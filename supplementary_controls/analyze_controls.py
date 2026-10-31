"""Post-hoc saved-score controls. Protocol is fixed before this script's first run.
No GPU inference or additional-sample fitting. Re-run into a NEW output directory.
"""
from pathlib import Path
import argparse, hashlib, json, platform, sys
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, kendalltau, pearsonr

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def selective(h,c):
    return np.where(((h>=75)&(c<=50))|((h<=50)&(c>=75)),(h+c)/2,h)
def rho(y,s):
    return float(spearmanr(y,s).statistic) if np.unique(y).size>1 and np.unique(s).size>1 else None
def metrics(y,s):
    e=s-y
    return dict(n=len(y),spearman=rho(y,s),kendall=float(kendalltau(y,s).statistic),pearson=float(pearsonr(y,s).statistic),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()),extreme_over=float(100*np.mean(e>=50)),extreme_under=float(100*np.mean(e<=-50)))
def intervals(draws,percentiles=(2.5,97.5)):
    return {k:[float(x) for x in np.percentile(v,percentiles)] for k,v in draws.items()}
def bootstrap(df,pairs,seed):
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero(df.lang.to_numpy()==l) for l in sorted(df.lang.unique())]
    out={a+' minus '+b:{'spearman':[],'mae':[]} for a,b in pairs}
    y=df.y.to_numpy();scores={s:df[s].to_numpy() for pair in pairs for s in pair}
    for _ in range(2000):
        ix=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        rs={s:rho(y[ix],v[ix]) for s,v in scores.items()}
        maes={s:float(np.abs(v[ix]-y[ix]).mean()) for s,v in scores.items()}
        for a,b in pairs:
            out[a+' minus '+b]['spearman'].append(rs[a]-rs[b])
            out[a+' minus '+b]['mae'].append(maes[a]-maes[b])
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,default=Path(__file__).parent/'data');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'RESULTS.json').exists():raise RuntimeError('Preserve existing results; choose a new --out directory.')
    plan=json.loads((Path(__file__).parent/'PROTOCOL_FREEZE.json').read_text())
    devpath=args.data_root/'development900.csv';labpath=args.data_root/'additional_labels.csv'
    assert sha(devpath)==plan['input_sha256']['development']
    assert sha(labpath)==plan['input_sha256']['additional_labels']
    dev=pd.read_csv(devpath);assert len(dev)==900
    for p in sorted((args.data_root/'additional_rows').glob('row_*.json')):assert sha(p)==plan['row_sha256'][p.name]
    # ONLY fitted parameter. Additional references are not involved here.
    delta=float((dev.human_0_100-dev.gemma_fullsource_score).mean())
    fit={'method':'clipped additive intercept','offset':delta,'fit_n':900,'fit_input_sha256':sha(devpath),'fit_uses_additional_scores_or_labels':False}
    (args.out/'CALIBRATION_FIT.json').write_text(json.dumps(fit,indent=2))
    labels=pd.read_csv(labpath).set_index('holdout_row',verify_integrity=True)
    records=[]
    for p in sorted((args.data_root/'additional_rows').glob('row_*.json')):
        r=json.loads(p.read_text());key=int(r['holdout_row']);assert int(r['id'])==int(labels.loc[key,'id'])
        h=float(r['H']['score']);c=float(r['S0']['capped']);nli=float(r['NLI_T']['trained3']['capped']);base=float(r['NLI_T']['base3']['capped'])
        records.append(dict(row_id=key,id=r['id'],lang=r['lang'],y=float(labels.loc[key,'q2_avg'])*100,H=h,capped=c,Mean=(h+c)/2,SV=float(selective(h,c)),CalH=float(np.clip(h+delta,0,100)),SV_NLI=float(selective(h,nli)),SV_NLI_base=float(selective(h,base)),translation_compliant=bool(r['T']['source_check']),gpu=r['runtime']['gpu']))
    add=pd.DataFrame(records).sort_values('row_id').reset_index(drop=True);assert len(add)==900 and add.row_id.nunique()==900 and add.groupby('lang').size().eq(20).all()
    devscores=pd.DataFrame(dict(row_id=dev.row_id,lang=dev.lang,y=dev.human_0_100,H=dev.gemma_fullsource_score,capped=dev.claim_capped_score))
    devscores['Mean']=(devscores.H+devscores.capped)/2;devscores['SV']=selective(devscores.H,devscores.capped);devscores['CalH']=np.clip(devscores.H+delta,0,100)
    assert np.allclose(devscores.SV,dev.final_symmetric_extreme_average,rtol=0,atol=1e-10)
    systems=['H','Mean','SV','CalH','SV_NLI','SV_NLI_base']
    overall={s:metrics(add.y.to_numpy(),add[s].to_numpy()) for s in systems}
    development={s:metrics(devscores.y.to_numpy(),devscores[s].to_numpy()) for s in ['H','Mean','SV','CalH']}
    pairs=[('SV','Mean'),('Mean','H'),('SV','CalH'),('CalH','H')]
    draws=bootstrap(add,pairs,20261024)
    controls={k:{'estimate':{m:overall[k.split(' minus ')[0]][m]-overall[k.split(' minus ')[1]][m] for m in ['spearman','mae']},'ci95':intervals(v)} for k,v in draws.items()}
    original=bootstrap(add,[('SV','H'),('SV_NLI','SV')],20261021)
    primary={k:{'estimate':overall[k.split(' minus ')[0]]['spearman']-overall[k.split(' minus ')[1]]['spearman'],'nominal95':intervals(v)['spearman'],'bonferroni_style97_5':intervals(v,(1.25,98.75))['spearman']} for k,v in original.items()}
    # Paired-language units: never replace undefined correlations with zero.
    within=[]
    for lang,g in add.groupby('lang',sort=True):
        for s in systems:within.append(dict(lang=lang,system=s,n=len(g),human_unique=g.y.nunique(),system_unique=g[s].nunique(),spearman=rho(g.y.to_numpy(),g[s].to_numpy())))
    within=pd.DataFrame(within);summary={}
    for s in systems:
        g=within[within.system==s];v=g.spearman.dropna().to_numpy()
        summary[s]={'defined':len(v),'undefined':int(g.spearman.isna().sum()),'undefined_languages':g.loc[g.spearman.isna(),'lang'].tolist(),'median':float(np.median(v)) if len(v) else None,'q1':float(np.percentile(v,25)) if len(v) else None,'q3':float(np.percentile(v,75)) if len(v) else None,'positive':int((v>0).sum())}
    result={'status':'POST_HOC_SUPPLEMENTARY_CONTROLS_COMPLETE','protocol_sha256':sha(Path(__file__).parent/'PROTOCOL_FREEZE.json'),'analysis_script_sha256':sha(__file__),'calibration':fit,'additional':overall,'development':development,'controls':controls,'original_primary_multiplicity_sensitivity':primary,'within_language':summary,'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},'limits':['Original additional-sample results were inspected before this protocol.','No additional-sample fitting, endpoint replacement, threshold tuning or model selection.','Language-level correlations are descriptive with n=20.','Bonferroni-style bootstrap intervals are an approximate supplementary family-wise sensitivity.','No independent semantic validation is performed by this analysis.']}
    (args.out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    add.to_csv(args.out/'additional_scores.csv',index=False);devscores.to_csv(args.out/'development_scores.csv',index=False);within.to_csv(args.out/'within_language.csv',index=False)
    np.savez_compressed(args.out/'bootstrap_draws.npz',**{('control__'+k+'__'+m):np.array(v) for k,d in draws.items() for m,v in d.items()},**{('primary__'+k+'__'+m):np.array(v) for k,d in original.items() for m,v in d.items()})
    print(json.dumps({k:result[k] for k in ['calibration','additional','controls','original_primary_multiplicity_sensitivity','within_language']},indent=2))

if __name__=='__main__':main()
