from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

import argparse
ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--out',type=Path);args=ap.parse_args()
root=args.out or Path(__file__).resolve().parent;root.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,'font.size':11})
fig,ax=plt.subplots(figsize=(8.2,3.3));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
def box(x,y,w,h,label):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',facecolor='#f5f7f9',edgecolor='#34495e',linewidth=1));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=12)
def arrow(a,b,label=None,rad=0):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#34495e',connectionstyle=f'arc3,rad={rad}',linewidth=1.1))
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.1,label,ha='center',fontsize=10)
box(.05,2.8,1.65,.8,'Source and\nsummary')
box(2.5,2.8,1.65,.8,'Holistic\nscore H')
box(6.25,2.8,1.8,.8,'Selective\n75/50 rule')
box(8.6,2.8,1.3,.8,'Final\nscore')
box(.8,.5,1.9,.85,'Translation\nto English')
box(3.25,.5,1.75,.85,'Claim\nextraction')
box(5.55,.5,1.75,.85,'Claim\nverification')
box(7.85,.5,1.9,.85,'Capped\naggregation')
arrow((1.7,3.2),(2.5,3.2));arrow((4.15,3.2),(6.25,3.2));arrow((8.05,3.2),(8.6,3.2))
arrow((.85,2.8),(1.65,1.35));arrow((2.7,.925),(3.25,.925),'summary');arrow((5,.925),(5.55,.925),'claims');arrow((7.3,.925),(7.85,.925))
arrow((8.8,1.35),(7.55,2.8),'claim score')
ax.plot([1.7,1.7,6.4],[.5,.15,.15],color='#34495e',lw=1);arrow((6.4,.15),(6.4,.5));ax.text(4.1,.03,'translated source',ha='center',fontsize=10)
ax.text(5,2.05,'Both scores are computed for every example.',ha='center',fontsize=11)
fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.08);fig.savefig(root/'pipeline.pdf',bbox_inches='tight');plt.close(fig)

r=json.loads(args.results.read_text())
fig,ax=plt.subplots(figsize=(6.4,2.2))
for y,(key,label) in enumerate([('SV minus H','SV − H'),('SV_NLI minus SV','SV-NLI − SV')]):
 d=r['original_primary_multiplicity_sensitivity'][key];lo,hi=d['nominal95'];a,b=d['bonferroni_style97_5'];v=d['estimate']
 ax.plot([a,b],[y,y],color='#8296a7',lw=2);ax.plot([lo,hi],[y,y],color='#1b4965',lw=5);ax.scatter([v],[y],s=35,c='#102a43',zorder=3)
ax.axvline(0,c='#666666',ls='--',lw=.9);ax.set_yticks([0,1],['SV − H','SV-NLI − SV']);ax.invert_yaxis();ax.set_ylim(1.55,-.55);ax.set_xlim(-.025,.14)
ax.set_xlabel('Difference in pooled Spearman correlation');ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.18);ax.tick_params(axis='y',length=0)
fig.tight_layout();fig.savefig(root/'primary_contrasts.pdf',bbox_inches='tight');plt.close(fig)
