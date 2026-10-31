#!/usr/bin/env python3
"""Separate LoRA NLI prototype, trained only on original human labels. No DPO continuation."""
import argparse, contextlib, hashlib, json, math, os, platform, random, sys, time
from pathlib import Path
from data_checks import ROOT, read_rows, validate_data

REVISION = '093f9f388b31de276ce2de164bdc2081324b9767'
OLD_CONFIG = '5a9bdc5bb8c31f118a3764651dbceeed3957e58ab875aa20eaf34f0c378f27b9'
OLD_WEIGHTS = '82d05d91d69396034ef9f69ecb36ad69ac313e23c8723c9b98d0ce9543d6ea19'
CLASSES = ['entailment', 'contradiction', 'neutral']
LETTERS = ['A', 'B', 'C']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_new(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def atomic_json(p, obj):
    t = Path(str(p) + '.tmp')
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(t, p)


def spec():
    s = json.loads((ROOT / 'PROTOCOL.json').read_text())
    for n, h in s['package_hashes'].items():
        if sha(ROOT / n) != h:
            raise RuntimeError('Package changed: ' + n)
    return s


def versions():
    import torch, transformers, peft, numpy, scipy, huggingface_hub

    return dict(
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        peft=peft.__version__,
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        huggingface_hub=huggingface_hub.__version__,
    )


def encode_prompt(proc, prompt):
    return proc.apply_chat_template(
        [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
    )['input_ids'][0].tolist()


def encode_train(proc, row):
    user = {'role': 'user', 'content': [{'type': 'text', 'text': row['prompt']}]}
    prompt = encode_prompt(proc, row['prompt'])
    full = proc.apply_chat_template(
        [
            user,
            {'role': 'assistant', 'content': [{'type': 'text', 'text': row['answer']}]},
        ],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
    )['input_ids'][0].tolist()
    if full[: len(prompt)] != prompt:
        raise RuntimeError(
            'Assistant template does not extend generation prefix exactly'
        )
    eot = proc.tokenizer.convert_tokens_to_ids('<end_of_turn>')
    if proc.tokenizer.convert_ids_to_tokens(eot) != '<end_of_turn>':
        raise RuntimeError('Cannot identify native assistant stop token')
    tail = full[len(prompt) :]
    if eot not in tail:
        raise RuntimeError('Native assistant stop token missing')
    answer = tail[: tail.index(eot) + 1]
    literal = proc.tokenizer(row['answer'], add_special_tokens=False)['input_ids']
    if answer != literal + [eot]:
        raise RuntimeError('Unexpected answer serialization')
    if len(prompt) + len(answer) > 512:
        raise RuntimeError('Input exceeds fixed short-evidence limit; no truncation')
    return dict(key=row['key'], prompt_ids=prompt, answer_ids=answer)


def preflight(args):
    s = spec()
    checks = validate_data()
    v = versions()
    out = Path(args.out).resolve()
    old = (
        Path.home()
        / 'korean_training/lora_experiment/gemma3_4b_ko_xlsum_FINAL_DPO_train100'
    )
    if out == old or old in out.parents:
        raise RuntimeError('Output must be separate from the historical adapter')
    for n, h in [
        ('adapter_config.json', OLD_CONFIG),
        ('adapter_model.safetensors', OLD_WEIGHTS),
    ]:
        if sha(old / n) != h:
            raise RuntimeError('Historical adapter guard mismatch')
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor

    snap = Path(
        snapshot_download(
            'google/gemma-3-4b-it', revision=REVISION, local_files_only=True
        )
    ).resolve()
    assert snap.name == REVISION
    proc = AutoProcessor.from_pretrained(str(snap), local_files_only=True)
    labelids = {
        x: proc.tokenizer(x, add_special_tokens=False)['input_ids'] for x in LETTERS
    }
    if (
        any(len(v) != 1 for v in labelids.values())
        or len({v[0] for v in labelids.values()}) != 3
    ):
        raise RuntimeError('Classifier requires three distinct single-token labels')
    encoded = {
        split: [
            encode_train(proc, r) for r in read_rows(ROOT / 'data' / f'{split}.jsonl')
        ]
        for split in ['train', 'validation', 'internal_test']
    }
    state = {
        'status': 'FROZEN_BEFORE_TRAINING',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'snapshot': str(snap),
        'revision': REVISION,
        'python': sys.executable,
        'versions': v,
        'protocol_sha256': sha(ROOT / 'PROTOCOL.json'),
        'old_adapter': str(old),
        'old_config_sha256': OLD_CONFIG,
        'old_weights_sha256': OLD_WEIGHTS,
        'chat_template_sha256': hashlib.sha256(
            (
                str(getattr(proc, 'chat_template', None))
                + '\n'
                + str(proc.tokenizer.chat_template)
            ).encode()
        ).hexdigest(),
        'label_ids': {k: v[0] for k, v in labelids.items()},
        'data_check': checks,
        'max_full_training_tokens': max(
            len(r['prompt_ids']) + len(r['answer_ids'])
            for rr in encoded.values()
            for r in rr
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'RUN_FREEZE.json').exists() or (out / 'ENCODED.json').exists():
        raise RuntimeError('Run directory already initialized; do not overwrite freeze')
    write_new(out / 'ENCODED.json', encoded)
    state['encoded_sha256'] = sha(out / 'ENCODED.json')
    write_new(out / 'RUN_FREEZE.json', state)
    print(json.dumps(state, indent=2))
    print('PREFLIGHT PASSED. GPU training smoke next.')


def metrics(rows):
    cm = [[0] * 3 for _ in range(3)]
    for r in rows:
        cm[CLASSES.index(r['human_label'])][CLASSES.index(r['prediction'])] += 1
    n = len(rows)
    f = []
    for i in range(3):
        den = sum(cm[i]) + sum(cm[j][i] for j in range(3))
        f.append(2 * cm[i][i] / den if den else 0.0)
    return {
        'n': n,
        'accuracy': sum(cm[i][i] for i in range(3)) / n,
        'macro_f1': sum(f) / 3,
        'confusion_true_rows_predicted_columns': cm,
        'class_order': CLASSES,
        'mean_restricted_label_nll': sum(r['nll'] for r in rows) / n,
    }


def runtime(args, new_adapter=None):
    s = spec()
    out = Path(args.out).resolve()
    f = json.loads((out / 'RUN_FREEZE.json').read_text())
    if (
        f['protocol_sha256'] != sha(ROOT / 'PROTOCOL.json')
        or versions() != f['versions']
        or sys.executable != f['python']
    ):
        raise RuntimeError('Frozen protocol/environment mismatch')
    if sha(out / 'ENCODED.json') != f['encoded_sha256']:
        raise RuntimeError('Encoded inputs changed')
    for n, h in [
        ('adapter_config.json', OLD_CONFIG),
        ('adapter_model.safetensors', OLD_WEIGHTS),
    ]:
        if sha(Path(f['old_adapter']) / n) != h:
            raise RuntimeError('Historical adapter changed')
    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    from peft import LoraConfig, get_peft_model, PeftModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires CUDA bfloat16 GPU')
    torch.manual_seed(2026)
    random.seed(2026)
    proc = AutoProcessor.from_pretrained(f['snapshot'], local_files_only=True)
    th = hashlib.sha256(
        (
            str(getattr(proc, 'chat_template', None))
            + '\n'
            + str(proc.tokenizer.chat_template)
        ).encode()
    ).hexdigest()
    if th != f['chat_template_sha256']:
        raise RuntimeError('Template changed')
    base = Gemma3ForConditionalGeneration.from_pretrained(
        f['snapshot'], torch_dtype=torch.bfloat16, local_files_only=True
    ).to('cuda')
    base.config.use_cache = False
    if new_adapter:
        model = PeftModel.from_pretrained(
            base, str(new_adapter), is_trainable=False
        ).eval()
    else:
        cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)',
        )
        model = get_peft_model(base, cfg)
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not trainable or any(
            'lora_' not in n or 'language_model' not in n for n in trainable
        ):
            raise RuntimeError('Unexpected trainable parameters')
    meta = {
        'gpu': torch.cuda.get_device_name(0),
        'cuda': torch.version.cuda,
        'dtype': 'bfloat16',
        'revision': f['revision'],
        'python': sys.executable,
        'versions': f['versions'],
    }
    mp = out / (args.task + '_runtime.json')
    if mp.exists() and json.loads(mp.read_text()) != meta:
        raise RuntimeError('Runtime/hardware differs on resume')
    if not mp.exists():
        write_new(mp, meta)
    print(json.dumps(meta, indent=2), file=sys.stderr, flush=True)
    return model, proc, f, out, s['training']


def classify(model, ids, f):
    import torch

    with torch.inference_mode():
        x = torch.tensor([ids], dtype=torch.long, device='cuda')
        output = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
        z = output.logits[0, -1, [f['label_ids'][a] for a in LETTERS]].float()
        lp = torch.log_softmax(z, dim=-1)
        if not torch.isfinite(lp).all():
            raise RuntimeError('Nonfinite classifier log probabilities')
        logp = lp.cpu().tolist()
        probs = lp.exp().cpu().tolist()
        ix = max(range(3), key=lambda i: logp[i])
        del output
    return {
        'prediction': CLASSES[ix],
        'restricted_class_probabilities': dict(zip(CLASSES, probs)),
        'restricted_class_logprobs': dict(zip(CLASSES, logp)),
        'probability_warning': 'Normalized over three label tokens; not calibrated factual-correctness probabilities',
    }


def evaluate_rows(model, f, out, split, system='new', path=None):
    enc = json.loads((out / 'ENCODED.json').read_text())[split]
    source = read_rows(ROOT / 'data' / f'{split}.jsonl')
    rows = []
    model.eval()
    if len(source) != len(enc):
        raise RuntimeError('Encoded/source row count mismatch')
    for row, e in zip(source, enc):
        assert row['key'] == e['key']
        pred = classify(model, e['prompt_ids'], f)
        r = {k: row[k] for k in ['key', 'premise_hash', 'genre', 'human_label']}
        r.update(pred)
        r.update(
            system=system,
            nll=-pred['restricted_class_logprobs'][row['human_label']],
            prompt_sha256=hashlib.sha256(row['prompt'].encode()).hexdigest(),
            input_ids_sha256=hashlib.sha256(
                json.dumps(e['prompt_ids']).encode()
            ).hexdigest(),
        )
        rows.append(r)
    if path:
        text = ''.join(
            json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows
        )
        if path.exists() and path.read_text() != text:
            raise RuntimeError('Existing completed evaluation differs; preserve it')
        if not path.exists():
            path.write_text(text)
    return metrics(rows)


def loss_one(model, e):
    import torch

    ids = e['prompt_ids'] + e['answer_ids']
    x = torch.tensor([ids], dtype=torch.long, device='cuda')
    answer = torch.tensor(e['answer_ids'], dtype=torch.long, device='cuda')
    P = len(e['prompt_ids'])
    L = len(e['answer_ids'])
    result = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
    logits = result.logits[0, P - 1 : P - 1 + L, :].float()
    loss = torch.nn.functional.cross_entropy(logits, answer)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss')
    return loss


def checkpoint(path, model, opt, state):
    import torch

    data = dict(
        state,
        params={
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        },
        optimizer=opt.state_dict(),
        rng_cpu=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state_all(),
    )
    tmp = Path(str(path) + '.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def restore(path, model, opt):
    import torch

    ck = torch.load(path, map_location='cpu', weights_only=False)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(ck['params']):
        raise RuntimeError('Checkpoint parameter mismatch')
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(ck['params'][n].to(p.device))
    opt.load_state_dict(ck['optimizer'])
    torch.set_rng_state(ck['rng_cpu'])
    torch.cuda.set_rng_state_all(ck['rng_cuda'])
    return {
        k: v
        for k, v in ck.items()
        if k not in ['params', 'optimizer', 'rng_cpu', 'rng_cuda']
    }


def train(args, smoke=False):
    import torch

    model, proc, f, out, cfg = runtime(args)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=0.0, eps=1e-8)
    enc = json.loads((out / 'ENCODED.json').read_text())['train']
    latest = out / 'latest_training.pt'
    if smoke:
        model.train()
        before = [p.detach().clone() for p in params]
        opt.zero_grad(set_to_none=True)
        loss = loss_one(model, enc[0])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        assert torch.isfinite(norm)
        opt.step()
        changed = any(not torch.equal(x, p.detach()) for x, p in zip(before, params))
        assert changed
        report = {
            'status': 'SMOKE_TRAINING_STEP_PASSED',
            'loss': float(loss.detach()),
            'gradient_norm_before_clip': float(norm),
            'some_lora_parameters_changed': changed,
            'training_example_key': enc[0]['key'],
            'uses_internal_test': False,
            'used_for_final_initialization': False,
        }
        atomic_json(out / 'SMOKE.json', report)
        print(json.dumps(report, indent=2))
        print('COMPLETE smoke')
        return
    if (out / 'TRAIN_COMPLETE.json').exists():
        print('Training already complete; preserve selected checkpoint.')
        return
    if (
        not (out / 'SMOKE.json').exists()
        or json.loads((out / 'SMOKE.json').read_text()).get('status')
        != 'SMOKE_TRAINING_STEP_PASSED'
    ):
        raise RuntimeError('Training smoke must complete first')
    state = {
        'epoch': 0,
        'next_group': 0,
        'steps': 0,
        'history': [],
        'best_epoch': 0,
        'best_nll': None,
    }
    if latest.exists():
        state = restore(latest, model, opt)
    else:
        initial = evaluate_rows(
            model, f, out, 'validation', path=out / 'dev_epoch_0.jsonl'
        )
        state['history'].append(dict(epoch=0, **initial))
        state['best_nll'] = initial['mean_restricted_label_nll']
        model.save_pretrained(out / 'checkpoints/epoch_0', safe_serialization=True)
        checkpoint(latest, model, opt, state)
    for epoch in range(state['epoch'], cfg['epochs']):
        indices = list(range(len(enc)))
        random.Random(2026 + epoch).shuffle(indices)
        groups = [
            indices[i : i + cfg['accumulation']]
            for i in range(0, len(indices), cfg['accumulation'])
        ]
        start = state['next_group'] if epoch == state['epoch'] else 0
        model.train()
        for gi in range(start, len(groups)):
            opt.zero_grad(set_to_none=True)
            meanloss = 0.0
            for i in groups[gi]:
                loss = loss_one(model, enc[i])
                meanloss += float(loss.detach()) / len(groups[gi])
                (loss / len(groups[gi])).backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(gn):
                raise RuntimeError('Nonfinite gradient norm')
            opt.step()
            opt.zero_grad(set_to_none=True)
            state.update(epoch=epoch, next_group=gi + 1, steps=state['steps'] + 1)
            if state['steps'] % 25 == 0:
                print(
                    'TRAIN',
                    epoch + 1,
                    'group',
                    gi + 1,
                    'of',
                    len(groups),
                    'step',
                    state['steps'],
                    'loss',
                    meanloss,
                    flush=True,
                )
            if state['steps'] % 50 == 0:
                checkpoint(latest, model, opt, state)
                atomic_json(
                    out / 'PROGRESS.json',
                    {k: v for k, v in state.items() if k != 'history'},
                )
        ep = epoch + 1
        dev = evaluate_rows(
            model, f, out, 'validation', path=out / f'dev_epoch_{ep}.jsonl'
        )
        state['history'].append(dict(epoch=ep, **dev))
        model.save_pretrained(out / f'checkpoints/epoch_{ep}', safe_serialization=True)
        if dev['mean_restricted_label_nll'] < state['best_nll']:
            state['best_epoch'] = ep
            state['best_nll'] = dev['mean_restricted_label_nll']
        state.update(epoch=ep, next_group=0)
        checkpoint(latest, model, opt, state)
        print('DEV', json.dumps(dict(epoch=ep, **dev)), flush=True)
    selected = out / f'checkpoints/epoch_{state["best_epoch"]}'
    record = {
        'status': 'TRAINING_COMPLETE_SELECTED_WITHOUT_INTERNAL_TEST',
        'selected_epoch': state['best_epoch'],
        'selected_adapter_relative_path': str(selected.relative_to(out)),
        'selected_adapter_hashes': {
            n: sha(selected / n)
            for n in ['adapter_config.json', 'adapter_model.safetensors']
        },
        'dev_history': state['history'],
        'selection': 'minimum validation restricted three-label NLL among epochs 0,1,2; earlier epoch wins ties',
        'optimizer_steps': state['steps'],
        'new_adapter_is_noop_epoch0': state['best_epoch'] == 0,
    }
    write_new(out / 'TRAIN_COMPLETE.json', record)
    for n, h in [
        ('adapter_config.json', OLD_CONFIG),
        ('adapter_model.safetensors', OLD_WEIGHTS),
    ]:
        assert sha(Path(f['old_adapter']) / n) == h
    print(json.dumps(record, indent=2))
    print('COMPLETE train')


def selected(out):
    rec = json.loads((out / 'TRAIN_COMPLETE.json').read_text())
    path = out / rec['selected_adapter_relative_path']
    for n, h in rec['selected_adapter_hashes'].items():
        if sha(path / n) != h:
            raise RuntimeError('Selected checkpoint changed after selection')
    return path, rec


def assess(args):
    out = Path(args.out).resolve()
    path, record = selected(out)
    model, proc, f, out, cfg = runtime(args, new_adapter=path)
    result = {
        'selected_epoch': record['selected_epoch'],
        'task': 'internal three-way NLI classification, not official KLUE test score',
    }
    with model.disable_adapter():
        result['base'] = evaluate_rows(
            model,
            f,
            out,
            'internal_test',
            system='base',
            path=out / 'heldout_base.jsonl',
        )
    result['new'] = evaluate_rows(
        model, f, out, 'internal_test', system='new', path=out / 'heldout_new.jsonl'
    )
    atomic_json(out / 'HELDOUT_SUMMARY.json', result)
    print(json.dumps(result, indent=2))
    print('COMPLETE assess')


def predict(args):
    if not args.input:
        raise ValueError('Supply --input JSONL containing key, premise and claim')
    out = Path(args.out).resolve()
    path, _ = selected(out)
    model, proc, f, out, cfg = runtime(args, new_adapter=path)
    model.eval()
    template = json.loads((ROOT / 'DATA_MANIFEST.json').read_text())['prompt_template']
    for r in read_rows(args.input):
        if not all(
            isinstance(r.get(k), str) and r[k].strip()
            for k in ['key', 'premise', 'claim']
        ):
            raise ValueError('Missing key/premise/claim')
        prompt = template.format(premise=r['premise'], hypothesis=r['claim'])
        ids = encode_prompt(proc, prompt)
        if len(ids) > 512:
            raise ValueError(
                'Evidence too long for this short-evidence prototype; no silent truncation'
            )
        print(
            json.dumps(
                {'key': r['key'], **classify(model, ids, f)}, ensure_ascii=False
            ),
            flush=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('task', choices=['preflight', 'smoke', 'train', 'assess', 'predict'])
    p.add_argument('--out', required=True)
    p.add_argument('--input')
    args = p.parse_args()
    if args.task == 'preflight':
        preflight(args)
    elif args.task == 'smoke':
        train(args, True)
    elif args.task == 'train':
        train(args)
    elif args.task == 'assess':
        assess(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
