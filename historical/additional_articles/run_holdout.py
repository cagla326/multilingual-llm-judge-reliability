#!/usr/bin/env python3
"""Holdout confirmation runner. Human scores are never read by this file.

Per row, in one process and one runtime:
  H       historical holistic Q2 judgment (256 new tokens, historical parsing + recovery)
  T       whole-document translation of source (4096) and summary (1024)
  claims  extraction from the translated summary (512)
  S0      four-label translated verifier on (T source, claims) -> capped -> trigger
  R       repaired translation: rows whose T source/summary fails the English check are
          re-translated with the frozen chunked procedure; claims re-extracted only if
          the summary was repaired
  NLI     bridge three-label verifier (adapter on = trained, off = base) on both
          (T source, claims) and (R source, R claims)
Which S* variant is primary is decided by the analysis from the frozen repair result.

Modes: preflight (CPU), smoke (GPU, rows 0 and the longest source), infer (GPU,
optionally sharded: --shard k --nshards n), status (CPU).
"""
import argparse
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path

import common as C

PROTOCOL = C.ROOT / "PROTOCOL.json"
DEFAULT_ADAPTER = str(Path.home() / "korean_training/lora_experiment/Korean_Verifier_Development_2026-10-19/run_v1/checkpoints/epoch_2")


def versions():
    import numpy, scipy, torch, transformers, peft, huggingface_hub
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "huggingface_hub": huggingface_hub.__version__}


def check_package():
    protocol = json.loads(PROTOCOL.read_text())
    for name, h in protocol["package_hashes"].items():
        if C.sha(C.ROOT / name) != h:
            raise RuntimeError("Package file changed after freeze: " + name)
    return protocol


def check_adapter(path):
    for name, h in [("adapter_config.json", C.ADAPTER_CONFIG_SHA), ("adapter_model.safetensors", C.ADAPTER_WEIGHTS_SHA)]:
        if C.sha(Path(path) / name) != h:
            raise RuntimeError("Adapter hash mismatch: " + name)


def chat_ids(proc, prompt):
    return proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")


def template_hash(proc):
    return C.digest([str(getattr(proc, "chat_template", None)), str(proc.tokenizer.chat_template)])


def snapshot():
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(C.MODEL_ID, revision=C.REVISION, local_files_only=True))
    if snap.name != C.REVISION:
        raise RuntimeError("Unexpected snapshot revision")
    return snap


# ------------------------------------------------------------------ preflight
def preflight(a):
    check_package()
    rows = C.load_inputs()
    check_adapter(a.adapter)
    from transformers import AutoConfig, AutoProcessor, Gemma3ForConditionalGeneration
    if "logits_to_keep" not in inspect.signature(Gemma3ForConditionalGeneration.forward).parameters:
        raise RuntimeError("Missing logits_to_keep API")
    snap = snapshot()
    proc = AutoProcessor.from_pretrained(str(snap), local_files_only=True)
    context = int(AutoConfig.from_pretrained(str(snap), local_files_only=True).text_config.max_position_embeddings)
    label_ids = {x: proc.tokenizer(x, add_special_tokens=False)["input_ids"] for x in ["A", "B", "C"]}
    if not all(len(v) == 1 for v in label_ids.values()):
        raise RuntimeError("Labels are not single tokens")
    max_h = max_t = 0
    for r in rows:
        max_h = max(max_h, chat_ids(proc, C.build_holistic_prompt(r["input"], r["summary"]))["input_ids"].shape[-1])
        max_t = max(max_t, chat_ids(proc, C.build_translation_prompt(r["input"], "source document"))["input_ids"].shape[-1])
    if max(max_h, max_t) + C.MAX_NEW_TOKENS_SOURCE_WHOLE > context:
        raise RuntimeError("Prompt would exceed context")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "RUN_FREEZE.json").exists():
        raise RuntimeError("Run already frozen")
    freeze = {
        "status": "FROZEN_BEFORE_ANY_HOLDOUT_MODEL_OUTPUT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_sha256": C.sha(PROTOCOL), "inputs_sha256": C.sha(C.INPUTS),
        "sealed_labels_sha256": C.sha(C.LABELS), "sealed_labels_opened": False,
        "rows": len(rows), "max_holistic_prompt_tokens": max_h, "max_translation_prompt_tokens": max_t,
        "model_context": context, "snapshot": str(snap), "revision": C.REVISION,
        "adapter": str(Path(a.adapter).resolve()), "chat_template_sha256": template_hash(proc),
        "label_ids": {k: v[0] for k, v in label_ids.items()}, "versions": versions(),
        "python": sys.executable, "human_scores_in_model_inputs": False, "new_training": False,
    }
    C.dump(out / "RUN_FREEZE.json", freeze)
    print(json.dumps(freeze, indent=2))
    print("PREFLIGHT PASSED; submit the Slurm smoke next. No inference on the login node.")


# ------------------------------------------------------------------ model
class Runtime:
    def __init__(self, freeze):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA bfloat16 GPU required")
        check_adapter(freeze["adapter"])
        self.torch, self.freeze = torch, freeze
        self.proc = AutoProcessor.from_pretrained(freeze["snapshot"], local_files_only=True)
        if template_hash(self.proc) != freeze["chat_template_sha256"]:
            raise RuntimeError("Chat template changed")
        torch.manual_seed(2026)
        base = Gemma3ForConditionalGeneration.from_pretrained(
            freeze["snapshot"], torch_dtype=torch.bfloat16, local_files_only=True).to("cuda")
        self.model = PeftModel.from_pretrained(base, freeze["adapter"], is_trainable=False).eval()
        self.calls = 0

    def generate(self, prompt, max_new_tokens):
        """Base model (adapter disabled); mirrors the historical generate calls."""
        inputs = chat_ids(self.proc, prompt).to("cuda")
        n_in = inputs["input_ids"].shape[-1]
        with self.torch.no_grad(), self.model.disable_adapter():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None)
        new = out[0][n_in:]
        self.calls += 1
        return {"text": self.proc.decode(new, skip_special_tokens=True), "new_tokens": int(new.shape[-1]),
                "hit_limit": int(new.shape[-1]) >= max_new_tokens}

    def classify(self, prompt, trained):
        """Bridge NLI classification: restricted next-token log-softmax over A/B/C."""
        torch = self.torch
        ids = chat_ids(self.proc, prompt)["input_ids"][0].tolist()
        lid = [self.freeze["label_ids"][c] for c in ["A", "B", "C"]]

        def run():
            with torch.inference_mode():
                x = torch.tensor([ids], dtype=torch.long, device="cuda")
                o = self.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
                lp = torch.log_softmax(o.logits[0, -1, lid].float(), dim=-1)
                if not torch.isfinite(lp).all():
                    raise RuntimeError("Nonfinite NLI logits")
                return lp.cpu().tolist()
        if trained:
            v = run()
        else:
            with self.model.disable_adapter():
                v = run()
        return {"label": C.NLI_CLASSES[max(range(3), key=lambda i: v[i])],
                "restricted_logprobs": dict(zip(C.NLI_CLASSES, v)), "prompt_tokens": len(ids)}


# ------------------------------------------------------------------ one row
def four_label(rt, source_en, claims):
    recs, labels = [], []
    for claim in claims:
        g = rt.generate(C.build_claim_verification_prompt(source_en, claim), C.MAX_NEW_TOKENS_VERIFICATION)
        rec = C.verification_record(claim, g["text"].strip())
        rec["hit_token_limit"] = g["hit_limit"]
        recs.append(rec)
        labels.append(rec["label"])
    return {"verifications": recs, "labels": labels, "capped": C.capped_score(labels)}


def nli_pair(rt, source_en, claims):
    out = {}
    for name, trained in [("base3", False), ("trained3", True)]:
        preds = [rt.classify(C.build_nli_prompt(source_en, c), trained) for c in claims]
        out[name] = {"predictions": preds, "labels": [p["label"] for p in preds],
                     "capped": C.nli_capped([p["label"] for p in preds])}
    return out


def process_row(rt, row):
    t0 = time.time()
    rec = {"holdout_row": row["holdout_row"], "id": row["id"], "lang": row["lang"]}
    # H
    g = rt.generate(C.build_holistic_prompt(row["input"], row["summary"]), C.MAX_NEW_TOKENS_HOLISTIC)
    h = C.holistic_score(g["text"])
    rec["H"] = dict(h, raw_response=g["text"], hit_token_limit=g["hit_limit"])
    # T: whole-document translation (historical)
    gs = rt.generate(C.build_translation_prompt(row["input"], "source document"), C.MAX_NEW_TOKENS_SOURCE_WHOLE)
    gm = rt.generate(C.build_translation_prompt(row["summary"], "candidate summary"), C.MAX_NEW_TOKENS_SUMMARY)
    src_t, sum_t = gs["text"].strip(), gm["text"].strip()
    ge = rt.generate(C.build_claim_extraction_prompt(sum_t), C.MAX_NEW_TOKENS_EXTRACTION)
    claims_t, valid_t = C.claims_from_extraction(ge["text"].strip())
    rec["T"] = {"source_en": src_t, "summary_en": sum_t, "raw_claim_extraction": ge["text"].strip(),
                "claims": claims_t, "claims_json_valid": valid_t,
                "source_hit_limit": gs["hit_limit"], "summary_hit_limit": gm["hit_limit"],
                "source_check": C.source_is_english(src_t), "summary_check": C.summary_is_english(sum_t)}
    rec["S0"] = four_label(rt, src_t, claims_t)
    rec["NLI_T"] = nli_pair(rt, src_t, claims_t)
    # R: repaired translation when the whole-document output fails the check
    hits = []

    def tr(chunk, text_type):
        x = rt.generate(C.build_translation_prompt(chunk, text_type), C.MAX_NEW_TOKENS_CHUNK)
        hits.append(x["hit_limit"])
        return x["text"].strip()
    src_r, sum_r, claims_r = src_t, sum_t, claims_t
    r = {"source_repaired": not rec["T"]["source_check"], "summary_repaired": not rec["T"]["summary_check"]}
    if r["source_repaired"]:
        src_r, r["source_chunks"] = C.translate_with_repair(row["input"], "source document", tr)
    if r["summary_repaired"]:
        sum_r, r["summary_chunks"] = C.translate_with_repair(row["summary"], "candidate summary", tr)
        ge2 = rt.generate(C.build_claim_extraction_prompt(sum_r), C.MAX_NEW_TOKENS_EXTRACTION)
        claims_r, r["claims_json_valid"] = C.claims_from_extraction(ge2["text"].strip())
        r["raw_claim_extraction"] = ge2["text"].strip()
    r.update({"source_en": src_r, "summary_en": sum_r, "claims": claims_r,
              "source_check": C.source_is_english(src_r), "summary_check": C.summary_is_english(sum_r),
              "chunk_token_limit_hits": sum(hits)})
    rec["R"] = r
    rec["NLI_R"] = rec["NLI_T"] if not (r["source_repaired"] or r["summary_repaired"]) else nli_pair(rt, src_r, claims_r)
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def inference(a):
    out = Path(a.out).resolve()
    freeze = json.loads((out / "RUN_FREEZE.json").read_text())
    check_package()
    if C.sha(PROTOCOL) != freeze["protocol_sha256"] or C.sha(C.INPUTS) != freeze["inputs_sha256"]:
        raise RuntimeError("Protocol or inputs changed after freeze")
    if versions() != freeze["versions"] or sys.executable != freeze["python"]:
        raise RuntimeError("Frozen environment mismatch")
    rows = C.load_inputs()
    rt = Runtime(freeze)
    import torch
    runtime = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "dtype": "bfloat16",
               "revision": C.REVISION, "versions": freeze["versions"], "slurm_job": os.environ.get("SLURM_JOB_ID"),
               "shard": a.shard, "nshards": a.nshards}
    print(json.dumps(runtime, indent=2), flush=True)
    if a.task == "smoke":
        longest = max(rows, key=lambda r: len(r["input"]))
        recs = [process_row(rt, r) for r in [rows[0], longest]]
        C.dump(out / "SMOKE.json", {"status": "SMOKE_PASSED", "protocol_sha256": freeze["protocol_sha256"],
                                    "runtime": runtime, "records": recs, "generation_calls": rt.calls})
        print(f"COMPLETE smoke: rows {rows[0]['holdout_row']} and {longest['holdout_row']} in "
              f"{recs[0]['seconds']} s and {recs[1]['seconds']} s", flush=True)
        return
    smoke = json.loads((out / "SMOKE.json").read_text())
    if smoke["status"] != "SMOKE_PASSED" or smoke["protocol_sha256"] != freeze["protocol_sha256"]:
        raise RuntimeError("Smoke missing or from another protocol")
    dest = out / "rows"
    dest.mkdir(exist_ok=True)
    mine = [r for r in rows if int(r["holdout_row"]) % a.nshards == a.shard]
    for i, row in enumerate(mine):
        path = dest / f"row_{int(row['holdout_row']):04d}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old["holdout_row"] != row["holdout_row"] or old["protocol_sha256"] != freeze["protocol_sha256"]:
                raise RuntimeError("Foreign or changed row file: " + path.name)
            continue
        rec = process_row(rt, row)
        rec.update({"protocol_sha256": freeze["protocol_sha256"], "runtime": runtime})
        tmp = path.with_suffix(".tmp")
        C.dump(tmp, rec)
        os.replace(tmp, path)
        print(f"SAVED row {row['holdout_row']} ({row['lang']}) {rec['seconds']} s  shard progress {i + 1}/{len(mine)}", flush=True)
    done = len(list(dest.glob("row_*.json")))
    print(f"SHARD {a.shard}/{a.nshards} COMPLETE; total rows saved {done}/900", flush=True)
    if done == 900:
        C.dump(out / "COMPLETE.json", {"status": "ALL_900_ROWS_COMPLETE", "protocol_sha256": freeze["protocol_sha256"]})
        print("COMPLETE infer", flush=True)


def status(a):
    out = Path(a.out).resolve()
    files = list((out / "rows").glob("row_*.json")) if (out / "rows").exists() else []
    secs = [json.loads(p.read_text())["seconds"] for p in files]
    left = (900 - len(files)) * (sum(secs) / len(secs)) / 3600 if secs else None
    print(json.dumps({"done": len(files), "total": 900, "complete": (out / "COMPLETE.json").exists(),
                      "mean_seconds_per_row": round(sum(secs) / len(secs), 1) if secs else None,
                      "single_gpu_hours_left": round(left, 2) if left is not None else None}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["preflight", "smoke", "infer", "status"])
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    a = p.parse_args()
    if not 0 <= a.shard < a.nshards:
        raise SystemExit("shard must be in [0, nshards)")
    {"preflight": preflight, "smoke": inference, "infer": inference, "status": status}[a.task](a)


if __name__ == "__main__":
    main()
