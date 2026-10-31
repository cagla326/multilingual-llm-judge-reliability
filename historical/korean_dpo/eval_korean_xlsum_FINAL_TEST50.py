import os
import sys
import csv
import json
import math
import time
import hashlib
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import peft

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel


# ============================================================
# FROZEN CONFIG
# ============================================================

MODEL_ID = "google/gemma-3-4b-it"

ADAPTER_DIR = "gemma3_4b_ko_xlsum_FINAL_DPO_train100"

ORIGINAL_FILE = "korean_xlsum_FROZEN_TEST50_ORIGINAL.jsonl"
SWAPPED_FILE = "korean_xlsum_FROZEN_TEST50_SWAPPED.jsonl"
PROTOCOL_FILE = "korean_xlsum_FROZEN_TEST50_EVAL_PROTOCOL.json"

OUT_DIR = "FINAL_KOREAN_TEST50_EVAL"

SEED = 2026
BOOTSTRAP_N = 10000
MAX_NEW_TOKENS = 4


# ============================================================
# UTILITIES
# ============================================================

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def make_prompt(row, template):
    return template.format(
        article=row["article"],
        summary_a=row["summary_a"],
        summary_b=row["summary_b"],
    )


def validate_inputs(original, swapped, protocol):
    assert len(original) == 50, f"Expected original n=50, got {len(original)}"
    assert len(swapped) == 50, f"Expected swapped n=50, got {len(swapped)}"

    o = {r["semantic_pair_id"]: r for r in original}
    s = {r["semantic_pair_id"]: r for r in swapped}

    assert len(o) == 50
    assert len(s) == 50
    assert set(o) == set(s)

    for pid in sorted(o):
        a = o[pid]
        b = s[pid]

        assert a["orientation"] == "original"
        assert b["orientation"] == "swapped"

        assert a["article"] == b["article"]
        assert a["summary_a"] == b["summary_b"]
        assert a["summary_b"] == b["summary_a"]

        assert a["gold_label"] in ["A", "B"]
        assert b["gold_label"] in ["A", "B"]
        assert a["gold_label"] != b["gold_label"]

        # Gold must correspond to grounded summary.
        gold_a = a["summary_a"] if a["gold_label"] == "A" else a["summary_b"]
        gold_b = b["summary_a"] if b["gold_label"] == "A" else b["summary_b"]

        assert gold_a == a["grounded_summary"]
        assert gold_b == b["grounded_summary"]

        # Frozen metadata should also match across orientations.
        for key in [
            "severity",
            "corruption_type",
            "corruption_family",
            "grounded_summary",
            "hallucinated_summary",
        ]:
            assert a[key] == b[key]

    assert Counter(r["gold_label"] for r in original) == Counter({"A": 25, "B": 25})
    assert Counter(r["severity"] for r in original) == Counter({"minor": 25, "major": 25})

    expected_families = Counter({
        "temporal_scope": 18,
        "relational_policy_status": 13,
        "scope_evidence_overstatement": 11,
        "numeric_quantitative": 8,
    })
    assert Counter(r["corruption_family"] for r in original) == expected_families

    assert protocol["generation_policy"]["do_sample"] is False
    assert protocol["generation_policy"]["max_new_tokens"] == 4
    assert protocol["statistics"]["paired_bootstrap_seed"] == 2026

    # Verify all prompts can be deterministically reconstructed.
    template = protocol["prompt_template"]
    for r in original + swapped:
        p = make_prompt(r, template)
        assert "{article}" not in p
        assert "{summary_a}" not in p
        assert "{summary_b}" not in p
        assert r["article"] in p
        assert r["summary_a"] in p
        assert r["summary_b"] in p

    print("INPUT VALIDATION PASSED", flush=True)
    print("original =", len(original), flush=True)
    print("swapped  =", len(swapped), flush=True)
    print("original labels =", Counter(r["gold_label"] for r in original), flush=True)
    print("severity =", Counter(r["severity"] for r in original), flush=True)
    print("families =", Counter(r["corruption_family"] for r in original), flush=True)


# ============================================================
# MODEL SCORING
#
# IMPORTANT:
# The final DPO training used raw prompt tokenization rather than
# an additional chat-template wrapper:
#
# prompt.strip() + "\n"
# add_special_tokens=False
#
# Evaluation preserves that exact input convention for BOTH Base
# and DPO to avoid implementation drift.
# ============================================================

def model_prompt_text(prompt):
    return prompt.strip() + "\n"


def free_generate(model, tokenizer, prompt, device):
    text = model_prompt_text(prompt)

    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    input_len = input_ids.shape[1]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=pad_id,
        )

    new_ids = generated[0, input_len:]

    decoded = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
    ).strip()

    pred = decoded if decoded in ["A", "B"] else None

    return {
        "generated_text": decoded,
        "prediction": pred,
        "valid": pred is not None,
        "prompt_tokens": int(input_len),
        "generated_tokens": int(new_ids.shape[0]),
    }


def conditional_label_logprob(model, tokenizer, prompt, label, device):
    prompt_text = model_prompt_text(prompt)

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    label_ids = tokenizer(
        label,
        add_special_tokens=False,
    )["input_ids"]

    if len(label_ids) == 0:
        raise RuntimeError(f"Empty label tokenization for {label!r}")

    ids = prompt_ids + label_ids

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    # logits[t] predicts token t+1.
    start = len(prompt_ids) - 1
    end = start + len(label_ids)

    relevant_logits = outputs.logits[:, start:end, :].float()
    targets = input_ids[:, len(prompt_ids):len(prompt_ids) + len(label_ids)]

    log_probs = F.log_softmax(relevant_logits, dim=-1)

    token_lp = log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    score = float(token_lp.sum().detach().cpu())

    if not math.isfinite(score):
        raise RuntimeError(f"Non-finite logprob for label {label}")

    del outputs, relevant_logits, log_probs, token_lp

    return score


def evaluate_system(system_name, model, tokenizer, original, swapped, template, device):
    rows = original + swapped
    results = []

    print("\n" + "=" * 90, flush=True)
    print(f"EVALUATING SYSTEM: {system_name}", flush=True)
    print("=" * 90, flush=True)

    start_time = time.time()

    for i, row in enumerate(rows, start=1):
        prompt = make_prompt(row, template)

        gen = free_generate(
            model,
            tokenizer,
            prompt,
            device,
        )

        lp_a = conditional_label_logprob(
            model,
            tokenizer,
            prompt,
            "A",
            device,
        )

        lp_b = conditional_label_logprob(
            model,
            tokenizer,
            prompt,
            "B",
            device,
        )

        if lp_a > lp_b:
            lp_pred = "A"
        elif lp_b > lp_a:
            lp_pred = "B"
        else:
            lp_pred = None

        result = {
            "system": system_name,
            "semantic_pair_id": row["semantic_pair_id"],
            "orientation": row["orientation"],
            "gold_label": row["gold_label"],
            "free_generated_text": gen["generated_text"],
            "free_prediction": gen["prediction"],
            "free_valid": int(gen["valid"]),
            "free_correct": int(gen["prediction"] == row["gold_label"]),
            "logprob_A": lp_a,
            "logprob_B": lp_b,
            "logprob_prediction": lp_pred,
            "logprob_correct": int(lp_pred == row["gold_label"]),
            "prompt_tokens": gen["prompt_tokens"],
            "generated_tokens": gen["generated_tokens"],
            "severity": row["severity"],
            "corruption_type": row["corruption_type"],
            "corruption_family": row["corruption_family"],
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
        }

        results.append(result)

        if i % 10 == 0 or i == len(rows):
            elapsed = time.time() - start_time
            print(
                f"{system_name}: {i}/{len(rows)} presentations complete "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    torch.cuda.synchronize()

    print(
        f"{system_name} evaluation complete in "
        f"{time.time() - start_time:.1f}s",
        flush=True,
    )

    return results


# ============================================================
# RESULTS
# ============================================================

def accuracy(rows, field):
    if not rows:
        return None
    return sum(int(r[field]) for r in rows) / len(rows)


def summarize_system(system_rows):
    original = [r for r in system_rows if r["orientation"] == "original"]
    swapped = [r for r in system_rows if r["orientation"] == "swapped"]
    combined = system_rows

    by_id = defaultdict(dict)
    for r in system_rows:
        by_id[r["semantic_pair_id"]][r["orientation"]] = r

    consistent = []

    for pid, pair in by_id.items():
        o = pair["original"]
        s = pair["swapped"]

        ok = (
            bool(o["free_valid"])
            and bool(s["free_valid"])
            and o["free_prediction"] != s["free_prediction"]
        )

        consistent.append(int(ok))

    return {
        "original_n": len(original),
        "original_free_accuracy": accuracy(original, "free_correct"),
        "original_invalid_count": sum(1 - int(r["free_valid"]) for r in original),

        "swapped_n": len(swapped),
        "swapped_free_accuracy": accuracy(swapped, "free_correct"),
        "swapped_invalid_count": sum(1 - int(r["free_valid"]) for r in swapped),

        "combined_n_presentations": len(combined),
        "combined_free_accuracy": accuracy(combined, "free_correct"),
        "combined_invalid_count": sum(1 - int(r["free_valid"]) for r in combined),

        "swap_consistency_n_pairs": len(consistent),
        "swap_consistency": float(np.mean(consistent)),

        "original_label_logprob_accuracy": accuracy(original, "logprob_correct"),
        "swapped_label_logprob_accuracy": accuracy(swapped, "logprob_correct"),
        "combined_label_logprob_accuracy": accuracy(combined, "logprob_correct"),
    }


def make_pair_rows(all_results):
    grouped = defaultdict(dict)

    for r in all_results:
        key = (r["system"], r["semantic_pair_id"])
        grouped[key][r["orientation"]] = r

    pair_rows = []

    for (system, pid), p in sorted(grouped.items()):
        o = p["original"]
        s = p["swapped"]

        swap_consistent = int(
            bool(o["free_valid"])
            and bool(s["free_valid"])
            and o["free_prediction"] != s["free_prediction"]
        )

        pair_rows.append({
            "system": system,
            "semantic_pair_id": pid,

            "original_gold": o["gold_label"],
            "original_prediction": o["free_prediction"],
            "original_generated_text": o["free_generated_text"],
            "original_valid": o["free_valid"],
            "original_correct": o["free_correct"],

            "swapped_gold": s["gold_label"],
            "swapped_prediction": s["free_prediction"],
            "swapped_generated_text": s["free_generated_text"],
            "swapped_valid": s["free_valid"],
            "swapped_correct": s["free_correct"],

            "combined_pair_accuracy": (
                o["free_correct"] + s["free_correct"]
            ) / 2.0,

            "swap_consistent": swap_consistent,

            "original_logprob_correct": o["logprob_correct"],
            "swapped_logprob_correct": s["logprob_correct"],

            "severity": o["severity"],
            "corruption_type": o["corruption_type"],
            "corruption_family": o["corruption_family"],
        })

    return pair_rows


def bootstrap_delta(base_values, dpo_values, seed=2026, n_boot=10000):
    base = np.asarray(base_values, dtype=float)
    dpo = np.asarray(dpo_values, dtype=float)

    assert len(base) == len(dpo)

    observed = float(dpo.mean() - base.mean())

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)

    n = len(base)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = dpo[idx].mean() - base[idx].mean()

    low, high = np.percentile(deltas, [2.5, 97.5])

    return {
        "n_pairs": n,
        "observed_delta_dpo_minus_base": observed,
        "bootstrap_n": n_boot,
        "seed": seed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_fraction_delta_gt_0": float(np.mean(deltas > 0)),
        "bootstrap_fraction_delta_eq_0": float(np.mean(deltas == 0)),
    }


def exact_mcnemar(base_correct, dpo_correct):
    b = sum(
        1
        for x, y in zip(base_correct, dpo_correct)
        if x == 1 and y == 0
    )

    c = sum(
        1
        for x, y in zip(base_correct, dpo_correct)
        if x == 0 and y == 1
    )

    n = b + c

    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        tail = sum(
            math.comb(n, i)
            for i in range(k + 1)
        ) / (2 ** n)

        p = min(1.0, 2.0 * tail)

    return {
        "base_correct_dpo_wrong": b,
        "base_wrong_dpo_correct": c,
        "discordant_total": n,
        "exact_two_sided_p": p,
    }


def write_csv(path, rows):
    if not rows:
        return

    keys = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_breakdowns(all_results):
    output = []

    original = [
        r for r in all_results
        if r["orientation"] == "original"
    ]

    for system in ["BASE", "FINAL_DPO"]:
        sr = [r for r in original if r["system"] == system]

        for field in [
            "corruption_family",
            "severity",
            "corruption_type",
        ]:
            values = sorted(set(r[field] for r in sr))

            for value in values:
                g = [r for r in sr if r[field] == value]

                output.append({
                    "system": system,
                    "group_field": field,
                    "group_value": value,
                    "n": len(g),
                    "free_accuracy": accuracy(g, "free_correct"),
                    "invalid_count": sum(
                        1 - int(r["free_valid"]) for r in g
                    ),
                    "label_logprob_accuracy": accuracy(
                        g,
                        "logprob_correct",
                    ),
                })

    return output


# ============================================================
# MAIN
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--validate-only",
    action="store_true",
)
args = parser.parse_args()

random_seed = SEED
np.random.seed(random_seed)
torch.manual_seed(random_seed)

with open(PROTOCOL_FILE, "r", encoding="utf-8") as f:
    protocol = json.load(f)

original = load_jsonl(ORIGINAL_FILE)
swapped = load_jsonl(SWAPPED_FILE)

validate_inputs(
    original,
    swapped,
    protocol,
)

if args.validate_only:
    print("\nVALIDATE-ONLY COMPLETE. NO MODEL INFERENCE WAS RUN.", flush=True)
    sys.exit(0)


if os.path.exists(OUT_DIR):
    raise RuntimeError(
        f"Output directory already exists: {OUT_DIR}. "
        "Refusing to overwrite a final frozen evaluation."
    )

os.makedirs(OUT_DIR)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

if not torch.cuda.is_bf16_supported():
    raise RuntimeError("BF16 unsupported")

device = torch.device("cuda")
dtype = torch.bfloat16

print("\nFINAL FROZEN TEST50 EVALUATION", flush=True)
print("device =", device, flush=True)
print("GPU =", torch.cuda.get_device_name(0), flush=True)
print("dtype =", dtype, flush=True)
print("torch =", torch.__version__, flush=True)
print("transformers =", transformers.__version__, flush=True)
print("peft =", peft.__version__, flush=True)

template = protocol["prompt_template"]

print("\nLoading processor/tokenizer...", flush=True)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    local_files_only=True,
)

tokenizer = processor.tokenizer

print("Loading Gemma-3-4B base...", flush=True)

base = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    local_files_only=True,
)

base.to(device)
base.eval()

torch.cuda.reset_peak_memory_stats()

# ------------------------------------------------------------
# BASE
# ------------------------------------------------------------

base_results = evaluate_system(
    "BASE",
    base,
    tokenizer,
    original,
    swapped,
    template,
    device,
)

# ------------------------------------------------------------
# FINAL DPO
# ------------------------------------------------------------

print("\nAttaching FINAL DPO LoRA...", flush=True)

dpo_model = PeftModel.from_pretrained(
    base,
    ADAPTER_DIR,
    is_trainable=False,
)

dpo_model.to(device)
dpo_model.eval()

dpo_results = evaluate_system(
    "FINAL_DPO",
    dpo_model,
    tokenizer,
    original,
    swapped,
    template,
    device,
)

all_results = base_results + dpo_results

# ------------------------------------------------------------
# RAW RESULTS
# ------------------------------------------------------------

raw_file = os.path.join(
    OUT_DIR,
    "FINAL_TEST50_all_predictions.csv",
)

write_csv(
    raw_file,
    all_results,
)

pair_rows = make_pair_rows(all_results)

write_csv(
    os.path.join(
        OUT_DIR,
        "FINAL_TEST50_pair_level.csv",
    ),
    pair_rows,
)

breakdowns = build_breakdowns(all_results)

write_csv(
    os.path.join(
        OUT_DIR,
        "FINAL_TEST50_breakdown_original.csv",
    ),
    breakdowns,
)

# ------------------------------------------------------------
# SYSTEM SUMMARIES
# ------------------------------------------------------------

base_summary = summarize_system(base_results)
dpo_summary = summarize_system(dpo_results)

summary_rows = []

for system, summary in [
    ("BASE", base_summary),
    ("FINAL_DPO", dpo_summary),
]:
    summary_rows.append({
        "system": system,
        **summary,
    })

write_csv(
    os.path.join(
        OUT_DIR,
        "FINAL_TEST50_system_summary.csv",
    ),
    summary_rows,
)

# ------------------------------------------------------------
# PAIRED STATISTICS
# ------------------------------------------------------------

pair_by_system = {
    "BASE": {},
    "FINAL_DPO": {},
}

for r in pair_rows:
    pair_by_system[r["system"]][r["semantic_pair_id"]] = r

ids = sorted(pair_by_system["BASE"].keys())

assert ids == sorted(pair_by_system["FINAL_DPO"].keys())
assert len(ids) == 50

base_primary = [
    pair_by_system["BASE"][pid]["original_correct"]
    for pid in ids
]

dpo_primary = [
    pair_by_system["FINAL_DPO"][pid]["original_correct"]
    for pid in ids
]

base_combined = [
    pair_by_system["BASE"][pid]["combined_pair_accuracy"]
    for pid in ids
]

dpo_combined = [
    pair_by_system["FINAL_DPO"][pid]["combined_pair_accuracy"]
    for pid in ids
]

base_swap = [
    pair_by_system["BASE"][pid]["swap_consistent"]
    for pid in ids
]

dpo_swap = [
    pair_by_system["FINAL_DPO"][pid]["swap_consistent"]
    for pid in ids
]

base_lp_original = [
    pair_by_system["BASE"][pid]["original_logprob_correct"]
    for pid in ids
]

dpo_lp_original = [
    pair_by_system["FINAL_DPO"][pid]["original_logprob_correct"]
    for pid in ids
]

statistics = {
    "primary_original_free_generation": {
        "bootstrap": bootstrap_delta(
            base_primary,
            dpo_primary,
            SEED,
            BOOTSTRAP_N,
        ),
        "mcnemar": exact_mcnemar(
            base_primary,
            dpo_primary,
        ),
    },

    "combined_original_plus_swapped_free_generation": {
        "bootstrap": bootstrap_delta(
            base_combined,
            dpo_combined,
            SEED,
            BOOTSTRAP_N,
        ),
    },

    "semantic_swap_consistency": {
        "bootstrap": bootstrap_delta(
            base_swap,
            dpo_swap,
            SEED,
            BOOTSTRAP_N,
        ),
    },

    "original_label_logprob_accuracy": {
        "bootstrap": bootstrap_delta(
            base_lp_original,
            dpo_lp_original,
            SEED,
            BOOTSTRAP_N,
        ),
    },
}

with open(
    os.path.join(
        OUT_DIR,
        "FINAL_TEST50_statistics.json",
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        statistics,
        f,
        ensure_ascii=False,
        indent=2,
    )

# ------------------------------------------------------------
# RUN METADATA
# ------------------------------------------------------------

metadata = {
    "experiment": "Korean XL-Sum FINAL frozen TEST50 evaluation",
    "seed": SEED,
    "bootstrap_n": BOOTSTRAP_N,

    "model_id": MODEL_ID,
    "adapter_dir": ADAPTER_DIR,

    "generation": {
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "valid_output_rule": "decoded_text.strip() must equal exactly A or B",
        "invalid_counted_as_incorrect": True,
    },

    "prompt_input_convention": {
        "source": "Frozen protocol prompt template",
        "suffix": "\\n",
        "chat_template_wrapper": False,
        "add_special_tokens": False,
        "reason": "Matches the frozen final DPO training input convention",
    },

    "files": {
        PROTOCOL_FILE: sha256_file(PROTOCOL_FILE),
        ORIGINAL_FILE: sha256_file(ORIGINAL_FILE),
        SWAPPED_FILE: sha256_file(SWAPPED_FILE),
        os.path.join(
            ADAPTER_DIR,
            "adapter_model.safetensors",
        ): sha256_file(
            os.path.join(
                ADAPTER_DIR,
                "adapter_model.safetensors",
            )
        ),
        os.path.join(
            ADAPTER_DIR,
            "adapter_config.json",
        ): sha256_file(
            os.path.join(
                ADAPTER_DIR,
                "adapter_config.json",
            )
        ),
    },

    "environment": {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    },

    "base_summary": base_summary,
    "final_dpo_summary": dpo_summary,
}

with open(
    os.path.join(
        OUT_DIR,
        "FINAL_TEST50_run_metadata.json",
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ------------------------------------------------------------
# FINAL CONSOLE REPORT
# ------------------------------------------------------------

def pct(x):
    return f"{100*x:.2f}%"


print("\n" + "=" * 90, flush=True)
print("PRIMARY RESULT — ORIGINAL ORIENTATION FREE GENERATION", flush=True)
print("=" * 90, flush=True)

print(
    "BASE      :",
    pct(base_summary["original_free_accuracy"]),
    f"invalid={base_summary['original_invalid_count']}",
    flush=True,
)

print(
    "FINAL_DPO :",
    pct(dpo_summary["original_free_accuracy"]),
    f"invalid={dpo_summary['original_invalid_count']}",
    flush=True,
)

print(
    "DPO - BASE:",
    f"{100 * (dpo_summary['original_free_accuracy'] - base_summary['original_free_accuracy']):+.2f} pp",
    flush=True,
)

print("\nSECONDARY:", flush=True)

print(
    "Combined free generation:",
    "BASE", pct(base_summary["combined_free_accuracy"]),
    "| DPO", pct(dpo_summary["combined_free_accuracy"]),
    flush=True,
)

print(
    "Swap consistency:",
    "BASE", pct(base_summary["swap_consistency"]),
    "| DPO", pct(dpo_summary["swap_consistency"]),
    flush=True,
)

print(
    "Original label-logprob:",
    "BASE", pct(base_summary["original_label_logprob_accuracy"]),
    "| DPO", pct(dpo_summary["original_label_logprob_accuracy"]),
    flush=True,
)

m = statistics["primary_original_free_generation"]["mcnemar"]

print(
    "McNemar:",
    f"base_correct/dpo_wrong={m['base_correct_dpo_wrong']}",
    f"base_wrong/dpo_correct={m['base_wrong_dpo_correct']}",
    f"p={m['exact_two_sided_p']:.6f}",
    flush=True,
)

b = statistics["primary_original_free_generation"]["bootstrap"]

print(
    "Primary paired bootstrap delta:",
    f"{100*b['observed_delta_dpo_minus_base']:+.2f} pp",
    f"95% CI [{100*b['ci95_low']:+.2f}, {100*b['ci95_high']:+.2f}] pp",
    flush=True,
)

print("\nResults saved to:", OUT_DIR, flush=True)
print("FINAL TEST50 EVALUATION COMPLETE", flush=True)
