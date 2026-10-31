import argparse
import json
import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


PROMPT_TEMPLATE = """
You are evaluating factual attribution.

Task:
Given a source document and a summary, judge how well the factual claims in the summary are supported by the source.

Use only the source document. Do not use outside knowledge.

Scoring scale:
100 = The summary is fully supported by the source. All important factual claims are attributable to the source.
75 = The summary is mostly supported, with only minor unsupported or imprecise details.
50 = The summary is partially supported, but it contains some important unsupported, vague, or overgeneralized claims.
25 = The summary has limited support. Most important claims are unsupported or only weakly related to the source.
0 = The summary is not supported by the source, contradicts the source, or describes a different event.

Pay special attention to:
- named entities
- roles
- who did what to whom
- dates
- numbers
- locations
- causal relations
- outcomes
- whether the summary adds information not in the source

Return valid JSON only:
{{
  "score": 0,
  "reason": "short explanation"
}}

SOURCE:
{source}

SUMMARY:
{summary}
"""


def extract_json(text):
    text = str(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None



# ============================================================
# Q1 OVERRIDE:
# Evaluate summary quality/comprehensibility ONLY.
# The source article is intentionally NOT shown.
# ============================================================

PROMPT_TEMPLATE = """You are evaluating the quality and comprehensibility of a summary.

Judge ONLY the linguistic quality and comprehensibility of the summary itself.
Do NOT judge whether the summary is factually supported by an article.

Consider:
- grammatical correctness
- fluency
- clarity
- coherence
- repetition or broken phrasing
- whether the summary is easy to understand

Use exactly one of these scores:

0 = Very poor: largely incomprehensible or seriously broken.
25 = Poor: understandable only with substantial difficulty; major quality problems.
50 = Fair: understandable overall, but with noticeable fluency, grammar, clarity, coherence, or repetition problems.
75 = Good: clear and generally fluent, with only minor problems.
100 = Excellent: fully comprehensible, clear, coherent, grammatical, and fluent.

Evaluate only the summary below.

Summary:
{summary}

Return ONLY valid JSON in exactly this form:
{{"score": 0, "reason": "brief explanation"}}

The score MUST be exactly one of:
0, 25, 50, 75, 100
"""


def judge_one(processor, model, source, summary, max_new_tokens=256):
    prompt = PROMPT_TEMPLATE.format(
        source=str(source),
        summary=str(summary),
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    decoded = processor.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    parsed = extract_json(decoded)

    if parsed is None:
        return {
            "score": None,
            "reason": None,
            "raw_response": decoded,
            "valid_json": False,
        }

    score = parsed.get("score", None)
    reason = parsed.get("reason", None)

    try:
        score = float(score)
    except Exception:
        score = None

    valid_json = score in [0, 25, 50, 75, 100]

    return {
        "score": score,
        "reason": reason,
        "raw_response": decoded,
        "valid_json": valid_json,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if os.path.exists(args.output):
        done = pd.read_csv(args.output)
        done_row_ids = set(done["row_id"].tolist())
        results = done.to_dict("records")
        print(f"Resuming from {args.output}. Already done: {len(done_row_ids)}")
    else:
        done_row_ids = set()
        results = []
        print("Starting fresh.")

    print("Loading model:", args.model)

    processor = AutoProcessor.from_pretrained(args.model)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    for row_id, row in tqdm(df.iterrows(), total=len(df)):
        if row_id in done_row_ids:
            continue

        out = judge_one(
            processor=processor,
            model=model,
            source=row["full_source"],
            summary=row["summary"],
        )

        record = {
            "row_id": row_id,
            "lang": row.get("lang", None),
            "system": row.get("system", None),
            "human_q1_avg": row.get("q1_avg", None),
            "human_0_100": float(row.get("q1_avg")) * 100 if pd.notna(row.get("q1_avg")) else None,
            "gemma_source1000_score": row.get("gemma_source1000_score", None),
            "gemma_q1_score": out["score"],
            "gemma_q1_reason": out["reason"],
            "raw_response": out["raw_response"],
            "valid_json": out["valid_json"],
            "source_chars": row.get("source_chars", None),
            "summary_chars": row.get("summary_chars", None),
            "summary": row.get("summary", None),
            "input": row.get("full_source", None),
        }

        results.append(record)

        # checkpoint after every row
        pd.DataFrame(results).to_csv(args.output, index=False)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()
