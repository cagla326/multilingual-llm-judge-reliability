import argparse
import json
import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


PROMPT_TEMPLATE = """
You are evaluating factual attribution of a summary to a source document.

Your task is to judge whether the SUMMARY is factually supported by the SOURCE.

Use the following structured verification procedure:

1. Break the summary into atomic factual claims.
2. For each claim, decide whether it is:
   - fully supported by the source,
   - partially supported,
   - unsupported,
   - contradicted by the source,
   - impossible to verify from the source.
3. Pay special attention to:
   - entity and role mismatches,
   - causal or relational mismatches,
   - numbers, dates, quantities, and locations,
   - unsupported added details,
   - cases where only part of the summary is supported.
4. Do not give a high score just because the topic is similar. The exact factual claims must be supported.

Scoring rubric:
100 = all important factual claims are directly supported by the source.
75 = mostly supported, with only minor omissions or minor unsupported details.
50 = partially supported, but some important claims are missing, vague, or unsupported.
25 = mostly unsupported, with only small factual overlap.
0 = unsupported or contradicted by the source.

Return valid JSON only in this exact format:
{{
  "score": 0,
  "claim_checks": [
    {{
      "claim": "short atomic claim",
      "status": "fully_supported / partially_supported / unsupported / contradicted / unverifiable",
      "evidence": "short explanation"
    }}
  ],
  "reason": "short final explanation"
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

    # Fallback: extract score even if JSON is malformed, truncated, or markdown-wrapped
    score_match = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if score_match:
        return {
            "score": float(score_match.group(1)),
            "reason": text[:1000],
            "valid_json": False
        }

    # Fallback: extract score even if JSON is malformed, truncated, or markdown-wrapped
    score_match = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if score_match:
        return {
            "score": float(score_match.group(1)),
            "reason": text[:1000],
            "valid_json": False
        }

    return None


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
            "human_q2_avg": row.get("human_q2_avg", None),
            "human_0_100": row.get("human_0_100", None),
            "gemma_source1000_score": row.get("gemma_source1000_score", None),
            "gemma_hybrid_score": out["score"],
            "gemma_hybrid_reason": out["reason"],
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
