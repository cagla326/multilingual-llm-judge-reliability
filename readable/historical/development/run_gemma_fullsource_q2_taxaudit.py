import argparse
import json
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


MODEL_ID = "google/gemma-3-4b-it"


def extract_json(text):
    if not isinstance(text, str):
        return None

    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def extract_score_fallback(text, keys=("score", "revised_score")):
    if not isinstance(text, str):
        return None

    for key in keys:
        pattern = rf'["\']?{key}["\']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            score = float(match.group(1))
            if 0 <= score <= 100:
                return score

    return None


def generate_response(model, processor, prompt, max_new_tokens=512):
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

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = output[0][input_len:]
    text = processor.decode(generated, skip_special_tokens=True)
    return text.strip()


def build_initial_prompt(source, summary):
    return f"""
You are evaluating factual attribution.

Given a source document and a candidate summary, decide how well the summary is supported by the source.

Use this scale:
100 = fully supported by the source.
75 = mostly supported, with only minor missing details.
50 = partially supported, but important information is missing or uncertain.
25 = weakly supported, with major unsupported or contradicted claims.
0 = unsupported or contradicted by the source.

Source document:
{source}

Candidate summary:
{summary}

Return valid JSON only:
{{
  "score": 0,
  "reason": "brief explanation"
}}
""".strip()


def build_audit_prompt(source, summary, initial_score, initial_reason):
    return f"""
You are auditing a previous factual attribution score.

You are given:
1. the source document,
2. the candidate summary,
3. the previous score,
4. the previous reason.

Your task is to decide whether the previous score is too high, too low, or appropriate.

Use these common factual-attribution failure types:

1. Partial support over-rewarded:
The summary contains some correct information, but important claims are missing, unsupported, contradicted, or only partly supported.

2. Entity or role mismatch:
The summary names the wrong person, organization, country, role, actor, or participant.

3. Causal or relational mismatch:
The summary gets the relation, cause, responsibility, event order, implication, or comparison wrong.

4. Number, date, or quantity mismatch:
The summary changes a number, date, amount, duration, age, location count, or other factual quantity.

If any of these problems affect an important claim, the previous score should be lowered.

However, do not lower the score just because the summary is short. If the summary's claims are directly supported by the source, keep or raise the score.

Source document:
{source}

Candidate summary:
{summary}

Previous score:
{initial_score}

Previous reason:
{initial_reason}

Return valid JSON only:
{{
  "revised_score": 0,
  "audit_decision": "keep/lower/raise",
  "failure_type": "none/partial_support/entity_role/causal_relation/number_date_quantity/other",
  "reason": "brief explanation"
}}
""".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Ensure stable row_id for later merging/comparison.
    # The frozen input may use "id" instead of "row_id".
    if "row_id" not in df.columns:
        if "id" in df.columns:
            df["row_id"] = df["id"]
        else:
            df = df.reset_index().rename(columns={"index": "row_id"})

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("Loading model:", MODEL_ID)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        source = str(row["input"])
        summary = str(row["summary"])

        initial_prompt = build_initial_prompt(source, summary)
        raw_initial = generate_response(
            model,
            processor,
            initial_prompt,
            max_new_tokens=512,
        )

        initial_json = extract_json(raw_initial)
        initial_valid_json = initial_json is not None

        if initial_json is not None:
            initial_score = initial_json.get("score", None)
            initial_reason = initial_json.get("reason", "")
        else:
            initial_score = extract_score_fallback(raw_initial, keys=("score",))
            initial_reason = ""

        try:
            initial_score = float(initial_score)
        except Exception:
            initial_score = None

        audit_prompt = build_audit_prompt(
            source=source,
            summary=summary,
            initial_score=initial_score,
            initial_reason=initial_reason,
        )

        raw_audit = generate_response(
            model,
            processor,
            audit_prompt,
            max_new_tokens=512,
        )

        audit_json = extract_json(raw_audit)
        audit_valid_json = audit_json is not None

        if audit_json is not None:
            audit_score = audit_json.get("revised_score", None)
            audit_decision = audit_json.get("audit_decision", "")
            failure_type = audit_json.get("failure_type", "")
            audit_reason = audit_json.get("reason", "")
        else:
            audit_score = extract_score_fallback(
                raw_audit, keys=("revised_score", "score")
            )
            audit_decision = ""
            failure_type = ""
            audit_reason = ""

        try:
            audit_score = float(audit_score)
        except Exception:
            audit_score = None

        final_score = audit_score if audit_score is not None else initial_score

        out = row.to_dict()
        out.update(
            {
                "initial_score": initial_score,
                "initial_reason": initial_reason,
                "initial_valid_json": initial_valid_json,
                "raw_initial_response": raw_initial,
                "audit_score": audit_score,
                "audit_decision": audit_decision,
                "failure_type": failure_type,
                "audit_reason": audit_reason,
                "audit_valid_json": audit_valid_json,
                "raw_audit_response": raw_audit,
                "gemma_fullsource_score": final_score,
                "valid_json": audit_valid_json,
            }
        )

        rows.append(out)
        pd.DataFrame(rows).to_csv(args.output, index=False)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()
