import argparse
import json
import re
import pandas as pd
import numpy as np
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


def generate_response(model, processor, prompt, max_new_tokens=1024):
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
    return processor.decode(generated, skip_special_tokens=True).strip()


def build_translation_prompt(text, text_type):
    return f"""
Translate the following {text_type} into English.

Rules:
- Preserve all names, dates, numbers, quantities, places, organizations, and roles exactly.
- Do not summarize.
- Do not add information.
- Do not omit factual details.
- Return only the English translation, with no commentary.

Text:
{text}
""".strip()


def build_claim_extraction_prompt(summary_en):
    return f"""
You are decomposing a candidate summary into atomic factual claims.

Rules:
- Extract only factual claims stated in the summary.
- Split compound claims into separate atomic claims.
- Each claim should be checkable against a source document.
- Do not add information that is not in the summary.
- Do not judge whether the claims are true yet.
- If the summary contains no factual claim, return an empty list.

Candidate summary in English:
{summary_en}

Return valid JSON only:
{{
  "claims": [
    "claim 1",
    "claim 2"
  ]
}}
""".strip()


def build_claim_verification_prompt(source_en, claim_en):
    return f"""
You are verifying ONE atomic factual claim against a source document.

Your job is strict factual attribution.

First, check the claim normally.
Then, before giving the final label, silently ask yourself:

"Am I overestimating support because the source is only topically related, only partially supports the claim, or supports a weaker version of the claim?"

A claim is fully_supported ONLY IF:
- the exact same main entity is present in the source,
- the exact same action, role, relation, or event is present in the source,
- all numbers, dates, quantities, locations, and names match,
- no important detail is missing,
- the evidence directly supports the full claim, not just a related weaker claim.

Use these labels:
- fully_supported: the complete claim is directly supported.
- partially_supported: some parts are supported, but an important detail is missing, weaker, vague, or uncertain.
- unsupported: the source is related but does not support the exact claim.
- contradicted: the source says something incompatible with the claim.

Important:
If you are uncertain between fully_supported and partially_supported, choose partially_supported.
If you are uncertain between partially_supported and unsupported, choose unsupported.

English source document:
{source_en}

English claim:
{claim_en}

Return valid JSON only:
{{
  "label": "fully_supported/partially_supported/unsupported/contradicted",
  "evidence": "short exact evidence span from the English source, or empty string",
  "missing_or_weaker_detail": "what is missing, weaker, or wrong; empty if none",
  "reason": "brief explanation"
}}
""".strip()


def label_to_weight(label):
    label = str(label).strip().lower()

    if label == "fully_supported":
        return 1.0
    if label == "partially_supported":
        return 0.5
    if label in ["unsupported", "contradicted"]:
        return 0.0

    return 0.0


def aggregate_scores(labels):
    labels = [str(x).strip().lower() for x in labels]
    n = len(labels)

    if n == 0:
        return {
            "claim_prop_score": 0.0,
            "claim_capped_score": 0.0,
            "claim_strict_score": 0.0,
        }

    weights = np.array([label_to_weight(x) for x in labels], dtype=float)
    prop = float(100 * weights.mean())

    has_partial = any(x == "partially_supported" for x in labels)
    has_unsupported = any(x == "unsupported" for x in labels)
    has_contradicted = any(x == "contradicted" for x in labels)

    all_full = all(x == "fully_supported" for x in labels)
    all_bad = all(x in ["unsupported", "contradicted"] for x in labels)

    capped = prop

    if has_contradicted:
        capped = min(capped, 25.0)
    elif has_unsupported:
        capped = min(capped, 50.0)
    elif has_partial:
        capped = min(capped, 75.0)

    if all_full:
        strict = 100.0
    elif all_bad:
        strict = 0.0
    elif has_contradicted:
        strict = 25.0
    elif has_unsupported:
        strict = 50.0
    elif has_partial:
        strict = 75.0
    else:
        strict = 0.0

    return {
        "claim_prop_score": prop,
        "claim_capped_score": capped,
        "claim_strict_score": strict,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if "row_id" not in df.columns:
        if "id" in df.columns:
            df["row_id"] = df["id"]
        else:
            df = df.reset_index().rename(columns={"index": "row_id"})

    df = df.head(args.limit).copy()

    print("Loading model:", MODEL_ID)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        source = str(row["input"])
        summary = str(row["summary"])

        source_en = generate_response(
            model,
            processor,
            build_translation_prompt(source, "source document"),
            max_new_tokens=4096,
        )

        summary_en = generate_response(
            model,
            processor,
            build_translation_prompt(summary, "candidate summary"),
            max_new_tokens=1024,
        )

        raw_claims = generate_response(
            model,
            processor,
            build_claim_extraction_prompt(summary_en),
            max_new_tokens=512,
        )

        claims_json = extract_json(raw_claims)
        claims_valid_json = claims_json is not None

        if claims_json is not None and isinstance(claims_json.get("claims", []), list):
            claims = [
                str(c).strip() for c in claims_json.get("claims", []) if str(c).strip()
            ]
        else:
            claims = []

        verification_results = []
        labels = []

        for claim in claims:
            raw_verification = generate_response(
                model,
                processor,
                build_claim_verification_prompt(source_en, claim),
                max_new_tokens=512,
            )

            ver_json = extract_json(raw_verification)
            ver_valid_json = ver_json is not None

            if ver_json is not None:
                label = ver_json.get("label", "unsupported")
                evidence = ver_json.get("evidence", "")
                missing_or_weaker_detail = ver_json.get("missing_or_weaker_detail", "")
                reason = ver_json.get("reason", "")
            else:
                label = "unsupported"
                evidence = ""
                missing_or_weaker_detail = ""
                reason = "Invalid JSON from verifier."

            labels.append(label)

            verification_results.append(
                {
                    "claim": claim,
                    "label": label,
                    "evidence": evidence,
                    "missing_or_weaker_detail": missing_or_weaker_detail,
                    "reason": reason,
                    "valid_json": ver_valid_json,
                    "raw_response": raw_verification,
                }
            )

        agg = aggregate_scores(labels)

        out = row.to_dict()
        out.update(
            {
                "translated_source_en": source_en,
                "translated_summary_en": summary_en,
                "num_claims": len(claims),
                "claims_json_valid": claims_valid_json,
                "claims": json.dumps(claims, ensure_ascii=False),
                "raw_claim_extraction": raw_claims,
                "claim_verifications": json.dumps(
                    verification_results, ensure_ascii=False
                ),
                "claim_labels": json.dumps(labels, ensure_ascii=False),
                **agg,
            }
        )

        rows.append(out)
        pd.DataFrame(rows).to_csv(args.output, index=False)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()
