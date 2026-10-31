"""Shared, pre-specified logic for the holdout confirmation (frozen by PROTOCOL.json).

HISTORICAL parts reproduce the recovered Q2 runner (run_gemma_fullsource_q2.py),
the Step 4C translated verifier, and the 19 Oct bridge NLI verifier. The translation
repair functions are byte-for-byte those of the frozen repair experiment.
"""

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "data" / "holdout900_inputs_NO_HUMAN_SCORES.csv"
LABELS = (
    ROOT / "data" / "holdout900_labels_SEALED.csv"
)  # read ONLY by analyze_holdout.py

MODEL_ID = "google/gemma-3-4b-it"
REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
ADAPTER_CONFIG_SHA = "cde99458eaf1fceff52cad78e46fe868eb3bfc268d4b9d54dd6b9ea01a457334"
ADAPTER_WEIGHTS_SHA = "818a9c918109536d0c3fd90598303030f0ffc5d3885b986f7153b65f95d2472d"
REPAIR_PROTOCOL_SHA = "903ff6e9cc5215fc853f9b501eebf845149433560949c1cf48092c8007b22db3"

MAX_NEW_TOKENS_HOLISTIC = 256
MAX_NEW_TOKENS_SOURCE_WHOLE = 4096
MAX_NEW_TOKENS_SUMMARY = 1024
MAX_NEW_TOKENS_EXTRACTION = 512
MAX_NEW_TOKENS_VERIFICATION = 512
MAX_CHUNK_CHARS = 1000
MAX_NEW_TOKENS_CHUNK = 1024

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261021

csv.field_size_limit(sys.maxsize)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def dump(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    )


def load_inputs():
    manifest = json.loads((ROOT / "data" / "HOLDOUT_MANIFEST.json").read_text())
    if (
        sha(INPUTS)
        != manifest["output_sha256"]["holdout900_inputs_NO_HUMAN_SCORES.csv"]
    ):
        raise RuntimeError("Holdout inputs changed")
    with INPUTS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 900 or any(
        k.startswith("q") and k.endswith("_avg") for k in rows[0]
    ):
        raise RuntimeError("Unexpected holdout input file")
    return rows


# ------------------------------------------- HISTORICAL holistic Q2 judge
Q2_PROMPT_TEMPLATE = """
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


def build_holistic_prompt(source, summary):  # HISTORICAL
    return Q2_PROMPT_TEMPLATE.format(source=str(source), summary=str(summary))


def holistic_extract_json(text):  # HISTORICAL (Q2 runner variant: no fence stripping)
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


_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)')


def holistic_score(raw):
    """HISTORICAL parse + recorded missing-only recovery. Off-anchor scores are kept."""
    parsed = holistic_extract_json(raw)
    score = None
    if isinstance(parsed, dict):
        score = parsed.get("score", None)
        try:
            score = float(score)
        except Exception:
            score = None
    valid = score in [0, 25, 50, 75, 100]
    recovered = False
    if score is None:
        m = _SCORE_RE.search(str(raw))
        if m:
            score, recovered = float(m.group(1)), True
    return {"score": score, "valid_json": valid, "recovered": recovered}


# ------------------------------------------------ bridge NLI verifier
NLI_CLASSES = ["entailment", "contradiction", "neutral"]


def build_nli_prompt(premise, hypothesis):  # BRIDGE
    return (
        (ROOT / "prompts" / "nli_ko.txt")
        .read_text(encoding="utf-8")
        .format(premise=premise, hypothesis=hypothesis)
    )


def nli_capped(labels):  # BRIDGE
    if not labels:
        return 0.0
    if any(x not in NLI_CLASSES for x in labels):
        raise ValueError("Unknown NLI label")
    p = 100 * labels.count("entailment") / len(labels)
    return (
        min(p, 25)
        if "contradiction" in labels
        else min(p, 50) if "neutral" in labels else p
    )


# ------------------------------------------------- English-likeness check
# Pre-specified rule, identical to the 20 Oct audit. It is a heuristic, not a
# validated language identifier. It uses only model outputs, never human scores.
FUNCTION_WORDS = frozenset(
    (
        "the of and to in a is that for on was with as by it he said are at from his has have be "
        "this were but they an not their which been who had will"
    ).split()
)
WORD_RE = re.compile(r"[a-zA-Z']+")


def english_metrics(text):
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    nonlatin = (
        0.0 if not letters else sum(ord(c) > 0x24F for c in letters) / len(letters)
    )
    words = WORD_RE.findall(text.lower())
    fw = 0.0 if not words else sum(w in FUNCTION_WORDS for w in words) / len(words)
    return nonlatin, fw, len(words)


def source_is_english(text):
    nonlatin, fw, _ = english_metrics(text)
    return nonlatin < 0.2 and fw >= 0.15


def summary_is_english(text):
    nonlatin, fw, _ = english_metrics(text)
    return nonlatin < 0.2 and fw >= 0.12


def _norm(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def chunk_accepted(output, source_chunk):
    """Acceptance test for one translated chunk (pre-specified)."""
    nonlatin, fw, n_words = english_metrics(output)
    if not str(output).strip() or nonlatin >= 0.2:
        return False
    if _norm(output) == _norm(source_chunk):
        return False
    if n_words < 15:
        return True
    return fw >= 0.10


# ------------------------------------------------------- segmentation
_TERMINALS = (
    ".!?\u2026\u0964\u0965\u06d4\u061f\u1362\u1367\u104b"  # . ! ? … । ॥ ۔ ؟ ። ፧ ။
)
_CJK_TERMINALS = "\u3002\uff01\uff1f"  # 。 ！ ？
_CLOSERS = "\"'\u201d\u2019\u00bb)]\u300d\u300f"


def segment(text):
    """Split text into sentence-like segments whose concatenation equals text exactly."""
    segs, start, i, n = [], 0, 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _TERMINALS or ch in _CJK_TERMINALS:
            j = i + 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            if ch in _CJK_TERMINALS:
                while j < n and text[j].isspace():
                    j += 1
                segs.append(text[start:j])
                start = i = j
                continue
            if j < n and text[j].isspace():
                while j < n and text[j].isspace():
                    j += 1
                segs.append(text[start:j])
                start = i = j
                continue
            i = j
            continue
        i += 1
    if start < n:
        segs.append(text[start:])
    return [s for s in segs if s]


def _hard_split(piece, limit):
    out = []
    while len(piece) > limit:
        cut = piece.rfind(" ", 0, limit)
        cut = cut + 1 if cut > limit // 2 else limit
        out.append(piece[:cut])
        piece = piece[cut:]
    if piece:
        out.append(piece)
    return out


def pack(text, limit=MAX_CHUNK_CHARS):
    """Greedy packing of segments into chunks of at most `limit` characters."""
    chunks, cur = [], ""
    for seg in segment(text):
        for part in _hard_split(seg, limit) if len(seg) > limit else [seg]:
            if cur and len(cur) + len(part) > limit:
                chunks.append(cur)
                cur = ""
            cur += part
    if cur:
        chunks.append(cur)
    return chunks


def split_half(chunk):
    """Deterministic split used once when a chunk translation is not accepted."""
    segs = segment(chunk)
    if len(segs) >= 2:
        total, acc, best, best_gap = len(chunk), 0, 1, None
        for k in range(1, len(segs)):
            acc += len(segs[k - 1])
            gap = abs(acc - total / 2)
            if best_gap is None or gap < best_gap:
                best, best_gap = k, gap
        return ["".join(segs[:best]), "".join(segs[best:])]
    mid = len(chunk) // 2
    cut = chunk.rfind(" ", 0, mid)
    cut = cut + 1 if cut > 0 else mid
    return [chunk[:cut], chunk[cut:]] if 0 < cut < len(chunk) else [chunk]


def translate_with_repair(text, text_type, translate_fn):
    """Chunked translation with one deterministic split-retry per rejected chunk."""
    records, outputs = [], []
    for idx, chunk in enumerate(pack(text)):
        out = translate_fn(chunk, text_type)
        rec = {
            "chunk_index": idx,
            "chunk_chars": len(chunk),
            "chunk_sha256": digest(chunk),
            "output": out,
            "accepted": chunk_accepted(out, chunk),
            "split_retry": None,
        }
        if not rec["accepted"]:
            halves = split_half(chunk)
            if len(halves) == 2:
                parts = [translate_fn(h, text_type) for h in halves]
                rec["split_retry"] = [
                    {"chars": len(h), "output": o, "accepted": chunk_accepted(o, h)}
                    for h, o in zip(halves, parts)
                ]
                out = " ".join(p.strip() for p in parts)
        outputs.append(out.strip())
        records.append(rec)
    return " ".join(o for o in outputs if o), records


# --------------------------------------------- HISTORICAL prompt builders
def _template(name):
    return (ROOT / "prompts" / name).read_text(encoding="utf-8").strip()


_PLACEHOLDER = re.compile(r"\{(text_type|text|summary_en|source_en|claim_en)\}")


def _fill(template, mapping):
    """Single-pass substitution, so inserted text is never re-scanned for placeholders."""
    return _PLACEHOLDER.sub(lambda m: mapping[m.group(1)], template)


def build_translation_prompt(text, text_type):  # HISTORICAL
    return _fill(_template("translation.txt"), {"text": text, "text_type": text_type})


def build_claim_extraction_prompt(summary_en):  # HISTORICAL
    return _fill(_template("claim_extraction.txt"), {"summary_en": summary_en})


def build_claim_verification_prompt(source_en, claim_en):  # HISTORICAL
    return _fill(
        _template("claim_verification.txt"),
        {"source_en": source_en, "claim_en": claim_en},
    )


# ----------------------------------------------- HISTORICAL parsing
def extract_json(text):  # HISTORICAL
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


def claims_from_extraction(raw):  # HISTORICAL behaviour
    js = extract_json(raw)
    valid = js is not None
    if js is not None and isinstance(js.get("claims", []), list):
        claims = [str(c).strip() for c in js.get("claims", []) if str(c).strip()]
    else:
        claims = []
    return claims, valid


def verification_record(claim, raw):  # HISTORICAL behaviour
    js = extract_json(raw)
    if js is not None:
        rec = {
            "label": js.get("label", "unsupported"),
            "evidence": js.get("evidence", ""),
            "missing_or_weaker_detail": js.get("missing_or_weaker_detail", ""),
            "reason": js.get("reason", ""),
        }
    else:
        rec = {
            "label": "unsupported",
            "evidence": "",
            "missing_or_weaker_detail": "",
            "reason": "Invalid JSON from verifier.",
        }
    rec.update({"claim": claim, "valid_json": js is not None, "raw_response": raw})
    return rec


# ------------------------------------------- HISTORICAL aggregation
def label_to_weight(label):  # HISTORICAL
    label = str(label).strip().lower()
    if label == "fully_supported":
        return 1.0
    if label == "partially_supported":
        return 0.5
    return 0.0


def capped_score(labels):  # HISTORICAL (claim_capped_score)
    labels = [str(x).strip().lower() for x in labels]
    if not labels:
        return 0.0
    prop = 100 * sum(label_to_weight(x) for x in labels) / len(labels)
    if any(x == "contradicted" for x in labels):
        return min(prop, 25.0)
    if any(x == "unsupported" for x in labels):
        return min(prop, 50.0)
    if any(x == "partially_supported" for x in labels):
        return min(prop, 75.0)
    return prop


def selective(
    b, c
):  # HISTORICAL symmetric extreme-average trigger (identical in bridge)
    if (b >= 75 and c <= 50) or (b <= 50 and c >= 75):
        return (b + c) / 2
    return b
