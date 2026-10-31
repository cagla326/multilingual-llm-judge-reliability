import os
import pandas as pd
import textwrap

INPUT = "Q2_ERROR_AUDIT50_MANUAL_INPUT.csv"
OUTPUT = "Q2_ERROR_AUDIT50_ANNOTATIONS.csv"

LABELS = {
    "1": "partial_support",
    "2": "entity_role",
    "3": "causal_relational",
    "4": "number_date_quantity",
    "5": "other",
}

df = pd.read_csv(INPUT)

# Separate editable annotation file.
if os.path.exists(OUTPUT):
    ann = pd.read_csv(OUTPUT)
else:
    ann = df[["audit_id", "row_id"]].copy()
    ann["manual_primary_type"] = ""
    ann["manual_secondary_type"] = ""
    ann["manual_confidence"] = ""
    ann["manual_notes"] = ""
    ann.to_csv(OUTPUT, index=False)


def clean(x):
    if pd.isna(x):
        return ""
    return str(x)


def wrap(text, width=110):
    text = clean(text)
    return "\n".join(textwrap.fill(p, width=width) for p in text.splitlines())


for i, row in df.iterrows():

    aid = row["audit_id"]

    old = ann.loc[ann["audit_id"] == aid, "manual_primary_type"].iloc[0]

    if isinstance(old, str) and old.strip():
        continue

    print("\n" + "=" * 120)
    print(f"{aid}   CASE {i+1}/50")
    print("=" * 120)

    print(
        f"Language: {row.get('lang')}   "
        f"Human Q2: {row.get('human_0_100')}   "
        f"Gemma: {row.get('gemma_fullsource_score')}   "
        f"Over-score: +{row.get('baseline_error')}"
    )

    print("\n--- ENGLISH SOURCE ---")
    src = row.get("translated_source_en")

    if pd.isna(src) or not str(src).strip():
        src = row.get("full_source", row.get("input", ""))

    print(wrap(src))

    print("\n--- ENGLISH SUMMARY ---")
    summ = row.get("translated_summary_en")

    if pd.isna(summ) or not str(summ).strip():
        summ = row.get("summary", "")

    print(wrap(summ))

    print("\n--- ORIGINAL SUMMARY ---")
    print(wrap(row.get("summary", "")))

    print("\nPRIMARY FAILURE TYPE")
    print("  1 = partial_support")
    print(
        "      Some content is supported, but unsupported/incorrect detail was rewarded too much."
    )
    print()
    print("  2 = entity_role")
    print("      Wrong person, entity, role, actor, attribution, location, etc.")
    print()
    print("  3 = causal_relational")
    print(
        "      Wrong causal relation, direction, condition, status, policy relationship, etc."
    )
    print()
    print("  4 = number_date_quantity")
    print("      Wrong number, date, age, percentage, duration, amount, quantity, etc.")
    print()
    print("  5 = other")
    print("      Main error does not fit the four categories.")
    print()
    print("  s = skip")
    print("  q = save and quit")

    while True:
        choice = input("\nPrimary label: ").strip().lower()

        if choice == "q":
            ann.to_csv(OUTPUT, index=False)
            print("\nSaved:", OUTPUT)
            raise SystemExit

        if choice == "s":
            break

        if choice in LABELS:
            primary = LABELS[choice]
            break

        print("Use 1, 2, 3, 4, 5, s, or q.")

    if choice == "s":
        continue

    secondary = input("Optional secondary label [1-5 or ENTER]: ").strip()

    secondary_label = LABELS.get(secondary, "")

    while True:
        confidence = input("Confidence [H/M/L]: ").strip().upper()

        if confidence in {"H", "M", "L"}:
            break

        print("Enter H, M, or L.")

    notes = input("Short note explaining the error: ").strip()

    mask = ann["audit_id"] == aid

    ann.loc[mask, "manual_primary_type"] = primary
    ann.loc[mask, "manual_secondary_type"] = secondary_label
    ann.loc[mask, "manual_confidence"] = confidence
    ann.loc[mask, "manual_notes"] = notes

    # Save after every case.
    ann.to_csv(OUTPUT, index=False)

    completed = (
        ann["manual_primary_type"].fillna("").astype(str).str.strip().ne("").sum()
    )

    print(f"\nSaved. Completed: {completed}/50")

print("\n" + "=" * 120)
print("ALL 50 CASES ANNOTATED")
print("=" * 120)
print("Saved:", OUTPUT)

print("\nPrimary type counts:")
print(ann["manual_primary_type"].value_counts().to_string())
