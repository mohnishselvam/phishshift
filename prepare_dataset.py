
from pathlib import Path
import hashlib
import re
from difflib import SequenceMatcher

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RAW_FILE = DATA / "messages_raw.csv"
CLEAN_FILE = DATA / "messages_prepared.csv"
DUPLICATES_FILE = DATA / "duplicate_messages.csv"
REVIEW_FILE = DATA / "near_duplicate_review.csv"

REQUIRED_COLUMNS = [
    "message_id",
    "channel",
    "subject_or_title",
    "body_text",
    "sanitized_url_features",
    "sender_domain_type",
    "label",
    "campaign_group",
    "collected_date",
]

VALID_LABELS = {"phishing", "legitimate"}
SIMILARITY_THRESHOLD = 0.90


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def make_fingerprint(row):
    combined = (
        normalize_text(row["subject_or_title"])
        + " "
        + normalize_text(row["body_text"])
    )
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def main():
    if not RAW_FILE.exists():
        print(f"Dataset not found: {RAW_FILE}")
        return

    original = pd.read_csv(RAW_FILE, dtype=str, keep_default_na=False)

    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in original.columns
    ]
    if missing_columns:
        print("Missing columns:", ", ".join(missing_columns))
        return

    df = original[REQUIRED_COLUMNS].copy()
    df["label"] = df["label"].str.strip().str.lower()

    if not df["label"].isin(VALID_LABELS).all():
        print("Invalid labels. Use 'phishing' or 'legitimate'.")
        return

    for col in ["message_id", "body_text", "campaign_group", "label"]:
        if (df[col].str.strip() == "").any():
            print(f"Empty required values in {col}.")
            return

    if df["message_id"].duplicated().any():
        print("Duplicate message_id values found.")
        return

    # Exact normalized-template duplicates.
    df["_fingerprint"] = df.apply(make_fingerprint, axis=1)
    exact_duplicate_mask = df["_fingerprint"].duplicated(keep="first")

    exact_duplicates = df.loc[exact_duplicate_mask].copy()
    retained = df.loc[~exact_duplicate_mask].copy()

    # Similarity review: flag similar messages but do not delete them
    # automatically. Human review avoids accidentally deleting useful cases.
    review_rows = []
    records = retained.reset_index(drop=True)

    normalized = [
        normalize_text(row["subject_or_title"] + " " + row["body_text"])
        for _, row in records.iterrows()
    ]

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records.loc[i, "campaign_group"] == records.loc[j, "campaign_group"]:
                # Still compare within a campaign; those are useful to inspect.
                pass

            score = SequenceMatcher(
                None, normalized[i], normalized[j]
            ).ratio()

            if score >= SIMILARITY_THRESHOLD:
                review_rows.append({
                    "message_id_1": records.loc[i, "message_id"],
                    "message_id_2": records.loc[j, "message_id"],
                    "campaign_group_1": records.loc[i, "campaign_group"],
                    "campaign_group_2": records.loc[j, "campaign_group"],
                    "label_1": records.loc[i, "label"],
                    "label_2": records.loc[j, "label"],
                    "similarity": round(score, 4),
                    "review_status": "needs_manual_review",
                })

    exact_duplicates.drop(columns=["_fingerprint"]).to_csv(
        DUPLICATES_FILE, index=False
    )
    retained.drop(columns=["_fingerprint"]).to_csv(
        CLEAN_FILE, index=False
    )
    pd.DataFrame(
        review_rows,
        columns=[
            "message_id_1", "message_id_2",
            "campaign_group_1", "campaign_group_2",
            "label_1", "label_2", "similarity", "review_status",
        ],
    ).to_csv(REVIEW_FILE, index=False)

    print("\nPHISHSHIFT DATA PREPARATION")
    print("-" * 40)
    print(f"Raw records: {len(original)}")
    print(f"Exact duplicates removed: {len(exact_duplicates)}")
    print(f"Records retained: {len(retained)}")
    print(f"Similar pairs flagged for review: {len(review_rows)}")
    print(f"Prepared data: {CLEAN_FILE}")
    print(f"Exact duplicates: {DUPLICATES_FILE}")
    print(f"Near-duplicate review: {REVIEW_FILE}")
    print("\nLabel distribution:")
    print(retained["label"].value_counts().to_string())
    print("\nNote: Similarity flags need manual review.")
    print("Campaign-level splitting is a separate later step.")


if __name__ == "__main__":
    main()
