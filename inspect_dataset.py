
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "messages_prepared.csv"

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


def main():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        print("Run prepare_dataset.py first.")
        return

    df = pd.read_csv(DATA_FILE, dtype=str, keep_default_na=False)

    print("=" * 55)
    print("PHISHSHIFT - DATASET QUALITY REPORT")
    print("=" * 55)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("\nMissing required columns:")
        for column in missing_columns:
            print(f"  - {column}")
        return

    print(f"\nTotal prepared records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Unique message IDs: {df['message_id'].nunique()}")
    print(f"Unique campaign groups: {df['campaign_group'].nunique()}")

    print("\n1. LABEL DISTRIBUTION")
    print(df["label"].value_counts().to_string())

    print("\n2. LABEL PERCENTAGES")
    percentages = df["label"].value_counts(normalize=True).mul(100)
    for label, percentage in percentages.items():
        print(f"{label}: {percentage:.1f}%")

    print("\n3. EMPTY VALUES BY COLUMN")
    empty_counts = (df[REQUIRED_COLUMNS] == "").sum()
    print(empty_counts.to_string())

    print("\n4. DUPLICATE MESSAGE IDs")
    print(df["message_id"].duplicated().sum())

    print("\n5. LABELS PER CAMPAIGN")
    campaign_labels = (
        df.groupby("campaign_group")["label"]
        .agg(lambda labels: ", ".join(sorted(set(labels))))
    )
    print(campaign_labels.to_string())

    mixed_campaigns = campaign_labels[
        campaign_labels.str.contains(",")
    ]

    print("\n6. CAMPAIGNS CONTAINING MULTIPLE LABELS")
    if mixed_campaigns.empty:
        print("None detected.")
    else:
        print(mixed_campaigns.to_string())

    print("\n7. COLLECTION DATE CHECK")
    dates = pd.to_datetime(df["collected_date"], errors="coerce")
    print(f"Invalid or missing dates: {dates.isna().sum()}")

    print("\n8. BODY TEXT LENGTH")
    lengths = df["body_text"].str.len()
    print(f"Shortest message: {lengths.min()} characters")
    print(f"Longest message: {lengths.max()} characters")
    print(f"Average message length: {lengths.mean():.1f} characters")

    print("\nQUALITY CHECK COMPLETE")
    print("This is a diagnostic report, not proof that the dataset")
    print("is ready for final model evaluation.")


if __name__ == "__main__":
    main()
