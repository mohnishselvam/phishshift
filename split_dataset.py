
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

INPUT_FILE = DATA_DIR / "messages_prepared.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

TEST_SIZE = 0.30
RANDOM_STATE = 42

REQUIRED_COLUMNS = {
    "message_id",
    "body_text",
    "label",
    "campaign_group",
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {INPUT_FILE}\n"
            "Run: python src\\prepare_dataset.py"
        )

    df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    if df.empty:
        raise ValueError("The prepared dataset is empty.")

    if df["message_id"].duplicated().any():
        raise ValueError("Duplicate message IDs exist in the prepared dataset.")

    if (df["campaign_group"].str.strip() == "").any():
        raise ValueError("Empty campaign groups exist.")

    if (df["label"].str.strip() == "").any():
        raise ValueError("Empty labels exist.")

    if df["campaign_group"].nunique() < 2:
        raise ValueError("At least two campaign groups are required.")

    if df["label"].nunique() < 2:
        raise ValueError("At least two labels are required.")

    # Assign complete campaigns to either training or testing.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            y=df["label"],
            groups=df["campaign_group"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Validate before writing any output files.
    train_groups = set(train_df["campaign_group"])
    test_groups = set(test_df["campaign_group"])
    campaign_overlap = train_groups & test_groups

    train_ids = set(train_df["message_id"])
    test_ids = set(test_df["message_id"])
    message_overlap = train_ids & test_ids

    if train_df.empty or test_df.empty:
        raise RuntimeError("The split produced an empty training or test set.")

    if campaign_overlap:
        raise RuntimeError(
            f"Campaign leakage detected: {sorted(campaign_overlap)}"
        )

    if message_overlap:
        raise RuntimeError(
            f"Message ID leakage detected: {sorted(message_overlap)}"
        )

    if train_df["label"].nunique() < 2 or test_df["label"].nunique() < 2:
        raise RuntimeError(
            "One split contains only one label. "
            "The split is unsuitable for binary classification evaluation."
        )

    # Save only after all mandatory checks pass.
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("\nPHISHMAN CAMPAIGN SPLIT")
    print("=" * 45)
    print(f"Total records:       {len(df)}")
    print(f"Training records:    {len(train_df)}")
    print(f"Testing records:     {len(test_df)}")
    print(f"Training campaigns:  {len(train_groups)}")
    print(f"Testing campaigns:   {len(test_groups)}")
    print(f"Campaign overlap:    {len(campaign_overlap)}")
    print(f"Message ID overlap:  {len(message_overlap)}")

    print("\nTraining label distribution:")
    print(train_df["label"].value_counts().to_string())

    print("\nTesting label distribution:")
    print(test_df["label"].value_counts().to_string())

    print("\nTraining campaign groups:")
    print(", ".join(sorted(train_groups)))

    print("\nTesting campaign groups:")
    print(", ".join(sorted(test_groups)))

    print("\nPASS: Campaign groups are separated.")
    print("PASS: Message IDs are separated.")
    print("PASS: Both splits contain both labels.")
    print("\nNote: This validates group separation, not real-world performance.")
    print(f"Training file: {TRAIN_FILE}")
    print(f"Testing file:  {TEST_FILE}")


if __name__ == "__main__":
    main()
