
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def test_required_columns_exist():
    df = pd.read_csv(DATA_DIR / "messages_prepared.csv")

    required = {
        "message_id",
        "channel",
        "subject_or_title",
        "body_text",
        "sanitized_url_features",
        "sender_domain_type",
        "label",
        "campaign_group",
        "collected_date",
    }

    assert required.issubset(df.columns)


def test_message_ids_are_unique():
    df = pd.read_csv(DATA_DIR / "messages_prepared.csv")
    assert df["message_id"].is_unique


def test_labels_are_valid():
    df = pd.read_csv(DATA_DIR / "messages_prepared.csv")
    assert set(df["label"].dropna().unique()) == {
        "legitimate",
        "phishing",
    }


def test_no_exact_duplicate_fingerprints():
    df = pd.read_csv(DATA_DIR / "messages_prepared.csv")

    normalized = (
        df["subject_or_title"].fillna("").astype(str).str.lower().str.strip()
        + " "
        + df["body_text"].fillna("").astype(str).str.lower().str.strip()
    )

    assert not normalized.duplicated().any()


def test_campaigns_do_not_overlap_between_splits():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    train_campaigns = set(train["campaign_group"])
    test_campaigns = set(test["campaign_group"])

    assert train_campaigns.isdisjoint(test_campaigns)


def test_splits_have_both_labels():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    expected = {"legitimate", "phishing"}

    assert set(train["label"].unique()) == expected
    assert set(test["label"].unique()) == expected


if __name__ == "__main__":
    tests = [
        test_required_columns_exist,
        test_message_ids_are_unique,
        test_labels_are_valid,
        test_no_exact_duplicate_fingerprints,
        test_campaigns_do_not_overlap_between_splits,
        test_splits_have_both_labels,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"\nAll {len(tests)} tests passed.")
