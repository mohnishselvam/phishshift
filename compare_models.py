
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"

INPUT_FILES = {
    "Rule-Based": REPORTS_DIR / "rule_based_results.csv",
    "Logistic Regression": REPORTS_DIR / "tfidf_results.csv",
    "Multinomial Naive Bayes": REPORTS_DIR / "naive_bayes_results.csv",
}

OUTPUT_FILE = REPORTS_DIR / "model_comparison.csv"

VALID_LABELS = {"legitimate", "phishing"}
VALID_PREDICTIONS = VALID_LABELS | {"review"}


def calculate_metrics(model_name, df):
    required = {"message_id", "label", "predicted_label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name}: missing columns {sorted(missing)}"
        )

    if df["message_id"].duplicated().any():
        raise ValueError(f"{model_name}: duplicate message IDs found.")

    if not set(df["label"]).issubset(VALID_LABELS):
        raise ValueError(f"{model_name}: unexpected ground-truth labels.")

    if not set(df["predicted_label"]).issubset(VALID_PREDICTIONS):
        invalid = sorted(set(df["predicted_label"]) - VALID_PREDICTIONS)
        raise ValueError(
            f"{model_name}: unexpected predictions: {invalid}"
        )

    review_mask = df["predicted_label"].eq("review")
    automatic = df.loc[~review_mask].copy()

    total = len(df)
    automatic_count = len(automatic)
    review_count = int(review_mask.sum())

    if automatic_count:
        true_phishing = automatic["label"].eq("phishing")
        predicted_phishing = automatic["predicted_label"].eq("phishing")

        tp = int((true_phishing & predicted_phishing).sum())
        fp = int((~true_phishing & predicted_phishing).sum())
        fn = int((true_phishing & ~predicted_phishing).sum())
        tn = int((~true_phishing & ~predicted_phishing).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        accuracy = (tp + tn) / automatic_count
    else:
        tp = fp = fn = tn = 0
        precision = recall = f1 = accuracy = 0.0

    return {
        "model": model_name,
        "test_messages": total,
        "automatic_predictions": automatic_count,
        "review_cases": review_count,
        "coverage": round(automatic_count / total, 4) if total else 0.0,
        "review_rate": round(review_count / total, 4) if total else 0.0,
        "accuracy_on_automatic_predictions": round(accuracy, 4),
        "phishing_precision": round(precision, 4),
        "phishing_recall": round(recall, 4),
        "phishing_f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    available = {}
    missing = []

    for model_name, file_path in INPUT_FILES.items():
        if file_path.exists():
            available[model_name] = pd.read_csv(
                file_path,
                dtype=str,
                keep_default_na=False,
            )
        else:
            missing.append(model_name)

    if missing:
        print("Results are missing for these models:")
        for model_name in missing:
            print(f" - {model_name}")
        print("\nRun the corresponding model scripts first.")
        return

    # Confirm all models evaluated exactly the same test records.
    reference = available["Rule-Based"][["message_id", "label"]].copy()

    for model_name, df in available.items():
        required = {"message_id", "label", "predicted_label"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{model_name}: required result columns are missing."
            )

        if df["message_id"].duplicated().any():
            raise ValueError(f"{model_name}: duplicate message IDs found.")

        comparison = df[["message_id", "label"]].copy()
        merged = reference.merge(
            comparison,
            on="message_id",
            how="outer",
            suffixes=("_reference", "_model"),
            indicator=True,
        )

        if not merged["_merge"].eq("both").all():
            raise ValueError(
                f"{model_name}: evaluated message IDs differ."
            )

        if not merged["label_reference"].eq(
            merged["label_model"]
        ).all():
            raise ValueError(
                f"{model_name}: ground-truth labels differ."
            )

    rows = [
        calculate_metrics(model_name, df)
        for model_name, df in available.items()
    ]

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_FILE, index=False)

    print("\nPHISHMAN MODEL COMPARISON")
    print("=" * 100)
    print(report.to_string(index=False))
    print(f"\nSaved comparison report: {OUTPUT_FILE}")
    print(
        "\nInterpret results cautiously: the dataset is synthetic. "
        "Coverage is the share classified automatically; accuracy is "
        "calculated only among those automatic predictions."
    )


if __name__ == "__main__":
    main()
