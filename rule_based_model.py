
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "test.csv"
REPORTS_DIR = ROOT / "reports"

PHISHING_PHRASES = [
    "verify your password",
    "confirm your password",
    "enter your password",
    "verify your account",
    "confirm your account",
    "login credentials",
    "click here to restore",
    "sign in immediately",
    "bank account details",
    "confirm your payment",
    "update your payment",
    "payment failed",
    "account will be suspended",
    "account has been suspended",
    "claim your reward",
    "you have won",
]

URGENT_PHRASES = [
    "urgent",
    "immediately",
    "right now",
    "within 24 hours",
    "final warning",
]


def classify_message(text):
    text = str(text).lower()

    matches = [phrase for phrase in PHISHING_PHRASES if phrase in text]
    urgent_matches = [phrase for phrase in URGENT_PHRASES if phrase in text]

    if matches:
        score = min(0.95, 0.60 + 0.10 * (len(matches) - 1))
        explanation = "Matched phrases: " + ", ".join(matches)
        return "phishing", score, explanation

    if urgent_matches:
        return (
            "review",
            0.50,
            "Urgency alone is insufficient; manual review recommended.",
        )

    return "legitimate", 0.10, "No configured phishing phrase matched."


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. "
            "Run python src\\split_dataset.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = {"message_id", "subject_or_title", "body_text", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rows = []

    for _, row in df.iterrows():
        message = f"{row['subject_or_title']} {row['body_text']}"
        prediction, score, explanation = classify_message(message)

        rows.append({
            "message_id": row["message_id"],
            "campaign_group": row.get("campaign_group", ""),
            "label": row["label"],
            "predicted_label": prediction,
            "risk_score": score,
            "explanation": explanation,
        })

    results = pd.DataFrame(rows)
    results_path = REPORTS_DIR / "rule_based_results.csv"
    metrics_path = REPORTS_DIR / "rule_based_metrics.txt"
    results.to_csv(results_path, index=False)

    evaluated = results[
        results["predicted_label"].isin(["phishing", "legitimate"])
    ]

    report = [
        "PHISHSHIFT RULE-BASED BASELINE",
        f"Total test messages: {len(results)}",
        f"Automatically classified: {len(evaluated)}",
        f"Sent for review: {(results['predicted_label'] == 'review').sum()}",
    ]

    if not evaluated.empty:
        y_true = evaluated["label"]
        y_pred = evaluated["predicted_label"]

        report.extend([
            f"Accuracy: {accuracy_score(y_true, y_pred):.4f}",
            f"Phishing precision: {precision_score(y_true, y_pred, pos_label='phishing', zero_division=0):.4f}",
            f"Phishing recall: {recall_score(y_true, y_pred, pos_label='phishing', zero_division=0):.4f}",
            f"Phishing F1: {f1_score(y_true, y_pred, pos_label='phishing', zero_division=0):.4f}",
            "",
            "Confusion matrix (rows=true, columns=predicted)",
            "Label order: legitimate, phishing",
        ])

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=["legitimate", "phishing"],
        )

        report.extend([
            str(matrix),
            f"True negatives: {matrix[0, 0]}",
            f"False positives: {matrix[0, 1]}",
            f"False negatives: {matrix[1, 0]}",
            f"True positives: {matrix[1, 1]}",
        ])
    else:
        report.append("No automatic predictions; metrics unavailable.")

    metrics_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nResults saved: {results_path}")
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
