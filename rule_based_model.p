
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "test.csv"
REPORTS_DIR = ROOT / "reports"

PHISHING_RULES = {
    "credential_request": [
        "verify your password",
        "confirm your password",
        "enter your password",
        "login credentials",
        "verify your account",
        "confirm your account",
    ],
    "suspicious_link": [
        "click here to restore",
        "click the link",
        "open the link",
        "sign in immediately",
        "secure your account",
    ],
    "financial_request": [
        "bank account details",
        "confirm your payment",
        "update your payment",
        "payment failed",
    ],
    "threat_or_reward": [
        "account will be suspended",
        "account has been suspended",
        "claim your reward",
        "you have won",
    ],
}

URGENT_PHRASES = [
    "urgent",
    "immediately",
    "right now",
    "within 24 hours",
    "final warning",
]


def classify_message(text):
    text = str(text).lower()
    matched_rules = []

    for rule_name, phrases in PHISHING_RULES.items():
        if any(phrase in text for phrase in phrases):
            matched_rules.append(rule_name)

    urgent_matches = [
        phrase for phrase in URGENT_PHRASES
        if phrase in text
    ]

    if matched_rules:
        score = min(0.95, 0.60 + 0.10 * (len(matched_rules) - 1))
        explanation = "Matched rules: " + ", ".join(matched_rules)
        return "phishing", score, explanation

    if urgent_matches:
        return (
            "review",
            0.50,
            "Urgency alone is insufficient; manual review recommended."
        )

    return "legitimate", 0.10, "No configured phishing rule matched."


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {DATA_FILE}\n"
            "Run python src\\split_dataset.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = {"message_id", "subject_or_title", "body_text", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    results = []

    for _, row in df.iterrows():
        text = (
            str(row["subject_or_title"]) + " " +
            str(row["body_text"])
        )

        prediction, score, explanation = classify_message(text)

        results.append({
            "message_id": row["message_id"],
            "campaign_group": row.get("campaign_group", ""),
            "true_label": row["label"],
            "predicted_label": prediction,
            "risk_score": score,
            "explanation": explanation,
        })

    result_df = pd.DataFrame(results)
    results_path = REPORTS_DIR / "rule_based_results.csv"
    metrics_path = REPORTS_DIR / "rule_based_metrics.txt"

    result_df.to_csv(results_path, index=False)

    # Review cases are abstentions, not automatic predictions.
    evaluated = result_df[
        result_df["predicted_label"].isin(["phishing", "legitimate"])
    ].copy()

    lines = [
        "PHISHSHIFT RULE-BASED BASELINE",
        f"Total test messages: {len(result_df)}",
        f"Automatically classified: {len(evaluated)}",
        f"Sent for review: {(result_df['predicted_label'] == 'review').sum()}",
    ]

    if not evaluated.empty:
        y_true = evaluated["true_label"]
        y_pred = evaluated["predicted_label"]

        lines.extend([
            f"Accuracy: {accuracy_score(y_true, y_pred):.4f}",
            f"Phishing precision: {precision_score(y_true, y_pred, pos_label='phishing', zero_division=0):.4f}",
            f"Phishing recall: {recall_score(y_true, y_pred, pos_label='phishing', zero_division=0):.4f}",
            f"Phishing F1: {f1_score(y_true, y_pred, pos_label='phishing', zero_division=0):.4f}",
        ])

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=["legitimate", "phishing"]
        )

        lines.extend([
            "",
            "Confusion matrix (rows=true, columns=predicted)",
            "Labels: legitimate, phishing",
            str(matrix),
            f"True negatives: {matrix[0, 0]}",
            f"False positives: {matrix[0, 1]}",
            f"False negatives: {matrix[1, 0]}",
            f"True positives: {matrix[1, 1]}",
        ])
    else:
        lines.append("No automatically classified records; metrics unavailable.")

    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nResults saved to: {results_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
