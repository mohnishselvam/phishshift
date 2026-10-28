
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

test = pd.read_csv(ROOT / "data" / "test.csv")

models = {
    "Logistic Regression": ROOT / "reports" / "tfidf_results.csv",
    "Multinomial Naive Bayes": ROOT / "reports" / "naive_bayes_results.csv",
}

summary_rows = []
example_rows = []

for model_name, result_path in models.items():
    predictions = pd.read_csv(result_path)[
        ["message_id", "predicted_label"]
    ]

    data = test.merge(predictions, on="message_id", how="inner")

    false_positive = (
        (data["label"] == "legitimate")
        & (data["predicted_label"] == "phishing")
    )

    false_negative = (
        (data["label"] == "phishing")
        & (data["predicted_label"] == "legitimate")
    )

    summary_rows.append({
        "model": model_name,
        "test_messages": len(data),
        "false_positives": int(false_positive.sum()),
        "false_negatives": int(false_negative.sum()),
        "total_errors": int((false_positive | false_negative).sum()),
    })

    for error_type, mask in [
        ("false_positive", false_positive),
        ("false_negative", false_negative),
    ]:
        errors = data.loc[mask]

        for _, row in errors.iterrows():
            example_rows.append({
                "model": model_name,
                "error_type": error_type,
                "message_id": row["message_id"],
                "campaign_group": row["campaign_group"],
                "true_label": row["label"],
                "predicted_label": row["predicted_label"],
                "subject_or_title": row["subject_or_title"],
                "body_text": row["body_text"],
            })

summary = pd.DataFrame(summary_rows)
examples = pd.DataFrame(example_rows)

reports_dir = ROOT / "reports"
reports_dir.mkdir(exist_ok=True)

summary.to_csv(reports_dir / "error_analysis_summary.csv", index=False)
examples.to_csv(reports_dir / "error_analysis_examples.csv", index=False)

print("\nERROR ANALYSIS SUMMARY")
print(summary.to_string(index=False))

print("\nSaved:")
print("reports/error_analysis_summary.csv")
print("reports/error_analysis_examples.csv")

print(
    "\nInterpretation note: The current dataset is synthetic. "
    "Error patterns require validation on appropriately sourced, "
    "independently labelled messages before real-world conclusions."
)
