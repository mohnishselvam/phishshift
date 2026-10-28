
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

test = pd.read_csv(ROOT / "data" / "test.csv")

models = {
    "Logistic Regression": ROOT / "reports" / "tfidf_results.csv",
    "Multinomial Naive Bayes": ROOT / "reports" / "naive_bayes_results.csv",
}

for model_name, result_path in models.items():
    predictions = pd.read_csv(result_path)[
        ["message_id", "predicted_label"]
    ]

    data = test.merge(predictions, on="message_id", how="inner")

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    error_groups = [
        (
            "FALSE POSITIVES",
            (data["label"] == "legitimate")
            & (data["predicted_label"] == "phishing"),
        ),
        (
            "FALSE NEGATIVES",
            (data["label"] == "phishing")
            & (data["predicted_label"] == "legitimate"),
        ),
    ]

    for title, mask in error_groups:
        errors = data.loc[
            mask,
            [
                "message_id",
                "campaign_group",
                "label",
                "predicted_label",
                "subject_or_title",
                "body_text",
            ],
        ]

        print(f"\n{title}: {len(errors)} total")
        print(errors.head(3).to_string(index=False))

print("\nError inspection complete.")
