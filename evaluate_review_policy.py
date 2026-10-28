
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = ROOT / "data" / "test.csv"
OUTPUT_PATH = ROOT / "reports" / "review_policy_results.csv"

MODEL_FILES = {
    "Logistic Regression": (
        ROOT / "models" / "tfidf_logistic_regression.joblib",
        ROOT / "models" / "tfidf_vectorizer.joblib",
    ),
    "Multinomial Naive Bayes": (
        ROOT / "models" / "naive_bayes_model.joblib",
        ROOT / "models" / "naive_bayes_vectorizer.joblib",
    ),
}

LOWER_THRESHOLD = 0.40
UPPER_THRESHOLD = 0.60


def get_phishing_probability(model, vectorizer, text):
    features = vectorizer.transform([text])
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    if "phishing" not in classes:
        raise ValueError(f"Unexpected model classes: {classes}")

    return float(probabilities[classes.index("phishing")])


def assign_decision(probability):
    if probability < LOWER_THRESHOLD:
        return "legitimate"
    if probability <= UPPER_THRESHOLD:
        return "review"
    return "phishing"


def main():
    test = pd.read_csv(TEST_PATH)

    required = {"message_id", "campaign_group", "subject_or_title",
                "body_text", "label"}
    missing = required - set(test.columns)

    if missing:
        raise ValueError(f"Missing test columns: {sorted(missing)}")

    results = []

    for model_name, (model_path, vectorizer_path) in MODEL_FILES.items():
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        for _, row in test.iterrows():
            text = f"{row['subject_or_title']} {row['body_text']}"
            probability = get_phishing_probability(
                model, vectorizer, text
            )

            results.append({
                "model": model_name,
                "message_id": row["message_id"],
                "campaign_group": row["campaign_group"],
                "true_label": row["label"],
                "phishing_probability": round(probability, 6),
                "decision": assign_decision(probability),
            })

    report = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print("\nPHISHSHIFT REVIEW POLICY EVALUATION")
    print("=" * 70)
    print(f"Review band: {LOWER_THRESHOLD:.2f} to {UPPER_THRESHOLD:.2f}")

    for model_name in MODEL_FILES:
        subset = report[report["model"] == model_name]
        automatic = subset[subset["decision"] != "review"]
        reviewed = subset[subset["decision"] == "review"]

        if len(automatic):
            tn, fp, fn, tp = confusion_matrix(
                automatic["true_label"],
                automatic["decision"],
                labels=["legitimate", "phishing"],
            ).ravel()
            accuracy = (tp + tn) / len(automatic)
        else:
            tn = fp = fn = tp = 0
            accuracy = float("nan")

        print(f"\nModel: {model_name}")
        print(f"Total test messages: {len(subset)}")
        print(f"Automatic decisions: {len(automatic)}")
        print(f"Sent for review: {len(reviewed)}")
        print(f"Review rate: {len(reviewed) / len(subset):.2%}")
        print(f"Accuracy on automatic decisions: {accuracy:.4f}")
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    print(f"\nDetailed results saved to: {OUTPUT_PATH}")
    print("Reminder: review is not the same as a legitimate classification.")


if __name__ == "__main__":
    main()
