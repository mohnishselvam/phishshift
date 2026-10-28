
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "robustness_test_cases.csv"
REPORT_PATH = ROOT / "reports" / "robustness_results.csv"

MODELS = {
    "Logistic Regression": (
        ROOT / "models" / "tfidf_logistic_regression.joblib",
        ROOT / "models" / "tfidf_vectorizer.joblib",
    ),
    "Multinomial Naive Bayes": (
        ROOT / "models" / "naive_bayes_model.joblib",
        ROOT / "models" / "naive_bayes_vectorizer.joblib",
    ),
}


def predict_risk(model, vectorizer, text):
    features = vectorizer.transform([text])
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    if "phishing" not in classes:
        raise ValueError(
            f"Expected a 'phishing' class, found: {classes}"
        )

    phishing_probability = float(
        probabilities[classes.index("phishing")]
    )
    predicted_label = str(model.predict(features)[0])

    return predicted_label, phishing_probability


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing test cases: {DATA_PATH}")

    cases = pd.read_csv(DATA_PATH)

    required_columns = {
        "case_id",
        "test_type",
        "original_text",
        "perturbed_text",
        "expected_note",
    }

    missing = required_columns - set(cases.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    results = []

    for model_name, (model_path, vectorizer_path) in MODELS.items():
        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError(
                f"Missing saved model or vectorizer for {model_name}"
            )

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        for _, case in cases.iterrows():
            original_label, original_risk = predict_risk(
                model, vectorizer, str(case["original_text"])
            )
            perturbed_label, perturbed_risk = predict_risk(
                model, vectorizer, str(case["perturbed_text"])
            )

            results.append({
                "model": model_name,
                "case_id": case["case_id"],
                "test_type": case["test_type"],
                "original_prediction": original_label,
                "original_phishing_probability": round(original_risk, 6),
                "perturbed_prediction": perturbed_label,
                "perturbed_phishing_probability": round(
                    perturbed_risk, 6
                ),
                "absolute_probability_change": round(
                    abs(perturbed_risk - original_risk), 6
                ),
                "expected_note": case["expected_note"],
            })

    report = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print("\nPHISHSHIFT ROBUSTNESS TEST")
    print("=" * 80)

    for model_name in MODELS:
        subset = report[report["model"] == model_name]
        changed = (
            subset["original_prediction"]
            != subset["perturbed_prediction"]
        ).sum()
        mean_change = subset["absolute_probability_change"].mean()
        largest = subset["absolute_probability_change"].max()

        print(f"\nModel: {model_name}")
        print(f"Cases tested: {len(subset)}")
        print(f"Prediction changes: {changed}")
        print(f"Mean absolute probability change: {mean_change:.4f}")
        print(f"Largest absolute probability change: {largest:.4f}")

    print(f"\nDetailed report saved to: {REPORT_PATH}")
    print(
        "\nNote: These eight synthetic cases are diagnostic examples, "
        "not a statistically reliable robustness benchmark."
    )


if __name__ == "__main__":
    main()
