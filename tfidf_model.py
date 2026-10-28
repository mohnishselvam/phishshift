
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

MODEL_FILE = MODELS_DIR / "tfidf_logistic_regression.joblib"
VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.joblib"
RESULTS_FILE = REPORTS_DIR / "tfidf_results.csv"
METRICS_FILE = REPORTS_DIR / "tfidf_metrics.txt"

REVIEW_LOW = 0.40
REVIEW_HIGH = 0.60


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_dataset.py and "
            "split_dataset.py first."
        )

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    required = {
        "message_id",
        "subject_or_title",
        "body_text",
        "label",
        "campaign_group",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")

    if df.empty:
        raise ValueError(f"{path.name} contains no records.")

    if not df["label"].isin(["phishing", "legitimate"]).all():
        raise ValueError(f"Unexpected labels found in {path.name}.")

    if df["message_id"].duplicated().any():
        raise ValueError(f"Duplicate message IDs in {path.name}.")

    return df


def apply_review_policy(probability):
    if probability < REVIEW_LOW:
        return "legitimate"

    if probability <= REVIEW_HIGH:
        return "review"

    return "phishing"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(TRAIN_FILE)
    test_df = load_dataset(TEST_FILE)

    overlap = set(train_df["campaign_group"]) & set(test_df["campaign_group"])
    if overlap:
        raise ValueError(f"Campaign leakage detected: {sorted(overlap)}")

    message_overlap = set(train_df["message_id"]) & set(test_df["message_id"])
    if message_overlap:
        raise ValueError(f"Message ID leakage detected: {sorted(message_overlap)}")

    if train_df["label"].nunique() < 2:
        raise ValueError("Training data must contain both labels.")

    if test_df["label"].nunique() < 2:
        raise ValueError("Test data must contain both labels.")

    # Fit text features on training data only.
    train_text = (
        train_df["subject_or_title"] + " " + train_df["body_text"]
    )
    test_text = (
        test_df["subject_or_title"] + " " + test_df["body_text"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, train_df["label"])

    probabilities = model.predict_proba(X_test)
    phishing_index = list(model.classes_).index("phishing")
    phishing_probabilities = probabilities[:, phishing_index]

    results = test_df.copy()
    results["phishing_probability"] = phishing_probabilities.round(4)
    results["predicted_label"] = [
        apply_review_policy(float(probability))
        for probability in phishing_probabilities
    ]

    results["explanation"] = results.apply(
        lambda row: (
            "Potential phishing: the model assigned a probability above "
            "the review threshold. Verify through an independently trusted "
            "channel; do not use links in the message."
            if row["predicted_label"] == "phishing"
            else
            "Uncertain result: manual review is recommended. The probability "
            "is not a guarantee that the message is safe or malicious."
            if row["predicted_label"] == "review"
            else
            "Likely legitimate according to this model, but safety is not "
            "guaranteed. Use an independently trusted channel for sensitive "
            "account actions."
        ),
        axis=1,
    )

    # Evaluate automatic decisions separately from review cases.
    automatic = results[results["predicted_label"] != "review"].copy()
    review_count = int((results["predicted_label"] == "review").sum())

    if not automatic.empty:
        y_true = automatic["label"]
        y_pred = automatic["predicted_label"]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, pos_label="phishing", zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, pos_label="phishing", zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, pos_label="phishing", zero_division=0
        )
        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=["legitimate", "phishing"],
        )
        report = classification_report(
            y_true,
            y_pred,
            labels=["legitimate", "phishing"],
            zero_division=0,
        )
    else:
        accuracy = precision = recall = f1 = 0.0
        matrix = [[0, 0], [0, 0]]
        report = "No automatic predictions; automatic metrics unavailable."

    metrics = {
        "train_records": len(train_df),
        "test_records": len(test_df),
        "train_campaigns": train_df["campaign_group"].nunique(),
        "test_campaigns": test_df["campaign_group"].nunique(),
        "campaign_overlap": len(overlap),
        "message_id_overlap": len(message_overlap),
        "review_low_threshold": REVIEW_LOW,
        "review_high_threshold": REVIEW_HIGH,
        "automatic_predictions": len(automatic),
        "review_cases": review_count,
        "coverage": len(automatic) / len(results),
        "review_rate": review_count / len(results),
        "accuracy_on_automatic_predictions": accuracy,
        "precision_phishing": precision,
        "recall_phishing": recall,
        "f1_phishing": f1,
        "false_positives": int(matrix[0][1]),
        "false_negatives": int(matrix[1][0]),
        "true_negatives": int(matrix[0][0]),
        "true_positives": int(matrix[1][1]),
    }

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    results.to_csv(RESULTS_FILE, index=False)

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        file.write("PHISHMAN TF-IDF + LOGISTIC REGRESSION\n\n")

        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")

        file.write("\nConfusion matrix among automatic predictions\n")
        file.write("Rows=true, columns=predicted; order: legitimate, phishing\n")
        file.write(str(matrix))
        file.write("\n\nClassification report\n")
        file.write(report)

    print("\nPHISHMAN TF-IDF + LOGISTIC REGRESSION")
    print("=" * 55)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nFirst 20 test predictions:")
    print(
        results[
            ["message_id", "label", "predicted_label", "phishing_probability"]
        ].head(20).to_string(index=False)
    )

    print(f"\nSaved model: {MODEL_FILE}")
    print(f"Saved vectorizer: {VECTORIZER_FILE}")
    print(f"Saved results: {RESULTS_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")

    print(
        "\nImportant: this is a synthetic-data experiment. "
        "Probabilities are model scores, not verified real-world risks."
    )


if __name__ == "__main__":
    main()
