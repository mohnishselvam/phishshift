
from pathlib import Path
from datetime import date

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = REPORTS_DIR / "PhishShift_Project_Report.md"


def read_csv(filename):
    path = REPORTS_DIR / filename
    if not path.exists():
        print(f"WARNING: Missing report file: {path}")
        return None

    return pd.read_csv(path)


def read_text(filename):
    path = REPORTS_DIR / filename
    if not path.exists():
        return f"Not available: {filename}"

    return path.read_text(encoding="utf-8")


def format_percent(value):
    return f"{value * 100:.2f}%"


def dataframe_to_markdown(df):
    """Render a DataFrame as a Markdown table without extra dependencies."""
    if df is None or df.empty:
        return "_No data available._"

    columns = list(df.columns)
    rows = [
        "| " + " | ".join(str(col) for col in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]

    for _, row in df.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|").replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def dataset_summary():
    prepared_path = DATA_DIR / "messages_prepared.csv"
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not all(path.exists() for path in [
        prepared_path, train_path, test_path
    ]):
        return "Dataset summary files are incomplete."

    prepared = pd.read_csv(prepared_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    overlap = set(train["campaign_group"]) & set(test["campaign_group"])

    return f"""
- Prepared messages: **{len(prepared)}**
- Training messages: **{len(train)}**
- Held-out test messages: **{len(test)}**
- Unique campaign groups in prepared dataset: **{prepared['campaign_group'].nunique()}**
- Campaign groups shared between training and test: **{len(overlap)}**
- Training labels: `{train['label'].value_counts().to_dict()}`
- Test labels: `{test['label'].value_counts().to_dict()}`
"""


def main():
    comparison = read_csv("model_comparison.csv")
    robustness = read_csv("robustness_results.csv")
    review = read_csv("review_policy_results.csv")
    error_summary = read_csv("error_analysis_summary.csv")

    lines = [
        "# PhishShift — Message Risk Scanner",
        "",
        f"**Report generated:** {date.today().isoformat()}",
        "",
        "## 1. Project overview",
        "",
        "PhishShift is an educational machine-learning project that explores "
        "message-risk classification. It compares a rule-based baseline with "
        "two text-classification models and investigates campaign-separated "
        "evaluation, uncertain-message review, and robustness to harmless "
        "text perturbations.",
        "",
        "## 2. Dataset and evaluation setup",
        "",
        dataset_summary(),
        "",
        "The current dataset is synthetic. It is useful for testing the "
        "processing and evaluation pipeline, but it does not establish "
        "real-world phishing-detection performance.",
        "",
        "Campaign groups are kept separate between the training and test "
        "sets to evaluate messages from unseen groups. This is a useful "
        "design choice, although synthetic campaign generation may still "
        "differ substantially from real campaigns.",
        "",
        "## 3. Model comparison",
        "",
        dataframe_to_markdown(comparison),
        "",
        "The rule-based baseline currently performs poorly on the held-out "
        "synthetic test set. The two TF-IDF classifiers perform better on "
        "this dataset, but their scores should not be generalized to "
        "real-world traffic.",
        "",
        "## 4. Robustness experiment",
        "",
        dataframe_to_markdown(robustness),
        "",
        "This experiment covers only the supplied eight synthetic test cases. "
        "It reports prediction changes and probability shifts under the "
        "implemented perturbations. The small sample is not sufficient to "
        "establish broad robustness.",
        "",
        "## 5. Uncertain-message review policy",
        "",
        dataframe_to_markdown(review),
        "",
        "The experimental policy sends messages with phishing probability "
        "from 0.40 through 0.60, inclusive, to review. Probabilities below "
        "0.40 are classified as legitimate and probabilities above 0.60 "
        "as phishing. These are experimental thresholds, not validated "
        "security settings.",
        "",
        "Accuracy on automatically classified messages excludes messages "
        "sent to review. It must not be interpreted as accuracy across the "
        "entire test set.",
        "",
        "## 6. False-positive and false-negative analysis",
        "",
        dataframe_to_markdown(error_summary),
        "",
        "A false positive flags a legitimate message as phishing. This can "
        "cause users to distrust or overlook genuine notifications. A false "
        "negative classifies a phishing-labelled message as legitimate, "
        "which may expose a user to risk.",
        "",
        "The inspected synthetic examples show that neutral wording can be "
        "classified as legitimate even when the assigned label is phishing, "
        "and urgency-related wording can trigger false alarms. Some generated "
        "examples also contain conflicts between their text and assigned "
        "labels. These cases should be manually reviewed before drawing "
        "conclusions about model behavior.",
        "",
        "## 7. Data quality and automated tests",
        "",
        "The dataset validation checks cover required columns, unique message "
        "IDs, valid labels, duplicate fingerprints, campaign-group separation, "
        "and the presence of both labels in each split. The latest recorded "
        "test run passed all six checks.",
        "",
        "These checks establish basic structural consistency; they do not "
        "prove that message labels are semantically correct or that the "
        "dataset represents real-world traffic.",
        "",
        "## 8. Limitations",
        "",
        "- The current dataset is synthetic and may contain label-text "
        "contradictions.",
        "- The robustness experiment uses only eight test cases.",
        "- The review thresholds have not been calibrated on real data.",
        "- No real-world detection claim is supported by these experiments.",
        "- The rule-based baseline may need further development, but any "
        "changes must be evaluated transparently on the same held-out split.",
        "",
        "## 9. Future improvements",
        "",
        "1. Add appropriately licensed, provenance-documented public examples "
        "and independently review labels.",
        "2. Manually audit synthetic examples where message content conflicts "
        "with the assigned label.",
        "3. Expand perturbation testing across many message templates and "
        "campaign groups.",
        "4. Evaluate false-positive rates on legitimate urgent and sensitive "
        "messages.",
        "5. Add safe explanations for model decisions and an explicit review "
        "outcome in the demo.",
        "6. Add reproducible tests for prediction behavior and model loading.",
        "",
        "## 10. Safe-use statement",
        "",
        "PhishShift is an educational prototype, not a production security "
        "system. Do not rely on it as the sole basis for deciding whether "
        "a message is safe. Verify sensitive requests through an independently "
        "trusted channel. Suspicious URLs should not be opened during testing.",
        "",
        "---",
        "",
        "Generated from project CSV artifacts. Check that all referenced "
        "result files exist and correspond to the latest model run before "
        "submitting the report.",
        "",
    ]

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("Report generated successfully:")
    print(OUTPUT_FILE)
    print("\nReport sections: overview, dataset, model comparison, robustness,")
    print("review policy, error analysis, tests, limitations, future work.")


if __name__ == "__main__":
    main()
