
from pathlib import Path
import html

import joblib
import streamlit as st

# =========================================================
# CONFIGURATION
# =========================================================
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

REVIEW_LOW = 0.40
REVIEW_HIGH = 0.60

st.set_page_config(
    page_title="PhishMan | Security Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# BLACK, RED, GREEN AND YELLOW THEME
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #070707;
        --panel: #101010;
        --border: #292929;
        --red: #ff3030;
        --green: #39e66b;
        --yellow: #ffcc33;
        --text: #f5f5f5;
        --muted: #999999;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7,7,7,0.95);
    }

    [data-testid="stSidebar"] {
        background: #0b0b0b;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: -0.5px;
    }

    .brand {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: var(--red);
    }

    .brand-sub {
        color: var(--muted);
        font-size: 0.76rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 18px 22px;
        background: linear-gradient(120deg, #171010, #0e0e0e 65%);
        border: 1px solid #382020;
        border-radius: 16px;
        margin-bottom: 22px;
    }

    .topbar-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff;
    }

    .topbar-subtitle {
        color: #a4a4a4;
        font-size: 0.9rem;
        margin-top: 5px;
        line-height: 1.5;
    }

    .status-pill {
        display: inline-block;
        padding: 7px 11px;
        border-radius: 20px;
        color: #ff7777;
        background: #291111;
        border: 1px solid #582121;
        font-size: 0.75rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .section-label {
        color: var(--red);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .panel-title {
        color: #ffffff;
        font-size: 1.04rem;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .panel-description {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.6;
    }

    .stat-card {
        background: linear-gradient(145deg, #151515, #0d0d0d);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 17px;
        min-height: 105px;
    }

    .stat-label {
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 12px;
    }

    .stat-value {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .stat-accent {
        color: var(--red);
    }

    .risk-card {
        padding: 23px;
        background: linear-gradient(145deg, #171010, #101010);
        border: 1px solid #432020;
        border-radius: 15px;
        margin: 12px 0 16px 0;
    }

    .risk-number {
        font-size: 2.8rem;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .risk-caption {
        font-size: 0.82rem;
        color: var(--muted);
        margin-top: 5px;
    }

    /* Risk status colors */
    .risk-high {
        color: #ff3030;
        font-size: 1.3rem;
        font-weight: 800;
    }

    .risk-review {
        color: #ffcc33;
        font-size: 1.3rem;
        font-weight: 800;
    }

    .risk-low {
        color: #39e66b;
        font-size: 1.3rem;
        font-weight: 800;
    }

    .indicator {
        background: #151515;
        border: 1px solid #303030;
        border-radius: 9px;
        padding: 12px 14px;
        margin: 8px 0;
        color: #dedede;
        font-size: 0.86rem;
        line-height: 1.55;
    }

    .muted {
        color: var(--muted);
        font-size: 0.85rem;
        line-height: 1.6;
    }

    div.stButton > button {
        background: var(--red);
        color: white;
        border: 1px solid var(--red);
        border-radius: 9px;
        min-height: 45px;
        font-weight: 700;
        transition: all .2s ease;
    }

    div.stButton > button:hover {
        background: #c91c1c;
        color: white;
        border-color: #ff6262;
        box-shadow: 0 0 16px rgba(255,48,48,.15);
    }

    div.stButton > button:focus {
        border-color: #ff6262;
        box-shadow: 0 0 0 2px #542020;
    }

    textarea {
        background: #0d0d0d !important;
        color: #f5f5f5 !important;
        border: 1px solid #383838 !important;
        border-radius: 10px !important;
    }

    textarea:focus {
        border-color: var(--red) !important;
        box-shadow: 0 0 0 1px var(--red) !important;
    }

    div[data-baseweb="select"] > div {
        background: #151515;
        border-color: #383838;
    }

    div[data-testid="stMetric"] {
        background: #111111;
        border: 1px solid var(--border);
        border-radius: 11px;
        padding: 13px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a1a1a1;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }

    /* Progress bar base */
    div[data-testid="stProgress"] > div {
        background: #292929;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Progress bar gets its color from the result category */
    div[data-testid="stProgress"] > div > div {
        background: var(--red);
    }

    .progress-green div[data-testid="stProgress"] > div > div {
        background: #39e66b !important;
    }

    .progress-yellow div[data-testid="stProgress"] > div > div {
        background: #ffcc33 !important;
    }

    .progress-red div[data-testid="stProgress"] > div > div {
        background: #ff3030 !important;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #292929;
    }

    .footer {
        text-align: center;
        color: #747474;
        font-size: 0.78rem;
        padding: 20px 0 8px 0;
        line-height: 1.8;
    }

    @media (max-width: 700px) {
        .topbar-title {
            font-size: 1.25rem;
        }

        .risk-number {
            font-size: 2.1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SAMPLE MESSAGES
# =========================================================
SAMPLES = {
    "Choose a sample...": "",
    "Legitimate — College notice": (
        "The updated class timetable is available on the college portal. "
        "Please check the official student portal when convenient. "
        "Contact your department office if you have questions."
    ),
    "Legitimate — Order update": (
        "Your order has been confirmed and is being prepared for dispatch. "
        "You can check the delivery status in the retailer's official app. "
        "Thank you for shopping with us."
    ),
    "Suspicious — Urgent account warning": (
        "URGENT: Your account will be suspended within 30 minutes. "
        "Verify your account immediately using "
        "hxxps://secure-account-check[.]example/login to avoid losing access."
    ),
    "Suspicious — Prize claim": (
        "Congratulations! You have been selected to receive a special cash "
        "reward. Claim it now at hxxps://claim-reward[.]example before "
        "the offer expires today."
    ),
    "Suspicious — Credential request": (
        "Your account needs verification. Reply with your password and "
        "one-time verification code immediately to prevent account restrictions."
    ),
    "Uncertain — Payment reminder": (
        "Reminder: Your payment is due today. Please review the invoice "
        "using your usual banking or service application. Contact the provider "
        "through its official support channel if you need help."
    ),
}

# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_models():
    model_files = {
        "Logistic Regression": (
            "tfidf_logistic_regression.joblib",
            "tfidf_vectorizer.joblib",
        ),
        "Multinomial Naive Bayes": (
            "naive_bayes_model.joblib",
            "naive_bayes_vectorizer.joblib",
        ),
    }

    loaded = {}

    for name, (model_file, vectorizer_file) in model_files.items():
        model_path = MODELS_DIR / model_file
        vectorizer_path = MODELS_DIR / vectorizer_file

        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError(
                f"Missing model files for {name}. "
                f"Check the models folder: {MODELS_DIR}"
            )

        loaded[name] = {
            "model": joblib.load(model_path),
            "vectorizer": joblib.load(vectorizer_path),
        }

    return loaded


# =========================================================
# CLASSIFICATION AND INDICATORS
# =========================================================
def classify_probability(probability):
    if probability < REVIEW_LOW:
        return "Likely legitimate", "risk-low", "progress-green"

    if probability <= REVIEW_HIGH:
        return "Needs manual review", "risk-review", "progress-yellow"

    return "Potential phishing", "risk-high", "progress-red"


def inspect_message(message):
    text = message.lower()

    indicators = {
        "Urgency or pressure": [
            "urgent",
            "immediately",
            "within 30 minutes",
            "act now",
            "expires today",
            "final warning",
            "suspended",
        ],
        "Credentials or verification": [
            "password",
            "verification code",
            "one-time password",
            "otp",
            "login details",
            "credentials",
        ],
        "Account or payment pressure": [
            "payment failed",
            "account locked",
            "account suspended",
            "unusual activity",
            "payment is due",
        ],
        "Link-related wording": [
            "click here",
            "verify your account",
            "open this link",
            "claim it now",
            "hxxps://",
            "[.]",
        ],
    }

    found = []

    for category, phrases in indicators.items():
        matches = [phrase for phrase in phrases if phrase in text]
        if matches:
            found.append((category, matches))

    return found


def analyze_message(message, model_info):
    vectorizer = model_info["vectorizer"]
    model = model_info["model"]

    features = vectorizer.transform([message])

    if not hasattr(model, "predict_proba"):
        raise ValueError("This model does not support probability estimates.")

    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    if "phishing" not in classes:
        raise ValueError("The model is missing the expected phishing class.")

    probability = float(probabilities[classes.index("phishing")])
    category, css_class, progress_class = classify_probability(probability)

    return probability, category, css_class, progress_class


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">PHISH<span style="color:#ffffff">MAN</span></div>
        <div class="brand-sub">Security intelligence</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Analysis settings")

    model_name = st.selectbox(
        "Detection model",
        ["Logistic Regression", "Multinomial Naive Bayes"],
    )

    st.markdown("### Risk thresholds")

    st.markdown(
        """
        <div class="indicator">
            <span style="color:#39e66b">●</span>
            <b>Below 40%</b><br>
            Likely legitimate
        </div>
        <div class="indicator">
            <span style="color:#ffcc33">●</span>
            <b>40%–60%</b><br>
            Manual review
        </div>
        <div class="indicator">
            <span style="color:#ff3030">●</span>
            <b>Above 60%</b><br>
            Potential phishing
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Privacy first**")
    st.caption(
        "Avoid entering real passwords, OTPs, personal information, "
        "or confidential messages."
    )
    st.caption("Prototype · Synthetic training data")

# =========================================================
# LOAD MODELS
# =========================================================
try:
    MODELS = load_models()
except Exception as exc:
    st.error(f"Unable to load the models: {exc}")
    st.stop()

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="topbar">
        <div>
            <div class="topbar-title">Message Risk Scanner</div>
            <div class="topbar-subtitle">
                Inspect messages, estimate phishing risk, and flag
                uncertain results for review.
            </div>
        </div>
        <div class="status-pill">● LOCAL DEMO</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DASHBOARD STATS
# =========================================================
stat1, stat2, stat3 = st.columns(3)

with stat1:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-label">AVAILABLE MODELS</div>
            <div class="stat-value"><span class="stat-accent">02</span> classifiers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat2:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-label">REVIEW WINDOW</div>
            <div class="stat-value">40% — 60%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with stat3:
    st.markdown(
        """
        <div class="stat-card">
            <div class="stat-label">ANALYSIS MODE</div>
            <div class="stat-value"><span class="stat-accent">Text</span> only</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# =========================================================
# MAIN WORKSPACE
# =========================================================
input_col, result_col = st.columns([1.12, 0.88], gap="large")

with input_col:
    st.markdown(
        '<div class="section-label">01 / Input</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Inspect a message")
    st.markdown(
        '<div class="muted">Paste a message or load a harmless sample '
        'to begin the analysis.</div>',
        unsafe_allow_html=True,
    )

    sample_choice = st.selectbox(
        "Sample message library",
        list(SAMPLES.keys()),
    )

    if "phishman_message" not in st.session_state:
        st.session_state.phishman_message = ""

    if st.button("Load selected sample", use_container_width=True):
        if SAMPLES[sample_choice]:
            st.session_state.phishman_message = SAMPLES[sample_choice]
            st.rerun()
        else:
            st.info("Choose a sample message first.")

    message = st.text_area(
        "Message content",
        key="phishman_message",
        height=245,
        placeholder=(
            "Paste the email, SMS, or message text here...\n\n"
            "Do not include real passwords, OTPs, or personal information."
        ),
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="muted">{len(message):,} characters · '
        f'{len(message.split()):,} words</div>',
        unsafe_allow_html=True,
    )

    analyze_clicked = st.button(
        "Analyze message  →",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        '<div class="muted">Analysis uses the selected trained model. '
        'No suspicious links are opened.</div>',
        unsafe_allow_html=True,
    )

with result_col:
    st.markdown(
        '<div class="section-label">02 / Results</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Risk assessment")

    if analyze_clicked:
        if not message.strip():
            st.warning("Enter a message or load a sample first.")
        else:
            try:
                probability, category, css_class, progress_class = (
                    analyze_message(message, MODELS[model_name])
                )

                st.session_state.phishman_result = {
                    "probability": probability,
                    "category": category,
                    "css_class": css_class,
                    "progress_class": progress_class,
                    "model_name": model_name,
                    "message": message,
                }

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("phishman_result")

    if result:
        probability = result["probability"]
        category = result["category"]
        css_class = result["css_class"]
        progress_class = result["progress_class"]

        st.markdown(
            f"""
            <div class="risk-card">
                <div class="section-label">
                    MODEL-ESTIMATED PHISHING PROBABILITY
                </div>
                <div class="risk-number" style="color:{
                    '#39e66b' if category == 'Likely legitimate'
                    else '#ffcc33' if category == 'Needs manual review'
                    else '#ff3030'
                }">{probability:.1%}</div>
                <div style="margin-top:12px" class="{css_class}">
                    {html.escape(category)}
                </div>
                <div class="risk-caption">
                    Model: {html.escape(result["model_name"])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Color the progress bar based on the risk category.
        st.markdown(
            f'<div class="{progress_class}">',
            unsafe_allow_html=True,
        )
        st.progress(probability, text="Phishing probability")
        st.markdown("</div>", unsafe_allow_html=True)

        if category == "Potential phishing":
            st.error(
                "Exercise caution. Do not open suspicious links or share "
                "credentials. Verify the request independently."
            )
        elif category == "Needs manual review":
            st.warning(
                "The score falls inside the experimental review window. "
                "Check the message using a trusted channel."
            )
        else:
            st.success(
                "The model estimates a lower phishing probability. "
                "This does not guarantee the message is safe."
            )

        metric_a, metric_b = st.columns(2)

        with metric_a:
            st.metric("Estimated phishing", f"{probability:.1%}")

        with metric_b:
            st.metric("Estimated legitimate", f"{1 - probability:.1%}")

    else:
        st.markdown(
            """
            <div class="panel" style="text-align:center;padding:40px 18px">
                <div style="font-size:2.5rem;color:#ff3030">⌕</div>
                <div class="panel-title">Awaiting message</div>
                <div class="panel-description">
                    Your analysis results will appear here after you
                    submit a message.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">What you'll see</div>
                <p class="muted">01 · Model-estimated phishing probability</p>
                <p class="muted">02 · Experimental risk category</p>
                <p class="muted">03 · Text indicators for further inspection</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# INVESTIGATION PANEL
# =========================================================
st.divider()

st.markdown(
    '<div class="section-label">03 / Investigation</div>',
    unsafe_allow_html=True,
)
st.markdown("### Message indicators")

st.markdown(
    '<div class="muted">These are simple phrase matches, not proof of '
    'phishing or explanations of the model’s internal reasoning.</div>',
    unsafe_allow_html=True,
)

latest = st.session_state.get("phishman_result")

if latest:
    indicators = inspect_message(latest["message"])

    if indicators:
        indicator_cols = st.columns(2)

        for index, (category, matches) in enumerate(indicators):
            with indicator_cols[index % 2]:
                safe_category = html.escape(category)
                safe_matches = ", ".join(
                    html.escape(match) for match in matches
                )

                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="panel-title">
                            <span style="color:#ff3030">●</span>
                            {safe_category}
                        </div>
                        <div class="panel-description">
                            Detected phrases: {safe_matches}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">No listed indicators detected</div>
                <div class="panel-description">
                    The phrase checklist found no matches. This does not
                    mean the message is trustworthy.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-description">
                Run an analysis to view the phrase indicators detected
                in your message.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown(
    """
    <div class="footer">
        <b style="color:#ff3030">PHISHMAN</b> · MESSAGE SECURITY INTELLIGENCE<br>
        Educational prototype trained on synthetic data.<br>
        Model probabilities are not calibrated security guarantees.
        Always verify sensitive requests independently.
    </div>
    """,
    unsafe_allow_html=True,
)
