
import os
import re
import json
import html
import sqlite3
import ipaddress
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit

import joblib
import pandas as pd
import streamlit as st

from src.security_analysis import (
    analyze_message_cues,
    analyze_url,
    normalize_defanged_url,
    RISK_PHRASES,
)

# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
DB_PATH = APP_DIR / "phishshift_history.db"

MODELS = {
    "Logistic Regression": {
        "model": MODEL_DIR / "tfidf_logistic_regression.joblib",
        "vectorizer": MODEL_DIR / "tfidf_vectorizer.joblib",
    },
    "Naive Bayes": {
        "model": MODEL_DIR / "naive_bayes_model.joblib",
        "vectorizer": MODEL_DIR / "naive_bayes_vectorizer.joblib",
    },
}

st.set_page_config(
    page_title="PhishShift | Message Risk Scanner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: Inter, sans-serif;
    }

    .stApp {
        background: #080d19;
        color: #edf2ff;
    }

    [data-testid="stSidebar"] {
        background: #0d1424;
        border-right: 1px solid #202c42;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }

    h1, h2, h3, h4 {
        color: #f3f6ff;
    }

    .hero {
        padding: 28px;
        border: 1px solid #263754;
        border-radius: 20px;
        background: linear-gradient(135deg, #111d34, #101729);
        margin-bottom: 22px;
    }

    .hero-label {
        color: #66d9ff;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 12px;
        font-weight: 800;
    }

    .hero h1 {
        font-size: 38px;
        margin: 10px 0;
    }

    .muted {
        color: #9caac3;
    }

    .panel {
        padding: 20px;
        border: 1px solid #263550;
        border-radius: 16px;
        background: #101827;
        margin-bottom: 14px;
    }

    .metric {
        background: #101827;
        border: 1px solid #293750;
        border-radius: 14px;
        padding: 18px;
        min-height: 110px;
    }

    .metric-label {
        color: #9caac3;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #f4f7ff;
        font-size: 29px;
        font-weight: 800;
        margin-top: 8px;
    }

    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 5px 11px;
        font-weight: 700;
        font-size: 12px;
        margin-bottom: 10px;
    }

    .danger {
        color: #ff8b96;
        background: #3c1823;
        border: 1px solid #713044;
    }

    .warning {
        color: #ffd580;
        background: #392b15;
        border: 1px solid #6c5126;
    }

    .success {
        color: #80f0bd;
        background: #12352c;
        border: 1px solid #28634e;
    }

    .highlighted-message {
        padding: 18px;
        border: 1px solid #2b3c58;
        border-radius: 12px;
        background: #0b1220;
        line-height: 2;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }

    .suspicious {
        background: #672b37;
        color: #ffe3e7;
        border-bottom: 2px solid #ff7284;
        border-radius: 4px;
        padding: 2px 3px;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #385c91;
        background: #1d4ed8;
        color: white;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton > button:hover {
        background: #2563eb;
        border-color: #60a5fa;
    }

    [data-testid="stMetric"] {
        background: #101827;
        border: 1px solid #293750;
        padding: 14px;
        border-radius: 12px;
    }

    hr {
        border-color: #25334b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATABASE
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                scan_type TEXT NOT NULL,
                input_text TEXT NOT NULL,
                prediction TEXT,
                confidence REAL,
                risk_score INTEGER,
                risk_level TEXT,
                model_name TEXT,
                findings_json TEXT,
                external_check TEXT
            )
            """
        )
        conn.commit()


def save_scan(
    scan_type,
    input_text,
    prediction,
    confidence,
    risk_score,
    risk_level,
    model_name,
    findings,
    external_check="Not requested",
):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO scan_history (
                timestamp, scan_type, input_text, prediction,
                confidence, risk_score, risk_level, model_name,
                findings_json, external_check
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().astimezone().isoformat(timespec="seconds"),
                scan_type,
                input_text,
                prediction,
                confidence,
                int(risk_score),
                risk_level,
                model_name,
                json.dumps(findings, ensure_ascii=False),
                external_check,
            ),
        )
        conn.commit()


def load_history():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM scan_history ORDER BY id DESC"
        ).fetchall()

    return pd.DataFrame([dict(row) for row in rows])


initialize_database()

# ============================================================
# MODEL LOADING AND PREDICTION
# ============================================================

@st.cache_resource
def load_model_pair(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def is_positive_label(label):
    return str(label).strip().lower() in {
        "spam", "phishing", "phish", "suspicious",
        "malicious", "fraud", "1", "true", "yes",
    }


def predict_message(message, model_name):
    paths = MODELS[model_name]

    if not paths["model"].is_file():
        raise FileNotFoundError(f"Model file not found: {paths['model']}")

    if not paths["vectorizer"].is_file():
        raise FileNotFoundError(
            f"Vectorizer file not found: {paths['vectorizer']}"
        )

    model, vectorizer = load_model_pair(
        str(paths["model"]),
        str(paths["vectorizer"]),
    )

    features = vectorizer.transform([message])
    raw_prediction = model.predict(features)[0]
    classes = list(getattr(model, "classes_", []))

    probabilities = None
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]

    # The prediction must be based on the model's actual predicted label.
    # Probability is used for confidence only after locating the matching class.
    suspicious = is_positive_label(raw_prediction)

    if classes and probabilities is not None:
        matching_indices = [
            i for i, label in enumerate(classes)
            if str(label) == str(raw_prediction)
        ]

        if matching_indices:
            predicted_index = matching_indices[0]
            confidence = float(probabilities[predicted_index]) * 100

    return {
        "prediction": "Suspicious" if suspicious else "Likely Safe",
        "confidence": confidence,
        "raw_prediction": str(raw_prediction),
    }


# ============================================================
# SAFE MESSAGE HIGHLIGHTING
# ============================================================

def render_highlighted_message(message, matches):
    valid_matches = sorted(
        matches,
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"]),
        ),
    )

    selected = []
    last_end = 0

    for item in valid_matches:
        start = max(0, int(item["start"]))
        end = min(len(message), int(item["end"]))

        if start < last_end or end <= start:
            continue

        selected.append({
            **item,
            "start": start,
            "end": end,
        })
        last_end = end

    output = []
    cursor = 0

    for item in selected:
        output.append(html.escape(message[cursor:item["start"]]))
        phrase = html.escape(message[item["start"]:item["end"]])
        explanation = html.escape(item.get("explanation", ""), quote=True)

        output.append(
            f'<span class="suspicious" title="{explanation}">'
            f'{phrase}</span>'
        )
        cursor = item["end"]

    output.append(html.escape(message[cursor:]))
    return "".join(output)


# ============================================================
# URL EXTRACTION
# ============================================================

URL_PATTERN = re.compile(
    r"""(?ix)
    (?:
        \b(?:https?://|hxxps?://)[^\s<>"']+
        |
        \bwww\.[^\s<>"']+
        |
        \b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+
        [a-z]{2,}(?::\d{1,5})?(?:/[^\s<>"']*)?
    )
    """
)


def extract_urls(message, limit=10):
    found = []

    for match in URL_PATTERN.finditer(message):
        candidate = match.group().rstrip(".,;:!?)]}")

        if candidate and candidate not in found:
            found.append(candidate)

        if len(found) >= limit:
            break

    return found


# ============================================================
# OPTIONAL GOOGLE SAFE BROWSING
# ============================================================

def google_safe_browsing_check(url):
    api_key = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")

    if not api_key:
        return {
            "status": "Unavailable",
            "detail": (
                "No API key configured. Set the "
                "GOOGLE_SAFE_BROWSING_API_KEY environment variable."
            ),
        }

    endpoint = (
        "https://safebrowsing.googleapis.com/v4/threatMatches:find"
        f"?key={api_key}"
    )

    payload = {
        "client": {
            "clientId": "phishshift-local-scanner",
            "clientVersion": "1.0",
        },
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            result = json.loads(response.read().decode("utf-8"))

        matches = result.get("matches", [])

        if matches:
            threat_types = sorted({
                item.get("threatType", "UNKNOWN")
                for item in matches
            })

            return {
                "status": "Threat found",
                "detail": "Google reported: " + ", ".join(threat_types),
            }

        return {
            "status": "No known threat found",
            "detail": (
                "No match was returned. This does not guarantee the URL is safe."
            ),
        }

    except urllib.error.HTTPError as exc:
        return {
            "status": "API error",
            "detail": f"Google Safe Browsing returned HTTP {exc.code}.",
        }
    except Exception as exc:
        return {
            "status": "Check failed",
            "detail": f"Reputation check failed ({type(exc).__name__}).",
        }


# ============================================================
# COMBINED RISK SCORING
# ============================================================

def calculate_combined_risk(
    ml_result=None,
    message_analysis=None,
    url_analyses=None,
    reputation_results=None,
):
    """
    Explainable heuristic score from 0 to 100.
    It is not a calibrated probability.
    """
    components = []
    reasons = []

    if ml_result:
        if ml_result["prediction"] == "Suspicious":
            confidence = ml_result.get("confidence")
            contribution = (
                25 if confidence is None
                else round(12 + 13 * confidence / 100)
            )
            components.append(("Machine-learning signal", contribution))
            reasons.append(
                f"ML model classified the message as suspicious (+{contribution})."
            )

    if message_analysis:
        matches = message_analysis.get("matches", [])
        unique = {}

        for match in matches:
            phrase = match.get("normalized_phrase", match.get("phrase", ""))
            unique[phrase.lower()] = match

        severity_points = {"high": 12, "medium": 8, "low": 4}
        cue_score = min(
            35,
            sum(
                severity_points.get(
                    str(item.get("severity", "low")).lower(), 4
                )
                for item in unique.values()
            ),
        )

        if cue_score:
            components.append(("Message phrase cues", cue_score))
            reasons.append(
                f"Suspicious message phrases contributed +{cue_score}."
            )

        urgency_count = len(message_analysis.get("urgency_matches", []))
        if urgency_count:
            urgency_score = min(10, urgency_count * 3)
            components.append(("Urgency cues", urgency_score))
            reasons.append(
                f"Urgency cues contributed +{urgency_score}."
            )

    if url_analyses:
        valid_scores = [
            int(item.get("score", 0))
            for item in url_analyses
        ]

        if valid_scores:
            url_score = round(max(valid_scores) * 0.4)
            if url_score:
                components.append(("URL structure", url_score))
                reasons.append(
                    f"Highest URL structure score contributed +{url_score}."
                )

    if reputation_results:
        if any(
            item.get("status") == "Threat found"
            for item in reputation_results
        ):
            components.append(("External threat match", 40))
            reasons.append(
                "Google Safe Browsing reported a threat (+40)."
            )

    # Use a weighted sum capped at 100. This is a heuristic, not probability.
    score = min(100, sum(points for _, points in components))

    if score >= 65:
        level = "High"
    elif score >= 30:
        level = "Medium"
    else:
        level = "Low"

    return {
        "score": score,
        "level": level,
        "components": components,
        "reasons": reasons,
    }


def risk_class(level):
    return {
        "High": "danger",
        "Medium": "warning",
        "Low": "success",
    }.get(level, "warning")


def show_risk(score, level, label="Combined risk score"):
    st.markdown(
        f"""
        <div class="panel">
            <div class="pill {risk_class(level)}">
                {html.escape(level)} Risk
            </div>
            <div class="metric-label">{html.escape(label)}</div>
            <div style="font-size:42px;font-weight:800;margin:8px 0">
                {int(score)}<span style="font-size:18px;color:#9caac3"> / 100</span>
            </div>
            <div class="muted">
                Heuristic indicator, not a probability or a guarantee.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_url_findings(analysis):
    for finding in analysis.get("findings", []):
        level = str(finding.get("level", "info")).lower()
        icon = "🔴" if level == "high" else (
            "🟠" if level == "medium" else "🔵"
        )

        title = html.escape(str(finding.get("title", "Finding")))
        explanation = html.escape(
            str(finding.get("explanation", ""))
        )

        st.markdown(
            f"""
            <div class="panel">
                <b>{icon} {title}</b>
                <div class="muted" style="margin-top:8px">
                    {explanation}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_header():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">Security Intelligence Platform</div>
            <h1>PhishShift</h1>
            <div class="muted">
                Detect suspicious messages, inspect risky URLs, and understand
                why a scan raises concern.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

if "navigation_target" not in st.session_state:
    st.session_state["navigation_target"] = "Overview"

pages = [
    "Overview",
    "Message Scanner",
    "URL Scanner",
    "Scan History",
]

if st.session_state["navigation_target"] not in pages:
    st.session_state["navigation_target"] = "Overview"

st.sidebar.markdown(
    """
    <div style="padding:10px 0 20px">
        <div style="font-size:30px">🛡️ <b>PhishShift</b></div>
        <div style="color:#91a2bf;font-size:12px;margin-top:5px">
            MESSAGE RISK INTELLIGENCE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "NAVIGATION",
    pages,
    key="navigation_target",
)

st.sidebar.markdown("---")
st.sidebar.caption("Local analysis · SQLite history")
st.sidebar.caption("Offline URL checks never visit the submitted destination.")
render_header()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    history = load_history()

    total_scans = len(history)
    high_risk_scans = (
        int((history["risk_level"] == "High").sum())
        if not history.empty else 0
    )
    average_risk = (
        round(float(history["risk_score"].mean()), 1)
        if not history.empty else 0
    )
    url_scans = (
        int((history["scan_type"] == "URL").sum())
        if not history.empty else 0
    )

    columns = st.columns(4)
    metrics = [
        ("Total Scans", total_scans),
        ("High-Risk Scans", high_risk_scans),
        ("Average Risk", average_risk),
        ("URL Scans", url_scans),
    ]

    for col, (label, value) in zip(columns, metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns(2)

    with left:
        st.markdown("### Message analysis")
        st.markdown(
            '<div class="panel">Run your trained models and inspect '
            'highlighted suspicious phrases.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Open Message Scanner", use_container_width=True):
            st.session_state["navigation_target"] = "Message Scanner"
            st.rerun()

    with right:
        st.markdown("### URL analysis")
        st.markdown(
            '<div class="panel">Inspect URL structure without opening '
            'the destination.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Open URL Scanner", use_container_width=True):
            st.session_state["navigation_target"] = "URL Scanner"
            st.rerun()

    st.markdown("### Recent scans")

    if history.empty:
        st.info("No scans yet. Run a scan to start building your history.")
    else:
        columns_to_show = [
            "timestamp", "scan_type", "prediction",
            "risk_score", "risk_level",
        ]
        st.dataframe(
            history[columns_to_show].head(10),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MESSAGE SCANNER
# ============================================================

elif page == "Message Scanner":
    st.markdown("## Message Risk Scanner")
    st.caption(
        "Analyze text using your trained model and explainable rule-based cues."
    )

    model_name = st.selectbox(
        "Machine-learning model",
        list(MODELS.keys()),
    )

    message = st.text_area(
        "Paste a suspicious message",
        height=220,
        placeholder=(
            "Example: Your account will be suspended. "
            "Verify your password immediately..."
        ),
    )

    include_url_scan = st.checkbox(
        "Analyze URLs found inside the message",
        value=True,
    )

    use_reputation = st.checkbox(
        "Check extracted URLs with Google Safe Browsing",
        value=False,
        help=(
            "This sends each extracted URL to Google. "
            "Do not enable this for confidential or internal URLs."
        ),
    )

    if use_reputation and not os.getenv("GOOGLE_SAFE_BROWSING_API_KEY"):
        st.warning(
            "No Google Safe Browsing API key is configured. "
            "The external check will be unavailable."
        )

    if st.button(
        "Analyze Message",
        type="primary",
        use_container_width=True,
    ):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            try:
                with st.spinner("Analyzing message..."):
                    ml_result = predict_message(message, model_name)
                    cue_analysis = analyze_message_cues(message)

                    url_results = []
                    reputation_results = []

                    if include_url_scan:
                        for candidate in extract_urls(message):
                            normalized = normalize_defanged_url(candidate)
                            url_analysis = analyze_url(normalized)

                            url_results.append({
                                "url": candidate,
                                "analysis": url_analysis,
                            })

                            if use_reputation:
                                reputation_results.append({
                                    "url": candidate,
                                    "result": google_safe_browsing_check(
                                        url_analysis.get(
                                            "normalized_url", normalized
                                        )
                                    ),
                                })

                    combined = calculate_combined_risk(
                        ml_result=ml_result,
                        message_analysis=cue_analysis,
                        url_analyses=[
                            item["analysis"] for item in url_results
                        ],
                        reputation_results=[
                            item["result"] for item in reputation_results
                        ],
                    )

                    findings_for_db = {
                        "message_cues": cue_analysis.get("matches", []),
                        "urgency_matches": cue_analysis.get(
                            "urgency_matches", []
                        ),
                        "reassuring_matches": cue_analysis.get(
                            "reassuring_matches", []
                        ),
                        "url_results": url_results,
                        "risk_reasons": combined["reasons"],
                        "reputation_results": reputation_results,
                    }

                    save_scan(
                        scan_type="Message",
                        input_text=message,
                        prediction=ml_result["prediction"],
                        confidence=ml_result["confidence"],
                        risk_score=combined["score"],
                        risk_level=combined["level"],
                        model_name=model_name,
                        findings=findings_for_db,
                        external_check=(
                            "Requested" if use_reputation else "Not requested"
                        ),
                    )

                    st.session_state["last_message_result"] = {
                        "message": message,
                        "model_name": model_name,
                        "ml": ml_result,
                        "cues": cue_analysis,
                        "urls": url_results,
                        "combined": combined,
                        "reputation": reputation_results,
                    }

            except Exception as exc:
                st.error(
                    f"Analysis failed ({type(exc).__name__}): {exc}"
                )

    result = st.session_state.get("last_message_result")

    if result:
        st.markdown("---")
        st.markdown("### Analysis results")

        ml_result = result["ml"]
        combined = result["combined"]
        cue_analysis = result["cues"]

        left, right = st.columns(2)

        with left:
            st.markdown("#### Model prediction")
            css_class = (
                "danger"
                if ml_result["prediction"] == "Suspicious"
                else "success"
            )

            st.markdown(
                f'<div class="pill {css_class}">'
                f'{html.escape(ml_result["prediction"])}</div>',
                unsafe_allow_html=True,
            )

            if ml_result["confidence"] is not None:
                st.metric(
                    "Confidence in predicted class",
                    f'{ml_result["confidence"]:.2f}%',
                )
            else:
                st.info("This model does not provide probability estimates.")

            st.caption(f'Model: {result["model_name"]}')
            st.caption(
                f'Raw predicted label: {ml_result["raw_prediction"]}'
            )

        with right:
            show_risk(combined["score"], combined["level"])

        st.markdown("#### Message with highlighted phrases")

        if cue_analysis.get("matches"):
            highlighted = render_highlighted_message(
                result["message"],
                cue_analysis["matches"],
            )
            st.markdown(
                f'<div class="highlighted-message">{highlighted}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "No configured suspicious phrases were matched. "
                "That does not establish that the message is safe."
            )
            st.markdown(
                '<div class="highlighted-message">'
                f'{html.escape(result["message"])}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Why was it flagged?")
        st.write(cue_analysis.get("summary", ""))

        for finding in cue_analysis.get("matches", []):
            st.markdown(
                f"""
                <div class="panel">
                    <b>{html.escape(finding.get("phrase", ""))}</b>
                    <div class="muted" style="margin-top:7px">
                        {html.escape(finding.get("explanation", ""))}
                    </div>
                    <div style="color:#9caac3;margin-top:6px">
                        Severity: {html.escape(finding.get("severity", "unknown"))}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for finding in cue_analysis.get("urgency_matches", []):
            st.caption(
                f'Urgency cue: {finding.get("phrase", "")} — '
                f'{finding.get("explanation", "")}'
            )

        if cue_analysis.get("reassuring_matches"):
            st.markdown("#### Reassuring wording detected")
            st.caption(
                "Reassuring wording does not establish that a message is safe."
            )
            for finding in cue_analysis["reassuring_matches"]:
                st.write(
                    f'- "{finding["phrase"]}": {finding["explanation"]}'
                )

        if combined["components"]:
            st.markdown("#### Combined risk breakdown")
            for label, points in combined["components"]:
                st.write(f"- **{label}:** +{points}")

        for reason in combined["reasons"]:
            st.caption(reason)

        if result["urls"]:
            st.markdown("#### URLs found in the message")

            for index, item in enumerate(result["urls"], start=1):
                analysis = item["analysis"]

                with st.expander(
                    f'URL {index}: {item["url"]} — '
                    f'{analysis.get("risk_level", "Unknown")} risk'
                ):
                    st.write(
                        f'**Hostname:** {analysis.get("hostname", "Unknown")}'
                    )
                    st.write(
                        f'**URL structure score:** '
                        f'{analysis.get("score", 0)}/100'
                    )
                    display_url_findings(analysis)

        if result["reputation"]:
            st.markdown("#### Google Safe Browsing results")
            st.warning(
                "The URLs listed below were sent to Google Safe Browsing."
            )

            for item in result["reputation"]:
                st.write(f'**URL:** `{item["url"]}`')
                st.write(f'**Status:** {item["result"]["status"]}')
                st.write(item["result"]["detail"])


# ============================================================
# URL SCANNER
# ============================================================

elif page == "URL Scanner":
    st.markdown("## URL Risk Scanner")
    st.caption(
        "Offline URL inspection. The submitted destination is never opened."
    )

    url_input = st.text_input(
        "Enter a URL",
        placeholder="https://example.com/login",
    )

    use_reputation = st.checkbox(
        "Also check Google Safe Browsing",
        value=False,
        help=(
            "This sends the URL to Google. Avoid using it for confidential "
            "or internal URLs."
        ),
    )

    if use_reputation and not os.getenv("GOOGLE_SAFE_BROWSING_API_KEY"):
        st.warning(
            "Configure GOOGLE_SAFE_BROWSING_API_KEY to enable this check."
        )

    if st.button(
        "Scan URL",
        type="primary",
        use_container_width=True,
    ):
        if not url_input.strip():
            st.warning("Enter a URL first.")
        else:
            with st.spinner("Inspecting URL structure..."):
                try:
                    analysis = analyze_url(url_input)
                    reputation = None

                    if (
                        use_reputation
                        and analysis.get("normalized_url")
                    ):
                        reputation = google_safe_browsing_check(
                            analysis["normalized_url"]
                        )

                    reputation_list = [reputation] if reputation else []

                    combined = calculate_combined_risk(
                        url_analyses=[analysis],
                        reputation_results=reputation_list,
                    )

                    findings_for_db = {
                        "url_analysis": analysis,
                        "risk_reasons": combined["reasons"],
                        "reputation": reputation,
                    }

                    prediction = (
                        "Threat found"
                        if reputation
                        and reputation["status"] == "Threat found"
                        else "URL inspected"
                    )

                    save_scan(
                        scan_type="URL",
                        input_text=url_input,
                        prediction=prediction,
                        confidence=None,
                        risk_score=combined["score"],
                        risk_level=combined["level"],
                        model_name="Offline URL analysis",
                        findings=findings_for_db,
                        external_check=(
                            reputation["status"]
                            if reputation else "Not requested"
                        ),
                    )

                    st.session_state["last_url_result"] = {
                        "analysis": analysis,
                        "combined": combined,
                        "reputation": reputation,
                    }

                except Exception as exc:
                    st.error(
                        f"URL analysis failed ({type(exc).__name__}): {exc}"
                    )

    result = st.session_state.get("last_url_result")

    if result:
        analysis = result["analysis"]
        combined = result["combined"]

        st.markdown("---")
        left, right = st.columns(2)

        with left:
            st.markdown("### URL details")
            st.write(
                f'**Normalized URL:** `{analysis.get("normalized_url", "")}`'
            )
            st.write(
                f'**Hostname:** `{analysis.get("hostname", "Unknown")}`'
            )
            st.write(f'**Scheme:** `{analysis.get("scheme", "Unknown")}`')

        with right:
            show_risk(combined["score"], combined["level"])

        st.markdown("### Structural findings")
        display_url_findings(analysis)

        st.markdown("### Risk score explanation")
        if combined["reasons"]:
            for reason in combined["reasons"]:
                st.write(f"- {reason}")
        else:
            st.write(
                "No score-raising indicator was identified by this heuristic."
            )

        if result["reputation"]:
            st.markdown("### Google Safe Browsing")
            st.write(f'**{result["reputation"]["status"]}**')
            st.write(result["reputation"]["detail"])

        st.caption(analysis.get("disclaimer", "Offline structural analysis."))
        st.caption(
            "A low score does not mean a URL is safe. Structural checks can "
            "miss new or carefully disguised attacks."
        )


# ============================================================
# SCAN HISTORY
# ============================================================

elif page == "Scan History":
    st.markdown("## Scan History")
    st.caption(
        "Scan records are saved locally in SQLite and persist across restarts."
    )

    history = load_history()

    if history.empty:
        st.info("No saved scans yet.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            types = ["All"] + sorted(
                history["scan_type"].dropna().unique().tolist()
            )
            selected_type = st.selectbox("Scan type", types)

        with col2:
            selected_level = st.selectbox(
                "Risk level",
                ["All", "High", "Medium", "Low", "Unknown"],
            )

        with col3:
            search_text = st.text_input("Search scanned text")

        filtered = history.copy()

        if selected_type != "All":
            filtered = filtered[
                filtered["scan_type"] == selected_type
            ]

        if selected_level != "All":
            filtered = filtered[
                filtered["risk_level"] == selected_level
            ]

        if search_text.strip():
            filtered = filtered[
                filtered["input_text"].str.contains(
                    search_text,
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

        columns_to_show = [
            "id", "timestamp", "scan_type", "input_text",
            "prediction", "confidence", "risk_score", "risk_level",
            "model_name", "external_check",
        ]

        st.dataframe(
            filtered[columns_to_show],
            use_container_width=True,
            hide_index=True,
        )

        csv_data = filtered.to_csv(index=False).encode("utf-8-sig")
        json_data = filtered.to_json(
            orient="records",
            force_ascii=False,
            indent=2,
        ).encode("utf-8")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "Export filtered history as CSV",
                data=csv_data,
                file_name="phishshift_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                "Export filtered history as JSON",
                data=json_data,
                file_name="phishshift_history.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")

        with st.expander("Delete saved scan history"):
            st.warning(
                "This permanently deletes all records in the local database."
            )

            confirm_delete = st.checkbox(
                "I understand that all saved scans will be deleted.",
                key="confirm_delete_history",
            )

            if st.button("Delete all history"):
                if confirm_delete:
                    with get_connection() as conn:
                        conn.execute("DELETE FROM scan_history")
                        conn.commit()

                    st.success("Saved scan history deleted.")
                    st.rerun()
                else:
                    st.error("Confirm the deletion before proceeding.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;color:#71819d;font-size:12px;padding:8px">
        PhishShift · Explainable message and URL risk analysis<br>
        For security awareness and triage; not a guarantee of safety.
    </div>
    """,
    unsafe_allow_html=True,
)
