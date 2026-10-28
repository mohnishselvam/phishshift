import ipaddress
import re
from urllib.parse import urlsplit, unquote


# ============================================================
# PHISHSHIFT SECURITY ANALYSIS
# Offline suspicious-phrase detection and URL risk assessment.
# No submitted URL is opened, fetched, or visited.
# ============================================================

RISK_PHRASES = [
    {
        "phrase": "verify your password",
        "severity": "high",
        "explanation": "Requests password verification. Verify independently before responding.",
    },
    {
        "phrase": "enter your password",
        "severity": "high",
        "explanation": "Requests a password. Never enter credentials through an unexpected message link.",
    },
    {
        "phrase": "share your otp",
        "severity": "high",
        "explanation": "Requests an OTP. Never share an OTP with another person.",
    },
    {
        "phrase": "send your otp",
        "severity": "high",
        "explanation": "Requests an OTP that could authorize an account action.",
    },
    {
        "phrase": "confirm your bank details",
        "severity": "high",
        "explanation": "Requests sensitive banking information.",
    },
    {
        "phrase": "account will be suspended",
        "severity": "medium",
        "explanation": "Uses a threat of suspension to pressure the recipient.",
    },
    {
        "phrase": "account has been suspended",
        "severity": "medium",
        "explanation": "Claims an account is suspended. Check through the official service.",
    },
    {
        "phrase": "immediate action required",
        "severity": "medium",
        "explanation": "Uses urgency to pressure the recipient into acting quickly.",
    },
    {
        "phrase": "act immediately",
        "severity": "medium",
        "explanation": "Pressures the recipient to act without checking the request.",
    },
    {
        "phrase": "claim your prize",
        "severity": "medium",
        "explanation": "Unexpected prize claims can be used to lure people into scams.",
    },
    {
        "phrase": "you have won",
        "severity": "medium",
        "explanation": "Unexpected winnings claims deserve independent verification.",
    },
    {
        "phrase": "click here to verify",
        "severity": "high",
        "explanation": "Combines a link instruction with an account-verification request.",
    },
    {
        "phrase": "confirm your identity",
        "severity": "medium",
        "explanation": "Requests identity verification. Check through an official service.",
    },
    {
        "phrase": "payment failed",
        "severity": "low",
        "explanation": "A payment issue may be genuine or fraudulent. Check the official application.",
    },
    {
        "phrase": "unusual login activity",
        "severity": "low",
        "explanation": "Security alerts can be legitimate or fraudulent. Verify independently.",
    },
    {
        "phrase": "gift card",
        "severity": "medium",
        "explanation": "Unexpected gift-card payment requests can be associated with scams.",
    },
    {
        "phrase": "keep this confidential",
        "severity": "medium",
        "explanation": "Secrecy pressure can be manipulative, especially alongside payment requests.",
    },
    {
        "phrase": "disable your security",
        "severity": "high",
        "explanation": "Encourages disabling a security protection.",
    },
]

URGENCY_PHRASES = [
    "urgent",
    "urgently",
    "within 24 hours",
    "expires today",
    "final warning",
    "last chance",
    "immediately",
    "limited time",
    "act now",
    "avoid suspension",
    "failure to respond",
]

REASSURING_PHRASES = [
    {
        "phrase": "no password is requested",
        "explanation": (
            "The message says it does not request a password. "
            "This alone does not prove the message is safe."
        ),
    },
    {
        "phrase": "no otp is requested",
        "explanation": (
            "The message says it does not request an OTP. "
            "Other parts may still be risky."
        ),
    },
    {
        "phrase": "does not require you to disclose confidential information",
        "explanation": (
            "The message says confidential information is not required. "
            "Treat this as context, not a safety guarantee."
        ),
    },
]


def normalize_defanged_url(url):
    """Normalize common defanged URL notation without visiting it."""
    value = str(url).strip()

    value = re.sub(r"(?i)^hxxps(?=://)", "https", value)
    value = re.sub(r"(?i)^hxxp(?=://)", "http", value)
    value = re.sub(r"(?i)\[\.\]|\(\.\)|\{\.\}", ".", value)
    value = re.sub(r"(?i)\[:\]", ":", value)

    return value


def analyze_message_cues(message):
    """Detect known suspicious phrases and return their positions."""
    text = str(message or "")
    lowered = text.lower()

    matches = []
    urgency_matches = []
    reassuring_matches = []

    for item in RISK_PHRASES:
        phrase = item["phrase"]
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"

        for found in re.finditer(pattern, lowered):
            matches.append({
                "phrase": text[found.start():found.end()],
                "normalized_phrase": phrase,
                "severity": item["severity"],
                "start": found.start(),
                "end": found.end(),
                "explanation": item["explanation"],
            })

    for phrase in URGENCY_PHRASES:
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"

        for found in re.finditer(pattern, lowered):
            urgency_matches.append({
                "phrase": text[found.start():found.end()],
                "start": found.start(),
                "end": found.end(),
                "explanation": (
                    "This wording may create urgency or pressure. "
                    "Urgency alone does not prove phishing."
                ),
            })

    for item in REASSURING_PHRASES:
        phrase = item["phrase"]
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"

        for found in re.finditer(pattern, lowered):
            reassuring_matches.append({
                "phrase": text[found.start():found.end()],
                "start": found.start(),
                "end": found.end(),
                "explanation": item["explanation"],
            })

    matches.sort(key=lambda item: item["start"])
    urgency_matches.sort(key=lambda item: item["start"])
    reassuring_matches.sort(key=lambda item: item["start"])

    if any(item["severity"] == "high" for item in matches):
        summary = "High-risk wording detected. Verify the request independently."
    elif matches or urgency_matches:
        summary = "Potential risk cues detected. Review the message carefully."
    else:
        summary = "No phrases from the current rule list were detected."

    return {
        "matches": matches,
        "urgency_matches": urgency_matches,
        "reassuring_matches": reassuring_matches,
        "summary": summary,
        "disclaimer": (
            "Phrase matching is rule-based. A detected phrase is not proof "
            "of phishing, and no detected phrase does not prove a message is safe."
        ),
    }


def is_ip_address(hostname):
    """Check whether a hostname is an IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def analyze_url(url):
    """
    Analyze URL structure offline.

    This function never opens or fetches a URL. The score is a heuristic,
    not a probability or a guarantee that a destination is safe.
    """
    original = str(url or "").strip()
    findings = []

    def add_finding(level, title, explanation):
        findings.append({
            "level": level,
            "title": title,
            "explanation": explanation,
        })

    if not original:
        return {
            "input": original,
            "normalized_url": "",
            "risk_level": "Unknown",
            "score": 0,
            "findings": [{
                "level": "info",
                "title": "No URL supplied",
                "explanation": "Paste a URL to analyze its structure.",
            }],
            "disclaimer": "No link was visited.",
        }

    candidate = normalize_defanged_url(original)

    if any(char in candidate for char in ("\r", "\n", "\t", "\x00")):
        add_finding(
            "high",
            "Unexpected control characters",
            "The input contains control characters. Check the copied URL carefully.",
        )

        return {
            "input": original,
            "normalized_url": candidate,
            "risk_level": "High",
            "score": 80,
            "findings": findings,
            "disclaimer": "Offline structural assessment only. No link was visited.",
        }

    try:
        parsed = urlsplit(candidate)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        parsed = None
        scheme = ""
        hostname = None
        port = None

    if parsed is None or scheme not in {"http", "https"} or not hostname:
        add_finding(
            "medium",
            "URL could not be parsed reliably",
            "Provide a complete http:// or https:// URL.",
        )

        return {
            "input": original,
            "normalized_url": candidate,
            "risk_level": "Unknown",
            "score": 0,
            "findings": findings,
            "disclaimer": (
                "Unable to make a reliable structural assessment. "
                "This is not a safety verdict."
            ),
        }

    score = 0

    if scheme != "https":
        score += 15
        add_finding(
            "medium",
            "No HTTPS",
            "The URL uses HTTP. This is a risk indicator, but HTTPS alone does not establish trust.",
        )

    if parsed.username is not None or parsed.password is not None:
        score += 25
        add_finding(
            "high",
            "Embedded username or password",
            "User-information fields can disguise the actual destination hostname.",
        )

    if is_ip_address(hostname):
        score += 25
        add_finding(
            "high",
            "IP address used as hostname",
            "The destination uses a raw IP address instead of a conventional domain.",
        )

    try:
        ascii_hostname = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError:
        ascii_hostname = hostname.lower()
        score += 15
        add_finding(
            "medium",
            "Unusual hostname encoding",
            "The hostname could not be normalized cleanly.",
        )

    if "xn--" in ascii_hostname:
        score += 20
        add_finding(
            "high",
            "Internationalized hostname",
            "Punycode can appear in lookalike domains. Inspect the domain carefully.",
        )

    if len(ascii_hostname) > 50:
        score += 10
        add_finding(
            "low",
            "Long hostname",
            "An unusually long hostname can be harder to inspect.",
        )

    if ascii_hostname.count(".") >= 4:
        score += 10
        add_finding(
            "low",
            "Many subdomain levels",
            "Inspect the registered domain carefully; multiple subdomains are not proof of phishing.",
        )

    if re.search(r"\d{5,}", ascii_hostname):
        score += 10
        add_finding(
            "low",
            "Long digit sequence in hostname",
            "The hostname contains a long digit sequence. This may be unusual but is not proof of malicious intent.",
        )

    if any(char in original for char in ("[.]", "(.)", "{.}")):
        add_finding(
            "info",
            "Defanged URL notation",
            "The input uses defanged notation. It was normalized for analysis only.",
        )

    if re.search(r"(?i)hxxps?://", original):
        add_finding(
            "info",
            "Defanged scheme",
            "The URL uses hxxp or hxxps notation. It was normalized for analysis only.",
        )

    if port is not None and port not in {80, 443}:
        score += 10
        add_finding(
            "low",
            "Unusual port",
            f"The URL specifies port {port}. This may be legitimate but deserves scrutiny.",
        )

    decoded_path = unquote(parsed.path or "").lower()
    decoded_query = unquote(parsed.query or "").lower()
    combined_path_query = decoded_path + " " + decoded_query

    sensitive_terms = [
        "login",
        "verify",
        "password",
        "credential",
        "account",
        "secure",
        "update",
        "wallet",
        "payment",
    ]

    matched_terms = [
        term for term in sensitive_terms
        if term in combined_path_query
    ]

    if matched_terms:
        add_finding(
            "info",
            "Account-related wording in URL",
            "The path or query contains: "
            + ", ".join(matched_terms)
            + ". These terms also appear on legitimate websites.",
        )

    score = min(score, 100)

    if score >= 50:
        risk_level = "High"
    elif score >= 25:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    if not findings:
        add_finding(
            "info",
            "No listed structural indicators found",
            "No indicators from this rule set were detected. This does not mean the URL is safe.",
        )

    return {
        "input": original,
        "normalized_url": candidate,
        "hostname": hostname,
        "scheme": scheme,
        "risk_level": risk_level,
        "score": score,
        "findings": findings,
        "disclaimer": (
            "Offline heuristic assessment only. No link was opened or fetched. "
            "This scanner does not check domain reputation, website content, "
            "or live threat intelligence. A low score does not guarantee safety."
        ),
    }