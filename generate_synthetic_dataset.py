
"""
PhishMan — Synthetic Dataset Generator
File: src/generate_synthetic_dataset.py

Generates:
    data/messages_synthetic_1000.csv
    data/messages_raw.csv
    data/source_manifest.csv
    data/robustness_cases.csv

All messages are synthetic. URLs use reserved .example domains.
No external URLs are fetched or opened.
"""

from __future__ import annotations

import csv
import random
import re
import shutil
from collections import Counter
from datetime import date, timedelta
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
TOTAL_CAMPAIGNS = 100
PER_LABEL_PER_CAMPAIGN = 5
EXPECTED_TOTAL = TOTAL_CAMPAIGNS * PER_LABEL_PER_CAMPAIGN * 2

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

SYNTHETIC_PATH = DATA_DIR / "messages_synthetic_1000.csv"
RAW_PATH = DATA_DIR / "messages_raw.csv"
MANIFEST_PATH = DATA_DIR / "source_manifest.csv"
ROBUSTNESS_PATH = DATA_DIR / "robustness_cases.csv"

FIELDS = [
    "message_id",
    "channel",
    "subject_or_title",
    "body_text",
    "sanitized_url_features",
    "sender_domain_type",
    "label",
    "campaign_group",
    "collected_date",
]

CHANNELS = ["email", "sms", "chat", "web_notification"]

SENDER_TYPES = [
    "known_organization",
    "third_party_service",
    "unverified_external",
    "unknown",
]

SERVICES = [
    "student portal",
    "online banking service",
    "shopping account",
    "delivery service",
    "subscription account",
    "utility billing service",
    "workplace account",
    "travel booking service",
    "customer support account",
    "digital payment service",
]

EVENTS = [
    ("monthly statement", "a statement is ready for review"),
    ("account notice", "an account notice has been published"),
    ("service update", "a service update has been recorded"),
    ("scheduled maintenance", "a maintenance notice is available"),
    ("transaction record", "a transaction record is available"),
    ("profile review", "profile information has been reviewed"),
    ("subscription renewal", "subscription information has been updated"),
    ("delivery update", "a delivery status update is available"),
    ("security notice", "a security notice has been published"),
    ("billing summary", "a billing summary is ready"),
]

# The wording pools deliberately offer many combinations.
LEGIT_SUBJECTS = [
    "Your {event}",
    "Information about your {service}",
    "A routine update for your {service}",
    "Your latest account records",
    "Notice: {event}",
    "Account information available",
    "An update from the service team",
    "Review your latest account notice",
    "Service information for your records",
    "A new entry in your account history",
    "Your requested account summary",
    "A routine service notification",
    "Information for your next account review",
    "Your account activity summary",
    "Update to your service records",
]

PHISH_SUBJECTS = [
    "Action requested for your {service}",
    "Urgent account verification",
    "Your account requires attention",
    "Final notice about account access",
    "Resolve a pending account issue",
    "Verification request awaiting completion",
    "Important account restriction notice",
    "Confirm information to restore access",
    "An unresolved account action",
    "Immediate review requested",
    "Your account status needs confirmation",
    "Pending verification notification",
    "Account access may be interrupted",
    "Security review requires a response",
    "Complete the requested account check",
]

LEGIT_OPENINGS = [
    "This notification is provided for your records.",
    "The service has published a routine account update.",
    "A new entry has been added to your account history.",
    "The account information system has recorded an update.",
    "This message summarises a routine service event.",
    "The following notice is available for your review.",
    "Your account records now include a recent update.",
    "The service team has made new information available.",
    "A routine notification has been generated for your account.",
    "This message contains an update about your service.",
    "The latest account information is ready to review.",
    "A service notification has been prepared for you.",
    "Your account summary has been refreshed.",
    "The service has recorded the latest status information.",
    "This is an informational message about your account.",
]

LEGIT_ACTIONS = [
    "Open the official application or use your usual saved bookmark to review it.",
    "If anything looks unfamiliar, contact support using the details on the official website.",
    "No immediate action is necessary unless the information appears incorrect.",
    "You can compare the notice with the records shown in your normal account interface.",
    "Keep this message if you need a record of the notification.",
    "For additional information, use the service's established help centre.",
    "Review the details through your usual sign-in process.",
    "You do not need to reply to this message.",
    "If you no longer use the service, you may disregard the notice.",
    "Contact the service through a known support channel if you have questions.",
    "The information is available through the service's usual account interface.",
    "Check your records independently rather than relying on message links.",
    "No password or one-time verification code is requested in this notice.",
    "You can review the update at a convenient time.",
    "Use the official application to check whether the update applies to you.",
]

LEGIT_ENDINGS = [
    "Thank you for using the service.",
    "This notice is for information only.",
    "Please retain it if it is useful to you.",
    "The service will not ask you to email your password.",
    "No reply is required.",
    "Support remains available through the usual official channel.",
    "This notification does not require you to disclose confidential information.",
    "Thank you for reviewing this update.",
    "Please follow your normal account-security practices.",
    "The notice is complete.",
    "You can verify the information independently.",
    "This message was generated for routine account communication.",
]

PHISH_OPENINGS = [
    "A problem has reportedly been detected with your {service}.",
    "The sender claims that your account needs immediate verification.",
    "An unresolved issue has allegedly been linked to your account.",
    "This message reports a pending account restriction.",
    "A recent account process is described as incomplete.",
    "The notice claims that account access may soon be limited.",
    "The sender reports an unexpected account status change.",
    "An account review is presented as urgent.",
    "The message claims that further confirmation is required.",
    "A service issue is reported and immediate action is requested.",
    "The sender says that a recent account check was unsuccessful.",
    "A pending account action is described as time-sensitive.",
    "This notice alleges that your account needs to be restored.",
    "The message reports a verification request awaiting completion.",
    "The sender claims that your account may be affected by a restriction.",
]

PHISH_ACTIONS = [
    "The message asks you to enter your password and one-time verification code.",
    "It directs the recipient to submit sign-in credentials to avoid a restriction.",
    "The sender requests account and payment information through a supplied page.",
    "The message asks for a security code to complete the claimed review.",
    "It requests personal information through an unverified destination.",
    "The sender asks the recipient to confirm credentials to restore access.",
    "It requests account details before the stated deadline.",
    "The message directs the recipient to a page to validate payment information.",
    "The sender asks for confidential information to release a pending action.",
    "It requests sign-in information to prevent a claimed service interruption.",
    "The message asks the recipient to complete a verification form.",
    "The sender requests a code that could be used to access the account.",
    "It asks the recipient to confirm sensitive details through the supplied destination.",
    "The message requests credentials as part of an alleged account check.",
    "It directs the recipient to provide security information to continue.",
]

PHISH_ENDINGS = [
    "The sender warns that access may be restricted if the request is ignored.",
    "The message describes the request as urgent.",
    "The sender claims that the request will expire shortly.",
    "The notice warns of a possible account limitation.",
    "The message urges the recipient to respond without delay.",
    "The sender claims that the issue must be resolved immediately.",
    "The request is presented as the final reminder.",
    "The message warns that service may be interrupted.",
    "The sender says that verification is necessary to maintain access.",
    "The notice claims that delayed action could affect the account.",
    "The sender requests completion before a stated deadline.",
    "The message claims that a restriction can be avoided by responding.",
]

PHISH_URLS = [
    "hxxps://account-check[.]example/verify",
    "hxxps://secure-review[.]example/session",
    "hxxps://billing-confirm[.]example/update",
    "hxxps://identity-check[.]example/confirm",
    "hxxps://service-alert[.]example/restore",
    "hxxps://payment-review[.]example/validate",
    "hxxps://access-support[.]example/login",
    "hxxps://profile-confirm[.]example/secure",
    "hxxps://notice-center[.]example/action",
    "hxxps://account-status[.]example/resolve",
]

LEGIT_CONTEXTS = [
    "The update concerns a recent change to the account record.",
    "The information may be useful when checking previous activity.",
    "The notice is associated with a routine service event.",
    "The account history contains further details about this entry.",
    "The notification was prepared as part of normal service communication.",
    "The information is intended to help the recipient review account activity.",
    "The event has been recorded in the service history.",
    "The update does not itself indicate that the account is compromised.",
    "The service may display more details after normal sign-in.",
    "The notice provides context for a recently recorded event.",
]

PHISH_CONTEXTS = [
    "The message claims that failure to respond could affect account access.",
    "The request is framed as necessary to prevent a service interruption.",
    "The sender presents the action as a condition for keeping the account active.",
    "The message does not establish that the sender is an authorised representative.",
    "The claimed issue has not been independently verified.",
    "The notice uses a deadline to encourage a quick response.",
    "The sender links the requested information to an alleged account review.",
    "The request relies on the recipient accepting the sender's claims.",
    "The message presents the supplied destination as part of the verification process.",
    "The sender claims that the requested action will resolve the reported issue.",
]

REFERENCE_SENTENCES = [
    "This is a synthetic example for security evaluation.",
    "The message is generated locally for the project dataset.",
    "This fictional notice contains no real account information.",
    "The example does not represent a real service communication.",
]


# ============================================================
# TEXT AND CONTEXT HELPERS
# ============================================================

def normalize(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def get_context(campaign_number: int) -> tuple[str, str, str]:
    service_index = (campaign_number - 1) // len(EVENTS)
    event_index = (campaign_number - 1) % len(EVENTS)

    service = SERVICES[service_index]
    event, description = EVENTS[event_index]

    return service, event, description


def build_message(
    label: str,
    campaign_number: int,
    variant: int,
    rng: random.Random,
) -> tuple[str, str, str]:
    """
    Build a message using seeded random selection.

    Each row receives a separate RNG derived from its stable coordinates.
    This avoids a repeating modular selection pattern while keeping the
    generated dataset reproducible.
    """
    service, event, description = get_context(campaign_number)

    if label == "legitimate":
        subject = rng.choice(LEGIT_SUBJECTS).format(
            service=service.title(),
            event=event,
        )

        opening = rng.choice(LEGIT_OPENINGS).format(
            service=service,
            event=event,
        )

        action = rng.choice(LEGIT_ACTIONS)
        ending = rng.choice(LEGIT_ENDINGS)
        context = rng.choice(LEGIT_CONTEXTS)

        parts = [
            opening,
            description.capitalize() + ".",
            context,
            action,
        ]

        # Some legitimate messages concern routine deadlines or time-sensitive
        # events, without asking for credentials or confidential information.
        if variant == 2:
            parts.append(
                "If this update relates to a deadline, check the date "
                "in the official account interface."
            )

        parts.append(ending)

        # A few harmless variants contain no external destination at all.
        url_feature = "none_detected"

    else:
        subject = rng.choice(PHISH_SUBJECTS).format(
            service=service.title(),
            event=event,
        )

        opening = rng.choice(PHISH_OPENINGS).format(
            service=service,
            event=event,
        )

        action = rng.choice(PHISH_ACTIONS)
        ending = rng.choice(PHISH_ENDINGS)
        context = rng.choice(PHISH_CONTEXTS)

        parts = [
            opening,
            description.capitalize() + ".",
            context,
        ]

        # Half of the examples contain a defanged reserved-domain URL;
        # the others refer to a destination without displaying one.
        if rng.random() < 0.5:
            parts.append(
                "The message supplies this destination: "
                + rng.choice(PHISH_URLS)
                + "."
            )
            url_feature = "reserved_example_domain"
        else:
            parts.append(
                "The sender refers to a verification page without "
                "providing a fully visible destination."
            )
            url_feature = "destination_not_verified"

        parts.extend([action, ending])

    # Add a neutral synthetic disclaimer to reduce exact repetitions.
    # This does not include real identifiers or real account data.
    parts.append(rng.choice(REFERENCE_SENTENCES))

    body = " ".join(parts)
    return subject, body, url_feature


def choose_metadata(
    campaign_number: int,
    variant: int,
) -> tuple[str, str, str]:
    """
    Choose metadata independently of the label.

    Both classes can occur with every channel and sender-domain type.
    """
    metadata_rng = random.Random(
        SEED + campaign_number * 1009 + variant * 9176
    )

    channel = metadata_rng.choice(CHANNELS)
    sender_type = metadata_rng.choice(SENDER_TYPES)

    start = date(2026, 1, 1)
    day_offset = metadata_rng.randint(0, 269)
    collected_date = (start + timedelta(days=day_offset)).isoformat()

    return channel, sender_type, collected_date


# ============================================================
# DATASET GENERATION
# ============================================================

def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    message_number = 1

    for campaign_number in range(1, TOTAL_CAMPAIGNS + 1):
        campaign_group = f"campaign_{campaign_number:03d}"

        labels = (
            ["legitimate"] * PER_LABEL_PER_CAMPAIGN
            + ["phishing"] * PER_LABEL_PER_CAMPAIGN
        )

        campaign_rng = random.Random(SEED + campaign_number * 7919)
        campaign_rng.shuffle(labels)

        variants = Counter()

        for label in labels:
            variant = variants[label]
            variants[label] += 1

            # Independent deterministic randomness for each message.
            message_rng = random.Random(
                SEED
                + campaign_number * 100_003
                + (1 if label == "phishing" else 0) * 10_007
                + variant * 503
            )

            subject, body, url_feature = build_message(
                label=label,
                campaign_number=campaign_number,
                variant=variant,
                rng=message_rng,
            )

            key = (normalize(subject), normalize(body))

            if key in seen:
                raise RuntimeError(
                    "Duplicate subject/body combination found: "
                    f"{campaign_group}, label={label}, variant={variant}. "
                    "The duplicate check has not been disabled. "
                    "Review the generation rules before proceeding."
                )

            seen.add(key)

            channel, sender_type, collected_date = choose_metadata(
                campaign_number,
                variant,
            )

            rows.append(
                {
                    "message_id": f"MSG{message_number:04d}",
                    "channel": channel,
                    "subject_or_title": subject,
                    "body_text": body,
                    "sanitized_url_features": url_feature,
                    "sender_domain_type": sender_type,
                    "label": label,
                    "campaign_group": campaign_group,
                    "collected_date": collected_date,
                }
            )

            message_number += 1

    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} messages; generated {len(rows)}."
        )

    ids = [row["message_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate message IDs detected.")

    campaign_counts = Counter(row["campaign_group"] for row in rows)
    if len(campaign_counts) != TOTAL_CAMPAIGNS:
        raise RuntimeError("Unexpected number of campaign groups.")

    for campaign, count in campaign_counts.items():
        if count != PER_LABEL_PER_CAMPAIGN * 2:
            raise RuntimeError(
                f"{campaign} has {count} rows; expected 10."
            )

        class_counts = Counter(
            row["label"]
            for row in rows
            if row["campaign_group"] == campaign
        )

        for label in ("legitimate", "phishing"):
            if class_counts[label] != PER_LABEL_PER_CAMPAIGN:
                raise RuntimeError(
                    f"{campaign} has {class_counts[label]} {label} rows."
                )

    return rows


# ============================================================
# OUTPUT WRITERS
# ============================================================

def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_robustness_cases() -> list[dict[str, str]]:
    return [
        {
            "case_id": "ROBUST_001",
            "base_label": "phishing",
            "perturbation": "baseline",
            "message_text": (
                "Your account requires verification. "
                "Use hxxps://account-check[.]example/verify and "
                "provide the requested sign-in details."
            ),
        },
        {
            "case_id": "ROBUST_002",
            "base_label": "phishing",
            "perturbation": "punctuation",
            "message_text": (
                "Your account requires verification!!! "
                "Use hxxps://account-check[.]example/verify... "
                "Provide the requested sign-in details."
            ),
        },
        {
            "case_id": "ROBUST_003",
            "base_label": "phishing",
            "perturbation": "case_change",
            "message_text": (
                "YOUR ACCOUNT REQUIRES VERIFICATION. "
                "Use hxxps://account-check[.]example/verify and "
                "provide the requested sign-in details."
            ),
        },
        {
            "case_id": "ROBUST_004",
            "base_label": "phishing",
            "perturbation": "spacing",
            "message_text": (
                "Your account requires verification.  "
                "Use hxxps://account-check[.]example/verify   "
                "and provide the requested sign-in details."
            ),
        },
        {
            "case_id": "ROBUST_005",
            "base_label": "legitimate",
            "perturbation": "baseline",
            "message_text": (
                "Your monthly statement is available. Open the official "
                "application or use your usual bookmark. "
                "Do not send passwords or verification codes by email."
            ),
        },
        {
            "case_id": "ROBUST_006",
            "base_label": "legitimate",
            "perturbation": "punctuation",
            "message_text": (
                "Your monthly statement is available! Open the official "
                "application, or use your usual bookmark. "
                "Do not send passwords or verification codes by email."
            ),
        },
        {
            "case_id": "ROBUST_007",
            "base_label": "legitimate",
            "perturbation": "case_change",
            "message_text": (
                "YOUR MONTHLY STATEMENT IS AVAILABLE. Open the official "
                "application or use your usual bookmark. "
                "Do not send passwords or verification codes by email."
            ),
        },
        {
            "case_id": "ROBUST_008",
            "base_label": "legitimate",
            "perturbation": "wording",
            "message_text": (
                "The monthly statement can now be reviewed. Use the official "
                "application or your saved bookmark. "
                "Never email your password or verification code."
            ),
        },
    ]


def write_manifest() -> None:
    fields = [
        "file_name",
        "source_type",
        "description",
        "license_or_provenance",
        "contains_real_personal_data",
        "external_urls_fetched",
    ]

    rows = [
        {
            "file_name": SYNTHETIC_PATH.name,
            "source_type": "synthetic",
            "description": (
                "Locally generated synthetic examples for the PhishMan project."
            ),
            "license_or_provenance": (
                "Generated locally; no external message dataset used."
            ),
            "contains_real_personal_data": "no",
            "external_urls_fetched": "no",
        },
        {
            "file_name": RAW_PATH.name,
            "source_type": "synthetic_copy",
            "description": "Copy of the locally generated synthetic dataset.",
            "license_or_provenance": "Copy of locally generated examples.",
            "contains_real_personal_data": "no",
            "external_urls_fetched": "no",
        },
        {
            "file_name": ROBUSTNESS_PATH.name,
            "source_type": "synthetic_test_cases",
            "description": "Eight synthetic wording and punctuation test cases.",
            "license_or_provenance": "Authored locally for robustness testing.",
            "contains_real_personal_data": "no",
            "external_urls_fetched": "no",
        },
    ]

    write_csv(MANIFEST_PATH, fields, rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("PhishMan — Synthetic Dataset Generator")
    print("=" * 64)

    rows = build_rows()

    backup_path = SYNTHETIC_PATH.with_suffix(".previous.csv")
    if SYNTHETIC_PATH.exists() and not backup_path.exists():
        shutil.copy2(SYNTHETIC_PATH, backup_path)
        print(f"Previous dataset backed up to: {backup_path}")

    write_csv(SYNTHETIC_PATH, FIELDS, rows)
    shutil.copy2(SYNTHETIC_PATH, RAW_PATH)

    robustness = make_robustness_cases()
    write_csv(
        ROBUSTNESS_PATH,
        ["case_id", "base_label", "perturbation", "message_text"],
        robustness,
    )

    write_manifest()

    print("\nGeneration completed successfully.")
    print(f"Total messages: {len(rows)}")
    print(
        "Campaign groups:",
        len({row["campaign_group"] for row in rows}),
    )
    print("Label distribution:", dict(Counter(row["label"] for row in rows)))
    print("Channel distribution:", dict(Counter(row["channel"] for row in rows)))
    print(
        "Sender-domain distribution:",
        dict(Counter(row["sender_domain_type"] for row in rows)),
    )
    print(f"Unique subject/body combinations: {len(rows)}")

    print("\nFiles written:")
    print(f"  {SYNTHETIC_PATH}")
    print(f"  {RAW_PATH}")
    print(f"  {MANIFEST_PATH}")
    print(f"  {ROBUSTNESS_PATH}")

    print("\nNext: prepare the dataset and inspect the near-duplicate report.")
    print("Do not train models until the new report has been reviewed.")


if __name__ == "__main__":
    main()
