"""Hash phone numbers, emails, and person names before any other step."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable

import pandas as pd

from pravah.ingest.constants import ANON_PREFIX, DEFAULT_PII_SALT, DATETIME_PARSE_RATE
from pravah.ingest.datetime_parse import is_stringy, parse_rate

logger = logging.getLogger("pravah.ingest.anonymise")

_NAME = re.compile(
    r"^[A-Za-z][A-Za-z'\-]{1,30}(?:\s+[A-Za-z][A-Za-z'\-]{1,30}){0,3}$"
)
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_NAME_HEADER = {
    "name",
    "fullname",
    "firstname",
    "lastname",
    "surname",
    "tutor",
    "teacher",
    "parent",
    "guardian",
    "counsellor",
    "counselor",
}
_PHONE_HEADER = {"phone", "mobile", "whatsapp", "telephone", "tel", "cell"}
_EMAIL_HEADER = {"email", "mail"}
_PERSON_HEADER = _NAME_HEADER | {
    "from",
    "to",
    "sender",
    "recipient",
    "author",
    "profile",
    "linkedin",
    "username",
    "handle",
    "contact",
}
_URL_HEADER = {"url", "link", "profile", "linkedin", "website"}


def _tokens(header: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", header.lower()))


def _text_values(series: pd.Series) -> list[str]:
    values: list[str] = []
    for raw in series.dropna():
        text = str(raw).strip()
        if text and text.lower() not in {"nan", "none", "nat"}:
            values.append(text)
    return values


def _match_rate(values: Iterable[str], pattern: re.Pattern[str]) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    hits = sum(1 for value in rows if pattern.match(value) is not None)
    return hits / len(rows)


def _is_phone(value: str) -> bool:
    if re.search(r"[A-Za-z]", value):
        return False
    digits = re.sub(r"\D", "", value)
    return 10 <= len(digits) <= 15


def phone_match_rate(series: pd.Series) -> float:
    """Share of non-missing values that look like a phone number."""
    values = _text_values(series)
    if not values:
        return 0.0
    return sum(1 for value in values if _is_phone(value)) / len(values)


def email_match_rate(series: pd.Series) -> float:
    """Share of non-missing values that look like an email address."""
    return _match_rate(_text_values(series), _EMAIL)


def url_match_rate(series: pd.Series) -> float:
    """Share of non-missing values that contain a web address."""
    return _match_rate(_text_values(series), _URL)


def name_match_rate(series: pd.Series) -> float:
    """Share of non-missing values that look like a personal name."""
    return _match_rate(_text_values(series), _NAME)


def _mean_word_count(series: pd.Series) -> float:
    values = _text_values(series)
    if not values:
        return 0.0
    return sum(len(value.split()) for value in values) / len(values)


def hash_value(value: str, *, salt: str, column: str) -> str:
    """Stable short hash. The column name is part of the input so two fields do not collide."""
    payload = f"{salt}|{column}|{value}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{ANON_PREFIX}{digest}"


def _should_hash(header: str, series: pd.Series) -> str | None:
    """Return the reason to hash this column, or None to leave it alone."""
    if parse_rate(series) >= DATETIME_PARSE_RATE:
        return None
    if not is_stringy(series) and not pd.api.types.is_numeric_dtype(series):
        return None
    tokens = _tokens(header)
    phone_rate = phone_match_rate(series)
    email_rate = email_match_rate(series)
    names = name_match_rate(series)
    urls = url_match_rate(series)
    n_rows = max(len(series), 1)
    nunique = series.dropna().nunique()
    uniqueness = nunique / n_rows

    if urls >= 0.40 or (tokens & _URL_HEADER and urls >= 0.20):
        return "url"
    if tokens & _PHONE_HEADER or phone_rate >= 0.70:
        if phone_rate >= 0.50 or tokens & _PHONE_HEADER:
            return "phone"
    if tokens & _EMAIL_HEADER or email_rate >= 0.70:
        if email_rate >= 0.50 or tokens & _EMAIL_HEADER:
            return "email"
    if tokens & _PERSON_HEADER and names >= 0.40:
        return "name"
    if tokens & _NAME_HEADER and names >= 0.50:
        return "name"
    if names >= 0.80 and _mean_word_count(series) >= 1.8 and uniqueness >= 0.30:
        return "name"
    return None


def anonymise_frame(
    frame: pd.DataFrame,
    *,
    salt: str = DEFAULT_PII_SALT,
) -> tuple[pd.DataFrame, list[str]]:
    """Replace phone numbers, emails, and person names with stable hashes.

    WHAT: A column is hashed when its values match a phone pattern (10 to 15
    digits, no letters), an email pattern, or a personal-name pattern, with
    the rates in ``_should_hash``. Header tokens such as ``phone``, ``email``,
    and ``name`` are extra evidence. They are matched as whole tokens, so
    ``student_id`` is not treated as a name. Each kept value becomes
    ``anon:`` plus the first 16 hex characters of SHA-256(salt, column, value).

    WHY: Later tables group by tutor and region. A hash keeps those groups
    without writing a phone number or a student's name into a result file.
    Including the column name in the hash stops a phone and a case id that
    happen to contain the same digits from looking like the same person.

    WHY NOT: Dropping the column would be safer and is rejected because the
    resource analysis needs a stable stand-in for the person. Encrypting with
    a hidden key would stop even the uploader from recognising a tutor they
    already know; a salt they can see is enough to keep the raw text out of
    logs. A header blocklist alone is rejected because ``mobile_no`` and a
    column of bare numbers would slip through, and a value pattern alone is
    rejected because the single name ``Anita`` is also a perfectly good
    activity label.

    ASSUMPTIONS: Personal data is a phone, an email, or a name-like string.
    The salt stays constant inside one deployment so the same tutor hashes to
    the same token. Callers pass the salt; the default is the project salt.

    COMPLEXITY: ``O(n m)`` time for ``n`` rows and ``m`` columns, and
    ``O(n)`` extra space for the copied frame.

    FAILS WHEN: A short local phone (under 10 digits) is kept. A single given
    name without a name-like header is kept, because it cannot be told apart
    from a category. A free-text column of full names with no header is hashed
    when most values are two or more words. A stolen salt lets someone hash
    likely phone numbers and look for a match; the hash is not a secret.
    """
    copied = frame.copy()
    hashed: list[str] = []
    for name in copied.columns:
        reason = _should_hash(str(name), copied[name])
        if reason is None:
            continue
        series = copied[name]

        def _cell(value: object, *, column: str = str(name)) -> object:
            if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
                return pd.NA
            text = str(value).strip()
            if text == "" or text.lower() in {"nan", "none", "nat"}:
                return pd.NA
            return hash_value(text, salt=salt, column=column)

        copied[name] = series.map(_cell)
        hashed.append(str(name))
        logger.info("Anonymised column %s as %s", name, reason)
    return copied, hashed
