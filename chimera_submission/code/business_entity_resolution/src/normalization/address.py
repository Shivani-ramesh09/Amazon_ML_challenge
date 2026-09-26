"""Cheap country-aware address signals with an open-set generic fallback."""

from __future__ import annotations

import re

from .text import clean_text

_INDIA_PIN = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")
_US_ZIP = re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)")
_FR_POSTAL = re.compile(r"(?<!\d)\d{5}(?!\d)")
_GENERIC_POSTAL = re.compile(r"\b(?:\d{5,6}|(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-z0-9]{4,8})\b")
_NUMBER = re.compile(r"(?<!\w)\d+[a-z]?(?!\w)")
_ALIASES = {"rd": "road", "ave": "avenue", "blvd": "boulevard", "ln": "lane", "dr": "drive", "apt": "apartment", "ste": "suite"}


def country_key(country: str) -> str:
    return clean_text(country)


def address_views(raw: str, country: str) -> tuple[str, str, str, str]:
    """Return clean address, sorted unique numeric tokens, postal candidate, house number."""
    key = country_key(country)
    cleaned = clean_text(raw)
    cleaned = " ".join(_ALIASES.get(token, token) for token in cleaned.split())
    if key in ("india", "in"):
        match = _INDIA_PIN.search(raw)
    elif key in ("us", "usa", "united states"):
        match = _US_ZIP.search(raw)
    elif key in ("france", "fr"):
        match = _FR_POSTAL.search(raw)
    else:
        match = _GENERIC_POSTAL.search(cleaned)
    postal = match.group(1) if match and key in ("us", "usa", "united states") else (match.group(0).casefold() if match else "")
    numbers = _NUMBER.findall(cleaned)
    numeric_tokens = " ".join(sorted(set(numbers)))
    house = next((number for number in numbers if number != postal), "")
    return cleaned, numeric_tokens, postal, house
