"""Pure, idempotent business-name normalization; raw values are retained elsewhere."""

from __future__ import annotations

import re
import unicodedata

_SPACE = re.compile(r"\s+")
_APOSTROPHE = re.compile(r"(?<=\w)['’](?=\w)")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_LEGAL_SUFFIXES = frozenset({
    "pvt", "private", "limited", "ltd", "llc", "inc", "incorporated",
    "corp", "corporation", "company",
})


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = _APOSTROPHE.sub("", value)
    value = value.replace("&", " and ")
    value = _PUNCT.sub(" ", value)
    return _SPACE.sub(" ", value).strip()


def name_views(raw: str) -> tuple[str, str]:
    clean = clean_text(raw)
    tokens = clean.split()
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    core = " ".join(tokens) or clean
    return clean, core
