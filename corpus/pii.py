from __future__ import annotations

import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?:(?:\+\d{1,3}[ -]?)?(?:\(\d{1,4}\)[ -]?)?\d{2,4}[ -]?\d{3,4}[ -]?\d{3,4})")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def redact_pii(text: str) -> str:
    s = _EMAIL.sub("<EMAIL>", text)
    s = _PHONE.sub("<PHONE>", s)
    s = _IP.sub("<IP>", s)
    return s


__all__ = ["redact_pii"]


