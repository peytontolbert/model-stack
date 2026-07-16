#!/usr/bin/env python3
from __future__ import annotations

import re


_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bF5\s*TTS\b", re.IGNORECASE), "F five T T S"),
    (re.compile(r"\bF5TTS\b", re.IGNORECASE), "F five T T S"),
    (re.compile(r"\bF5\b", re.IGNORECASE), "F five"),
    (re.compile(r"\bTTS\b", re.IGNORECASE), "T T S"),
    (re.compile(r"\bWebGPU\b", re.IGNORECASE), "Web G P U"),
    (re.compile(r"\bGPU\b", re.IGNORECASE), "G P U"),
    (re.compile(r"\bWebAssembly\b", re.IGNORECASE), "Web Assembly"),
    (re.compile(r"\bWASM\b", re.IGNORECASE), "Web Assembly"),
    (re.compile(r"\b(?:int4|q4)\b", re.IGNORECASE), "four bit"),
    (re.compile(r"\b4\s*bit\b", re.IGNORECASE), "four bit"),
    (re.compile(r"\b24\s*kHz\b", re.IGNORECASE), "twenty four kilo hertz"),
    (re.compile(r"\bkHz\b", re.IGNORECASE), "kilo hertz"),
    (re.compile(r"\bVocos\b", re.IGNORECASE), "Voh coes"),
    (re.compile(r"\bvoice cloning quality\b", re.IGNORECASE), "voice clone quality"),
)


def normalize_f5tts_speech_text(text: str) -> str:
    normalized = str(text)
    for pattern, replacement in _REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
