"""Detect a user request to switch the spoken/response language.

Conservative, phrase-anchored detection so normal talk about languages ("I learned
French in school") doesn't trigger a switch — we require an explicit instruction
("speak French", "respond in Spanish", "switch to Japanese", "habla español").

Returns an ISO-ish language code + English name; the caller updates the active
session language, hands the code to OmniVoice, and nudges the LLM to reply in it.
Pure/stateless so it's trivially unit testable.
"""

from __future__ import annotations

import re

# code -> English display name
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "ar": "Arabic",
    "tr": "Turkish",
    "pl": "Polish",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
}

# lowercased name/alias (English + common native) -> code
_NAME_TO_CODE: dict[str, str] = {}
for _code, _name in LANGUAGE_NAMES.items():
    _NAME_TO_CODE[_name.lower()] = _code
_NAME_TO_CODE.update(
    {
        "english": "en",
        "spanish": "es", "espanol": "es", "español": "es", "castellano": "es",
        "french": "fr", "francais": "fr", "français": "fr",
        "german": "de", "deutsch": "de",
        "italian": "it", "italiano": "it",
        "portuguese": "pt", "portugues": "pt", "português": "pt",
        "dutch": "nl", "nederlands": "nl",
        "russian": "ru",
        "chinese": "zh", "mandarin": "zh",
        "japanese": "ja",
        "korean": "ko",
        "hindi": "hi",
        "arabic": "ar",
        "turkish": "tr",
        "polish": "pl",
        "swedish": "sv",
        "norwegian": "no",
        "danish": "da",
        "finnish": "fi",
        "greek": "el",
        "hebrew": "he",
        "thai": "th",
        "vietnamese": "vi",
        "indonesian": "id",
        "ukrainian": "uk",
        "czech": "cs",
        "romanian": "ro",
        "hungarian": "hu",
    }
)

# Alternation of every known language name/alias, longest first so e.g. a
# multi-word match would win. Putting it directly in the capture group lets the
# regex backtrack past filler ("talk to me in Portuguese") to find a real language.
_LANGS = "|".join(re.escape(name) for name in sorted(_NAME_TO_CODE, key=len, reverse=True))
# "...switch/respond/speak... in/to <language>"
_SWITCH_IN_RE = re.compile(
    r"\b(?:speak|talk|respond|reply|answer|converse|switch(?:ing)?|change)\b"
    r"(?:\s+[\w']+){0,4}?\s+(?:in|to|into)\s+(?:the\s+)?(" + _LANGS + r")\b",
    re.IGNORECASE,
)
# "speak/talk <language>" directly, e.g. "can you speak Spanish?"
_SPEAK_DIRECT_RE = re.compile(
    r"\b(?:speak|talk)\b(?:\s+[\w']+){0,3}?\s+(" + _LANGS + r")\b",
    re.IGNORECASE,
)
# Native imperatives: "habla español", "parle français", "sprich deutsch", ...
_NATIVE_RE = re.compile(
    r"\b(?:habla|parle|parla|sprich|spreek|fala)\b(?:\s+en)?\s+(" + _LANGS + r")\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().replace("’", "'")).lower()


def detect_language_request(text: str) -> tuple[str, str] | None:
    """Return (code, English name) if the user asked to switch languages, else None."""
    normalized = _normalize(text)
    if not normalized:
        return None
    for regex in (_SWITCH_IN_RE, _SPEAK_DIRECT_RE, _NATIVE_RE):
        m = regex.search(normalized)
        if not m:
            continue
        code = _NAME_TO_CODE.get(m.group(1))
        if code:
            return code, LANGUAGE_NAMES[code]
    return None


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get((code or "").strip().lower(), code)
