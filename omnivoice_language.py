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

# code -> English display name. Covers Google Translate's language set (so beta
# is at parity) and grows from there. Codes are ISO 639-1 where one exists, else
# 639-2/3 / Google's code; the sidecar maps the code to OmniVoice's own scheme.
LANGUAGE_NAMES: dict[str, str] = {
    "af": "Afrikaans", "sq": "Albanian", "am": "Amharic", "ar": "Arabic",
    "hy": "Armenian", "as": "Assamese", "ay": "Aymara", "az": "Azerbaijani",
    "bm": "Bambara", "eu": "Basque", "be": "Belarusian", "bn": "Bengali",
    "bho": "Bhojpuri", "bs": "Bosnian", "bg": "Bulgarian", "ca": "Catalan",
    "ceb": "Cebuano", "ny": "Chichewa", "zh": "Chinese", "co": "Corsican",
    "hr": "Croatian", "cs": "Czech", "da": "Danish", "dv": "Dhivehi",
    "doi": "Dogri", "nl": "Dutch", "en": "English", "eo": "Esperanto",
    "et": "Estonian", "ee": "Ewe", "tl": "Filipino", "fi": "Finnish",
    "fr": "French", "fy": "Frisian", "gl": "Galician", "ka": "Georgian",
    "de": "German", "el": "Greek", "gn": "Guarani", "gu": "Gujarati",
    "ht": "Haitian Creole", "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew",
    "hi": "Hindi", "hmn": "Hmong", "hu": "Hungarian", "is": "Icelandic",
    "ig": "Igbo", "ilo": "Ilocano", "id": "Indonesian", "ga": "Irish",
    "it": "Italian", "ja": "Japanese", "jv": "Javanese", "kn": "Kannada",
    "kk": "Kazakh", "km": "Khmer", "rw": "Kinyarwanda", "gom": "Konkani",
    "ko": "Korean", "kri": "Krio", "ku": "Kurdish", "ckb": "Sorani Kurdish",
    "ky": "Kyrgyz", "lo": "Lao", "la": "Latin", "lv": "Latvian",
    "ln": "Lingala", "lt": "Lithuanian", "lg": "Luganda", "lb": "Luxembourgish",
    "mk": "Macedonian", "mai": "Maithili", "mg": "Malagasy", "ms": "Malay",
    "ml": "Malayalam", "mt": "Maltese", "mi": "Maori", "mr": "Marathi",
    "mni": "Meiteilon", "lus": "Mizo", "mn": "Mongolian", "my": "Burmese",
    "ne": "Nepali", "no": "Norwegian", "or": "Odia", "om": "Oromo",
    "ps": "Pashto", "fa": "Persian", "pl": "Polish", "pt": "Portuguese",
    "pa": "Punjabi", "qu": "Quechua", "ro": "Romanian", "ru": "Russian",
    "sm": "Samoan", "sa": "Sanskrit", "gd": "Scots Gaelic", "nso": "Sepedi",
    "sr": "Serbian", "st": "Sesotho", "sn": "Shona", "sd": "Sindhi",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "so": "Somali",
    "es": "Spanish", "su": "Sundanese", "sw": "Swahili", "sv": "Swedish",
    "tg": "Tajik", "ta": "Tamil", "tt": "Tatar", "te": "Telugu",
    "th": "Thai", "ti": "Tigrinya", "ts": "Tsonga", "tr": "Turkish",
    "tk": "Turkmen", "ak": "Twi", "uk": "Ukrainian", "ur": "Urdu",
    "ug": "Uyghur", "uz": "Uzbek", "vi": "Vietnamese", "cy": "Welsh",
    "xh": "Xhosa", "yi": "Yiddish", "yo": "Yoruba", "zu": "Zulu",
}

# lowercased name/alias (English + common native/alternate) -> code. Display names
# are auto-added; this set adds native forms and common alternates users might say.
_NAME_TO_CODE: dict[str, str] = {}
for _code, _name in LANGUAGE_NAMES.items():
    _NAME_TO_CODE[_name.lower()] = _code
_NAME_TO_CODE.update(
    {
        "espanol": "es", "español": "es", "castellano": "es",
        "francais": "fr", "français": "fr",
        "deutsch": "de",
        "italiano": "it",
        "portugues": "pt", "português": "pt", "brazilian": "pt",
        "nederlands": "nl", "flemish": "nl",
        "mandarin": "zh", "cantonese": "zh", "putonghua": "zh",
        "tagalog": "tl", "pilipino": "tl",
        "farsi": "fa", "persian": "fa",
        "myanmar": "my", "burmese": "my",
        "odia": "or", "oriya": "or",
        "kurmanji": "ku",
        "sorani": "ckb",
        "manipuri": "mni",
        "twi": "ak", "akan": "ak",
        "punjabi": "pa", "panjabi": "pa",
        "scottish gaelic": "gd",
        "haitian": "ht",
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
