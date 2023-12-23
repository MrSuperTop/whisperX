from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Final, Literal

import ctranslate2._ext
from faster_whisper.tokenizer import (
    _LANGUAGE_CODES,  # pyright: ignore [reportPrivateUsage]
)

DeviceType = Literal["cpu", "cuda", "auto"]
DEVICE_TYPES: Final[set[DeviceType]] = set(["cpu", "cuda", "auto"])

TaskType = Literal["transcribe", "translate"]
LanguageCode = Literal[
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "haw",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jw",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "yi",
    "yo",
    "zh",
]

ALL_LANGUAGES = _LANGUAGE_CODES
ComputeType = Literal[
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
]

compute_types: dict[DeviceType, set[ComputeType]] = {}
for device_type in DEVICE_TYPES:
    compute_types[device_type] = ctranslate2._ext.get_supported_compute_types(
        device_type
    )

COMPUTE_TYPES: Final[dict[DeviceType, set[ComputeType]]] = compute_types
ALL_COMPUTE_TYPES = set(chain.from_iterable(COMPUTE_TYPES.values()))

ModelSize = Literal[
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large",
]

ModelArchive = str
ModelArchiveOrSize = ModelSize | ModelArchive


@dataclass(slots=True)
class SingleWordSegment:
    """
    A single word of a speech.
    """

    word: str
    start: float
    end: float
    score: float


@dataclass(slots=True)
class SingleCharSegment:
    """
    A single char of a speech.
    """

    char: str
    start: float | None
    end: float | None
    score: float | None
    word_idx: int = -1
