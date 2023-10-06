from dataclasses import dataclass
from typing import List, Literal, Optional, TypedDict

DeviceType = Literal['cpu', 'cuda', 'auto']
TaskType = Literal['transcribe', 'translate']
LanguageCode = Literal[
    'af',
    'am',
    'ar',
    'as',
    'az',
    'ba',
    'be',
    'bg',
    'bn',
    'bo',
    'br',
    'bs',
    'ca',
    'cs',
    'cy',
    'da',
    'de',
    'el',
    'en',
    'es',
    'et',
    'eu',
    'fa',
    'fi',
    'fo',
    'fr',
    'gl',
    'gu',
    'ha',
    'haw',
    'he',
    'hi',
    'hr',
    'ht',
    'hu',
    'hy',
    'id',
    'is',
    'it',
    'ja',
    'jw',
    'ka',
    'kk',
    'km',
    'kn',
    'ko',
    'la',
    'lb',
    'ln',
    'lo',
    'lt',
    'lv',
    'mg',
    'mi',
    'mk',
    'ml',
    'mn',
    'mr',
    'ms',
    'mt',
    'my',
    'ne',
    'nl',
    'nn',
    'no',
    'oc',
    'pa',
    'pl',
    'ps',
    'pt',
    'ro',
    'ru',
    'sa',
    'sd',
    'si',
    'sk',
    'sl',
    'sn',
    'so',
    'sq',
    'sr',
    'su',
    'sv',
    'sw',
    'ta',
    'te',
    'tg',
    'th',
    'tk',
    'tl',
    'tr',
    'tt',
    'uk',
    'ur',
    'uz',
    'vi',
    'yi',
    'yo',
    'zh',
]

ComputeType = Literal[
    'int8', 'int8_float32', 'int8_float16', 'int8_bfloat16',
    'int16', 'float16', 'bfloat16', 'float32'
]

ModelSize = Literal[
    'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en',
    'medium', 'medium.en', 'large-v1', 'large-v2', 'large'
]

ModelArchive = str
ModelArchiveOrSize = ModelSize | ModelArchive


class SingleSegment(TypedDict):
    """
    A single word of a speech.
    """
    word: str
    start: float
    end: float
    score: float


class SingleWordSegment(TypedDict):
    """
    A single word of a speech.
    """
    word: str
    start: float
    end: float
    score: float


class SingleCharSegment(TypedDict):
    """
    A single char of a speech.
    """
    char: str
    start: float
    end: float
    score: float


class SingleAlignedSegment(TypedDict):
    """
    A single segment (up to multiple sentences) of a speech with word alignment.
    """

    start: float
    end: float
    text: str
    words: List[SingleWordSegment]
    chars: Optional[List[SingleCharSegment]]


class AlignedTranscriptionResult(TypedDict):
    """
    A list of segments and word segments of a speech.
    """
    segments: List[SingleAlignedSegment]
    word_segments: List[SingleWordSegment]
