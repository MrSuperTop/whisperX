from typing import Literal, TypeAlias, cast

from whisperx.types import ALL_LANGUAGES

PUNKT_ABBREVIATIONS = {'dr', 'vs', 'mr', 'mrs', 'prof'}

LANGUAGES_WITHOUT_SPACES = {'ja', 'zh'}

AlignLanguageCodeTorch: TypeAlias = Literal['en', 'fr', 'de', 'es', 'it']
DEFAULT_ALIGN_MODELS_TORCH: dict[AlignLanguageCodeTorch, str] = {
    'en': 'WAV2VEC2_ASR_BASE_960H',
    'fr': 'VOXPOPULI_ASR_BASE_10K_FR',
    'de': 'VOXPOPULI_ASR_BASE_10K_DE',
    'es': 'VOXPOPULI_ASR_BASE_10K_ES',
    'it': 'VOXPOPULI_ASR_BASE_10K_IT',
}

AlignLanguageCodeHF: TypeAlias = Literal[
    'ja',
    'zh',
    'nl',
    'uk',
    'pt',
    'ar',
    'cs',
    'ru',
    'pl',
    'hu',
    'fi',
    'fa',
    'el',
    'tr',
    'da',
    'he',
    'vi',
    'ko',
    'ur',
    'te',
    'hi',
]

DEFAULT_ALIGN_MODELS_HF: dict[AlignLanguageCodeHF, str] = {
    'ja': 'jonatasgrosman/wav2vec2-large-xlsr-53-japanese',
    'zh': 'jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn',
    'nl': 'jonatasgrosman/wav2vec2-large-xlsr-53-dutch',
    'uk': 'Yehor/wav2vec2-xls-r-300m-uk-with-small-lm',
    'pt': 'jonatasgrosman/wav2vec2-large-xlsr-53-portuguese',
    'ar': 'jonatasgrosman/wav2vec2-large-xlsr-53-arabic',
    'cs': 'comodoro/wav2vec2-xls-r-300m-cs-250',
    'ru': 'jonatasgrosman/wav2vec2-large-xlsr-53-russian',
    'pl': 'jonatasgrosman/wav2vec2-large-xlsr-53-polish',
    'hu': 'jonatasgrosman/wav2vec2-large-xlsr-53-hungarian',
    'fi': 'jonatasgrosman/wav2vec2-large-xlsr-53-finnish',
    'fa': 'jonatasgrosman/wav2vec2-large-xlsr-53-persian',
    'el': 'jonatasgrosman/wav2vec2-large-xlsr-53-greek',
    'tr': 'mpoyraz/wav2vec2-xls-r-300m-cv7-turkish',
    'da': 'saattrupdan/wav2vec2-xls-r-300m-ftspeech',
    'he': 'imvladikon/wav2vec2-xls-r-300m-hebrew',
    'vi': 'nguyenvulebinh/wav2vec2-base-vi',
    'ko': 'kresnik/wav2vec2-large-xlsr-korean',
    'ur': 'kingabzpro/wav2vec2-large-xls-r-300m-Urdu',
    'te': 'anuragshas/wav2vec2-large-xlsr-53-telugu',
    'hi': 'theainerd/Wav2Vec2-large-xlsr-hindi',
}

AlignableLanguageCode: TypeAlias = AlignLanguageCodeHF | AlignLanguageCodeTorch
ALIGNABLE_LANGUAGE_CODES: set[AlignableLanguageCode] = {
    *DEFAULT_ALIGN_MODELS_TORCH.keys(),
    *DEFAULT_ALIGN_MODELS_HF.keys(),
}

NotAlignableLanguageCode: TypeAlias = Literal[
    'cy',
    'lo',
    'ha',
    'gl',
    'af',
    'fo',
    'yo',
    'sr',
    'mg',
    'uz',
    'et',
    'jw',
    'tk',
    'hy',
    'ln',
    'sv',
    'br',
    'gu',
    'ta',
    'my',
    'sa',
    'sw',
    'bs',
    'ps',
    'oc',
    'si',
    'ms',
    'bn',
    'yi',
    'ba',
    'az',
    'lb',
    'su',
    'am',
    'tg',
    'mt',
    'tl',
    'kk',
    'ml',
    'lv',
    'tt',
    'sl',
    'ht',
    'sn',
    'th',
    'mi',
    'hr',
    'haw',
    'id',
    'kn',
    'pa',
    'is',
    'ca',
    'no',
    'la',
    'nn',
    'as',
    'eu',
    'mr',
    'lt',
    'yue',
    'bg',
    'sd',
    'ro',
    'so',
    'km',
    'ka',
    'sk',
    'bo',
    'mn',
    'be',
    'mk',
    'sq',
    'ne',
]

NOT_ALIGNABLE_LANGUAGE_CODES: set[NotAlignableLanguageCode] = cast(
    set[NotAlignableLanguageCode], ALL_LANGUAGES.difference(ALIGNABLE_LANGUAGE_CODES)
)
