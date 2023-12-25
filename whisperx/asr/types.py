from __future__ import annotations

from dataclasses import dataclass, field

from whisperx.types import LanguageCode


@dataclass()
class LanguageDetails:
    language: LanguageCode
    language_probability: float
    all_language_probs: list[tuple[LanguageCode, float]] | None = field(default=None)

    @classmethod
    def get_defined(cls, for_language: LanguageCode) -> LanguageDetails:
        return cls(
            language=for_language, language_probability=1, all_language_probs=None
        )


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    segments: list[Segment]
    language: LanguageCode
    language_details: LanguageDetails
