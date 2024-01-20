from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from pyannote.core.annotation import Segment

from whisperx.alignment.types import SingleCharSegment, SingleWordSegment

SpeakerId = NewType('SpeakerId', str)


@dataclass(frozen=True)
class SegmentDiarized:
    start: float
    end: float
    speaker: SpeakerId

    @classmethod
    def plain_from_segment(
        cls, create_from: Segment, speaker: SpeakerId = SpeakerId('UNKNOWN')
    ) -> SegmentDiarized:
        return cls(start=create_from.start, end=create_from.end, speaker=speaker)


@dataclass(slots=True)
class SingleDiarizedWordSegment(SingleWordSegment):
    speaker: str


@dataclass(slots=True)
class SingleDiarizedSegment:
    start: float
    end: float
    text: str
    words: list[SingleDiarizedWordSegment]
    chars: list[SingleCharSegment] | None
    speaker: str


@dataclass(slots=True)
class DiarizedTranscriptionResult:
    segments: list[SingleDiarizedSegment]
    word_segments: list[SingleDiarizedWordSegment]
