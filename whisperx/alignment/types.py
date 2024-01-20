from dataclasses import dataclass


@dataclass(slots=True)
class SingleWordSegment:
    """
    A single word of a speech.
    """

    word: str
    start: float | None
    end: float | None
    score: float | None


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


@dataclass(slots=True)
class SingleAlignedSegment:
    """
    A single segment (up to multiple sentences) of a speech with word alignment.
    """

    start: float
    end: float
    text: str
    words: list[SingleWordSegment]
    chars: list[SingleCharSegment] | None = None


@dataclass(slots=True)
class AlignedSegment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f'{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})'

    @property
    def length(self):
        return self.end - self.start


@dataclass(slots=True)
class AlignedTranscriptionResult:
    """
    A list of segments and word segments of a speech.
    """

    segments: list[SingleAlignedSegment]
    word_segments: list[SingleWordSegment]


@dataclass(slots=True)
class PreprocessResult:
    clean_char: list[str]
    clean_cdx: list[int]
    clean_wdx: list[int]
    sentence_spans: list[tuple[int, int]]
