from collections.abc import Generator
from typing import Mapping, Union

from _typeshed import Incomplete
from pyannote.core import (
    Annotation as Annotation,
)
from pyannote.core import (
    SlidingWindow as SlidingWindow,
)
from pyannote.core import (
    SlidingWindowFeature,
)
from pyannote.core.utils.types import Label as Label

class SpeakerDiarizationMixin:
    @staticmethod
    def set_num_speakers(
        num_speakers: int = ..., min_speakers: int = ..., max_speakers: int = ...
    ): ...
    @staticmethod
    def optimal_mapping(
        reference: Union[Mapping, Annotation],
        hypothesis: Annotation,
        return_mapping: bool = ...,
    ) -> Union[Annotation, tuple[Annotation, dict[Label, Label]]]: ...
    @staticmethod
    def speaker_count(
        segmentations: SlidingWindowFeature,
        onset: float = ...,
        offset: float = ...,
        warm_up: tuple[float, float] = ...,
        frames: SlidingWindow = ...,
    ) -> SlidingWindowFeature: ...
    @staticmethod
    def to_annotation(
        discrete_diarization: SlidingWindowFeature,
        min_duration_on: float = ...,
        min_duration_off: float = ...,
    ) -> Annotation: ...
    @staticmethod
    def to_diarization(
        segmentations: SlidingWindowFeature, count: SlidingWindowFeature
    ) -> SlidingWindowFeature: ...
    def classes(self) -> Generator[Incomplete, None, None]: ...
