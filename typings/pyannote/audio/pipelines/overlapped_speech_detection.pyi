from typing import Callable, Optional, Union

from _typeshed import Incomplete
from pyannote.audio.core.io import AudioFile as AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import PipelineModel as PipelineModel
from pyannote.core import (
    Annotation as Annotation,
)
from pyannote.core import (
    SlidingWindowFeature as SlidingWindowFeature,
)
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure

def to_overlap(annotation: Annotation) -> Annotation: ...

class OracleOverlappedSpeechDetection(Pipeline):
    def apply(self, file: AudioFile) -> Annotation: ...

class OverlappedSpeechDetection(Pipeline):
    segmentation: Incomplete
    onset: float
    offset: Incomplete
    min_duration_on: Incomplete
    min_duration_off: Incomplete
    precision: Incomplete
    recall: Incomplete
    def __init__(
        self,
        segmentation: PipelineModel = ...,
        precision: Optional[float] = ...,
        recall: Optional[float] = ...,
        use_auth_token: Union[str, None] = ...,
        **inference_kwargs,
    ) -> None: ...
    def default_parameters(self): ...
    def classes(self): ...
    def initialize(self) -> None: ...
    CACHED_SEGMENTATION: str
    def apply(self, file: AudioFile, hook: Optional[Callable] = ...) -> Annotation: ...
    def get_metric(self, **kwargs) -> DetectionPrecisionRecallFMeasure: ...
    def loss(self, file: AudioFile, hypothesis: Annotation) -> float: ...
    def get_direction(self): ...
