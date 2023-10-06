from _typeshed import Incomplete
from pyannote.audio import Model as Model
from pyannote.audio.core.io import AudioFile as AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import PipelineModel as PipelineModel, SpeakerDiarizationMixin
from pyannote.core import Annotation as Annotation
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from typing import Callable, Optional, Text, Union

class Resegmentation(SpeakerDiarizationMixin, Pipeline):
    segmentation: Incomplete
    diarization: Incomplete
    der_variant: Incomplete
    warm_up: Incomplete
    onset: Incomplete
    offset: Incomplete
    min_duration_on: Incomplete
    min_duration_off: Incomplete
    def __init__(self, segmentation: PipelineModel = ..., diarization: Text = ..., der_variant: dict = ..., use_auth_token: Union[Text, None] = ...) -> None: ...
    def default_parameters(self): ...
    def classes(self) -> None: ...
    CACHED_SEGMENTATION: str
    def apply(self, file: AudioFile, diarization: Annotation = ..., hook: Optional[Callable] = ...) -> Annotation: ...
    def get_metric(self) -> GreedyDiarizationErrorRate: ...
