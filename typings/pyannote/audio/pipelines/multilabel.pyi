from ..utils.signal import Binarize as Binarize
from .utils import PipelineModel as PipelineModel, get_model as get_model
from _typeshed import Incomplete
from pyannote.audio.core.io import AudioFile as AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.utils.metric import MacroAverageFMeasure
from pyannote.core import Annotation
from pyannote.metrics.identification import IdentificationErrorRate
from typing import Callable, Optional, Text, Union

class MultiLabelSegmentation(Pipeline):
    segmentation: Incomplete
    fscore: Incomplete
    share_min_duration: Incomplete
    min_duration_on: Incomplete
    min_duration_off: Incomplete
    thresholds: Incomplete
    def __init__(self, segmentation: PipelineModel = ..., fscore: bool = ..., share_min_duration: bool = ..., use_auth_token: Union[Text, None] = ..., **inference_kwargs) -> None: ...
    def classes(self): ...
    def initialize(self) -> None: ...
    CACHED_SEGMENTATION: str
    def apply(self, file: AudioFile, hook: Optional[Callable] = ...) -> Annotation: ...
    def get_metric(self) -> Union[MacroAverageFMeasure, IdentificationErrorRate]: ...
    def get_direction(self): ...
