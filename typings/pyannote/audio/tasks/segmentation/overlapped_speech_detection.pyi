from _typeshed import Incomplete
from pyannote.audio.core.task import Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.database import Protocol as Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform as BaseWaveformTransform
from torchmetrics import Metric as Metric
from typing import Dict, Sequence, Text, Tuple, Union

class OverlappedSpeechDetection(SegmentationTaskMixin, Task):
    OVERLAP_DEFAULTS: Incomplete
    specifications: Incomplete
    overlap: Incomplete
    balance: Incomplete
    weight: Incomplete
    def __init__(self, protocol: Protocol, duration: float = ..., warm_up: Union[float, Tuple[float, float]] = ..., overlap: dict = ..., balance: Sequence[Text] = ..., weight: Text = ..., batch_size: int = ..., num_workers: int = ..., pin_memory: bool = ..., augmentation: BaseWaveformTransform = ..., metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = ...) -> None: ...
    def prepare_chunk(self, file_id: int, start_time: float, duration: float): ...
