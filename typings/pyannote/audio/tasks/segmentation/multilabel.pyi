from _typeshed import Incomplete
from pyannote.audio.core.task import Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.database import Protocol as Protocol
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform as BaseWaveformTransform
from torchmetrics import Metric as Metric
from typing import Dict, List, Optional, Sequence, Text, Tuple, Union

class MultiLabelSegmentation(SegmentationTaskMixin, Task):
    balance: Incomplete
    weight: Incomplete
    classes: Incomplete
    def __init__(self, protocol: Protocol, classes: Optional[List[str]] = ..., duration: float = ..., warm_up: Union[float, Tuple[float, float]] = ..., balance: Sequence[Text] = ..., weight: Text = ..., batch_size: int = ..., num_workers: int = ..., pin_memory: bool = ..., augmentation: BaseWaveformTransform = ..., metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = ...) -> None: ...
    specifications: Incomplete
    def setup(self) -> None: ...
    def prepare_chunk(self, file_id: int, start_time: float, duration: float): ...
    def training_step(self, batch, batch_idx: int): ...
    def validation_step(self, batch, batch_idx: int): ...
    @property
    def val_monitor(self): ...
