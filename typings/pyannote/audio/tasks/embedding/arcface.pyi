from typing import Sequence, Union

from _typeshed import Incomplete
from pyannote.audio.core.task import Task
from pyannote.database import Protocol as Protocol
from torch_audiomentations.core.transforms_interface import (
    BaseWaveformTransform as BaseWaveformTransform,
)
from torchmetrics import Metric as Metric

from .mixins import (
    SupervisedRepresentationLearningTaskMixin as SupervisedRepresentationLearningTaskMixin,
)

class SupervisedRepresentationLearningWithArcFace(
    SupervisedRepresentationLearningTaskMixin, Task
):
    num_chunks_per_class: Incomplete
    num_classes_per_batch: Incomplete
    margin: Incomplete
    scale: Incomplete
    def __init__(
        self,
        protocol: Protocol,
        min_duration: float = ...,
        duration: float = ...,
        num_classes_per_batch: int = ...,
        num_chunks_per_class: int = ...,
        margin: float = ...,
        scale: float = ...,
        num_workers: int = ...,
        pin_memory: bool = ...,
        augmentation: BaseWaveformTransform = ...,
        metric: Union[Metric, Sequence[Metric], dict[str, Metric]] = ...,
    ) -> None: ...
    def setup_loss_func(self) -> None: ...
