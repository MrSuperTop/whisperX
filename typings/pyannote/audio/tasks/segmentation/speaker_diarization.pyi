from typing import Literal, Sequence, Union

import torch.nn.functional
from _typeshed import Incomplete
from pyannote.audio.core.task import Task
from pyannote.audio.tasks.segmentation.mixins import SegmentationTaskMixin
from pyannote.database.protocol import SpeakerDiarizationProtocol
from torch_audiomentations.core.transforms_interface import (
    BaseWaveformTransform as BaseWaveformTransform,
)
from torchmetrics import Metric as Metric

Subsets: Incomplete
Scopes: Incomplete

class SpeakerDiarization(SegmentationTaskMixin, Task):
    max_speakers_per_chunk: Incomplete
    max_speakers_per_frame: Incomplete
    weigh_by_cardinality: Incomplete
    balance: Incomplete
    weight: Incomplete
    vad_loss: Incomplete
    def __init__(
        self,
        protocol: SpeakerDiarizationProtocol,
        duration: float = ...,
        max_speakers_per_chunk: int = ...,
        max_speakers_per_frame: int = ...,
        weigh_by_cardinality: bool = ...,
        warm_up: Union[float, tuple[float, float]] = ...,
        balance: Sequence[str] = ...,
        weight: str = ...,
        batch_size: int = ...,
        num_workers: int = ...,
        pin_memory: bool = ...,
        augmentation: BaseWaveformTransform = ...,
        vad_loss: Literal['bce', 'mse'] = ...,
        metric: Union[Metric, Sequence[Metric], dict[str, Metric]] = ...,
        max_num_speakers: int = ...,
        loss: Literal['bce', 'mse'] = ...,
    ) -> None: ...
    specifications: Incomplete
    def setup(self) -> None: ...
    def setup_loss_func(self) -> None: ...
    def prepare_chunk(self, file_id: int, start_time: float, duration: float): ...
    def collate_y(self, batch) -> torch.Tensor: ...
    def segmentation_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = ...,
    ) -> torch.Tensor: ...
    def voice_activity_detection_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = ...,
    ) -> torch.Tensor: ...
    def training_step(self, batch, batch_idx: int): ...
    def default_metric(self) -> Union[Metric, Sequence[Metric], dict[str, Metric]]: ...
    def validation_step(self, batch, batch_idx: int): ...

def main(protocol: str, subset: str = ..., model: str = ...): ...
