import torch
import torch.nn as nn
from _typeshed import Incomplete
from pyannote.audio.core.model import Model
from torch_audiomentations.core.transforms_interface import (
    BaseWaveformTransform as BaseWaveformTransform,
)

def register_augmentation(
    augmentation: nn.Module, module: nn.Module, when: str = ...
): ...
def unregister_augmentation(module: nn.Module, when: str = ...): ...
def wrap_augmentation(augmentation, model: Model, when: str = ...): ...

class TorchAudiomentationsWaveformTransformWrapper(nn.Module):
    augmentation: Incomplete
    sample_rate_: Incomplete
    def __init__(
        self, augmentation: BaseWaveformTransform, model: Model, when: str = ...
    ) -> None: ...
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor: ...
