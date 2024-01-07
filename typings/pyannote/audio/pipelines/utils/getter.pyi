from typing import Any, Mapping

import torch
from pyannote.audio import Inference, Model
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

PipelineModel = Model | str | Mapping[Any, Any]

def get_model(model: PipelineModel, use_auth_token: str | None = ...) -> Model: ...

PipelineInference = Inference | Model | str | Mapping[Any, Any]

def get_inference(inference: PipelineInference) -> Inference: ...

PipelineAugmentation = BaseWaveformTransform | Mapping[Any, Any]

def get_augmentation(augmentation: PipelineAugmentation) -> BaseWaveformTransform: ...
def get_devices(needs: int | None = None) -> list[torch.device]: ...
