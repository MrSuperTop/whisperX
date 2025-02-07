from functools import cached_property as cached_property
from typing import Union

import numpy as np
import torch
from _typeshed import Incomplete
from pyannote.audio import Model as Model
from pyannote.audio import Pipeline
from pyannote.audio.core.inference import BaseInference
from pyannote.audio.core.io import AudioFile as AudioFile
from pyannote.audio.pipelines.utils import PipelineModel as PipelineModel

backend: Incomplete
SPEECHBRAIN_IS_AVAILABLE: bool
NEMO_IS_AVAILABLE: bool
ONNX_IS_AVAILABLE: bool

class NeMoPretrainedSpeakerEmbedding(BaseInference):
    embedding: Incomplete
    device: Incomplete
    model_: Incomplete
    def __init__(self, embedding: str = ..., device: torch.device = ...) -> None: ...
    def to(self, device: torch.device): ...
    @cached_property
    def sample_rate(self) -> int: ...
    @cached_property
    def dimension(self) -> int: ...
    @cached_property
    def metric(self) -> str: ...
    @cached_property
    def min_num_samples(self) -> int: ...
    def __call__(
        self, waveforms: torch.Tensor, masks: torch.Tensor = ...
    ) -> np.ndarray: ...

class SpeechBrainPretrainedSpeakerEmbedding(BaseInference):
    embedding: Incomplete
    revision: Incomplete
    device: Incomplete
    use_auth_token: Incomplete
    classifier_: Incomplete
    def __init__(
        self,
        embedding: str = ...,
        device: torch.device = ...,
        use_auth_token: Union[str, None] = ...,
    ) -> None: ...
    def to(self, device: torch.device): ...
    @cached_property
    def sample_rate(self) -> int: ...
    @cached_property
    def dimension(self) -> int: ...
    @cached_property
    def metric(self) -> str: ...
    @cached_property
    def min_num_samples(self) -> int: ...
    def __call__(
        self, waveforms: torch.Tensor, masks: torch.Tensor = ...
    ) -> np.ndarray: ...

class WeSpeakerPretrainedSpeakerEmbedding(BaseInference):
    embedding: Incomplete
    def __init__(self, embedding: str = ..., device: torch.device = ...) -> None: ...
    session_: Incomplete
    device: Incomplete
    def to(self, device: torch.device): ...
    @cached_property
    def sample_rate(self) -> int: ...
    @cached_property
    def dimension(self) -> int: ...
    @cached_property
    def metric(self) -> str: ...
    @cached_property
    def min_num_samples(self) -> int: ...
    @cached_property
    def min_num_frames(self) -> int: ...
    def compute_fbank(
        self,
        waveforms: torch.Tensor,
        num_mel_bins: int = ...,
        frame_length: int = ...,
        frame_shift: int = ...,
        dither: float = ...,
    ) -> torch.Tensor: ...
    def __call__(
        self, waveforms: torch.Tensor, masks: torch.Tensor = ...
    ) -> np.ndarray: ...

class PyannoteAudioPretrainedSpeakerEmbedding(BaseInference):
    embedding: Incomplete
    device: Incomplete
    model_: Incomplete
    def __init__(
        self,
        embedding: PipelineModel = ...,
        device: torch.device = ...,
        use_auth_token: Union[str, None] = ...,
    ) -> None: ...
    def to(self, device: torch.device): ...
    @cached_property
    def sample_rate(self) -> int: ...
    @cached_property
    def dimension(self) -> int: ...
    @cached_property
    def metric(self) -> str: ...
    @cached_property
    def min_num_samples(self) -> int: ...
    def __call__(
        self, waveforms: torch.Tensor, masks: torch.Tensor = ...
    ) -> np.ndarray: ...

def PretrainedSpeakerEmbedding(
    embedding: PipelineModel,
    device: torch.device = ...,
    use_auth_token: Union[str, None] = ...,
): ...

class SpeakerEmbedding(Pipeline):
    embedding: Incomplete
    segmentation: Incomplete
    embedding_model_: Incomplete
    def __init__(
        self,
        embedding: PipelineModel = ...,
        segmentation: PipelineModel = ...,
        use_auth_token: Union[str, None] = ...,
    ) -> None: ...
    def apply(self, file: AudioFile) -> np.ndarray: ...

def main(
    protocol: str = ...,
    subset: str = ...,
    embedding: str = ...,
    segmentation: str = ...,
): ...
