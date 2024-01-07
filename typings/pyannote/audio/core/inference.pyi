from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from _typeshed import Incomplete
from pyannote.audio.core.io import AudioFile as AudioFile
from pyannote.audio.core.model import Model
from pyannote.audio.pipelines.voice_activity_detection import Hook
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature

class BaseInference: ...

class Inference(BaseInference):
    model: Incomplete
    device: Incomplete
    window: Incomplete
    duration: Incomplete
    skip_conversion: Incomplete
    conversion: Incomplete
    skip_aggregation: Incomplete
    pre_aggregation_hook: Incomplete
    warm_up: Incomplete
    step: Incomplete
    batch_size: Incomplete
    def __init__(
        self,
        model: Union[Model, str, Path],
        window: str = ...,
        duration: float = ...,
        step: float = ...,
        pre_aggregation_hook: Callable[[np.ndarray], np.ndarray] = ...,
        skip_aggregation: bool = ...,
        skip_conversion: bool = ...,
        device: torch.device = ...,
        batch_size: int = ...,
        use_auth_token: Union[str, None] = ...,
    ) -> None: ...
    def to(self, device: torch.device) -> Inference: ...
    def infer(self, chunks: torch.Tensor) -> Union[np.ndarray, tuple[np.ndarray]]: ...
    def slide(
        self, waveform: torch.Tensor, sample_rate: int, hook: Optional[Callable]
    ) -> Union[SlidingWindowFeature, tuple[SlidingWindowFeature]]: ...
    def __call__(
        self, file: AudioFile, hook: Hook = None
    ) -> Union[
        tuple[Union[SlidingWindowFeature, np.ndarray[Any, Any]]],
        Union[SlidingWindowFeature, np.ndarray[Any, Any]],
    ]: ...
    def crop(
        self,
        file: AudioFile,
        chunk: Union[Segment, list[Segment]],
        duration: Optional[float] = ...,
        hook: Optional[Callable] = ...,
    ) -> Union[
        tuple[Union[SlidingWindowFeature, np.ndarray]],
        Union[SlidingWindowFeature, np.ndarray],
    ]: ...
    @staticmethod
    def aggregate(
        scores: SlidingWindowFeature,
        frames: SlidingWindow = ...,
        warm_up: tuple[float, float] = ...,
        epsilon: float = ...,
        hamming: bool = ...,
        missing: float = ...,
        skip_average: bool = ...,
    ) -> SlidingWindowFeature: ...
    @staticmethod
    def trim(
        scores: SlidingWindowFeature, warm_up: tuple[float, float] = ...
    ) -> SlidingWindowFeature: ...
    @staticmethod
    def stitch(
        activations: SlidingWindowFeature,
        frames: SlidingWindow = ...,
        lookahead: Optional[tuple[int, int]] = ...,
        cost_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ...,
        match_func: Callable[[np.ndarray, np.ndarray, float], bool] = ...,
    ) -> SlidingWindowFeature: ...
