from collections.abc import Generator
from typing import Callable, Optional, Union

import numpy as np
from _typeshed import Incomplete
from pyannote.audio import Model as Model
from pyannote.audio import Pipeline
from pyannote.audio.core.io import AudioFile as AudioFile
from pyannote.audio.pipelines.utils import (
    PipelineModel as PipelineModel,
)
from pyannote.audio.pipelines.utils import (
    SpeakerDiarizationMixin,
)
from pyannote.core import (
    Annotation,
    SlidingWindowFeature,
)
from pyannote.core import (
    SlidingWindow as SlidingWindow,
)
from pyannote.metrics.diarization import GreedyDiarizationErrorRate

def batchify(iterable, batch_size: int = ..., fillvalue: Incomplete | None = ...): ...

class SpeakerDiarization(SpeakerDiarizationMixin, Pipeline):
    segmentation_model: Incomplete
    segmentation_step: Incomplete
    embedding: Incomplete
    embedding_batch_size: Incomplete
    embedding_exclude_overlap: Incomplete
    klustering: Incomplete
    der_variant: Incomplete
    segmentation: Incomplete
    clustering: Incomplete
    def __init__(
        self,
        segmentation: PipelineModel = ...,
        segmentation_step: float = ...,
        embedding: PipelineModel = ...,
        embedding_exclude_overlap: bool = ...,
        clustering: str = ...,
        embedding_batch_size: int = ...,
        segmentation_batch_size: int = ...,
        der_variant: dict = ...,
        use_auth_token: Union[str, None] = ...,
    ) -> None: ...
    @property
    def segmentation_batch_size(self) -> int: ...
    def default_parameters(self) -> None: ...
    def classes(self) -> Generator[Incomplete, None, None]: ...
    @property
    def CACHED_SEGMENTATION(self): ...
    def get_segmentations(
        self, file, hook: Incomplete | None = ...
    ) -> SlidingWindowFeature: ...
    def get_embeddings(
        self,
        file,
        binary_segmentations: SlidingWindowFeature,
        exclude_overlap: bool = ...,
        hook: Optional[Callable] = ...,
    ): ...
    def reconstruct(
        self,
        segmentations: SlidingWindowFeature,
        hard_clusters: np.ndarray,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature: ...
    def apply(
        self,
        file: AudioFile,
        num_speakers: int = ...,
        min_speakers: int = ...,
        max_speakers: int = ...,
        return_embeddings: bool = ...,
        hook: Optional[Callable] = ...,
    ) -> Annotation: ...
    def get_metric(self) -> GreedyDiarizationErrorRate: ...
