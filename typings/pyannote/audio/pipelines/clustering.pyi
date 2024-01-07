from enum import Enum

import numpy as np
from _typeshed import Incomplete
from pyannote.audio.core.io import AudioFile as AudioFile
from pyannote.core import (
    SlidingWindow as SlidingWindow,
)
from pyannote.core import (
    SlidingWindowFeature as SlidingWindowFeature,
)
from pyannote.pipeline import Pipeline

class BaseClustering(Pipeline):
    metric: Incomplete
    max_num_embeddings: Incomplete
    constrained_assignment: Incomplete
    def __init__(
        self,
        metric: str = ...,
        max_num_embeddings: int = ...,
        constrained_assignment: bool = ...,
    ) -> None: ...
    def set_num_clusters(
        self,
        num_embeddings: int,
        num_clusters: int = ...,
        min_clusters: int = ...,
        max_clusters: int = ...,
    ): ...
    def filter_embeddings(
        self, embeddings: np.ndarray, segmentations: SlidingWindowFeature = ...
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def constrained_argmax(self, soft_clusters: np.ndarray) -> np.ndarray: ...
    def assign_embeddings(
        self,
        embeddings: np.ndarray,
        train_chunk_idx: np.ndarray,
        train_speaker_idx: np.ndarray,
        train_clusters: np.ndarray,
        constrained: bool = ...,
    ): ...
    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature = ...,
        num_clusters: int = ...,
        min_clusters: int = ...,
        max_clusters: int = ...,
        **kwargs,
    ) -> np.ndarray: ...

class AgglomerativeClustering(BaseClustering):
    threshold: Incomplete
    method: Incomplete
    min_cluster_size: Incomplete
    def __init__(
        self,
        metric: str = ...,
        max_num_embeddings: int = ...,
        constrained_assignment: bool = ...,
    ) -> None: ...
    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: int = ...,
    ): ...

class OracleClustering(BaseClustering):
    def __call__(
        self,
        embeddings: np.ndarray = ...,
        segmentations: SlidingWindowFeature = ...,
        file: AudioFile = ...,
        frames: SlidingWindow = ...,
        **kwargs,
    ) -> np.ndarray: ...

class Clustering(Enum):
    AgglomerativeClustering = AgglomerativeClustering
    OracleClustering = OracleClustering
