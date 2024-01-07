from typing import Optional, Union

import numpy as np
from _typeshed import Incomplete
from pyannote.core import Annotation, SlidingWindowFeature

def binarize(
    scores,
    onset: float = ...,
    offset: Optional[float] = ...,
    initial_state: Optional[Union[bool, np.ndarray]] = ...,
): ...
def binarize_ndarray(
    scores: np.ndarray,
    onset: float = ...,
    offset: Optional[float] = ...,
    initial_state: Optional[Union[bool, np.ndarray]] = ...,
): ...
def binarize_swf(
    scores: SlidingWindowFeature,
    onset: float = ...,
    offset: Optional[float] = ...,
    initial_state: Optional[bool] = ...,
): ...

class Binarize:
    onset: Incomplete
    offset: Incomplete
    pad_onset: Incomplete
    pad_offset: Incomplete
    min_duration_on: Incomplete
    min_duration_off: Incomplete
    def __init__(
        self,
        onset: float = ...,
        offset: Optional[float] = ...,
        min_duration_on: float = ...,
        min_duration_off: float = ...,
        pad_onset: float = ...,
        pad_offset: float = ...,
    ) -> None: ...
    def __call__(self, scores: SlidingWindowFeature) -> Annotation: ...

class Peak:
    alpha: Incomplete
    min_duration: Incomplete
    def __init__(self, alpha: float = ..., min_duration: float = ...) -> None: ...
    def __call__(self, scores: SlidingWindowFeature): ...
