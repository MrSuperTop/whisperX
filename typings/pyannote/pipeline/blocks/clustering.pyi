import numpy as np
from ..parameter import Uniform as Uniform
from ..pipeline import Pipeline as Pipeline
from _typeshed import Incomplete
from typing import Optional

class HierarchicalAgglomerativeClustering(Pipeline):
    method: Incomplete
    metric: Incomplete
    normalize: Incomplete
    use_threshold: Incomplete
    threshold: Incomplete
    def __init__(self, method: Optional[str] = ..., metric: Optional[str] = ..., use_threshold: Optional[bool] = ..., normalize: Optional[bool] = ...) -> None: ...
    def __call__(self, X: np.ndarray) -> np.ndarray: ...

class AffinityPropagationClustering(Pipeline):
    metric: Incomplete
    damping: Incomplete
    preference: Incomplete
    def __init__(self, metric: Optional[str] = ...) -> None: ...
    affinity_propagation_: Incomplete
    def initialize(self) -> None: ...
    def __call__(self, X: np.ndarray) -> np.ndarray: ...
