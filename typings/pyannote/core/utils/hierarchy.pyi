from typing import Callable, Union

import numpy as np

from .distance import (
    cdist as cdist,
)
from .distance import (
    l2_normalize as l2_normalize,
)
from .distance import (
    pdist as pdist,
)
from .distance import (
    to_condensed as to_condensed,
)
from .distance import (
    to_squared as to_squared,
)

def linkage(X, method: str = ..., metric: str = ..., **kwargs): ...
def propagate_constraints(
    cannot_link: list[tuple[int, int]], must_link: list[tuple[int, int]]
): ...
def pool(
    X: np.ndarray,
    metric: str = ...,
    pooling_func: Union[str, Callable] = ...,
    cannot_link: list[tuple[int, int]] = ...,
    must_link: list[tuple[int, int]] = ...,
    must_link_method: str = ...,
): ...
def fcluster_auto(X, Z, metric: str = ...): ...
