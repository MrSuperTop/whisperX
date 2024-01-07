from typing import Callable, Optional

import networkx as nx
import numpy as np
import torch
from pyannote.core import SlidingWindowFeature as SlidingWindowFeature

def permutate(y1, y2, cost_func: Optional[Callable] = ..., return_cost: bool = ...): ...
def mse_cost_func(Y, y, **kwargs): ...
def mae_cost_func(Y, y, **kwargs): ...
def permutate_torch(
    y1: torch.Tensor,
    y2: torch.Tensor,
    cost_func: Optional[Callable] = ...,
    return_cost: bool = ...,
) -> tuple[torch.Tensor, list[tuple[int]]]: ...
def permutate_numpy(
    y1: np.ndarray,
    y2: np.ndarray,
    cost_func: Optional[Callable] = ...,
    return_cost: bool = ...,
) -> tuple[np.ndarray, list[tuple[int]]]: ...
def build_permutation_graph(
    segmentations: SlidingWindowFeature,
    onset: float = ...,
    cost_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = ...,
) -> nx.Graph: ...
