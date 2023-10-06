import torch
from _typeshed import Incomplete
from torchmetrics import Metric
from typing import Optional

class EqualErrorRate(Metric):
    is_differentiable: Optional[bool]
    higher_is_better: Optional[bool]
    full_state_update: bool
    distances: Incomplete
    def __init__(self, distances: bool = ..., compute_on_cpu: bool = ..., **kwargs) -> None: ...
    def update(self, scores: torch.Tensor, y_true: torch.Tensor) -> None: ...
    def compute(self) -> torch.Tensor: ...
