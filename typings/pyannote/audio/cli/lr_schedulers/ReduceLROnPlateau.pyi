from typing import Optional

from torch.optim import Optimizer as Optimizer

def ReduceLROnPlateau(
    optimizer: Optimizer,
    monitor: Optional[str] = ...,
    direction: Optional[str] = ...,
    min_lr: float = ...,
    max_lr: float = ...,
    factor: float = ...,
    patience: int = ...,
    **kwargs,
): ...
