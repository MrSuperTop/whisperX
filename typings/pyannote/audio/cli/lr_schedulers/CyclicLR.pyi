from torch.optim import Optimizer as Optimizer

def CyclicLR(
    optimizer: Optimizer,
    min_lr: float = ...,
    max_lr: float = ...,
    mode: str = ...,
    patience: int = ...,
    num_batches_per_epoch: int = ...,
    **kwargs,
): ...
