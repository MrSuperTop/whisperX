from torch.optim import Optimizer as Optimizer

def CosineAnnealingWarmRestarts(
    optimizer: Optimizer,
    min_lr: float = ...,
    max_lr: float = ...,
    patience: int = ...,
    num_batches_per_epoch: int = ...,
    **kwargs,
): ...
