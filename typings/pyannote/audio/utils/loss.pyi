import torch

def interpolate(target: torch.Tensor, weight: torch.Tensor = ...): ...
def binary_cross_entropy(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = ...
) -> torch.Tensor: ...
def mse_loss(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = ...
) -> torch.Tensor: ...
def nll_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_weight: torch.Tensor = ...,
    weight: torch.Tensor = ...,
) -> torch.Tensor: ...
