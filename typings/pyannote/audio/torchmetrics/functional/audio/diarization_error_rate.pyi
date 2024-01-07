from typing import Optional, Union

import torch

def diarization_error_rate(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: Union[torch.Tensor, float] = ...,
) -> torch.Tensor: ...
def optimal_diarization_error_rate(
    preds: torch.Tensor, target: torch.Tensor, threshold: Optional[torch.Tensor] = ...
) -> torch.Tensor: ...
