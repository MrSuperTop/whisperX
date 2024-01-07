from typing import Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch_audiomentations import Mix

class MixSpeakerDiarization(Mix):
    supported_modes: Incomplete
    supports_multichannel: bool
    requires_sample_rate: bool
    supports_target: bool
    requires_target: bool
    max_num_speakers: Incomplete
    def __init__(
        self,
        min_snr_in_db: float = ...,
        max_snr_in_db: float = ...,
        mode: str = ...,
        p: float = ...,
        p_mode: str = ...,
        sample_rate: int = ...,
        target_rate: int = ...,
        max_num_speakers: int = ...,
        output_type: str = ...,
    ) -> None: ...
    def randomize_parameters(
        self,
        samples: Tensor = ...,
        sample_rate: Optional[int] = ...,
        targets: Optional[Tensor] = ...,
        target_rate: Optional[int] = ...,
    ): ...
