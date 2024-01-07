from io import IOBase
from typing import Any, Mapping, MutableMapping, Optional

from _typeshed import Incomplete, StrPath
from pyannote.core import Segment as Segment
from pyannote.database import ProtocolFile
from torch import Tensor as Tensor

AudioFile = StrPath | IOBase | ProtocolFile | MutableMapping[Any, Any]
AudioFileDocString: str

def get_torchaudio_info(file: AudioFile): ...

class Audio:
    PRECISION: float
    @staticmethod
    def power_normalize(waveform: Tensor) -> Tensor: ...
    @staticmethod
    def validate_file(file: AudioFile) -> Mapping: ...
    sample_rate: Incomplete
    mono: Incomplete
    def __init__(
        self, sample_rate: Incomplete | None = ..., mono: Incomplete | None = ...
    ) -> None: ...
    def downmix_and_resample(self, waveform: Tensor, sample_rate: int) -> Tensor: ...
    def get_duration(self, file: AudioFile) -> float: ...
    def get_num_samples(self, duration: float, sample_rate: int = ...) -> int: ...
    def __call__(self, file: AudioFile) -> tuple[Tensor, int]: ...
    def crop(
        self,
        file: AudioFile,
        segment: Segment,
        duration: Optional[float] = ...,
        mode: str = ...,
    ) -> tuple[Tensor, int]: ...
