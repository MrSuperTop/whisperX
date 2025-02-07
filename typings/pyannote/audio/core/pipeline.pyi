from collections.abc import Iterator
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from _typeshed import Incomplete
from pyannote.audio.core.io import AudioFile as AudioFile
from pyannote.pipeline import Pipeline as _Pipeline

PIPELINE_PARAMS_NAME: str

class Pipeline(_Pipeline):
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        hparams_file: Union[str, Path] = ...,
        use_auth_token: Union[str, None] = ...,
        cache_dir: Union[Path, str] = ...,
    ) -> Pipeline: ...
    def __init__(self) -> None: ...
    def __getattr__(self, name): ...
    def __setattr__(self, name, value) -> None: ...
    def __delattr__(self, name) -> None: ...
    @staticmethod
    def setup_hook(file: AudioFile, hook: Optional[Callable] = ...) -> Callable: ...
    def default_parameters(self) -> None: ...
    def classes(self) -> Union[list, Iterator]: ...
    def __call__(self, file: AudioFile, **kwargs): ...
    device: Incomplete
    def to(self, device: torch.device): ...
