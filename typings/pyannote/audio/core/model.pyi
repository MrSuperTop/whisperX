from functools import cached_property as cached_property
from pathlib import Path
from typing import Any, Optional, Union

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim
from _typeshed import Incomplete, StrPath
from pyannote.audio.core.task import Specifications
from pyannote.audio.core.task import Task as Task
from pyannote.core import SlidingWindow
from torch.utils.data import DataLoader as DataLoader

CACHE_DIR: Incomplete
HF_PYTORCH_WEIGHTS_NAME: str
HF_LIGHTNING_CONFIG_NAME: str

class Introspection: ...

class Output:
    num_frames: int
    dimension: int
    frames: SlidingWindow
    def __init__(self, num_frames, dimension, frames) -> None: ...

class Model(pl.LightningModule):
    audio: Incomplete
    def __init__(
        self,
        sample_rate: int = ...,
        num_channels: int = ...,
        task: Optional[Task] = ...,
    ) -> None: ...
    @property
    def task(self) -> Task: ...
    def build(self) -> None: ...
    @property
    def specifications(self) -> Union[Specifications, tuple[Specifications]]: ...
    @property
    def example_input_array(self) -> torch.Tensor: ...
    @cached_property
    def example_output(self) -> Union[Output, tuple[Output]]: ...
    task_dependent: Incomplete
    def setup(self, stage: Incomplete | None = ...) -> None: ...
    def on_save_checkpoint(self, checkpoint) -> None: ...
    def on_load_checkpoint(self, checkpoint: dict[str, Any]): ...
    def forward(
        self, waveforms: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, tuple[torch.Tensor]]: ...
    def default_activation(self) -> Union[nn.Module, tuple[nn.Module]]: ...
    def train_dataloader(self) -> DataLoader: ...
    def training_step(self, batch, batch_idx): ...
    def val_dataloader(self) -> DataLoader: ...
    def validation_step(self, batch, batch_idx): ...
    def configure_optimizers(self): ...
    def freeze_up_to(self, module_name: str) -> list[str]: ...
    def unfreeze_up_to(self, module_name: str) -> list[str]: ...
    def freeze_by_name(
        self, modules: Union[str, list[str]], recurse: bool = ...
    ) -> list[str]: ...
    def unfreeze_by_name(
        self, modules: Union[list[str], str], recurse: bool = ...
    ) -> list[str]: ...
    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Union[Path, str],
        map_location: StrPath | None = ...,
        hparams_file: StrPath = ...,
        strict: bool = ...,
        use_auth_token: Union[str, None] = ...,
        cache_dir: Union[Path, str] = ...,
        **kwargs: Any,
    ) -> Model: ...
