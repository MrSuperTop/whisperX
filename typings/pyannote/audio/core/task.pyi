from enum import Enum
from functools import cached_property as cached_property
from typing import Literal, Optional, Sequence, Union

import pytorch_lightning as pl
import torch
from _typeshed import Incomplete
from pyannote.database import Protocol as Protocol
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch_audiomentations.core.transforms_interface import (
    BaseWaveformTransform as BaseWaveformTransform,
)
from torchmetrics import Metric as Metric
from torchmetrics import MetricCollection

class Problem(Enum):
    BINARY_CLASSIFICATION: int
    MONO_LABEL_CLASSIFICATION: int
    MULTI_LABEL_CLASSIFICATION: int
    REPRESENTATION: int
    REGRESSION: int

class Resolution(Enum):
    FRAME: int
    CHUNK: int

class UnknownSpecificationsError(Exception): ...

class Specifications:
    problem: Problem
    resolution: Resolution
    duration: float
    min_duration: Optional[float]
    warm_up: Optional[tuple[float, float]]
    classes: Optional[list[str]]
    powerset_max_classes: Optional[int]
    permutation_invariant: bool
    @cached_property
    def powerset(self) -> bool: ...
    @cached_property
    def num_powerset_classes(self) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def __init__(
        self,
        problem,
        resolution,
        duration,
        min_duration,
        warm_up,
        classes,
        powerset_max_classes,
        permutation_invariant,
    ) -> None: ...

class TrainDataset(IterableDataset):
    task: Incomplete
    def __init__(self, task: Task) -> None: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...

class ValDataset(Dataset):
    task: Incomplete
    def __init__(self, task: Task) -> None: ...
    def __getitem__(self, idx): ...
    def __len__(self) -> int: ...

class Task(pl.LightningDataModule):
    has_validation: Incomplete
    has_scope: Incomplete
    has_classes: Incomplete
    duration: Incomplete
    min_duration: Incomplete
    batch_size: Incomplete
    warm_up: Incomplete
    num_workers: Incomplete
    pin_memory: Incomplete
    augmentation: Incomplete
    def __init__(
        self,
        protocol: Protocol,
        duration: float = ...,
        min_duration: float = ...,
        warm_up: Union[float, tuple[float, float]] = ...,
        batch_size: int = ...,
        num_workers: int = ...,
        pin_memory: bool = ...,
        augmentation: BaseWaveformTransform = ...,
        metric: Union[Metric, Sequence[Metric], dict[str, Metric]] = ...,
    ) -> None: ...
    def prepare_data(self) -> None: ...
    @property
    def specifications(self) -> Union[Specifications, tuple[Specifications]]: ...
    @property
    def has_setup_metadata(self): ...
    def setup_metadata(self) -> None: ...
    def setup_loss_func(self) -> None: ...
    def train__iter__(self) -> None: ...
    def train__len__(self) -> None: ...
    def collate_fn(self, batch, stage: str = ...) -> None: ...
    def train_dataloader(self) -> DataLoader: ...
    def default_loss(
        self,
        specifications: Specifications,
        target,
        prediction,
        weight: Incomplete | None = ...,
    ) -> torch.Tensor: ...
    def common_step(self, batch, batch_idx: int, stage: Literal['train', 'val']): ...
    def training_step(self, batch, batch_idx: int): ...
    def val__getitem__(self, idx) -> None: ...
    def val__len__(self) -> None: ...
    def val_dataloader(self) -> Optional[DataLoader]: ...
    def validation_step(self, batch, batch_idx: int): ...
    def default_metric(self) -> Union[Metric, Sequence[Metric], dict[str, Metric]]: ...
    @cached_property
    def metric(self) -> MetricCollection: ...
    def setup_validation_metric(self) -> None: ...
    @property
    def val_monitor(self): ...
