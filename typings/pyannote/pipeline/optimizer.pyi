from pathlib import Path
from typing import Callable, Generator, Iterable, Optional, Union

from _typeshed import Incomplete
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from optuna.trial import Trial as Trial

from .pipeline import Pipeline as Pipeline
from .typing import PipelineInput as PipelineInput

class Optimizer:
    pipeline: Incomplete
    db: Incomplete
    storage_: Incomplete
    study_name: Incomplete
    sampler: Incomplete
    pruner: Incomplete
    study_: Incomplete
    average_case: Incomplete
    def __init__(
        self,
        pipeline: Pipeline,
        db: Optional[Path] = ...,
        study_name: Optional[str] = ...,
        sampler: Optional[Union[str, BaseSampler]] = ...,
        pruner: Optional[Union[str, BasePruner]] = ...,
        seed: Optional[int] = ...,
        average_case: bool = ...,
    ) -> None: ...
    @property
    def best_loss(self) -> float: ...
    @property
    def best_params(self) -> dict: ...
    @property
    def best_pipeline(self) -> Pipeline: ...
    def get_objective(
        self, inputs: Iterable[PipelineInput], show_progress: Union[bool, dict] = ...
    ) -> Callable[[Trial], float]: ...
    def tune(
        self,
        inputs: Iterable[PipelineInput],
        n_iterations: int = ...,
        warm_start: dict = ...,
        show_progress: Union[bool, dict] = ...,
    ) -> dict: ...
    def tune_iter(
        self,
        inputs: Iterable[PipelineInput],
        warm_start: dict = ...,
        show_progress: Union[bool, dict] = ...,
    ) -> Generator[dict, None, None]: ...
