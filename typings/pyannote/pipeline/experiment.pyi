from .optimizer import Optimizer as Optimizer
from _typeshed import Incomplete
from pathlib import Path
from typing import Optional

class Experiment:
    CONFIG_YML: str
    TRAIN_DIR: str
    APPLY_DIR: str
    @classmethod
    def from_train_dir(cls, train_dir: Path, training: bool = ...) -> Experiment: ...
    experiment_dir: Incomplete
    config_: Incomplete
    preprocessors_: Incomplete
    pipeline_: Incomplete
    def __init__(self, experiment_dir: Path, training: bool = ...) -> None: ...
    def train(self, protocol_name: str, subset: Optional[str] = ..., pretrained: Optional[Path] = ..., n_iterations: int = ..., sampler: Optional[str] = ..., pruner: Optional[str] = ..., average_case: bool = ...): ...
    def best(self, protocol_name: str, subset: str = ...): ...
    def apply(self, protocol_name: str, output_dir: Path, subset: Optional[str] = ...): ...

def main() -> None: ...
