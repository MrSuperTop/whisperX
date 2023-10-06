from ..parameter import Uniform as Uniform
from ..pipeline import Pipeline as Pipeline
from _typeshed import Incomplete
from typing import Optional

class ClosestAssignment(Pipeline):
    metric: Incomplete
    normalize: Incomplete
    threshold: Incomplete
    def __init__(self, metric: Optional[str] = ..., normalize: Optional[bool] = ...) -> None: ...
    def __call__(self, X_target, X): ...
