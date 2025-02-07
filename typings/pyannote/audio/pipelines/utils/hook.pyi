from typing import Any, Mapping, Optional

from _typeshed import Incomplete

def logging_hook(
    step_name: str,
    step_artifact: Any,
    file: Optional[Mapping] = ...,
    completed: Optional[int] = ...,
    total: Optional[int] = ...,
): ...

class ProgressHook:
    transient: Incomplete
    def __init__(self, transient: bool = ...) -> None: ...
    progress: Incomplete
    def __enter__(self): ...
    def __exit__(self, *args) -> None: ...
    step_name: Incomplete
    step: Incomplete
    def __call__(
        self,
        step_name: str,
        step_artifact: Any,
        file: Optional[Mapping] = ...,
        total: Optional[int] = ...,
        completed: Optional[int] = ...,
    ): ...
