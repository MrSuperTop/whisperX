from typing import Optional

from _typeshed import Incomplete

from .protocol.protocol import Preprocessors as Preprocessors

class Database:
    def __init__(self, preprocessors: Incomplete | None = ...) -> None: ...
    protocols_: Incomplete
    def register_protocol(self, task_name, protocol_name, protocol) -> None: ...
    def get_tasks(self): ...
    def get_protocols(self, task): ...
    def get_protocol(
        self, task, protocol, preprocessors: Optional[Preprocessors] = ...
    ): ...
