from enum import Enum
from pathlib import Path
from typing import Optional, Union

from _typeshed import Incomplete
from pyannote.database.protocol.protocol import (
    Preprocessors as Preprocessors,
)
from pyannote.database.protocol.protocol import (
    Protocol as Protocol,
)

from .custom import create_protocol as create_protocol
from .custom import get_init as get_init
from .database import Database as Database

class LoadingMode(Enum):
    OVERRIDE: int
    KEEP: int
    ERROR: int

class Registry:
    configs: Incomplete
    sources: Incomplete
    databases: Incomplete
    def __init__(self) -> None: ...
    def load_database(self, path: Union[str, Path], mode: LoadingMode = ...): ...
    def get_database(self, database_name, **kwargs) -> Database: ...
    def get_protocol(
        self, name, preprocessors: Optional[Preprocessors] = ...
    ) -> Protocol: ...

registry: Incomplete
