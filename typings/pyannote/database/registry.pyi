from .custom import create_protocol as create_protocol, get_init as get_init
from .database import Database as Database
from _typeshed import Incomplete
from enum import Enum
from pathlib import Path
from pyannote.database.protocol.protocol import Preprocessors as Preprocessors, Protocol as Protocol
from typing import Optional, Text, Union

class LoadingMode(Enum):
    OVERRIDE: int
    KEEP: int
    ERROR: int

class Registry:
    configs: Incomplete
    sources: Incomplete
    databases: Incomplete
    def __init__(self) -> None: ...
    def load_database(self, path: Union[Text, Path], mode: LoadingMode = ...): ...
    def get_database(self, database_name, **kwargs) -> Database: ...
    def get_protocol(self, name, preprocessors: Optional[Preprocessors] = ...) -> Protocol: ...

registry: Incomplete
