from .registry import Registry as Registry
from _typeshed import Incomplete
from pathlib import Path
from pyannote.database.protocol.protocol import ProtocolFile as ProtocolFile
from typing import Text

class FileFinder:
    registry: Incomplete
    def __init__(self, registry: Registry = ..., database_yml: Text = ...) -> None: ...
    def __call__(self, current_file: ProtocolFile) -> Path: ...
