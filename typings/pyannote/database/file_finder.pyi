from pathlib import Path

from _typeshed import Incomplete
from pyannote.database.protocol.protocol import ProtocolFile as ProtocolFile

from .registry import Registry as Registry

class FileFinder:
    registry: Incomplete
    def __init__(self, registry: Registry = ..., database_yml: str = ...) -> None: ...
    def __call__(self, current_file: ProtocolFile) -> Path: ...
