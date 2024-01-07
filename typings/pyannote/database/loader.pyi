from collections.abc import Generator
from pathlib import Path
from typing import Any, Union

from _typeshed import Incomplete
from pyannote.core import Annotation, Timeline
from pyannote.core import Segment as Segment
from pyannote.database.protocol.protocol import ProtocolFile as ProtocolFile
from spacy.tokens import Doc

def load_lst(file_lst): ...
def load_trial(file_trial) -> Generator[Incomplete, None, None]: ...

class RTTMLoader:
    path: Incomplete
    placeholders_: Incomplete
    loaded_: Incomplete
    def __init__(self, path: str = ...) -> None: ...
    def __call__(self, file: ProtocolFile) -> Annotation: ...

class STMLoader:
    path: Incomplete
    placeholders_: Incomplete
    loaded_: Incomplete
    def __init__(self, path: str = ...) -> None: ...
    def __call__(self, file: ProtocolFile) -> Annotation: ...

class UEMLoader:
    path: Incomplete
    placeholders_: Incomplete
    loaded_: Incomplete
    def __init__(self, path: str = ...) -> None: ...
    def __call__(self, file: ProtocolFile) -> Timeline: ...

class LABLoader:
    path: Incomplete
    placeholders_: Incomplete
    def __init__(self, path: str = ...) -> None: ...
    def __call__(self, file: ProtocolFile) -> Annotation: ...

class CTMLoader:
    ctm: Incomplete
    data_: Incomplete
    def __init__(self, ctm: Path) -> None: ...
    def __call__(self, current_file: ProtocolFile) -> Union[Doc, None]: ...

class MAPLoader:
    mapping: Incomplete
    data_: Incomplete
    dtype: Incomplete
    def __init__(self, mapping: Path) -> None: ...
    def __call__(self, current_file: ProtocolFile) -> Any: ...
