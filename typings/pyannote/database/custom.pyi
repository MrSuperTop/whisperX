from pathlib import Path
from typing import Any, Callable, Union

from _typeshed import Incomplete
from pyannote.database.protocol.protocol import ProtocolFile

from .loader import load_lst as load_lst
from .loader import load_trial as load_trial
from .protocol.protocol import Scope as Scope
from .protocol.protocol import Subset as Subset
from .protocol.segmentation import SegmentationProtocol as SegmentationProtocol
from .protocol.speaker_diarization import (
    SpeakerDiarizationProtocol as SpeakerDiarizationProtocol,
)
from .util import get_annotated as get_annotated

LOADERS: Incomplete

def Template(template: str, database_yml: Path) -> Callable[[ProtocolFile], Any]: ...
def NumericValue(value): ...
def resolve_path(path: Path, database_yml: Path) -> Path: ...
def meta_subset_iter(
    meta_database: str,
    meta_task: str,
    meta_protocol: str,
    meta_subset: Subset,
    subset_entries: dict,
    database_yml: Path,
): ...
def gather_loaders(entries: dict, database_yml: Path) -> dict: ...
def subset_iter(
    self,
    database: str = ...,
    task: str = ...,
    protocol: str = ...,
    subset: Subset = ...,
    entries: dict = ...,
    database_yml: Path = ...,
    **metadata,
): ...
def subset_trial(
    self,
    database: str = ...,
    task: str = ...,
    protocol: str = ...,
    subset: Subset = ...,
    entries: dict = ...,
    database_yml: Path = ...,
): ...
def get_init(protocols): ...
def get_custom_protocol_class_name(database: str, task: str, protocol: str): ...
def create_protocol(
    database: str,
    task: str,
    protocol: str,
    protocol_entries: dict,
    database_yml: Path,
) -> Union[type, None]: ...
