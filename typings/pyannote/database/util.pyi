from pathlib import Path as Path

from _typeshed import Incomplete
from pyannote.core import Annotation

from .protocol.protocol import ProtocolFile as ProtocolFile

DatabaseName = str
PathTemplate = str

def get_unique_identifier(item): ...
def get_annotated(current_file): ...
def get_label_identifier(label, current_file): ...
def load_rttm(file_rttm, keep_type: str = ...): ...
def load_stm(file_stm): ...
def load_mdtm(file_mdtm): ...
def load_uem(file_uem): ...
def load_lab(path, uri: str = ...) -> Annotation: ...
def load_lst(file_lst): ...
def load_mapping(mapping_txt): ...

class LabelMapper:
    mapping: Incomplete
    keep_missing: Incomplete
    def __init__(self, mapping, keep_missing: bool = ...) -> None: ...
    def __call__(self, current_file): ...
