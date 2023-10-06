from _typeshed import Incomplete
from itertools import chain as chain
from pyannote.core import Annotation as Annotation
from pyannote.database import ProtocolFile as ProtocolFile
from typing import Dict, List, Optional

class LowerTemporalResolution:
    preprocessed_key: str
    resolution: Incomplete
    def __init__(self, resolution: float = ...) -> None: ...
    def __call__(self, current_file: ProtocolFile) -> Annotation: ...

class DeriveMetaLabels:
    classes: Incomplete
    unions: Incomplete
    intersections: Incomplete
    def __init__(self, classes: List[str], unions: Optional[Dict[str, List[str]]] = ..., intersections: Optional[Dict[str, List[str]]] = ...) -> None: ...
    @property
    def all_classes(self) -> List[str]: ...
    def __call__(self, current_file: ProtocolFile) -> Annotation: ...
