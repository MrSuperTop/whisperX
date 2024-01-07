from itertools import chain as chain
from typing import Optional

from _typeshed import Incomplete
from pyannote.core import Annotation as Annotation
from pyannote.database import ProtocolFile as ProtocolFile

class LowerTemporalResolution:
    preprocessed_key: str
    resolution: Incomplete
    def __init__(self, resolution: float = ...) -> None: ...
    def __call__(self, current_file: ProtocolFile) -> Annotation: ...

class DeriveMetaLabels:
    classes: Incomplete
    unions: Incomplete
    intersections: Incomplete
    def __init__(
        self,
        classes: list[str],
        unions: Optional[dict[str, list[str]]] = ...,
        intersections: Optional[dict[str, list[str]]] = ...,
    ) -> None: ...
    @property
    def all_classes(self) -> list[str]: ...
    def __call__(self, current_file: ProtocolFile) -> Annotation: ...
