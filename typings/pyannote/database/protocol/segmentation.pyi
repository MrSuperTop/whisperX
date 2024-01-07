from typing import Optional

from pyannote.core import Annotation as Annotation
from pyannote.core import Timeline as Timeline

from .protocol import (
    Preprocessor as Preprocessor,
)
from .protocol import (
    Preprocessors as Preprocessors,
)
from .protocol import (
    Protocol as Protocol,
)
from .protocol import (
    ProtocolFile as ProtocolFile,
)
from .protocol import (
    Subset as Subset,
)

def crop_annotated(
    current_file: ProtocolFile, existing_preprocessor: Optional[Preprocessor] = ...
) -> Timeline: ...
def crop_annotation(
    current_file: ProtocolFile, existing_preprocessor: Optional[Preprocessor] = ...
) -> Annotation: ...

class SegmentationProtocol(Protocol):
    def __init__(self, preprocessors: Optional[Preprocessors] = ...) -> None: ...
    def stats(self, subset: Subset = ...) -> dict: ...
