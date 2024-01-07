from typing import Union

from pyannote.core import Annotation, Timeline

class SegmentationErrorAnalysis:
    def __init__(self) -> None: ...
    def __call__(
        self,
        reference: Union[Timeline, Annotation],
        hypothesis: Union[Timeline, Annotation],
    ) -> Annotation: ...
