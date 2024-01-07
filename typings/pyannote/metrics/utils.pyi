from typing import Optional, Union

from pyannote.core import Annotation as Annotation
from pyannote.core import Timeline

class UEMSupportMixin:
    def extrude(
        self,
        uem: Timeline,
        reference: Annotation,
        collar: float = ...,
        skip_overlap: bool = ...,
    ) -> Timeline: ...
    def common_timeline(
        self, reference: Annotation, hypothesis: Annotation
    ) -> Timeline: ...
    def project(self, annotation: Annotation, timeline: Timeline) -> Annotation: ...
    def uemify(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        collar: float = ...,
        skip_overlap: bool = ...,
        returns_uem: bool = ...,
        returns_timeline: bool = ...,
    ) -> Union[
        tuple[Annotation, Annotation],
        tuple[Annotation, Annotation, Timeline],
        tuple[Annotation, Annotation, Timeline, Timeline],
    ]: ...
