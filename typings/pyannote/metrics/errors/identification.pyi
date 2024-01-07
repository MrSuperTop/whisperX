from typing import Optional

from _typeshed import Incomplete
from pyannote.core import Annotation
from pyannote.core import Timeline as Timeline
from xarray import DataArray

from ..identification import UEMSupportMixin as UEMSupportMixin
from ..matcher import (
    MATCH_CONFUSION as MATCH_CONFUSION,
)
from ..matcher import (
    MATCH_CORRECT as MATCH_CORRECT,
)
from ..matcher import (
    MATCH_FALSE_ALARM as MATCH_FALSE_ALARM,
)
from ..matcher import (
    MATCH_MISSED_DETECTION as MATCH_MISSED_DETECTION,
)
from ..matcher import (
    LabelMatcher as LabelMatcher,
)

REFERENCE_TOTAL: str
HYPOTHESIS_TOTAL: str
REGRESSION: str
IMPROVEMENT: str
BOTH_CORRECT: str
BOTH_INCORRECT: str

class IdentificationErrorAnalysis(UEMSupportMixin):
    matcher: Incomplete
    collar: Incomplete
    skip_overlap: Incomplete
    def __init__(self, collar: float = ..., skip_overlap: bool = ...) -> None: ...
    def difference(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        uemified: bool = ...,
    ): ...
    def regression(
        self,
        reference: Annotation,
        before: Annotation,
        after: Annotation,
        uem: Optional[Timeline] = ...,
        uemified: bool = ...,
    ): ...
    def matrix(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
    ) -> DataArray: ...
