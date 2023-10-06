from ..identification import UEMSupportMixin as UEMSupportMixin
from ..matcher import LabelMatcher as LabelMatcher, MATCH_CONFUSION as MATCH_CONFUSION, MATCH_CORRECT as MATCH_CORRECT, MATCH_FALSE_ALARM as MATCH_FALSE_ALARM, MATCH_MISSED_DETECTION as MATCH_MISSED_DETECTION
from _typeshed import Incomplete
from pyannote.core import Annotation, Timeline as Timeline
from typing import Optional
from xarray import DataArray

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
    def difference(self, reference: Annotation, hypothesis: Annotation, uem: Optional[Timeline] = ..., uemified: bool = ...): ...
    def regression(self, reference: Annotation, before: Annotation, after: Annotation, uem: Optional[Timeline] = ..., uemified: bool = ...): ...
    def matrix(self, reference: Annotation, hypothesis: Annotation, uem: Optional[Timeline] = ...) -> DataArray: ...
