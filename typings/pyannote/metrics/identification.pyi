from .base import BaseMetric as BaseMetric, PRECISION_RELEVANT_RETRIEVED as PRECISION_RELEVANT_RETRIEVED, PRECISION_RETRIEVED as PRECISION_RETRIEVED, Precision as Precision, RECALL_RELEVANT as RECALL_RELEVANT, RECALL_RELEVANT_RETRIEVED as RECALL_RELEVANT_RETRIEVED, Recall as Recall
from .matcher import LabelMatcher as LabelMatcher, MATCH_CONFUSION as MATCH_CONFUSION, MATCH_CORRECT as MATCH_CORRECT, MATCH_FALSE_ALARM as MATCH_FALSE_ALARM, MATCH_MISSED_DETECTION as MATCH_MISSED_DETECTION, MATCH_TOTAL as MATCH_TOTAL
from .types import Details as Details, MetricComponents as MetricComponents
from .utils import UEMSupportMixin as UEMSupportMixin
from _typeshed import Incomplete
from pyannote.core import Annotation as Annotation, Timeline as Timeline
from typing import Optional

IER_TOTAL = MATCH_TOTAL
IER_CORRECT = MATCH_CORRECT
IER_CONFUSION = MATCH_CONFUSION
IER_FALSE_ALARM = MATCH_FALSE_ALARM
IER_MISS = MATCH_MISSED_DETECTION
IER_NAME: str

class IdentificationErrorRate(UEMSupportMixin, BaseMetric):
    @classmethod
    def metric_name(cls) -> str: ...
    @classmethod
    def metric_components(cls) -> MetricComponents: ...
    matcher_: Incomplete
    confusion: Incomplete
    miss: Incomplete
    false_alarm: Incomplete
    collar: Incomplete
    skip_overlap: Incomplete
    def __init__(self, confusion: float = ..., miss: float = ..., false_alarm: float = ..., collar: float = ..., skip_overlap: bool = ..., **kwargs) -> None: ...
    def compute_components(self, reference: Annotation, hypothesis: Annotation, uem: Optional[Timeline] = ..., collar: Optional[float] = ..., skip_overlap: Optional[float] = ..., **kwargs) -> Details: ...
    def compute_metric(self, detail: Details) -> float: ...

class IdentificationPrecision(UEMSupportMixin, Precision):
    collar: Incomplete
    skip_overlap: Incomplete
    matcher_: Incomplete
    def __init__(self, collar: float = ..., skip_overlap: bool = ..., **kwargs) -> None: ...
    def compute_components(self, reference: Annotation, hypothesis: Annotation, uem: Optional[Timeline] = ..., **kwargs) -> Details: ...

class IdentificationRecall(UEMSupportMixin, Recall):
    collar: Incomplete
    skip_overlap: Incomplete
    matcher_: Incomplete
    def __init__(self, collar: float = ..., skip_overlap: bool = ..., **kwargs) -> None: ...
    def compute_components(self, reference: Annotation, hypothesis: Annotation, uem: Optional[Timeline] = ..., **kwargs) -> Details: ...
