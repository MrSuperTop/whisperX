from typing import Optional

from _typeshed import Incomplete
from pyannote.core import Annotation as Annotation
from pyannote.core import Timeline as Timeline

from .base import BaseMetric as BaseMetric
from .base import f_measure as f_measure
from .types import Details as Details
from .types import MetricComponents as MetricComponents
from .utils import UEMSupportMixin as UEMSupportMixin

DER_NAME: str
DER_TOTAL: str
DER_FALSE_ALARM: str
DER_MISS: str

class DetectionErrorRate(UEMSupportMixin, BaseMetric):
    @classmethod
    def metric_name(cls) -> str: ...
    @classmethod
    def metric_components(cls) -> MetricComponents: ...
    collar: Incomplete
    skip_overlap: Incomplete
    def __init__(
        self, collar: float = ..., skip_overlap: bool = ..., **kwargs
    ) -> None: ...
    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        **kwargs,
    ) -> Details: ...
    def compute_metric(self, detail: Details) -> float: ...

ACCURACY_NAME: str
ACCURACY_TRUE_POSITIVE: str
ACCURACY_TRUE_NEGATIVE: str
ACCURACY_FALSE_POSITIVE: str
ACCURACY_FALSE_NEGATIVE: str

class DetectionAccuracy(DetectionErrorRate):
    @classmethod
    def metric_name(cls): ...
    @classmethod
    def metric_components(cls) -> MetricComponents: ...
    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        **kwargs,
    ) -> Details: ...
    def compute_metric(self, detail: Details) -> float: ...

PRECISION_NAME: str
PRECISION_RETRIEVED: str
PRECISION_RELEVANT_RETRIEVED: str

class DetectionPrecision(DetectionErrorRate):
    @classmethod
    def metric_name(cls): ...
    @classmethod
    def metric_components(cls) -> MetricComponents: ...
    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        **kwargs,
    ) -> Details: ...
    def compute_metric(self, detail: Details) -> float: ...

RECALL_NAME: str
RECALL_RELEVANT: str
RECALL_RELEVANT_RETRIEVED: str

class DetectionRecall(DetectionErrorRate):
    @classmethod
    def metric_name(cls): ...
    @classmethod
    def metric_components(cls) -> MetricComponents: ...
    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        **kwargs,
    ) -> Details: ...
    def compute_metric(self, detail: Details) -> float: ...

DFS_NAME: str
DFS_PRECISION_RETRIEVED: str
DFS_RECALL_RELEVANT: str
DFS_RELEVANT_RETRIEVED: str

class DetectionPrecisionRecallFMeasure(UEMSupportMixin, BaseMetric):
    @classmethod
    def metric_name(cls): ...
    @classmethod
    def metric_components(cls): ...
    collar: Incomplete
    skip_overlap: Incomplete
    beta: Incomplete
    def __init__(
        self, collar: float = ..., skip_overlap: bool = ..., beta: float = ..., **kwargs
    ) -> None: ...
    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        **kwargs,
    ) -> Details: ...
    def compute_metric(self, detail: Details) -> float: ...
    def compute_metrics(
        self, detail: Optional[Details] = ...
    ) -> tuple[float, float, float]: ...

DCF_NAME: str
DCF_POS_TOTAL: str
DCF_NEG_TOTAL: str
DCF_FALSE_ALARM: str
DCF_MISS: str

class DetectionCostFunction(UEMSupportMixin, BaseMetric):
    collar: Incomplete
    skip_overlap: Incomplete
    fa_weight: Incomplete
    miss_weight: Incomplete
    def __init__(
        self,
        collar: float = ...,
        skip_overlap: bool = ...,
        fa_weight: float = ...,
        miss_weight: float = ...,
        **kwargs,
    ) -> None: ...
    @classmethod
    def metric_name(cls): ...
    @classmethod
    def metric_components(cls) -> MetricComponents: ...
    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        **kwargs,
    ) -> Details: ...
    def compute_metric(self, components: Details) -> float: ...
