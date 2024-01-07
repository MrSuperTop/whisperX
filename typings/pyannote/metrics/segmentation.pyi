from typing import Optional, Union

from _typeshed import Incomplete
from pyannote.core import Annotation, Timeline

from .base import BaseMetric as BaseMetric
from .base import f_measure as f_measure
from .types import Details as Details
from .types import MetricComponents as MetricComponents
from .utils import UEMSupportMixin as UEMSupportMixin

PURITY_NAME: str
COVERAGE_NAME: str
PURITY_COVERAGE_NAME: str
PTY_CVG_TOTAL: str
PTY_CVG_INTER: str
PTY_TOTAL: str
PTY_INTER: str
CVG_TOTAL: str
CVG_INTER: str
PRECISION_NAME: str
RECALL_NAME: str
PR_BOUNDARIES: str
PR_MATCHES: str

class SegmentationCoverage(BaseMetric):
    tolerance: Incomplete
    def __init__(self, tolerance: float = ..., **kwargs) -> None: ...
    @classmethod
    def metric_name(cls): ...
    @classmethod
    def metric_components(cls) -> MetricComponents: ...
    def compute_components(
        self, reference: Annotation, hypothesis: Union[Annotation, Timeline], **kwargs
    ): ...
    def compute_metric(self, detail: Details) -> float: ...

class SegmentationPurity(SegmentationCoverage):
    @classmethod
    def metric_name(cls) -> str: ...
    def compute_components(
        self, reference: Annotation, hypothesis: Union[Annotation, Timeline], **kwargs
    ) -> Details: ...

class SegmentationPurityCoverageFMeasure(SegmentationCoverage):
    beta: Incomplete
    def __init__(self, tolerance: float = ..., beta: int = ..., **kwargs) -> None: ...
    def compute_components(
        self, reference: Annotation, hypothesis: Union[Annotation, Timeline], **kwargs
    ) -> Details: ...
    def compute_metric(self, detail: Details) -> float: ...
    def compute_metrics(
        self, detail: Optional[Details] = ...
    ) -> tuple[float, float, float]: ...
    @classmethod
    def metric_name(cls) -> str: ...
    @classmethod
    def metric_components(cls) -> MetricComponents: ...

class SegmentationPrecision(UEMSupportMixin, BaseMetric):
    @classmethod
    def metric_name(cls): ...
    @classmethod
    def metric_components(cls): ...
    tolerance: Incomplete
    def __init__(self, tolerance: float = ..., **kwargs) -> None: ...
    def compute_components(
        self,
        reference: Union[Annotation, Timeline],
        hypothesis: Union[Annotation, Timeline],
        **kwargs,
    ) -> Details: ...
    def compute_metric(self, detail: Details) -> float: ...

class SegmentationRecall(SegmentationPrecision):
    @classmethod
    def metric_name(cls): ...
    def compute_components(
        self,
        reference: Union[Annotation, Timeline],
        hypothesis: Union[Annotation, Timeline],
        **kwargs,
    ) -> Details: ...
