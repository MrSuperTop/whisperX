from typing import Optional

from _typeshed import Incomplete
from pyannote.core import Annotation as Annotation
from pyannote.core import Timeline as Timeline
from pyannote.core.utils.types import Label as Label

from .base import BaseMetric as BaseMetric
from .base import f_measure as f_measure
from .identification import IdentificationErrorRate as IdentificationErrorRate
from .matcher import GreedyMapper as GreedyMapper
from .matcher import HungarianMapper as HungarianMapper
from .types import Details as Details
from .types import MetricComponents as MetricComponents
from .utils import UEMSupportMixin as UEMSupportMixin

DER_NAME: str

class DiarizationErrorRate(IdentificationErrorRate):
    @classmethod
    def metric_name(cls) -> str: ...
    mapper_: Incomplete
    def __init__(
        self, collar: float = ..., skip_overlap: bool = ..., **kwargs
    ) -> None: ...
    def optimal_mapping(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
    ) -> dict[Label, Label]: ...
    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        **kwargs,
    ) -> Details: ...

class GreedyDiarizationErrorRate(IdentificationErrorRate):
    @classmethod
    def metric_name(cls): ...
    mapper_: Incomplete
    def __init__(
        self, collar: float = ..., skip_overlap: bool = ..., **kwargs
    ) -> None: ...
    def greedy_mapping(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
    ) -> dict[Label, Label]: ...
    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        **kwargs,
    ) -> Details: ...

JER_NAME: str
JER_SPEAKER_ERROR: str
JER_SPEAKER_COUNT: str

class JaccardErrorRate(DiarizationErrorRate):
    @classmethod
    def metric_name(cls): ...
    @classmethod
    def metric_components(cls) -> MetricComponents: ...
    mapper_: Incomplete
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

PURITY_NAME: str
PURITY_TOTAL: str
PURITY_CORRECT: str

class DiarizationPurity(UEMSupportMixin, BaseMetric):
    @classmethod
    def metric_name(cls): ...
    @classmethod
    def metric_components(cls): ...
    weighted: Incomplete
    collar: Incomplete
    skip_overlap: Incomplete
    def __init__(
        self,
        collar: float = ...,
        skip_overlap: bool = ...,
        weighted: bool = ...,
        **kwargs,
    ) -> None: ...
    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        **kwargs,
    ) -> Details: ...
    def compute_metric(self, detail: Details) -> float: ...

COVERAGE_NAME: str

class DiarizationCoverage(DiarizationPurity):
    @classmethod
    def metric_name(cls): ...
    def __init__(
        self,
        collar: float = ...,
        skip_overlap: bool = ...,
        weighted: bool = ...,
        **kwargs,
    ) -> None: ...
    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        **kwargs,
    ) -> Details: ...

PURITY_COVERAGE_NAME: str
PURITY_COVERAGE_LARGEST_CLASS: str
PURITY_COVERAGE_TOTAL_CLUSTER: str
PURITY_COVERAGE_LARGEST_CLUSTER: str
PURITY_COVERAGE_TOTAL_CLASS: str

class DiarizationPurityCoverageFMeasure(UEMSupportMixin, BaseMetric):
    @classmethod
    def metric_name(cls): ...
    @classmethod
    def metric_components(cls) -> MetricComponents: ...
    collar: Incomplete
    skip_overlap: Incomplete
    weighted: Incomplete
    beta: Incomplete
    def __init__(
        self,
        collar: float = ...,
        skip_overlap: bool = ...,
        weighted: bool = ...,
        beta: float = ...,
        **kwargs,
    ) -> None: ...
    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        **kwargs,
    ) -> Details: ...
    def compute_metric(self, detail): ...
    def compute_metrics(self, detail: Incomplete | None = ...): ...

HOMOGENEITY_NAME: str
HOMOGENEITY_ENTROPY: str
HOMOGENEITY_CROSS_ENTROPY: str

class DiarizationHomogeneity(UEMSupportMixin, BaseMetric):
    @classmethod
    def metric_name(cls): ...
    @classmethod
    def metric_components(cls): ...
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
    def compute_metric(self, detail): ...

COMPLETENESS_NAME: str

class DiarizationCompleteness(DiarizationHomogeneity):
    @classmethod
    def metric_name(cls): ...
    def compute_components(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Optional[Timeline] = ...,
        **kwargs,
    ) -> Details: ...
