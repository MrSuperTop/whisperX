from .base import BaseMetric as BaseMetric
from .binary_classification import det_curve as det_curve
from .types import Details as Details, MetricComponents as MetricComponents
from _typeshed import Incomplete
from numpy.typing import ArrayLike as ArrayLike
from pyannote.core import Annotation, SlidingWindowFeature, Timeline as Timeline
from typing import Dict, Iterable, Optional, Tuple, Union

SPOTTING_TARGET: str
SPOTTING_SPK_LATENCY: str
SPOTTING_SPK_SCORE: str
SPOTTING_ABS_LATENCY: str
SPOTTING_ABS_SCORE: str
SPOTTING_SCORE: str

class LowLatencySpeakerSpotting(BaseMetric):
    @classmethod
    def metric_name(cls) -> str: ...
    def metric_components(self) -> Dict[str, float]: ...
    thresholds: Incomplete
    latencies: Incomplete
    def __init__(self, thresholds: Optional[ArrayLike] = ..., latencies: Optional[ArrayLike] = ...) -> None: ...
    def compute_metric(self, detail: MetricComponents): ...
    def compute_components(self, reference: Union[Timeline, Annotation], hypothesis: Union[SlidingWindowFeature, Iterable[Tuple[float, float]]], **kwargs) -> Details: ...
    @property
    def absolute_latency(self): ...
    @property
    def speaker_latency(self): ...
    def det_curve(self, cost_miss: float = ..., cost_fa: float = ..., prior_target: float = ..., return_latency: bool = ...): ...
