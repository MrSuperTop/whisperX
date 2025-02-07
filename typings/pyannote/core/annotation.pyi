from typing import (
    Hashable,
    Iterable,
    Iterator,
    Optional,
    TextIO,
    Union,
)

import numpy as np
import pandas as pd
from _typeshed import Incomplete

from . import PYANNOTE_LABEL as PYANNOTE_LABEL
from . import PYANNOTE_SEGMENT as PYANNOTE_SEGMENT
from . import PYANNOTE_TRACK as PYANNOTE_TRACK
from .feature import SlidingWindowFeature as SlidingWindowFeature
from .segment import Segment as Segment
from .segment import SlidingWindow as SlidingWindow
from .timeline import Timeline as Timeline
from .utils.generators import int_generator as int_generator
from .utils.generators import string_generator as string_generator
from .utils.types import CropMode as CropMode
from .utils.types import Key as Key
from .utils.types import Label as Label
from .utils.types import LabelGenerator as LabelGenerator
from .utils.types import Support as Support
from .utils.types import TrackName as TrackName

class Annotation:
    @classmethod
    def from_df(
        cls, df: pd.DataFrame, uri: Optional[str] = ..., modality: Optional[str] = ...
    ) -> Annotation: ...
    modality: Incomplete
    def __init__(
        self, uri: Optional[str] = ..., modality: Optional[str] = ...
    ) -> None: ...
    @property
    def uri(self): ...
    def __len__(self) -> int: ...
    def __nonzero__(self): ...
    def __bool__(self) -> bool: ...
    def itersegments(self) -> Iterator[Segment]: ...
    def itertracks(
        self, yield_label: bool = ...
    ) -> Iterator[
        Union[tuple[Segment, TrackName], tuple[Segment, TrackName, Label]]
    ]: ...
    def get_timeline(self, copy: bool = ...) -> Timeline: ...
    def __eq__(self, other: Annotation): ...
    def __ne__(self, other: Annotation): ...
    def __contains__(self, included: Union[Segment, Timeline]): ...
    def to_rttm(self) -> str: ...
    def write_rttm(self, file: TextIO): ...
    def to_lab(self) -> str: ...
    def write_lab(self, file: TextIO): ...
    def crop(self, support: Support, mode: CropMode = ...) -> Annotation: ...
    def extrude(self, removed: Support, mode: CropMode = ...) -> Annotation: ...
    def get_overlap(self, labels: Optional[Iterable[Label]] = ...) -> Timeline: ...
    def get_tracks(self, segment: Segment) -> set[TrackName]: ...
    def has_track(self, segment: Segment, track: TrackName) -> bool: ...
    def copy(self) -> Annotation: ...
    def new_track(
        self,
        segment: Segment,
        candidate: Optional[TrackName] = ...,
        prefix: Optional[str] = ...,
    ) -> TrackName: ...
    def __delitem__(self, key: Key): ...
    def __getitem__(self, key: Key) -> Label: ...
    def __setitem__(self, key: Key, label: Label): ...
    def empty(self) -> Annotation: ...
    def labels(self) -> list[Label]: ...
    def get_labels(
        self, segment: Segment, unique: bool = ...
    ) -> Union[set[Label], list[Label]]: ...
    def subset(self, labels: Iterable[Label], invert: bool = ...) -> Annotation: ...
    def update(self, annotation: Annotation, copy: bool = ...) -> Annotation: ...
    def label_timeline(self, label: Label, copy: bool = ...) -> Timeline: ...
    def label_support(self, label: Label) -> Timeline: ...
    def label_duration(self, label: Label) -> float: ...
    def chart(self, percent: bool = ...) -> list[tuple[Label, float]]: ...
    def argmax(self, support: Optional[Support] = ...) -> Optional[Label]: ...
    def rename_tracks(self, generator: LabelGenerator = ...) -> Annotation: ...
    def rename_labels(
        self,
        mapping: Optional[dict] = ...,
        generator: LabelGenerator = ...,
        copy: bool = ...,
    ) -> Annotation: ...
    def relabel_tracks(self, generator: LabelGenerator = ...) -> Annotation: ...
    def support(self, collar: float = ...) -> Annotation: ...
    def co_iter(
        self, other: Annotation
    ) -> Iterator[tuple[tuple[Segment, TrackName], tuple[Segment, TrackName]]]: ...
    def __mul__(self, other: Annotation) -> np.ndarray: ...
    def discretize(
        self,
        support: Optional[Segment] = ...,
        resolution: Union[float, SlidingWindow] = ...,
        labels: Optional[list[Hashable]] = ...,
        duration: Optional[float] = ...,
    ): ...
    @classmethod
    def from_records(
        cls,
        records: Iterator[tuple[Segment, TrackName, Label]],
        uri: Optional[str] = ...,
        modality: Optional[str] = ...,
    ) -> Annotation: ...
