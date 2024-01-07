from typing import Optional, Union

from _typeshed import Incomplete

from ..annotation import Annotation as Annotation
from ..feature import SlidingWindowFeature as SlidingWindowFeature
from ..segment import Segment as Segment
from ..segment import SlidingWindow as SlidingWindow
from ..timeline import Timeline as Timeline
from .generators import string_generator as string_generator

def one_hot_encoding(
    annotation: Annotation,
    support: Union[Segment, Timeline],
    window: Union[SlidingWindow, SlidingWindowFeature],
    labels: Optional[list[str]] = ...,
    mode: str = ...,
) -> SlidingWindowFeature: ...
def one_hot_decoding(y, window, labels: Incomplete | None = ...): ...
