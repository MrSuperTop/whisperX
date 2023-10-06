from ..annotation import Annotation as Annotation
from ..feature import SlidingWindowFeature as SlidingWindowFeature
from ..segment import Segment as Segment, SlidingWindow as SlidingWindow
from ..timeline import Timeline as Timeline
from .generators import string_generator as string_generator
from _typeshed import Incomplete
from typing import List, Optional, Text, Union

def one_hot_encoding(annotation: Annotation, support: Union[Segment, Timeline], window: Union[SlidingWindow, SlidingWindowFeature], labels: Optional[List[Text]] = ..., mode: Text = ...) -> SlidingWindowFeature: ...
def one_hot_decoding(y, window, labels: Incomplete | None = ...): ...
