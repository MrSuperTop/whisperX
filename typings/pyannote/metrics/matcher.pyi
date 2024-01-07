from typing import Iterable

from pyannote.core import Annotation as Annotation
from pyannote.core.utils.types import Label as Label

MATCH_CORRECT: str
MATCH_CONFUSION: str
MATCH_MISSED_DETECTION: str
MATCH_FALSE_ALARM: str
MATCH_TOTAL: str

class LabelMatcher:
    def match(self, rlabel: Label, hlabel: Label) -> bool: ...
    def __call__(
        self, rlabels: Iterable[Label], hlabels: Iterable[Label]
    ) -> tuple[dict[str, int], dict[str, list[Label]]]: ...

class HungarianMapper:
    def __call__(self, A: Annotation, B: Annotation) -> dict[Label, Label]: ...

class GreedyMapper:
    def __call__(self, A: Annotation, B: Annotation) -> dict[Label, Label]: ...
