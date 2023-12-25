from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

SegmentId: TypeAlias = tuple[float, float]


@dataclass(frozen=True)
class SegmentsBoundsMerge:
    start: float
    end: float
    segments: list[SegmentId]
