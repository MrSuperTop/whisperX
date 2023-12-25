from __future__ import annotations

import typing
from collections.abc import Iterator
from typing import cast

from pyannote.core import Annotation, Segment
from pyannote.core.annotation import TrackName

if typing.TYPE_CHECKING:
    pass


def remove_shorter_than(min_duration_on: float, annotation: Annotation) -> None:
    tracks_generator = cast(
        Iterator[tuple[Segment, TrackName]], annotation.itertracks(yield_label=False)
    )

    for segment, track in tracks_generator:
        if segment.duration < min_duration_on:
            del annotation[segment, track]
