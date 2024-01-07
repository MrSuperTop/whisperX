from typing import Union

from pyannote.audio.core.io import AudioFile as AudioFile
from pyannote.core import Annotation as Annotation
from pyannote.core import SlidingWindow, SlidingWindowFeature

def oracle_segmentation(
    file: AudioFile,
    window: SlidingWindow,
    frames: Union[SlidingWindow, float],
    num_speakers: int = ...,
) -> SlidingWindowFeature: ...
