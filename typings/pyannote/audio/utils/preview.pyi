from IPython.display import Video as IPythonVideo
from pyannote.audio.core.io import AudioFile as AudioFile
from pyannote.audio.core.model import Model as Model
from pyannote.core import Segment
from typing import Union

IPYTHON_INSTALLED: bool
MOVIEPY_INSTALLED: bool

def listen(audio_file: AudioFile, segment: Segment = ...) -> None: ...
def preview(audio_file: AudioFile, segment: Segment = ..., zoom: float = ..., video_fps: int = ..., video_ext: str = ..., display: bool = ..., **views): ...
def BROKEN_preview_training_samples(model: Model, blank: float = ..., video_fps: int = ..., video_ext: str = ..., display: bool = ...) -> Union[IPythonVideo, str]: ...
