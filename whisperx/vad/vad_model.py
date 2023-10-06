from dataclasses import dataclass, field
from typing import Any, Callable, MutableMapping, cast

from pyannote.audio import pipelines
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.core import Annotation
from pyannote.core.feature import SlidingWindowFeature
from pyannote.pipeline import Pipeline


@dataclass
class VadOptions:
    onset: float = field(default=0.500)
    offset: float = field(default=0.363)


Hook = Callable[[str, Any, Any], Any]

DEFAULT_VAD_MODEL = "pyannote/segmentation"


class VoiceActivityDetectionPipeline(pipelines.VoiceActivityDetection):
    def __init__(
        self,
        segmentation: PipelineModel = DEFAULT_VAD_MODEL,
        fscore: bool = False,
        use_auth_token: str | None = None,
        **inference_kwargs: Any
    ) -> None:
        super().__init__(
            segmentation=segmentation,
            fscore=fscore,
            use_auth_token=use_auth_token,
            **inference_kwargs
        )

    def instantiate(self, params: dict[Any, Any]) -> Pipeline:
        return super().instantiate(params)

    def apply(self, file: AudioFile, hook: Hook | None = None) -> Annotation:
        """Apply voice activity detection

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hook : callable, optional
            Hook called after each major step of the pipeline with the following
            signature: hook("step_name", step_artefact, file=file)

        Returns
        -------
        speech : Annotation
            Speech regions.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, 1)
        if self.training and isinstance(file, MutableMapping):
            if self.CACHED_SEGMENTATION in file:
                segmentations = cast(Any, file[self.CACHED_SEGMENTATION])
            else:
                segmentations = self._segmentation(file)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations = self._segmentation(file)

        return cast(Annotation, segmentations)

    def __call__(self, file: AudioFile, **kwargs: Any) -> SlidingWindowFeature:
        return cast(SlidingWindowFeature, super().__call__(file, **kwargs)) # pyright: ignore [reportUnknownMemberType]
