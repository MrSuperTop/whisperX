from __future__ import annotations

import os
import typing
from pathlib import Path
from typing import Literal, TypeAlias, cast

import pandas as pd
import torch
from pyannote.audio.pipelines.speaker_diarization import (
    SpeakerDiarization as PyannoteSpeakerDiarization,
)
from pyannote.core.annotation import Annotation

from whisperx.audio import SAMPLE_RATE, AudioData, load_audio
from whisperx.types import DeviceType
from whisperx.utils import get_device

if typing.TYPE_CHECKING:
    from _typeshed import StrPath


DiarizeModelName: TypeAlias = Literal['pyannote/speaker-diarization-3.1']
DEFAULT_DIARIZE_MODEL_NAME: DiarizeModelName = 'pyannote/speaker-diarization-3.1'


class DiarizationPipeline:
    def __init__(
        self,
        model_name: DiarizeModelName = DEFAULT_DIARIZE_MODEL_NAME,
        use_auth_token: str | None = None,
        device: DeviceType | torch.device = 'cpu',
    ) -> None:
        device = get_device(device)

        self.model = cast(
            PyannoteSpeakerDiarization,
            PyannoteSpeakerDiarization.from_pretrained(
                model_name, use_auth_token=use_auth_token
            ).to(device),  # pyright: ignore[reportUnknownMemberType]
        )

    def __call__(
        self,
        audio: StrPath | AudioData,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> pd.DataFrame:
        if isinstance(audio, os.PathLike | Path | str):
            loaded_audio = load_audio(audio)
        else:
            loaded_audio = audio

        audio_data = {
            'waveform': torch.from_numpy(loaded_audio[None, :]),  # pyright: ignore[reportUnknownMemberType]
            'sample_rate': SAMPLE_RATE,
        }

        segments: Annotation = cast(
            Annotation,
            self.model(
                audio_data,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            ),
        )

        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True))
        diarize_df['start'] = diarize_df[0].apply(lambda x: x.start)  # pyright: ignore[reportUnknownMemberType, reportUnknownLambdaType]
        diarize_df['end'] = diarize_df[0].apply(lambda x: x.end)  # pyright: ignore[reportUnknownMemberType, reportUnknownLambdaType]
        diarize_df.rename(columns={2: 'speaker'}, inplace=True)

        return diarize_df
