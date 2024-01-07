from __future__ import annotations

from typing import BinaryIO, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional
from faster_whisper.audio import decode_audio
from torch.functional import Tensor
from transformers.dynamic_module_utils import typing

from whisperx.utils import exact_div
from whisperx.utils.convert_path import convert_path

if typing.TYPE_CHECKING:
    from _typeshed import StrPath

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


AudioData = npt.NDArray[np.float32]


def load_audio(file: StrPath | BinaryIO, sr: int = SAMPLE_RATE) -> AudioData:
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    formatted_path = convert_path(file)
    audio_data = cast(
        npt.NDArray[np.float32], decode_audio(formatted_path, sr, split_stereo=False)
    )

    return audio_data


def pad_or_trim(
    array: torch.Tensor | AudioData, length: int = N_SAMPLES, *, axis: int = -1
) -> torch.Tensor | AudioData:
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """

    pad_widths: list[tuple[int, int]]
    if isinstance(array, Tensor):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = torch.nn.functional.pad(
                array, [pad for sizes in pad_widths[::-1] for pad in sizes]
            )
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array
