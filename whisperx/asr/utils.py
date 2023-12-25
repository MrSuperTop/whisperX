from __future__ import annotations

import typing
from functools import lru_cache
from pathlib import Path
from typing import Any

import faster_whisper.tokenizer
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torch.utils.data
from transformers.pipelines.base import os

from whisperx.audio import HOP_LENGTH, N_FFT, N_MELS, AudioData, load_audio
from whisperx.types import DeviceType

if typing.TYPE_CHECKING:
    from _typeshed import StrPath


@lru_cache(maxsize=None)
def mel_filters(device: torch.device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """

    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"

    with np.load(
        os.path.join(os.path.dirname(__file__), "..", "assets", "mel_filters.npz")
    ) as handle:
        data: npt.NDArray[Any] = handle[f"mel_{n_mels}"]

        return torch.from_numpy(data).to(device)  # pyright: ignore


def log_mel_spectrogram(
    audio: StrPath | AudioData | torch.Tensor,
    n_mels: int = N_MELS,
    padding: int = 0,
    device: DeviceType | torch.device | None = None,
) -> torch.Tensor:
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """

    if not isinstance(audio, torch.Tensor):
        if isinstance(audio, str | Path | os.PathLike):
            audio = load_audio(audio)

        audio = torch.from_numpy(audio)  # pyright: ignore

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)  # pyright: ignore [reportUnknownMemberType]
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def find_numeral_symbol_tokens(
    tokenizer: faster_whisper.tokenizer.Tokenizer
) -> list[int]:
    numeral_symbol_tokens: list[int] = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)

    return numeral_symbol_tokens
