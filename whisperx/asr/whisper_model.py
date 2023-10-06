from __future__ import annotations

import typing
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import ctranslate2
import ctranslate2.models
import faster_whisper
import faster_whisper.tokenizer
import faster_whisper.transcribe
import numpy as np
import numpy.typing as npt
import tokenizers
import torch
import torch.nn.functional as F
import torch.utils.data
from transformers.pipelines.base import os

from whisperx.audio import (
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    N_SAMPLES,
    SAMPLE_RATE,
    AudioData,
    load_audio,
)
from whisperx.types import DeviceType, LanguageCode

if typing.TYPE_CHECKING:
    from _typeshed import StrPath


class Features(torch.Tensor):
    ...


@dataclass()
class AsrOptions:
    beam_size: int = field(default=5)
    best_of: int = field(default=5)
    patience: float = field(default=1)
    length_penalty: float = field(default=1)
    repetition_penalty: float = field(default=1)
    no_repeat_ngram_size: int = field(default=0)
    temperatures: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    log_prob_threshold: float | None = field(default=-1.0)
    no_speech_threshold: float | None = field(default=0.6)
    compression_ratio_threshold: float | None = field(default=2.4)
    condition_on_previous_text: bool = field(default=False)
    prompt_reset_on_temperature: float = field(default=0.5)
    initial_prompt: str | Iterable[int] | None = field(default=None)
    prefix: str | None = field(default=None)
    suppress_blank: bool = field(default=True)
    suppress_tokens: list[int] | None = field(default_factory=lambda: [-1])
    without_timestamps: bool = field(default=True)
    max_initial_timestamp: float = field(default=0.0)
    word_timestamps: bool = field(default=False)
    prepend_punctuations: str = field(default="\"'“¿([{-")
    append_punctuations: str = field(default="\"'.。,，!！?？:：”)]}、")

    suppress_numerals: bool = field(default=True)


@dataclass()
class LanguageDetails:
    language: LanguageCode
    language_probability: float
    all_language_probs: list[tuple[LanguageCode, float]] | None = field(default=None)

    @classmethod
    def get_defined(cls, for_language: LanguageCode) -> LanguageDetails:
        return cls(
            language=for_language,
            language_probability=1,
            all_language_probs=None
        )


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

        return torch.from_numpy(data).to(device) # pyright: ignore


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

        audio = torch.from_numpy(audio) # pyright: ignore

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
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0) # pyright: ignore [reportUnknownMemberType]
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class BatchingWhisperModel(faster_whisper.WhisperModel):
    """
    BatchingWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    """

    hf_tokenizer: tokenizers.Tokenizer
    is_multilingual: bool
    model: ctranslate2.models.Whisper

    def generate_segment_batched(
        self,
        features: AudioData,
        tokenizer: faster_whisper.tokenizer.Tokenizer,
        options: AsrOptions
    ) -> list[str]:
        batch_size = features.shape[0]
        all_tokens: list[int] = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            if isinstance(options.initial_prompt, str):
                initial_prompt = " " + options.initial_prompt.strip()
                initial_prompt_tokens = tokenizer.encode(initial_prompt)
                all_tokens.extend(initial_prompt_tokens)
            else:
                all_tokens.extend(options.initial_prompt)


        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        # FIXME: Allow accepting properly
        encoder_output = self.encode(features)

        result = self.model.generate(
            encoder_output,
            [prompt] * batch_size,
            length_penalty=options.length_penalty,
            max_length=self.max_length,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
        )

        tokens_batch = [x.sequences_ids[0] for x in result]

        def decode_batch(tokens: list[list[int]]) -> list[str]:
            res: list[list[int]] = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            # text_tokens = [token for token in tokens if token < self.eot]
            return cast(
                list[str],
                tokenizer.tokenizer.decode_batch(res) # pyright: ignore [reportUnknownMemberType]
            )

        text = decode_batch(tokens_batch)

        return text

    def encode(self, features: AudioData) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1

        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)

        features = faster_whisper.transcribe.get_ctranslate2_storage(features) # pyright: ignore [reportUnknownMemberType]

        return self.model.encode(features, to_cpu=to_cpu)

    def detect_language(self, data: Features | AudioData) -> LanguageDetails:
        if isinstance(data, Features):
            features = data
        else:
            features = Features(
                log_mel_spectrogram(
                    data[:SAMPLE_RATE],
                    padding=0 if data.shape[0] >= N_SAMPLES else N_SAMPLES - data.shape[0]
                )
            )

        return self._detect_language(features)

    def _detect_language(self, features: Features) -> LanguageDetails:
        segment = features[:SAMPLE_RATE]
        encoder_output = self.encode(segment.numpy())
        # results is a list of tuple[str, float] with language names and
        # probabilities.
        results = self.model.detect_language(encoder_output)[0]
        # Parse language names to strip out markers
        all_language_probs = cast(
            list[tuple[LanguageCode, float]],
            [(token[2:-2], prob) for (token, prob) in results]
        )

        # Get top language token and probability
        language, language_probability = all_language_probs[0]

        # FIXME: Transition to using a custom logger
        self.logger.info(
            "Detected language '%s' with probability %.2f",
            language,
            language_probability,
        )

        return LanguageDetails(
            language=language,
            language_probability=language_probability,
            all_language_probs=all_language_probs
        )
