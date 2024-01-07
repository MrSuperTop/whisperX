from __future__ import annotations

import os
import typing
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, NotRequired, TypedDict, TypeGuard, TypeVar, cast

import faster_whisper
import faster_whisper.tokenizer
import faster_whisper.transcribe
import torch
import torch.utils.data
from tqdm import tqdm
from transformers import Pipeline
from transformers.pipelines.base import ModelOutput
from transformers.pipelines.pt_utils import PipelineIterator
from typing_extensions import override

from whisperx.asr.batching_whisper import (
    AsrOptions,
    BatchingWhisperModel,
    Features,
    LanguageDetails,
    log_mel_spectrogram,
)
from whisperx.asr.types import Segment, TranscriptionResult
from whisperx.asr.utils import find_numeral_symbol_tokens
from whisperx.audio import N_SAMPLES, SAMPLE_RATE, AudioData, load_audio
from whisperx.logging import get_logger
from whisperx.types import DeviceType, LanguageCode, TaskType
from whisperx.utils import get_device
from whisperx.vad import SegmentsBoundsMerge, merge_chunks
from whisperx.vad.vad_pipeline import VoiceActivityDetectionPipeline

if typing.TYPE_CHECKING:
    from _typeshed import StrPath


class PreprocessParams(TypedDict):
    tokenizer: NotRequired[Any]
    maybe_arg: NotRequired[Any]


class ForwardParams(TypedDict):
    ...


class PostprocessParams(TypedDict):
    ...


T = TypeVar('T', torch.Tensor, AudioData)


@dataclass()
class StackedAudio(Generic[T]):
    inputs: T


@dataclass()
class BatchingWhisperOutput(ModelOutput):
    text: list[str] | str


def is_tokenizer_wrong_or_none(
    tokenizer: faster_whisper.tokenizer.Tokenizer | None,
    is_multilingual: bool,
    check_language: LanguageCode | None,
    check_task: TaskType,
) -> TypeGuard[faster_whisper.tokenizer.Tokenizer]:
    if tokenizer is None:
        return True

    formatted_given_task: str
    if is_multilingual:
        formatted_given_task = cast(
            str, tokenizer.tokenizer.token_to_id('<|%s|>' % check_task)
        )  # pyright: ignore [reportUnknownMemberType]
    else:
        formatted_given_task = check_task

    is_wrong_tokenizer = check_task != formatted_given_task or check_language != cast(
        LanguageCode, tokenizer.language_code
    )

    return is_wrong_tokenizer


class BatchingWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """

    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
        self,
        model: BatchingWhisperModel,
        vad_model: VoiceActivityDetectionPipeline,
        options: AsrOptions = AsrOptions(),
        tokenizer: faster_whisper.tokenizer.Tokenizer | None = None,
        device: int | DeviceType | torch.device = -1,
        framework: str = 'pt',
        language: LanguageCode | None = None,
        suppress_numerals: bool = False,
        batch_size: int | None = None,
        **kwargs: PreprocessParams,
    ) -> None:
        self.logger = get_logger(__name__)

        self.model = model

        # TODO: Consider moving the tokenizer to BatchingWhisperModel class or straigt up feed the tokens into the model during forward instead of passing the tokenizer
        self._tokenizer = tokenizer
        self.options = options
        self.preset_language: LanguageCode | None = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = batch_size

        (
            self._preprocess_params,
            self._forward_params,
            self._postprocess_params,
        ) = self._sanitize_parameters(**kwargs)

        self.call_count = 0
        self.framework = framework

        if self.framework == 'pt':
            self.device = get_device(device)
        else:
            self.device = device

        super(Pipeline, self).__init__()
        self.vad_model = vad_model

    def _sanitize_parameters(
        self, **kwargs: PreprocessParams
    ) -> tuple[PreprocessParams, ForwardParams, PostprocessParams]:
        preprocess_kwargs: PreprocessParams = {}
        if 'tokenizer' in kwargs:
            preprocess_kwargs['maybe_arg'] = kwargs['maybe_arg']

        return preprocess_kwargs, {}, {}

    @override
    def preprocess(
        self, input_: StackedAudio[AudioData], **preprocessor_params: PreprocessParams
    ) -> StackedAudio[torch.Tensor]:
        audio = input_.inputs

        features = log_mel_spectrogram(audio, padding=N_SAMPLES - audio.shape[0])

        return StackedAudio(inputs=features)

    @override
    def _forward(
        self, input_tensors: StackedAudio[torch.Tensor]
    ) -> BatchingWhisperOutput:
        if self._tokenizer is None:
            raise ValueError(
                'Could not do forward, the tokenizer on BatchingWhisperModel has to be defined'
            )

        outputs = self.model.generate_segment_batched(
            input_tensors.inputs.numpy(), self._tokenizer, self.options
        )

        return BatchingWhisperOutput(text=outputs)

    @override
    def postprocess(
        self,
        model_outputs: BatchingWhisperOutput,
        **postprocess_params: PostprocessParams,
    ) -> BatchingWhisperOutput:
        return model_outputs

    def get_iterator(
        self,
        inputs: torch.utils.data.DataLoader[torch.Tensor] | Iterable[torch.Tensor],
        num_workers: int,
        batch_size: int,
        preprocess_params: PreprocessParams,
        forward_params: ForwardParams,
        postprocess_params: PostprocessParams,
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)

        if 'TOKENIZERS_PARALLELISM' not in os.environ:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # TODO: hack by collating feature_extractor and image_processor

        def stack(
            items: Iterable[StackedAudio[torch.Tensor]],
        ) -> StackedAudio[torch.Tensor]:
            return StackedAudio(inputs=torch.stack([x.inputs for x in items]))

        dataloader = torch.utils.data.DataLoader[torch.Tensor](
            dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack
        )

        model_iterator = PipelineIterator(
            dataloader, self._forward, forward_params, loader_batch_size=batch_size
        )

        final_iterator = PipelineIterator(
            model_iterator, self.postprocess, postprocess_params
        )

        return final_iterator

    def __call__(
        self,
        inputs: Iterable[StackedAudio[AudioData]],
        *args: Any,
        num_workers: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[BatchingWhisperOutput]:
        result = cast(
            list[BatchingWhisperOutput],
            super().__call__(  # pyright: ignore [reportUnknownMemberType]
                inputs, *args, num_workers=num_workers, batch_size=batch_size, **kwargs
            ),
        )

        return result

    def _get_or_infer_language_details(
        self, features: AudioData | Features, given_language: LanguageCode | None = None
    ) -> LanguageDetails:
        language: LanguageCode
        if given_language is None:
            if self.preset_language is not None:
                return LanguageDetails.get_defined(self.preset_language)

            if not self.model.model.is_multilingual:
                return LanguageDetails(
                    language='en', language_probability=1, all_language_probs=None
                )
            else:
                return self.model.detect_language(features)
        else:
            if not self.model.model.is_multilingual and given_language != 'en':
                self.logger.warning(
                    "The current model is English-only but the language parameter is set to '%s'; "
                    "using 'en' instead." % given_language
                )

                language = 'en'
            else:
                language = given_language

            return LanguageDetails.get_defined(language)

    @property
    def tokenizer(self) -> faster_whisper.tokenizer.Tokenizer | None:
        return self._tokenizer

    def _get_tokenizer(
        self,
        for_language: LanguageCode | None,
        for_task: TaskType = 'transcribe',
        for_audio: AudioData | None = None,
    ) -> tuple[LanguageDetails, faster_whisper.tokenizer.Tokenizer]:
        if (
            for_language is None and self.preset_language is None
        ) and for_audio is not None:
            language_details = self.model.detect_language(for_audio)
        elif for_audio is not None:
            language_details = self._get_or_infer_language_details(
                for_audio, for_language
            )
        else:
            raise ValueError(
                f'Invalid configuration, atleast one param: {for_language = } or {for_audio = } has to be not None'
            )

        if is_tokenizer_wrong_or_none(
            self._tokenizer,
            self.model.model.is_multilingual,
            language_details.language,
            for_task,
        ):
            new_tokenizer = faster_whisper.tokenizer.Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=for_task,
                language=language_details.language,
            )

            self._tokenizer = new_tokenizer

            return language_details, new_tokenizer

        # * Will be not None as it should have been recreated and returned above. I am not sure if it's possible to make the type checker infer that
        # * the self._tokenizer is certainly not None. As far as I am concerned, it's not really possible to with TypeGuards or any means that I am aware of

        return language_details, cast(
            faster_whisper.tokenizer.Tokenizer, self._tokenizer
        )

    def transcribe(
        self,
        audio: StrPath | AudioData,
        batch_size: int | None = None,
        num_workers: int = 0,
        language: LanguageCode | None = None,
        task: TaskType = 'transcribe',
        chunk_size: int = 30,
    ) -> TranscriptionResult:
        if isinstance(audio, os.PathLike | Path | str):
            loaded_audio = load_audio(audio)
        else:
            loaded_audio = audio

        audio_duration = loaded_audio.shape[0] / SAMPLE_RATE
        self.logger.info(f'Running VAD on audio, {audio_duration = }s')

        vad_segments = self.vad_model(
            {
                'waveform': torch.from_numpy(loaded_audio).unsqueeze(0),  # pyright: ignore [reportUnknownMemberType]
                'sample_rate': SAMPLE_RATE,
            }
        )

        vad_segments = merge_chunks(vad_segments, chunk_size)
        self.logger.info(
            f'Split audio into {len(vad_segments)} chunk(s) (around {chunk_size} second(s) in length)'
        )

        language_details, tokenizer = self._get_tokenizer(language, task, loaded_audio)

        previous_suppress_tokens = None
        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(tokenizer)

            self.logger.info(
                f'Suppressing numeral and symbol tokens: {numeral_symbol_tokens}'
            )

            if self.options.suppress_tokens is None:
                raise ValueError('Options supress tokens is None but it should not.')

            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options.suppress_tokens = new_suppressed_tokens

        segments: list[Segment] = []
        batch_size = batch_size or self._batch_size

        def slice_data_by_vad_segments(
            audio: AudioData, segments: list[SegmentsBoundsMerge]
        ) -> Iterable[StackedAudio[AudioData]]:
            for seg in segments:
                f1 = int(seg.start * SAMPLE_RATE)
                f2 = int(seg.end * SAMPLE_RATE)

                yield StackedAudio(inputs=audio[f1:f2])

        for idx, out in tqdm(
            enumerate(
                self.__call__(
                    slice_data_by_vad_segments(loaded_audio, vad_segments),
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            ),
            desc='Performing transription',
            total=len(vad_segments),
        ):
            if batch_size in [0, 1, None]:
                text = out.text[0]
            else:
                text = cast(str, out.text)

            segments.append(
                Segment(
                    text=text,
                    start=round(vad_segments[idx].start, 3),
                    end=round(vad_segments[idx].end, 3),
                )
            )

        # revert suppressed tokens if suppress_numerals is enabled
        if self.suppress_numerals:
            self.options.suppress_tokens = previous_suppress_tokens

        return TranscriptionResult(
            segments=segments,
            language=language_details.language,
            language_details=language_details,
        )
