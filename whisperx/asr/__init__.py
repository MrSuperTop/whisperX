from __future__ import annotations

import typing

import faster_whisper
import faster_whisper.tokenizer
import faster_whisper.transcribe

from whisperx.asr.batching_whisper import AsrOptions
from whisperx.asr.batching_whisper import BatchingWhisperModel as BatchingWhisperModel
from whisperx.asr.batching_whisper_pipeline import BatchingWhisperPipeline
from whisperx.types import (
    ComputeType,
    DeviceType,
    LanguageCode,
    ModelArchiveOrSize,
    TaskType,
)
from whisperx.utils.convert_path import convert_path
from whisperx.vad import VadOptions, load_vad_model

if typing.TYPE_CHECKING:
    from _typeshed import StrPath

# TODO: Logging
# TODO: Check for compute type compatibility, inform the user which compute types are compatitble with their hardware


# TODO: https://opennmt.net/CTranslate2/python/ctranslate2.converters.TransformersConverter.html
def load_model(
    whisper_arch_or_path: ModelArchiveOrSize | StrPath,
    device: DeviceType = 'auto',
    device_index: int | list[int] = 0,
    compute_type: ComputeType = 'float16',
    language: LanguageCode | None = None,
    asr_options: AsrOptions = AsrOptions(),
    vad_options: VadOptions = VadOptions(),
    task: TaskType = 'transcribe',
    use_auth_token: str | None = None,
    download_root: StrPath | None = None,
    threads: int = 4,
) -> BatchingWhisperPipeline:
    # TODO: Proper docstring
    """Load a Whisper model for inference.
    Args:
        whisper_arch_or_path: str | Path - The name of the Whisper model to load or the path to the
            local model checkpoint transformed with ctranslate2.
        device: DeviceType - The device to load the model on.
        compute_type: str - The compute type to use for the model.
        options: dict - A dictionary of options to use for the model.
        language: str - The language of the model. (use English for now)
        download_root: Optional[str] - The root directory to download the model to.
        threads: int - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    """

    if isinstance(whisper_arch_or_path, str) and whisper_arch_or_path.endswith('.en'):
        language = 'en'

    whisper_arch_or_path = convert_path(whisper_arch_or_path)

    if download_root is not None:
        download_root = convert_path(download_root)

    model = BatchingWhisperModel(
        whisper_arch_or_path,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        download_root=download_root,
        cpu_threads=threads,
        num_workers=threads,
    )

    if language is not None:
        tokenizer = faster_whisper.tokenizer.Tokenizer(
            model.hf_tokenizer,
            model.model.is_multilingual,
            task=task,
            language=language,
        )
    else:
        print(
            'No language specified, language will be first be detected for each audio file (increases inference time).'
        )
        tokenizer = None

    vad_model = load_vad_model(
        device, vad_options, use_auth_token=use_auth_token, model_dir=download_root
    )

    return BatchingWhisperPipeline(
        model=model,
        vad_model=vad_model,
        options=asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=asr_options.suppress_numerals,
    )
