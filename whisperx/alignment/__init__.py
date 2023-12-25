"""
Forced Alignment with Whisper
C. Max Bain
"""

from __future__ import annotations

import os
import typing
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias, cast

import dacite
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torchaudio
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
from pandas._typing import InterpolateOptions
from tqdm import tqdm
from transformers import PreTrainedTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

from whisperx.alignment.constants import (
    DEFAULT_ALIGN_MODELS_HF,
    DEFAULT_ALIGN_MODELS_TORCH,
    LANGUAGES_WITHOUT_SPACES,
    PUNKT_ABBREVIATIONS,
    AlignLanguageCode,
)
from whisperx.alignment.types import (
    AlignedTranscriptionResult,
    PreprocessResult,
    SingleAlignedSegment,
    SingleCharSegment,
    SingleWordSegment,
)
from whisperx.alignment.utils import backtrack, get_trellis, merge_repeats
from whisperx.asr.batching_whisper_pipeline import Segment
from whisperx.audio import SAMPLE_RATE, AudioData, load_audio
from whisperx.logging import get_logger
from whisperx.types import DeviceType
from whisperx.utils import get_device, interpolate_nans

if typing.TYPE_CHECKING:
    from _typeshed import StrPath

DACITE_CONVERSION_CONFIG = dacite.Config(check_types=False)

logger = get_logger(__name__)


class TypedWav2Vec2Processor(Wav2Vec2Processor):
    tokenizer: PreTrainedTokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)  # pyright: ignore[reportUnknownMemberType]


PipelineType: TypeAlias = Literal['torchaudio', 'huggingface']
Dictionary: TypeAlias = dict[str, int]


class AlignModelWrapper:
    def __init__(
        self,
        model: Wav2Vec2ForCTC | torch.nn.Module,
        device_type: DeviceType | torch.device,
        language: AlignLanguageCode,
        pipeline_type: PipelineType,
        dictionary: Dictionary,
    ) -> None:
        self.device = get_device(device_type)

        self.model = model.to(self.device)  # pyright: ignore

        self.language = language
        self.pipeline_type = pipeline_type
        self.dictionary = dictionary

        self._is_language_with_spaces = self.language not in LANGUAGES_WITHOUT_SPACES

        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        self.sentence_splitter = PunktSentenceTokenizer(punkt_param)

        self.logger = get_logger(__name__)

    @classmethod
    def load(
        cls,
        language_code: AlignLanguageCode,
        device: DeviceType | torch.device = 'auto',
        model_name_or_path: str | None = None,
        model_dir: StrPath | None = None,
    ) -> AlignModelWrapper:
        if model_name_or_path is None:
            # use default model
            if language_code in DEFAULT_ALIGN_MODELS_TORCH:
                model_name_or_path = DEFAULT_ALIGN_MODELS_TORCH[language_code]
            elif language_code in DEFAULT_ALIGN_MODELS_HF:
                model_name_or_path = DEFAULT_ALIGN_MODELS_HF[language_code]
            else:
                logger.error(
                    f'There is no default alignment model set for this language ({language_code}).\
                    Please find a wav2vec2.0 model fine-tuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]'
                )
                raise ValueError(
                    f'No default align-model for language: {language_code}'
                )

        pipeline_type: PipelineType
        if model_name_or_path in torchaudio.pipelines.__all__:
            pipeline_type = 'torchaudio'
            bundle = torchaudio.pipelines.__dict__[model_name_or_path]
            model = bundle.get_model(dl_kwargs={'model_dir': model_dir})
            labels = bundle.get_labels()
            dictionary = {c.lower(): i for i, c in enumerate(labels)}
        else:
            try:
                processor = TypedWav2Vec2Processor.from_pretrained(model_name_or_path)
                model = cast(
                    Wav2Vec2ForCTC, Wav2Vec2ForCTC.from_pretrained(model_name_or_path)
                )
            except Exception as e:
                logger.error(e)
                logger.error(
                    'Error loading model from huggingface, check https://huggingface.co/models for fine-tuned wav2vec2.0 models'
                )
                raise ValueError(
                    f'The chosen align_model "{model_name_or_path}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)'
                )

            pipeline_type = 'huggingface'

            labels = processor.tokenizer.get_vocab()
            dictionary = {char.lower(): code for char, code in labels.items()}

        return cls(model, device, language_code, pipeline_type, dictionary)

    def _preprocess_segments(
        self, segments: Sequence[Segment]
    ) -> list[PreprocessResult]:
        self.logger.debug(f'Starting segments preprocessing, {len(segments) = }')

        results: list[PreprocessResult] = []

        for segment in tqdm(
            segments, desc='Preprocessing segments', total=len(segments)
        ):
            text = segment.text
            num_leading = len(text) - len(text.lstrip())
            num_trailing = len(text) - len(text.rstrip())

            # split into words
            if self._is_language_with_spaces:
                per_word = text.split(' ')
            else:
                per_word = text

            clean_char: list[str] = []
            clean_cdx: list[int] = []
            for cdx, char in enumerate(text):
                char_ = char.lower()
                # wav2vec2 models use "|" character to represent spaces
                if self._is_language_with_spaces:
                    char_ = char_.replace(' ', '|')

                # ignore whitespace at beginning and end of transcript
                if cdx < num_leading:
                    pass
                elif cdx > len(text) - num_trailing - 1:
                    pass
                elif char_ in self.dictionary:
                    clean_char.append(char_)
                    clean_cdx.append(cdx)

            clean_wdx: list[int] = []
            for wdx, wrd in enumerate(per_word):
                if any([c in self.dictionary for c in wrd]):
                    clean_wdx.append(wdx)

            sentence_spans = list(self.sentence_splitter.span_tokenize(text))

            result = PreprocessResult(
                clean_char=clean_char,
                clean_cdx=clean_cdx,
                clean_wdx=clean_wdx,
                sentence_spans=sentence_spans,
            )

            results.append(result)

        return results

    def align(
        self,
        segments: Sequence[Segment],
        audio: StrPath | AudioData | torch.Tensor,
        interpolate_method: InterpolateOptions = 'nearest',
        return_char_alignments: bool = False,
    ) -> AlignedTranscriptionResult:
        """
        Align phoneme recognition predictions to known transcription.
        """

        if not isinstance(audio, torch.Tensor):
            if isinstance(audio, os.PathLike | Path | str):
                audio = load_audio(audio)

            audio = torch.from_numpy(audio)  # pyright: ignore[reportUnknownMemberType]

        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)

        max_duration = audio.shape[1] / SAMPLE_RATE

        model_dictionary = self.dictionary

        # 1. Preprocess to keep only characters in dictionary
        preprocess_results = self._preprocess_segments(segments)
        aligned_segments: list[SingleAlignedSegment] = []

        # 2. Get prediction matrix from alignment model & align
        for segment, preprocess in tqdm(
            zip(segments, preprocess_results),
            desc='Aligning segments',
            total=len(segments),
        ):
            t1 = segment.start
            t2 = segment.end
            text = segment.text

            aligned = SingleAlignedSegment(start=t1, end=t2, text=text, words=[])

            if return_char_alignments:
                aligned.chars = []

            # check we can align
            if len(preprocess.clean_char) == 0:
                self.logger.info(
                    'Failed to align segment ("text"): no characters in this segment found in model dictionary, resorting to original...'
                )
                aligned_segments.append(aligned)
                continue

            if t1 >= max_duration or t2 - t1 < 0.02:
                self.logger.info(
                    'Failed to align segment: original start time longer than audio duration, skipping...'
                )
                aligned_segments.append(aligned)
                continue

            text_clean = ''.join(preprocess.clean_char)
            tokens: npt.NDArray[np.int64] = np.array(
                [model_dictionary[c] for c in text_clean], dtype=np.int64
            )

            f1 = int(t1 * SAMPLE_RATE)
            f2 = int(t2 * SAMPLE_RATE)

            # TODO: Probably can get some speedup gain with batched inference here
            waveform_segment = audio[:, f1:f2]

            with torch.inference_mode():
                if self.pipeline_type == 'torchaudio':
                    emissions, _ = self.model(waveform_segment.to(self.device))
                elif self.pipeline_type == 'huggingface':
                    emissions = self.model(waveform_segment.to(self.device)).logits
                else:
                    raise NotImplementedError(
                        f'Align model of type {self.pipeline_type} not supported.'
                    )
                emissions = torch.log_softmax(emissions, dim=-1)

            emission = emissions[0].cpu().detach()

            blank_id = 0
            for char, code in self.dictionary.items():
                if char == '[pad]' or char == '<pad>':
                    blank_id = code

            trellis = get_trellis(emission, tokens, blank_id)
            path = backtrack(trellis, emission, tokens, blank_id)

            if path is None:
                self.logger.info(
                    f'Failed to align segment ("{text}"): backtrack failed, resorting to original...'
                )
                aligned_segments.append(aligned)
                continue

            no_repeat_aligned_segments = merge_repeats(path, text_clean)

            duration = t2 - t1
            ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

            # assign timestamps to aligned characters
            char_segments: list[SingleCharSegment] = []
            word_idx = 0
            for cdx, char in enumerate(text):
                start, end, score = None, None, None
                if cdx in preprocess.clean_cdx:
                    char_seg = no_repeat_aligned_segments[
                        preprocess.clean_cdx.index(cdx)
                    ]

                    start = round(char_seg.start * ratio + t1, 3)
                    end = round(char_seg.end * ratio + t1, 3)
                    score = round(char_seg.score, 3)

                char_segment: SingleCharSegment = SingleCharSegment(
                    char=char,
                    start=start,
                    end=end,
                    score=score,
                    word_idx=word_idx,
                )

                char_segments.append(char_segment)

                # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
                if self.language in LANGUAGES_WITHOUT_SPACES:
                    word_idx += 1
                elif cdx == len(text) - 1 or text[cdx + 1] == ' ':
                    word_idx += 1

            char_segments_df = pd.DataFrame(char_segments)

            aligned_subsegments: list[SingleAlignedSegment] = []
            # assign sentence_idx to each character index
            char_segments_df['sentence-idx'] = None
            for sdx, (sent_start, sent_end) in enumerate(preprocess.sentence_spans):
                select_range = cast(
                    pd.Series,
                    (char_segments_df.index >= sent_start)
                    & (char_segments_df.index <= sent_end),
                )

                curr_chars = cast(Any, char_segments_df.loc[select_range])
                char_segments_df.loc[
                    select_range,
                    'sentence-idx',
                ] = sdx

                sentence_text = text[sent_start:sent_end]
                sentence_start = curr_chars['start'].min()
                sentence_end = curr_chars['end'].max()
                sentence_words: list[SingleWordSegment] = []

                for word_idx in curr_chars['word_idx'].unique():
                    word_chars = curr_chars.loc[curr_chars['word_idx'] == word_idx]
                    word_text = ''.join(word_chars['char'].tolist()).strip()
                    if len(word_text) == 0:
                        continue

                    # Don't use space character for alignment
                    word_chars = word_chars[word_chars['char'] != ' ']

                    word_start = word_chars['start'].min()
                    word_end = word_chars['end'].max()
                    word_score = round(word_chars['score'].mean(), 3)

                    if (
                        np.isnan(word_start)
                        or np.isnan(word_end)
                        or np.isnan(word_score)
                    ):
                        logger.error(
                            "Some of the values are NaN for this word's segment, skipping it"
                        )
                        continue

                    # NOTE: Values can be NaN for some reason?
                    word_segment = SingleWordSegment(
                        word=word_text, start=word_start, end=word_end, score=word_score
                    )
                    sentence_words.append(word_segment)

                sentence_segment = SingleAlignedSegment(
                    text=sentence_text,
                    start=sentence_start,
                    end=sentence_end,
                    words=sentence_words,
                )

                aligned_subsegments.append(sentence_segment)

                if return_char_alignments:
                    curr_chars = curr_chars[['char', 'start', 'end', 'score']]
                    curr_chars.fillna(-1, inplace=True)
                    curr_chars = curr_chars.to_dict('records')
                    curr_chars = [
                        {key: val for key, val in char.items() if val != -1}
                        for char in curr_chars
                    ]

                    chars = [
                        dacite.from_dict(
                            SingleCharSegment, data, DACITE_CONVERSION_CONFIG
                        )
                        for data in curr_chars
                    ]

                    aligned_subsegments[-1].chars = chars

            aligned_subsegments_df = pd.DataFrame(aligned_subsegments)
            del aligned_subsegments

            aligned_subsegments_df['start'] = interpolate_nans(
                cast(pd.Series, aligned_subsegments_df['start']),
                method=interpolate_method,
            )
            aligned_subsegments_df['end'] = interpolate_nans(
                cast(pd.Series, aligned_subsegments_df['end']),
                method=interpolate_method,
            )

            # concatenate sentences with same timestamps
            agg_dict = {'text': ' '.join, 'words': 'sum'}
            if self.language in LANGUAGES_WITHOUT_SPACES:
                agg_dict['text'] = ''.join
            if return_char_alignments:
                agg_dict['chars'] = 'sum'

            aligned_subsegments = [
                dacite.from_dict(SingleAlignedSegment, data, DACITE_CONVERSION_CONFIG)
                for data in cast(
                    list[dict[str, Any]],
                    aligned_subsegments_df.to_dict('records'),  # pyright: ignore[reportUnknownMemberType]
                )
            ]

            aligned_segments += aligned_subsegments

        # create word_segments list
        word_segments: list[SingleWordSegment] = []
        for segment in aligned_segments:
            word_segments += segment.words

        return AlignedTranscriptionResult(
            segments=aligned_segments, word_segments=word_segments
        )
