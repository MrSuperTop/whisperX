"""
Forced Alignment with Whisper
C. Max Bain
"""

from __future__ import annotations

import os
import typing
from dataclasses import dataclass
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
from transformers import PreTrainedTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

from whisperx.alignment.constants import (
    DEFAULT_ALIGN_MODELS_HF,
    DEFAULT_ALIGN_MODELS_TORCH,
    LANGUAGES_WITHOUT_SPACES,
    PUNKT_ABBREVIATIONS,
    AlignLanguageCode,
)
from whisperx.asr.faster_whisper_pipeline import Segment
from whisperx.audio import SAMPLE_RATE, AudioData, load_audio
from whisperx.logging import get_logger
from whisperx.types import DeviceType, SingleCharSegment, SingleWordSegment
from whisperx.utils import get_device, interpolate_nans

if typing.TYPE_CHECKING:
    from _typeshed import StrPath

logger = get_logger(__name__)
DACITE_CONVERTION_CONFIG = dacite.Config(check_types=False)


class TypedWav2Vec2Processor(Wav2Vec2Processor):
    tokenizer: PreTrainedTokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)  # pyright: ignore[reportUnknownMemberType]


PipelineType: TypeAlias = Literal["torchaudio", "huggingface"]
Dictionary: TypeAlias = dict[str, int]


@dataclass(slots=True)
class PreprocessResult:
    clean_char: list[str]
    clean_cdx: list[int]
    clean_wdx: list[int]
    sentence_spans: list[tuple[int, int]]


@dataclass(slots=True)
class SingleAlignedSegment:
    """
    A single segment (up to multiple sentences) of a speech with word alignment.
    """

    start: float
    end: float
    text: str
    words: list[SingleWordSegment]
    chars: list[SingleCharSegment] | None = None


@dataclass(slots=True)
class AlignedTranscriptionResult:
    """
    A list of segments and word segments of a speech.
    """

    segments: list[SingleAlignedSegment]
    word_segments: list[SingleWordSegment]


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
        device: DeviceType | torch.device = "auto",
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
                    f"There is no default alignment model set for this language ({language_code}).\
                    Please find a wav2vec2.0 model finetuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]"
                )
                raise ValueError(
                    f"No default align-model for language: {language_code}"
                )

        pipeline_type: PipelineType
        if model_name_or_path in torchaudio.pipelines.__all__:
            pipeline_type = "torchaudio"
            bundle = torchaudio.pipelines.__dict__[model_name_or_path]
            model = bundle.get_model(dl_kwargs={"model_dir": model_dir})
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
                    "Error loading model from huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models"
                )
                raise ValueError(
                    f'The chosen align_model "{model_name_or_path}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)'
                )

            pipeline_type = "huggingface"

            labels = processor.tokenizer.get_vocab()
            dictionary = {char.lower(): code for char, code in labels.items()}

        return cls(model, device, language_code, pipeline_type, dictionary)

    def _preprocess_segments(
        self, segments: Sequence[Segment]
    ) -> list[PreprocessResult]:
        results: list[PreprocessResult] = []

        for segment in segments:
            text = segment.text
            num_leading = len(text) - len(text.lstrip())
            num_trailing = len(text) - len(text.rstrip())

            # split into words
            if self._is_language_with_spaces:
                per_word = text.split(" ")
            else:
                per_word = text

            clean_char: list[str] = []
            clean_cdx: list[int] = []
            for cdx, char in enumerate(text):
                char_ = char.lower()
                # wav2vec2 models use "|" character to represent spaces
                if self._is_language_with_spaces:
                    char_ = char_.replace(" ", "|")

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
        interpolate_method: InterpolateOptions = "nearest",
        return_char_alignments: bool = False,
        # FIXME: Implement???
        print_progress: bool = False,
        combined_progress: bool = False,
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

        # TODO: Imeplement process printing???
        # if print_progress:
        #     base_progress = ((sdx + 1) / total_segments) * 100
        #     percent_complete = (
        #         (50 + base_progress / 2) if combined_progress else base_progress
        #     )
        #     print(f"Progress: {percent_complete:.2f}%...")

        # 1. Preprocess to keep only characters in dictionary
        preprocess_results = self._preprocess_segments(segments)
        aligned_segments: list[SingleAlignedSegment] = []

        # 2. Get prediction matrix from alignment model & align
        for sdx, (segment, preprocess) in enumerate(zip(segments, preprocess_results)):
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
                    "Failed to align segment: original start time longer than audio duration, skipping..."
                )
                aligned_segments.append(aligned)
                continue

            text_clean = "".join(preprocess.clean_char)
            tokens: npt.NDArray[np.int64] = np.array(
                [model_dictionary[c] for c in text_clean], dtype=np.int64
            )

            f1 = int(t1 * SAMPLE_RATE)
            f2 = int(t2 * SAMPLE_RATE)

            # TODO: Probably can get some speedup gain with batched inference here
            waveform_segment = audio[:, f1:f2]

            with torch.inference_mode():
                if self.pipeline_type == "torchaudio":
                    emissions, _ = self.model(waveform_segment.to(self.device))
                elif self.pipeline_type == "huggingface":
                    emissions = self.model(waveform_segment.to(self.device)).logits
                else:
                    raise NotImplementedError(
                        f"Align model of type {self.pipeline_type} not supported."
                    )
                emissions = torch.log_softmax(emissions, dim=-1)

            emission = emissions[0].cpu().detach()

            blank_id = 0
            for char, code in self.dictionary.items():
                if char == "[pad]" or char == "<pad>":
                    blank_id = code

            trellis = get_trellis(emission, tokens, blank_id)
            path = backtrack(trellis, emission, tokens, blank_id)

            if path is None:
                logger.info(
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
                elif cdx == len(text) - 1 or text[cdx + 1] == " ":
                    word_idx += 1

            char_segments_df = pd.DataFrame(char_segments)

            aligned_subsegments: list[SingleAlignedSegment] = []
            # assign sentence_idx to each character index
            char_segments_df["sentence-idx"] = None
            for sdx, (sstart, send) in enumerate(preprocess.sentence_spans):
                select_range = cast(
                    pd.Series,
                    (char_segments_df.index >= sstart)
                    & (char_segments_df.index <= send),
                )

                curr_chars = cast(Any, char_segments_df.loc[select_range])
                char_segments_df.loc[
                    select_range,
                    "sentence-idx",
                ] = sdx

                sentence_text = text[sstart:send]
                sentence_start = curr_chars["start"].min()
                sentence_end = curr_chars["end"].max()
                sentence_words: list[SingleWordSegment] = []

                for word_idx in curr_chars["word_idx"].unique():
                    word_chars = curr_chars.loc[curr_chars["word_idx"] == word_idx]
                    word_text = "".join(word_chars["char"].tolist()).strip()
                    if len(word_text) == 0:
                        continue

                    # dont use space character for alignment
                    word_chars = word_chars[word_chars["char"] != " "]

                    word_start = word_chars["start"].min()
                    word_end = word_chars["end"].max()
                    word_score = round(word_chars["score"].mean(), 3)

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
                    curr_chars = curr_chars[["char", "start", "end", "score"]]
                    curr_chars.fillna(-1, inplace=True)
                    curr_chars = curr_chars.to_dict("records")
                    curr_chars = [
                        {key: val for key, val in char.items() if val != -1}
                        for char in curr_chars
                    ]

                    chars = [
                        dacite.from_dict(
                            SingleCharSegment, data, DACITE_CONVERTION_CONFIG
                        )
                        for data in curr_chars
                    ]

                    aligned_subsegments[-1].chars = chars

            aligned_subsegments_df = pd.DataFrame(aligned_subsegments)
            del aligned_subsegments

            aligned_subsegments_df["start"] = interpolate_nans(
                cast(pd.Series, aligned_subsegments_df["start"]),
                method=interpolate_method,
            )
            aligned_subsegments_df["end"] = interpolate_nans(
                cast(pd.Series, aligned_subsegments_df["end"]),
                method=interpolate_method,
            )

            # concatenate sentences with same timestamps
            agg_dict = {"text": " ".join, "words": "sum"}
            if self.language in LANGUAGES_WITHOUT_SPACES:
                agg_dict["text"] = "".join
            if return_char_alignments:
                agg_dict["chars"] = "sum"

            aligned_subsegments = [
                dacite.from_dict(SingleAlignedSegment, data, DACITE_CONVERTION_CONFIG)
                for data in cast(
                    list[dict[str, Any]],
                    aligned_subsegments_df.to_dict("records"),  # pyright: ignore[reportUnknownMemberType]
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


"""
source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
"""


def get_trellis(
    emission: torch.Tensor, tokens: npt.NDArray[np.int64], blank_id: int = 0
):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


@dataclass(slots=True)
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(
    trellis: torch.Tensor,
    emission: torch.Tensor,
    tokens: npt.NDArray[np.int64],
    blank_id: int = 0,
) -> list[Point] | None:
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = int(torch.argmax(trellis[:, j]).item())

    path: list[Point] = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        logger.error("Backtracking failed")
        # failed
        return None
    return path[::-1]


# Merge the labels
@dataclass
class AlignedSegment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path: list[Point], transcript: str) -> list[AlignedSegment]:
    i1, i2 = 0, 0
    segments: list[AlignedSegment] = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            AlignedSegment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2

    return segments
