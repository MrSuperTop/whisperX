from __future__ import annotations

import hashlib
import os
import typing
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, cast

import pandas as pd
import torch
from pyannote.audio.core.model import Model
from pyannote.core import Annotation, Segment
from pyannote.core.feature import SlidingWindowFeature
from tqdm import tqdm

from whisperx.diarize import SegmentDiarized, SpeakerId
from whisperx.types import DeviceType
from whisperx.utils import get_device
from whisperx.utils.convert_path import convert_path
from whisperx.vad.binarize import Binarize, remove_shorter_than
from whisperx.vad.vad_model import VadOptions, VoiceActivityDetectionPipeline

if typing.TYPE_CHECKING:
    from _typeshed import StrPath

VAD_SEGMENTATION_URL = "https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"


def _download_model(from_url: str, model_dir: StrPath) -> StrPath:
    os.makedirs(model_dir, exist_ok=True)

    model_fp = os.path.join(model_dir, "whisperx-vad-segmentation.bin")
    if os.path.exists(model_fp) and not os.path.isfile(model_fp):
        raise RuntimeError(f"{model_fp} exists and is not a regular file")

    if not os.path.isfile(model_fp):
        with urllib.request.urlopen(from_url) as source, open(model_fp, "wb") as output:
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

    model_bytes = open(model_fp, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != from_url.split("/")[-2]:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model."
        )

    return model_fp


def load_vad_model(
    device: DeviceType | int | torch.device,
    vad_options: VadOptions,
    use_auth_token: str | None = None,
    model_dir: StrPath | None = None,
) -> VoiceActivityDetectionPipeline:
    model_dir = convert_path(model_dir)

    if model_dir is None or not os.path.isdir(model_dir):
        model_dir = cast(str, torch.hub.get_dir())

    model_fp = _download_model(VAD_SEGMENTATION_URL, model_dir)
    device = get_device(device)

    vad_model = Model.from_pretrained(
        convert_path(model_fp), use_auth_token=use_auth_token
    )

    hyperparameters = {
        "min_duration_on": 0.1,
        "min_duration_off": 0.1,
        **asdict(vad_options),
    }

    vad_pipeline = VoiceActivityDetectionPipeline(
        segmentation=vad_model, device=torch.device(device)
    )

    vad_pipeline.instantiate(hyperparameters)

    return vad_pipeline


def merge_vad(
    vad_arr: Iterable[Any],
    pad_onset: float = 0.0,
    pad_offset: float = 0.0,
    min_duration_off: float = 0.0,
    min_duration_on: float = 0.0,
) -> pd.DataFrame:
    active = Annotation()
    for k, vad_t in enumerate(vad_arr):
        region = Segment(vad_t[0] - pad_onset, vad_t[1] + pad_offset)
        active[region, k] = 1

    if pad_offset > 0.0 or pad_onset > 0.0 or min_duration_off > 0.0:
        active = active.support(collar=min_duration_off)

    remove_shorter_than(min_duration_on, active)

    active_json = active.for_json()
    active_segs = pd.DataFrame([x["segment"] for x in active_json["content"]])
    return active_segs


SegmentId = tuple[float, float]


# FIXME: Get rid of the TypedDict, when it will be clear how does the whole project structure go.
@dataclass(frozen=True)
class SegmentsBoundsMerge:
    start: float
    end: float
    segments: list[SegmentId]


def merge_chunks(
    segments: SlidingWindowFeature, chunk_size: int
) -> list[SegmentsBoundsMerge]:
    """
    Merge operation described in paper
    """

    curr_end = 0
    merged_segments: list[SegmentsBoundsMerge] = []
    slice_segements_ids = []
    speaker_idxs = []

    assert chunk_size > 0

    binarized = Binarize(max_duration=chunk_size)
    binarized_segments = binarized(segments)

    segments_list: list[SegmentDiarized] = []
    for speech_turn in iter(binarized_segments.get_timeline()):
        segments_list.append(SegmentDiarized.plain_from_segment(speech_turn))

    if len(segments_list) == 0:
        print("No active speech found in audio")
        return []

    # assert segments_list, "segments_list is empty."
    # Make sur the starting point is the start of the segment.
    curr_start = segments_list[0].start

    for seg in segments_list:
        if seg.end - curr_start > chunk_size and curr_end - curr_start > 0:
            merged_segments.append(
                SegmentsBoundsMerge(
                    start=curr_start,
                    end=curr_end,
                    segments=slice_segements_ids,
                )
            )

            curr_start = seg.start
            slice_segements_ids: list[SegmentId] = []
            speaker_idxs: list[SpeakerId] = []

        curr_end = seg.end
        slice_segements_ids.append((seg.start, seg.end))
        speaker_idxs.append(seg.speaker)

    # add final
    merged_segments.append(
        SegmentsBoundsMerge(
            start=curr_start,
            end=curr_end,
            segments=slice_segements_ids,
        )
    )

    return merged_segments
