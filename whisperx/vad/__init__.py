from __future__ import annotations

import os
import typing
from dataclasses import asdict
from typing import cast

import torch
from pyannote.audio.core.model import Model
from pyannote.core.feature import SlidingWindowFeature

from whisperx.diarize import SegmentDiarized, SpeakerId
from whisperx.logging import get_logger
from whisperx.types import DeviceType
from whisperx.utils import get_device
from whisperx.utils.convert_path import convert_path
from whisperx.vad.binarize import Binarize
from whisperx.vad.download_model import download_model
from whisperx.vad.types import SegmentId, SegmentsBoundsMerge
from whisperx.vad.vad_pipeline import VadOptions, VoiceActivityDetectionPipeline

if typing.TYPE_CHECKING:
    from _typeshed import StrPath

logger = get_logger(__name__)


def load_vad_model(
    device: DeviceType | int | torch.device,
    vad_options: VadOptions,
    use_auth_token: str | None = None,
    model_dir: StrPath | None = None,
) -> VoiceActivityDetectionPipeline:
    model_dir = convert_path(model_dir)

    if model_dir is None or not os.path.isdir(model_dir):
        model_dir = cast(str, torch.hub.get_dir())

    model_fp = download_model(model_dir)
    if model_fp is None:
        raise ValueError(
            'Could not download the vad model, please check the configured download links'
        )

    device = get_device(device)

    vad_model = Model.from_pretrained(
        convert_path(model_fp), use_auth_token=use_auth_token
    )

    hyperparameters = {
        'min_duration_on': 0.1,
        'min_duration_off': 0.1,
        **asdict(vad_options),
    }

    vad_pipeline = VoiceActivityDetectionPipeline(
        segmentation=vad_model, device=torch.device(device)
    )

    vad_pipeline.instantiate(hyperparameters)

    return vad_pipeline


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
        logger.info('No active speech found in audio')
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
