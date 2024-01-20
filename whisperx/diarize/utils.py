# pyright: basic

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from whisperx.alignment.types import AlignedTranscriptionResult
from whisperx.diarize.types import (
    DiarizedTranscriptionResult,
    SingleDiarizedSegment,
    SingleDiarizedWordSegment,
)


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: AlignedTranscriptionResult,
    fill_nearest: bool = False,
) -> DiarizedTranscriptionResult:
    transcript_segments = transcript_result.segments
    new_segments: list[SingleDiarizedSegment] = []

    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(
            diarize_df['end'], seg.end
        ) - np.maximum(diarize_df['start'], seg.start)
        diarize_df['union'] = np.maximum(diarize_df['end'], seg.end) - np.minimum(
            diarize_df['start'], seg.start
        )

        # remove no hit, otherwise we look for closest (even negative intersection...)
        new_segment: SingleDiarizedSegment | None = None
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker = cast(
                str,
                (
                    dia_tmp.groupby('speaker')['intersection']  # pyright: ignore
                    .sum()
                    .sort_values(ascending=False)
                    .index[0]
                ),
            )

            new_segment = SingleDiarizedSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=speaker,
                words=[],
                chars=seg.chars,
            )

            new_segments.append(new_segment)

        new_words: list[SingleDiarizedWordSegment] = []

        # assign speaker to words
        if len(seg.words) > 0:
            for word in seg.words:
                if word.start is not None:
                    diarize_df['intersection'] = np.minimum(
                        diarize_df['end'], word.end
                    ) - np.maximum(diarize_df['start'], word.start)
                    diarize_df['union'] = np.maximum(
                        diarize_df['end'], word.end
                    ) - np.minimum(diarize_df['start'], word.start)
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker = cast(
                            str,
                            (
                                dia_tmp.groupby('speaker')['intersection']  # pyright: ignore
                                .sum()
                                .sort_values(ascending=False)
                                .index[0]
                            ),
                        )

                        new_word = SingleDiarizedWordSegment(
                            start=word.start,
                            end=word.end,
                            word=word.word,
                            score=word.score,
                            speaker=speaker,
                        )

                        new_words.append(new_word)

    diarized_transcript_result = DiarizedTranscriptionResult(
        segments=new_segments,
        word_segments=[word for segment in new_segments for word in segment.words],
    )

    return diarized_transcript_result
