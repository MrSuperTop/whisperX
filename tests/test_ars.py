from itertools import islice
from pathlib import Path

import numpy as np
from jiwer import wer

import datasets
import whisperx
from tests.utils.load_dataset import DatasetEntry
from whisperx.asr.whisper_model import AsrOptions

DOWNLOAD_ROOT = Path('./models')


def test_asr(dataset: datasets.Dataset) -> None:
    model = whisperx.load_model(
        'guillaumekln/faster-whisper-medium',
        compute_type='float32',
        device='cuda',
        asr_options=AsrOptions(suppress_numerals=False, beam_size=10),
        download_root=DOWNLOAD_ROOT,
    )

    transcription_results: list[str] = []
    ground_truth: list[str] = []

    entry: DatasetEntry
    for entry in islice(dataset, 250):
        audio_data = entry['audio']['array'].astype(np.float32)

        result = model.transcribe(audio_data, language='en')

        ground_truth.append(entry['sentence'])

        joined_segments = ' '.join([segment.text for segment in result.segments]).strip()
        transcription_results.append(joined_segments)

    error = wer(ground_truth, transcription_results)
    print(error)
