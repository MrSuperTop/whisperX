from pathlib import Path

import pytest

import whisperx
from whisperx.asr.faster_whisper_pipeline import FasterWhisperPipeline
from whisperx.asr.whisper_model import AsrOptions

DOWNLOAD_ROOT = Path("./models")


@pytest.fixture(scope="class")
def model() -> FasterWhisperPipeline:
    model = whisperx.load_model(
        "guillaumekln/faster-whisper-medium",
        compute_type="float32",
        device="cuda",
        asr_options=AsrOptions(suppress_numerals=False, beam_size=10),
        download_root=DOWNLOAD_ROOT,
    )

    return model
