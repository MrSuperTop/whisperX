from itertools import islice

from jiwer import wer

from tests.utils.load_dataset import AudioDataset
from whisperx.alignment import AlignModelWrapper
from whisperx.asr.faster_whisper_pipeline import FasterWhisperPipeline


class TestArs:
    def test_asr(self, dataset: AudioDataset, model: FasterWhisperPipeline) -> None:
        transcription_results: list[str] = []
        ground_truth: list[str] = []

        for entry in islice(dataset, 25):
            result = model.transcribe(entry["audio"]["array"], language="en")

            ground_truth.append(entry["sentence"])

            joined_segments = " ".join(
                [segment.text for segment in result.segments]
            ).strip()
            transcription_results.append(joined_segments)

        error = wer(ground_truth, transcription_results)
        print(error)

    def test_align(self, dataset: AudioDataset, model: FasterWhisperPipeline) -> None:
        align_model = AlignModelWrapper.load("en")
        for entry in islice(dataset, 1):
            result = model.transcribe(entry["audio"]["array"])
            print(result)
            align_result = align_model.align(result.segments, entry["audio"]["array"])
            print(align_result)
