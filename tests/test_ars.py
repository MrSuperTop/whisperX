from collections.abc import Iterator
from itertools import islice

from jiwer import wer

from tests.utils.load_dataset import AudioDataset, DatasetEntry
from whisperx.alignment import AlignModelWrapper
from whisperx.alignment.types import AlignedTranscriptionResult
from whisperx.asr.batching_whisper_pipeline import BatchingWhisperPipeline
from whisperx.diarize import DiarizationPipeline
from whisperx.diarize.utils import assign_word_speakers

DEFAULT_LIMIT = 25


class TestArs:
    def test_asr(self, dataset: AudioDataset, model: BatchingWhisperPipeline) -> None:
        transcription_results: list[str] = []
        ground_truth: list[str] = []

        for entry in islice(dataset, DEFAULT_LIMIT):
            result = model.transcribe(entry['audio']['array'], language='en')

            ground_truth.append(entry['sentence'])

            joined_segments = ' '.join(
                [segment.text for segment in result.segments]
            ).strip()
            transcription_results.append(joined_segments)

        error = wer(ground_truth, transcription_results)
        print(error)

    def _align(
        self,
        dataset: AudioDataset,
        model: BatchingWhisperPipeline,
        limit: int = DEFAULT_LIMIT,
    ) -> Iterator[tuple[DatasetEntry, AlignedTranscriptionResult]]:
        align_model = AlignModelWrapper.load('en')
        for entry in islice(dataset, limit):
            result = model.transcribe(entry['audio']['array'])
            align_result = align_model.align(result.segments, entry['audio']['array'])

            yield entry, align_result

    def test_align(self, dataset: AudioDataset, model: BatchingWhisperPipeline) -> None:
        align_results = self._align(dataset, model, DEFAULT_LIMIT)
        for _, aligned in align_results:
            print(aligned)

    def test_diarize(
        self, dataset: AudioDataset, model: BatchingWhisperPipeline, hf_token: str
    ) -> None:
        diarization_pipeline = DiarizationPipeline(use_auth_token=hf_token)
        align_results = self._align(dataset, model, DEFAULT_LIMIT)
        for audio, aligned in align_results:
            diarize_result = diarization_pipeline(audio['audio']['array'])
            result = assign_word_speakers(diarize_result, aligned)
            print(result)
