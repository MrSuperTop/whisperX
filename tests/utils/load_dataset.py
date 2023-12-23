from __future__ import annotations

import os.path
import typing
from pathlib import Path
from typing import Iterator, Literal, TypedDict, cast

import numpy as np

import datasets
from datasets import Audio
from whisperx.audio import AudioData
from whisperx.types import LanguageCode
from whisperx.utils.convert_path import convert_path

if typing.TYPE_CHECKING:
    from _typeshed import StrPath

DatasetName = Literal["mozilla-foundation/common_voice_13_0"]


class AudioDatasetObject(TypedDict):
    path: str
    array: AudioData
    sampling_rate: int


class DatasetEntry(TypedDict):
    client_id: str
    path: str
    audio: AudioDatasetObject
    sentence: str
    up_votes: int
    down_votes: int
    age: str
    gender: Literal["male", "female"]
    accent: str
    locale: LanguageCode
    segment: str


# TODO: Into dataloader?
class AudioDataset:
    def __init__(self, base_dataset: datasets.Dataset) -> None:
        self._dataset = cast(Iterator[DatasetEntry], base_dataset)

    def __iter__(self) -> Iterator[DatasetEntry]:  # pyright: ignore[reportIncompatibleMethodOverride]
        # NOTE: Could not figure out how to properly case nested columns, does not seem to be a proper way to do this
        def wrapper() -> Iterator[DatasetEntry]:
            entry: DatasetEntry
            for entry in self._dataset:
                converted = entry["audio"]["array"].astype(dtype=np.float32)

                entry["audio"]["array"] = converted

                yield entry

        return wrapper()


def load_dataset(
    dataset_name: DatasetName,
    streaming: bool = True,
    cache_dir: StrPath = Path("./datasets"),
) -> AudioDataset:
    os.makedirs(cache_dir, exist_ok=True)

    base_dataset = cast(
        datasets.Dataset,
        datasets.load_dataset(  # pyright: ignore [reportUnknownMemberType]
            dataset_name,
            "en",
            split="test",
            cache_dir=convert_path(cache_dir),
            streaming=streaming,
        ),
    )

    base_dataset = base_dataset.cast_column(  # pyright: ignore [reportUnknownMemberType]
        "audio", Audio(sampling_rate=16000, mono=True)
    ).shuffle()

    return AudioDataset(base_dataset)
