from pathlib import Path

import pytest

import datasets
from tests.utils.load_dataset import load_dataset

CACHE_DIR = Path('./datasets')


@pytest.fixture(scope='module')
def dataset() -> datasets.Dataset:
    return load_dataset(
        'mozilla-foundation/common_voice_13_0',
        cache_dir=CACHE_DIR
    )
