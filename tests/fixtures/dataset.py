from pathlib import Path

import pytest

from tests.utils.load_dataset import AudioDataset, load_dataset

CACHE_DIR = Path('./datasets')


@pytest.fixture(scope='module')
def dataset() -> AudioDataset:
    return load_dataset('mozilla-foundation/common_voice_13_0', cache_dir=CACHE_DIR)
