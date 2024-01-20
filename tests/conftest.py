from pathlib import Path

from dotenv import load_dotenv

from whisperx.logging import setup_loggers

TEST_DOTENV_FILE = Path('./.test.env')
load_dotenv(TEST_DOTENV_FILE)

setup_loggers()


pytest_plugins = [
    'tests.fixtures.hf_token',
    'tests.fixtures.dataset',
    'tests.fixtures.model',
]
