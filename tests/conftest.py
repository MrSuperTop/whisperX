from whisperx.logging import setup_loggers

setup_loggers()

pytest_plugins = ['tests.fixtures.dataset', 'tests.fixtures.model']
