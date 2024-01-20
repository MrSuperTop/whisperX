import os

import pytest


@pytest.fixture(scope='session')
def hf_token() -> str:
    token = os.environ.get('HF_TOKEN')

    if token is None:
        raise ValueError(
            'Please create a .test.env file in the root of the project following the .test.env.example file'
        )

    return token
