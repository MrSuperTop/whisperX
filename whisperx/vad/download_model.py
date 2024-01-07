from __future__ import annotations

import hashlib
import os
import typing
import urllib.error
import urllib.request
from collections.abc import Callable

from tqdm import tqdm

from whisperx.logging import get_logger

if typing.TYPE_CHECKING:
    from _typeshed import StrPath

SHA256_HASH_LENGTH = 64
MODEL_WEIGHTS_LINKS: list[tuple[str, Callable[[str], str]]] = [
    (
        'https://huggingface.co/pyannote/segmentation/resolve/main/pytorch_model.bin?download=true',
        lambda _: '0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea',
    ),
    (
        'https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin',
        lambda link: link.split('/')[-2],
    ),
]

logger = get_logger(__name__)


def _download_model(from_url: str, checksum: str, model_dir: StrPath) -> StrPath:
    os.makedirs(model_dir, exist_ok=True)

    model_fp = os.path.join(model_dir, 'whisperx-vad-segmentation.bin')
    if os.path.exists(model_fp) and not os.path.isfile(model_fp):
        raise RuntimeError(f'{model_fp} exists and is not a regular file')

    if not os.path.isfile(model_fp):
        try:
            with urllib.request.urlopen(from_url) as source, open(
                model_fp, 'wb'
            ) as output:
                total_size = int(source.info().get('Content-Length'))

                with tqdm(
                    total=total_size,
                    ncols=80,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as loop:
                    while True:
                        buffer = source.read(8192)
                        if not buffer:
                            break

                        output.write(buffer)
                        loop.update(len(buffer))
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            logger.error(f'Could not retrieve the model weights from {from_url}')
            logger.error(e)

            raise ValueError(f'Incorrect from_url: {from_url}')

    model_bytes = open(model_fp, 'rb').read()
    if hashlib.sha256(model_bytes).hexdigest() != checksum:
        logger.warn(
            'Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model.'
        )

    return model_fp


def download_model(model_dir: StrPath) -> StrPath | None:
    for link, checksum_getter in MODEL_WEIGHTS_LINKS:
        checksum = checksum_getter(link)

        if len(checksum) != SHA256_HASH_LENGTH:
            raise ValueError(
                'Checksum getter functions should return a SHA256_HASH of length 64 to check the downloaded files against.'
            )

        try:
            download_result = _download_model(link, checksum, model_dir)
        except ValueError:
            continue

        if os.path.exists(download_result):
            return download_result

    return None
