from __future__ import annotations

import typing
from pathlib import Path
from typing import BinaryIO, overload

if typing.TYPE_CHECKING:
    from _typeshed import StrPath


@overload
def convert_path(path: StrPath) -> str:
    ...


@overload
def convert_path(path: None) -> None:
    ...


@overload
def convert_path(path: BinaryIO) -> BinaryIO:
    ...


def convert_path(path: StrPath | None | BinaryIO) -> str | None | BinaryIO:
    if isinstance(path, Path):
        return str(path.resolve().absolute())
    elif isinstance(path, BinaryIO):
        return path

    return str(path)
