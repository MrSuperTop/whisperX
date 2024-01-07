from typing import Iterator

from .protocol import Protocol as Protocol

class CollectionProtocol(Protocol):
    def files_iter(self) -> Iterator[dict]: ...
    def train_iter(self) -> Iterator[dict]: ...
