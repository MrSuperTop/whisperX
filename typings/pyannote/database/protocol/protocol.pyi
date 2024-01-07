import collections
from typing import Iterator, Optional, Union

from _typeshed import Incomplete

Subset: Incomplete
LEGACY_SUBSET_MAPPING: Incomplete
Scope: Incomplete
Preprocessor: Incomplete
Preprocessors = dict[str, Preprocessor]

class ProtocolFile(collections.abc.MutableMapping):
    lazy: Incomplete
    lock_: Incomplete
    evaluating_: Incomplete
    def __init__(
        self, precomputed: Union[dict, ProtocolFile], lazy: dict = ...
    ) -> None: ...
    def __abs__(self): ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...
    def __delitem__(self, key) -> None: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def files(self) -> Iterator[ProtocolFile]: ...

class Protocol:
    preprocessors: Incomplete
    def __init__(self, preprocessors: Optional[Preprocessors] = ...) -> None: ...
    def preprocess(self, current_file: Union[dict, ProtocolFile]) -> ProtocolFile: ...
    def train_iter(self) -> Iterator[Union[dict, ProtocolFile]]: ...
    def development_iter(self) -> Iterator[Union[dict, ProtocolFile]]: ...
    def test_iter(self) -> Iterator[Union[dict, ProtocolFile]]: ...
    def subset_helper(self, subset: Subset) -> Iterator[ProtocolFile]: ...
    def train(self) -> Iterator[ProtocolFile]: ...
    def development(self) -> Iterator[ProtocolFile]: ...
    def test(self) -> Iterator[ProtocolFile]: ...
    def files(self) -> Iterator[ProtocolFile]: ...
