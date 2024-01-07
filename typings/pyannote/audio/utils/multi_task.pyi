from typing import Any, Callable, Union

from pyannote.audio.core.model import Specifications

def map_with_specifications(
    specifications: Union[Specifications, tuple[Specifications]],
    func: Callable,
    *iterables,
) -> Union[Any, tuple[Any]]: ...
