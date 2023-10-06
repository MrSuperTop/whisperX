from pyannote.audio.core.model import Specifications
from typing import Any, Callable, Tuple, Union

def map_with_specifications(specifications: Union[Specifications, Tuple[Specifications]], func: Callable, *iterables) -> Union[Any, Tuple[Any]]: ...
