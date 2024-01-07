import abc
import argparse
from typing import Optional

from ctranslate2.specs.model_spec import (
    ACCEPTED_MODEL_TYPES as ACCEPTED_MODEL_TYPES,
)
from ctranslate2.specs.model_spec import (
    ModelSpec as ModelSpec,
)

class Converter(abc.ABC, metaclass=abc.ABCMeta):
    @staticmethod
    def declare_arguments(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser: ...
    def convert_from_args(self, args: argparse.Namespace) -> str: ...
    def convert(
        self,
        output_dir: str,
        vmap: Optional[str] = ...,
        quantization: Optional[str] = ...,
        force: bool = ...,
    ) -> str: ...
