import abc
import argparse
from ctranslate2.specs.model_spec import ACCEPTED_MODEL_TYPES as ACCEPTED_MODEL_TYPES, ModelSpec as ModelSpec
from typing import Optional

class Converter(abc.ABC, metaclass=abc.ABCMeta):
    @staticmethod
    def declare_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: ...
    def convert_from_args(self, args: argparse.Namespace) -> str: ...
    def convert(self, output_dir: str, vmap: Optional[str] = ..., quantization: Optional[str] = ..., force: bool = ...) -> str: ...
