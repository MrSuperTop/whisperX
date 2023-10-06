from ctranslate2.converters import utils as utils
from ctranslate2.converters.converter import Converter as Converter
from ctranslate2.specs import common_spec as common_spec, transformer_spec as transformer_spec
from typing import Optional, Union

class OpenNMTTFConverter(Converter):
    @classmethod
    def from_config(cls, config: Union[str, dict], auto_config: bool = ..., checkpoint_path: Optional[str] = ..., model: Optional[str] = ...): ...
    def __init__(self, model) -> None: ...

class TransformerSpecBuilder:
    def __call__(self, model): ...
    def set_transformer_encoder(self, spec, module, inputter) -> None: ...
    def set_transformer_decoder(self, spec, module, inputter) -> None: ...
    def set_ffn(self, spec, module) -> None: ...
    def set_multi_head_attention(self, spec, module, self_attention: bool = ...) -> None: ...
    def set_layer_norm_from_wrapper(self, spec, module) -> None: ...
    def set_layer_norm(self, spec, module) -> None: ...
    def set_linear(self, spec, module) -> None: ...
    def set_embeddings(self, spec, module) -> None: ...
    def set_position_encodings(self, spec, module) -> None: ...

class TransformerDecoderSpecBuilder(TransformerSpecBuilder):
    def __call__(self, model): ...

def main() -> None: ...
