from ctranslate2.converters.converter import Converter as Converter
from ctranslate2.specs import common_spec as common_spec, model_spec as model_spec, transformer_spec as transformer_spec

class OpenAIGPT2Converter(Converter):
    def __init__(self, model_dir: str) -> None: ...

def set_decoder(spec, weights, scope) -> None: ...
def set_layer_norm(spec, weights, scope) -> None: ...
def set_linear(spec, weights, scope) -> None: ...
def set_layer(spec, weights, scope) -> None: ...
def main() -> None: ...
