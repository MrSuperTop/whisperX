from typing import Optional

from _typeshed import Incomplete
from ctranslate2.specs import (
    common_spec as common_spec,
)
from ctranslate2.specs import (
    model_spec as model_spec,
)
from ctranslate2.specs import (
    transformer_spec as transformer_spec,
)

class WhisperConfig(model_spec.ModelConfig):
    def __init__(
        self,
        suppress_ids: Optional[list[int]] = ...,
        suppress_ids_begin: Optional[list[int]] = ...,
        lang_ids: Optional[list[int]] = ...,
        alignment_heads: Optional[list[tuple[int, int]]] = ...,
    ) -> None: ...

class WhisperSpec(model_spec.LanguageModelSpec):
    encoder: Incomplete
    decoder: Incomplete
    def __init__(self, num_layers, num_heads) -> None: ...
    @property
    def name(self): ...
    @property
    def revision(self): ...
    def get_default_config(self): ...
    def get_vocabulary_size(self): ...

class WhisperEncoderSpec(model_spec.LayerSpec):
    num_heads: Incomplete
    conv1: Incomplete
    conv2: Incomplete
    position_encodings: Incomplete
    layer_norm: Incomplete
    layer: Incomplete
    def __init__(self, num_layers, num_heads) -> None: ...
