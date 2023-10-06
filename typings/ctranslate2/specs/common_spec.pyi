import enum
from _typeshed import Incomplete
from ctranslate2.specs import model_spec as model_spec

class Activation(enum.IntEnum):
    RELU: int
    GELUTanh: int
    SWISH: int
    GELU: int
    GELUSigmoid: int
    Tanh: int

class EmbeddingsMerge(enum.IntEnum):
    CONCAT: int
    ADD: int

class LayerNormSpec(model_spec.LayerSpec):
    gamma: Incomplete
    beta: Incomplete
    def __init__(self, rms_norm: bool = ...) -> None: ...

class LinearSpec(model_spec.LayerSpec):
    weight: Incomplete
    weight_scale: Incomplete
    bias: Incomplete
    def __init__(self) -> None: ...
    def has_bias(self): ...

class Conv1DSpec(model_spec.LayerSpec):
    weight: Incomplete
    bias: Incomplete
    def __init__(self) -> None: ...

class EmbeddingsSpec(model_spec.LayerSpec):
    weight: Incomplete
    weight_scale: Incomplete
    def __init__(self) -> None: ...
