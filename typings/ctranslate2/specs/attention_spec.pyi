import enum

from _typeshed import Incomplete
from ctranslate2.specs import common_spec as common_spec
from ctranslate2.specs import model_spec as model_spec

class RotaryScalingType(enum.IntEnum):
    Linear: int

class MultiHeadAttentionSpec(model_spec.LayerSpec):
    queries_scale: Incomplete
    layer_norm: Incomplete
    linear: Incomplete
    relative_position_keys: Incomplete
    relative_position_values: Incomplete
    relative_attention_bias: Incomplete
    relative_attention_max_distance: Incomplete
    rotary_dim: Incomplete
    rotary_interleave: Incomplete
    rotary_base: Incomplete
    rotary_scaling_type: Incomplete
    rotary_scaling_factor: Incomplete
    num_heads_kv: Incomplete
    def __init__(
        self,
        self_attention: bool = ...,
        relative_position: bool = ...,
        relative_attention_bias: bool = ...,
        rms_norm: bool = ...,
        rotary_dim: Incomplete | None = ...,
        rotary_interleave: bool = ...,
        rotary_scaling_type: Incomplete | None = ...,
        rotary_scaling_factor: int = ...,
        rotary_base: int = ...,
        num_heads_kv: Incomplete | None = ...,
    ) -> None: ...
