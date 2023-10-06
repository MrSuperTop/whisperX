from _typeshed import Incomplete
from ctranslate2.specs import attention_spec as attention_spec, common_spec as common_spec, model_spec as model_spec
from typing import Optional, Tuple, Union

class TransformerEncoderSpec(model_spec.LayerSpec):
    num_heads: Incomplete
    pre_norm: Incomplete
    activation: Incomplete
    embeddings_merge: Incomplete
    embeddings: Incomplete
    scale_embeddings: bool
    position_encodings: Incomplete
    layer_norm: Incomplete
    layernorm_embedding: Incomplete
    layer: Incomplete
    def __init__(self, num_layers: int, num_heads: int, pre_norm: bool = ..., no_final_norm: bool = ..., activation: common_spec.Activation = ..., num_source_embeddings: int = ..., embeddings_merge: common_spec.EmbeddingsMerge = ..., layernorm_embedding: bool = ..., relative_position: bool = ..., relative_attention_bias: bool = ..., ffn_glu: bool = ..., rms_norm: bool = ..., multi_query_attention: bool = ...) -> None: ...

class TransformerDecoderSpec(model_spec.LayerSpec):
    num_heads: Incomplete
    pre_norm: Incomplete
    activation: Incomplete
    alignment_layer: Incomplete
    alignment_heads: Incomplete
    embeddings: Incomplete
    scale_embeddings: bool
    scale_outputs: Incomplete
    alibi: Incomplete
    alibi_use_positive_positions: Incomplete
    scale_alibi: Incomplete
    position_encodings: Incomplete
    layer_norm: Incomplete
    layernorm_embedding: Incomplete
    projection: Incomplete
    layer: Incomplete
    start_from_zero_embedding: bool
    project_in: Incomplete
    project_out: Incomplete
    def __init__(self, num_layers: int, num_heads: int, pre_norm: bool = ..., activation: common_spec.Activation = ..., layernorm_embedding: bool = ..., with_encoder_attention: bool = ..., no_final_norm: bool = ..., project_in_out: bool = ..., relative_position: bool = ..., relative_attention_bias: bool = ..., alignment_layer: int = ..., alignment_heads: int = ..., ffn_glu: bool = ..., rms_norm: bool = ..., alibi: bool = ..., alibi_use_positive_positions: bool = ..., scale_alibi: bool = ..., rotary_dim: Optional[int] = ..., rotary_interleave: bool = ..., rotary_scaling_type: Optional[attention_spec.RotaryScalingType] = ..., rotary_scaling_factor: float = ..., rotary_base: float = ..., parallel_residual: bool = ..., shared_layer_norm: bool = ..., multi_query_attention: bool = ..., num_heads_kv: Optional[int] = ...) -> None: ...

class TransformerEncoderLayerSpec(model_spec.LayerSpec):
    self_attention: Incomplete
    ffn: Incomplete
    def __init__(self, relative_position: bool = ..., relative_attention_bias: bool = ..., ffn_glu: bool = ..., rms_norm: bool = ..., num_heads_kv: Incomplete | None = ...) -> None: ...

class TransformerDecoderLayerSpec(model_spec.LayerSpec):
    self_attention: Incomplete
    attention: Incomplete
    ffn: Incomplete
    shared_layer_norm: Incomplete
    input_layer_norm: Incomplete
    post_attention_layer_norm: Incomplete
    def __init__(self, with_encoder_attention: bool = ..., relative_position: bool = ..., relative_attention_bias: bool = ..., ffn_glu: bool = ..., rms_norm: bool = ..., rotary_dim: Incomplete | None = ..., rotary_interleave: bool = ..., rotary_scaling_type: Incomplete | None = ..., rotary_scaling_factor: int = ..., rotary_base: int = ..., parallel_residual: bool = ..., shared_layer_norm: bool = ..., num_heads_kv: Incomplete | None = ...) -> None: ...

class FeedForwardSpec(model_spec.LayerSpec):
    layer_norm: Incomplete
    linear_0: Incomplete
    linear_1: Incomplete
    linear_0_noact: Incomplete
    def __init__(self, glu: bool = ..., rms_norm: bool = ...) -> None: ...

class PositionEncoderSpec(model_spec.LayerSpec):
    encodings: Incomplete
    def __init__(self) -> None: ...

class TransformerConfig(model_spec.SequenceToSequenceModelConfig):
    def __init__(self, layer_norm_epsilon: Optional[float] = ..., **kwargs) -> None: ...

class TransformerSpec(model_spec.SequenceToSequenceModelSpec):
    encoder: Incomplete
    decoder: Incomplete
    def __init__(self, encoder: TransformerEncoderSpec, decoder: TransformerDecoderSpec) -> None: ...
    @classmethod
    def from_config(cls, num_layers: Union[int, Tuple[int, int]], num_heads: int, with_relative_position: bool = ..., pre_norm: bool = ..., no_final_norm: bool = ..., activation: common_spec.Activation = ..., alignment_layer: int = ..., alignment_heads: int = ..., num_source_embeddings: int = ..., embeddings_merge: common_spec.EmbeddingsMerge = ..., layernorm_embedding: bool = ..., relative_attention_bias: bool = ..., ffn_glu: bool = ..., rms_norm: bool = ..., multi_query_attention: bool = ...): ...
    @property
    def name(self): ...
    @property
    def revision(self): ...
    def get_default_config(self): ...
    def get_source_vocabulary_size(self): ...
    def get_target_vocabulary_size(self): ...

class TransformerDecoderModelConfig(model_spec.LanguageModelConfig):
    def __init__(self, layer_norm_epsilon: Optional[float] = ..., **kwargs) -> None: ...

class TransformerDecoderModelSpec(model_spec.LanguageModelSpec):
    decoder: Incomplete
    def __init__(self, decoder: TransformerDecoderSpec) -> None: ...
    @classmethod
    def from_config(cls, num_layers: int, num_heads: int, pre_norm: bool = ..., activation: common_spec.Activation = ..., layernorm_embedding: bool = ..., no_final_norm: bool = ..., project_in_out: bool = ..., with_relative_position: bool = ..., ffn_glu: bool = ..., rms_norm: bool = ..., alibi: bool = ..., alibi_use_positive_positions: bool = ..., scale_alibi: bool = ..., rotary_dim: Optional[int] = ..., rotary_interleave: bool = ..., rotary_scaling_type: Optional[attention_spec.RotaryScalingType] = ..., rotary_scaling_factor: float = ..., rotary_base: float = ..., parallel_residual: bool = ..., shared_layer_norm: bool = ..., multi_query_attention: bool = ..., num_heads_kv: Optional[int] = ...): ...
    @property
    def name(self): ...
    @property
    def revision(self): ...
    def get_default_config(self): ...
    def get_vocabulary_size(self): ...

class TransformerEncoderModelConfig(model_spec.LanguageModelConfig):
    def __init__(self, layer_norm_epsilon: Optional[float] = ..., **kwargs) -> None: ...

class TransformerEncoderModelSpec(model_spec.LanguageModelSpec):
    encoder: Incomplete
    pooler_dense: Incomplete
    pooler_activation: Incomplete
    def __init__(self, encoder: TransformerEncoderSpec, pooling_layer: bool = ..., pooling_activation: common_spec.Activation = ...) -> None: ...
    @property
    def name(self): ...
    @property
    def revision(self): ...
    def get_default_config(self): ...
    def get_vocabulary_size(self): ...
