from typing import AsyncIterable, Iterable, Optional, Union

from _typeshed import Incomplete
from ctranslate2._ext import (
    GenerationResult as GenerationResult,
)
from ctranslate2._ext import (
    GenerationStepResult as GenerationStepResult,
)
from ctranslate2._ext import (
    Generator as Generator,
)
from ctranslate2._ext import (
    ScoringResult as ScoringResult,
)
from ctranslate2._ext import (
    TranslationResult as TranslationResult,
)
from ctranslate2._ext import (
    Translator as Translator,
)

def register_extensions() -> None: ...
def translator_translate_iterable(
    translator: Translator,
    source: Iterable[list[str]],
    target_prefix: Optional[Iterable[list[str]]] = ...,
    max_batch_size: int = ...,
    batch_type: str = ...,
    **kwargs,
) -> Iterable[TranslationResult]: ...
def translator_score_iterable(
    translator: Translator,
    source: Iterable[list[str]],
    target: Iterable[list[str]],
    max_batch_size: int = ...,
    batch_type: str = ...,
    **kwargs,
) -> Iterable[ScoringResult]: ...
def generator_generate_iterable(
    generator: Generator,
    start_tokens: Iterable[list[str]],
    max_batch_size: int = ...,
    batch_type: str = ...,
    **kwargs,
) -> Iterable[GenerationResult]: ...
def generator_score_iterable(
    generator: Generator,
    tokens: Iterable[list[str]],
    max_batch_size: int = ...,
    batch_type: str = ...,
    **kwargs,
) -> Iterable[ScoringResult]: ...
def translator_generate_tokens(
    translator: Translator,
    source: list[str],
    target_prefix: Optional[list[str]] = ...,
    *,
    max_decoding_length: int = ...,
    min_decoding_length: int = ...,
    sampling_topk: int = ...,
    sampling_topp: float = ...,
    sampling_temperature: float = ...,
    return_log_prob: bool = ...,
    repetition_penalty: float = ...,
    no_repeat_ngram_size: int = ...,
    disable_unk: bool = ...,
    suppress_sequences: Optional[list[list[str]]] = ...,
    end_token: Optional[Union[str, list[str], list[int]]] = ...,
    max_input_length: int = ...,
    use_vmap: bool = ...,
) -> Iterable[GenerationStepResult]: ...
def generator_generate_tokens(
    generator: Generator,
    prompt: Union[list[str], list[list[str]]],
    max_batch_size: int = ...,
    batch_type: str = ...,
    *,
    max_length: int = ...,
    min_length: int = ...,
    sampling_topk: int = ...,
    sampling_topp: float = ...,
    sampling_temperature: float = ...,
    return_log_prob: bool = ...,
    repetition_penalty: float = ...,
    no_repeat_ngram_size: int = ...,
    disable_unk: bool = ...,
    suppress_sequences: Optional[list[list[str]]] = ...,
    end_token: Optional[Union[str, list[str], list[int]]] = ...,
    static_prompt: Optional[list[str]] = ...,
    cache_static_prompt: bool = ...,
) -> Iterable[GenerationStepResult]: ...
async def generator_async_generate_tokens(
    generator: Generator,
    prompt: Union[list[str], list[list[str]]],
    max_batch_size: int = ...,
    batch_type: str = ...,
    *,
    max_length: int = ...,
    min_length: int = ...,
    sampling_topk: int = ...,
    sampling_topp: float = ...,
    sampling_temperature: float = ...,
    return_log_prob: bool = ...,
    repetition_penalty: float = ...,
    no_repeat_ngram_size: int = ...,
    disable_unk: bool = ...,
    suppress_sequences: Optional[list[list[str]]] = ...,
    end_token: Optional[Union[str, list[str], list[int]]] = ...,
    static_prompt: Optional[list[str]] = ...,
    cache_static_prompt: bool = ...,
) -> AsyncIterable[GenerationStepResult]: ...

class AsyncGenerator:
    queue: Incomplete
    shutdown_event: Incomplete
    iterator_task: Incomplete
    process_func: Incomplete
    args: Incomplete
    kwargs: Incomplete
    def __init__(self, process_func, *args, **kwargs) -> None: ...
    async def producer(self) -> None: ...
    def __aiter__(self): ...
    async def __anext__(self): ...
