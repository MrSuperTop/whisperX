# module ctranslate2._ext
# from /mnt/data/code/Python/Tests/whisperx-test/whisperX/.venv/lib/python3.11/site-packages/ctranslate2/_ext.cpython-311-x86_64-linux-gnu.so
# by generator 1.147
# no doc

# imports
from typing import Literal, overload

import ctranslate2
import pybind11_builtins as __pybind11_builtins

from whisperx.types import ComputeType, DeviceType

# functions

def contains_model(path):  # real signature unknown; restored from __doc__
    """
    contains_model(path: str) -> bool

    Helper function to check if a directory seems to contain a CTranslate2 model.
    """
    return False

def get_cuda_device_count():  # real signature unknown; restored from __doc__
    """
    get_cuda_device_count() -> int

    Returns the number of visible GPU devices.
    """
    return 0

def get_log_level():  # real signature unknown; restored from __doc__
    """get_log_level() -> ctranslate2._ext.LogLevel"""
    pass

def get_supported_compute_types(
    device: DeviceType, device_index: int = 0
) -> set[ComputeType]:  # real signature unknown; restored from __doc__
    """
    get_supported_compute_types(device: str, device_index: int = 0) -> Set[str]


                 Returns the set of supported compute types on a device.

                 Arguments:
                   device: Device name (cpu or cuda).
                   device_index: Device index.

                 Example:
                     >>> ctranslate2.get_supported_compute_types("cpu")
                     {'int16', 'float32', 'int8', 'int8_float32'}
                     >>> ctranslate2.get_supported_compute_types("cuda")
                     {'float32', 'int8_float16', 'float16', 'int8', 'int8_float32'}
    """
    pass

def set_log_level(arg0):  # real signature unknown; restored from __doc__
    """set_log_level(arg0: ctranslate2._ext.LogLevel) -> None"""
    pass

def set_random_seed(seed):  # real signature unknown; restored from __doc__
    """
    set_random_seed(seed: int) -> None

    Sets the seed of random generators.
    """
    pass

# classes

class AsyncGenerationResult(__pybind11_builtins.pybind11_object):
    """Asynchronous wrapper around a result object."""
    def done(self):  # real signature unknown; restored from __doc__
        """
        done(self: ctranslate2._ext.AsyncGenerationResult) -> bool

        Returns ``True`` if the result is available.
        """
        return False

    def result(self):  # real signature unknown; restored from __doc__
        """
        result(self: ctranslate2._ext.AsyncGenerationResult) -> ctranslate2._ext.GenerationResult


                         Blocks until the result is available and returns it.

                         If an exception was raised when computing the result,
                         this method raises the exception.
        """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

class AsyncScoringResult(__pybind11_builtins.pybind11_object):
    """Asynchronous wrapper around a result object."""
    def done(self):  # real signature unknown; restored from __doc__
        """
        done(self: ctranslate2._ext.AsyncScoringResult) -> bool

        Returns ``True`` if the result is available.
        """
        return False

    def result(self):  # real signature unknown; restored from __doc__
        """
        result(self: ctranslate2._ext.AsyncScoringResult) -> ctranslate2._ext.ScoringResult


                         Blocks until the result is available and returns it.

                         If an exception was raised when computing the result,
                         this method raises the exception.
        """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

class AsyncTranslationResult(__pybind11_builtins.pybind11_object):
    """Asynchronous wrapper around a result object."""
    def done(self):  # real signature unknown; restored from __doc__
        """
        done(self: ctranslate2._ext.AsyncTranslationResult) -> bool

        Returns ``True`` if the result is available.
        """
        return False

    def result(self):  # real signature unknown; restored from __doc__
        """
        result(self: ctranslate2._ext.AsyncTranslationResult) -> ctranslate2._ext.TranslationResult


                         Blocks until the result is available and returns it.

                         If an exception was raised when computing the result,
                         this method raises the exception.
        """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

class DataType(__pybind11_builtins.pybind11_object):
    # no doc
    def __eq__(self, other):  # real signature unknown; restored from __doc__
        """__eq__(self: object, other: object) -> bool"""
        return False

    def __getstate__(self):  # real signature unknown; restored from __doc__
        """__getstate__(self: object) -> int"""
        return 0

    def __hash__(self):  # real signature unknown; restored from __doc__
        """__hash__(self: object) -> int"""
        return 0

    def __index__(self):  # real signature unknown; restored from __doc__
        """__index__(self: ctranslate2._ext.DataType) -> int"""
        return 0

    def __init__(self, value):  # real signature unknown; restored from __doc__
        """__init__(self: ctranslate2._ext.DataType, value: int) -> None"""
        pass

    def __int__(self):  # real signature unknown; restored from __doc__
        """__int__(self: ctranslate2._ext.DataType) -> int"""
        return 0

    def __ne__(self, other):  # real signature unknown; restored from __doc__
        """__ne__(self: object, other: object) -> bool"""
        return False

    def __repr__(self):  # real signature unknown; restored from __doc__
        """__repr__(self: object) -> str"""
        return ''

    def __setstate__(self, state):  # real signature unknown; restored from __doc__
        """__setstate__(self: ctranslate2._ext.DataType, state: int) -> None"""
        pass

    def __str__(self, *args, **kwargs):  # real signature unknown
        """name(self: handle) -> str"""
        pass

    name = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """name(self: handle) -> str
"""

    value = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default

    bfloat16 = None  # (!) real value is '<DataType.bfloat16: 5>'
    float16 = None  # (!) real value is '<DataType.float16: 4>'
    float32 = None  # (!) real value is '<DataType.float32: 0>'
    int16 = None  # (!) real value is '<DataType.int16: 2>'
    int32 = None  # (!) real value is '<DataType.int32: 3>'
    int8 = None  # (!) real value is '<DataType.int8: 1>'
    __entries = {
        'bfloat16': (
            None,  # (!) real value is '<DataType.bfloat16: 5>'
            None,
        ),
        'float16': (
            None,  # (!) real value is '<DataType.float16: 4>'
            None,
        ),
        'float32': (
            None,  # (!) real value is '<DataType.float32: 0>'
            None,
        ),
        'int16': (
            None,  # (!) real value is '<DataType.int16: 2>'
            None,
        ),
        'int32': (
            None,  # (!) real value is '<DataType.int32: 3>'
            None,
        ),
        'int8': (
            None,  # (!) real value is '<DataType.int8: 1>'
            None,
        ),
    }
    __members__ = {
        'bfloat16': None,  # (!) real value is '<DataType.bfloat16: 5>'
        'float16': None,  # (!) real value is '<DataType.float16: 4>'
        'float32': None,  # (!) real value is '<DataType.float32: 0>'
        'int16': None,  # (!) real value is '<DataType.int16: 2>'
        'int32': None,  # (!) real value is '<DataType.int32: 3>'
        'int8': None,  # (!) real value is '<DataType.int8: 1>'
    }

class Encoder(__pybind11_builtins.pybind11_object):
    """
    A text encoder.

                Example:

                    >>> encoder = ctranslate2.Encoder("model/", device="cpu")
                    >>> encoder.forward_batch([["▁Hello", "▁world", "!"]])
    """
    def forward_batch(
        self, inputs, *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        forward_batch(self: ctranslate2._ext.Encoder, inputs: Union[List[List[str]], List[List[int]], ctranslate2._ext.StorageView], lengths: Optional[ctranslate2._ext.StorageView] = None, token_type_ids: Optional[List[List[int]]] = None) -> ctranslate2._ext.EncoderForwardOutput


                         Forwards a batch of sequences in the encoder.

                         Arguments:
                           inputs: A batch of sequences either as string tokens or token IDs.
                             This argument can also be a dense int32 array with shape
                             ``[batch_size, max_length]`` (e.g. created from a Numpy array or PyTorch tensor).
                           lengths: The length of each sequence as a int32 array with shape
                             ``[batch_size]``. Required when :obj:`inputs` is a dense array.
                           token_type_ids: A batch of token type IDs of same shape as :obj:`inputs`.
                             ``[batch_size, max_length]``.

                         Returns:
                           The encoder model output.
        """
        pass

    def __init__(
        self, model_path, device='cpu', *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        __init__(self: ctranslate2._ext.Encoder, model_path: str, device: str = 'cpu', *, device_index: Union[int, List[int]] = 0, compute_type: Union[str, Dict[str, str]] = 'default', inter_threads: int = 1, intra_threads: int = 0, max_queued_batches: int = 0, files: object = None) -> None


                         Initializes the encoder.

                         Arguments:
                           model_path: Path to the CTranslate2 model directory.
                           device: Device to use (possible values are: cpu, cuda, auto).
                           device_index: Device IDs where to place this encoder on.
                           compute_type: Model computation type or a dictionary mapping a device name
                             to the computation type (possible values are: default, auto, int8, int8_float32,
                             int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
                           inter_threads: Maximum number of parallel generations.
                           intra_threads: Number of OpenMP threads per encoder (0 to use a default value).
                           max_queued_batches: Maximum numbers of batches in the queue (-1 for unlimited,
                             0 for an automatic value). When the queue is full, future requests will block
                             until a free slot is available.
                           files: Load model files from the memory. This argument is a dictionary mapping
                             file names to file contents as file-like or bytes objects. If this is set,
                             :obj:`model_path` acts as an identifier for this model.
        """
        pass

    compute_type = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Computation type used by the model."""

    device = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Device this encoder is running on."""

    device_index = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """List of device IDs where this encoder is running on."""

    num_active_batches = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of batches waiting to be processed or currently processed."""

    num_encoders = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of encoders backing this instance."""

    num_queued_batches = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of batches waiting to be processed."""

class EncoderForwardOutput(__pybind11_builtins.pybind11_object):
    """Forward output of an encoder model."""
    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __repr__(self):  # real signature unknown; restored from __doc__
        """__repr__(self: ctranslate2._ext.EncoderForwardOutput) -> str"""
        return ''

    last_hidden_state = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Output of the last layer."""

    pooler_output = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Output of the pooling layer."""

class ExecutionStats(__pybind11_builtins.pybind11_object):
    """A structure containing some execution statistics."""
    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __repr__(self):  # real signature unknown; restored from __doc__
        """__repr__(self: ctranslate2._ext.ExecutionStats) -> str"""
        return ''

    num_examples = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of processed examples."""

    num_tokens = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of output tokens."""

    total_time_in_ms = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Total processing time in milliseconds."""

class GenerationResult(__pybind11_builtins.pybind11_object):
    """A generation result."""
    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __repr__(self):  # real signature unknown; restored from __doc__
        """__repr__(self: ctranslate2._ext.GenerationResult) -> str"""
        return ''

    scores = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Score of each sequence (empty if :obj:`return_scores` was disabled)."""

    sequences = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Generated sequences of tokens."""

    sequences_ids = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Generated sequences of token IDs."""

class GenerationStepResult(__pybind11_builtins.pybind11_object):
    """The result for a single generation step."""
    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __repr__(self):  # real signature unknown; restored from __doc__
        """__repr__(self: ctranslate2._ext.GenerationStepResult) -> str"""
        return ''

    batch_id = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """The batch index."""

    hypothesis_id = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Index of the hypothesis in the batch."""

    is_last = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Whether this step is the last decoding step for this batch."""

    log_prob = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Log probability of the token (``None`` if :obj:`return_log_prob` was disabled)."""

    step = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """The decoding step."""

    token = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """String value of the generated token."""

    token_id = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """ID of the generated token."""

class Generator(__pybind11_builtins.pybind11_object):
    """
    A text generator.

                Example:

                    >>> generator = ctranslate2.Generator("model/", device="cpu")
                    >>> generator.generate_batch([["<s>"]], max_length=50, sampling_topk=20)
    """
    def async_generate_tokens(
        generator,
        prompt,
        max_batch_size=0,
        batch_type=None,
        *,
        max_length=512,
        min_length=0,
        sampling_topk=1,
        sampling_topp=1,
        sampling_temperature=1,
        return_log_prob=False,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        disable_unk=False,
        suppress_sequences=None,
        end_token=None,
        static_prompt=None,
        cache_static_prompt=True,
    ):  # reliably restored by inspect
        """
        Yields tokens asynchronously as they are generated by the model.

            Arguments:
              prompt: Batch of start tokens. If the decoder starts from a
                special start token like <s>, this token should be added to this input.
              max_batch_size: The maximum batch size.
              batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
              max_length: Maximum generation length.
              min_length: Minimum generation length.
              sampling_topk: Randomly sample predictions from the top K candidates.
              sampling_topp: Keep the most probable tokens whose cumulative probability exceeds this value.
              sampling_temperature: Sampling temperature to generate more random samples.
              return_log_prob: Include the token log probability in the result.
              repetition_penalty: Penalty applied to the score of previously generated tokens
                (set > 1 to penalize).
              no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                (set 0 to disable).
              disable_unk: Disable the generation of the unknown token.
              suppress_sequences: Disable the generation of some sequences of tokens.
              end_token: Stop the decoding on one of these tokens (defaults to the model EOS token).
              static_prompt: If the model expects a static prompt (a.k.a. system prompt)
                it can be set here to simplify the inputs and optionally cache the model
                state for this prompt to accelerate future generations.
              cache_static_prompt: Cache the model state after the static prompt and
                reuse it for future generations using the same static prompt.

            Returns:
              An async generator iterator over :class:`ctranslate2.GenerationStepResult` instances.

            Note:
              This generation method is not compatible with beam search which requires a complete decoding.
        """
        pass

    def forward_batch(
        self, inputs, *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        forward_batch(self: ctranslate2._ext.Generator, inputs: Union[List[List[str]], List[List[int]], ctranslate2._ext.StorageView], lengths: Optional[ctranslate2._ext.StorageView] = None, *, return_log_probs: bool = False) -> ctranslate2._ext.StorageView


                         Forwards a batch of sequences in the generator.

                         Arguments:
                           inputs: A batch of sequences either as string tokens or token IDs.
                             This argument can also be a dense int32 array with shape
                             ``[batch_size, max_length]`` (e.g. created from a Numpy array or PyTorch tensor).
                           lengths: The length of each sequence as a int32 array with shape
                             ``[batch_size]``. Required when :obj:`inputs` is a dense array.
                           return_log_probs: If ``True``, the method returns the log probabilties instead
                             of the unscaled logits.

                         Returns:
                           The output logits, or the output log probabilities if :obj:`return_log_probs`
                           is enabled.
        """
        pass

    def generate_batch(
        self, start_tokens, List=None, p_str=None, *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        generate_batch(self: ctranslate2._ext.Generator, start_tokens: List[List[str]], *, max_batch_size: int = 0, batch_type: str = 'examples', asynchronous: bool = False, beam_size: int = 1, patience: float = 1, num_hypotheses: int = 1, length_penalty: float = 1, repetition_penalty: float = 1, no_repeat_ngram_size: int = 0, disable_unk: bool = False, suppress_sequences: Optional[List[List[str]]] = None, end_token: Optional[Union[str, List[str], List[int]]] = None, return_end_token: bool = False, max_length: int = 512, min_length: int = 0, static_prompt: Optional[List[str]] = None, cache_static_prompt: bool = True, include_prompt_in_result: bool = True, return_scores: bool = False, return_alternatives: bool = False, min_alternative_expansion_prob: float = 0, sampling_topk: int = 1, sampling_topp: float = 1, sampling_temperature: float = 1, callback: Callable[[ctranslate2._ext.GenerationStepResult], bool] = None) -> Union[List[ctranslate2._ext.GenerationResult], List[ctranslate2._ext.AsyncGenerationResult]]


                         Generates from a batch of start tokens.

                         Note:
                           The way the start tokens are forwarded in the decoder depends on the argument
                           :obj:`include_prompt_in_result`:

                           * If :obj:`include_prompt_in_result` is ``True`` (the default), the decoding loop
                             is constrained to generate the start tokens that are then included in the result.
                           * If :obj:`include_prompt_in_result` is ``False``, the start tokens are forwarded
                             in the decoder at once to initialize its state (i.e. the KV cache for
                             Transformer models). For variable-length inputs, only the tokens up to the
                             minimum length in the batch are forwarded at once. The remaining tokens are
                             generated in the decoding loop with constrained decoding.

                           Consider setting ``include_prompt_in_result=False`` to increase the performance
                           for long inputs.

                         Arguments:
                           start_tokens: Batch of start tokens. If the decoder starts from a special
                             start token like ``<s>``, this token should be added to this input.
                           max_batch_size: The maximum batch size. If the number of inputs is greater than
                             :obj:`max_batch_size`, the inputs are sorted by length and split by chunks of
                             :obj:`max_batch_size` examples so that the number of padding positions is
                             minimized.
                           batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
                           asynchronous: Run the generation asynchronously.
                           beam_size: Beam size (1 for greedy search).
                           patience: Beam search patience factor, as described in
                             https://arxiv.org/abs/2204.05424. The decoding will continue until
                             beam_size*patience hypotheses are finished.
                           num_hypotheses: Number of hypotheses to return.
                           length_penalty: Exponential penalty applied to the length during beam search.
                           repetition_penalty: Penalty applied to the score of previously generated tokens
                             (set > 1 to penalize).
                           no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                             (set 0 to disable).
                           disable_unk: Disable the generation of the unknown token.
                           suppress_sequences: Disable the generation of some sequences of tokens.
                           end_token: Stop the decoding on one of these tokens (defaults to the model EOS token).
                           return_end_token: Include the end token in the results.
                           max_length: Maximum generation length.
                           min_length: Minimum generation length.
                           static_prompt: If the model expects a static prompt (a.k.a. system prompt)
                             it can be set here to simplify the inputs and optionally cache the model
                             state for this prompt to accelerate future generations.
                           cache_static_prompt: Cache the model state after the static prompt and
                             reuse it for future generations using the same static prompt.
                           include_prompt_in_result: Include the :obj:`start_tokens` in the result.
                           return_scores: Include the scores in the output.
                           return_alternatives: Return alternatives at the first unconstrained decoding position.
                           min_alternative_expansion_prob: Minimum initial probability to expand an alternative.
                           sampling_topk: Randomly sample predictions from the top K candidates.
                           sampling_topp: Keep the most probable tokens whose cumulative probability exceeds
                             this value.
                           sampling_temperature: Sampling temperature to generate more random samples.
                           callback: Optional function that is called for each generated token when
                             :obj:`beam_size` is 1. If the callback function returns ``True``, the
                             decoding will stop for this batch.

                         Returns:
                           A list of generation results.

                         See Also:
                           `GenerationOptions <https://github.com/OpenNMT/CTranslate2/blob/master/include/ctranslate2/generation.h>`_ structure in the C++ library.
        """
        pass

    def generate_iterable(
        generator, start_tokens, max_batch_size=32, batch_type=None, **kwargs
    ):  # reliably restored by inspect
        """
        Generates from an iterable of tokenized prompts.

            This method is built on top of :meth:`ctranslate2.Generator.generate_batch`
            to efficiently run generation on an arbitrarily large stream of data. It enables
            the following optimizations:

            * stream processing (the iterable is not fully materialized in memory)
            * parallel generations (if the generator has multiple workers)
            * asynchronous batch prefetching
            * local sorting by length

            Arguments:
              start_tokens: An iterable of tokenized prompts.
              max_batch_size: The maximum batch size.
              batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
              **kwargs: Any generation options accepted by
                :meth:`ctranslate2.Generator.generate_batch`.

            Returns:
              A generator iterator over :class:`ctranslate2.GenerationResult` instances.
        """
        pass

    def generate_tokens(
        generator,
        prompt,
        max_batch_size=0,
        batch_type=None,
        *,
        max_length=512,
        min_length=0,
        sampling_topk=1,
        sampling_topp=1,
        sampling_temperature=1,
        return_log_prob=False,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        disable_unk=False,
        suppress_sequences=None,
        end_token=None,
        static_prompt=None,
        cache_static_prompt=True,
    ):  # reliably restored by inspect
        """
        Yields tokens as they are generated by the model.

            Arguments:
              prompt: Batch of start tokens. If the decoder starts from a
                special start token like <s>, this token should be added to this input.
              max_batch_size: The maximum batch size.
              batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
              max_length: Maximum generation length.
              min_length: Minimum generation length.
              sampling_topk: Randomly sample predictions from the top K candidates.
              sampling_topp: Keep the most probable tokens whose cumulative probability exceeds this value.
              sampling_temperature: Sampling temperature to generate more random samples.
              return_log_prob: Include the token log probability in the result.
              repetition_penalty: Penalty applied to the score of previously generated tokens
                (set > 1 to penalize).
              no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                (set 0 to disable).
              disable_unk: Disable the generation of the unknown token.
              suppress_sequences: Disable the generation of some sequences of tokens.
              end_token: Stop the decoding on one these tokens (defaults to the model EOS token).
              static_prompt: If the model expects a static prompt (a.k.a. system prompt)
                it can be set here to simplify the inputs and optionally cache the model
                state for this prompt to accelerate future generations.
              cache_static_prompt: Cache the model state after the static prompt and
                reuse it for future generations using the same static prompt.

            Returns:
              A generator iterator over :class:`ctranslate2.GenerationStepResult` instances.

            Note:
              This generation method is not compatible with beam search which requires a complete decoding.
        """
        pass

    def score_batch(
        self, tokens, List=None, p_str=None, *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        score_batch(self: ctranslate2._ext.Generator, tokens: List[List[str]], *, max_batch_size: int = 0, batch_type: str = 'examples', max_input_length: int = 1024, asynchronous: bool = False) -> Union[List[ctranslate2._ext.ScoringResult], List[ctranslate2._ext.AsyncScoringResult]]


                         Scores a batch of tokens.

                         Arguments:
                           tokens: Batch of tokens to score. If the model expects special start or end tokens,
                             they should also be added to this input.
                           max_batch_size: The maximum batch size. If the number of inputs is greater than
                             :obj:`max_batch_size`, the inputs are sorted by length and split by chunks of
                             :obj:`max_batch_size` examples so that the number of padding positions is
                             minimized.
                           batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
                           max_input_length: Truncate inputs after this many tokens (0 to disable).
                           asynchronous: Run the scoring asynchronously.

                         Returns:
                           A list of scoring results.
        """
        pass

    def score_iterable(
        generator, tokens, max_batch_size=64, batch_type=None, **kwargs
    ):  # reliably restored by inspect
        """
        Scores an iterable of tokenized examples.

            This method is built on top of :meth:`ctranslate2.Generator.score_batch`
            to efficiently score an arbitrarily large stream of data. It enables
            the following optimizations:

            * stream processing (the iterable is not fully materialized in memory)
            * parallel scoring (if the generator has multiple workers)
            * asynchronous batch prefetching
            * local sorting by length

            Arguments:
              tokens: An iterable of tokenized examples.
              max_batch_size: The maximum batch size.
              batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
              **kwargs: Any score options accepted by
                :meth:`ctranslate2.Generator.score_batch`.

            Returns:
              A generator iterator over :class:`ctranslate2.ScoringResult` instances.
        """
        pass

    def __init__(
        self, model_path, device='cpu', *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        __init__(self: ctranslate2._ext.Generator, model_path: str, device: str = 'cpu', *, device_index: Union[int, List[int]] = 0, compute_type: Union[str, Dict[str, str]] = 'default', inter_threads: int = 1, intra_threads: int = 0, max_queued_batches: int = 0, files: object = None) -> None


                         Initializes the generator.

                         Arguments:
                           model_path: Path to the CTranslate2 model directory.
                           device: Device to use (possible values are: cpu, cuda, auto).
                           device_index: Device IDs where to place this generator on.
                           compute_type: Model computation type or a dictionary mapping a device name
                             to the computation type (possible values are: default, auto, int8, int8_float32,
                             int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
                           inter_threads: Maximum number of parallel generations.
                           intra_threads: Number of OpenMP threads per generator (0 to use a default value).
                           max_queued_batches: Maximum numbers of batches in the queue (-1 for unlimited,
                             0 for an automatic value). When the queue is full, future requests will block
                             until a free slot is available.
                           files: Load model files from the memory. This argument is a dictionary mapping
                             file names to file contents as file-like or bytes objects. If this is set,
                             :obj:`model_path` acts as an identifier for this model.
        """
        pass

    compute_type = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Computation type used by the model."""

    device = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Device this generator is running on."""

    device_index = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """List of device IDs where this generator is running on."""

    num_active_batches = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of batches waiting to be processed or currently processed."""

    num_generators = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of generators backing this instance."""

    num_queued_batches = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of batches waiting to be processed."""

class LogLevel(__pybind11_builtins.pybind11_object):
    # no doc
    def __eq__(self, other):  # real signature unknown; restored from __doc__
        """__eq__(self: object, other: object) -> bool"""
        return False

    def __getstate__(self):  # real signature unknown; restored from __doc__
        """__getstate__(self: object) -> int"""
        return 0

    def __hash__(self):  # real signature unknown; restored from __doc__
        """__hash__(self: object) -> int"""
        return 0

    def __index__(self):  # real signature unknown; restored from __doc__
        """__index__(self: ctranslate2._ext.LogLevel) -> int"""
        return 0

    def __init__(self, value):  # real signature unknown; restored from __doc__
        """__init__(self: ctranslate2._ext.LogLevel, value: int) -> None"""
        pass

    def __int__(self):  # real signature unknown; restored from __doc__
        """__int__(self: ctranslate2._ext.LogLevel) -> int"""
        return 0

    def __ne__(self, other):  # real signature unknown; restored from __doc__
        """__ne__(self: object, other: object) -> bool"""
        return False

    def __repr__(self):  # real signature unknown; restored from __doc__
        """__repr__(self: object) -> str"""
        return ''

    def __setstate__(self, state):  # real signature unknown; restored from __doc__
        """__setstate__(self: ctranslate2._ext.LogLevel, state: int) -> None"""
        pass

    def __str__(self, *args, **kwargs):  # real signature unknown
        """name(self: handle) -> str"""
        pass

    name = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """name(self: handle) -> str
"""

    value = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default

    Critical = None  # (!) forward: Critical, real value is '<LogLevel.Critical: -2>'
    Debug = None  # (!) forward: Debug, real value is '<LogLevel.Debug: 2>'
    Error = None  # (!) forward: Error, real value is '<LogLevel.Error: -1>'
    Info = None  # (!) forward: Info, real value is '<LogLevel.Info: 1>'
    Off = None  # (!) forward: Off, real value is '<LogLevel.Off: -3>'
    Trace = None  # (!) forward: Trace, real value is '<LogLevel.Trace: 3>'
    Warning = None  # (!) forward: Warning, real value is '<LogLevel.Warning: 0>'
    __entries = {
        'Critical': (
            None,  # (!) forward: Critical, real value is '<LogLevel.Critical: -2>'
            None,
        ),
        'Debug': (
            None,  # (!) forward: Debug, real value is '<LogLevel.Debug: 2>'
            None,
        ),
        'Error': (
            None,  # (!) forward: Error, real value is '<LogLevel.Error: -1>'
            None,
        ),
        'Info': (
            None,  # (!) forward: Info, real value is '<LogLevel.Info: 1>'
            None,
        ),
        'Off': (
            None,  # (!) forward: Off, real value is '<LogLevel.Off: -3>'
            None,
        ),
        'Trace': (
            None,  # (!) forward: Trace, real value is '<LogLevel.Trace: 3>'
            None,
        ),
        'Warning': (
            None,  # (!) forward: Warning, real value is '<LogLevel.Warning: 0>'
            None,
        ),
    }
    __members__ = {
        'Critical': None,  # (!) forward: Critical, real value is '<LogLevel.Critical: -2>'
        'Debug': None,  # (!) forward: Debug, real value is '<LogLevel.Debug: 2>'
        'Error': None,  # (!) forward: Error, real value is '<LogLevel.Error: -1>'
        'Info': None,  # (!) forward: Info, real value is '<LogLevel.Info: 1>'
        'Off': None,  # (!) forward: Off, real value is '<LogLevel.Off: -3>'
        'Trace': None,  # (!) forward: Trace, real value is '<LogLevel.Trace: 3>'
        'Warning': None,  # (!) forward: Warning, real value is '<LogLevel.Warning: 0>'
    }

class ScoringResult(__pybind11_builtins.pybind11_object):
    """A scoring result."""
    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __repr__(self):  # real signature unknown; restored from __doc__
        """__repr__(self: ctranslate2._ext.ScoringResult) -> str"""
        return ''

    log_probs = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Log probability of each token"""

    tokens = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """The scored tokens."""

class StorageView(__pybind11_builtins.pybind11_object):
    """
    An allocated buffer with shape information.

                The object implements the
                `Array Interface <https://numpy.org/doc/stable/reference/arrays.interface.html>`_
                and the
                `CUDA Array Interface <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>`_
                so that it can be passed to Numpy or PyTorch without copy.

                Example:

                    >>> x = np.ones((2, 4), dtype=np.int32)
                    >>> y = ctranslate2.StorageView.from_array(x)
                    >>> print(y)
                     1 1 1 ... 1 1 1
                    [cpu:0 int32 storage viewed as 2x4]
                    >>> z = np.array(y)
                    ...
                    >>> x = torch.ones((2, 4), dtype=torch.int32, device="cuda")
                    >>> y = ctranslate2.StorageView.from_array(x)
                    >>> print(y)
                     1 1 1 ... 1 1 1
                    [cuda:0 int32 storage viewed as 2x4]
                    >>> z = torch.as_tensor(y, device="cuda")
    """
    def from_array(self, array):  # real signature unknown; restored from __doc__
        """
        from_array(array: object) -> ctranslate2._ext.StorageView


                                Creates a ``StorageView`` from an object implementing the array interface.

                                Arguments:
                                  array: An object implementing the array interface (e.g. a Numpy array
                                    or a PyTorch Tensor).

                                Returns:
                                  A new ``StorageView`` instance sharing the same data as the input array.

                                Raises:
                                  ValueError: if the object does not implement the array interface or
                                    uses an unsupported array specification.
        """
        pass

    def to(self, dtype):  # real signature unknown; restored from __doc__
        """
        to(self: ctranslate2._ext.StorageView, dtype: ctranslate2._ext.DataType) -> ctranslate2._ext.StorageView


                         Converts the storage to another type.

                         Arguments:
                           dtype: The data type to convert to.

                         Returns:
                           A new ``StorageView`` instance.
        """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __str__(self):  # real signature unknown; restored from __doc__
        """__str__(self: ctranslate2._ext.StorageView) -> str"""
        return ''

    device = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Device where the storage is allocated ("cpu" or "cuda")."""

    device_index = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Device index."""

    dtype = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Data type used by the storage."""

    shape = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Shape of the storage view."""

    __array_interface__ = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default

    __cuda_array_interface__ = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default

class TranslationResult(__pybind11_builtins.pybind11_object):
    """A translation result."""
    def __getitem__(self, arg0):  # real signature unknown; restored from __doc__
        """__getitem__(self: ctranslate2._ext.TranslationResult, arg0: int) -> dict"""
        return {}

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __len__(self):  # real signature unknown; restored from __doc__
        """__len__(self: ctranslate2._ext.TranslationResult) -> int"""
        return 0

    def __repr__(self):  # real signature unknown; restored from __doc__
        """__repr__(self: ctranslate2._ext.TranslationResult) -> str"""
        return ''

    attention = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Attention matrix of each translation hypothesis (empty if :obj:`return_attention` was disabled)."""

    hypotheses = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Translation hypotheses."""

    scores = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Score of each translation hypothesis (empty if :obj:`return_scores` was disabled)."""

class Translator(__pybind11_builtins.pybind11_object):
    """
    A text translator.

                Example:

                    >>> translator = ctranslate2.Translator("model/", device="cpu")
                    >>> translator.translate_batch([["▁Hello", "▁world", "!"]])
    """
    def generate_tokens(
        translator,
        source,
        target_prefix=None,
        *,
        max_decoding_length=256,
        min_decoding_length=1,
        sampling_topk=1,
        sampling_topp=1,
        sampling_temperature=1,
        return_log_prob=False,
        repetition_penalty=1,
        no_repeat_ngram_size=0,
        disable_unk=False,
        suppress_sequences=None,
        end_token=None,
        max_input_length=1024,
        use_vmap=False,
    ):  # reliably restored by inspect
        """
        Yields tokens as they are generated by the model.

            Arguments:
              source: Source tokens.
              target_prefix: Optional target prefix tokens.
              max_decoding_length: Maximum prediction length.
              min_decoding_length: Minimum prediction length.
              sampling_topk: Randomly sample predictions from the top K candidates.
              sampling_topp: Keep the most probable tokens whose cumulative probability exceeds this value.
              sampling_temperature: Sampling temperature to generate more random samples.
              return_log_prob: Include the token log probability in the result.
              repetition_penalty: Penalty applied to the score of previously generated tokens
                (set > 1 to penalize).
              no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                (set 0 to disable).
              disable_unk: Disable the generation of the unknown token.
              suppress_sequences: Disable the generation of some sequences of tokens.
              end_token: Stop the decoding on one of these tokens (defaults to the model EOS token).
              max_input_length: Truncate inputs after this many tokens (set 0 to disable).
              use_vmap: Use the vocabulary mapping file saved in this model

            Returns:
              A generator iterator over :class:`ctranslate2.GenerationStepResult` instances.

            Note:
              This generation method is not compatible with beam search which requires a complete decoding.
        """
        pass

    def load_model(self):  # real signature unknown; restored from __doc__
        """
        load_model(self: ctranslate2._ext.Translator) -> None

        Loads the model back to the initial device.
        """
        pass

    def score_batch(
        self, source, List=None, p_str=None, *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        score_batch(self: ctranslate2._ext.Translator, source: List[List[str]], target: List[List[str]], *, max_batch_size: int = 0, batch_type: str = 'examples', max_input_length: int = 1024, asynchronous: bool = False) -> Union[List[ctranslate2._ext.ScoringResult], List[ctranslate2._ext.AsyncScoringResult]]


                         Scores a batch of parallel tokens.

                         Arguments:
                           source: Batch of source tokens.
                           target: Batch of target tokens.
                           max_batch_size: The maximum batch size. If the number of inputs is greater than
                             :obj:`max_batch_size`, the inputs are sorted by length and split by chunks of
                             :obj:`max_batch_size` examples so that the number of padding positions is
                             minimized.
                           batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
                           max_input_length: Truncate inputs after this many tokens (0 to disable).
                           asynchronous: Run the scoring asynchronously.

                         Returns:
                           A list of scoring results.
        """
        pass

    def score_file(
        self, source_path, target_path, output_path, *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        score_file(self: ctranslate2._ext.Translator, source_path: str, target_path: str, output_path: str, *, max_batch_size: int = 32, read_batch_size: int = 0, batch_type: str = 'examples', max_input_length: int = 1024, with_tokens_score: bool = False, source_tokenize_fn: Callable[[str], List[str]] = None, target_tokenize_fn: Callable[[str], List[str]] = None, target_detokenize_fn: Callable[[List[str]], str] = None) -> ctranslate2._ext.ExecutionStats


                         Scores a parallel tokenized text file.

                         Each line in :obj:`output_path` will have the format:

                         .. code-block:: text

                             <score> ||| <target> [||| <score_token_0> <score_token_1> ...]

                         The score is normalized by the target length which includes the end of sentence
                         token ``</s>``.

                         Arguments:
                           source_path: Path to the source file.
                           target_path: Path to the target file.
                           output_path: Path to the output file.
                           max_batch_size: The maximum batch size.
                           read_batch_size: The number of examples to read from the file before sorting
                             by length and splitting by chunks of :obj:`max_batch_size` examples
                             (set 0 for an automatic value).
                           batch_type: Whether :obj:`max_batch_size` and :obj:`read_batch_size` are the
                             number of "examples" or "tokens".
                           max_input_length: Truncate inputs after this many tokens (0 to disable).
                           with_tokens_score: Include the token-level scores in the output file.
                           source_tokenize_fn: Function to tokenize source lines.
                           target_tokenize_fn: Function to tokenize target lines.
                           target_detokenize_fn: Function to detokenize target outputs.

                         Returns:
                           A statistics object.
        """
        pass

    def score_iterable(
        translator, source, target, max_batch_size=64, batch_type=None, **kwargs
    ):  # reliably restored by inspect
        """
        Scores an iterable of tokenized examples.

            This method is built on top of :meth:`ctranslate2.Translator.score_batch`
            to efficiently score an arbitrarily large stream of data. It enables the
            following optimizations:

            * stream processing (the iterable is not fully materialized in memory)
            * parallel scoring (if the translator has multiple workers)
            * asynchronous batch prefetching
            * local sorting by length

            Arguments:
              source: An iterable of tokenized source examples.
              target: An iterable of tokenized target examples.
              max_batch_size: The maximum batch size.
              batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
              **kwargs: Any scoring options accepted by
                :meth:`ctranslate2.Translator.score_batch`.

            Returns:
              A generator iterator over :class:`ctranslate2.ScoringResult` instances.
        """
        pass

    def translate_batch(
        self, source, List=None, p_str=None, *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        translate_batch(self: ctranslate2._ext.Translator, source: List[List[str]], target_prefix: Optional[List[Optional[List[str]]]] = None, *, max_batch_size: int = 0, batch_type: str = 'examples', asynchronous: bool = False, beam_size: int = 2, patience: float = 1, num_hypotheses: int = 1, length_penalty: float = 1, coverage_penalty: float = 0, repetition_penalty: float = 1, no_repeat_ngram_size: int = 0, disable_unk: bool = False, suppress_sequences: Optional[List[List[str]]] = None, end_token: Optional[Union[str, List[str], List[int]]] = None, return_end_token: bool = False, prefix_bias_beta: float = 0, max_input_length: int = 1024, max_decoding_length: int = 256, min_decoding_length: int = 1, use_vmap: bool = False, return_scores: bool = False, return_attention: bool = False, return_alternatives: bool = False, min_alternative_expansion_prob: float = 0, sampling_topk: int = 1, sampling_topp: float = 1, sampling_temperature: float = 1, replace_unknowns: bool = False, callback: Callable[[ctranslate2._ext.GenerationStepResult], bool] = None) -> Union[List[ctranslate2._ext.TranslationResult], List[ctranslate2._ext.AsyncTranslationResult]]


                         Translates a batch of tokens.

                         Arguments:
                           source: Batch of source tokens.
                           target_prefix: Optional batch of target prefix tokens.
                           max_batch_size: The maximum batch size. If the number of inputs is greater than
                             :obj:`max_batch_size`, the inputs are sorted by length and split by chunks of
                             :obj:`max_batch_size` examples so that the number of padding positions is
                             minimized.
                           batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
                           asynchronous: Run the translation asynchronously.
                           beam_size: Beam size (1 for greedy search).
                           patience: Beam search patience factor, as described in
                             https://arxiv.org/abs/2204.05424. The decoding will continue until
                             beam_size*patience hypotheses are finished.
                           num_hypotheses: Number of hypotheses to return.
                           length_penalty: Exponential penalty applied to the length during beam search.
                           coverage_penalty: Coverage penalty weight applied during beam search.
                           repetition_penalty: Penalty applied to the score of previously generated tokens
                             (set > 1 to penalize).
                           no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                             (set 0 to disable).
                           disable_unk: Disable the generation of the unknown token.
                           suppress_sequences: Disable the generation of some sequences of tokens.
                           end_token: Stop the decoding on one of these tokens (defaults to the model EOS token).
                           return_end_token: Include the end token in the results.
                           prefix_bias_beta: Parameter for biasing translations towards given prefix.
                           max_input_length: Truncate inputs after this many tokens (set 0 to disable).
                           max_decoding_length: Maximum prediction length.
                           min_decoding_length: Minimum prediction length.
                           use_vmap: Use the vocabulary mapping file saved in this model
                           return_scores: Include the scores in the output.
                           return_attention: Include the attention vectors in the output.
                           return_alternatives: Return alternatives at the first unconstrained decoding position.
                           min_alternative_expansion_prob: Minimum initial probability to expand an alternative.
                           sampling_topk: Randomly sample predictions from the top K candidates.
                           sampling_topp: Keep the most probable tokens whose cumulative probability exceeds
                             this value.
                           sampling_temperature: Sampling temperature to generate more random samples.
                           replace_unknowns: Replace unknown target tokens by the source token with the highest attention.
                           callback: Optional function that is called for each generated token when
                             :obj:`beam_size` is 1. If the callback function returns ``True``, the
                             decoding will stop for this batch.

                         Returns:
                           A list of translation results.

                         See Also:
                           `TranslationOptions <https://github.com/OpenNMT/CTranslate2/blob/master/include/ctranslate2/translation.h>`_ structure in the C++ library.
        """
        pass

    def translate_file(
        self, source_path, output_path, target_path, p_str=None, *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        translate_file(self: ctranslate2._ext.Translator, source_path: str, output_path: str, target_path: Optional[str] = None, *, max_batch_size: int = 32, read_batch_size: int = 0, batch_type: str = 'examples', beam_size: int = 2, patience: float = 1, num_hypotheses: int = 1, length_penalty: float = 1, coverage_penalty: float = 0, repetition_penalty: float = 1, no_repeat_ngram_size: int = 0, disable_unk: bool = False, suppress_sequences: Optional[List[List[str]]] = None, end_token: Optional[Union[str, List[str], List[int]]] = None, prefix_bias_beta: float = 0, max_input_length: int = 1024, max_decoding_length: int = 256, min_decoding_length: int = 1, use_vmap: bool = False, with_scores: bool = False, sampling_topk: int = 1, sampling_topp: float = 1, sampling_temperature: float = 1, replace_unknowns: bool = False, source_tokenize_fn: Callable[[str], List[str]] = None, target_tokenize_fn: Callable[[str], List[str]] = None, target_detokenize_fn: Callable[[List[str]], str] = None) -> ctranslate2._ext.ExecutionStats


                         Translates a tokenized text file.

                         Arguments:
                           source_path: Path to the source file.
                           output_path: Path to the output file.
                           target_path: Path to the target prefix file.
                           max_batch_size: The maximum batch size.
                           read_batch_size: The number of examples to read from the file before sorting
                             by length and splitting by chunks of :obj:`max_batch_size` examples
                             (set 0 for an automatic value).
                           batch_type: Whether :obj:`max_batch_size` and :obj:`read_batch_size` are the
                             numbers of "examples" or "tokens".
                           asynchronous: Run the translation asynchronously.
                           beam_size: Beam size (1 for greedy search).
                           patience: Beam search patience factor, as described in
                             https://arxiv.org/abs/2204.05424. The decoding will continue until
                             beam_size*patience hypotheses are finished.
                           num_hypotheses: Number of hypotheses to return.
                           length_penalty: Exponential penalty applied to the length during beam search.
                           coverage_penalty: Coverage penalty weight applied during beam search.
                           repetition_penalty: Penalty applied to the score of previously generated tokens
                             (set > 1 to penalize).
                           no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                             (set 0 to disable).
                           disable_unk: Disable the generation of the unknown token.
                           suppress_sequences: Disable the generation of some sequences of tokens.
                           end_token: Stop the decoding on one of these tokens (defaults to the model EOS token).
                           prefix_bias_beta: Parameter for biasing translations towards given prefix.
                           max_input_length: Truncate inputs after this many tokens (set 0 to disable).
                           max_decoding_length: Maximum prediction length.
                           min_decoding_length: Minimum prediction length.
                           use_vmap: Use the vocabulary mapping file saved in this model
                           with_scores: Include the scores in the output.
                           sampling_topk: Randomly sample predictions from the top K candidates.
                           sampling_topp: Keep the most probable tokens whose cumulative probability exceeds
                             this value.
                           sampling_temperature: Sampling temperature to generate more random samples.
                           replace_unknowns: Replace unknown target tokens by the source token with the highest attention.
                           source_tokenize_fn: Function to tokenize source lines.
                           target_tokenize_fn: Function to tokenize target lines.
                           target_detokenize_fn: Function to detokenize target outputs.

                         Returns:
                           A statistics object.

                         See Also:
                           `TranslationOptions <https://github.com/OpenNMT/CTranslate2/blob/master/include/ctranslate2/translation.h>`_ structure in the C++ library.
        """
        pass

    def translate_iterable(
        translator,
        source,
        target_prefix=None,
        max_batch_size=32,
        batch_type=None,
        **kwargs,
    ):  # reliably restored by inspect
        """
        Translates an iterable of tokenized examples.

            This method is built on top of :meth:`ctranslate2.Translator.translate_batch`
            to efficiently translate an arbitrarily large stream of data. It enables the
            following optimizations:

            * stream processing (the iterable is not fully materialized in memory)
            * parallel translations (if the translator has multiple workers)
            * asynchronous batch prefetching
            * local sorting by length

            Arguments:
              source: An iterable of tokenized source examples.
              target_prefix: An optional iterable of tokenized target prefixes.
              max_batch_size: The maximum batch size.
              batch_type: Whether :obj:`max_batch_size` is the number of "examples" or "tokens".
              **kwargs: Any translation options accepted by
                :meth:`ctranslate2.Translator.translate_batch`.

            Returns:
              A generator iterator over :class:`ctranslate2.TranslationResult` instances.

            Example:
              This method can be used to efficiently translate text files:

              .. code-block:: python

                  # Replace by your own tokenization and detokenization functions.
                  tokenize_fn = lambda line: line.strip().split()
                  detokenize_fn = lambda tokens: " ".join(tokens)

                  with open("input.txt") as input_file:
                      source = map(tokenize_fn, input_file)
                      results = translator.translate_iterable(source, max_batch_size=64)

                      for result in results:
                          tokens = result.hypotheses[0]
                          target = detokenize_fn(tokens)
                          print(target)
        """
        pass

    def unload_model(
        self, to_cpu=False
    ):  # real signature unknown; restored from __doc__
        """
        unload_model(self: ctranslate2._ext.Translator, to_cpu: bool = False) -> None


                         Unloads the model attached to this translator but keep enough runtime context
                         to quickly resume translation on the initial device. The model is not guaranteed
                         to be unloaded if translations are running concurrently.

                         Arguments:
                           to_cpu: If ``True``, the model is moved to the CPU memory and not fully unloaded.
        """
        pass

    def __init__(
        self, model_path, device='cpu', *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        __init__(self: ctranslate2._ext.Translator, model_path: str, device: str = 'cpu', *, device_index: Union[int, List[int]] = 0, compute_type: Union[str, Dict[str, str]] = 'default', inter_threads: int = 1, intra_threads: int = 0, max_queued_batches: int = 0, files: object = None) -> None


                         Initializes the translator.

                         Arguments:
                           model_path: Path to the CTranslate2 model directory.
                           device: Device to use (possible values are: cpu, cuda, auto).
                           device_index: Device IDs where to place this generator on.
                           compute_type: Model computation type or a dictionary mapping a device name
                             to the computation type (possible values are: default, auto, int8, int8_float32,
                             int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
                           inter_threads: Maximum number of parallel translations.
                           intra_threads: Number of OpenMP threads per translator (0 to use a default value).
                           max_queued_batches: Maximum numbers of batches in the queue (-1 for unlimited,
                             0 for an automatic value). When the queue is full, future requests will block
                             until a free slot is available.
                           files: Load model files from the memory. This argument is a dictionary mapping
                             file names to file contents as file-like or bytes objects. If this is set,
                             :obj:`model_path` acts as an identifier for this model.
        """
        pass

    compute_type = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Computation type used by the model."""

    device = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Device this translator is running on."""

    device_index = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """List of device IDs where this translator is running on."""

    model_is_loaded = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Whether the model is loaded on the initial device and ready to be used."""

    num_active_batches = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of batches waiting to be processed or currently processed."""

    num_queued_batches = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of batches waiting to be processed."""

    num_translators = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of translators backing this instance."""

class Whisper:
    """
    Implements the Whisper speech recognition model published by OpenAI.

                See Also:
                   https://github.com/openai/whisper
    """
    def align(
        self, features, start_sequence, p_int=None, *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        align(self: ctranslate2._ext.Whisper, features: ctranslate2._ext.StorageView, start_sequence: List[int], text_tokens: List[List[int]], num_frames: Union[int, List[int]], *, median_filter_width: int = 7) -> List[ctranslate2._ext.WhisperAlignmentResult]


                         Computes the alignments between the text tokens and the audio.

                         Arguments:
                           features: Mel spectogram of the audio, as a float array with shape
                             ``[batch_size, 80, chunk_length]``. This method also accepts the encoded
                             features returned by the method :meth:`ctranslate2.models.Whisper.encode`,
                             which have shape ``[batch_size, chunk_length // 2, d_model]``.
                           start_sequence: The start sequence tokens.
                           text_tokens: Batch of text tokens to align.
                           num_frames: Number of non padding frames in the features.
                           median_filter_width: Width of the median filter kernel.

                         Returns:
                           A list of alignment results.
        """
        pass

    def detect_language(
        self, features: ctranslate2.StorageView
    ) -> list[list[tuple[str, float]]]:  # real signature unknown; restored from __doc__
        """
        detect_language(self: ctranslate2._ext.Whisper, features: ctranslate2._ext.StorageView) -> List[List[Tuple[str, float]]]


                         Returns the probability of each language.

                         Arguments:
                           features: Mel spectogram of the audio, as a float array with shape
                             ``[batch_size, 80, chunk_length]``. This method also accepts the encoded
                             features returned by the method :meth:`ctranslate2.models.Whisper.encode`,
                             which have shape ``[batch_size, chunk_length // 2, d_model]``.

                         Returns:
                           For each batch, a list of pairs (language, probability) ordered from
                           best to worst probability.

                         Raises:
                           RuntimeError: if the model is not multilingual.
        """
        return []

    def encode(
        self, features: ctranslate2.StorageView, to_cpu: bool = False
    ) -> ctranslate2.StorageView:  # real signature unknown; restored from __doc__
        """
        encode(self: ctranslate2._ext.Whisper, features: ctranslate2._ext.StorageView, to_cpu: bool = False) -> ctranslate2._ext.StorageView


                         Encodes the input features.

                         Arguments:
                           features: Mel spectogram of the audio, as a float array with shape
                             ``[batch_size, 80, chunk_length]``.
                           to_cpu: Copy the encoder output to the CPU before returning the value.

                         Returns:
                           The encoder output.
        """
        pass

    @overload
    def generate(
        self,
        features: ctranslate2.StorageView,
        prompts: list[list[str]] | list[list[int]],
        *,
        asyncronous: Literal[False] = ...,
        beam_size: int = 5,
        patience: float = 1,
        num_hypotheses: int = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        max_length: int = 448,
        return_scores: bool = False,
        return_no_speech_prob: bool = False,
        max_initial_timestamp_index: int = 50,
        suppress_blank: bool = True,
        suppress_tokens: list[int] | None = [-1],
        sampling_topk: int = 1,
        sampling_temperature: float = 1,
    ) -> list[ctranslate2.WhisperGenerationResult]: ...
    @overload
    def generate(
        self,
        features: ctranslate2.StorageView,
        prompts: list[list[str]] | list[list[int]],
        *,
        asyncronous: Literal[True] = ...,
        beam_size: int = 5,
        patience: float = 1,
        num_hypotheses: int = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        max_length: int = 448,
        return_scores: bool = False,
        return_no_speech_prob: bool = False,
        max_initial_timestamp_index: int = 50,
        suppress_blank: bool = True,
        suppress_tokens: list[int] | None = [-1],
        sampling_topk: int = 1,
        sampling_temperature: float = 1,
    ) -> list[ctranslate2.WhisperGenerationResultAsync]: ...
    def generate(
        self,
        features: ctranslate2.StorageView,
        prompts: list[list[str]] | list[list[int]],
        *,
        asyncronous: bool = False,
        beam_size: int = 5,
        patience: float = 1,
        num_hypotheses: int = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        max_length: int = 448,
        return_scores: bool = False,
        return_no_speech_prob: bool = False,
        max_initial_timestamp_index: int = 50,
        suppress_blank: bool = True,
        suppress_tokens: list[int] | None = [-1],
        sampling_topk: int = 1,
        sampling_temperature: float = 1,
    ) -> (
        list[ctranslate2.WhisperGenerationResult]
        | list[ctranslate2.WhisperGenerationResultAsync]
    ):
        """
        generate(self: ctranslate2._ext.Whisper, features: ctranslate2._ext.StorageView, prompts: Union[List[List[str]], List[List[int]]], *, asynchronous: bool = False, beam_size: int = 5, patience: float = 1, num_hypotheses: int = 1, length_penalty: float = 1, repetition_penalty: float = 1, no_repeat_ngram_size: int = 0, max_length: int = 448, return_scores: bool = False, return_no_speech_prob: bool = False, max_initial_timestamp_index: int = 50, suppress_blank: bool = True, suppress_tokens: Optional[List[int]] = [-1], sampling_topk: int = 1, sampling_temperature: float = 1) -> Union[List[ctranslate2._ext.WhisperGenerationResult], List[ctranslate2._ext.WhisperGenerationResultAsync]]


                         Encodes the input features and generates from the given prompt.

                         Arguments:
                           features: Mel spectogram of the audio, as a float array with shape
                             ``[batch_size, 80, chunk_length]``. This method also accepts the encoded
                             features returned by the method :meth:`ctranslate2.models.Whisper.encode`,
                             which have shape ``[batch_size, chunk_length // 2, d_model]``.
                           prompts: Batch of initial string tokens or token IDs.
                           asynchronous: Run the model asynchronously.
                           beam_size: Beam size (1 for greedy search).
                           patience: Beam search patience factor, as described in
                             https://arxiv.org/abs/2204.05424. The decoding will continue until
                             beam_size*patience hypotheses are finished.
                           num_hypotheses: Number of hypotheses to return.
                           length_penalty: Exponential penalty applied to the length during beam search.
                           repetition_penalty: Penalty applied to the score of previously generated tokens
                             (set > 1 to penalize).
                           no_repeat_ngram_size: Prevent repetitions of ngrams with this size
                             (set 0 to disable).
                           max_length: Maximum generation length.
                           return_scores: Include the scores in the output.
                           return_no_speech_prob: Include the probability of the no speech token in the
                             result.
                           max_initial_timestamp_index: Maximum index of the first predicted timestamp.
                           suppress_blank: Suppress blank outputs at the beginning of the sampling.
                           suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
                             of symbols as defined in the model ``config.json`` file.
                           sampling_topk: Randomly sample predictions from the top K candidates.
                           sampling_temperature: Sampling temperature to generate more random samples.

                         Returns:
                           A list of generation results.
        """
        pass

    def __init__(
        self, model_path, device='cpu', *args, **kwargs
    ):  # real signature unknown; NOTE: unreliably restored from __doc__
        """
        __init__(self: ctranslate2._ext.Whisper, model_path: str, device: str = 'cpu', *, device_index: Union[int, List[int]] = 0, compute_type: Union[str, Dict[str, str]] = 'default', inter_threads: int = 1, intra_threads: int = 0, max_queued_batches: int = 0, files: object = None) -> None


                         Initializes a Whisper model from a converted model.

                         Arguments:
                           model_path: Path to the CTranslate2 model directory.
                           device: Device to use (possible values are: cpu, cuda, auto).
                           device_index: Device IDs where to place this model on.
                           compute_type: Model computation type or a dictionary mapping a device name
                             to the computation type (possible values are: default, auto, int8, int8_float32,
                             int8_float16, int8_bfloat16, int16, float16, bfloat16, float32).
                           inter_threads: Number of workers to allow executing multiple batches in parallel.
                           intra_threads: Number of OpenMP threads per worker (0 to use a default value).
                           max_queued_batches: Maximum numbers of batches in the worker queue (-1 for unlimited,
                             0 for an automatic value). When the queue is full, future requests will block
                             until a free slot is available.
                           files: Load model files from the memory. This argument is a dictionary mapping
                             file names to file contents as file-like or bytes objects. If this is set,
                             :obj:`model_path` acts as an identifier for this model.
        """
        pass

    compute_type = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Computation type used by the model."""

    device = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Device this model is running on."""

    device_index = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """List of device IDs where this model is running on."""

    is_multilingual = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Returns ``True`` if this model is multilingual."""

    num_active_batches = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of batches waiting to be processed or currently processed."""

    num_queued_batches = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of batches waiting to be processed."""

    num_workers = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Number of model workers backing this instance."""

class WhisperAlignmentResult(__pybind11_builtins.pybind11_object):
    """An alignment result from the Whisper model."""
    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __repr__(self):  # real signature unknown; restored from __doc__
        """__repr__(self: ctranslate2._ext.WhisperAlignmentResult) -> str"""
        return ''

    alignments = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """List of aligned text and time indices."""

    text_token_probs = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Probabilities of text tokens."""

class WhisperGenerationResult(__pybind11_builtins.pybind11_object):
    """A generation result from the Whisper model."""
    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __repr__(self):  # real signature unknown; restored from __doc__
        """__repr__(self: ctranslate2._ext.WhisperGenerationResult) -> str"""
        return ''

    no_speech_prob = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Probability of the no speech token (0 if :obj:`return_no_speech_prob` was disabled)."""

    scores = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Score of each sequence (empty if :obj:`return_scores` was disabled)."""

    sequences = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Generated sequences of tokens."""

    sequences_ids = property(
        lambda self: object(), lambda self, v: None, lambda self: None
    )  # default
    """Generated sequences of token IDs."""

class WhisperGenerationResultAsync(__pybind11_builtins.pybind11_object):
    """Asynchronous wrapper around a result object."""
    def done(self):  # real signature unknown; restored from __doc__
        """
        done(self: ctranslate2._ext.WhisperGenerationResultAsync) -> bool

        Returns ``True`` if the result is available.
        """
        return False

    def result(self):  # real signature unknown; restored from __doc__
        """
        result(self: ctranslate2._ext.WhisperGenerationResultAsync) -> ctranslate2._ext.WhisperGenerationResult


                         Blocks until the result is available and returns it.

                         If an exception was raised when computing the result,
                         this method raises the exception.
        """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

# variables with complex values

Critical = None  # (!) real value is '<LogLevel.Critical: -2>'

Debug = None  # (!) real value is '<LogLevel.Debug: 2>'

Error = None  # (!) real value is '<LogLevel.Error: -1>'

Info = None  # (!) real value is '<LogLevel.Info: 1>'

Off = None  # (!) real value is '<LogLevel.Off: -3>'

Trace = None  # (!) real value is '<LogLevel.Trace: 3>'

Warning = None  # (!) real value is '<LogLevel.Warning: 0>'

__loader__ = None  # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x7fc697cccf10>'

__spec__ = None  # (!) real value is "ModuleSpec(name='ctranslate2._ext', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x7fc697cccf10>, origin='/mnt/data/code/Python/Tests/whisperx-test/whisperX/.venv/lib/python3.11/site-packages/ctranslate2/_ext.cpython-311-x86_64-linux-gnu.so')"
