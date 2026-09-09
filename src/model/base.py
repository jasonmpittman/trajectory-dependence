__author__ = "Jason M. Pittman"
__date__ = "August 23, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


class ModelAdapterError(RuntimeError):
    """Defined failure raised by a model adapter."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.message = message
        self.details = deepcopy(
            details or {}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": deepcopy(
                self.details
            ),
        }


@dataclass(frozen=True)
class InferenceConfig:
    """
    Explicit inference parameters for one model generation.

    The current experiment uses greedy argmax decoding. No sampler is
    supplied to mlx-lm.

    These fields are preserved even where they retain mlx-lm defaults so
    that runtime behavior can later be frozen explicitly.
    """

    max_tokens: int
    seed: int | None

    thinking_mode: bool = False
    decoding_strategy: str = "greedy_argmax"

    prefill_step_size: int = 2048
    max_kv_size: int | None = None

    kv_bits: int | None = None
    kv_group_size: int = 64
    quantized_kv_start: int = 0

    speculative_decoding: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.max_tokens,
                int,
            )
            or self.max_tokens <= 0
        ):
            raise ValueError(
                "max_tokens must be a positive integer."
            )

        if (
            self.seed is not None
            and not isinstance(
                self.seed,
                int,
            )
        ):
            raise ValueError(
                "seed must be an integer or None."
            )

        if (
            not isinstance(
                self.prefill_step_size,
                int,
            )
            or self.prefill_step_size <= 0
        ):
            raise ValueError(
                "prefill_step_size must be a positive integer."
            )

        if (
            self.max_kv_size is not None
            and (
                not isinstance(
                    self.max_kv_size,
                    int,
                )
                or self.max_kv_size <= 0
            )
        ):
            raise ValueError(
                "max_kv_size must be a positive integer or None."
            )

        if (
            self.kv_bits is not None
            and (
                not isinstance(
                    self.kv_bits,
                    int,
                )
                or self.kv_bits <= 0
            )
        ):
            raise ValueError(
                "kv_bits must be a positive integer or None."
            )

        if (
            not isinstance(
                self.kv_group_size,
                int,
            )
            or self.kv_group_size <= 0
        ):
            raise ValueError(
                "kv_group_size must be a positive integer."
            )

        if (
            not isinstance(
                self.quantized_kv_start,
                int,
            )
            or self.quantized_kv_start < 0
        ):
            raise ValueError(
                "quantized_kv_start must be a non-negative integer."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "thinking_mode": self.thinking_mode,
            "decoding_strategy": self.decoding_strategy,
            "prefill_step_size": self.prefill_step_size,
            "max_kv_size": self.max_kv_size,
            "kv_bits": self.kv_bits,
            "kv_group_size": self.kv_group_size,
            "quantized_kv_start": self.quantized_kv_start,
            "speculative_decoding": self.speculative_decoding,
        }


@dataclass(frozen=True)
class ChatMessage:
    """One exact chat-template message supplied by the adapter."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass(frozen=True)
class ModelInputRecord:
    """
    Exact auditable input prepared for one model invocation.

    `prompt_token_ids` is the token sequence that must be passed directly
    to the inference engine.
    """

    model_visible_text: str

    messages: tuple[ChatMessage, ...]

    rendered_prompt: str
    rendered_prompt_sha256: str

    prompt_token_ids: tuple[int, ...]
    prompt_token_ids_sha256: str

    prompt_token_count: int

    thinking_mode: bool
    chat_template_arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_visible_text": self.model_visible_text,
            "messages": [
                message.to_dict()
                for message in self.messages
            ],
            "rendered_prompt": self.rendered_prompt,
            "rendered_prompt_sha256": (
                self.rendered_prompt_sha256
            ),
            "prompt_token_ids": list(
                self.prompt_token_ids
            ),
            "prompt_token_ids_sha256": (
                self.prompt_token_ids_sha256
            ),
            "prompt_token_count": (
                self.prompt_token_count
            ),
            "thinking_mode": self.thinking_mode,
            "chat_template_arguments": deepcopy(
                self.chat_template_arguments
            ),
        }


@dataclass(frozen=True)
class ModelRuntimeMetadata:
    """Reproducibility metadata for the loaded model runtime."""

    model_identifier: str
    model_revision: str | None
    model_checksum: str | None
    quantization: str

    backend: str
    backend_version: str | None
    mlx_version: str | None

    python_version: str
    platform: str

    model_path_env: str

    device_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_identifier": self.model_identifier,
            "model_revision": self.model_revision,
            "model_checksum": self.model_checksum,
            "quantization": self.quantization,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "mlx_version": self.mlx_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "model_path_env": self.model_path_env,
            "device_info": deepcopy(
                self.device_info
            ),
        }


@dataclass(frozen=True)
class ModelGenerationResult:
    """
    Complete raw result from one model invocation.

    No task interpretation, action normalization, success grading, or
    causal classification occurs in this object.
    """

    raw_text: str
    raw_text_sha256: str

    generated_token_ids: tuple[int, ...]
    generated_token_ids_sha256: str

    finish_reason: str | None

    prompt_tokens: int
    generation_tokens: int

    prompt_tps: float | None
    generation_tps: float | None
    peak_memory_gb: float | None

    input_record: ModelInputRecord
    inference_config: InferenceConfig
    runtime_metadata: ModelRuntimeMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "raw_text_sha256": (
                self.raw_text_sha256
            ),
            "generated_token_ids": list(
                self.generated_token_ids
            ),
            "generated_token_ids_sha256": (
                self.generated_token_ids_sha256
            ),
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "generation_tokens": (
                self.generation_tokens
            ),
            "prompt_tps": self.prompt_tps,
            "generation_tps": (
                self.generation_tps
            ),
            "peak_memory_gb": (
                self.peak_memory_gb
            ),
            "input_record": (
                self.input_record.to_dict()
            ),
            "inference_config": (
                self.inference_config.to_dict()
            ),
            "runtime_metadata": (
                self.runtime_metadata.to_dict()
            ),
        }


class ModelAdapter(ABC):
    """
    Minimal model boundary used by the experimental harness.

    Implementations must not know about:

    - C0-C3;
    - tools;
    - memory;
    - environmental state;
    - task success;
    - focal decisions;
    - interventions;
    - self-explanation scoring.
    """

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return whether model and tokenizer are loaded."""
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        """Load model and tokenizer."""
        raise NotImplementedError

    @abstractmethod
    def runtime_metadata(
        self,
    ) -> ModelRuntimeMetadata:
        """Return metadata for the loaded runtime."""
        raise NotImplementedError

    @abstractmethod
    def prepare_input(
        self,
        model_visible_text: str,
    ) -> ModelInputRecord:
        """
        Render and tokenize the exact model-visible context.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        *,
        input_record: ModelInputRecord,
        config: InferenceConfig,
    ) -> ModelGenerationResult:
        """
        Generate from the exact prepared token sequence.
        """
        raise NotImplementedError