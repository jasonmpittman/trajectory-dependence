__author__ = "Jason M. Pittman"
__date__ = "September 9, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.3"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import hashlib
import json
import os
import platform
import sys
from copy import deepcopy
from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx
    from mlx_lm import (
        load as mlx_load,
        stream_generate,
    )
except Exception as mlx_import_error:
    # MLX may be installed on a headless macOS worker but still fail while
    # initializing Metal. Keep the model-free harness importable so its
    # preflight, replay, normalization, and provenance tests can run.
    _MLX_IMPORT_ERROR = mlx_import_error

    class _UnavailableRandom:
        @staticmethod
        def seed(_seed: int) -> None:
            return None

    class _UnavailableMLX:
        random = _UnavailableRandom()

        @staticmethod
        def reset_peak_memory() -> None:
            return None

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def device_info() -> dict[str, Any]:
            return {}

    mx = _UnavailableMLX()

    def mlx_load(*_args, **_kwargs):
        raise RuntimeError(
            "mlx-lm is unavailable or Metal could not be initialized."
        ) from _MLX_IMPORT_ERROR

    def stream_generate(*_args, **_kwargs):
        raise RuntimeError(
            "mlx-lm is unavailable or Metal could not be initialized."
        ) from _MLX_IMPORT_ERROR

from .base import (
    ChatMessage,
    InferenceConfig,
    ModelAdapter,
    ModelAdapterError,
    ModelGenerationResult,
    ModelInputRecord,
    ModelRuntimeMetadata,
)


class MLXQwenAdapter(ModelAdapter):
    """
    Local text-only Qwen adapter backed by mlx-lm.

    The adapter loads only from the filesystem location specified through
    `model_path_env`.

    It performs:

    - Qwen chat-template rendering;
    - explicit non-thinking configuration;
    - exact tokenization;
    - greedy argmax generation;
    - raw generation capture;
    - runtime/provenance metadata capture.

    It performs no agent or experimental interpretation.
    """

    def __init__(
        self,
        *,
        model_identifier: str,
        model_revision: str | None,
        model_checksum: str | None,
        quantization: str = "8bit",
        model_path_env: str = "TRAJECTORY_MODEL_PATH",
    ) -> None:
        if (
            not isinstance(
                model_identifier,
                str,
            )
            or not model_identifier
        ):
            raise ValueError(
                "model_identifier must be a non-empty string."
            )

        if (
            not isinstance(
                model_path_env,
                str,
            )
            or not model_path_env
        ):
            raise ValueError(
                "model_path_env must be a non-empty string."
            )

        self.model_identifier = (
            model_identifier
        )
        self.model_revision = (
            model_revision
        )
        self.model_checksum = (
            model_checksum
        )
        self.quantization = (
            quantization
        )
        self.model_path_env = (
            model_path_env
        )

        self._model = None
        self._tokenizer = None
        self._model_config = None
        self._runtime_metadata = None

    @property
    def is_loaded(self) -> bool:
        return (
            self._model is not None
            and self._tokenizer is not None
        )

    def load(self) -> None:
        """
        Load the model exclusively from the configured local path.

        Remote custom code is explicitly disabled.
        """

        configured = os.environ.get(
            self.model_path_env
        )

        if not configured:
            raise ModelAdapterError(
                code="model_path_not_configured",
                message=(
                    f"{self.model_path_env} is not set."
                ),
            )

        model_path = Path(
            configured
        ).expanduser().resolve()

        if not model_path.is_dir():
            raise ModelAdapterError(
                code="model_path_not_found",
                message=(
                    f"Configured model directory does not exist: "
                    f"{model_path}"
                ),
            )

        self._validate_local_model_directory(
            model_path
        )

        try:
            (
                model,
                tokenizer,
                model_config,
            ) = mlx_load(
                str(model_path),
                tokenizer_config={
                    "trust_remote_code": False,
                },
                return_config=True,
            )

        except Exception as exc:
            raise ModelAdapterError(
                code="model_load_failure",
                message=(
                    "mlx-lm failed to load the local model."
                ),
                details={
                    "model_path_env": self.model_path_env,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "exception_repr": repr(exc),
                },
            ) from exc

        self._model = model
        self._tokenizer = tokenizer
        self._model_config = deepcopy(
            model_config
        )

        self._runtime_metadata = (
            self._capture_runtime_metadata()
        )

    def runtime_metadata(
        self,
    ) -> ModelRuntimeMetadata:
        self._require_loaded()

        if self._runtime_metadata is None:
            raise ModelAdapterError(
                code="runtime_metadata_unavailable",
                message=(
                    "Runtime metadata was not captured."
                ),
            )

        return self._runtime_metadata

    def prepare_input(
        self,
        model_visible_text: str,
    ) -> ModelInputRecord:
        """
        Render and tokenize one exact user message.

        No system message is inserted by the adapter. Any task-level
        instruction required by the experiment must already be represented
        in model_visible_text.
        """

        self._require_loaded()

        if (
            not isinstance(
                model_visible_text,
                str,
            )
            or not model_visible_text
        ):
            raise ModelAdapterError(
                code="invalid_model_visible_text",
                message=(
                    "model_visible_text must be a non-empty string."
                ),
            )

        message = ChatMessage(
            role="user",
            content=model_visible_text,
        )

        template_arguments = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }

        try:
            rendered_prompt = (
                self._tokenizer
                .apply_chat_template(
                    [
                        message.to_dict()
                    ],
                    **template_arguments,
                )
            )

        except Exception as exc:
            raise ModelAdapterError(
                code="chat_template_failure",
                message=(
                    "Qwen chat-template rendering failed."
                ),
                details={
                    "exception_type": (
                        type(exc).__name__
                    ),
                },
            ) from exc

        if not isinstance(
            rendered_prompt,
            str,
        ):
            raise ModelAdapterError(
                code="invalid_rendered_prompt",
                message=(
                    "Chat template did not return a string."
                ),
            )

        if self._ends_with_open_thinking_block(
            rendered_prompt
        ):
            raise ModelAdapterError(
                code="open_thinking_block",
                message=(
                    "Non-thinking chat template ended with "
                    "an open <think> block."
                ),
            )

        try:
            token_ids = tuple(
                int(token_id)
                for token_id in (
                    self._tokenizer.encode(
                        rendered_prompt,
                        add_special_tokens=False,
                    )
                )
            )

        except Exception as exc:
            raise ModelAdapterError(
                code="tokenization_failure",
                message=(
                    "Qwen prompt tokenization failed."
                ),
                details={
                    "exception_type": (
                        type(exc).__name__
                    ),
                },
            ) from exc

        if not token_ids:
            raise ModelAdapterError(
                code="empty_tokenized_prompt",
                message=(
                    "Tokenized prompt contained no tokens."
                ),
            )

        return ModelInputRecord(
            model_visible_text=(
                model_visible_text
            ),
            messages=(
                message,
            ),
            rendered_prompt=(
                rendered_prompt
            ),
            rendered_prompt_sha256=(
                self._sha256_text(
                    rendered_prompt
                )
            ),
            prompt_token_ids=(
                token_ids
            ),
            prompt_token_ids_sha256=(
                self._sha256_token_ids(
                    token_ids
                )
            ),
            prompt_token_count=len(
                token_ids
            ),
            thinking_mode=False,
            chat_template_arguments=(
                deepcopy(
                    template_arguments
                )
            ),
        )

    def generate(
        self,
        *,
        input_record: ModelInputRecord,
        config: InferenceConfig,
    ) -> ModelGenerationResult:
        """
        Generate from the exact token sequence stored in input_record.

        No sampler is supplied to mlx-lm. The generation path therefore
        uses mlx-lm's greedy argmax default.
        """

        self._require_loaded()

        self._validate_inference_config(
            config
        )

        self._validate_input_record(
            input_record
        )

        if config.seed is not None:
            mx.random.seed(
                config.seed
            )

        # Reset the per-invocation peak measurement without clearing
        # model or KV-cache memory.
        mx.reset_peak_memory()

        text_parts: list[str] = []
        generated_token_ids: list[int] = []

        last_response = None

        try:
            responses = stream_generate(
                self._model,
                self._tokenizer,

                # These are the exact previously audited token IDs.
                prompt=list(
                    input_record.prompt_token_ids
                ),

                max_tokens=config.max_tokens,

                # Explicitly retain greedy argmax behavior.
                sampler=None,

                max_kv_size=(
                    config.max_kv_size
                ),
                prefill_step_size=(
                    config.prefill_step_size
                ),
                kv_bits=(
                    config.kv_bits
                ),
                kv_group_size=(
                    config.kv_group_size
                ),
                quantized_kv_start=(
                    config.quantized_kv_start
                ),
            )

            for response in responses:
                text_parts.append(
                    response.text
                )

                generated_token_ids.append(
                    int(
                        response.token
                    )
                )

                last_response = (
                    response
                )

            mx.synchronize()

        except Exception as exc:
            raise ModelAdapterError(
                code="generation_failure",
                message=(
                    "mlx-lm generation failed."
                ),
                details={
                    "exception_type": (
                        type(exc).__name__
                    ),
                    "input_prompt_sha256": (
                        input_record
                        .rendered_prompt_sha256
                    ),
                    "input_token_ids_sha256": (
                        input_record
                        .prompt_token_ids_sha256
                    ),
                },
            ) from exc

        if last_response is None:
            raise ModelAdapterError(
                code="empty_generation",
                message=(
                    "mlx-lm returned no generation responses."
                ),
            )

        raw_text = "".join(
            text_parts
        )

        if self._contains_nonempty_thinking_block(
            raw_text
        ):
            raise ModelAdapterError(
                code="unexpected_thinking_output",
                message=(
                    "Generation contained a non-empty "
                    "<think>...</think> block despite "
                    "thinking_mode=False."
                ),
                details={
                    "raw_text": raw_text,
                    "generated_token_ids": (
                        generated_token_ids
                    ),
                },
            )

        generated_ids = tuple(
            generated_token_ids
        )

        return ModelGenerationResult(
            raw_text=raw_text,
            raw_text_sha256=(
                self._sha256_text(
                    raw_text
                )
            ),
            generated_token_ids=(
                generated_ids
            ),
            generated_token_ids_sha256=(
                self._sha256_token_ids(
                    generated_ids
                )
            ),
            finish_reason=getattr(
                last_response,
                "finish_reason",
                None,
            ),
            prompt_tokens=int(
                getattr(
                    last_response,
                    "prompt_tokens",
                    input_record.prompt_token_count,
                )
            ),
            generation_tokens=int(
                getattr(
                    last_response,
                    "generation_tokens",
                    len(
                        generated_ids
                    ),
                )
            ),
            prompt_tps=self._optional_float(
                getattr(
                    last_response,
                    "prompt_tps",
                    None,
                )
            ),
            generation_tps=self._optional_float(
                getattr(
                    last_response,
                    "generation_tps",
                    None,
                )
            ),
            peak_memory_gb=self._optional_float(
                getattr(
                    last_response,
                    "peak_memory",
                    None,
                )
            ),
            input_record=input_record,
            inference_config=config,
            runtime_metadata=(
                self.runtime_metadata()
            ),
        )

    def _validate_input_record(
        self,
        input_record: ModelInputRecord,
    ) -> None:
        """
        Verify that the recorded prompt and token fingerprints remain
        internally consistent before inference.
        """

        expected_prompt_hash = (
            self._sha256_text(
                input_record.rendered_prompt
            )
        )

        if (
            expected_prompt_hash
            != input_record.rendered_prompt_sha256
        ):
            raise ModelAdapterError(
                code="rendered_prompt_checksum_mismatch",
                message=(
                    "Prepared rendered-prompt checksum does not match "
                    "its recorded value."
                ),
            )

        expected_token_hash = (
            self._sha256_token_ids(
                input_record.prompt_token_ids
            )
        )

        if (
            expected_token_hash
            != input_record.prompt_token_ids_sha256
        ):
            raise ModelAdapterError(
                code="prompt_token_checksum_mismatch",
                message=(
                    "Prepared token-ID checksum does not match "
                    "its recorded value."
                ),
            )

        if (
            len(
                input_record.prompt_token_ids
            )
            != input_record.prompt_token_count
        ):
            raise ModelAdapterError(
                code="prompt_token_count_mismatch",
                message=(
                    "Prepared token count does not match the "
                    "recorded token sequence."
                ),
            )

        try:
            retokenized = tuple(
                int(token_id)
                for token_id in (
                    self._tokenizer.encode(
                        input_record.rendered_prompt,
                        add_special_tokens=False,
                    )
                )
            )

        except Exception as exc:
            raise ModelAdapterError(
                code="retokenization_failure",
                message=(
                    "Unable to validate prepared prompt tokenization."
                ),
                details={
                    "exception_type": (
                        type(exc).__name__
                    ),
                },
            ) from exc

        if (
            retokenized
            != input_record.prompt_token_ids
        ):
            raise ModelAdapterError(
                code="prompt_tokenization_mismatch",
                message=(
                    "Retokenizing the rendered prompt produced "
                    "different token IDs."
                ),
            )

    @staticmethod
    def _validate_inference_config(
        config: InferenceConfig,
    ) -> None:
        if config.thinking_mode:
            raise ModelAdapterError(
                code="thinking_mode_not_supported",
                message=(
                    "The current experimental adapter requires "
                    "thinking_mode=False."
                ),
            )

        if (
            config.decoding_strategy
            != "greedy_argmax"
        ):
            raise ModelAdapterError(
                code="unsupported_decoding_strategy",
                message=(
                    "The current experimental adapter supports only "
                    "greedy_argmax decoding."
                ),
            )

        if config.speculative_decoding:
            raise ModelAdapterError(
                code="speculative_decoding_not_supported",
                message=(
                    "Speculative decoding is disabled for the current "
                    "experiment."
                ),
            )

    def _capture_runtime_metadata(
        self,
    ) -> ModelRuntimeMetadata:
        try:
            device_info = dict(
                mx.device_info()
            )
        except Exception:
            device_info = {}

        return ModelRuntimeMetadata(
            model_identifier=(
                self.model_identifier
            ),
            model_revision=(
                self.model_revision
            ),
            model_checksum=(
                self.model_checksum
            ),
            quantization=(
                self.quantization
            ),
            backend="mlx-lm",
            backend_version=(
                self._package_version(
                    "mlx-lm"
                )
            ),
            mlx_version=(
                self._package_version(
                    "mlx"
                )
            ),
            python_version=(
                sys.version.split()[0]
            ),
            platform=(
                platform.platform()
            ),
            model_path_env=(
                self.model_path_env
            ),
            device_info=device_info,
        )

    @staticmethod
    def _validate_local_model_directory(
        model_path: Path,
    ) -> None:
        required = (
            "config.json",
            "model.safetensors.index.json",
        )

        missing = [
            filename
            for filename in required
            if not (
                model_path
                / filename
            ).is_file()
        ]

        tokenizer_present = any(
            (
                model_path
                / filename
            ).is_file()
            for filename in (
                "tokenizer.json",
                "tokenizer.model",
            )
        )

        if not tokenizer_present:
            missing.append(
                "tokenizer.json or tokenizer.model"
            )

        if missing:
            raise ModelAdapterError(
                code="incomplete_local_model",
                message=(
                    "Local model directory is missing required "
                    "artifact(s): "
                    + ", ".join(
                        missing
                    )
                    + "."
                ),
            )

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise ModelAdapterError(
                code="model_not_loaded",
                message=(
                    "Model adapter has not been loaded."
                ),
            )

    @staticmethod
    def _package_version(
        package: str,
    ) -> str | None:
        try:
            return version(
                package
            )
        except PackageNotFoundError:
            return None

    @staticmethod
    def _sha256_text(
        value: str,
    ) -> str:
        return hashlib.sha256(
            value.encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _sha256_token_ids(
        token_ids,
    ) -> str:
        serialized = json.dumps(
            list(
                token_ids
            ),
            separators=(",", ":"),
        )

        return hashlib.sha256(
            serialized.encode(
                "utf-8"
            )
        ).hexdigest()

    @staticmethod
    def _optional_float(
        value,
    ) -> float | None:
        if value is None:
            return None

        return float(
            value
        )

    @staticmethod
    def _ends_with_open_thinking_block(
        text: str,
    ) -> bool:
        return (
            text
            .rstrip()
            .endswith(
                "<think>"
            )
        )

    @staticmethod
    def _contains_nonempty_thinking_block(
        text: str,
    ) -> bool:
        start_token = "<think>"
        end_token = "</think>"

        start = text.find(
            start_token
        )

        while start != -1:
            end = text.find(
                end_token,
                start
                + len(
                    start_token
                ),
            )

            if end == -1:
                return True

            body = text[
                start
                + len(
                    start_token
                ):
                end
            ]

            if body.strip():
                return True

            start = text.find(
                start_token,
                end
                + len(
                    end_token
                ),
            )

        return False
