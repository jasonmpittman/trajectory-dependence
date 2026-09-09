__author__ = "Jason M. Pittman"
__date__ = "August 23, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.model import (
    InferenceConfig,
    MLXQwenAdapter,
    ModelAdapterError,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.template_calls = []
        self.encode_calls = []

    def apply_chat_template(
        self,
        messages,
        **kwargs,
    ):
        self.template_calls.append(
            (
                messages,
                kwargs,
            )
        )

        content = messages[0][
            "content"
        ]

        return (
            "<|im_start|>user\n"
            f"{content}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

    def encode(
        self,
        text,
        add_special_tokens=False,
    ):
        self.encode_calls.append(
            (
                text,
                add_special_tokens,
            )
        )

        return [
            101,
            202,
            303,
        ]


class OpenThinkingTokenizer(
    FakeTokenizer
):
    def apply_chat_template(
        self,
        messages,
        **kwargs,
    ):
        return (
            "<|im_start|>user\n"
            "test"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>"
        )


def make_model_directory(
    tmp_path,
):
    model_path = (
        tmp_path
        / "model"
    )

    model_path.mkdir()

    (
        model_path
        / "config.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    (
        model_path
        / "tokenizer.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    (
        model_path
        / "model.safetensors.index.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    return model_path


def make_adapter() -> MLXQwenAdapter:
    return MLXQwenAdapter(
        model_identifier=(
            "mlx-community/"
            "Qwen3.5-27B-8bit"
        ),
        model_revision="abc123",
        model_checksum="def456",
    )


def fake_load_into_adapter(
    adapter,
    tokenizer=None,
) -> None:
    adapter._model = object()
    adapter._tokenizer = (
        tokenizer
        or FakeTokenizer()
    )
    adapter._model_config = {}
    adapter._runtime_metadata = (
        adapter._capture_runtime_metadata()
    )


def test_inference_config_accepts_valid_defaults() -> None:
    config = InferenceConfig(
        max_tokens=64,
        seed=12345,
    )

    assert (
        config.decoding_strategy
        == "greedy_argmax"
    )

    assert (
        config.thinking_mode
        is False
    )

    assert (
        config.speculative_decoding
        is False
    )


def test_inference_config_rejects_invalid_max_tokens() -> None:
    with pytest.raises(
        ValueError
    ):
        InferenceConfig(
            max_tokens=0,
            seed=12345,
        )


def test_load_requires_model_path_environment(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "TRAJECTORY_MODEL_PATH",
        raising=False,
    )

    adapter = make_adapter()

    with pytest.raises(
        ModelAdapterError
    ) as exc_info:
        adapter.load()

    assert (
        exc_info.value.code
        == "model_path_not_configured"
    )


def test_load_rejects_missing_directory(
    monkeypatch,
    tmp_path,
) -> None:
    missing = (
        tmp_path
        / "missing"
    )

    monkeypatch.setenv(
        "TRAJECTORY_MODEL_PATH",
        str(
            missing
        ),
    )

    adapter = make_adapter()

    with pytest.raises(
        ModelAdapterError
    ) as exc_info:
        adapter.load()

    assert (
        exc_info.value.code
        == "model_path_not_found"
    )


def test_load_rejects_incomplete_model_directory(
    monkeypatch,
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "model"
    )

    path.mkdir()

    monkeypatch.setenv(
        "TRAJECTORY_MODEL_PATH",
        str(
            path
        ),
    )

    adapter = make_adapter()

    with pytest.raises(
        ModelAdapterError
    ) as exc_info:
        adapter.load()

    assert (
        exc_info.value.code
        == "incomplete_local_model"
    )


def test_load_uses_local_path_and_disables_remote_tokenizer_code(
    monkeypatch,
    tmp_path,
) -> None:
    path = make_model_directory(
        tmp_path
    )

    monkeypatch.setenv(
        "TRAJECTORY_MODEL_PATH",
        str(
            path
        ),
    )

    tokenizer = FakeTokenizer()

    captured = {}

    def fake_mlx_load(
        model_path,
        **kwargs,
    ):
        captured["path"] = (
            model_path
        )

        captured["kwargs"] = (
            kwargs
        )

        return (
            object(),
            tokenizer,
            {
                "model_type": "qwen3_5",
            },
        )

    monkeypatch.setattr(
        "src.model.mlx_qwen.mlx_load",
        fake_mlx_load,
    )

    adapter = make_adapter()
    adapter.load()

    assert adapter.is_loaded is True

    assert (
        captured["path"]
        == str(
            path.resolve()
        )
    )

    assert (
        captured["kwargs"][
            "tokenizer_config"
        ][
            "trust_remote_code"
        ]
        is False
    )

    assert (
        captured["kwargs"][
            "return_config"
        ]
        is True
    )


def test_prepare_input_requires_loaded_model() -> None:
    adapter = make_adapter()

    with pytest.raises(
        ModelAdapterError
    ) as exc_info:
        adapter.prepare_input(
            "test"
        )

    assert (
        exc_info.value.code
        == "model_not_loaded"
    )


def test_prepare_input_disables_thinking_explicitly() -> None:
    tokenizer = FakeTokenizer()

    adapter = make_adapter()

    fake_load_into_adapter(
        adapter,
        tokenizer,
    )

    adapter.prepare_input(
        "TEST"
    )

    _, kwargs = (
        tokenizer.template_calls[0]
    )

    assert (
        kwargs[
            "enable_thinking"
        ]
        is False
    )

    assert (
        kwargs[
            "tokenize"
        ]
        is False
    )

    assert (
        kwargs[
            "add_generation_prompt"
        ]
        is True
    )


def test_prepare_input_records_exact_token_ids() -> None:
    adapter = make_adapter()

    fake_load_into_adapter(
        adapter
    )

    record = adapter.prepare_input(
        "TEST"
    )

    assert (
        record.prompt_token_ids
        == (
            101,
            202,
            303,
        )
    )

    assert (
        record.prompt_token_count
        == 3
    )

    assert len(
        record.rendered_prompt_sha256
    ) == 64

    assert len(
        record.prompt_token_ids_sha256
    ) == 64


def test_prepare_input_does_not_add_system_message() -> None:
    adapter = make_adapter()

    fake_load_into_adapter(
        adapter
    )

    record = adapter.prepare_input(
        "TEST"
    )

    assert len(
        record.messages
    ) == 1

    assert (
        record.messages[0].role
        == "user"
    )

    assert (
        record.messages[0].content
        == "TEST"
    )


def test_prepare_input_rejects_open_thinking_block() -> None:
    adapter = make_adapter()

    fake_load_into_adapter(
        adapter,
        OpenThinkingTokenizer(),
    )

    with pytest.raises(
        ModelAdapterError
    ) as exc_info:
        adapter.prepare_input(
            "TEST"
        )

    assert (
        exc_info.value.code
        == "open_thinking_block"
    )


def test_generate_uses_exact_prepared_token_ids(
    monkeypatch,
) -> None:
    adapter = make_adapter()

    fake_load_into_adapter(
        adapter
    )

    input_record = (
        adapter.prepare_input(
            "TEST"
        )
    )

    captured = {}

    def fake_stream_generate(
        model,
        tokenizer,
        prompt,
        **kwargs,
    ):
        captured["prompt"] = (
            prompt
        )

        captured["kwargs"] = (
            kwargs
        )

        yield SimpleNamespace(
            text="MODEL_",
            token=501,
            prompt_tokens=3,
            generation_tokens=1,
            prompt_tps=10.0,
            generation_tps=20.0,
            peak_memory=1.0,
            finish_reason=None,
        )

        yield SimpleNamespace(
            text="SMOKE_OK",
            token=502,
            prompt_tokens=3,
            generation_tokens=2,
            prompt_tps=10.0,
            generation_tps=20.0,
            peak_memory=1.0,
            finish_reason="stop",
        )

    monkeypatch.setattr(
        "src.model.mlx_qwen.stream_generate",
        fake_stream_generate,
    )

    monkeypatch.setattr(
        "src.model.mlx_qwen.mx.reset_peak_memory",
        lambda: None,
    )

    monkeypatch.setattr(
        "src.model.mlx_qwen.mx.synchronize",
        lambda: None,
    )

    result = adapter.generate(
        input_record=input_record,
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        ),
    )

    assert (
        captured["prompt"]
        == [
            101,
            202,
            303,
        ]
    )

    assert (
        captured["kwargs"][
            "sampler"
        ]
        is None
    )

    assert (
        result.raw_text
        == "MODEL_SMOKE_OK"
    )

    assert (
        result.generated_token_ids
        == (
            501,
            502,
        )
    )

    assert (
        result.finish_reason
        == "stop"
    )


def test_generate_records_generation_metadata(
    monkeypatch,
) -> None:
    adapter = make_adapter()

    fake_load_into_adapter(
        adapter
    )

    input_record = (
        adapter.prepare_input(
            "TEST"
        )
    )

    def fake_stream_generate(
        *args,
        **kwargs,
    ):
        yield SimpleNamespace(
            text="OK",
            token=700,
            prompt_tokens=3,
            generation_tokens=1,
            prompt_tps=11.5,
            generation_tps=22.5,
            peak_memory=33.25,
            finish_reason="stop",
        )

    monkeypatch.setattr(
        "src.model.mlx_qwen.stream_generate",
        fake_stream_generate,
    )

    monkeypatch.setattr(
        "src.model.mlx_qwen.mx.reset_peak_memory",
        lambda: None,
    )

    monkeypatch.setattr(
        "src.model.mlx_qwen.mx.synchronize",
        lambda: None,
    )

    result = adapter.generate(
        input_record=input_record,
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        ),
    )

    assert result.prompt_tokens == 3
    assert result.generation_tokens == 1
    assert result.prompt_tps == 11.5
    assert result.generation_tps == 22.5
    assert result.peak_memory_gb == 33.25


def test_generate_rejects_thinking_mode() -> None:
    adapter = make_adapter()

    fake_load_into_adapter(
        adapter
    )

    input_record = (
        adapter.prepare_input(
            "TEST"
        )
    )

    with pytest.raises(
        ModelAdapterError
    ) as exc_info:
        adapter.generate(
            input_record=input_record,
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
                thinking_mode=True,
            ),
        )

    assert (
        exc_info.value.code
        == "thinking_mode_not_supported"
    )


def test_generate_rejects_non_greedy_strategy() -> None:
    adapter = make_adapter()

    fake_load_into_adapter(
        adapter
    )

    input_record = (
        adapter.prepare_input(
            "TEST"
        )
    )

    with pytest.raises(
        ModelAdapterError
    ) as exc_info:
        adapter.generate(
            input_record=input_record,
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
                decoding_strategy="sampling",
            ),
        )

    assert (
        exc_info.value.code
        == "unsupported_decoding_strategy"
    )


def test_generate_rejects_speculative_decoding() -> None:
    adapter = make_adapter()

    fake_load_into_adapter(
        adapter
    )

    input_record = (
        adapter.prepare_input(
            "TEST"
        )
    )

    with pytest.raises(
        ModelAdapterError
    ) as exc_info:
        adapter.generate(
            input_record=input_record,
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
                speculative_decoding=True,
            ),
        )

    assert (
        exc_info.value.code
        == "speculative_decoding_not_supported"
    )


def test_tampered_prompt_token_hash_is_rejected() -> None:
    adapter = make_adapter()

    fake_load_into_adapter(
        adapter
    )

    input_record = (
        adapter.prepare_input(
            "TEST"
        )
    )

    tampered = replace(
        input_record,
        prompt_token_ids_sha256=(
            "0" * 64
        ),
    )

    with pytest.raises(
        ModelAdapterError
    ) as exc_info:
        adapter.generate(
            input_record=tampered,
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
            ),
        )

    assert (
        exc_info.value.code
        == "prompt_token_checksum_mismatch"
    )


def test_retokenization_mismatch_is_rejected() -> None:
    adapter = make_adapter()

    tokenizer = FakeTokenizer()

    fake_load_into_adapter(
        adapter,
        tokenizer,
    )

    input_record = (
        adapter.prepare_input(
            "TEST"
        )
    )

    tokenizer.encode = lambda *args, **kwargs: [
        999
    ]

    with pytest.raises(
        ModelAdapterError
    ) as exc_info:
        adapter.generate(
            input_record=input_record,
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
            ),
        )

    assert (
        exc_info.value.code
        == "prompt_tokenization_mismatch"
    )


def test_unexpected_thinking_output_is_rejected(
    monkeypatch,
) -> None:
    adapter = make_adapter()

    fake_load_into_adapter(
        adapter
    )

    input_record = (
        adapter.prepare_input(
            "TEST"
        )
    )

    def fake_stream_generate(
        *args,
        **kwargs,
    ):
        yield SimpleNamespace(
            text=(
                "<think>"
                "hidden reasoning"
                "</think>"
                "OK"
            ),
            token=800,
            prompt_tokens=3,
            generation_tokens=1,
            prompt_tps=10.0,
            generation_tps=20.0,
            peak_memory=1.0,
            finish_reason="stop",
        )

    monkeypatch.setattr(
        "src.model.mlx_qwen.stream_generate",
        fake_stream_generate,
    )

    monkeypatch.setattr(
        "src.model.mlx_qwen.mx.reset_peak_memory",
        lambda: None,
    )

    monkeypatch.setattr(
        "src.model.mlx_qwen.mx.synchronize",
        lambda: None,
    )

    with pytest.raises(
        ModelAdapterError
    ) as exc_info:
        adapter.generate(
            input_record=input_record,
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
            ),
        )

    assert (
        exc_info.value.code
        == "unexpected_thinking_output"
    )


def test_runtime_metadata_does_not_store_absolute_model_path(
    monkeypatch,
) -> None:
    adapter = make_adapter()

    monkeypatch.setattr(
        "src.model.mlx_qwen.mx.device_info",
        lambda: {
            "device_name": "Test Device",
        },
    )

    fake_load_into_adapter(
        adapter
    )

    metadata = (
        adapter.runtime_metadata()
        .to_dict()
    )

    assert (
        metadata["model_path_env"]
        == "TRAJECTORY_MODEL_PATH"
    )

    assert "model_path" not in metadata