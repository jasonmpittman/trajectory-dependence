__author__ = "Jason M. Pittman"
__date__ = "August 23, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import hashlib
import json

import pytest

from src.agent import (
    AgentRuntime,
    AgentRuntimeError,
    RunIdentity,
)
from src.logging import (
    Condition,
    ContextElement,
    EventType,
    RawEventWriter,
    SourceClass,
    TaskClassification,
    read_events,
)
from src.model import (
    ChatMessage,
    InferenceConfig,
    ModelAdapter,
    ModelAdapterError,
    ModelGenerationResult,
    ModelInputRecord,
    ModelRuntimeMetadata,
)


def sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def sha256_tokens(
    values,
) -> str:
    serialized = json.dumps(
        list(
            values
        ),
        separators=(",", ":"),
    )

    return sha256_text(
        serialized
    )


class FakeModelAdapter(
    ModelAdapter
):
    def __init__(
        self,
        *,
        loaded=True,
        fail_prepare=False,
        fail_generate=False,
    ) -> None:
        self._loaded = loaded
        self.fail_prepare = (
            fail_prepare
        )
        self.fail_generate = (
            fail_generate
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def runtime_metadata(
        self,
    ) -> ModelRuntimeMetadata:
        if not self._loaded:
            raise ModelAdapterError(
                code="model_not_loaded",
                message="not loaded",
            )

        return ModelRuntimeMetadata(
            model_identifier="fake-model",
            model_revision="fake-revision",
            model_checksum="fake-checksum",
            quantization="test",
            backend="fake",
            backend_version="1.0",
            mlx_version="test",
            python_version="test",
            platform="test",
            model_path_env="TRAJECTORY_MODEL_PATH",
            device_info={
                "device": "fake",
            },
        )

    def prepare_input(
        self,
        model_visible_text: str,
    ) -> ModelInputRecord:
        if self.fail_prepare:
            raise ModelAdapterError(
                code="fake_prepare_failure",
                message="prepare failed",
            )

        rendered = (
            "<user>"
            + model_visible_text
            + "</user>"
        )

        token_ids = (
            11,
            22,
            33,
        )

        return ModelInputRecord(
            model_visible_text=(
                model_visible_text
            ),
            messages=(
                ChatMessage(
                    role="user",
                    content=(
                        model_visible_text
                    ),
                ),
            ),
            rendered_prompt=rendered,
            rendered_prompt_sha256=(
                sha256_text(
                    rendered
                )
            ),
            prompt_token_ids=token_ids,
            prompt_token_ids_sha256=(
                sha256_tokens(
                    token_ids
                )
            ),
            prompt_token_count=3,
            thinking_mode=False,
            chat_template_arguments={
                "enable_thinking": False,
            },
        )

    def generate(
        self,
        *,
        input_record: ModelInputRecord,
        config: InferenceConfig,
    ) -> ModelGenerationResult:
        if self.fail_generate:
            raise ModelAdapterError(
                code="fake_generation_failure",
                message="generation failed",
            )

        text = "MODEL_OUTPUT"

        generated_ids = (
            44,
            55,
        )

        return ModelGenerationResult(
            raw_text=text,
            raw_text_sha256=(
                sha256_text(
                    text
                )
            ),
            generated_token_ids=(
                generated_ids
            ),
            generated_token_ids_sha256=(
                sha256_tokens(
                    generated_ids
                )
            ),
            finish_reason="stop",
            prompt_tokens=3,
            generation_tokens=2,
            prompt_tps=10.0,
            generation_tps=20.0,
            peak_memory_gb=1.0,
            input_record=input_record,
            inference_config=config,
            runtime_metadata=(
                self.runtime_metadata()
            ),
        )


def make_identity(
    condition=Condition.C0,
) -> RunIdentity:
    return RunIdentity(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        task_classification=(
            TaskClassification.INTEGRATION
        ),
        condition=condition,
        repetition_id=1,
        seed=12345,
    )


def make_runtime(
    tmp_path,
    *,
    condition=Condition.C0,
    adapter=None,
):
    writer = RawEventWriter(
        tmp_path
        / "events.jsonl"
    )

    runtime = AgentRuntime(
        identity=make_identity(
            condition
        ),
        model_adapter=(
            adapter
            or FakeModelAdapter()
        ),
        event_writer=writer,
    )

    return runtime, writer


def test_start_records_system_event(
    tmp_path,
) -> None:
    runtime, _ = make_runtime(
        tmp_path
    )

    event = runtime.start()

    assert (
        event.event_type
        == EventType.SYSTEM
    )

    assert (
        event.sequence_index
        == 0
    )

    assert (
        len(
            runtime.events
        )
        == 1
    )


def test_runtime_requires_loaded_model(
    tmp_path,
) -> None:
    runtime, _ = make_runtime(
        tmp_path,
        adapter=FakeModelAdapter(
            loaded=False
        ),
    )

    with pytest.raises(
        AgentRuntimeError
    ) as exc_info:
        runtime.start()

    assert (
        exc_info.value.code
        == "model_not_loaded"
    )


def test_begin_task_creates_event_backed_prompt(
    tmp_path,
) -> None:
    runtime, _ = make_runtime(
        tmp_path
    )

    runtime.start()

    prompt = runtime.begin_task(
        "Choose A."
    )

    assert (
        prompt.source
        == SourceClass.P
    )

    assert (
        prompt.source_id
        == "prompt:task_001"
    )

    assert (
        prompt.origin_event_id
        == runtime.events[1].event_id
    )

    assert (
        runtime.events[1].event_type
        == EventType.TASK
    )


def test_component_event_is_appended(
    tmp_path,
) -> None:
    runtime, _ = make_runtime(
        tmp_path,
        condition=Condition.C1,
    )

    runtime.start()
    runtime.begin_task(
        "Choose A."
    )

    event = (
        runtime.record_component_event(
            event_type=(
                EventType.TOOL_RETURN
            ),
            source_component="test_tool",
            raw_payload={
                "threshold": 7,
            },
            normalized_payload={
                "threshold": 7,
            },
            tool_call_id="call_001",
        )
    )

    assert (
        event.sequence_index
        == 2
    )

    assert (
        runtime.events[-1]
        == event
    )


def test_c0_invocation_logs_expected_event_order(
    tmp_path,
) -> None:
    runtime, _ = make_runtime(
        tmp_path
    )

    runtime.start()

    runtime.begin_task(
        "Choose A."
    )

    runtime.invoke_model(
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        )
    )

    assert [
        event.event_type
        for event in runtime.events
    ] == [
        EventType.SYSTEM,
        EventType.TASK,
        EventType.CONTEXT_BUILD,
        EventType.PROVENANCE_AUDIT,
        EventType.GENERATION,
    ]


def test_context_build_preserves_all_input_layers(
    tmp_path,
) -> None:
    runtime, _ = make_runtime(
        tmp_path
    )

    runtime.start()

    runtime.begin_task(
        "Choose A."
    )

    invocation = runtime.invoke_model(
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        )
    )

    context_event = next(
        event
        for event in runtime.events
        if (
            event.event_type
            == EventType.CONTEXT_BUILD
        )
    )

    raw = (
        context_event.raw_payload
    )

    assert (
        raw["structured_context"]
        == invocation
        .structured_context
        .to_dict()
    )

    assert (
        raw["serialized_context"][
            "model_text"
        ]
        == invocation
        .serialized_context
        .model_text
    )

    assert (
        raw["model_input"][
            "rendered_prompt"
        ]
        == invocation
        .model_input
        .rendered_prompt
    )

    assert (
        raw["model_input"][
            "prompt_token_ids"
        ]
        == list(
            invocation
            .model_input
            .prompt_token_ids
        )
    )


def test_provenance_audit_is_logged_before_generation(
    tmp_path,
) -> None:
    runtime, _ = make_runtime(
        tmp_path
    )

    runtime.start()
    runtime.begin_task(
        "Choose A."
    )

    invocation = runtime.invoke_model(
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        )
    )

    audit_event = next(
        event
        for event in runtime.events
        if (
            event.event_type
            == EventType.PROVENANCE_AUDIT
        )
    )

    generation_event = next(
        event
        for event in runtime.events
        if (
            event.event_type
            == EventType.GENERATION
        )
    )

    assert (
        audit_event.sequence_index
        < generation_event.sequence_index
    )

    assert (
        audit_event.raw_payload
        == invocation
        .provenance_audit
        .to_dict()
    )


def test_generation_event_uses_raw_text_as_normalized_payload(
    tmp_path,
) -> None:
    runtime, _ = make_runtime(
        tmp_path
    )

    runtime.start()
    runtime.begin_task(
        "Choose A."
    )

    invocation = runtime.invoke_model(
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        )
    )

    generation_event = (
        runtime.events[-1]
    )

    assert (
        generation_event.event_type
        == EventType.GENERATION
    )

    assert (
        generation_event.normalized_payload
        == invocation.generation.raw_text
    )

    assert (
        generation_event.raw_payload
        == invocation.generation.to_dict()
    )


def test_tool_backed_c1_context_audits_successfully(
    tmp_path,
) -> None:
    runtime, _ = make_runtime(
        tmp_path,
        condition=Condition.C1,
    )

    runtime.start()

    runtime.begin_task(
        "Use the tool result."
    )

    tool_event = (
        runtime.record_component_event(
            event_type=(
                EventType.TOOL_RETURN
            ),
            source_component="test_tool",
            raw_payload={
                "threshold": 7,
            },
            normalized_payload={
                "threshold": 7,
            },
        )
    )

    tool_element = ContextElement(
        source=SourceClass.T,
        source_id=(
            f"tool_return:"
            f"{tool_event.event_id}"
        ),
        content={
            "threshold": 7,
        },
        origin_event_id=(
            tool_event.event_id
        ),
        origin_source=SourceClass.T,
        created_sequence=(
            tool_event.sequence_index
        ),
    )

    invocation = runtime.invoke_model(
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        ),
        tool_elements=(
            tool_element,
        ),
    )

    assert [
        record.source
        for record in (
            invocation
            .provenance_audit
            .records
        )
    ] == [
        SourceClass.P,
        SourceClass.T,
    ]


def test_invalid_provenance_is_preserved_as_failure(
    tmp_path,
) -> None:
    runtime, _ = make_runtime(
        tmp_path,
        condition=Condition.C1,
    )

    runtime.start()

    runtime.begin_task(
        "Use the tool result."
    )

    tool_event = (
        runtime.record_component_event(
            event_type=(
                EventType.TOOL_RETURN
            ),
            source_component="test_tool",
            raw_payload={
                "threshold": 7,
            },
            normalized_payload={
                "threshold": 7,
            },
        )
    )

    bad_element = ContextElement(
        source=SourceClass.T,
        source_id=(
            f"tool_return:"
            f"{tool_event.event_id}"
        ),

        # Deliberately does not match event payload.
        content={
            "threshold": 999,
        },
        origin_event_id=(
            tool_event.event_id
        ),
        origin_source=SourceClass.T,
    )

    with pytest.raises(
        AgentRuntimeError
    ) as exc_info:
        runtime.invoke_model(
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
            ),
            tool_elements=(
                bad_element,
            ),
        )

    assert (
        exc_info.value.code
        == "provenance_audit_failure"
    )

    assert (
        runtime.events[-2].event_type
        == EventType.CONTEXT_BUILD
    )

    assert (
        runtime.events[-1].event_type
        == EventType.TECHNICAL_FAILURE
    )

    assert all(
        event.event_type
        != EventType.GENERATION
        for event in runtime.events
    )


def test_prepare_failure_is_logged(
    tmp_path,
) -> None:
    runtime, _ = make_runtime(
        tmp_path,
        adapter=FakeModelAdapter(
            fail_prepare=True
        ),
    )

    runtime.start()
    runtime.begin_task(
        "Choose A."
    )

    with pytest.raises(
        AgentRuntimeError
    ) as exc_info:
        runtime.invoke_model(
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
            )
        )

    assert (
        exc_info.value.code
        == "model_input_preparation_failure"
    )

    assert (
        runtime.events[-1].event_type
        == EventType.TECHNICAL_FAILURE
    )


def test_generation_failure_is_logged_after_audit(
    tmp_path,
) -> None:
    runtime, _ = make_runtime(
        tmp_path,
        adapter=FakeModelAdapter(
            fail_generate=True
        ),
    )

    runtime.start()
    runtime.begin_task(
        "Choose A."
    )

    with pytest.raises(
        AgentRuntimeError
    ) as exc_info:
        runtime.invoke_model(
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
            )
        )

    assert (
        exc_info.value.code
        == "model_generation_failure"
    )

    assert [
        event.event_type
        for event in runtime.events[-3:]
    ] == [
        EventType.CONTEXT_BUILD,
        EventType.PROVENANCE_AUDIT,
        EventType.TECHNICAL_FAILURE,
    ]


def test_jsonl_reconstructs_runtime_event_stream(
    tmp_path,
) -> None:
    runtime, writer = make_runtime(
        tmp_path
    )

    runtime.start()
    runtime.begin_task(
        "Choose A."
    )

    runtime.invoke_model(
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        )
    )

    loaded = read_events(
        writer.path
    )

    assert [
        event["sequence_index"]
        for event in loaded
    ] == [
        0,
        1,
        2,
        3,
        4,
    ]

    assert [
        event["event_type"]
        for event in loaded
    ] == [
        "system",
        "task",
        "context_build",
        "provenance_audit",
        "generation",
    ]