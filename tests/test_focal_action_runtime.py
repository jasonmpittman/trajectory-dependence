__author__ = "Jason M. Pittman"
__date__ = "August 23, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.1"
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
    EventType,
    RawEventWriter,
    TaskClassification,
)
from src.model import (
    ChatMessage,
    InferenceConfig,
    ModelAdapter,
    ModelGenerationResult,
    ModelInputRecord,
    ModelRuntimeMetadata,
)
from src.normalization import (
    ActionField,
    FocalActionPromptBuilder,
    FocalActionSchema,
)


def sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def sha256_tokens(
    values,
) -> str:
    return sha256_text(
        json.dumps(
            list(values),
            separators=(",", ":"),
        )
    )


def make_schema() -> FocalActionSchema:
    return FocalActionSchema(
        focal_decision_id="final_route",
        fields=(
            ActionField(
                name="action",
                json_type="string",
                allowed_values=(
                    "select_route",
                ),
            ),
            ActionField(
                name="value",
                json_type="string",
                allowed_values=(
                    "A",
                    "B",
                    "C",
                ),
            ),
        ),
    )


class FocalFakeAdapter(
    ModelAdapter
):
    def __init__(
        self,
        output: str,
    ) -> None:
        self.output = output

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def runtime_metadata(
        self,
    ) -> ModelRuntimeMetadata:
        return ModelRuntimeMetadata(
            model_identifier="fake",
            model_revision="fake",
            model_checksum="fake",
            quantization="test",
            backend="fake",
            backend_version="1",
            mlx_version="test",
            python_version="test",
            platform="test",
            model_path_env="TRAJECTORY_MODEL_PATH",
            device_info={},
        )

    def prepare_input(
        self,
        model_visible_text: str,
    ) -> ModelInputRecord:
        rendered = (
            "<user>"
            + model_visible_text
            + "</user>"
        )

        ids = (
            1,
            2,
            3,
        )

        return ModelInputRecord(
            model_visible_text=model_visible_text,
            messages=(
                ChatMessage(
                    role="user",
                    content=model_visible_text,
                ),
            ),
            rendered_prompt=rendered,
            rendered_prompt_sha256=(
                sha256_text(rendered)
            ),
            prompt_token_ids=ids,
            prompt_token_ids_sha256=(
                sha256_tokens(ids)
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
        input_record,
        config,
    ) -> ModelGenerationResult:
        ids = (
            10,
            11,
        )

        return ModelGenerationResult(
            raw_text=self.output,
            raw_text_sha256=(
                sha256_text(
                    self.output
                )
            ),
            generated_token_ids=ids,
            generated_token_ids_sha256=(
                sha256_tokens(ids)
            ),
            finish_reason="stop",
            prompt_tokens=3,
            generation_tokens=2,
            prompt_tps=None,
            generation_tps=None,
            peak_memory_gb=None,
            input_record=input_record,
            inference_config=config,
            runtime_metadata=(
                self.runtime_metadata()
            ),
        )


def make_runtime(
    tmp_path,
    output: str,
) -> AgentRuntime:
    return AgentRuntime(
        identity=RunIdentity(
            experiment_id="exp",
            run_id="run_001",
            task_id="task_001",
            task_classification=(
                TaskClassification.INTEGRATION
            ),
            condition=Condition.C0,
            repetition_id=1,
            seed=12345,
        ),
        model_adapter=(
            FocalFakeAdapter(
                output
            )
        ),
        event_writer=RawEventWriter(
            tmp_path
            / "events.jsonl"
        ),
    )


def test_action_prompt_is_deterministic() -> None:
    builder = (
        FocalActionPromptBuilder()
    )

    first = builder.build(
        task_prompt="Select a route.",
        schema=make_schema(),
    )

    second = builder.build(
        task_prompt="Select a route.",
        schema=make_schema(),
    )

    assert (
        first.rendered_prompt
        == second.rendered_prompt
    )

    assert (
        first.rendered_prompt_sha256
        == second.rendered_prompt_sha256
    )


def test_action_prompt_requires_exact_json() -> None:
    prompt = (
        FocalActionPromptBuilder()
        .build(
            task_prompt="Select B.",
            schema=make_schema(),
        )
    )

    assert (
        "exactly one JSON object"
        in prompt.rendered_prompt
    )

    assert (
        'Return only the fields defined under "properties".'
        in prompt.rendered_prompt
    )

    assert (
        "Do not use Markdown code fences."
        in prompt.rendered_prompt
    )

    assert (
        '"A"'
        in prompt.action_contract_json
    )

    assert (
        '"B"'
        in prompt.action_contract_json
    )

    assert (
        "focal_decision_id"
        not in prompt.action_contract
    )

    assert (
        prompt.focal_decision_id
        == "final_route"
    )

def test_action_contract_contains_only_response_schema() -> None:
    prompt = (
        FocalActionPromptBuilder()
        .build(
            task_prompt="Select B.",
            schema=make_schema(),
        )
    )

    assert set(
        prompt.action_contract.keys()
    ) == {
        "type",
        "properties",
        "required",
        "additionalProperties",
    }

    assert set(
        prompt.action_contract[
            "properties"
        ].keys()
    ) == {
        "action",
        "value",
    }
    
def test_begin_focal_task_logs_rendered_prompt_as_p(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        '{"action":"select_route","value":"B"}',
    )

    runtime.start()

    focal_prompt = (
        runtime.begin_focal_task(
            task_prompt="Select B.",
            action_schema=(
                make_schema()
            ),
        )
    )

    task_event = (
        runtime.events[1]
    )

    assert (
        task_event.event_type
        == EventType.TASK
    )

    assert (
        task_event.normalized_payload
        == focal_prompt.rendered_prompt
    )

    assert (
        task_event.raw_payload[
            "focal_action_schema"
        ][
            "focal_decision_id"
        ]
        == "final_route"
    )


def test_valid_focal_action_creates_action_event(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        '{"action":"select_route","value":"B"}',
    )

    runtime.start()

    runtime.begin_focal_task(
        task_prompt="Select B.",
        action_schema=make_schema(),
    )

    result = (
        runtime.invoke_focal_action(
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
            )
        )
    )

    assert result.valid is True

    assert (
        result.parse_result
        .normalized_action
        .payload
        == {
            "action": "select_route",
            "value": "B",
        }
    )

    assert (
        runtime.events[-1].event_type
        == EventType.ACTION
    )


def test_action_event_follows_generation(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        '{"action":"select_route","value":"B"}',
    )

    runtime.start()

    runtime.begin_focal_task(
        task_prompt="Select B.",
        action_schema=make_schema(),
    )

    result = runtime.invoke_focal_action(
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        )
    )

    generation = (
        runtime.events[-2]
    )

    action = runtime.events[-1]

    assert (
        generation.event_type
        == EventType.GENERATION
    )

    assert (
        action.event_type
        == EventType.ACTION
    )

    assert (
        action.parent_event_id
        == generation.event_id
    )

    assert (
        action.focal_decision_id
        == "final_route"
    )

    assert (
        action.model_invocation_id
        == result
        .model_invocation
        .model_invocation_id
    )


def test_valid_action_event_contains_normalized_action(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        '{"value":"B","action":"select_route"}',
    )

    runtime.start()

    runtime.begin_focal_task(
        task_prompt="Select B.",
        action_schema=make_schema(),
    )

    runtime.invoke_focal_action(
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        )
    )

    action = runtime.events[-1]

    assert (
        action.normalized_payload[
            "payload"
        ]
        == {
            "action": "select_route",
            "value": "B",
        }
    )

    assert (
        action.metadata[
            "parse_valid"
        ]
        is True
    )


def test_invalid_action_is_not_technical_failure(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        "I choose route B.",
    )

    runtime.start()

    runtime.begin_focal_task(
        task_prompt="Select B.",
        action_schema=make_schema(),
    )

    result = runtime.invoke_focal_action(
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        )
    )

    assert result.valid is False

    assert (
        result.parse_error[
            "code"
        ]
        == "invalid_action_json"
    )

    assert (
        runtime.events[-1].event_type
        == EventType.ACTION
    )

    assert (
        runtime.events[-1]
        .normalized_payload
        is None
    )

    assert all(
        event.event_type
        != EventType.TECHNICAL_FAILURE
        for event in runtime.events
    )


def test_markdown_fenced_action_is_preserved_as_invalid(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        (
            "```json\n"
            '{"action":"select_route","value":"B"}'
            "\n```"
        ),
    )

    runtime.start()

    runtime.begin_focal_task(
        task_prompt="Select B.",
        action_schema=make_schema(),
    )

    result = runtime.invoke_focal_action(
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        )
    )

    assert result.valid is False

    assert (
        runtime.events[-1]
        .raw_payload[
            "raw_model_output"
        ]
        .startswith("```json")
    )


def test_focal_action_can_only_be_committed_once(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        '{"action":"select_route","value":"B"}',
    )

    runtime.start()

    runtime.begin_focal_task(
        task_prompt="Select B.",
        action_schema=make_schema(),
    )

    runtime.invoke_focal_action(
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        )
    )

    with pytest.raises(
        AgentRuntimeError
    ) as exc_info:
        runtime.invoke_focal_action(
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
            )
        )

    assert (
        exc_info.value.code
        == "focal_action_already_committed"
    )


def test_generic_task_has_no_focal_schema(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        '{"action":"select_route","value":"B"}',
    )

    runtime.start()

    runtime.begin_task(
        "Generic task."
    )

    with pytest.raises(
        AgentRuntimeError
    ) as exc_info:
        runtime.invoke_focal_action(
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
            )
        )

    assert (
        exc_info.value.code
        == "focal_action_schema_unavailable"
    )


def test_focal_action_event_order(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        '{"action":"select_route","value":"B"}',
    )

    runtime.start()

    runtime.begin_focal_task(
        task_prompt="Select B.",
        action_schema=make_schema(),
    )

    runtime.invoke_focal_action(
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
        EventType.ACTION,
    ]