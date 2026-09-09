__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
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
    ExplanationSource,
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
    FocalActionSchema,
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
    return sha256_text(
        json.dumps(
            list(
                values
            ),
            separators=(",", ":"),
        )
    )


VALID_ACTION = (
    '{"action":"select_route","value":"B"}'
)

VALID_EXPLANATION = """
{
  "causal_sources": ["P"],
  "justification": {
    "P": "The task instructed me to select B.",
    "T": null,
    "H": null,
    "M": null,
    "E": null,
    "Other": null
  }
}
"""


class SequentialFakeAdapter(
    ModelAdapter
):
    def __init__(
        self,
        outputs: list[str],
    ) -> None:
        self.outputs = list(
            outputs
        )

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
            model_path_env=(
                "TRAJECTORY_MODEL_PATH"
            ),
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
            rendered_prompt=(
                rendered
            ),
            rendered_prompt_sha256=(
                sha256_text(
                    rendered
                )
            ),
            prompt_token_ids=(
                ids
            ),
            prompt_token_ids_sha256=(
                sha256_tokens(
                    ids
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
        input_record,
        config,
    ) -> ModelGenerationResult:
        if not self.outputs:
            raise RuntimeError(
                "No fake output remains."
            )

        text = self.outputs.pop(
            0
        )

        generated_ids = (
            10,
            11,
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
            prompt_tps=None,
            generation_tps=None,
            peak_memory_gb=None,
            input_record=(
                input_record
            ),
            inference_config=(
                config
            ),
            runtime_metadata=(
                self.runtime_metadata()
            ),
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
                ),
            ),
        ),
    )


def make_runtime(
    tmp_path,
    outputs: list[str],
) -> AgentRuntime:
    runtime = AgentRuntime(
        identity=RunIdentity(
            experiment_id="exp",
            run_id="run_001",
            task_id="task_001",
            task_classification=(
                TaskClassification.DIAGNOSTIC
            ),
            condition=Condition.C0,
            repetition_id=1,
            seed=12345,
        ),
        model_adapter=(
            SequentialFakeAdapter(
                outputs
            )
        ),
        event_writer=RawEventWriter(
            tmp_path
            / "events.jsonl"
        ),
    )

    runtime.start()

    runtime.begin_focal_task(
        task_prompt=(
            "Select route B."
        ),
        action_schema=(
            make_schema()
        ),
    )

    return runtime


def commit_action(
    runtime: AgentRuntime,
):
    return runtime.invoke_focal_action(
        config=InferenceConfig(
            max_tokens=64,
            seed=12345,
        )
    )


def test_explanation_requires_committed_action(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        [
            VALID_ACTION,
            VALID_EXPLANATION,
        ],
    )

    with pytest.raises(
        AgentRuntimeError
    ) as exc_info:
        runtime.elicit_self_explanation(
            focal_action=object(),
            config=InferenceConfig(
                max_tokens=128,
                seed=12345,
            ),
        )

    assert (
        exc_info.value.code
        == "focal_action_not_committed"
    )

def test_valid_self_explanation_is_committed(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        [
            VALID_ACTION,
            VALID_EXPLANATION,
        ],
    )

    action = commit_action(
        runtime
    )

    explanation = (
        runtime.elicit_self_explanation(
            focal_action=action,
            config=InferenceConfig(
                max_tokens=128,
                seed=12345,
            ),
        )
    )

    assert explanation.valid is True

    assert (
        explanation.parse_result
        is not None
    )

    assert (
        explanation.parse_result
        .reported_sources
        == (
            ExplanationSource.P,
        )
    )

    assert (
        runtime.events[-1].event_type
        == EventType.EXPLANATION
    )


def test_explanation_occurs_after_action(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        [
            VALID_ACTION,
            VALID_EXPLANATION,
        ],
    )

    action = commit_action(
        runtime
    )

    runtime.elicit_self_explanation(
        focal_action=action,
        config=InferenceConfig(
            max_tokens=128,
            seed=12345,
        ),
    )

    action_index = next(
        event.sequence_index
        for event in runtime.events
        if (
            event.event_type
            == EventType.ACTION
        )
    )

    explanation_index = next(
        event.sequence_index
        for event in runtime.events
        if (
            event.event_type
            == EventType.EXPLANATION
        )
    )

    assert (
        action_index
        < explanation_index
    )


def test_explanation_reuses_exact_decision_context(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        [
            VALID_ACTION,
            VALID_EXPLANATION,
        ],
    )

    action = commit_action(
        runtime
    )

    explanation = (
        runtime.elicit_self_explanation(
            focal_action=action,
            config=InferenceConfig(
                max_tokens=128,
                seed=12345,
            ),
        )
    )

    assert (
        explanation.prompt
        .decision_context_text
        == action
        .model_invocation
        .serialized_context
        .model_text
    )

    assert (
        explanation.prompt
        .decision_context_sha256
        == action
        .model_invocation
        .serialized_context
        .model_text_sha256
    )


def test_explanation_uses_exact_focal_action_output(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        [
            VALID_ACTION,
            VALID_EXPLANATION,
        ],
    )

    action = commit_action(
        runtime
    )

    explanation = (
        runtime.elicit_self_explanation(
            focal_action=action,
            config=InferenceConfig(
                max_tokens=128,
                seed=12345,
            ),
        )
    )

    assert (
        explanation.prompt
        .focal_action_text
        == VALID_ACTION
    )


def test_explanation_generation_links_to_action(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        [
            VALID_ACTION,
            VALID_EXPLANATION,
        ],
    )

    action = commit_action(
        runtime
    )

    explanation = (
        runtime.elicit_self_explanation(
            focal_action=action,
            config=InferenceConfig(
                max_tokens=128,
                seed=12345,
            ),
        )
    )

    generation_event = next(
        event
        for event in runtime.events
        if (
            event.event_id
            == explanation
            .generation_event_id
        )
    )

    assert (
        generation_event.event_type
        == EventType.GENERATION
    )

    assert (
        generation_event.parent_event_id
        == action.action_event_id
    )

    assert (
        generation_event.metadata[
            "generation_role"
        ]
        == "self_explanation"
    )


def test_explanation_event_links_to_generation(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        [
            VALID_ACTION,
            VALID_EXPLANATION,
        ],
    )

    action = commit_action(
        runtime
    )

    explanation = (
        runtime.elicit_self_explanation(
            focal_action=action,
            config=InferenceConfig(
                max_tokens=128,
                seed=12345,
            ),
        )
    )

    explanation_event = (
        runtime.events[-1]
    )

    assert (
        explanation_event.event_type
        == EventType.EXPLANATION
    )

    assert (
        explanation_event.parent_event_id
        == explanation
        .generation_event_id
    )


def test_invalid_explanation_is_not_technical_failure(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        [
            VALID_ACTION,
            "I used the prompt.",
        ],
    )

    action = commit_action(
        runtime
    )

    explanation = (
        runtime.elicit_self_explanation(
            focal_action=action,
            config=InferenceConfig(
                max_tokens=128,
                seed=12345,
            ),
        )
    )

    assert explanation.valid is False

    assert (
        explanation.parse_error[
            "code"
        ]
        == "invalid_explanation_json"
    )

    assert (
        runtime.events[-1].event_type
        == EventType.EXPLANATION
    )

    assert (
        runtime.events[-1]
        .normalized_payload
        is None
    )

    assert all(
        not (
            event.event_type
            == EventType.TECHNICAL_FAILURE
            and event.raw_payload.get(
                "stage"
            )
            == "self_explanation_parsing"
        )
        for event in runtime.events
    )


def test_only_one_self_explanation_is_allowed(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        [
            VALID_ACTION,
            VALID_EXPLANATION,
            VALID_EXPLANATION,
        ],
    )

    action = commit_action(
        runtime
    )

    runtime.elicit_self_explanation(
        focal_action=action,
        config=InferenceConfig(
            max_tokens=128,
            seed=12345,
        ),
    )

    with pytest.raises(
        AgentRuntimeError
    ) as exc_info:
        runtime.elicit_self_explanation(
            focal_action=action,
            config=InferenceConfig(
                max_tokens=128,
                seed=12345,
            ),
        )

    assert (
        exc_info.value.code
        == "self_explanation_already_committed"
    )


def test_expected_event_order(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        [
            VALID_ACTION,
            VALID_EXPLANATION,
        ],
    )

    action = commit_action(
        runtime
    )

    runtime.elicit_self_explanation(
        focal_action=action,
        config=InferenceConfig(
            max_tokens=128,
            seed=12345,
        ),
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
        EventType.GENERATION,
        EventType.EXPLANATION,
    ]