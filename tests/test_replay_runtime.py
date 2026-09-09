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
    RunIdentity,
)
from src.interventions import (
    InterventionEngine,
    InterventionKind,
    InterventionOperation,
    InterventionPair,
    InterventionPairValidator,
    InterventionSpec,
    MatchedReplayExecutor,
    ReplayExecutionError,
)
from src.logging import (
    Condition,
    ContextElement,
    EventType,
    RawEventWriter,
    SourceClass,
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
from src.snapshots import (
    SnapshotManager,
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


class SequentialReplayAdapter(
    ModelAdapter
):
    """
    Deterministic fake adapter.

    Output order in make_baseline():

    1. baseline focal action;
    2. targeted replay focal action;
    3. sham replay focal action.
    """

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

        token_ids = (
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
                token_ids
            ),
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
        input_record,
        config,
    ) -> ModelGenerationResult:
        if not self.outputs:
            raise RuntimeError(
                "No fake generation remains."
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
            inference_config=config,
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


def make_baseline(
    tmp_path,
):
    adapter = SequentialReplayAdapter(
        [
            (
                '{"action":"select_route",'
                '"value":"B"}'
            ),
            (
                '{"action":"select_route",'
                '"value":"A"}'
            ),
            (
                '{"action":"select_route",'
                '"value":"B"}'
            ),
        ]
    )

    runtime = AgentRuntime(
        identity=RunIdentity(
            experiment_id="exp",
            run_id="run_001",
            task_id="task_001",
            task_classification=(
                TaskClassification.INTEGRATION
            ),
            condition=Condition.C1,
            repetition_id=1,
            seed=12345,
        ),
        model_adapter=adapter,
        event_writer=RawEventWriter(
            tmp_path
            / "baseline.jsonl"
        ),
    )

    runtime.start()

    runtime.begin_focal_task(
        task_prompt=(
            "If threshold is 7, select B. "
            "Otherwise select A. "
            "The label field is irrelevant."
        ),
        action_schema=(
            make_schema()
        ),
    )

    relevant_event = (
        runtime.record_component_event(
            event_type=(
                EventType.TOOL_RETURN
            ),
            source_component=(
                "test_tool"
            ),
            raw_payload={
                "threshold": 7,
            },
            normalized_payload={
                "threshold": 7,
            },
        )
    )

    irrelevant_event = (
        runtime.record_component_event(
            event_type=(
                EventType.TOOL_RETURN
            ),
            source_component=(
                "test_tool"
            ),
            raw_payload={
                "label": "blue",
            },
            normalized_payload={
                "label": "blue",
            },
        )
    )

    relevant = ContextElement(
        source=SourceClass.T,
        source_id=(
            f"tool_return:"
            f"{relevant_event.event_id}"
        ),
        content={
            "threshold": 7,
        },
        origin_event_id=(
            relevant_event.event_id
        ),
        origin_source=SourceClass.T,
        created_sequence=(
            relevant_event.sequence_index
        ),
    )

    irrelevant = ContextElement(
        source=SourceClass.T,
        source_id=(
            f"tool_return:"
            f"{irrelevant_event.event_id}"
        ),
        content={
            "label": "blue",
        },
        origin_event_id=(
            irrelevant_event.event_id
        ),
        origin_source=SourceClass.T,
        created_sequence=(
            irrelevant_event.sequence_index
        ),
    )

    prompt = (
        runtime.prompt_element
    )

    assert prompt is not None

    snapshot = (
        SnapshotManager()
        .capture(
            experiment_id="exp",
            run_id="run_001",
            task_id="task_001",
            condition=Condition.C1,
            repetition_id=1,
            sequence_position=(
                runtime.events[-1]
                .sequence_index
            ),
            prompt_elements=(
                prompt,
            ),
            tool_elements=(
                relevant,
                irrelevant,
            ),
            history_elements=(),
            memory_store=None,
            memory_namespace=None,
            environment=None,
            runtime_configuration={
                "decoding_strategy": (
                    "greedy_argmax"
                ),
            },
            task_control_state={
                "focal_decision_id": (
                    "final_route"
                ),
            },
            seed_metadata={
                "inference_seed": 12345,
            },
        )
    )

    config = InferenceConfig(
        max_tokens=64,
        seed=12345,
    )

    baseline_action = (
        runtime.invoke_focal_action(
            config=config,
            tool_elements=(
                relevant,
                irrelevant,
            ),
        )
    )

    pair = InterventionPair(
        pair_id="pair_001",
        targeted=InterventionSpec(
            intervention_id=(
                "targeted_threshold"
            ),
            pair_id="pair_001",
            kind=(
                InterventionKind.TARGETED
            ),
            source=SourceClass.T,
            operation=(
                InterventionOperation.REPLACE
            ),
            target_id=(
                relevant.source_id
            ),
            replacement_value={
                "threshold": 9,
            },
        ),
        sham=InterventionSpec(
            intervention_id=(
                "sham_label"
            ),
            pair_id="pair_001",
            kind=(
                InterventionKind.SHAM
            ),
            source=SourceClass.T,
            operation=(
                InterventionOperation.REPLACE
            ),
            target_id=(
                irrelevant.source_id
            ),
            replacement_value={
                "label": "cyan",
            },
        ),
    )

    validated = (
        InterventionPairValidator(
            InterventionEngine()
        )
        .validate(
            snapshot=snapshot,
            pair=pair,
        )
    )

    return (
        adapter,
        snapshot,
        baseline_action,
        validated,
        config,
    )


def execute_pair(
    tmp_path,
):
    (
        adapter,
        snapshot,
        baseline_action,
        validated,
        config,
    ) = make_baseline(
        tmp_path
    )

    targeted_path = (
        tmp_path
        / "targeted.jsonl"
    )

    sham_path = (
        tmp_path
        / "sham.jsonl"
    )

    result = (
        MatchedReplayExecutor()
        .execute_pair(
            snapshot=snapshot,
            baseline_action=(
                baseline_action
            ),
            targeted_application=(
                validated
                .targeted_application
            ),
            sham_application=(
                validated
                .sham_application
            ),
            action_schema=(
                make_schema()
            ),
            task_classification=(
                TaskClassification.INTEGRATION
            ),
            model_adapter=adapter,
            config=config,
            targeted_event_writer=(
                RawEventWriter(
                    targeted_path
                )
            ),
            sham_event_writer=(
                RawEventWriter(
                    sham_path
                )
            ),
        )
    )

    return (
        result,
        snapshot,
        targeted_path,
        sham_path,
    )


def read_event_types(
    path,
) -> list[str]:
    values: list[str] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if not line.strip():
                continue

            event = json.loads(
                line
            )

            values.append(
                event[
                    "event_type"
                ]
            )

    return values


def test_replay_pair_executes_from_same_snapshot(
    tmp_path,
) -> None:
    (
        result,
        snapshot,
        _,
        _,
    ) = execute_pair(
        tmp_path
    )

    assert (
        result.baseline_snapshot_id
        == snapshot.snapshot_id
    )

    assert (
        result.targeted
        .application
        .baseline_snapshot_id
        == snapshot.snapshot_id
    )

    assert (
        result.sham
        .application
        .baseline_snapshot_id
        == snapshot.snapshot_id
    )

    assert (
        result.targeted
        .application
        .baseline_state_checksum
        == result.sham
        .application
        .baseline_state_checksum
    )


def test_targeted_replay_uses_counterfactual_t(
    tmp_path,
) -> None:
    (
        result,
        _,
        _,
        _,
    ) = execute_pair(
        tmp_path
    )

    contents = [
        element.content
        for element
        in (
            result
            .targeted
            .projected_context
            .tool_elements
        )
    ]

    assert {
        "threshold": 9,
    } in contents

    assert {
        "label": "blue",
    } in contents


def test_sham_replay_preserves_relevant_t(
    tmp_path,
) -> None:
    (
        result,
        _,
        _,
        _,
    ) = execute_pair(
        tmp_path
    )

    contents = [
        element.content
        for element
        in (
            result
            .sham
            .projected_context
            .tool_elements
        )
    ]

    assert {
        "threshold": 7,
    } in contents

    assert {
        "label": "cyan",
    } in contents


def test_replay_preserves_event_backed_provenance(
    tmp_path,
) -> None:
    (
        result,
        _,
        _,
        _,
    ) = execute_pair(
        tmp_path
    )

    targeted_audit = (
        result
        .targeted
        .focal_action
        .model_invocation
        .provenance_audit
    )

    sham_audit = (
        result
        .sham
        .focal_action
        .model_invocation
        .provenance_audit
    )

    assert [
        record.source
        for record
        in targeted_audit.records
    ] == [
        SourceClass.P,
        SourceClass.T,
        SourceClass.T,
    ]

    assert [
        record.source
        for record
        in sham_audit.records
    ] == [
        SourceClass.P,
        SourceClass.T,
        SourceClass.T,
    ]


def test_replay_projection_uses_replay_local_event_ids(
    tmp_path,
) -> None:
    (
        result,
        _,
        _,
        _,
    ) = execute_pair(
        tmp_path
    )

    targeted_elements = (
        result
        .targeted
        .projected_context
        .tool_elements
    )

    for element in targeted_elements:
        assert (
            element.origin_event_id
            is not None
        )

        assert (
            element.source_id
            == (
                f"tool_return:"
                f"{element.origin_event_id}"
            )
        )

        assert (
            element.metadata[
                "replay_projection"
            ]
            is True
        )

        assert (
            element.metadata[
                "baseline_source_id"
            ]
            is not None
        )


def test_targeted_and_sham_event_types_are_logged(
    tmp_path,
) -> None:
    (
        _,
        _,
        targeted_path,
        sham_path,
    ) = execute_pair(
        tmp_path
    )

    targeted_types = (
        read_event_types(
            targeted_path
        )
    )

    sham_types = (
        read_event_types(
            sham_path
        )
    )

    assert (
        EventType.INTERVENTION.value
        in targeted_types
    )

    assert (
        EventType.SHAM_INTERVENTION.value
        in sham_types
    )

    assert (
        EventType.SHAM_INTERVENTION.value
        not in targeted_types
    )

    assert (
        EventType.INTERVENTION.value
        not in sham_types
    )


def test_replay_uses_exact_baseline_inference_config(
    tmp_path,
) -> None:
    (
        adapter,
        snapshot,
        baseline_action,
        validated,
        _,
    ) = make_baseline(
        tmp_path
    )

    changed_config = (
        InferenceConfig(
            max_tokens=32,
            seed=12345,
        )
    )

    with pytest.raises(
        ReplayExecutionError
    ) as exc_info:
        (
            MatchedReplayExecutor()
            .execute_pair(
                snapshot=snapshot,
                baseline_action=(
                    baseline_action
                ),
                targeted_application=(
                    validated
                    .targeted_application
                ),
                sham_application=(
                    validated
                    .sham_application
                ),
                action_schema=(
                    make_schema()
                ),
                task_classification=(
                    TaskClassification.INTEGRATION
                ),
                model_adapter=adapter,
                config=(
                    changed_config
                ),
                targeted_event_writer=(
                    RawEventWriter(
                        tmp_path
                        / "targeted.jsonl"
                    )
                ),
                sham_event_writer=(
                    RawEventWriter(
                        tmp_path
                        / "sham.jsonl"
                    )
                ),
            )
        )

    assert (
        exc_info.value.code
        == "inference_config_mismatch"
    )


def test_replay_actions_are_strictly_normalized(
    tmp_path,
) -> None:
    (
        result,
        _,
        _,
        _,
    ) = execute_pair(
        tmp_path
    )

    assert (
        result.targeted
        .focal_action
        .valid
        is True
    )

    assert (
        result.sham
        .focal_action
        .valid
        is True
    )

    assert (
        result.targeted
        .focal_action
        .parse_result
        .normalized_action
        .payload
        == {
            "action": (
                "select_route"
            ),
            "value": "A",
        }
    )

    assert (
        result.sham
        .focal_action
        .parse_result
        .normalized_action
        .payload
        == {
            "action": (
                "select_route"
            ),
            "value": "B",
        }
    )