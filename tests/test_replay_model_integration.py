__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import os

import pytest

from src.agent import (
    AgentRuntime,
    RunIdentity,
)
from src.interventions import (
    InterventionEngine,
    InterventionKind,
    InterventionOperation,
    InterventionOutcome,
    InterventionPair,
    InterventionPairValidator,
    InterventionSpec,
    MatchedReplayClassifier,
    MatchedReplayExecutor,
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
    InferenceConfig,
    MLXQwenAdapter,
)
from src.normalization import (
    ActionField,
    FocalActionSchema,
)
from src.snapshots import (
    SnapshotManager,
)


pytestmark = pytest.mark.skipif(
    os.environ.get(
        "RUN_MODEL_INTEGRATION"
    )
    != "1",
    reason=(
        "Set RUN_MODEL_INTEGRATION=1 "
        "to run local 27B model tests."
    ),
)


def test_real_qwen_targeted_tool_replay(
    tmp_path,
) -> None:
    """
    End-to-end diagnostic validation of matched causal replay.

    Expected normalized behavior:

        baseline:
            threshold = 7
            -> B

        targeted T intervention:
            threshold = 9
            -> A

        source-matched T sham:
            threshold = 7
            label = blue -> cyan
            -> B

    Therefore:

        a_target != a_base
        a_sham == a_base

    and the matched replay classification must be
    intervention_positive.
    """

    adapter = MLXQwenAdapter(
        model_identifier=(
            "mlx-community/"
            "Qwen3.5-27B-8bit"
        ),
        model_revision=None,
        model_checksum=None,
        quantization="8bit",
    )

    adapter.load()

    schema = FocalActionSchema(
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

    config = InferenceConfig(
        max_tokens=64,
        seed=12345,
    )

    runtime = AgentRuntime(
        identity=RunIdentity(
            experiment_id=(
                "matched_replay_integration"
            ),
            run_id="run_001",
            task_id=(
                "matched_replay_smoke"
            ),
            task_classification=(
                TaskClassification.DIAGNOSTIC
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
            "Use the tool information below to choose a route. "
            "The decision rule is exact: if threshold is exactly 7, "
            "select route B. For every other threshold value, "
            "select route A. The label field is irrelevant and must "
            "not affect the route decision."
        ),
        action_schema=schema,
    )

    threshold_event = (
        runtime.record_component_event(
            event_type=(
                EventType.TOOL_RETURN
            ),
            source_component=(
                "integration_fixture"
            ),
            raw_payload={
                "threshold": 7,
            },
            normalized_payload={
                "threshold": 7,
            },
        )
    )

    label_event = (
        runtime.record_component_event(
            event_type=(
                EventType.TOOL_RETURN
            ),
            source_component=(
                "integration_fixture"
            ),
            raw_payload={
                "label": "blue",
            },
            normalized_payload={
                "label": "blue",
            },
        )
    )

    threshold_element = (
        ContextElement(
            source=SourceClass.T,
            source_id=(
                f"tool_return:"
                f"{threshold_event.event_id}"
            ),
            content={
                "threshold": 7,
            },
            origin_event_id=(
                threshold_event.event_id
            ),
            origin_source=(
                SourceClass.T
            ),
            created_sequence=(
                threshold_event.sequence_index
            ),
        )
    )

    label_element = (
        ContextElement(
            source=SourceClass.T,
            source_id=(
                f"tool_return:"
                f"{label_event.event_id}"
            ),
            content={
                "label": "blue",
            },
            origin_event_id=(
                label_event.event_id
            ),
            origin_source=(
                SourceClass.T
            ),
            created_sequence=(
                label_event.sequence_index
            ),
        )
    )

    prompt_element = (
        runtime.prompt_element
    )

    assert (
        prompt_element is not None
    )

    snapshot = (
        SnapshotManager()
        .capture(
            experiment_id=(
                "matched_replay_integration"
            ),
            run_id="run_001",
            task_id=(
                "matched_replay_smoke"
            ),
            condition=Condition.C1,
            repetition_id=1,
            sequence_position=(
                runtime.events[-1]
                .sequence_index
            ),
            prompt_elements=(
                prompt_element,
            ),
            tool_elements=(
                threshold_element,
                label_element,
            ),
            history_elements=(),
            memory_store=None,
            memory_namespace=None,
            environment=None,
            runtime_configuration={
                "inference": (
                    config.to_dict()
                ),
            },
            task_control_state={
                "focal_decision_id": (
                    "final_route"
                ),
            },
            seed_metadata={
                "inference_seed": (
                    12345
                ),
            },
        )
    )

    baseline_action = (
        runtime.invoke_focal_action(
            config=config,
            tool_elements=(
                threshold_element,
                label_element,
            ),
        )
    )

    assert (
        baseline_action.valid
        is True
    )

    assert (
        baseline_action.parse_result
        is not None
    )

    assert (
        baseline_action
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

    pair = InterventionPair(
        pair_id="tool_pair_001",
        targeted=InterventionSpec(
            intervention_id=(
                "target_threshold"
            ),
            pair_id="tool_pair_001",
            kind=(
                InterventionKind.TARGETED
            ),
            source=SourceClass.T,
            operation=(
                InterventionOperation.REPLACE
            ),
            target_id=(
                threshold_element.source_id
            ),

            # Same serialized size as the baseline value.
            replacement_value={
                "threshold": 9,
            },
        ),
        sham=InterventionSpec(
            intervention_id=(
                "sham_label"
            ),
            pair_id="tool_pair_001",
            kind=(
                InterventionKind.SHAM
            ),
            source=SourceClass.T,
            operation=(
                InterventionOperation.REPLACE
            ),
            target_id=(
                label_element.source_id
            ),

            # "blue" and "cyan" are the same character length.
            replacement_value={
                "label": "cyan",
            },
        ),
    )

    validated_pair = (
        InterventionPairValidator(
            InterventionEngine()
        )
        .validate(
            snapshot=snapshot,
            pair=pair,
        )
    )

    assert (
        validated_pair
        .targeted_application
        .baseline_snapshot_id
        == snapshot.snapshot_id
    )

    assert (
        validated_pair
        .sham_application
        .baseline_snapshot_id
        == snapshot.snapshot_id
    )

    assert (
        validated_pair
        .targeted_application
        .baseline_state_checksum
        == validated_pair
        .sham_application
        .baseline_state_checksum
    )

    execution = (
        MatchedReplayExecutor()
        .execute_pair(
            snapshot=snapshot,
            baseline_action=(
                baseline_action
            ),
            targeted_application=(
                validated_pair
                .targeted_application
            ),
            sham_application=(
                validated_pair
                .sham_application
            ),
            action_schema=schema,
            task_classification=(
                TaskClassification.DIAGNOSTIC
            ),
            model_adapter=adapter,
            config=config,
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
        execution.targeted
        .focal_action
        .valid
        is True
    )

    assert (
        execution.sham
        .focal_action
        .valid
        is True
    )

    assert (
        execution.targeted
        .focal_action
        .parse_result
        is not None
    )

    assert (
        execution.sham
        .focal_action
        .parse_result
        is not None
    )

    assert (
        execution.targeted
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
        execution.sham
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

    classification = (
        MatchedReplayClassifier()
        .classify(
            execution
        )
    )

    assert (
        classification.outcome
        == InterventionOutcome.POSITIVE
    )

    assert (
        classification.targeted_changed
        is True
    )

    assert (
        classification.sham_changed
        is False
    )

    assert (
        classification.baseline_action
        == {
            "action": (
                "select_route"
            ),
            "value": "B",
        }
    )

    assert (
        classification.targeted_action
        == {
            "action": (
                "select_route"
            ),
            "value": "A",
        }
    )

    assert (
        classification.sham_action
        == {
            "action": (
                "select_route"
            ),
            "value": "B",
        }
    )