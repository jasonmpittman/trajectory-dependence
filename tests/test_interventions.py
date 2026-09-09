__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from copy import deepcopy

import pytest

from src.environment import (
    EnvironmentState,
)
from src.interventions import (
    InterventionEngine,
    InterventionExecutionError,
    InterventionKind,
    InterventionOperation,
    InterventionPair,
    InterventionPairValidator,
    InterventionSchemaError,
    InterventionSpec,
    InterventionValidationError,
)
from src.logging import (
    Condition,
    ContextElement,
    SourceClass,
)
from src.memory import (
    SQLiteMemoryStore,
)
from src.snapshots import (
    SnapshotManager,
)


def make_prompt_elements():
    return (
        ContextElement(
            source=SourceClass.P,
            source_id="prompt:task_001",
            content=(
                "Current mode is ALPHA. "
                "Choose the correct route."
            ),
        ),
    )


def make_tool_elements():
    return (
        ContextElement(
            source=SourceClass.T,
            source_id="tool_return:evt_tool_relevant",
            content={
                "threshold": 7,
            },
            origin_event_id=(
                "evt_tool_relevant"
            ),
            origin_source=SourceClass.T,
        ),
        ContextElement(
            source=SourceClass.T,
            source_id="tool_return:evt_tool_irrelevant",
            content={
                "label": "blue",
            },
            origin_event_id=(
                "evt_tool_irrelevant"
            ),
            origin_source=SourceClass.T,
        ),
    )


def make_history_elements():
    return (
        ContextElement(
            source=SourceClass.H,
            source_id="history:evt_history_relevant",
            content={
                "previous_route": "A",
            },
            origin_event_id=(
                "evt_history_relevant"
            ),
            origin_source=None,
        ),
        ContextElement(
            source=SourceClass.H,
            source_id="history:evt_history_irrelevant",
            content={
                "marker": "unused",
            },
            origin_event_id=(
                "evt_history_irrelevant"
            ),
            origin_source=None,
        ),
    )


def make_memory_store(
    tmp_path,
):
    store = SQLiteMemoryStore(
        tmp_path
        / "memory.sqlite3"
    )

    store.initialize()

    store.write(
        namespace="run_001",
        memory_id="mem_relevant",
        key="preferred_route",
        value="B",
    )

    store.write(
        namespace="run_001",
        memory_id="mem_irrelevant",
        key="color",
        value="blue",
    )

    return store


def make_environment():
    return EnvironmentState(
        {
            "entities": {
                "door_1": {
                    "status": "locked",
                    "color": "red",
                }
            },
            "global_flags": {
                "alarm": False,
            },
        }
    )


def make_snapshot(
    tmp_path,
):
    store = make_memory_store(
        tmp_path
    )

    environment = (
        make_environment()
    )

    manager = SnapshotManager()

    snapshot = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C3,
        repetition_id=1,
        sequence_position=10,
        prompt_elements=(
            make_prompt_elements()
        ),
        tool_elements=(
            make_tool_elements()
        ),
        history_elements=(
            make_history_elements()
        ),
        memory_store=store,
        memory_namespace="run_001",
        environment=environment,
        runtime_configuration={
            "temperature": None,
            "decoding_strategy": (
                "greedy_argmax"
            ),
        },
        task_control_state={
            "step": 3,
        },
        seed_metadata={
            "inference_seed": 12345,
        },
    )

    return (
        snapshot,
        store,
        environment,
    )


def targeted_replace(
    *,
    source,
    target_id=None,
    target_path=None,
    replacement_value=None,
    pair_id="pair_001",
):
    return InterventionSpec(
        intervention_id="targeted_001",
        pair_id=pair_id,
        kind=(
            InterventionKind.TARGETED
        ),
        source=source,
        operation=(
            InterventionOperation.REPLACE
        ),
        target_id=target_id,
        target_path=target_path,
        replacement_value=(
            replacement_value
        ),
    )


def sham_replace(
    *,
    source,
    target_id=None,
    target_path=None,
    replacement_value=None,
    pair_id="pair_001",
):
    return InterventionSpec(
        intervention_id="sham_001",
        pair_id=pair_id,
        kind=(
            InterventionKind.SHAM
        ),
        source=source,
        operation=(
            InterventionOperation.REPLACE
        ),
        target_id=target_id,
        target_path=target_path,
        replacement_value=(
            replacement_value
        ),
    )


def test_p_replacement_changes_only_p(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    spec = targeted_replace(
        source=SourceClass.P,
        target_id="prompt:task_001",
        replacement_value=(
            "Current mode is BETA. "
            "Choose the correct route."
        ),
    )

    result = InterventionEngine().apply(
        snapshot=snapshot,
        spec=spec,
    )

    assert (
        result.changed_sources
        == (
            SourceClass.P,
        )
    )

    assert (
        result.counterfactual_state
        .prompt_elements[0]
        .content
        .startswith(
            "Current mode is BETA"
        )
    )


def test_p_removal_is_rejected() -> None:
    with pytest.raises(
        InterventionSchemaError
    ) as exc_info:
        InterventionSpec(
            intervention_id="targeted",
            pair_id="pair",
            kind=(
                InterventionKind.TARGETED
            ),
            source=SourceClass.P,
            operation=(
                InterventionOperation.REMOVE
            ),
            target_id="prompt:task_001",
        )

    assert (
        exc_info.value.code
        == "unsupported_prompt_operation"
    )


def test_t_replacement_changes_only_t(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    spec = targeted_replace(
        source=SourceClass.T,
        target_id=(
            "tool_return:"
            "evt_tool_relevant"
        ),
        replacement_value={
            "threshold": 9,
        },
    )

    result = InterventionEngine().apply(
        snapshot=snapshot,
        spec=spec,
    )

    assert (
        result.changed_sources
        == (
            SourceClass.T,
        )
    )

    assert (
        result.diff.before
        == {
            "threshold": 7,
        }
    )

    assert (
        result.diff.after
        == {
            "threshold": 9,
        }
    )


def test_t_remove_removes_only_target(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    spec = InterventionSpec(
        intervention_id="targeted",
        pair_id="pair",
        kind=(
            InterventionKind.TARGETED
        ),
        source=SourceClass.T,
        operation=(
            InterventionOperation.REMOVE
        ),
        target_id=(
            "tool_return:"
            "evt_tool_relevant"
        ),
    )

    result = InterventionEngine().apply(
        snapshot=snapshot,
        spec=spec,
    )

    remaining = [
        element.source_id
        for element
        in (
            result
            .counterfactual_state
            .tool_elements
        )
    ]

    assert remaining == [
        "tool_return:evt_tool_irrelevant"
    ]


def test_h_remove_changes_only_h(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    spec = InterventionSpec(
        intervention_id="targeted",
        pair_id="pair",
        kind=(
            InterventionKind.TARGETED
        ),
        source=SourceClass.H,
        operation=(
            InterventionOperation.REMOVE
        ),
        target_id=(
            "history:"
            "evt_history_relevant"
        ),
    )

    result = InterventionEngine().apply(
        snapshot=snapshot,
        spec=spec,
    )

    assert (
        result.changed_sources
        == (
            SourceClass.H,
        )
    )


def test_m_replacement_preserves_memory_identity(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    spec = targeted_replace(
        source=SourceClass.M,
        target_id="mem_relevant",
        replacement_value="A",
    )

    result = InterventionEngine().apply(
        snapshot=snapshot,
        spec=spec,
    )

    record = next(
        record
        for record
        in (
            result
            .counterfactual_state
            .memory_records
        )
        if (
            record.memory_id
            == "mem_relevant"
        )
    )

    assert (
        record.memory_id
        == "mem_relevant"
    )

    assert (
        record.key
        == "preferred_route"
    )

    assert record.value == "A"


def test_m_remove_removes_only_target(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    spec = InterventionSpec(
        intervention_id="targeted",
        pair_id="pair",
        kind=(
            InterventionKind.TARGETED
        ),
        source=SourceClass.M,
        operation=(
            InterventionOperation.REMOVE
        ),
        target_id="mem_relevant",
    )

    result = InterventionEngine().apply(
        snapshot=snapshot,
        spec=spec,
    )

    ids = [
        record.memory_id
        for record
        in (
            result
            .counterfactual_state
            .memory_records
        )
    ]

    assert ids == [
        "mem_irrelevant"
    ]


def test_e_replacement_changes_only_e(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    spec = targeted_replace(
        source=SourceClass.E,
        target_path=(
            "entities",
            "door_1",
            "status",
        ),
        replacement_value=(
            "unlocked"
        ),
    )

    result = InterventionEngine().apply(
        snapshot=snapshot,
        spec=spec,
    )

    assert (
        result.changed_sources
        == (
            SourceClass.E,
        )
    )

    assert (
        result
        .counterfactual_state
        .environment_state[
            "entities"
        ][
            "door_1"
        ][
            "status"
        ]
        == "unlocked"
    )


def test_e_remove_is_rejected() -> None:
    with pytest.raises(
        InterventionSchemaError
    ) as exc_info:
        InterventionSpec(
            intervention_id="targeted",
            pair_id="pair",
            kind=(
                InterventionKind.TARGETED
            ),
            source=SourceClass.E,
            operation=(
                InterventionOperation.REMOVE
            ),
            target_path=(
                "entities",
                "door_1",
                "status",
            ),
        )

    assert (
        exc_info.value.code
        == "unsupported_environment_operation"
    )


def test_missing_target_is_rejected(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    spec = targeted_replace(
        source=SourceClass.T,
        target_id="missing",
        replacement_value={
            "threshold": 9,
        },
    )

    with pytest.raises(
        InterventionExecutionError
    ) as exc_info:
        InterventionEngine().apply(
            snapshot=snapshot,
            spec=spec,
        )

    assert (
        exc_info.value.code
        == "intervention_target_not_found"
    )


def test_same_replacement_is_rejected_as_no_effect(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    spec = targeted_replace(
        source=SourceClass.T,
        target_id=(
            "tool_return:"
            "evt_tool_relevant"
        ),
        replacement_value={
            "threshold": 7,
        },
    )

    with pytest.raises(
        InterventionExecutionError
    ) as exc_info:
        InterventionEngine().apply(
            snapshot=snapshot,
            spec=spec,
        )

    assert (
        exc_info.value.code
        == "intervention_no_effect"
    )


def test_baseline_snapshot_is_not_modified(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    before = deepcopy(
        snapshot.to_dict()
    )

    spec = targeted_replace(
        source=SourceClass.M,
        target_id="mem_relevant",
        replacement_value="A",
    )

    InterventionEngine().apply(
        snapshot=snapshot,
        spec=spec,
    )

    assert (
        snapshot.to_dict()
        == before
    )


def test_intervention_application_is_deterministic(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    spec = targeted_replace(
        source=SourceClass.T,
        target_id=(
            "tool_return:"
            "evt_tool_relevant"
        ),
        replacement_value={
            "threshold": 9,
        },
    )

    engine = InterventionEngine()

    first = engine.apply(
        snapshot=snapshot,
        spec=spec,
    )

    second = engine.apply(
        snapshot=snapshot,
        spec=spec,
    )

    assert (
        first.to_dict()
        == second.to_dict()
    )


def test_pair_requires_same_source() -> None:
    pair = InterventionPair(
        pair_id="pair_001",
        targeted=targeted_replace(
            source=SourceClass.T,
            target_id=(
                "tool_return:"
                "evt_tool_relevant"
            ),
            replacement_value={
                "threshold": 9,
            },
        ),
        sham=sham_replace(
            source=SourceClass.M,
            target_id="mem_irrelevant",
            replacement_value="green",
        ),
    )

    with pytest.raises(
        InterventionValidationError
    ) as exc_info:
        (
            InterventionPairValidator()
            .validate(
                snapshot=None,
                pair=pair,
            )
        )

    assert (
        exc_info.value.code
        == "sham_source_mismatch"
    )


def test_targeted_and_sham_use_same_baseline(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    pair = InterventionPair(
        pair_id="pair_001",
        targeted=targeted_replace(
            source=SourceClass.T,
            target_id=(
                "tool_return:"
                "evt_tool_relevant"
            ),
            replacement_value={
                "threshold": 9,
            },
        ),
        sham=sham_replace(
            source=SourceClass.T,
            target_id=(
                "tool_return:"
                "evt_tool_irrelevant"
            ),
            replacement_value={
                "label": "green",
            },
        ),
    )

    result = (
        InterventionPairValidator()
        .validate(
            snapshot=snapshot,
            pair=pair,
        )
    )

    assert (
        result
        .targeted_application
        .baseline_snapshot_id
        == snapshot.snapshot_id
    )

    assert (
        result
        .sham_application
        .baseline_snapshot_id
        == snapshot.snapshot_id
    )

    assert (
        result
        .targeted_application
        .baseline_state_checksum
        == result
        .sham_application
        .baseline_state_checksum
    )


def test_pair_applications_each_change_only_same_source(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    pair = InterventionPair(
        pair_id="pair_001",
        targeted=targeted_replace(
            source=SourceClass.T,
            target_id=(
                "tool_return:"
                "evt_tool_relevant"
            ),
            replacement_value={
                "threshold": 9,
            },
        ),
        sham=sham_replace(
            source=SourceClass.T,
            target_id=(
                "tool_return:"
                "evt_tool_irrelevant"
            ),
            replacement_value={
                "label": "green",
            },
        ),
    )

    result = (
        InterventionPairValidator()
        .validate(
            snapshot=snapshot,
            pair=pair,
        )
    )

    assert (
        result
        .targeted_application
        .changed_sources
        == (
            SourceClass.T,
        )
    )

    assert (
        result
        .sham_application
        .changed_sources
        == (
            SourceClass.T,
        )
    )


def test_diff_records_length_information(
    tmp_path,
) -> None:
    snapshot, _, _ = (
        make_snapshot(
            tmp_path
        )
    )

    spec = targeted_replace(
        source=SourceClass.T,
        target_id=(
            "tool_return:"
            "evt_tool_relevant"
        ),
        replacement_value={
            "threshold": 9,
        },
    )

    result = InterventionEngine().apply(
        snapshot=snapshot,
        spec=spec,
    )

    assert (
        result.diff
        .before_serialized_chars
        > 0
    )

    assert (
        result.diff
        .after_serialized_chars
        > 0
    )