__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from src.environment import (
    DeterministicTransitionEngine,
    EnvironmentState,
    TransitionRule,
)
from src.logging import (
    Condition,
    ContextElement,
    SourceClass,
)
from src.memory import SQLiteMemoryStore
from src.snapshots import SnapshotManager
from src.tools import (
    EnvironmentQueryTool,
    KeyValueLookupTool,
)


def make_environment() -> EnvironmentState:
    return EnvironmentState(
        {
            "entities": {
                "door_1": {
                    "status": "locked",
                }
            },
            "global_flags": {
                "alarm": False,
            },
        }
    )


def make_transition_engine() -> DeterministicTransitionEngine:
    return DeterministicTransitionEngine(
        [
            TransitionRule(
                rule_id="unlock_door_1",
                match={
                    "action": "unlock",
                    "target": "door_1",
                },
                path=(
                    "entities",
                    "door_1",
                    "status",
                ),
                required_value="locked",
                new_value="unlocked",
            )
        ]
    )


def make_memory_store(
    tmp_path,
    filename: str = "memory.sqlite3",
) -> SQLiteMemoryStore:
    store = SQLiteMemoryStore(
        tmp_path / filename
    )

    store.initialize()

    return store


def make_prompt_elements():
    return (
        ContextElement(
            source=SourceClass.P,
            source_id="prompt:task_001",
            content={
                "instruction": "Choose the correct action.",
            },
        ),
    )


def make_tool_elements():
    return (
        ContextElement(
            source=SourceClass.T,
            source_id="tool_return:event_004",
            content={
                "threshold": 7,
            },
            origin_event_id="event_004",
            origin_source=SourceClass.T,
            created_sequence=4,
        ),
    )


def make_history_elements():
    return (
        ContextElement(
            source=SourceClass.H,
            source_id="history:event_004",
            content={
                "threshold": 7,
            },
            origin_event_id="event_004",
            origin_source=SourceClass.T,
            created_sequence=4,
            retrieved_sequence=8,
        ),
    )


def test_combined_memory_environment_restore_returns_exact_state(
    tmp_path,
) -> None:
    store = make_memory_store(
        tmp_path
    )

    store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="threshold",
        value=7,
    )

    environment = make_environment()

    manager = SnapshotManager()

    snapshot = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C3,
        repetition_id=1,
        sequence_position=8,
        prompt_elements=make_prompt_elements(),
        tool_elements=make_tool_elements(),
        history_elements=make_history_elements(),
        memory_store=store,
        memory_namespace="run_001",
        environment=environment,
        runtime_configuration={
            "temperature": 0.0,
        },
        task_control_state={
            "step": 2,
        },
        seed_metadata={
            "inference_seed": 12345,
        },
    )

    original_memory_checksum = (
        snapshot.memory_snapshot.checksum_sha256
    )

    original_environment_checksum = (
        snapshot.environment_snapshot.checksum_sha256
    )

    store.targeted_replace(
        namespace="run_001",
        memory_id="mem_001",
        replacement_value=999,
    )

    environment.targeted_modify(
        path=[
            "global_flags",
            "alarm",
        ],
        value=True,
    )

    manager.restore(
        snapshot,
        memory_store=store,
        environment=environment,
    )

    restored_memory = store.snapshot(
        namespace="run_001"
    )

    restored_environment = environment.snapshot()

    assert (
        restored_memory.checksum_sha256
        == original_memory_checksum
    )

    assert (
        restored_environment.checksum_sha256
        == original_environment_checksum
    )


def test_restoring_one_memory_namespace_does_not_affect_another(
    tmp_path,
) -> None:
    store = make_memory_store(
        tmp_path
    )

    store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="value",
        value=1,
    )

    store.write(
        namespace="run_002",
        memory_id="mem_001",
        key="value",
        value=100,
    )

    environment = make_environment()

    manager = SnapshotManager()

    run_two_before = store.snapshot(
        namespace="run_002"
    )

    snapshot = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C3,
        repetition_id=1,
        sequence_position=2,
        prompt_elements=make_prompt_elements(),
        memory_store=store,
        memory_namespace="run_001",
        environment=environment,
    )

    store.targeted_replace(
        namespace="run_001",
        memory_id="mem_001",
        replacement_value=999,
    )

    manager.restore(
        snapshot,
        memory_store=store,
        environment=environment,
    )

    run_two_after = store.snapshot(
        namespace="run_002"
    )

    assert (
        run_two_after.checksum_sha256
        == run_two_before.checksum_sha256
    )

    assert (
        run_two_after.records
        == run_two_before.records
    )


def test_same_initial_state_and_transition_produce_same_final_state(
    tmp_path,
) -> None:
    first_environment = make_environment()
    second_environment = make_environment()

    engine = make_transition_engine()

    action = {
        "action": "unlock",
        "target": "door_1",
    }

    first_result = engine.apply(
        environment=first_environment,
        action=action,
    )

    second_result = engine.apply(
        environment=second_environment,
        action=action,
    )

    assert (
        first_result.to_dict()
        == second_result.to_dict()
    )

    assert (
        first_environment.checksum_sha256()
        == second_environment.checksum_sha256()
    )


def test_same_complete_pre_focal_state_produces_same_checksum(
    tmp_path,
) -> None:
    first_store = make_memory_store(
        tmp_path,
        "first.sqlite3",
    )

    second_store = make_memory_store(
        tmp_path,
        "second.sqlite3",
    )

    for store in (
        first_store,
        second_store,
    ):
        store.write(
            namespace="run_001",
            memory_id="mem_001",
            key="threshold",
            value=7,
        )

    first_environment = make_environment()
    second_environment = make_environment()

    manager = SnapshotManager()

    common = {
        "experiment_id": "exp_001",
        "run_id": "run_001",
        "task_id": "task_001",
        "condition": Condition.C3,
        "repetition_id": 1,
        "sequence_position": 8,
        "prompt_elements": make_prompt_elements(),
        "tool_elements": make_tool_elements(),
        "history_elements": make_history_elements(),
        "memory_namespace": "run_001",
        "runtime_configuration": {
            "temperature": 0.0,
            "thinking_mode": False,
        },
        "task_control_state": {
            "step": 2,
        },
        "seed_metadata": {
            "inference_seed": 12345,
        },
    }

    first = manager.capture(
        **common,
        memory_store=first_store,
        environment=first_environment,
    )

    second = manager.capture(
        **common,
        memory_store=second_store,
        environment=second_environment,
    )

    assert first.snapshot_id != second.snapshot_id

    assert (
        first.state_checksum_sha256
        == second.state_checksum_sha256
    )


def test_memory_change_changes_complete_state_checksum(
    tmp_path,
) -> None:
    store = make_memory_store(
        tmp_path
    )

    store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="threshold",
        value=7,
    )

    environment = make_environment()
    manager = SnapshotManager()

    first = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C3,
        repetition_id=1,
        sequence_position=8,
        prompt_elements=make_prompt_elements(),
        memory_store=store,
        memory_namespace="run_001",
        environment=environment,
    )

    store.targeted_replace(
        namespace="run_001",
        memory_id="mem_001",
        replacement_value=8,
    )

    second = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C3,
        repetition_id=1,
        sequence_position=8,
        prompt_elements=make_prompt_elements(),
        memory_store=store,
        memory_namespace="run_001",
        environment=environment,
    )

    assert (
        first.state_checksum_sha256
        != second.state_checksum_sha256
    )


def test_environment_change_changes_complete_state_checksum(
    tmp_path,
) -> None:
    store = make_memory_store(
        tmp_path
    )

    environment = make_environment()
    manager = SnapshotManager()

    first = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C3,
        repetition_id=1,
        sequence_position=8,
        prompt_elements=make_prompt_elements(),
        memory_store=store,
        memory_namespace="run_001",
        environment=environment,
    )

    environment.targeted_modify(
        path=[
            "entities",
            "door_1",
            "status",
        ],
        value="unlocked",
    )

    second = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C3,
        repetition_id=1,
        sequence_position=8,
        prompt_elements=make_prompt_elements(),
        memory_store=store,
        memory_namespace="run_001",
        environment=environment,
    )

    assert (
        first.state_checksum_sha256
        != second.state_checksum_sha256
    )


def test_prompt_change_changes_complete_state_checksum(
    tmp_path,
) -> None:
    store = make_memory_store(
        tmp_path
    )

    environment = make_environment()
    manager = SnapshotManager()

    first_prompt = (
        ContextElement(
            source=SourceClass.P,
            source_id="prompt:task_001",
            content={
                "instruction": "Choose A.",
            },
        ),
    )

    second_prompt = (
        ContextElement(
            source=SourceClass.P,
            source_id="prompt:task_001",
            content={
                "instruction": "Choose B.",
            },
        ),
    )

    common = {
        "experiment_id": "exp_001",
        "run_id": "run_001",
        "task_id": "task_001",
        "condition": Condition.C3,
        "repetition_id": 1,
        "sequence_position": 8,
        "memory_store": store,
        "memory_namespace": "run_001",
        "environment": environment,
    }

    first = manager.capture(
        **common,
        prompt_elements=first_prompt,
    )

    second = manager.capture(
        **common,
        prompt_elements=second_prompt,
    )

    assert (
        first.state_checksum_sha256
        != second.state_checksum_sha256
    )


def test_tool_query_does_not_change_environment_checksum() -> None:
    environment = make_environment()

    tool = EnvironmentQueryTool()

    before = environment.checksum_sha256()

    result = tool.execute(
        {
            "path": [
                "entities",
                "door_1",
                "status",
            ]
        },
        environment.get_state(),
    )

    after = environment.checksum_sha256()

    assert result.normalized_result["value"] == "locked"

    assert before == after


def test_key_value_lookup_does_not_modify_source_state() -> None:
    tool = KeyValueLookupTool()

    state = {
        "threshold": 7,
    }

    before = dict(state)

    result = tool.execute(
        {
            "key": "threshold",
        },
        state,
    )

    assert result.normalized_result["value"] == 7

    assert state == before