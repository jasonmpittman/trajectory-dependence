__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from dataclasses import replace

import pytest

from src.environment import EnvironmentState
from src.logging import (
    Condition,
    ContextElement,
    SourceClass,
)
from src.memory import SQLiteMemoryStore
from src.snapshots import (
    SnapshotError,
    SnapshotManager,
)


def make_prompt_elements():
    return (
        ContextElement(
            source=SourceClass.P,
            source_id="prompt:task_001",
            content={
                "instruction": "Choose A or B.",
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


def make_environment():
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


def make_memory_store(tmp_path):
    store = SQLiteMemoryStore(
        tmp_path / "memory.sqlite3"
    )

    store.initialize()

    store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="threshold",
        value=7,
        created_by_task="setup_001",
        created_sequence=3,
    )

    return store


def test_capture_preserves_structured_state() -> None:
    manager = SnapshotManager()

    snapshot = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C2,
        repetition_id=1,
        sequence_position=8,
        prompt_elements=make_prompt_elements(),
        tool_elements=make_tool_elements(),
        history_elements=make_history_elements(),
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

    assert snapshot.sequence_position == 8

    assert (
        snapshot.prompt_elements[0].source
        == SourceClass.P
    )

    assert (
        snapshot.tool_elements[0].source
        == SourceClass.T
    )

    assert (
        snapshot.history_elements[0].source
        == SourceClass.H
    )

    assert (
        snapshot.history_elements[0].origin_source
        == SourceClass.T
    )


def test_capture_defensively_copies_context() -> None:
    manager = SnapshotManager()

    mutable_content = {
        "value": [1, 2],
    }

    prompt = (
        ContextElement(
            source=SourceClass.P,
            source_id="prompt:mutable",
            content=mutable_content,
        ),
    )

    snapshot = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C0,
        repetition_id=1,
        sequence_position=1,
        prompt_elements=prompt,
    )

    mutable_content["value"].append(3)

    assert snapshot.prompt_elements[0].content == {
        "value": [1, 2],
    }


def test_source_bucket_mismatch_is_rejected() -> None:
    manager = SnapshotManager()

    incorrect = (
        ContextElement(
            source=SourceClass.M,
            source_id="memory:mem_001",
            content=7,
        ),
    )

    with pytest.raises(
        SnapshotError
    ) as exc_info:
        manager.capture(
            experiment_id="exp_001",
            run_id="run_001",
            task_id="task_001",
            condition=Condition.C0,
            repetition_id=1,
            sequence_position=1,
            prompt_elements=incorrect,
        )

    assert (
        exc_info.value.code
        == "source_bucket_mismatch"
    )


def test_c0_rejects_tool_state() -> None:
    manager = SnapshotManager()

    with pytest.raises(
        SnapshotError
    ) as exc_info:
        manager.capture(
            experiment_id="exp_001",
            run_id="run_001",
            task_id="task_001",
            condition=Condition.C0,
            repetition_id=1,
            sequence_position=1,
            prompt_elements=make_prompt_elements(),
            tool_elements=make_tool_elements(),
        )

    assert (
        exc_info.value.code
        == "condition_isolation_violation"
    )


def test_c1_rejects_history_state() -> None:
    manager = SnapshotManager()

    with pytest.raises(
        SnapshotError
    ) as exc_info:
        manager.capture(
            experiment_id="exp_001",
            run_id="run_001",
            task_id="task_001",
            condition=Condition.C1,
            repetition_id=1,
            sequence_position=4,
            prompt_elements=make_prompt_elements(),
            tool_elements=make_tool_elements(),
            history_elements=make_history_elements(),
        )

    assert (
        exc_info.value.code
        == "condition_isolation_violation"
    )


def test_c2_rejects_persistent_memory(tmp_path) -> None:
    manager = SnapshotManager()

    store = make_memory_store(
        tmp_path
    )

    with pytest.raises(
        SnapshotError
    ) as exc_info:
        manager.capture(
            experiment_id="exp_001",
            run_id="run_001",
            task_id="task_001",
            condition=Condition.C2,
            repetition_id=1,
            sequence_position=8,
            prompt_elements=make_prompt_elements(),
            tool_elements=make_tool_elements(),
            history_elements=make_history_elements(),
            memory_store=store,
            memory_namespace="run_001",
        )

    assert (
        exc_info.value.code
        == "condition_isolation_violation"
    )


def test_c3_captures_persistent_memory(tmp_path) -> None:
    manager = SnapshotManager()

    store = make_memory_store(
        tmp_path
    )

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
    )

    assert snapshot.memory_snapshot is not None

    assert (
        snapshot.memory_snapshot.records[0].memory_id
        == "mem_001"
    )


def test_snapshot_captures_and_restores_environment() -> None:
    manager = SnapshotManager()

    environment = make_environment()

    snapshot = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C0,
        repetition_id=1,
        sequence_position=1,
        prompt_elements=make_prompt_elements(),
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

    manager.restore(
        snapshot,
        environment=environment,
    )

    result = environment.query(
        [
            "entities",
            "door_1",
            "status",
        ]
    )

    assert result.value == "locked"


def test_unified_restore_recovers_memory_and_environment(
    tmp_path,
) -> None:
    manager = SnapshotManager()

    store = make_memory_store(
        tmp_path
    )

    environment = make_environment()

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

    restored = manager.restore(
        snapshot,
        memory_store=store,
        environment=environment,
    )

    memory = store.read(
        namespace="run_001",
        memory_id="mem_001",
    )

    alarm = environment.query(
        [
            "global_flags",
            "alarm",
        ]
    )

    assert memory is not None
    assert memory.value == 7
    assert alarm.value is False

    assert restored.sequence_position == 8

    assert restored.seed_metadata == {
        "inference_seed": 12345,
    }


def test_same_state_produces_same_state_checksum() -> None:
    manager = SnapshotManager()

    arguments = {
        "experiment_id": "exp_001",
        "run_id": "run_001",
        "task_id": "task_001",
        "condition": Condition.C2,
        "repetition_id": 1,
        "sequence_position": 8,
        "prompt_elements": make_prompt_elements(),
        "tool_elements": make_tool_elements(),
        "history_elements": make_history_elements(),
        "runtime_configuration": {
            "temperature": 0.0,
        },
        "task_control_state": {
            "step": 2,
        },
        "seed_metadata": {
            "inference_seed": 12345,
        },
    }

    first = manager.capture(
        **arguments
    )

    second = manager.capture(
        **arguments
    )

    assert first.snapshot_id != second.snapshot_id

    assert (
        first.state_checksum_sha256
        == second.state_checksum_sha256
    )


def test_corrupted_unified_checksum_is_rejected() -> None:
    manager = SnapshotManager()

    snapshot = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C0,
        repetition_id=1,
        sequence_position=1,
        prompt_elements=make_prompt_elements(),
    )

    corrupted = replace(
        snapshot,
        state_checksum_sha256="0" * 64,
    )

    with pytest.raises(
        SnapshotError
    ) as exc_info:
        manager.validate(
            corrupted
        )

    assert (
        exc_info.value.code
        == "snapshot_checksum_mismatch"
    )


def test_snapshot_save_load_round_trip(tmp_path) -> None:
    manager = SnapshotManager()

    snapshot = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C2,
        repetition_id=1,
        sequence_position=8,
        prompt_elements=make_prompt_elements(),
        tool_elements=make_tool_elements(),
        history_elements=make_history_elements(),
        runtime_configuration={
            "temperature": 0.0,
        },
        seed_metadata={
            "inference_seed": 12345,
        },
    )

    path = tmp_path / "snapshot.json"

    manager.save(
        snapshot,
        path,
    )

    loaded = manager.load(
        path
    )

    assert loaded.to_dict() == snapshot.to_dict()


def test_snapshot_save_refuses_overwrite(tmp_path) -> None:
    manager = SnapshotManager()

    snapshot = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C0,
        repetition_id=1,
        sequence_position=1,
        prompt_elements=make_prompt_elements(),
    )

    path = tmp_path / "snapshot.json"

    manager.save(
        snapshot,
        path,
    )

    with pytest.raises(
        FileExistsError
    ):
        manager.save(
            snapshot,
            path,
        )


def test_restore_requires_memory_store(tmp_path) -> None:
    manager = SnapshotManager()

    store = make_memory_store(
        tmp_path
    )

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
    )

    with pytest.raises(
        SnapshotError
    ) as exc_info:
        manager.restore(
            snapshot
        )

    assert (
        exc_info.value.code
        == "memory_store_required"
    )


def test_restore_requires_environment() -> None:
    manager = SnapshotManager()

    environment = make_environment()

    snapshot = manager.capture(
        experiment_id="exp_001",
        run_id="run_001",
        task_id="task_001",
        condition=Condition.C0,
        repetition_id=1,
        sequence_position=1,
        prompt_elements=make_prompt_elements(),
        environment=environment,
    )

    with pytest.raises(
        SnapshotError
    ) as exc_info:
        manager.restore(
            snapshot
        )

    assert (
        exc_info.value.code
        == "environment_required"
    )