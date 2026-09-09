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

from src.memory import (
    MemoryStoreError,
    SQLiteMemoryStore,
)


def make_store(tmp_path) -> SQLiteMemoryStore:
    store = SQLiteMemoryStore(
        tmp_path / "memory.sqlite3"
    )
    store.initialize()
    return store


def test_write_and_read_round_trip(tmp_path) -> None:
    store = make_store(tmp_path)

    written = store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="threshold",
        value={
            "value": 7,
            "units": "points",
        },
        created_by_task="setup_001",
        created_sequence=4,
        metadata={
            "source": "preregistered_seed",
        },
    )

    read = store.read(
        namespace="run_001",
        memory_id="mem_001",
    )

    assert read == written

    assert read.value == {
        "value": 7,
        "units": "points",
    }


def test_memory_persists_across_store_instances(tmp_path) -> None:
    path = tmp_path / "memory.sqlite3"

    first_store = SQLiteMemoryStore(path)
    first_store.initialize()

    first_store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="mode",
        value="safe",
    )

    second_store = SQLiteMemoryStore(path)

    result = second_store.read(
        namespace="run_001",
        memory_id="mem_001",
    )

    assert result is not None
    assert result.value == "safe"


def test_duplicate_memory_id_is_rejected(tmp_path) -> None:
    store = make_store(tmp_path)

    store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="threshold",
        value=7,
    )

    with pytest.raises(MemoryStoreError) as exc_info:
        store.write(
            namespace="run_001",
            memory_id="mem_001",
            key="threshold",
            value=9,
        )

    assert exc_info.value.code == "duplicate_memory_id"


def test_same_memory_id_isolated_by_namespace(tmp_path) -> None:
    store = make_store(tmp_path)

    store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="threshold",
        value=7,
    )

    store.write(
        namespace="run_002",
        memory_id="mem_001",
        key="threshold",
        value=11,
    )

    first = store.read(
        namespace="run_001",
        memory_id="mem_001",
    )

    second = store.read(
        namespace="run_002",
        memory_id="mem_001",
    )

    assert first is not None
    assert second is not None

    assert first.value == 7
    assert second.value == 11


def test_missing_memory_returns_none(tmp_path) -> None:
    store = make_store(tmp_path)

    result = store.read(
        namespace="run_001",
        memory_id="missing",
    )

    assert result is None


def test_list_records_is_deterministically_ordered(tmp_path) -> None:
    store = make_store(tmp_path)

    for memory_id in [
        "mem_003",
        "mem_001",
        "mem_002",
    ]:
        store.write(
            namespace="run_001",
            memory_id=memory_id,
            key="test",
            value=memory_id,
        )

    records = store.list_records(
        namespace="run_001"
    )

    assert [
        record.memory_id
        for record in records
    ] == [
        "mem_001",
        "mem_002",
        "mem_003",
    ]


def test_targeted_replace_preserves_memory_identity(tmp_path) -> None:
    store = make_store(tmp_path)

    store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="threshold",
        value=7,
        created_by_task="setup_001",
        created_sequence=5,
        metadata={"kind": "preference"},
    )

    mutation = store.targeted_replace(
        namespace="run_001",
        memory_id="mem_001",
        replacement_value=12,
    )

    assert mutation.operation == "replace"

    assert mutation.before is not None
    assert mutation.after is not None

    assert mutation.before.value == 7
    assert mutation.after.value == 12

    assert mutation.after.memory_id == mutation.before.memory_id
    assert mutation.after.key == mutation.before.key

    assert (
        mutation.after.created_by_task
        == mutation.before.created_by_task
    )

    assert (
        mutation.after.created_sequence
        == mutation.before.created_sequence
    )

    assert mutation.after.metadata == mutation.before.metadata


def test_targeted_remove_removes_only_target_memory(tmp_path) -> None:
    store = make_store(tmp_path)

    store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="first",
        value=1,
    )

    store.write(
        namespace="run_001",
        memory_id="mem_002",
        key="second",
        value=2,
    )

    mutation = store.targeted_remove(
        namespace="run_001",
        memory_id="mem_001",
    )

    assert mutation.operation == "remove"
    assert mutation.before is not None
    assert mutation.after is None

    assert (
        store.read(
            namespace="run_001",
            memory_id="mem_001",
        )
        is None
    )

    remaining = store.read(
        namespace="run_001",
        memory_id="mem_002",
    )

    assert remaining is not None
    assert remaining.value == 2


def test_snapshot_restore_recovers_exact_namespace_state(
    tmp_path,
) -> None:
    store = make_store(tmp_path)

    store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="threshold",
        value=7,
    )

    store.write(
        namespace="run_001",
        memory_id="mem_002",
        key="mode",
        value="ready",
    )

    before = store.list_records(
        namespace="run_001"
    )

    snapshot = store.snapshot(
        namespace="run_001"
    )

    store.targeted_replace(
        namespace="run_001",
        memory_id="mem_001",
        replacement_value=999,
    )

    store.targeted_remove(
        namespace="run_001",
        memory_id="mem_002",
    )

    store.write(
        namespace="run_001",
        memory_id="mem_003",
        key="extra",
        value=True,
    )

    store.restore(snapshot)

    after = store.list_records(
        namespace="run_001"
    )

    assert after == before


def test_restore_does_not_modify_other_namespace(tmp_path) -> None:
    store = make_store(tmp_path)

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

    snapshot = store.snapshot(
        namespace="run_001"
    )

    store.targeted_replace(
        namespace="run_001",
        memory_id="mem_001",
        replacement_value=2,
    )

    store.restore(snapshot)

    other = store.read(
        namespace="run_002",
        memory_id="mem_001",
    )

    assert other is not None
    assert other.value == 100


def test_snapshot_checksum_is_stable_for_same_state(
    tmp_path,
) -> None:
    store = make_store(tmp_path)

    store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="threshold",
        value={
            "b": 2,
            "a": 1,
        },
    )

    first = store.snapshot(
        namespace="run_001"
    )

    second = store.snapshot(
        namespace="run_001"
    )

    assert (
        first.checksum_sha256
        == second.checksum_sha256
    )


def test_restore_rejects_invalid_snapshot_checksum(
    tmp_path,
) -> None:
    store = make_store(tmp_path)

    store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="threshold",
        value=7,
    )

    snapshot = store.snapshot(
        namespace="run_001"
    )

    corrupted = replace(
        snapshot,
        checksum_sha256="0" * 64,
    )

    with pytest.raises(MemoryStoreError) as exc_info:
        store.restore(corrupted)

    assert (
        exc_info.value.code
        == "snapshot_checksum_mismatch"
    )


def test_non_json_memory_value_is_rejected(tmp_path) -> None:
    store = make_store(tmp_path)

    with pytest.raises(MemoryStoreError) as exc_info:
        store.write(
            namespace="run_001",
            memory_id="mem_001",
            key="invalid",
            value={1, 2, 3},
        )

    assert exc_info.value.code == "non_serializable_value"