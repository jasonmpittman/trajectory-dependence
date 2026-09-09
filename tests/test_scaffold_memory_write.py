__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.2"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.runtime import (
    AgentRuntime,
)
from src.agent.scaffold import (
    MemoryWriteObservation,
    ScaffoldController,
    ScaffoldExecutionError,
)
from src.logging import (
    Condition,
    EventType,
    SourceClass,
)
from src.memory import (
    SQLiteMemoryStore,
)

class FakeEvent:
    """
    Minimal event object matching the attribute-access surface consumed
    by ScaffoldController.
    """

    def __init__(
        self,
        *,
        event_id,
        sequence_index,
        event_type,
        source_component,
        raw_payload,
        normalized_payload,
        parent_event_id=None,
        tool_call_id=None,
        memory_id=None,
        metadata=None,
    ):
        self.event_id = event_id
        self.sequence_index = (
            sequence_index
        )
        self.event_type = event_type
        self.source_component = (
            source_component
        )
        self.raw_payload = (
            raw_payload
        )
        self.normalized_payload = (
            normalized_payload
        )
        self.parent_event_id = (
            parent_event_id
        )
        self.tool_call_id = (
            tool_call_id
        )
        self.memory_id = memory_id
        self.metadata = (
            metadata
            if metadata is not None
            else {}
        )


class FakeRuntime:
    """
    Minimal deterministic AgentRuntime substitute for scaffold state tests.

    It deliberately implements only the runtime surface used by
    ScaffoldController's memory lifecycle operations.
    """

    def __init__(
        self,
        *,
        task_id,
        condition,
    ):
        self.identity = SimpleNamespace(
            task_id=task_id,
            condition=condition,
        )

        self._events = []
        self._technical_failures = []

    @property
    def events(self):
        return tuple(
            self._events
        )

    @property
    def task_has_begun(self):
        return True

    @property
    def technical_failures(self):
        return tuple(
            self._technical_failures
        )

    def record_component_event(
        self,
        *,
        event_type,
        source_component,
        raw_payload,
        normalized_payload,
        parent_event_id=None,
        tool_call_id=None,
        memory_id=None,
        metadata=None,
    ):
        event = FakeEvent(
            event_id=(
                f"event-"
                f"{len(self._events)}"
            ),
            sequence_index=len(
                self._events
            ),
            event_type=event_type,
            source_component=(
                source_component
            ),
            raw_payload=(
                raw_payload
            ),
            normalized_payload=(
                normalized_payload
            ),
            parent_event_id=(
                parent_event_id
            ),
            tool_call_id=(
                tool_call_id
            ),
            memory_id=memory_id,
            metadata=metadata,
        )

        self._events.append(
            event
        )

        return event

    def record_technical_failure(
        self,
        *,
        stage,
        error,
        model_invocation_id=None,
    ):
        self._technical_failures.append(
            {
                "stage": stage,
                "error": error,
                "model_invocation_id": (
                    model_invocation_id
                ),
            }
        )
    
def build_runtime(
    *,
    condition: Condition,
):
    runtime = FakeRuntime(
        task_id=(
            "memory_write_test"
        ),
        condition=condition,
    )

    return (
        runtime,
        runtime._events,
    )

def build_store(
    tmp_path: Path,
) -> SQLiteMemoryStore:
    store = SQLiteMemoryStore(
        tmp_path
        / "memory.sqlite3"
    )

    store.initialize()

    return store


def test_write_memory_persists_record_and_logs_event(
    tmp_path,
) -> None:
    runtime, events = build_runtime(
        condition=Condition.C3
    )

    store = build_store(
        tmp_path
    )

    controller = ScaffoldController(
        runtime=runtime,
        memory_store=store,
        memory_namespace=(
            "memory_write_test"
        ),
    )

    result = controller.write_memory(
        memory_id="route_memory",
        key="preferred_route",
        value="west",
        metadata={
            "purpose": (
                "supplemental_test"
            )
        },
    )

    assert isinstance(
        result,
        MemoryWriteObservation,
    )

    assert (
        result.record.namespace
        == "memory_write_test"
    )

    assert (
        result.record.memory_id
        == "route_memory"
    )

    assert (
        result.record.key
        == "preferred_route"
    )

    assert (
        result.record.value
        == "west"
    )

    assert (
        result.record.created_by_task
        == "memory_write_test"
    )

    assert (
        result.record.created_sequence
        == 0
    )

    assert (
        result.record.metadata
        == {
            "purpose": (
                "supplemental_test"
            )
        }
    )

    assert len(
        events
    ) == 1

    event = events[
        0
    ]

    assert (
        event.event_type
        == EventType.MEMORY_WRITE
    )

    assert (
        event.sequence_index
        == 0
    )

    assert (
        event.memory_id
        == "route_memory"
    )

    assert (
        event.source_component
        == "persistent_memory"
    )

    assert (
        event.normalized_payload
        == {
            "key": "preferred_route",
            "value": "west",
        }
    )


def test_write_memory_sequence_matches_record_creation_sequence(
    tmp_path,
) -> None:
    runtime, events = build_runtime(
        condition=Condition.C3
    )

    # Simulate one prior event in the trajectory.
    events.append(
        FakeEvent(
            event_id="prior",
            sequence_index=0,
            event_type=(
                EventType.ACTION
            ),
            source_component="synthetic",
            raw_payload={},
            normalized_payload={},
            parent_event_id=None,
            tool_call_id=None,
            memory_id=None,
            metadata={},
        )
    )

    store = build_store(
        tmp_path
    )

    controller = ScaffoldController(
        runtime=runtime,
        memory_store=store,
        memory_namespace="namespace",
    )

    result = controller.write_memory(
        memory_id="memory",
        key="route",
        value="west",
    )

    assert (
        result.record.created_sequence
        == 1
    )

    assert (
        result.event.sequence_index
        == 1
    )


def test_write_then_read_anchors_model_visible_m_to_memory_read(
    tmp_path,
) -> None:
    runtime, events = build_runtime(
        condition=Condition.C3
    )

    store = build_store(
        tmp_path
    )

    controller = ScaffoldController(
        runtime=runtime,
        memory_store=store,
        memory_namespace="namespace",
    )

    written = controller.write_memory(
        memory_id="route_memory",
        key="preferred_route",
        value="west",
    )

    read = controller.read_memory(
        memory_id="route_memory"
    )

    assert read.found is True

    assert (
        read.record
        == written.record
    )

    assert (
        len(
            events
        )
        == 2
    )

    assert (
        events[
            0
        ].event_type
        == EventType.MEMORY_WRITE
    )

    assert (
        events[
            1
        ].event_type
        == EventType.MEMORY_READ
    )

    assert (
        read.memory_element
        is not None
    )

    assert (
        read.memory_element.source
        == SourceClass.M
    )

    assert (
        read.memory_element.origin_source
        == SourceClass.M
    )

    # Critical provenance invariant:
    # M is anchored to the read, not the prior write.
    assert (
        read.memory_element.origin_event_id
        == events[
            1
        ].event_id
    )

    assert (
        read.memory_element.origin_event_id
        != events[
            0
        ].event_id
    )

    assert (
        read.memory_element.content
        == {
            "key": "preferred_route",
            "value": "west",
        }
    )


@pytest.mark.parametrize(
    "condition",
    [
        Condition.C0,
        Condition.C1,
        Condition.C2,
    ],
)
def test_write_memory_rejected_outside_c3(
    tmp_path,
    condition,
) -> None:
    runtime, events = build_runtime(
        condition=condition
    )

    store = build_store(
        tmp_path
    )

    controller = ScaffoldController(
        runtime=runtime,
        memory_store=store,
        memory_namespace="namespace",
    )

    with pytest.raises(
        ScaffoldExecutionError
    ) as exc_info:
        controller.write_memory(
            memory_id="memory",
            key="route",
            value="west",
        )

    assert (
        exc_info.value.code
        == "memory_not_allowed"
    )

    assert events == []

    assert (
        store.read(
            namespace="namespace",
            memory_id="memory",
        )
        is None
    )


def test_duplicate_memory_write_records_technical_failure(
    tmp_path,
) -> None:
    runtime, events = build_runtime(
        condition=Condition.C3
    )

    store = build_store(
        tmp_path
    )

    controller = ScaffoldController(
        runtime=runtime,
        memory_store=store,
        memory_namespace="namespace",
    )

    controller.write_memory(
        memory_id="memory",
        key="route",
        value="west",
    )

    with pytest.raises(
        ScaffoldExecutionError
    ) as exc_info:
        controller.write_memory(
            memory_id="memory",
            key="route",
            value="east",
        )

    assert (
        exc_info.value.code
        == "memory_write_failure"
    )

    assert len(
        runtime.technical_failures
    ) == 1

    failure = (
        runtime
        .technical_failures[
            0
        ]
    )

    assert (
        failure[
            "stage"
        ]
        == "memory_write"
    )

    assert (
        failure[
            "error"
        ]
        is not None
    )

    assert (
        failure[
            "model_invocation_id"
        ]
        is None
    )

    # The original record remains intact.
    record = store.read(
        namespace="namespace",
        memory_id="memory",
    )

    assert (
        record is not None
    )

    assert (
        record.value
        == "west"
    )

    # Only the successful first write generated MEMORY_WRITE.
    assert len(
        events
    ) == 1