__author__ = "Jason M. Pittman"
__date__ = "August 23, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from dataclasses import replace

import pytest

from src.agent import (
    ContextBuilder,
    EventBackedProvenanceAuditor,
    ProvenanceAuditError,
    TaskEnvironmentPolicy,
)
from src.logging import (
    Condition,
    ContextElement,
    EventType,
    ExperimentEvent,
    SourceClass,
    TaskClassification,
)


TIMESTAMP = "2026-08-23T12:00:00+00:00"


def event(
    *,
    event_id: str,
    sequence_index: int,
    event_type: EventType,
    normalized_payload,
    experiment_id: str = "exp_001",
    run_id: str = "run_001",
    task_id: str = "task_001",
    condition: Condition = Condition.C3,
    memory_id: str | None = None,
) -> ExperimentEvent:
    return ExperimentEvent(
        experiment_id=experiment_id,
        run_id=run_id,
        task_id=task_id,
        task_classification=(
            TaskClassification.INTEGRATION
        ),
        condition=condition,
        repetition_id=1,
        seed=12345,
        event_id=event_id,
        sequence_index=sequence_index,
        timestamp=TIMESTAMP,
        event_type=event_type,
        source_component="test",
        raw_payload=normalized_payload,
        normalized_payload=normalized_payload,
        memory_id=memory_id,
    )


def p(
    *,
    origin_event_id="evt_task",
    content="Choose A or B.",
):
    return ContextElement(
        source=SourceClass.P,
        source_id="prompt:task_001",
        content=content,
        origin_event_id=origin_event_id,
        origin_source=SourceClass.P,
    )


def t(
    *,
    origin_event_id="evt_tool",
    content=None,
):
    return ContextElement(
        source=SourceClass.T,
        source_id=(
            f"tool_return:{origin_event_id}"
        ),
        content=(
            {"threshold": 7}
            if content is None
            else content
        ),
        origin_event_id=origin_event_id,
        origin_source=SourceClass.T,
    )


def h_from_tool(
    *,
    origin_event_id="evt_tool",
    content=None,
):
    return ContextElement(
        source=SourceClass.H,
        source_id=(
            f"history:{origin_event_id}"
        ),
        content=(
            {"threshold": 7}
            if content is None
            else content
        ),
        origin_event_id=origin_event_id,
        origin_source=SourceClass.T,
    )


def h_from_action(
    *,
    origin_event_id="evt_action",
    content=None,
):
    return ContextElement(
        source=SourceClass.H,
        source_id=(
            f"history:{origin_event_id}"
        ),
        content=(
            {
                "action": "select",
                "value": "A",
            }
            if content is None
            else content
        ),
        origin_event_id=origin_event_id,
        origin_source=None,
    )


def m(
    *,
    origin_event_id="evt_memory",
    memory_id="mem_001",
    content=None,
):
    return ContextElement(
        source=SourceClass.M,
        source_id=f"memory:{memory_id}",
        content=(
            {
                "key": "threshold",
                "value": 7,
            }
            if content is None
            else content
        ),
        origin_event_id=origin_event_id,
        origin_source=SourceClass.M,
        metadata={
            "memory_id": memory_id,
        },
    )


def e(
    *,
    origin_event_id="evt_environment",
    content=None,
):
    return ContextElement(
        source=SourceClass.E,
        source_id="environment:door_1.status",
        content=(
            {
                "path": [
                    "entities",
                    "door_1",
                    "status",
                ],
                "found": True,
                "value": "locked",
            }
            if content is None
            else content
        ),
        origin_event_id=origin_event_id,
        origin_source=SourceClass.E,
    )


def build_context(
    *,
    condition=Condition.C3,
    prompt_elements=None,
    tool_elements=(),
    history_elements=(),
    memory_elements=(),
    environment_elements=(),
):
    builder = ContextBuilder()

    return builder.build(
        task_id="task_001",
        condition=condition,
        prompt_elements=(
            (p(),)
            if prompt_elements is None
            else prompt_elements
        ),
        tool_elements=tool_elements,
        history_elements=history_elements,
        memory_elements=memory_elements,
        environment_elements=environment_elements,
        environment_policy=TaskEnvironmentPolicy(
            allow_environment=bool(
                environment_elements
            )
        ),
    )


def context_build_event(
    context,
    *,
    event_id="evt_context",
    sequence_index=10,
    run_id="run_001",
    task_id="task_001",
    condition=Condition.C3,
):
    return event(
        event_id=event_id,
        sequence_index=sequence_index,
        event_type=EventType.CONTEXT_BUILD,
        normalized_payload=context.to_dict(),
        run_id=run_id,
        task_id=task_id,
        condition=condition,
    )


def test_prompt_binds_to_task_event() -> None:
    context = build_context()

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        context_build_event(
            context
        ),
    ]

    result = EventBackedProvenanceAuditor().audit(
        context=context,
        events=events,
        context_build_event_id="evt_context",
    )

    assert (
        result.records[0].origin_event_type
        == EventType.TASK
    )


def test_tool_context_binds_to_tool_return() -> None:
    context = build_context(
        tool_elements=(t(),)
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_tool",
            sequence_index=4,
            event_type=EventType.TOOL_RETURN,
            normalized_payload={
                "threshold": 7,
            },
        ),
        context_build_event(
            context
        ),
    ]

    result = EventBackedProvenanceAuditor().audit(
        context=context,
        events=events,
        context_build_event_id="evt_context",
    )

    assert (
        result.records[1].source
        == SourceClass.T
    )


def test_history_preserves_tool_origin() -> None:
    context = build_context(
        history_elements=(
            h_from_tool(),
        )
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_tool",
            sequence_index=4,
            event_type=EventType.TOOL_RETURN,
            normalized_payload={
                "threshold": 7,
            },
        ),
        context_build_event(
            context
        ),
    ]

    result = EventBackedProvenanceAuditor().audit(
        context=context,
        events=events,
        context_build_event_id="evt_context",
    )

    history = result.records[1]

    assert history.source == SourceClass.H
    assert history.origin_source == SourceClass.T
    assert (
        history.origin_event_type
        == EventType.TOOL_RETURN
    )


def test_history_can_bind_to_prior_action() -> None:
    context = build_context(
        history_elements=(
            h_from_action(),
        )
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_action",
            sequence_index=5,
            event_type=EventType.ACTION,
            normalized_payload={
                "action": "select",
                "value": "A",
            },
        ),
        context_build_event(
            context
        ),
    ]

    result = EventBackedProvenanceAuditor().audit(
        context=context,
        events=events,
        context_build_event_id="evt_context",
    )

    assert (
        result.records[1].origin_event_type
        == EventType.ACTION
    )

    assert (
        result.records[1].origin_source
        is None
    )


def test_memory_binds_to_memory_read_and_memory_id() -> None:
    context = build_context(
        memory_elements=(m(),)
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_memory",
            sequence_index=6,
            event_type=EventType.MEMORY_READ,
            normalized_payload={
                "key": "threshold",
                "value": 7,
            },
            memory_id="mem_001",
        ),
        context_build_event(
            context
        ),
    ]

    result = EventBackedProvenanceAuditor().audit(
        context=context,
        events=events,
        context_build_event_id="evt_context",
    )

    assert (
        result.records[1].origin_event_type
        == EventType.MEMORY_READ
    )


def test_environment_binds_to_environment_query() -> None:
    context = build_context(
        environment_elements=(e(),)
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_environment",
            sequence_index=7,
            event_type=EventType.ENVIRONMENT_QUERY,
            normalized_payload=e().content,
        ),
        context_build_event(
            context
        ),
    ]

    result = EventBackedProvenanceAuditor().audit(
        context=context,
        events=events,
        context_build_event_id="evt_context",
    )

    assert (
        result.records[1].origin_event_type
        == EventType.ENVIRONMENT_QUERY
    )


def test_wrong_origin_event_type_is_rejected() -> None:
    context = build_context(
        tool_elements=(t(),)
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_tool",
            sequence_index=4,
            event_type=EventType.GENERATION,
            normalized_payload={
                "threshold": 7,
            },
        ),
        context_build_event(
            context
        ),
    ]

    with pytest.raises(
        ProvenanceAuditError
    ) as exc_info:
        EventBackedProvenanceAuditor().audit(
            context=context,
            events=events,
            context_build_event_id="evt_context",
        )

    assert (
        exc_info.value.code
        == "origin_event_type_mismatch"
    )


def test_payload_mismatch_is_rejected() -> None:
    context = build_context(
        tool_elements=(t(),)
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_tool",
            sequence_index=4,
            event_type=EventType.TOOL_RETURN,
            normalized_payload={
                "threshold": 999,
            },
        ),
        context_build_event(
            context
        ),
    ]

    with pytest.raises(
        ProvenanceAuditError
    ) as exc_info:
        EventBackedProvenanceAuditor().audit(
            context=context,
            events=events,
            context_build_event_id="evt_context",
        )

    assert (
        exc_info.value.code
        == "origin_payload_mismatch"
    )


def test_missing_origin_event_is_rejected() -> None:
    context = build_context(
        tool_elements=(t(),)
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        context_build_event(
            context
        ),
    ]

    with pytest.raises(
        ProvenanceAuditError
    ) as exc_info:
        EventBackedProvenanceAuditor().audit(
            context=context,
            events=events,
            context_build_event_id="evt_context",
        )

    assert (
        exc_info.value.code
        == "origin_event_not_found"
    )


def test_origin_event_must_precede_context_build() -> None:
    context = build_context(
        tool_elements=(t(),)
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_tool",
            sequence_index=11,
            event_type=EventType.TOOL_RETURN,
            normalized_payload={
                "threshold": 7,
            },
        ),
        context_build_event(
            context,
            sequence_index=10,
        ),
    ]

    with pytest.raises(
        ProvenanceAuditError
    ) as exc_info:
        EventBackedProvenanceAuditor().audit(
            context=context,
            events=events,
            context_build_event_id="evt_context",
        )

    assert (
        exc_info.value.code
        == "origin_event_not_prior"
    )


def test_cross_run_provenance_is_rejected() -> None:
    context = build_context(
        tool_elements=(t(),)
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_tool",
            sequence_index=4,
            event_type=EventType.TOOL_RETURN,
            normalized_payload={
                "threshold": 7,
            },
            run_id="run_999",
        ),
        context_build_event(
            context
        ),
    ]

    with pytest.raises(
        ProvenanceAuditError
    ) as exc_info:
        EventBackedProvenanceAuditor().audit(
            context=context,
            events=events,
            context_build_event_id="evt_context",
        )

    assert (
        exc_info.value.code
        == "cross_run_provenance"
    )


def test_cross_task_provenance_is_rejected() -> None:
    context = build_context(
        tool_elements=(t(),)
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_tool",
            sequence_index=4,
            event_type=EventType.TOOL_RETURN,
            normalized_payload={
                "threshold": 7,
            },
            task_id="task_999",
        ),
        context_build_event(
            context
        ),
    ]

    with pytest.raises(
        ProvenanceAuditError
    ) as exc_info:
        EventBackedProvenanceAuditor().audit(
            context=context,
            events=events,
            context_build_event_id="evt_context",
        )

    assert (
        exc_info.value.code
        == "cross_task_provenance"
    )


def test_context_build_payload_must_match_context() -> None:
    context = build_context()

    bad_context_event = event(
        event_id="evt_context",
        sequence_index=10,
        event_type=EventType.CONTEXT_BUILD,
        normalized_payload={
            "not": "the context"
        },
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        bad_context_event,
    ]

    with pytest.raises(
        ProvenanceAuditError
    ) as exc_info:
        EventBackedProvenanceAuditor().audit(
            context=context,
            events=events,
            context_build_event_id="evt_context",
        )

    assert (
        exc_info.value.code
        == "context_build_payload_mismatch"
    )


def test_duplicate_event_ids_are_rejected() -> None:
    context = build_context()

    duplicate_one = event(
        event_id="evt_task",
        sequence_index=1,
        event_type=EventType.TASK,
        normalized_payload="Choose A or B.",
    )

    duplicate_two = replace(
        duplicate_one,
        sequence_index=2,
    )

    events = [
        duplicate_one,
        duplicate_two,
        context_build_event(
            context
        ),
    ]

    with pytest.raises(
        ProvenanceAuditError
    ) as exc_info:
        EventBackedProvenanceAuditor().audit(
            context=context,
            events=events,
            context_build_event_id="evt_context",
        )

    assert (
        exc_info.value.code
        == "duplicate_event_id"
    )


def test_wrong_history_origin_source_is_rejected() -> None:
    bad_history = ContextElement(
        source=SourceClass.H,
        source_id="history:evt_tool",
        content={
            "threshold": 7,
        },
        origin_event_id="evt_tool",
        origin_source=SourceClass.E,
    )

    context = build_context(
        history_elements=(
            bad_history,
        )
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_tool",
            sequence_index=4,
            event_type=EventType.TOOL_RETURN,
            normalized_payload={
                "threshold": 7,
            },
        ),
        context_build_event(
            context
        ),
    ]

    with pytest.raises(
        ProvenanceAuditError
    ) as exc_info:
        EventBackedProvenanceAuditor().audit(
            context=context,
            events=events,
            context_build_event_id="evt_context",
        )

    assert (
        exc_info.value.code
        == "origin_source_mismatch"
    )


def test_memory_id_mismatch_is_rejected() -> None:
    context = build_context(
        memory_elements=(
            m(memory_id="mem_001"),
        )
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_memory",
            sequence_index=6,
            event_type=EventType.MEMORY_READ,
            normalized_payload={
                "key": "threshold",
                "value": 7,
            },
            memory_id="mem_999",
        ),
        context_build_event(
            context
        ),
    ]

    with pytest.raises(
        ProvenanceAuditError
    ) as exc_info:
        EventBackedProvenanceAuditor().audit(
            context=context,
            events=events,
            context_build_event_id="evt_context",
        )

    assert (
        exc_info.value.code
        == "memory_id_mismatch"
    )


def test_source_id_binding_is_enforced() -> None:
    bad_tool = ContextElement(
        source=SourceClass.T,
        source_id="tool_return:wrong_event",
        content={
            "threshold": 7,
        },
        origin_event_id="evt_tool",
        origin_source=SourceClass.T,
    )

    context = build_context(
        tool_elements=(
            bad_tool,
        )
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_tool",
            sequence_index=4,
            event_type=EventType.TOOL_RETURN,
            normalized_payload={
                "threshold": 7,
            },
        ),
        context_build_event(
            context
        ),
    ]

    with pytest.raises(
        ProvenanceAuditError
    ) as exc_info:
        EventBackedProvenanceAuditor().audit(
            context=context,
            events=events,
            context_build_event_id="evt_context",
        )

    assert (
        exc_info.value.code
        == "source_id_binding_mismatch"
    )


def test_complete_c3_context_audits_successfully() -> None:
    context = build_context(
        tool_elements=(t(),),
        history_elements=(
            h_from_action(),
        ),
        memory_elements=(m(),),
        environment_elements=(e(),),
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_tool",
            sequence_index=3,
            event_type=EventType.TOOL_RETURN,
            normalized_payload={
                "threshold": 7,
            },
        ),
        event(
            event_id="evt_action",
            sequence_index=4,
            event_type=EventType.ACTION,
            normalized_payload={
                "action": "select",
                "value": "A",
            },
        ),
        event(
            event_id="evt_memory",
            sequence_index=5,
            event_type=EventType.MEMORY_READ,
            normalized_payload={
                "key": "threshold",
                "value": 7,
            },
            memory_id="mem_001",
        ),
        event(
            event_id="evt_environment",
            sequence_index=6,
            event_type=EventType.ENVIRONMENT_QUERY,
            normalized_payload=e().content,
        ),
        context_build_event(
            context
        ),
    ]

    result = EventBackedProvenanceAuditor().audit(
        context=context,
        events=events,
        context_build_event_id="evt_context",
    )

    assert [
        record.source
        for record in result.records
    ] == [
        SourceClass.P,
        SourceClass.T,
        SourceClass.H,
        SourceClass.M,
        SourceClass.E,
    ]


def test_audit_result_is_deterministic() -> None:
    context = build_context(
        tool_elements=(t(),)
    )

    events = [
        event(
            event_id="evt_task",
            sequence_index=1,
            event_type=EventType.TASK,
            normalized_payload="Choose A or B.",
        ),
        event(
            event_id="evt_tool",
            sequence_index=4,
            event_type=EventType.TOOL_RETURN,
            normalized_payload={
                "threshold": 7,
            },
        ),
        context_build_event(
            context
        ),
    ]

    auditor = (
        EventBackedProvenanceAuditor()
    )

    first = auditor.audit(
        context=context,
        events=events,
        context_build_event_id="evt_context",
    )

    second = auditor.audit(
        context=context,
        events=events,
        context_build_event_id="evt_context",
    )

    assert (
        first.to_dict()
        == second.to_dict()
    )