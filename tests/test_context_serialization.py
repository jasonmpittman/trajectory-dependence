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

from src.agent import (
    ContextBuilder,
    ContextSerializationError,
    ContextSerializer,
    StructuredContext,
    TaskEnvironmentPolicy,
)
from src.logging import (
    Condition,
    ContextElement,
    SourceClass,
)


def p(
    content="Choose A or B.",
    source_id="prompt:task_001",
):
    return ContextElement(
        source=SourceClass.P,
        source_id=source_id,
        content=content,
    )


def t(
    content=None,
    source_id="tool_return:event_004",
):
    return ContextElement(
        source=SourceClass.T,
        source_id=source_id,
        content=(
            {"threshold": 7}
            if content is None
            else content
        ),
        origin_event_id="event_004",
        origin_source=SourceClass.T,
    )


def h(
    content="Earlier observation.",
    source_id="history:event_004",
):
    return ContextElement(
        source=SourceClass.H,
        source_id=source_id,
        content=content,
        origin_event_id="event_004",
        origin_source=SourceClass.T,
    )


def m(
    content=None,
    source_id="memory:mem_001",
):
    return ContextElement(
        source=SourceClass.M,
        source_id=source_id,
        content=(
            {"threshold": 7}
            if content is None
            else content
        ),
    )


def e(
    content=None,
    source_id="environment:door_1.status",
):
    return ContextElement(
        source=SourceClass.E,
        source_id=source_id,
        content=(
            {"status": "locked"}
            if content is None
            else content
        ),
    )


def test_c0_exact_serialization() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C0,
        prompt_elements=(
            p("Choose A or B."),
        ),
    )

    serialized = serializer.serialize(
        context
    )

    assert serialized.model_text == (
        "=== CURRENT TASK ===\n"
        "Choose A or B."
    )


def test_c3_serializes_sections_in_fixed_order() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C3,
        prompt_elements=(p(),),
        tool_elements=(t(),),
        history_elements=(h(),),
        memory_elements=(m(),),
        environment_elements=(e(),),
        environment_policy=TaskEnvironmentPolicy(
            allow_environment=True
        ),
    )

    text = serializer.serialize(
        context
    ).model_text

    headers = [
        "=== CURRENT TASK ===",
        "=== TOOL INTERACTIONS ===",
        "=== CURRENT-TASK HISTORY ===",
        "=== PERSISTENT MEMORY ===",
        "=== ENVIRONMENT STATE ===",
    ]

    positions = [
        text.index(header)
        for header in headers
    ]

    assert positions == sorted(
        positions
    )


def test_same_context_produces_byte_identical_serialization() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C2,
        prompt_elements=(p(),),
        tool_elements=(t(),),
        history_elements=(h(),),
    )

    first = serializer.serialize(
        context
    )

    second = serializer.serialize(
        context
    )

    assert (
        first.model_text_bytes
        == second.model_text_bytes
    )

    assert (
        first.model_text_sha256
        == second.model_text_sha256
    )


def test_source_ids_are_not_added_to_model_text() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    distinctive_source_id = (
        "INTERNAL_SOURCE_ID_DO_NOT_EXPOSE"
    )

    context = builder.build(
        task_id="task_001",
        condition=Condition.C0,
        prompt_elements=(
            p(
                source_id=distinctive_source_id
            ),
        ),
    )

    serialized = serializer.serialize(
        context
    )

    assert (
        distinctive_source_id
        not in serialized.model_text
    )

    assert (
        serialized.spans[0].source_id
        == distinctive_source_id
    )


def test_origin_event_id_is_not_added_to_model_text() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    distinctive_event_id = (
        "EVENT_ID_DO_NOT_EXPOSE"
    )

    history = ContextElement(
        source=SourceClass.H,
        source_id="history:test",
        content="Prior observation.",
        origin_event_id=distinctive_event_id,
        origin_source=SourceClass.T,
    )

    context = builder.build(
        task_id="task_001",
        condition=Condition.C2,
        prompt_elements=(p(),),
        history_elements=(history,),
    )

    serialized = serializer.serialize(
        context
    )

    assert (
        distinctive_event_id
        not in serialized.model_text
    )

    assert (
        serialized.spans[1].origin_event_id
        == distinctive_event_id
    )


def test_internal_source_codes_are_not_used_as_headers() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C3,
        prompt_elements=(p(),),
        tool_elements=(t(),),
        history_elements=(h(),),
        memory_elements=(m(),),
    )

    text = serializer.serialize(
        context
    ).model_text

    assert "=== P ===" not in text
    assert "=== T ===" not in text
    assert "=== H ===" not in text
    assert "=== M ===" not in text
    assert "=== E ===" not in text


def test_span_maps_exact_serialized_content() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C0,
        prompt_elements=(
            p("Exact focal content."),
        ),
    )

    serialized = serializer.serialize(
        context
    )

    span = serialized.spans[0]

    extracted = serialized.model_text[
        span.start_char:
        span.end_char
    ]

    assert extracted == "Exact focal content."


def test_span_preserves_origin_source() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C2,
        prompt_elements=(p(),),
        history_elements=(h(),),
    )

    serialized = serializer.serialize(
        context
    )

    history_span = serialized.spans[1]

    assert (
        history_span.source
        == SourceClass.H
    )

    assert (
        history_span.origin_source
        == SourceClass.T
    )


def test_strings_are_preserved_exactly() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    content = (
        "Line one.\n"
        "  Indented line two.\n\n"
        "Line four."
    )

    context = builder.build(
        task_id="task_001",
        condition=Condition.C0,
        prompt_elements=(
            p(content),
        ),
    )

    serialized = serializer.serialize(
        context
    )

    span = serialized.spans[0]

    assert (
        serialized.model_text[
            span.start_char:
            span.end_char
        ]
        == content
    )


def test_json_objects_use_deterministic_key_order() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    first_content = {
        "z": 3,
        "a": 1,
        "m": 2,
    }

    second_content = {
        "m": 2,
        "z": 3,
        "a": 1,
    }

    first_context = builder.build(
        task_id="task_001",
        condition=Condition.C0,
        prompt_elements=(
            p(first_content),
        ),
    )

    second_context = builder.build(
        task_id="task_001",
        condition=Condition.C0,
        prompt_elements=(
            p(second_content),
        ),
    )

    first = serializer.serialize(
        first_context
    )

    second = serializer.serialize(
        second_context
    )

    assert (
        first.model_text
        == second.model_text
    )

    assert (
        first.model_text_sha256
        == second.model_text_sha256
    )


def test_content_change_changes_serialization_checksum() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    first_context = builder.build(
        task_id="task_001",
        condition=Condition.C0,
        prompt_elements=(
            p("Choose A."),
        ),
    )

    second_context = builder.build(
        task_id="task_001",
        condition=Condition.C0,
        prompt_elements=(
            p("Choose B."),
        ),
    )

    first = serializer.serialize(
        first_context
    )

    second = serializer.serialize(
        second_context
    )

    assert (
        first.model_text_sha256
        != second.model_text_sha256
    )


def test_multiple_elements_use_fixed_separator() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    first = ContextElement(
        source=SourceClass.H,
        source_id="history:first",
        content="First event.",
    )

    second = ContextElement(
        source=SourceClass.H,
        source_id="history:second",
        content="Second event.",
    )

    context = builder.build(
        task_id="task_001",
        condition=Condition.C2,
        prompt_elements=(p(),),
        history_elements=(
            first,
            second,
        ),
    )

    text = serializer.serialize(
        context
    ).model_text

    assert (
        "First event.\n\n---\n\nSecond event."
        in text
    )


def test_span_order_matches_context_element_order() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C3,
        prompt_elements=(p(),),
        tool_elements=(t(),),
        history_elements=(h(),),
        memory_elements=(m(),),
    )

    serialized = serializer.serialize(
        context
    )

    assert [
        span.source
        for span in serialized.spans
    ] == [
        SourceClass.P,
        SourceClass.T,
        SourceClass.H,
        SourceClass.M,
    ]


def test_audit_metadata_does_not_contain_model_text() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C0,
        prompt_elements=(p(),),
    )

    serialized = serializer.serialize(
        context
    )

    audit = serialized.to_audit_dict()

    assert "model_text" not in audit

    assert (
        audit["model_text_sha256"]
        == serialized.model_text_sha256
    )


def test_environment_serializes_only_when_authorized() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C1,
        prompt_elements=(p(),),
        environment_elements=(e(),),
        environment_policy=TaskEnvironmentPolicy(
            allow_environment=True
        ),
    )

    text = serializer.serialize(
        context
    ).model_text

    assert (
        "=== ENVIRONMENT STATE ==="
        in text
    )


def test_manually_constructed_c1_history_is_rejected() -> None:
    serializer = ContextSerializer()

    forged = StructuredContext(
        task_id="task_001",
        condition=Condition.C1,
        elements=(
            p(),
            h(),
        ),
        environment_policy=(
            TaskEnvironmentPolicy(
                allow_environment=False
            )
        ),
    )

    with pytest.raises(
        ContextSerializationError
    ) as exc_info:
        serializer.serialize(
            forged
        )

    assert (
        exc_info.value.code
        == "condition_source_violation"
    )


def test_manually_constructed_c0_environment_is_rejected() -> None:
    serializer = ContextSerializer()

    forged = StructuredContext(
        task_id="task_001",
        condition=Condition.C0,
        elements=(
            p(),
            e(),
        ),
        environment_policy=(
            TaskEnvironmentPolicy(
                allow_environment=True
            )
        ),
    )

    with pytest.raises(
        ContextSerializationError
    ) as exc_info:
        serializer.serialize(
            forged
        )

    assert (
        exc_info.value.code
        == "c0_environment_violation"
    )


def test_manually_constructed_unauthorized_environment_is_rejected() -> None:
    serializer = ContextSerializer()

    forged = StructuredContext(
        task_id="task_001",
        condition=Condition.C1,
        elements=(
            p(),
            e(),
        ),
        environment_policy=(
            TaskEnvironmentPolicy(
                allow_environment=False
            )
        ),
    )

    with pytest.raises(
        ContextSerializationError
    ) as exc_info:
        serializer.serialize(
            forged
        )

    assert (
        exc_info.value.code
        == "environment_not_authorized"
    )


def test_manually_constructed_wrong_source_order_is_rejected() -> None:
    serializer = ContextSerializer()

    forged = StructuredContext(
        task_id="task_001",
        condition=Condition.C2,
        elements=(
            p(),
            h(),
            t(),
        ),
        environment_policy=(
            TaskEnvironmentPolicy(
                allow_environment=False
            )
        ),
    )

    with pytest.raises(
        ContextSerializationError
    ) as exc_info:
        serializer.serialize(
            forged
        )

    assert (
        exc_info.value.code
        == "non_deterministic_source_order"
    )


def test_empty_structured_context_is_rejected() -> None:
    serializer = ContextSerializer()

    forged = StructuredContext(
        task_id="task_001",
        condition=Condition.C0,
        elements=(),
        environment_policy=(
            TaskEnvironmentPolicy(
                allow_environment=False
            )
        ),
    )

    with pytest.raises(
        ContextSerializationError
    ) as exc_info:
        serializer.serialize(
            forged
        )

    assert (
        exc_info.value.code
        == "empty_context"
    )


def test_missing_prompt_in_manually_constructed_context_is_rejected() -> None:
    serializer = ContextSerializer()

    forged = StructuredContext(
        task_id="task_001",
        condition=Condition.C1,
        elements=(
            t(),
        ),
        environment_policy=(
            TaskEnvironmentPolicy(
                allow_environment=False
            )
        ),
    )

    with pytest.raises(
        ContextSerializationError
    ) as exc_info:
        serializer.serialize(
            forged
        )

    assert (
        exc_info.value.code
        == "missing_prompt"
    )


def test_serialization_version_is_recorded() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C0,
        prompt_elements=(p(),),
    )

    serialized = serializer.serialize(
        context
    )

    assert (
        serialized.serialization_version
        == "context-v1"
    )


def test_model_byte_length_is_auditable() -> None:
    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C0,
        prompt_elements=(
            p("Unicode: α β γ"),
        ),
    )

    serialized = serializer.serialize(
        context
    )

    audit = serialized.to_audit_dict()

    assert (
        audit["model_text_length_bytes"]
        == len(
            serialized.model_text.encode(
                "utf-8"
            )
        )
    )