__author__ = "Jason M. Pittman"
__date__ = "August 23, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import pytest

from src.agent import (
    ContextBuildError,
    ContextBuilder,
    ContextSerializer,
    TaskEnvironmentPolicy,
)
from src.logging import (
    Condition,
    ContextElement,
    SourceClass,
)


P_SENTINEL = "SENTINEL_CURRENT_PROMPT_6A17"
T_SENTINEL = "SENTINEL_TOOL_STATE_B42C"
H_SENTINEL = "SENTINEL_HISTORY_STATE_91DE"
M_SENTINEL = "SENTINEL_MEMORY_STATE_F530"
E_SENTINEL = "SENTINEL_ENVIRONMENT_STATE_7CC8"


P_HEADER = "=== CURRENT TASK ==="
T_HEADER = "=== TOOL INTERACTIONS ==="
H_HEADER = "=== CURRENT-TASK HISTORY ==="
M_HEADER = "=== PERSISTENT MEMORY ==="
E_HEADER = "=== ENVIRONMENT STATE ==="


def prompt_element(
    *,
    source_id: str = "prompt:task_001",
    content: str = P_SENTINEL,
) -> ContextElement:
    return ContextElement(
        source=SourceClass.P,
        source_id=source_id,
        content=content,
    )


def tool_element(
    *,
    source_id: str = "tool_return:event_004",
    content: str = T_SENTINEL,
) -> ContextElement:
    return ContextElement(
        source=SourceClass.T,
        source_id=source_id,
        content=content,
        origin_event_id="event_004",
        origin_source=SourceClass.T,
        created_sequence=4,
    )


def history_element(
    *,
    source_id: str = "history:event_004",
    content: str = H_SENTINEL,
    origin_event_id: str = "event_004",
) -> ContextElement:
    return ContextElement(
        source=SourceClass.H,
        source_id=source_id,
        content=content,
        origin_event_id=origin_event_id,
        origin_source=SourceClass.T,
        created_sequence=4,
        retrieved_sequence=8,
    )


def memory_element(
    *,
    source_id: str = "memory:mem_001",
    content: str = M_SENTINEL,
) -> ContextElement:
    return ContextElement(
        source=SourceClass.M,
        source_id=source_id,
        content=content,
        metadata={
            "memory_id": "mem_001",
        },
    )


def environment_element(
    *,
    source_id: str = "environment:door_1.status",
    content: str = E_SENTINEL,
) -> ContextElement:
    return ContextElement(
        source=SourceClass.E,
        source_id=source_id,
        content=content,
    )


def build_and_serialize(
    *,
    condition: Condition,
    tool_elements=(),
    history_elements=(),
    memory_elements=(),
    environment_elements=(),
    environment_policy=None,
):
    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=condition,
        prompt_elements=(
            prompt_element(),
        ),
        tool_elements=tool_elements,
        history_elements=history_elements,
        memory_elements=memory_elements,
        environment_elements=environment_elements,
        environment_policy=environment_policy,
    )

    return serializer.serialize(
        context
    )


@pytest.mark.parametrize(
    (
        "condition",
        "tool_elements",
        "history_elements",
        "memory_elements",
        "expected_sentinels",
        "prohibited_sentinels",
        "expected_headers",
        "prohibited_headers",
    ),
    [
        (
            Condition.C0,
            (),
            (),
            (),
            {
                P_SENTINEL,
            },
            {
                T_SENTINEL,
                H_SENTINEL,
                M_SENTINEL,
                E_SENTINEL,
            },
            {
                P_HEADER,
            },
            {
                T_HEADER,
                H_HEADER,
                M_HEADER,
                E_HEADER,
            },
        ),
        (
            Condition.C1,
            (
                tool_element(),
            ),
            (),
            (),
            {
                P_SENTINEL,
                T_SENTINEL,
            },
            {
                H_SENTINEL,
                M_SENTINEL,
                E_SENTINEL,
            },
            {
                P_HEADER,
                T_HEADER,
            },
            {
                H_HEADER,
                M_HEADER,
                E_HEADER,
            },
        ),
        (
            Condition.C2,
            (
                tool_element(),
            ),
            (
                history_element(),
            ),
            (),
            {
                P_SENTINEL,
                T_SENTINEL,
                H_SENTINEL,
            },
            {
                M_SENTINEL,
                E_SENTINEL,
            },
            {
                P_HEADER,
                T_HEADER,
                H_HEADER,
            },
            {
                M_HEADER,
                E_HEADER,
            },
        ),
        (
            Condition.C3,
            (
                tool_element(),
            ),
            (
                history_element(),
            ),
            (
                memory_element(),
            ),
            {
                P_SENTINEL,
                T_SENTINEL,
                H_SENTINEL,
                M_SENTINEL,
            },
            {
                E_SENTINEL,
            },
            {
                P_HEADER,
                T_HEADER,
                H_HEADER,
                M_HEADER,
            },
            {
                E_HEADER,
            },
        ),
    ],
)
def test_model_visible_context_contains_only_authorized_cumulative_state(
    condition,
    tool_elements,
    history_elements,
    memory_elements,
    expected_sentinels,
    prohibited_sentinels,
    expected_headers,
    prohibited_headers,
) -> None:
    serialized = build_and_serialize(
        condition=condition,
        tool_elements=tool_elements,
        history_elements=history_elements,
        memory_elements=memory_elements,
    )

    text = serialized.model_text

    for sentinel in expected_sentinels:
        assert sentinel in text

    for sentinel in prohibited_sentinels:
        assert sentinel not in text

    for header in expected_headers:
        assert header in text

    for header in prohibited_headers:
        assert header not in text


@pytest.mark.parametrize(
    (
        "condition",
        "extra_kwargs",
    ),
    [
        (
            Condition.C0,
            {
                "tool_elements": (
                    tool_element(),
                ),
            },
        ),
        (
            Condition.C0,
            {
                "history_elements": (
                    history_element(),
                ),
            },
        ),
        (
            Condition.C0,
            {
                "memory_elements": (
                    memory_element(),
                ),
            },
        ),
        (
            Condition.C1,
            {
                "history_elements": (
                    history_element(),
                ),
            },
        ),
        (
            Condition.C1,
            {
                "memory_elements": (
                    memory_element(),
                ),
            },
        ),
        (
            Condition.C2,
            {
                "memory_elements": (
                    memory_element(),
                ),
            },
        ),
    ],
)
def test_prohibited_cumulative_state_fails_closed(
    condition,
    extra_kwargs,
) -> None:
    builder = ContextBuilder()

    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        builder.build(
            task_id="task_001",
            condition=condition,
            prompt_elements=(
                prompt_element(),
            ),
            **extra_kwargs,
        )

    assert (
        exc_info.value.code
        == "condition_source_violation"
    )


@pytest.mark.parametrize(
    "condition",
    [
        Condition.C1,
        Condition.C2,
        Condition.C3,
    ],
)
def test_authorized_environment_reaches_model_visible_context(
    condition,
) -> None:
    serialized = build_and_serialize(
        condition=condition,
        environment_elements=(
            environment_element(),
        ),
        environment_policy=TaskEnvironmentPolicy(
            allow_environment=True
        ),
    )

    assert (
        E_SENTINEL
        in serialized.model_text
    )

    assert (
        E_HEADER
        in serialized.model_text
    )


def test_c0_environment_fails_even_when_task_policy_authorizes_it() -> None:
    builder = ContextBuilder()

    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        builder.build(
            task_id="task_001",
            condition=Condition.C0,
            prompt_elements=(
                prompt_element(),
            ),
            environment_elements=(
                environment_element(),
            ),
            environment_policy=TaskEnvironmentPolicy(
                allow_environment=True
            ),
        )

    assert (
        exc_info.value.code
        == "c0_environment_violation"
    )


def test_environment_authorization_does_not_create_environment_context() -> None:
    serialized = build_and_serialize(
        condition=Condition.C3,
        environment_policy=TaskEnvironmentPolicy(
            allow_environment=True
        ),
    )

    assert (
        E_SENTINEL
        not in serialized.model_text
    )

    assert (
        E_HEADER
        not in serialized.model_text
    )


def test_c3_does_not_force_permitted_scaffold_sources_into_context() -> None:
    serialized = build_and_serialize(
        condition=Condition.C3,
    )

    assert (
        serialized.model_text
        == (
            "=== CURRENT TASK ===\n"
            f"{P_SENTINEL}"
        )
    )

    assert T_HEADER not in serialized.model_text
    assert H_HEADER not in serialized.model_text
    assert M_HEADER not in serialized.model_text


def test_internal_provenance_identifiers_do_not_leak_to_model_text() -> None:
    internal_source_id = (
        "INTERNAL_SOURCE_ID_SENTINEL_418A"
    )

    internal_event_id = (
        "INTERNAL_EVENT_ID_SENTINEL_C109"
    )

    builder = ContextBuilder()
    serializer = ContextSerializer()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C2,
        prompt_elements=(
            prompt_element(),
        ),
        history_elements=(
            history_element(
                source_id=internal_source_id,
                origin_event_id=internal_event_id,
            ),
        ),
    )

    serialized = serializer.serialize(
        context
    )

    assert (
        internal_source_id
        not in serialized.model_text
    )

    assert (
        internal_event_id
        not in serialized.model_text
    )

    history_span = serialized.spans[1]

    assert (
        history_span.source_id
        == internal_source_id
    )

    assert (
        history_span.origin_event_id
        == internal_event_id
    )


def test_same_content_in_different_source_roles_changes_serialization() -> None:
    shared_content = (
        "IDENTICAL_INFORMATION_CONTENT"
    )

    tool_serialized = build_and_serialize(
        condition=Condition.C1,
        tool_elements=(
            tool_element(
                content=shared_content
            ),
        ),
    )

    history_serialized = build_and_serialize(
        condition=Condition.C2,
        history_elements=(
            history_element(
                content=shared_content
            ),
        ),
    )

    assert (
        shared_content
        in tool_serialized.model_text
    )

    assert (
        shared_content
        in history_serialized.model_text
    )

    assert (
        T_HEADER
        in tool_serialized.model_text
    )

    assert (
        H_HEADER
        in history_serialized.model_text
    )

    assert (
        tool_serialized.model_text_sha256
        != history_serialized.model_text_sha256
    )


def test_independently_built_equivalent_contexts_are_byte_identical() -> None:
    environment_policy = (
        TaskEnvironmentPolicy(
            allow_environment=True
        )
    )

    first = build_and_serialize(
        condition=Condition.C3,
        tool_elements=(
            tool_element(),
        ),
        history_elements=(
            history_element(),
        ),
        memory_elements=(
            memory_element(),
        ),
        environment_elements=(
            environment_element(),
        ),
        environment_policy=environment_policy,
    )

    second = build_and_serialize(
        condition=Condition.C3,
        tool_elements=(
            tool_element(),
        ),
        history_elements=(
            history_element(),
        ),
        memory_elements=(
            memory_element(),
        ),
        environment_elements=(
            environment_element(),
        ),
        environment_policy=environment_policy,
    )

    assert (
        first.model_text_bytes
        == second.model_text_bytes
    )

    assert (
        first.model_text_sha256
        == second.model_text_sha256
    )


def test_complete_context_has_exact_source_section_order() -> None:
    serialized = build_and_serialize(
        condition=Condition.C3,
        tool_elements=(
            tool_element(),
        ),
        history_elements=(
            history_element(),
        ),
        memory_elements=(
            memory_element(),
        ),
        environment_elements=(
            environment_element(),
        ),
        environment_policy=TaskEnvironmentPolicy(
            allow_environment=True
        ),
    )

    text = serialized.model_text

    positions = [
        text.index(P_HEADER),
        text.index(T_HEADER),
        text.index(H_HEADER),
        text.index(M_HEADER),
        text.index(E_HEADER),
    ]

    assert positions == sorted(
        positions
    )