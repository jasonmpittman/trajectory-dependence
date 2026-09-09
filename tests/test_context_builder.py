__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import math

import pytest

from src.agent import (
    ContextBuildError,
    ContextBuilder,
    TaskEnvironmentPolicy,
)
from src.logging import (
    Condition,
    ContextElement,
    SourceClass,
)


def p(
    source_id: str = "prompt:task_001",
    content=None,
) -> ContextElement:
    return ContextElement(
        source=SourceClass.P,
        source_id=source_id,
        content=(
            {"instruction": "Choose A or B."}
            if content is None
            else content
        ),
    )


def t(
    source_id: str = "tool_return:event_004",
) -> ContextElement:
    return ContextElement(
        source=SourceClass.T,
        source_id=source_id,
        content={
            "threshold": 7,
        },
        origin_event_id="event_004",
        origin_source=SourceClass.T,
        created_sequence=4,
    )


def h(
    source_id: str = "history:event_004",
) -> ContextElement:
    return ContextElement(
        source=SourceClass.H,
        source_id=source_id,
        content={
            "threshold": 7,
        },
        origin_event_id="event_004",
        origin_source=SourceClass.T,
        created_sequence=4,
        retrieved_sequence=8,
    )


def m(
    source_id: str = "memory:mem_001",
) -> ContextElement:
    return ContextElement(
        source=SourceClass.M,
        source_id=source_id,
        content={
            "key": "threshold",
            "value": 7,
        },
        metadata={
            "memory_id": "mem_001",
        },
    )


def e(
    source_id: str = "environment:door_1.status",
) -> ContextElement:
    return ContextElement(
        source=SourceClass.E,
        source_id=source_id,
        content={
            "status": "locked",
        },
    )


@pytest.mark.parametrize(
    (
        "condition",
        "tool_elements",
        "history_elements",
        "memory_elements",
        "expected_sources",
    ),
    [
        (
            Condition.C0,
            (),
            (),
            (),
            {"P"},
        ),
        (
            Condition.C1,
            (t(),),
            (),
            (),
            {"P", "T"},
        ),
        (
            Condition.C2,
            (t(),),
            (h(),),
            (),
            {"P", "T", "H"},
        ),
        (
            Condition.C3,
            (t(),),
            (h(),),
            (m(),),
            {"P", "T", "H", "M"},
        ),
    ],
)
def test_valid_cumulative_contexts(
    condition,
    tool_elements,
    history_elements,
    memory_elements,
    expected_sources,
) -> None:
    builder = ContextBuilder()

    context = builder.build(
        task_id="task_001",
        condition=condition,
        prompt_elements=(p(),),
        tool_elements=tool_elements,
        history_elements=history_elements,
        memory_elements=memory_elements,
    )

    assert {
        source.value
        for source in context.source_classes
    } == expected_sources


def test_authorized_subset_is_valid_in_c3() -> None:
    builder = ContextBuilder()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C3,
        prompt_elements=(p(),),
        history_elements=(h(),),
    )

    assert context.source_classes == frozenset(
        {
            SourceClass.P,
            SourceClass.H,
        }
    )


def test_c1_rejects_history() -> None:
    builder = ContextBuilder()

    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        builder.build(
            task_id="task_001",
            condition=Condition.C1,
            prompt_elements=(p(),),
            history_elements=(h(),),
        )

    assert (
        exc_info.value.code
        == "condition_source_violation"
    )


def test_c2_rejects_memory() -> None:
    builder = ContextBuilder()

    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        builder.build(
            task_id="task_001",
            condition=Condition.C2,
            prompt_elements=(p(),),
            memory_elements=(m(),),
        )

    assert (
        exc_info.value.code
        == "condition_source_violation"
    )


def test_c0_rejects_explicit_environment_even_if_task_allows() -> None:
    builder = ContextBuilder()

    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        builder.build(
            task_id="task_001",
            condition=Condition.C0,
            prompt_elements=(p(),),
            environment_elements=(e(),),
            environment_policy=TaskEnvironmentPolicy(
                allow_environment=True
            ),
        )

    assert (
        exc_info.value.code
        == "c0_environment_violation"
    )


def test_environment_requires_task_authorization() -> None:
    builder = ContextBuilder()

    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        builder.build(
            task_id="task_001",
            condition=Condition.C1,
            prompt_elements=(p(),),
            environment_elements=(e(),),
        )

    assert (
        exc_info.value.code
        == "environment_not_authorized"
    )


def test_environment_is_allowed_when_task_authorizes_it() -> None:
    builder = ContextBuilder()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C1,
        prompt_elements=(p(),),
        environment_elements=(e(),),
        environment_policy=TaskEnvironmentPolicy(
            allow_environment=True
        ),
    )

    assert context.source_classes == frozenset(
        {
            SourceClass.P,
            SourceClass.E,
        }
    )


def test_environment_policy_can_restrict_source_ids() -> None:
    builder = ContextBuilder()

    permitted = e(
        "environment:door_1.status"
    )

    prohibited = e(
        "environment:alarm.status"
    )

    policy = TaskEnvironmentPolicy(
        allow_environment=True,
        allowed_source_ids=frozenset(
            {
                "environment:door_1.status",
            }
        ),
    )

    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        builder.build(
            task_id="task_001",
            condition=Condition.C2,
            prompt_elements=(p(),),
            environment_elements=(
                permitted,
                prohibited,
            ),
            environment_policy=policy,
        )

    assert (
        exc_info.value.code
        == "environment_source_not_authorized"
    )


def test_environment_policy_accepts_named_source() -> None:
    builder = ContextBuilder()

    policy = TaskEnvironmentPolicy(
        allow_environment=True,
        allowed_source_ids=frozenset(
            {
                "environment:door_1.status",
            }
        ),
    )

    context = builder.build(
        task_id="task_001",
        condition=Condition.C2,
        prompt_elements=(p(),),
        environment_elements=(e(),),
        environment_policy=policy,
    )

    assert (
        context.elements_for(
            SourceClass.E
        )[0].source_id
        == "environment:door_1.status"
    )


def test_missing_prompt_is_rejected() -> None:
    builder = ContextBuilder()

    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        builder.build(
            task_id="task_001",
            condition=Condition.C0,
            prompt_elements=(),
        )

    assert (
        exc_info.value.code
        == "missing_prompt"
    )


def test_wrong_source_in_bucket_is_rejected() -> None:
    builder = ContextBuilder()

    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        builder.build(
            task_id="task_001",
            condition=Condition.C1,
            prompt_elements=(p(),),
            tool_elements=(h(),),
        )

    assert (
        exc_info.value.code
        == "source_bucket_mismatch"
    )


def test_duplicate_source_ids_are_rejected() -> None:
    builder = ContextBuilder()

    prompt = p(
        source_id="duplicate"
    )

    tool = t(
        source_id="duplicate"
    )

    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        builder.build(
            task_id="task_001",
            condition=Condition.C1,
            prompt_elements=(prompt,),
            tool_elements=(tool,),
        )

    assert (
        exc_info.value.code
        == "duplicate_source_id"
    )


def test_history_preserves_tool_origin() -> None:
    builder = ContextBuilder()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C2,
        prompt_elements=(p(),),
        history_elements=(h(),),
    )

    history = context.elements_for(
        SourceClass.H
    )[0]

    assert history.source == SourceClass.H
    assert history.origin_source == SourceClass.T
    assert history.origin_event_id == "event_004"


def test_memory_remains_memory_provenance() -> None:
    builder = ContextBuilder()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C3,
        prompt_elements=(p(),),
        memory_elements=(m(),),
    )

    memory = context.elements_for(
        SourceClass.M
    )[0]

    assert memory.source == SourceClass.M


def test_context_uses_deterministic_bucket_order() -> None:
    builder = ContextBuilder()

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

    assert [
        element.source
        for element in context.elements
    ] == [
        SourceClass.P,
        SourceClass.T,
        SourceClass.H,
        SourceClass.M,
        SourceClass.E,
    ]


def test_order_inside_bucket_is_preserved() -> None:
    builder = ContextBuilder()

    first = ContextElement(
        source=SourceClass.H,
        source_id="history:first",
        content="first",
    )

    second = ContextElement(
        source=SourceClass.H,
        source_id="history:second",
        content="second",
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

    assert [
        element.source_id
        for element in context.elements_for(
            SourceClass.H
        )
    ] == [
        "history:first",
        "history:second",
    ]


def test_builder_defensively_copies_content() -> None:
    builder = ContextBuilder()

    mutable = {
        "values": [1, 2],
    }

    prompt = p(
        content=mutable
    )

    context = builder.build(
        task_id="task_001",
        condition=Condition.C0,
        prompt_elements=(prompt,),
    )

    mutable["values"].append(3)

    assert (
        context.elements[0].content
        == {
            "values": [1, 2],
        }
    )


def test_non_json_context_content_is_rejected() -> None:
    builder = ContextBuilder()

    invalid = p(
        content={
            "value": {
                1,
                2,
                3,
            }
        }
    )

    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        builder.build(
            task_id="task_001",
            condition=Condition.C0,
            prompt_elements=(invalid,),
        )

    assert (
        exc_info.value.code
        == "non_serializable_context_element"
    )


def test_non_finite_number_is_rejected() -> None:
    builder = ContextBuilder()

    invalid = p(
        content={
            "value": math.nan,
        }
    )

    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        builder.build(
            task_id="task_001",
            condition=Condition.C0,
            prompt_elements=(invalid,),
        )

    assert (
        exc_info.value.code
        == "non_serializable_context_element"
    )


def test_structured_context_to_dict_is_deterministic() -> None:
    builder = ContextBuilder()

    context = builder.build(
        task_id="task_001",
        condition=Condition.C2,
        prompt_elements=(p(),),
        tool_elements=(t(),),
        history_elements=(h(),),
    )

    first = context.to_dict()
    second = context.to_dict()

    assert first == second

    assert first["source_classes"] == [
        "P",
        "T",
        "H",
    ]


def test_environment_policy_rejects_ids_when_disabled() -> None:
    with pytest.raises(
        ContextBuildError
    ) as exc_info:
        TaskEnvironmentPolicy(
            allow_environment=False,
            allowed_source_ids=frozenset(
                {
                    "environment:door_1.status",
                }
            ),
        )

    assert (
        exc_info.value.code
        == "inconsistent_environment_policy"
    )