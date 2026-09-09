__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from dataclasses import FrozenInstanceError

import pytest

from src.agent import (
    CONDITION_POLICIES,
    ConditionConfigurationError,
    get_condition_policy,
    validate_condition_policy_table,
    validate_condition_source_set,
)
from src.logging import (
    Condition,
    SourceClass,
)


@pytest.mark.parametrize(
    ("condition", "expected_sources"),
    [
        (
            Condition.C0,
            {
                SourceClass.P,
            },
        ),
        (
            Condition.C1,
            {
                SourceClass.P,
                SourceClass.T,
            },
        ),
        (
            Condition.C2,
            {
                SourceClass.P,
                SourceClass.T,
                SourceClass.H,
            },
        ),
        (
            Condition.C3,
            {
                SourceClass.P,
                SourceClass.T,
                SourceClass.H,
                SourceClass.M,
            },
        ),
    ],
)
def test_condition_cumulative_sources(
    condition,
    expected_sources,
) -> None:
    policy = get_condition_policy(
        condition
    )

    assert (
        policy.cumulative_sources
        == frozenset(expected_sources)
    )


def test_complete_condition_policy_table_is_valid() -> None:
    validate_condition_policy_table()


def test_condition_policies_are_cumulative() -> None:
    ordered = [
        get_condition_policy(
            condition
        ).cumulative_sources
        for condition in (
            Condition.C0,
            Condition.C1,
            Condition.C2,
            Condition.C3,
        )
    ]

    assert ordered[0].issubset(
        ordered[1]
    )

    assert ordered[1].issubset(
        ordered[2]
    )

    assert ordered[2].issubset(
        ordered[3]
    )


def test_c0_has_no_scaffold_sources() -> None:
    policy = get_condition_policy(
        Condition.C0
    )

    assert (
        policy.scaffold_sources
        == frozenset()
    )


def test_c3_has_three_cumulative_scaffold_sources() -> None:
    policy = get_condition_policy(
        Condition.C3
    )

    assert policy.scaffold_sources == frozenset(
        {
            SourceClass.T,
            SourceClass.H,
            SourceClass.M,
        }
    )


def test_environment_is_not_a_cumulative_condition_source() -> None:
    policy = get_condition_policy(
        Condition.C3
    )

    with pytest.raises(
        ConditionConfigurationError
    ) as exc_info:
        policy.permits_cumulative_source(
            SourceClass.E
        )

    assert (
        exc_info.value.code
        == "environment_not_cumulative"
    )


def test_source_validation_accepts_authorized_subset() -> None:
    result = validate_condition_source_set(
        condition=Condition.C3,
        sources=[
            SourceClass.P,
            SourceClass.H,
        ],
    )

    assert result == frozenset(
        {
            SourceClass.P,
            SourceClass.H,
        }
    )


def test_source_validation_rejects_disallowed_source() -> None:
    with pytest.raises(
        ConditionConfigurationError
    ) as exc_info:
        validate_condition_source_set(
            condition=Condition.C1,
            sources=[
                SourceClass.P,
                SourceClass.H,
            ],
        )

    assert (
        exc_info.value.code
        == "condition_source_violation"
    )


def test_source_validation_requires_separate_e_policy() -> None:
    with pytest.raises(
        ConditionConfigurationError
    ) as exc_info:
        validate_condition_source_set(
            condition=Condition.C3,
            sources=[
                SourceClass.P,
                SourceClass.E,
            ],
        )

    assert (
        exc_info.value.code
        == "environment_requires_task_policy"
    )


def test_string_condition_identifier_is_supported() -> None:
    policy = get_condition_policy(
        "C2"
    )

    assert policy.condition == Condition.C2
    assert policy.allow_history is True
    assert policy.allow_persistent_memory is False


def test_unknown_condition_is_rejected() -> None:
    with pytest.raises(
        ConditionConfigurationError
    ) as exc_info:
        get_condition_policy(
            "C99"
        )

    assert (
        exc_info.value.code
        == "unknown_condition"
    )


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(
        ConditionConfigurationError
    ) as exc_info:
        validate_condition_source_set(
            condition=Condition.C3,
            sources=[
                "P",
                "X",
            ],
        )

    assert (
        exc_info.value.code
        == "unknown_source"
    )


def test_condition_policy_is_immutable() -> None:
    policy = get_condition_policy(
        Condition.C1
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        policy.allow_history = True


def test_condition_policy_mapping_is_immutable() -> None:
    with pytest.raises(
        TypeError
    ):
        CONDITION_POLICIES[
            Condition.C0
        ] = get_condition_policy(
            Condition.C3
        )


def test_manifest_representation_is_deterministic() -> None:
    policy = get_condition_policy(
        Condition.C2
    )

    assert policy.to_dict() == {
        "condition": "C2",
        "label": "trajectory-enabled",
        "allow_prompt": True,
        "allow_tools": True,
        "allow_history": True,
        "allow_persistent_memory": False,
        "environment_policy": "task_controlled",
        "cumulative_sources": [
            "P",
            "T",
            "H",
        ],
    }