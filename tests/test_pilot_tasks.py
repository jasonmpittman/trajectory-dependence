__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from pathlib import Path

from src.logging import (
    Condition,
    SourceClass,
    TaskClassification,
)
from src.tasks import (
    TaskFamily,
    load_task_suite,
)


PILOT_PATH = (
    Path("config")
    / "tasks"
    / "pilot-tasks.json"
)


def load_pilot():
    return load_task_suite(
        PILOT_PATH
    )


def task_by_id(
    task_id: str,
):
    suite = load_pilot()

    return next(
        task
        for task
        in suite.tasks
        if task.task_id == task_id
    )


def test_pilot_suite_loads() -> None:
    suite = load_pilot()

    assert (
        suite.suite_id
        == "trajectory_dependence_pilot"
    )


def test_pilot_contains_exactly_six_tasks() -> None:
    suite = load_pilot()

    assert len(
        suite.tasks
    ) == 6


def test_pilot_covers_all_six_families_once() -> None:
    suite = load_pilot()

    families = [
        task.family
        for task
        in suite.tasks
    ]

    assert set(
        families
    ) == set(
        TaskFamily
    )

    assert len(
        families
    ) == len(
        set(
            families
        )
    )


def test_pilot_has_four_diagnostic_and_two_integration_tasks() -> None:
    suite = load_pilot()

    diagnostic = [
        task
        for task
        in suite.tasks
        if (
            task.task_classification
            == TaskClassification.DIAGNOSTIC
        )
    ]

    integration = [
        task
        for task
        in suite.tasks
        if (
            task.task_classification
            == TaskClassification.INTEGRATION
        )
    ]

    assert len(
        diagnostic
    ) == 4

    assert len(
        integration
    ) == 2


def test_every_task_runs_all_four_conditions() -> None:
    suite = load_pilot()

    expected = (
        Condition.C0,
        Condition.C1,
        Condition.C2,
        Condition.C3,
    )

    for task in suite.tasks:
        assert (
            task.conditions
            == expected
        )


def test_c0_is_prompt_only_for_every_task() -> None:
    suite = load_pilot()

    for task in suite.tasks:
        assert (
            task.available_sources[
                Condition.C0
            ]
            == (
                SourceClass.P,
            )
        )


def test_memory_diagnostic_exposes_m_only_in_c3() -> None:
    task = task_by_id(
        "pilot_memory_001"
    )

    assert (
        SourceClass.M
        not in task.available_sources[
            Condition.C2
        ]
    )

    assert (
        task.available_sources[
            Condition.C3
        ]
        == (
            SourceClass.P,
            SourceClass.M,
        )
    )


def test_tool_diagnostic_exposes_t_from_c1() -> None:
    task = task_by_id(
        "pilot_tool_001"
    )

    for condition in (
        Condition.C1,
        Condition.C2,
        Condition.C3,
    ):
        assert (
            SourceClass.T
            in task.available_sources[
                condition
            ]
        )


def test_history_diagnostic_exposes_h_from_c2() -> None:
    task = task_by_id(
        "pilot_history_001"
    )

    assert (
        SourceClass.H
        not in task.available_sources[
            Condition.C1
        ]
    )

    assert (
        SourceClass.H
        in task.available_sources[
            Condition.C2
        ]
    )

    assert (
        SourceClass.H
        in task.available_sources[
            Condition.C3
        ]
    )


def test_environment_is_exogenous_to_condition_ladder() -> None:
    task = task_by_id(
        "pilot_environment_001"
    )

    assert (
        SourceClass.E
        not in task.available_sources[
            Condition.C0
        ]
    )

    for condition in (
        Condition.C1,
        Condition.C2,
        Condition.C3,
    ):
        assert (
            SourceClass.E
            in task.available_sources[
                condition
            ]
        )


def test_conflict_task_has_cumulative_p_t_h_m_sources() -> None:
    task = task_by_id(
        "pilot_conflict_001"
    )

    assert (
        task.available_sources[
            Condition.C0
        ]
        == (
            SourceClass.P,
        )
    )

    assert (
        task.available_sources[
            Condition.C1
        ]
        == (
            SourceClass.P,
            SourceClass.T,
        )
    )

    assert (
        task.available_sources[
            Condition.C2
        ]
        == (
            SourceClass.P,
            SourceClass.T,
            SourceClass.H,
        )
    )

    assert (
        task.available_sources[
            Condition.C3
        ]
        == (
            SourceClass.P,
            SourceClass.T,
            SourceClass.H,
            SourceClass.M,
        )
    )


def test_multisource_c3_exposes_all_experimental_sources() -> None:
    task = task_by_id(
        "pilot_multisource_001"
    )

    assert (
        task.available_sources[
            Condition.C3
        ]
        == (
            SourceClass.P,
            SourceClass.T,
            SourceClass.H,
            SourceClass.M,
            SourceClass.E,
        )
    )


def test_pilot_exercises_all_scaffold_intervention_sources() -> None:
    suite = load_pilot()

    observed = {
        pair.source
        for task
        in suite.tasks
        for pair
        in task.intervention_pairs
    }

    assert {
        SourceClass.T,
        SourceClass.H,
        SourceClass.M,
        SourceClass.E,
    }.issubset(
        observed
    )


def test_pilot_contains_prompt_intervention() -> None:
    suite = load_pilot()

    assert any(
        pair.source
        == SourceClass.P
        for task
        in suite.tasks
        for pair
        in task.intervention_pairs
    )


def test_targeted_and_sham_fixture_targets_are_distinct_for_non_p_non_e() -> None:
    suite = load_pilot()

    for task in suite.tasks:
        for pair in (
            task.intervention_pairs
        ):
            if (
                pair.source
                in {
                    SourceClass.P,
                    SourceClass.E,
                }
            ):
                continue

            assert (
                pair.targeted
                .target_fixture_id
                != pair.sham
                .target_fixture_id
            )


def test_environment_target_and_sham_paths_are_distinct() -> None:
    suite = load_pilot()

    for task in suite.tasks:
        for pair in (
            task.intervention_pairs
        ):
            if (
                pair.source
                != SourceClass.E
            ):
                continue

            assert (
                pair.targeted.target_path
                != pair.sham.target_path
            )


def test_non_prompt_pilot_interventions_request_length_matching() -> None:
    suite = load_pilot()

    for task in suite.tasks:
        for pair in (
            task.intervention_pairs
        ):
            if (
                pair.source
                == SourceClass.P
            ):
                continue

            assert (
                pair.targeted
                .length_match_required
                is True
            )

            assert (
                pair.sham
                .length_match_required
                is True
            )


def test_integration_tasks_have_multiple_intervention_sources() -> None:
    suite = load_pilot()

    for task in suite.tasks:
        if (
            task.task_classification
            != TaskClassification.INTEGRATION
        ):
            continue

        sources = {
            pair.source
            for pair
            in task.intervention_pairs
        }

        assert len(
            sources
        ) >= 2


def test_multisource_task_has_four_scaffold_intervention_sources() -> None:
    task = task_by_id(
        "pilot_multisource_001"
    )

    sources = {
        pair.source
        for pair
        in task.intervention_pairs
    }

    assert sources == {
        SourceClass.T,
        SourceClass.H,
        SourceClass.M,
        SourceClass.E,
    }


def test_multisource_expected_actions_are_deliberately_nonmonotonic() -> None:
    task = task_by_id(
        "pilot_multisource_001"
    )

    expected = (
        task.focal_decision
        .expected_by_condition
    )

    assert (
        expected[
            Condition.C0
        ][
            "value"
        ]
        == "A"
    )

    assert (
        expected[
            Condition.C1
        ][
            "value"
        ]
        == "A"
    )

    assert (
        expected[
            Condition.C2
        ][
            "value"
        ]
        == "B"
    )

    assert (
        expected[
            Condition.C3
        ][
            "value"
        ]
        == "A"
    )


def test_no_task_encodes_empirical_causal_truth() -> None:
    suite = load_pilot()

    for task in suite.tasks:
        assert (
            task.empirical_status
            == "determined_only_by_intervention"
        )


def test_all_tasks_use_same_focal_action_shape() -> None:
    suite = load_pilot()

    for task in suite.tasks:
        schema = (
            task.focal_decision
            .schema
        )

        assert (
            schema.focal_decision_id
            == "final_route"
        )

        assert [
            field.name
            for field
            in schema.fields
        ] == [
            "action",
            "value",
        ]

def test_key_value_tool_interventions_preserve_result_shape() -> None:
    suite = load_pilot()

    tool_pair_ids = {
        "tool_threshold_pair",
        "conflict_tool_pair",
        "multi_tool_pair",
    }

    pairs = [
        pair
        for task in suite.tasks
        for pair in task.intervention_pairs
        if pair.pair_id in tool_pair_ids
    ]

    assert len(
        pairs
    ) == 3

    for pair in pairs:
        for mutation in (
            pair.targeted,
            pair.sham,
        ):
            replacement = (
                mutation
                .replacement_value
            )

            assert isinstance(
                replacement,
                dict,
            )

            assert (
                replacement.get(
                    "found"
                )
                is True
            )

            assert (
                "key"
                in replacement
            )

            assert (
                "value"
                in replacement
            )

            assert set(
                replacement.keys()
            ) == {
                "found",
                "key",
                "value",
            }