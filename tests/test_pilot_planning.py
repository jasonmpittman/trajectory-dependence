__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from dataclasses import replace
from pathlib import Path

import pytest

from src.logging import (
    Condition,
)
from src.tasks import (
    PilotPlanner,
    PilotPreflightError,
    PilotPreflightValidator,
    load_configured_task_suite,
    load_pilot_configuration,
)


CONFIG_PATH = (
    Path("config")
    / "pilot-run.json"
)


def load_config():
    return load_pilot_configuration(
        CONFIG_PATH
    )


def load_suite(
    config,
):
    return load_configured_task_suite(
        config
    )


def build_plan(
    config,
    suite,
):
    return PilotPlanner().build(
        config=config,
        suite=suite,
    )


def test_pilot_configuration_loads() -> None:
    config = load_config()

    assert (
        config.experiment_id
        == "trajectory_dependence_pilot"
    )

    assert (
        config.repetitions
        == 1
    )


def test_pilot_plan_contains_24_runs() -> None:
    config = load_config()

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    assert len(
        plan.runs
    ) == 24


def test_every_task_has_four_conditions() -> None:
    config = load_config()

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    for task in suite.tasks:
        runs = [
            run
            for run
            in plan.runs
            if (
                run.task_id
                == task.task_id
            )
        ]

        assert [
            run.condition
            for run
            in runs
        ] == [
            Condition.C0,
            Condition.C1,
            Condition.C2,
            Condition.C3,
        ]


def test_run_ids_are_unique() -> None:
    config = load_config()

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    values = [
        run.run_id
        for run
        in plan.runs
    ]

    assert len(
        values
    ) == len(
        set(
            values
        )
    )


def test_output_directories_are_unique() -> None:
    config = load_config()

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    values = [
        run.output_directory
        for run
        in plan.runs
    ]

    assert len(
        values
    ) == len(
        set(
            values
        )
    )


def test_action_seed_is_matched_across_conditions() -> None:
    config = load_config()

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    for task in suite.tasks:
        seeds = {
            run.action_seed
            for run
            in plan.runs
            if (
                run.task_id
                == task.task_id
            )
        }

        assert len(
            seeds
        ) == 1


def test_explanation_seed_is_matched_across_conditions() -> None:
    config = load_config()

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    for task in suite.tasks:
        seeds = {
            run.explanation_seed
            for run
            in plan.runs
            if (
                run.task_id
                == task.task_id
            )
        }

        assert len(
            seeds
        ) == 1


def test_action_and_explanation_seed_roles_are_distinct() -> None:
    config = load_config()

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    for run in plan.runs:
        assert (
            run.action_seed
            != run.explanation_seed
        )


def test_plan_is_deterministic() -> None:
    config = load_config()

    suite = load_suite(
        config
    )

    planner = PilotPlanner()

    first = planner.build(
        config=config,
        suite=suite,
    )

    second = planner.build(
        config=config,
        suite=suite,
    )

    assert (
        first.to_dict()
        == second.to_dict()
    )


def test_plan_checksum_is_stable() -> None:
    config = load_config()

    suite = load_suite(
        config
    )

    first = build_plan(
        config,
        suite,
    )

    second = build_plan(
        config,
        suite,
    )

    assert (
        first.plan_sha256
        == second.plan_sha256
    )


def test_preflight_checks_all_24_runs(
    tmp_path,
) -> None:
    config = replace(
        load_config(),
        output_root=str(
            tmp_path
            / "pilot"
        ),
    )

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    result = (
        PilotPreflightValidator()
        .validate(
            config=config,
            suite=suite,
            plan=plan,
        )
    )

    assert (
        result.checked_runs
        == 24
    )

    assert len(
        result.run_results
    ) == 24


def test_preflight_checks_expected_25_intervention_pairs(
    tmp_path,
) -> None:
    config = replace(
        load_config(),
        output_root=str(
            tmp_path
            / "pilot"
        ),
    )

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    result = (
        PilotPreflightValidator()
        .validate(
            config=config,
            suite=suite,
            plan=plan,
        )
    )

    assert (
        result
        .checked_intervention_pairs
        == 25
    )


def test_preflight_does_not_create_planned_run_directories(
    tmp_path,
) -> None:
    output_root = (
        tmp_path
        / "pilot"
    )

    config = replace(
        load_config(),
        output_root=str(
            output_root
        ),
    )

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    (
        PilotPreflightValidator()
        .validate(
            config=config,
            suite=suite,
            plan=plan,
        )
    )

    assert (
        output_root.exists()
        is False
    )


def test_existing_planned_run_directory_fails_preflight(
    tmp_path,
) -> None:
    config = replace(
        load_config(),
        output_root=str(
            tmp_path
            / "pilot"
        ),
    )

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    first = Path(
        plan.runs[0]
        .output_directory
    )

    first.mkdir(
        parents=True,
    )

    with pytest.raises(
        PilotPreflightError
    ) as exc_info:
        (
            PilotPreflightValidator()
            .validate(
                config=config,
                suite=suite,
                plan=plan,
            )
        )

    assert (
        exc_info.value.code
        == "planned_output_exists"
    )


def test_plan_action_configs_use_planned_seeds() -> None:
    config = load_config()

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    for run in plan.runs:
        inference = (
            run.action_config(
                max_tokens=(
                    config
                    .action_max_tokens
                )
            )
        )

        assert (
            inference.seed
            == run.action_seed
        )

        assert (
            inference.max_tokens
            == config.action_max_tokens
        )

def test_preflight_allows_memory_backing_without_m_in_c0(
    tmp_path,
) -> None:
    config = replace(
        load_config(),
        output_root=str(
            tmp_path
            / "pilot"
        ),
    )

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    result = (
        PilotPreflightValidator()
        .validate(
            config=config,
            suite=suite,
            plan=plan,
        )
    )

    memory_c0 = next(
        item
        for item
        in result.run_results
        if (
            item.task_id
            == "pilot_memory_001"
            and item.condition
            == Condition.C0
        )
    )

    assert (
        memory_c0.task_id
        == "pilot_memory_001"
    )

    assert (
        memory_c0.condition
        == Condition.C0
    )


def test_preflight_allows_environment_backing_without_e_in_c0(
    tmp_path,
) -> None:
    config = replace(
        load_config(),
        output_root=str(
            tmp_path
            / "pilot"
        ),
    )

    suite = load_suite(
        config
    )

    plan = build_plan(
        config,
        suite,
    )

    result = (
        PilotPreflightValidator()
        .validate(
            config=config,
            suite=suite,
            plan=plan,
        )
    )

    environment_c0 = next(
        item
        for item
        in result.run_results
        if (
            item.task_id
            == "pilot_environment_001"
            and item.condition
            == Condition.C0
        )
    )

    assert (
        environment_c0.task_id
        == "pilot_environment_001"
    )

    assert (
        environment_c0.condition
        == Condition.C0
    )