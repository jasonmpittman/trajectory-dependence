__author__ = "Jason M. Pittman"
__date__ = "August 27, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from dataclasses import replace
from pathlib import Path

from src.logging import (
    Condition,
)
from src.tasks import (
    PilotPlanner,
    PilotPreflightValidator,
    load_configured_task_suite,
    load_pilot_configuration,
)


CONFIG_PATH = (
    "config/final-batch1-preflight.json"
)

EXPECTED_TASK_IDS = {
    "final_memory_001",
    "final_tool_001",
    "final_history_001",
    "final_environment_001",
    "final_memory_002",
}

EXPECTED_CONDITIONS = {
    Condition.C0,
    Condition.C1,
    Condition.C2,
    Condition.C3,
}


def build_isolated(
    tmp_path,
):
    config = load_pilot_configuration(
        CONFIG_PATH
    )

    config = replace(
        config,
        output_root=str(
            tmp_path
            / "preflight"
        ),
    )

    suite = (
        load_configured_task_suite(
            config
        )
    )

    plan = (
        PilotPlanner()
        .build(
            config=config,
            suite=suite,
        )
    )

    return (
        config,
        suite,
        plan,
    )


def test_final_batch1_preflight_config_loads():
    config = load_pilot_configuration(
        CONFIG_PATH
    )

    assert (
        config.experiment_id
        == (
            "trajectory_dependence_"
            "final_batch1_preflight"
        )
    )

    assert (
        config.task_suite_path
        == (
            "config/tasks/"
            "final-tasks-batch1.json"
        )
    )

    assert (
        config.repetitions
        == 1
    )


def test_final_batch1_suite_contains_five_tasks(
    tmp_path,
):
    (
        _config,
        suite,
        _plan,
    ) = build_isolated(
        tmp_path
    )

    assert len(
        suite.tasks
    ) == 5

    assert {
        task.task_id
        for task
        in suite.tasks
    } == EXPECTED_TASK_IDS


def test_final_batch1_plan_contains_twenty_runs(
    tmp_path,
):
    (
        _config,
        _suite,
        plan,
    ) = build_isolated(
        tmp_path
    )

    assert len(
        plan.runs
    ) == 20


def test_every_batch1_task_has_all_four_conditions(
    tmp_path,
):
    (
        _config,
        _suite,
        plan,
    ) = build_isolated(
        tmp_path
    )

    by_task = {}

    for run in plan.runs:
        by_task.setdefault(
            run.task_id,
            set(),
        ).add(
            run.condition
        )

    assert set(
        by_task
    ) == EXPECTED_TASK_IDS

    for conditions in (
        by_task.values()
    ):
        assert (
            conditions
            == EXPECTED_CONDITIONS
        )


def test_batch1_plan_is_deterministic(
    tmp_path,
):
    (
        config,
        suite,
        plan_a,
    ) = build_isolated(
        tmp_path
    )

    plan_b = (
        PilotPlanner()
        .build(
            config=config,
            suite=suite,
        )
    )

    assert (
        plan_a.plan_sha256
        == plan_b.plan_sha256
    )

    assert (
        plan_a.to_dict()
        == plan_b.to_dict()
    )


def test_batch1_preflight_checks_all_twenty_runs(
    tmp_path,
):
    (
        config,
        suite,
        plan,
    ) = build_isolated(
        tmp_path
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
        == 20
    )


def test_batch1_preflight_checks_expected_thirteen_pairs(
    tmp_path,
):
    (
        config,
        suite,
        plan,
    ) = build_isolated(
        tmp_path
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
        result.checked_intervention_pairs
        == 13
    )


def test_batch1_preflight_creates_no_planned_output_directories(
    tmp_path,
):
    (
        config,
        suite,
        plan,
    ) = build_isolated(
        tmp_path
    )

    (
        PilotPreflightValidator()
        .validate(
            config=config,
            suite=suite,
            plan=plan,
        )
    )

    for run in plan.runs:
        assert not Path(
            run.output_directory
        ).exists()


def test_batch1_preflight_does_not_require_model_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(
        "TRAJECTORY_MODEL_PATH",
        raising=False,
    )

    (
        config,
        suite,
        plan,
    ) = build_isolated(
        tmp_path
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
        == 20
    )

    assert (
        result.checked_intervention_pairs
        == 13
    )