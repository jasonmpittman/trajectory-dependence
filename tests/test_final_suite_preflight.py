__author__ = "Jason M. Pittman"
__date__ = "August 27, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from dataclasses import replace
from pathlib import Path

from src.logging import (
    Condition,
    TaskClassification,
)
from src.tasks import (
    PilotPlanner,
    PilotPreflightValidator,
    load_configured_task_suite,
    load_pilot_configuration,
)


CONFIG_PATH = (
    "config/final-suite-preflight.json"
)

MATRIX_PATH = (
    "config/tasks/final-task-matrix.json"
)

EXPECTED_TASK_IDS = {
    "final_memory_001",
    "final_memory_002",
    "final_memory_003",
    "final_tool_001",
    "final_tool_002",
    "final_tool_003",
    "final_history_001",
    "final_history_002",
    "final_history_003",
    "final_environment_001",
    "final_environment_002",
    "final_environment_003",
    "final_conflict_001",
    "final_conflict_002",
    "final_conflict_003",
    "final_conflict_004",
    "final_multisource_001",
    "final_multisource_002",
    "final_multisource_003",
    "final_multisource_004",
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


def test_final_suite_preflight_config_loads():
    config = load_pilot_configuration(
        CONFIG_PATH
    )

    assert (
        config.experiment_id
        == (
            "trajectory_dependence_"
            "final_suite_preflight"
        )
    )

    assert (
        config.task_suite_path
        == (
            "config/tasks/"
            "final-tasks.json"
        )
    )

    assert (
        config.repetitions
        == 1
    )


def test_final_suite_contains_twenty_tasks(
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
    ) == 20

    assert {
        task.task_id
        for task
        in suite.tasks
    } == EXPECTED_TASK_IDS


def test_final_suite_matches_frozen_matrix_ids(
    tmp_path,
):
    (
        _config,
        suite,
        _plan,
    ) = build_isolated(
        tmp_path
    )

    matrix = json.loads(
        Path(
            MATRIX_PATH
        ).read_text(
            encoding="utf-8"
        )
    )

    matrix_ids = {
        task_id
        for family
        in matrix[
            "families"
        ].values()
        for task_id
        in (
            family[
                "diagnostic"
            ]
            + family[
                "integration"
            ]
        )
    }

    assert matrix_ids == EXPECTED_TASK_IDS

    assert {
        task.task_id
        for task
        in suite.tasks
    } == matrix_ids


def test_final_suite_has_four_diagnostics_and_sixteen_integrations(
    tmp_path,
):
    (
        _config,
        suite,
        _plan,
    ) = build_isolated(
        tmp_path
    )

    diagnostics = [
        task
        for task
        in suite.tasks
        if (
            task.task_classification
            == TaskClassification.DIAGNOSTIC
        )
    ]

    integrations = [
        task
        for task
        in suite.tasks
        if (
            task.task_classification
            == TaskClassification.INTEGRATION
        )
    ]

    assert len(
        diagnostics
    ) == 4

    assert len(
        integrations
    ) == 16


def test_final_suite_plan_contains_eighty_runs(
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
    ) == 80


def test_every_final_task_has_all_four_conditions(
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


def test_final_suite_plan_is_deterministic(
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


def test_final_suite_preflight_checks_all_eighty_runs(
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
        == 80
    )


def test_final_suite_preflight_checks_expected_122_pairs(
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
        == 122
    )


def test_final_suite_preflight_is_model_free_and_non_materializing(
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
        == 80
    )

    assert (
        result.checked_intervention_pairs
        == 122
    )

    for run in plan.runs:
        assert not Path(
            run.output_directory
        ).exists()
