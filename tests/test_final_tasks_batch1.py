__author__ = "Jason M. Pittman"
__date__ = "August 27, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from src.logging import (
    SourceClass,
    TaskClassification,
)
from src.tasks import (
    FinalTaskMatrixValidator,
    load_task_suite,
)


BATCH_PATH = (
    "config/tasks/"
    "final-tasks-batch1.json"
)

MATRIX_PATH = (
    "config/tasks/"
    "final-task-matrix.json"
)


EXPECTED_IDS = {
    "final_memory_001",
    "final_tool_001",
    "final_history_001",
    "final_environment_001",
    "final_memory_002",
}


def load_batch():
    return load_task_suite(
        BATCH_PATH
    )


def test_batch1_loads():
    suite = load_batch()

    assert len(
        suite.tasks
    ) == 5


def test_batch1_contains_expected_ids():
    suite = load_batch()

    assert {
        task.task_id
        for task in suite.tasks
    } == EXPECTED_IDS


def test_batch1_ids_exist_in_frozen_matrix():
    matrix = (
        FinalTaskMatrixValidator()
        .load(
            MATRIX_PATH
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

    assert (
        EXPECTED_IDS
        <= matrix_ids
    )


def test_batch1_has_four_diagnostics_one_integration():
    suite = load_batch()

    diagnostics = [
        task
        for task in suite.tasks
        if (
            task.task_classification
            == TaskClassification.DIAGNOSTIC
        )
    ]

    integration = [
        task
        for task in suite.tasks
        if (
            task.task_classification
            == TaskClassification.INTEGRATION
        )
    ]

    assert len(
        diagnostics
    ) == 4

    assert len(
        integration
    ) == 1


def test_memory_diagnostic_exposes_m_only_in_c3():
    suite = load_batch()

    task = next(
        task
        for task in suite.tasks
        if (
            task.task_id
            == "final_memory_001"
        )
    )

    assert (
        SourceClass.M
        not in task.available_sources[
            task.conditions[0]
        ]
    )

    assert (
        SourceClass.M
        in task.available_sources[
            task.conditions[-1]
        ]
    )


def test_memory_002_is_integration():
    suite = load_batch()

    task = next(
        task
        for task in suite.tasks
        if (
            task.task_id
            == "final_memory_002"
        )
    )

    assert (
        task.task_classification
        == TaskClassification.INTEGRATION
    )


def test_memory_002_has_t_and_m_opportunities():
    suite = load_batch()

    task = next(
        task
        for task in suite.tasks
        if (
            task.task_id
            == "final_memory_002"
        )
    )

    sources = {
        pair.source
        for pair
        in task.intervention_pairs
    }

    assert (
        SourceClass.T
        in sources
    )

    assert (
        SourceClass.M
        in sources
    )


def test_batch1_empirical_status_is_intervention_defined():
    suite = load_batch()

    for task in suite.tasks:
        assert (
            task.empirical_status
            == "determined_only_by_intervention"
        )