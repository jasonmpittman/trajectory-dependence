__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from copy import deepcopy

import pytest

from src.tasks import (
    FinalTaskMatrixError,
    FinalTaskMatrixValidator,
)


MATRIX_PATH = (
    "config/tasks/"
    "final-task-matrix.json"
)


def load_matrix():
    return (
        FinalTaskMatrixValidator()
        .load(
            MATRIX_PATH
        )
    )


def test_final_matrix_is_valid():
    validator = (
        FinalTaskMatrixValidator()
    )

    result = validator.validate(
        load_matrix()
    )

    assert result.valid is True
    assert result.total_tasks == 20
    assert result.diagnostic_tasks == 4
    assert result.integration_tasks == 16


def test_all_six_families_are_present():
    result = (
        FinalTaskMatrixValidator()
        .validate(
            load_matrix()
        )
    )

    assert len(
        result.family_counts
    ) == 6


def test_every_family_has_at_least_three_tasks():
    result = (
        FinalTaskMatrixValidator()
        .validate(
            load_matrix()
        )
    )

    assert all(
        count >= 3
        for count
        in result.family_counts.values()
    )


def test_integration_tasks_are_majority():
    result = (
        FinalTaskMatrixValidator()
        .validate(
            load_matrix()
        )
    )

    assert (
        result.integration_tasks
        > result.diagnostic_tasks
    )

    assert (
        result.integration_tasks
        >= (
            result.total_tasks
            / 2
        )
    )


def test_real_model_candidate_screening_is_prohibited():
    result = (
        FinalTaskMatrixValidator()
        .validate(
            load_matrix()
        )
    )

    assert (
        result
        .real_model_candidate_screening_allowed
        is False
    )


def test_duplicate_task_id_is_rejected():
    matrix = load_matrix()

    matrix[
        "families"
    ][
        "tool_dependence"
    ][
        "integration"
    ][
        0
    ] = "final_memory_001"

    with pytest.raises(
        FinalTaskMatrixError
    ) as exc_info:
        (
            FinalTaskMatrixValidator()
            .validate(
                matrix
            )
        )

    assert (
        exc_info.value.code
        == "duplicate_task_id"
    )


def test_underrepresented_family_is_rejected():
    matrix = deepcopy(
        load_matrix()
    )

    matrix[
        "families"
    ][
        "memory_dependence"
    ][
        "integration"
    ] = []

    with pytest.raises(
        FinalTaskMatrixError
    ) as exc_info:
        (
            FinalTaskMatrixValidator()
            .validate(
                matrix
            )
        )

    assert (
        exc_info.value.code
        == "family_underrepresented"
    )


def test_model_screening_flag_cannot_be_enabled():
    matrix = deepcopy(
        load_matrix()
    )

    matrix[
        "real_model_candidate_screening_allowed"
    ] = True

    with pytest.raises(
        FinalTaskMatrixError
    ) as exc_info:
        (
            FinalTaskMatrixValidator()
            .validate(
                matrix
            )
        )

    assert (
        exc_info.value.code
        == (
            "real_model_screening_not_prohibited"
        )
    )