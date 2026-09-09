__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import pytest

from src.analysis import (
    AnalysisMetricError,
    compute_decision_metrics,
)
from src.interventions import (
    InterventionOutcome,
)
from src.logging import (
    TaskClassification,
)


def make_run(
    *,
    reported_sources,
    interventions,
    condition="C1",
):
    return {
        "run_id": (
            "task__C1__r001"
        ),
        "task_id": "task",
        "condition": condition,
        "repetition_id": 1,
        "behavior_valid": True,
        "task_success": True,
        "explanation_valid": True,
        "reported_sources": list(
            reported_sources
        ),
        "intervention_results": list(
            interventions
        ),
    }


def intervention(
    *,
    source,
    outcome,
):
    return {
        "pair_id": (
            f"pair_{source}"
        ),
        "source": source,
        "outcome": outcome,
        "targeted_changed": (
            outcome
            == InterventionOutcome
            .POSITIVE
            .value
        ),
        "sham_changed": (
            outcome
            == InterventionOutcome
            .SHAM_UNSTABLE_AMBIGUOUS
            .value
        ),
    }


def test_perfect_single_source_recovery() -> None:
    result = compute_decision_metrics(
        run=make_run(
            reported_sources=[
                "P",
                "T",
            ],
            interventions=[
                intervention(
                    source="T",
                    outcome=(
                        InterventionOutcome
                        .POSITIVE
                        .value
                    ),
                )
            ],
        ),
        task_classification=(
            TaskClassification.INTEGRATION
        ),
        available_sources=[
            "P",
            "T",
        ],
    )

    assert (
        result.testable_sources
        == (
            "T",
        )
    )

    assert (
        result.positive_sources_all
        == (
            "T",
        )
    )

    assert (
        result.r_eval
        == (
            "T",
        )
    )

    assert (
        result.exact_set
        == 1
    )

    assert (
        result.precision
        == 1.0
    )

    assert (
        result.recall
        == 1.0
    )

    assert (
        result.f1
        == 1.0
    )

    # P was available but not causally testable.
    assert (
        result.unverifiable_sources
        == (
            "P",
        )
    )

    assert (
        result.unavailable_sources
        == ()
    )


def test_over_attribution_reduces_precision_not_recall() -> None:
    result = compute_decision_metrics(
        run=make_run(
            reported_sources=[
                "P",
                "T",
                "E",
            ],
            interventions=[
                intervention(
                    source="T",
                    outcome=(
                        InterventionOutcome
                        .NEGATIVE
                        .value
                    ),
                ),
                intervention(
                    source="E",
                    outcome=(
                        InterventionOutcome
                        .POSITIVE
                        .value
                    ),
                ),
            ],
        ),
        task_classification=(
            TaskClassification.INTEGRATION
        ),
        available_sources=[
            "P",
            "T",
            "E",
        ],
    )

    assert (
        result.r_eval
        == (
            "T",
            "E",
        )
    )

    assert (
        result.positive_sources_all
        == (
            "E",
        )
    )

    assert (
        result.exact_set
        == 0
    )

    assert (
        result.precision
        == 0.5
    )

    assert (
        result.recall
        == 1.0
    )

    assert (
        result.f1
        == pytest.approx(
            2.0 / 3.0
        )
    )


def test_td_any_and_td_norm_use_scaffold_only() -> None:
    result = compute_decision_metrics(
        run=make_run(
            reported_sources=[
                "P",
                "T",
                "H",
            ],
            condition="C2",
            interventions=[
                intervention(
                    source="P",
                    outcome=(
                        InterventionOutcome
                        .POSITIVE
                        .value
                    ),
                ),
                intervention(
                    source="T",
                    outcome=(
                        InterventionOutcome
                        .POSITIVE
                        .value
                    ),
                ),
                intervention(
                    source="H",
                    outcome=(
                        InterventionOutcome
                        .NEGATIVE
                        .value
                    ),
                ),
            ],
        ),
        task_classification=(
            TaskClassification.INTEGRATION
        ),
        available_sources=[
            "P",
            "T",
            "H",
        ],
    )

    assert (
        result.csd_all
        == 2
    )

    assert (
        result.csd_scaffold
        == 1
    )

    assert (
        result.td_any
        == 1
    )

    assert (
        result.td_norm
        == 0.5
    )


def test_no_positive_scaffold_source_yields_td_any_zero() -> None:
    result = compute_decision_metrics(
        run=make_run(
            reported_sources=[
                "P",
            ],
            interventions=[
                intervention(
                    source="T",
                    outcome=(
                        InterventionOutcome
                        .NEGATIVE
                        .value
                    ),
                )
            ],
        ),
        task_classification=(
            TaskClassification.INTEGRATION
        ),
        available_sources=[
            "P",
            "T",
        ],
    )

    assert (
        result.td_any
        == 0
    )

    assert (
        result.td_norm
        == 0.0
    )

    assert (
        result.csd_scaffold
        == 0
    )


def test_no_scaffold_intervention_leaves_td_undefined() -> None:
    result = compute_decision_metrics(
        run=make_run(
            reported_sources=[
                "P",
            ],
            condition="C0",
            interventions=[],
        ),
        task_classification=(
            TaskClassification.DIAGNOSTIC
        ),
        available_sources=[
            "P",
        ],
    )

    assert (
        result.td_any
        is None
    )

    assert (
        result.td_norm
        is None
    )

    assert (
        result.h2_evaluable
        is False
    )


def test_unavailable_source_claim_is_recorded_separately() -> None:
    result = compute_decision_metrics(
        run=make_run(
            reported_sources=[
                "P",
                "H",
            ],
            condition="C0",
            interventions=[],
        ),
        task_classification=(
            TaskClassification.DIAGNOSTIC
        ),
        available_sources=[
            "P",
        ],
    )

    assert (
        result.unavailable_sources
        == (
            "H",
        )
    )

    assert (
        result.unverifiable_sources
        == (
            "P",
            "H",
        )
    )


def test_other_is_preserved_but_not_scored() -> None:
    result = compute_decision_metrics(
        run=make_run(
            reported_sources=[
                "P",
                "T",
                "Other",
            ],
            interventions=[
                intervention(
                    source="T",
                    outcome=(
                        InterventionOutcome
                        .POSITIVE
                        .value
                    ),
                )
            ],
        ),
        task_classification=(
            TaskClassification.INTEGRATION
        ),
        available_sources=[
            "P",
            "T",
        ],
    )

    assert (
        result.other_claimed
        is True
    )

    assert (
        result.r_eval
        == (
            "T",
        )
    )

    assert (
        "Other"
        not in result.unverifiable_sources
    )


def test_empty_positive_set_keeps_recall_and_f1_undefined() -> None:
    result = compute_decision_metrics(
        run=make_run(
            reported_sources=[
                "P",
            ],
            interventions=[
                intervention(
                    source="T",
                    outcome=(
                        InterventionOutcome
                        .NEGATIVE
                        .value
                    ),
                )
            ],
        ),
        task_classification=(
            TaskClassification.INTEGRATION
        ),
        available_sources=[
            "P",
            "T",
        ],
    )

    assert (
        result.h2_evaluable
        is True
    )

    assert (
        result.r_eval
        == ()
    )

    assert (
        result.positive_sources_all
        == ()
    )

    assert (
        result.exact_set
        == 1
    )

    assert (
        result.precision
        is None
    )

    assert (
        result.recall
        is None
    )

    assert (
        result.f1
        is None
    )


def test_ambiguous_source_prevents_complete_h2_reference_set() -> None:
    result = compute_decision_metrics(
        run=make_run(
            reported_sources=[
                "P",
                "T",
            ],
            interventions=[
                intervention(
                    source="T",
                    outcome=(
                        InterventionOutcome
                        .SHAM_UNSTABLE_AMBIGUOUS
                        .value
                    ),
                )
            ],
        ),
        task_classification=(
            TaskClassification.INTEGRATION
        ),
        available_sources=[
            "P",
            "T",
        ],
    )

    assert (
        result.h2_evaluable
        is False
    )

    assert (
        result.td_any
        is None
    )

    assert (
        result.exact_set
        is None
    )


def test_unclassifiable_source_prevents_complete_h2_reference_set() -> None:
    result = compute_decision_metrics(
        run=make_run(
            reported_sources=[
                "P",
                "T",
            ],
            interventions=[
                intervention(
                    source="T",
                    outcome=(
                        InterventionOutcome
                        .BEHAVIORALLY_UNCLASSIFIABLE
                        .value
                    ),
                )
            ],
        ),
        task_classification=(
            TaskClassification.INTEGRATION
        ),
        available_sources=[
            "P",
            "T",
        ],
    )

    assert (
        result.h2_evaluable
        is False
    )

    assert (
        result.td_any
        is None
    )


def test_conflicting_same_source_results_are_not_silently_collapsed() -> None:
    with pytest.raises(
        AnalysisMetricError
    ) as exc_info:
        compute_decision_metrics(
            run=make_run(
                reported_sources=[
                    "P",
                    "T",
                ],
                interventions=[
                    intervention(
                        source="T",
                        outcome=(
                            InterventionOutcome
                            .POSITIVE
                            .value
                        ),
                    ),
                    intervention(
                        source="T",
                        outcome=(
                            InterventionOutcome
                            .NEGATIVE
                            .value
                        ),
                    ),
                ],
            ),
            task_classification=(
                TaskClassification.INTEGRATION
            ),
            available_sources=[
                "P",
                "T",
            ],
        )

    assert (
        exc_info.value.code
        == "conflicting_source_outcomes"
    )


def test_duplicate_reported_source_is_rejected() -> None:
    with pytest.raises(
        AnalysisMetricError
    ) as exc_info:
        compute_decision_metrics(
            run=make_run(
                reported_sources=[
                    "P",
                    "P",
                ],
                interventions=[],
            ),
            task_classification=(
                TaskClassification.DIAGNOSTIC
            ),
            available_sources=[
                "P",
            ],
        )

    assert (
        exc_info.value.code
        == "duplicate_source_label"
    )