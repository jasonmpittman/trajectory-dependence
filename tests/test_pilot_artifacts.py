__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from pathlib import Path

import pytest

from src.analysis import (
    PilotAnalysisWriter,
    PilotArtifactAnalyzer,
    PilotArtifactError,
)


PLAN_HASH = (
    "0123456789abcdef"
    "0123456789abcdef"
    "0123456789abcdef"
    "0123456789abcdef"
)


def write_json(
    path: Path,
    value,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )

        handle.write(
            "\n"
        )


def baseline_action(
    value: str,
):
    return {
        "valid": True,
        "parse_result": {
            "normalized_action": {
                "focal_decision_id": (
                    "final_route"
                ),
                "payload": {
                    "action": (
                        "select_route"
                    ),
                    "value": value,
                },
            }
        },
    }


def explanation(
    sources,
):
    return {
        "valid": True,
        "parse_result": {
            "reported_sources": list(
                sources
            )
        },
    }


def classification(
    *,
    source: str,
    outcome: str,
):
    return {
        "source": source,
        "outcome": outcome,
        "baseline_action": None,
        "targeted_action": None,
        "sham_action": None,
        "targeted_changed": (
            outcome
            == "intervention_positive"
        ),
        "sham_changed": (
            outcome
            == "sham_unstable_ambiguous"
        ),
        "reason": "synthetic test",
    }


def run_summary(
    *,
    run_id: str,
    condition: str,
    value: str,
    reported_sources,
    interventions,
):
    return {
        "experiment_id": (
            "synthetic_pilot"
        ),
        "run_id": run_id,
        "task_id": (
            "integration_task"
        ),
        "condition": condition,
        "repetition_id": 1,
        "output_directory": (
            "synthetic"
        ),
        "artifacts": {},
        "snapshot_id": (
            f"snapshot_{condition}"
        ),
        "snapshot_checksum": (
            f"checksum_{condition}"
        ),
        "baseline_action": (
            baseline_action(
                value
            )
        ),
        "self_explanation": (
            explanation(
                reported_sources
            )
        ),
        "behavior_valid": True,
        "task_success": True,
        "expected_action": {
            "action": (
                "select_route"
            ),
            "value": value,
        },
        "intervention_results": [
            {
                "pair_id": pair_id,
                "validated_pair": {},
                "replay_execution": {},
                "classification": (
                    classification(
                        source=source,
                        outcome=outcome,
                    )
                ),
                "targeted_events_path": (
                    "targeted.jsonl"
                ),
                "sham_events_path": (
                    "sham.jsonl"
                ),
            }
            for (
                pair_id,
                source,
                outcome
            )
            in interventions
        ],
    }


def concise_run(
    *,
    run_id: str,
    condition: str,
    value: str,
    reported_sources,
    interventions,
):
    return {
        "behavior_valid": True,
        "condition": condition,
        "explanation_valid": True,
        "intervention_results": [
            {
                "outcome": outcome,
                "pair_id": pair_id,
                "sham_changed": False,
                "sham_events_path": (
                    "sham.jsonl"
                ),
                "source": source,
                "targeted_changed": (
                    outcome
                    == "intervention_positive"
                ),
                "targeted_events_path": (
                    "targeted.jsonl"
                ),
            }
            for (
                pair_id,
                source,
                outcome
            )
            in interventions
        ],
        "normalized_action": {
            "action": (
                "select_route"
            ),
            "value": value,
        },
        "ordinal": (
            1
            if condition == "C1"
            else 2
        ),
        "repetition_id": 1,
        "reported_sources": list(
            reported_sources
        ),
        "run_id": run_id,
        "run_summary_path": (
            "synthetic"
        ),
        "task_id": (
            "integration_task"
        ),
        "task_success": True,
    }


def make_bundle(
    tmp_path,
):
    root = (
        tmp_path
        / "pilot"
    )

    root.mkdir()

    task_suite = {
        "metadata": {},
        "schema_version": (
            "pilot-task-v1"
        ),
        "suite_id": (
            "synthetic_suite"
        ),
        "suite_version": (
            "0.1.0"
        ),
        "tasks": [
            {
                "task_id": (
                    "integration_task"
                ),
                "task_version": (
                    "0.1.0"
                ),
                "family": (
                    "multi_source_dependence"
                ),
                "class": (
                    "integration"
                ),
                "conditions": [
                    "C1",
                    "C2",
                ],
                "task_prompt": (
                    "Synthetic."
                ),
                "focal_decision": {},
                "available_sources": {
                    "C1": [
                        "P",
                        "T"
                    ],
                    "C2": [
                        "P",
                        "T",
                        "H"
                    ],
                },
                "tool_fixtures": [],
                "history_fixtures": [],
                "memory_fixtures": [],
                "environment": None,
                "intervention_pairs": [],
                "causal_ground_truth": {},
                "notes": None,
            }
        ],
    }

    run1_interventions = [
        (
            "t_pair",
            "T",
            "intervention_positive",
        )
    ]

    run2_interventions = [
        (
            "t_pair",
            "T",
            "intervention_negative",
        ),
        (
            "h_pair",
            "H",
            "intervention_positive",
        ),
    ]

    summary = {
        "completed_runs": 2,
        "experiment_id": (
            "synthetic_pilot"
        ),
        "intervention_outcome_counts": {
            "behaviorally_unclassifiable": 0,
            "intervention_negative": 1,
            "intervention_positive": 2,
            "sham_unstable_ambiguous": 0,
        },
        "invalid_behavior_runs": 0,
        "invalid_explanations": 0,
        "model_runtime_metadata": {},
        "plan_sha256": PLAN_HASH,
        "planned_runs": 2,
        "runs": [
            concise_run(
                run_id=(
                    "integration_task__C1__r001"
                ),
                condition="C1",
                value="B",
                reported_sources=[
                    "P",
                    "T",
                ],
                interventions=(
                    run1_interventions
                ),
            ),
            concise_run(
                run_id=(
                    "integration_task__C2__r001"
                ),
                condition="C2",
                value="B",
                reported_sources=[
                    "P",
                    "T",
                    "H",
                ],
                interventions=(
                    run2_interventions
                ),
            ),
        ],
        "successful_task_runs": 2,
        "suite_id": (
            "synthetic_suite"
        ),
        "suite_version": (
            "0.1.0"
        ),
        "unscored_task_runs": 0,
        "unsuccessful_task_runs": 0,
        "valid_behavior_runs": 2,
        "valid_explanations": 2,
    }

    write_json(
        root
        / "task_suite.json",
        task_suite,
    )

    write_json(
        root
        / "pilot_summary.json",
        summary,
    )

    write_json(
        root
        / "pilot_config.json",
        {
            "experiment_id": (
                "synthetic_pilot"
            )
        },
    )

    write_json(
        root
        / "execution_plan.json",
        {
            "experiment_id": (
                "synthetic_pilot"
            ),
            "suite_id": (
                "synthetic_suite"
            ),
            "suite_version": (
                "0.1.0"
            ),
            "plan_sha256": (
                PLAN_HASH
            ),
        },
    )

    write_json(
        root
        / "preflight.json",
        {
            "plan": {
                "plan_sha256": (
                    PLAN_HASH
                )
            }
        },
    )

    write_json(
        root
        / "model_runtime.json",
        {
            "model_identifier": (
                "synthetic"
            )
        },
    )

    for (
        condition,
        value,
        sources,
        interventions,
    ) in (
        (
            "C1",
            "B",
            [
                "P",
                "T",
            ],
            run1_interventions,
        ),
        (
            "C2",
            "B",
            [
                "P",
                "T",
                "H",
            ],
            run2_interventions,
        ),
    ):
        run_id = (
            f"integration_task"
            f"__{condition}"
            f"__r001"
        )

        write_json(
            root
            / "integration_task"
            / condition
            / "r001"
            / "run_summary.json",
            run_summary(
                run_id=run_id,
                condition=condition,
                value=value,
                reported_sources=(
                    sources
                ),
                interventions=(
                    interventions
                ),
            ),
        )

    return root


def test_analyzer_reconstructs_two_runs(
    tmp_path,
) -> None:
    root = make_bundle(
        tmp_path
    )

    analysis = (
        PilotArtifactAnalyzer()
        .analyze(
            root
        )
    )

    assert len(
        analysis.decision_metrics
    ) == 2

    assert len(
        analysis.intervention_metrics
    ) == 3


def test_analyzer_reproduces_outcome_counts(
    tmp_path,
) -> None:
    root = make_bundle(
        tmp_path
    )

    analysis = (
        PilotArtifactAnalyzer()
        .analyze(
            root
        )
    )

    outcomes = (
        analysis
        .aggregates[
            "overall"
        ][
            "intervention_outcomes"
        ]
    )

    assert (
        outcomes[
            "intervention_positive"
        ]
        == 2
    )

    assert (
        outcomes[
            "intervention_negative"
        ]
        == 1
    )


def test_analyzer_computes_c1_td(
    tmp_path,
) -> None:
    root = make_bundle(
        tmp_path
    )

    analysis = (
        PilotArtifactAnalyzer()
        .analyze(
            root
        )
    )

    c1 = (
        analysis
        .aggregates[
            "h1_integration"
        ][
            "conditions"
        ][
            "C1"
        ]
    )

    assert (
        c1[
            "td_any"
        ]
        == 1.0
    )

    assert (
        c1[
            "mean_td_norm"
        ]
        == 1.0
    )


def test_analyzer_computes_over_attribution(
    tmp_path,
) -> None:
    root = make_bundle(
        tmp_path
    )

    analysis = (
        PilotArtifactAnalyzer()
        .analyze(
            root
        )
    )

    c2 = next(
        decision
        for decision
        in analysis.decision_metrics
        if (
            decision.condition
            == "C2"
        )
    )

    assert (
        c2.r_eval
        == (
            "T",
            "H",
        )
    )

    assert (
        c2.positive_sources_all
        == (
            "H",
        )
    )

    assert (
        c2.precision
        == 0.5
    )

    assert (
        c2.recall
        == 1.0
    )

    assert (
        c2.f1
        == pytest.approx(
            2.0 / 3.0
        )
    )


def test_summary_count_mismatch_is_rejected(
    tmp_path,
) -> None:
    root = make_bundle(
        tmp_path
    )

    summary_path = (
        root
        / "pilot_summary.json"
    )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(
            handle
        )

    summary[
        "valid_behavior_runs"
    ] = 1

    write_json(
        summary_path,
        summary,
    )

    with pytest.raises(
        PilotArtifactError
    ) as exc_info:
        (
            PilotArtifactAnalyzer()
            .analyze(
                root
            )
        )

    assert (
        exc_info.value.code
        == "summary_count_mismatch"
    )


def test_missing_run_summary_is_rejected(
    tmp_path,
) -> None:
    root = make_bundle(
        tmp_path
    )

    path = (
        root
        / "integration_task"
        / "C2"
        / "r001"
        / "run_summary.json"
    )

    path.unlink()

    with pytest.raises(
        PilotArtifactError
    ) as exc_info:
        (
            PilotArtifactAnalyzer()
            .analyze(
                root
            )
        )

    assert (
        exc_info.value.code
        == "run_summary_missing"
    )


def test_writer_creates_processed_outputs(
    tmp_path,
) -> None:
    root = make_bundle(
        tmp_path
    )

    analysis = (
        PilotArtifactAnalyzer()
        .analyze(
            root
        )
    )

    outputs = (
        PilotAnalysisWriter()
        .write(
            analysis=analysis,
            processed_directory=(
                tmp_path
                / "processed"
            ),
            report_path=(
                tmp_path
                / "results"
                / "pilot-report.md"
            ),
        )
    )

    assert Path(
        outputs.decision_metrics_csv
    ).is_file()

    assert Path(
        outputs.intervention_metrics_csv
    ).is_file()

    assert Path(
        outputs.pilot_metrics_json
    ).is_file()

    assert Path(
        outputs.pilot_report_markdown
    ).is_file()


def test_writer_refuses_overwrite_by_default(
    tmp_path,
) -> None:
    root = make_bundle(
        tmp_path
    )

    analysis = (
        PilotArtifactAnalyzer()
        .analyze(
            root
        )
    )

    writer = (
        PilotAnalysisWriter()
    )

    writer.write(
        analysis=analysis,
        processed_directory=(
            tmp_path
            / "processed"
        ),
        report_path=(
            tmp_path
            / "results"
            / "pilot-report.md"
        ),
    )

    with pytest.raises(
        PilotArtifactError
    ) as exc_info:
        writer.write(
            analysis=analysis,
            processed_directory=(
                tmp_path
                / "processed"
            ),
            report_path=(
                tmp_path
                / "results"
                / "pilot-report.md"
            ),
        )

    assert (
        exc_info.value.code
        == "analysis_output_exists"
    )


def test_generated_report_is_labeled_methodological(
    tmp_path,
) -> None:
    root = make_bundle(
        tmp_path
    )

    analysis = (
        PilotArtifactAnalyzer()
        .analyze(
            root
        )
    )

    outputs = (
        PilotAnalysisWriter()
        .write(
            analysis=analysis,
            processed_directory=(
                tmp_path
                / "processed"
            ),
            report_path=(
                tmp_path
                / "results"
                / "pilot-report.md"
            ),
        )
    )

    text = Path(
        outputs.pilot_report_markdown
    ).read_text(
        encoding="utf-8"
    )

    assert (
        "methodological pilot"
        in text.lower()
    )

    assert (
        "not a confirmatory test"
        in text.lower()
    )