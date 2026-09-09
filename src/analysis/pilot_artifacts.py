__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from ..interventions import (
    InterventionOutcome,
)
from ..logging import (
    TaskClassification,
)
from .pilot_metrics import (
    DecisionMetrics,
    STANDARD_SOURCES,
    compute_decision_metrics,
)


class PilotArtifactError(RuntimeError):
    """Defined failure analyzing a frozen pilot artifact bundle."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
    ) -> None:
        super().__init__(
            message
        )

        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class PilotArtifactAnalysis:
    """Complete reproducible analysis extracted from one pilot bundle."""

    pilot_root: str

    experiment_id: str

    suite_id: str
    suite_version: str

    plan_sha256: str

    integrity: dict[
        str,
        Any,
    ]

    decision_metrics: tuple[
        DecisionMetrics,
        ...
    ]

    intervention_metrics: tuple[
        dict[str, Any],
        ...
    ]

    aggregates: dict[
        str,
        Any,
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pilot_root": (
                self.pilot_root
            ),
            "experiment_id": (
                self.experiment_id
            ),
            "suite_id": (
                self.suite_id
            ),
            "suite_version": (
                self.suite_version
            ),
            "plan_sha256": (
                self.plan_sha256
            ),
            "integrity": deepcopy(
                self.integrity
            ),
            "decision_metrics": [
                item.to_dict()
                for item
                in self.decision_metrics
            ],
            "intervention_metrics": [
                deepcopy(
                    item
                )
                for item
                in self.intervention_metrics
            ],
            "aggregates": deepcopy(
                self.aggregates
            ),
        }


@dataclass(frozen=True)
class PilotAnalysisOutputs:
    """Paths written by PilotAnalysisWriter."""

    decision_metrics_csv: str
    intervention_metrics_csv: str
    pilot_metrics_json: str
    pilot_report_markdown: str

    def to_dict(self) -> dict[str, str]:
        return {
            "decision_metrics_csv": (
                self.decision_metrics_csv
            ),
            "intervention_metrics_csv": (
                self.intervention_metrics_csv
            ),
            "pilot_metrics_json": (
                self.pilot_metrics_json
            ),
            "pilot_report_markdown": (
                self.pilot_report_markdown
            ),
        }


class PilotArtifactAnalyzer:
    """
    Reconstruct pilot metrics from immutable experiment artifacts.

    The analyzer does not modify raw pilot files.

    It cross-checks:

    - aggregate pilot metadata;
    - task-suite metadata;
    - execution-plan hash;
    - every individual run_summary.json;
    - behavior/explanation validity;
    - reported source sets;
    - intervention classifications;
    - top-level aggregate counts.
    """

    _REQUIRED_AGGREGATE_FILES = (
        "pilot_config.json",
        "task_suite.json",
        "execution_plan.json",
        "preflight.json",
        "model_runtime.json",
        "pilot_summary.json",
    )

    def analyze(
        self,
        pilot_root: str | Path,
    ) -> PilotArtifactAnalysis:
        root = Path(
            pilot_root
        )

        if not root.is_dir():
            raise PilotArtifactError(
                code="pilot_root_not_found",
                message=(
                    f"Pilot artifact root "
                    f"{str(root)!r} does not exist."
                ),
            )

        self._validate_aggregate_files(
            root
        )

        summary = self._read_json(
            root
            / "pilot_summary.json"
        )

        suite = self._read_json(
            root
            / "task_suite.json"
        )

        plan = self._read_json(
            root
            / "execution_plan.json"
        )

        preflight = self._read_json(
            root
            / "preflight.json"
        )

        pilot_config = self._read_json(
            root
            / "pilot_config.json"
        )

        self._validate_bundle_identity(
            summary=summary,
            suite=suite,
            plan=plan,
            preflight=preflight,
            pilot_config=pilot_config,
        )

        task_map = self._task_map(
            suite
        )

        concise_runs = summary.get(
            "runs"
        )

        if not isinstance(
            concise_runs,
            list,
        ):
            raise PilotArtifactError(
                code="invalid_pilot_summary",
                message=(
                    "pilot_summary.json field "
                    "'runs' must be an array."
                ),
            )

        decisions: list[
            DecisionMetrics
        ] = []

        interventions: list[
            dict[str, Any]
        ] = []

        run_summaries_checked = 0

        for concise in concise_runs:
            if not isinstance(
                concise,
                Mapping,
            ):
                raise PilotArtifactError(
                    code="invalid_pilot_run",
                    message=(
                        "Every pilot-summary run "
                        "must be a mapping."
                    ),
                )

            task_id = self._required_string(
                concise,
                "task_id",
            )

            condition = self._required_string(
                concise,
                "condition",
            )

            repetition_id = concise.get(
                "repetition_id"
            )

            if (
                not isinstance(
                    repetition_id,
                    int,
                )
                or repetition_id < 0
            ):
                raise PilotArtifactError(
                    code="invalid_repetition_id",
                    message=(
                        f"Pilot run for task "
                        f"{task_id!r} has invalid "
                        "repetition_id."
                    ),
                )

            task = task_map.get(
                task_id
            )

            if task is None:
                raise PilotArtifactError(
                    code="task_not_in_suite",
                    message=(
                        f"Pilot summary references "
                        f"unknown task {task_id!r}."
                    ),
                )

            run_summary_path = (
                root
                / task_id
                / condition
                / (
                    f"r"
                    f"{repetition_id:03d}"
                )
                / "run_summary.json"
            )

            if not run_summary_path.is_file():
                raise PilotArtifactError(
                    code="run_summary_missing",
                    message=(
                        f"Expected run summary "
                        f"{str(run_summary_path)!r} "
                        "does not exist."
                    ),
                )

            run_summary = self._read_json(
                run_summary_path
            )

            derived_run = (
                self._derive_concise_run(
                    run_summary
                )
            )

            self._validate_concise_run(
                declared=concise,
                derived=derived_run,
                expected_path=(
                    run_summary_path
                ),
            )

            try:
                classification = (
                    TaskClassification(
                        task[
                            "class"
                        ]
                    )
                )

            except (
                KeyError,
                ValueError,
                TypeError,
            ) as exc:
                raise PilotArtifactError(
                    code="invalid_task_classification",
                    message=(
                        f"Task {task_id!r} contains "
                        "an invalid class."
                    ),
                ) from exc

            available_sources = (
                task.get(
                    "available_sources",
                    {}
                )
                .get(
                    condition
                )
            )

            if not isinstance(
                available_sources,
                list,
            ):
                raise PilotArtifactError(
                    code="available_sources_missing",
                    message=(
                        f"Task {task_id!r} does "
                        f"not define available_sources "
                        f"for {condition}."
                    ),
                )

            decision = (
                compute_decision_metrics(
                    run=derived_run,
                    task_classification=(
                        classification
                    ),
                    available_sources=(
                        available_sources
                    ),
                )
            )

            decisions.append(
                decision
            )

            for intervention in (
                derived_run[
                    "intervention_results"
                ]
            ):
                interventions.append(
                    {
                        "run_id": (
                            decision.run_id
                        ),
                        "task_id": (
                            decision.task_id
                        ),
                        "task_classification": (
                            classification.value
                        ),
                        "condition": (
                            decision.condition
                        ),
                        "repetition_id": (
                            decision.repetition_id
                        ),
                        "pair_id": (
                            intervention[
                                "pair_id"
                            ]
                        ),
                        "source": (
                            intervention[
                                "source"
                            ]
                        ),
                        "outcome": (
                            intervention[
                                "outcome"
                            ]
                        ),
                        "targeted_changed": (
                            intervention[
                                "targeted_changed"
                            ]
                        ),
                        "sham_changed": (
                            intervention[
                                "sham_changed"
                            ]
                        ),
                    }
                )

            run_summaries_checked += 1

        self._validate_summary_totals(
            summary=summary,
            decisions=decisions,
            interventions=interventions,
        )

        aggregates = self._aggregate(
            decisions=decisions,
            interventions=interventions,
        )

        integrity = {
            "required_aggregate_files": (
                list(
                    self
                    ._REQUIRED_AGGREGATE_FILES
                )
            ),
            "required_aggregate_files_present": (
                True
            ),
            "pilot_failure_present": (
                (
                    root
                    / "pilot_failure.json"
                ).exists()
            ),
            "pilot_partial_summary_present": (
                (
                    root
                    / "pilot_partial_summary.json"
                ).exists()
            ),
            "run_summaries_checked": (
                run_summaries_checked
            ),
            "summary_counts_reproduced": (
                True
            ),
            "plan_hash_consistent": (
                True
            ),
            "task_suite_consistent": (
                True
            ),
        }

        return PilotArtifactAnalysis(
            pilot_root=str(
                root
            ),
            experiment_id=(
                self._required_string(
                    summary,
                    "experiment_id",
                )
            ),
            suite_id=(
                self._required_string(
                    summary,
                    "suite_id",
                )
            ),
            suite_version=(
                self._required_string(
                    summary,
                    "suite_version",
                )
            ),
            plan_sha256=(
                self._required_string(
                    summary,
                    "plan_sha256",
                )
            ),
            integrity=integrity,
            decision_metrics=tuple(
                decisions
            ),
            intervention_metrics=tuple(
                interventions
            ),
            aggregates=aggregates,
        )

    def _validate_aggregate_files(
        self,
        root: Path,
    ) -> None:
        missing = [
            name
            for name
            in self._REQUIRED_AGGREGATE_FILES
            if not (
                root
                / name
            ).is_file()
        ]

        if missing:
            raise PilotArtifactError(
                code="pilot_bundle_incomplete",
                message=(
                    "Pilot artifact bundle is "
                    "missing required aggregate "
                    "file(s): "
                    + ", ".join(
                        missing
                    )
                ),
            )

        if (
            root
            / "pilot_failure.json"
        ).exists():
            raise PilotArtifactError(
                code="pilot_failure_artifact_present",
                message=(
                    "pilot_failure.json is present. "
                    "Analyze the failure/partial bundle "
                    "separately rather than treating it "
                    "as a completed pilot."
                ),
            )

    def _validate_bundle_identity(
        self,
        *,
        summary: Mapping[
            str,
            Any,
        ],
        suite: Mapping[
            str,
            Any,
        ],
        plan: Mapping[
            str,
            Any,
        ],
        preflight: Mapping[
            str,
            Any,
        ],
        pilot_config: Mapping[
            str,
            Any,
        ],
    ) -> None:
        summary_experiment = (
            self._required_string(
                summary,
                "experiment_id",
            )
        )

        config_experiment = (
            self._required_string(
                pilot_config,
                "experiment_id",
            )
        )

        plan_experiment = (
            self._required_string(
                plan,
                "experiment_id",
            )
        )

        if not (
            summary_experiment
            == config_experiment
            == plan_experiment
        ):
            raise PilotArtifactError(
                code="experiment_identity_mismatch",
                message=(
                    "Pilot config, plan, and summary "
                    "do not share one experiment_id."
                ),
            )

        summary_suite = (
            self._required_string(
                summary,
                "suite_id",
            )
        )

        suite_id = (
            self._required_string(
                suite,
                "suite_id",
            )
        )

        plan_suite = (
            self._required_string(
                plan,
                "suite_id",
            )
        )

        if not (
            summary_suite
            == suite_id
            == plan_suite
        ):
            raise PilotArtifactError(
                code="suite_identity_mismatch",
                message=(
                    "Task suite, execution plan, "
                    "and pilot summary disagree "
                    "on suite_id."
                ),
            )

        summary_suite_version = (
            self._required_string(
                summary,
                "suite_version",
            )
        )

        suite_version = (
            self._required_string(
                suite,
                "suite_version",
            )
        )

        plan_suite_version = (
            self._required_string(
                plan,
                "suite_version",
            )
        )

        if not (
            summary_suite_version
            == suite_version
            == plan_suite_version
        ):
            raise PilotArtifactError(
                code="suite_version_mismatch",
                message=(
                    "Task suite, execution plan, "
                    "and pilot summary disagree "
                    "on suite_version."
                ),
            )

        summary_hash = (
            self._required_string(
                summary,
                "plan_sha256",
            )
        )

        plan_hash = (
            self._required_string(
                plan,
                "plan_sha256",
            )
        )

        preflight_plan = (
            preflight.get(
                "plan"
            )
        )

        if not isinstance(
            preflight_plan,
            Mapping,
        ):
            raise PilotArtifactError(
                code="invalid_preflight_artifact",
                message=(
                    "preflight.json does not "
                    "contain a plan object."
                ),
            )

        preflight_hash = (
            self._required_string(
                preflight_plan,
                "plan_sha256",
            )
        )

        if not (
            summary_hash
            == plan_hash
            == preflight_hash
        ):
            raise PilotArtifactError(
                code="plan_hash_mismatch",
                message=(
                    "Execution plan hash differs "
                    "across pilot artifacts."
                ),
            )

    def _task_map(
        self,
        suite: Mapping[
            str,
            Any,
        ],
    ) -> dict[
        str,
        Mapping[
            str,
            Any,
        ],
    ]:
        tasks = suite.get(
            "tasks"
        )

        if not isinstance(
            tasks,
            list,
        ):
            raise PilotArtifactError(
                code="invalid_task_suite",
                message=(
                    "task_suite.json field "
                    "'tasks' must be an array."
                ),
            )

        values: dict[
            str,
            Mapping[
                str,
                Any,
            ],
        ] = {}

        for task in tasks:
            if not isinstance(
                task,
                Mapping,
            ):
                raise PilotArtifactError(
                    code="invalid_task_suite",
                    message=(
                        "Every task-suite task "
                        "must be a mapping."
                    ),
                )

            task_id = (
                self._required_string(
                    task,
                    "task_id",
                )
            )

            if task_id in values:
                raise PilotArtifactError(
                    code="duplicate_task_id",
                    message=(
                        f"Duplicate task_id "
                        f"{task_id!r} in "
                        "task_suite.json."
                    ),
                )

            values[
                task_id
            ] = task

        return values

    def _derive_concise_run(
        self,
        run_summary: Mapping[
            str,
            Any,
        ],
    ) -> dict[str, Any]:
        run_id = self._required_string(
            run_summary,
            "run_id",
        )

        baseline = run_summary.get(
            "baseline_action"
        )

        explanation = run_summary.get(
            "self_explanation"
        )

        if not isinstance(
            baseline,
            Mapping,
        ):
            raise PilotArtifactError(
                code="invalid_run_summary",
                message=(
                    f"Run {run_id!r} does "
                    "not contain baseline_action."
                ),
            )

        if not isinstance(
            explanation,
            Mapping,
        ):
            raise PilotArtifactError(
                code="invalid_run_summary",
                message=(
                    f"Run {run_id!r} does "
                    "not contain self_explanation."
                ),
            )

        baseline_valid = baseline.get(
            "valid"
        )

        explanation_valid = (
            explanation.get(
                "valid"
            )
        )

        if not isinstance(
            baseline_valid,
            bool,
        ):
            raise PilotArtifactError(
                code="invalid_run_summary",
                message=(
                    f"Run {run_id!r} has "
                    "invalid baseline valid flag."
                ),
            )

        if not isinstance(
            explanation_valid,
            bool,
        ):
            raise PilotArtifactError(
                code="invalid_run_summary",
                message=(
                    f"Run {run_id!r} has "
                    "invalid explanation valid flag."
                ),
            )

        normalized_action = None

        if baseline_valid:
            parse_result = baseline.get(
                "parse_result"
            )

            if not isinstance(
                parse_result,
                Mapping,
            ):
                raise PilotArtifactError(
                    code="invalid_run_summary",
                    message=(
                        f"Valid baseline action "
                        f"for {run_id!r} lacks "
                        "parse_result."
                    ),
                )

            normalized = (
                parse_result.get(
                    "normalized_action"
                )
            )

            if not isinstance(
                normalized,
                Mapping,
            ):
                raise PilotArtifactError(
                    code="invalid_run_summary",
                    message=(
                        f"Valid baseline action "
                        f"for {run_id!r} lacks "
                        "normalized_action."
                    ),
                )

            normalized_action = deepcopy(
                normalized.get(
                    "payload"
                )
            )

        reported_sources: list[
            str
        ] = []

        if explanation_valid:
            parse_result = (
                explanation.get(
                    "parse_result"
                )
            )

            if not isinstance(
                parse_result,
                Mapping,
            ):
                raise PilotArtifactError(
                    code="invalid_run_summary",
                    message=(
                        f"Valid explanation "
                        f"for {run_id!r} lacks "
                        "parse_result."
                    ),
                )

            reported = (
                parse_result.get(
                    "reported_sources"
                )
            )

            if not isinstance(
                reported,
                list,
            ):
                raise PilotArtifactError(
                    code="invalid_run_summary",
                    message=(
                        f"Explanation for "
                        f"{run_id!r} lacks "
                        "reported_sources array."
                    ),
                )

            reported_sources = list(
                reported
            )

        intervention_results = (
            run_summary.get(
                "intervention_results",
                []
            )
        )

        if not isinstance(
            intervention_results,
            list,
        ):
            raise PilotArtifactError(
                code="invalid_run_summary",
                message=(
                    f"Run {run_id!r} "
                    "intervention_results "
                    "must be an array."
                ),
            )

        concise_interventions: list[
            dict[str, Any]
        ] = []

        for intervention in (
            intervention_results
        ):
            if not isinstance(
                intervention,
                Mapping,
            ):
                raise PilotArtifactError(
                    code="invalid_intervention_result",
                    message=(
                        f"Run {run_id!r} contains "
                        "an invalid intervention row."
                    ),
                )

            classification = (
                intervention.get(
                    "classification"
                )
            )

            if not isinstance(
                classification,
                Mapping,
            ):
                raise PilotArtifactError(
                    code="invalid_intervention_result",
                    message=(
                        f"Run {run_id!r} intervention "
                        "lacks classification."
                    ),
                )

            concise_interventions.append(
                {
                    "pair_id": (
                        self._required_string(
                            intervention,
                            "pair_id",
                        )
                    ),
                    "source": (
                        self._required_string(
                            classification,
                            "source",
                        )
                    ),
                    "outcome": (
                        self._required_string(
                            classification,
                            "outcome",
                        )
                    ),
                    "targeted_changed": (
                        classification.get(
                            "targeted_changed"
                        )
                    ),
                    "sham_changed": (
                        classification.get(
                            "sham_changed"
                        )
                    ),
                }
            )

        return {
            "run_id": run_id,
            "task_id": (
                self._required_string(
                    run_summary,
                    "task_id",
                )
            ),
            "condition": (
                self._required_string(
                    run_summary,
                    "condition",
                )
            ),
            "repetition_id": (
                run_summary.get(
                    "repetition_id"
                )
            ),
            "behavior_valid": (
                run_summary.get(
                    "behavior_valid"
                )
            ),
            "task_success": (
                run_summary.get(
                    "task_success"
                )
            ),
            "explanation_valid": (
                explanation_valid
            ),
            "normalized_action": (
                normalized_action
            ),
            "reported_sources": (
                reported_sources
            ),
            "intervention_results": (
                concise_interventions
            ),
        }

    def _validate_concise_run(
        self,
        *,
        declared: Mapping[
            str,
            Any,
        ],
        derived: Mapping[
            str,
            Any,
        ],
        expected_path: Path,
    ) -> None:
        fields = (
            "run_id",
            "task_id",
            "condition",
            "repetition_id",
            "behavior_valid",
            "task_success",
            "explanation_valid",
            "normalized_action",
            "reported_sources",
        )

        for field_name in fields:
            if (
                declared.get(
                    field_name
                )
                != derived.get(
                    field_name
                )
            ):
                raise PilotArtifactError(
                    code="run_summary_mismatch",
                    message=(
                        f"Aggregate pilot summary "
                        f"does not match "
                        f"{str(expected_path)!r} "
                        f"for field "
                        f"{field_name!r}."
                    ),
                )

        declared_interventions = (
            declared.get(
                "intervention_results",
                []
            )
        )

        if not isinstance(
            declared_interventions,
            list,
        ):
            raise PilotArtifactError(
                code="run_summary_mismatch",
                message=(
                    "Aggregate intervention_results "
                    "must be an array."
                ),
            )

        def normalized(
            values,
        ):
            return sorted(
                (
                    {
                        "pair_id": (
                            value.get(
                                "pair_id"
                            )
                        ),
                        "source": (
                            value.get(
                                "source"
                            )
                        ),
                        "outcome": (
                            value.get(
                                "outcome"
                            )
                        ),
                        "targeted_changed": (
                            value.get(
                                "targeted_changed"
                            )
                        ),
                        "sham_changed": (
                            value.get(
                                "sham_changed"
                            )
                        ),
                    }
                    for value
                    in values
                ),
                key=lambda item: (
                    str(
                        item[
                            "pair_id"
                        ]
                    )
                ),
            )

        if (
            normalized(
                declared_interventions
            )
            != normalized(
                derived[
                    "intervention_results"
                ]
            )
        ):
            raise PilotArtifactError(
                code="run_summary_mismatch",
                message=(
                    "Aggregate pilot intervention "
                    "results do not match "
                    f"{str(expected_path)!r}."
                ),
            )

    def _validate_summary_totals(
        self,
        *,
        summary: Mapping[
            str,
            Any,
        ],
        decisions: Sequence[
            DecisionMetrics,
        ],
        interventions: Sequence[
            Mapping[
                str,
                Any,
            ],
        ],
    ) -> None:
        derived = {
            "completed_runs": len(
                decisions
            ),
            "valid_behavior_runs": sum(
                1
                for decision
                in decisions
                if (
                    decision
                    .behavior_valid
                )
            ),
            "invalid_behavior_runs": sum(
                1
                for decision
                in decisions
                if (
                    not decision
                    .behavior_valid
                )
            ),
            "successful_task_runs": sum(
                1
                for decision
                in decisions
                if (
                    decision
                    .task_success
                    is True
                )
            ),
            "unsuccessful_task_runs": sum(
                1
                for decision
                in decisions
                if (
                    decision
                    .task_success
                    is False
                )
            ),
            "unscored_task_runs": sum(
                1
                for decision
                in decisions
                if (
                    decision
                    .task_success
                    is None
                )
            ),
            "valid_explanations": sum(
                1
                for decision
                in decisions
                if (
                    decision
                    .explanation_valid
                )
            ),
            "invalid_explanations": sum(
                1
                for decision
                in decisions
                if (
                    not decision
                    .explanation_valid
                )
            ),
        }

        for (
            field_name,
            value,
        ) in derived.items():
            if (
                summary.get(
                    field_name
                )
                != value
            ):
                raise PilotArtifactError(
                    code="summary_count_mismatch",
                    message=(
                        f"pilot_summary.json field "
                        f"{field_name!r} does not "
                        f"match run-level artifacts: "
                        f"declared="
                        f"{summary.get(field_name)!r}, "
                        f"derived={value!r}."
                    ),
                )

        planned_runs = summary.get(
            "planned_runs"
        )

        if (
            planned_runs
            != len(
                decisions
            )
        ):
            raise PilotArtifactError(
                code="pilot_incomplete",
                message=(
                    "Completed pilot summary "
                    "does not contain every "
                    "planned run."
                ),
            )

        observed_outcomes = {
            outcome.value: 0
            for outcome
            in InterventionOutcome
        }

        for intervention in (
            interventions
        ):
            outcome = intervention[
                "outcome"
            ]

            if (
                outcome
                not in observed_outcomes
            ):
                raise PilotArtifactError(
                    code="unknown_intervention_outcome",
                    message=(
                        f"Unknown intervention "
                        f"outcome {outcome!r}."
                    ),
                )

            observed_outcomes[
                outcome
            ] += 1

        declared_outcomes = (
            summary.get(
                "intervention_outcome_counts"
            )
        )

        if (
            not isinstance(
                declared_outcomes,
                Mapping,
            )
            or dict(
                declared_outcomes
            )
            != observed_outcomes
        ):
            raise PilotArtifactError(
                code="summary_intervention_count_mismatch",
                message=(
                    "pilot_summary.json intervention "
                    "outcome counts do not match "
                    "run-level artifacts."
                ),
            )

    def _aggregate(
        self,
        *,
        decisions: Sequence[
            DecisionMetrics,
        ],
        interventions: Sequence[
            Mapping[
                str,
                Any,
            ],
        ],
    ) -> dict[str, Any]:
        return {
            "overall": (
                self._aggregate_overall(
                    decisions=decisions,
                    interventions=(
                        interventions
                    ),
                )
            ),
            "diagnostic_validation": (
                self
                ._aggregate_diagnostic(
                    interventions
                )
            ),
            "h1_integration": (
                self._aggregate_h1(
                    decisions=decisions,
                    interventions=(
                        interventions
                    ),
                )
            ),
            "h2_all_evaluable": (
                self._aggregate_h2(
                    [
                        decision
                        for decision
                        in decisions
                        if (
                            decision
                            .h2_evaluable
                        )
                    ]
                )
            ),
            "h2_integration_evaluable": (
                self._aggregate_h2(
                    [
                        decision
                        for decision
                        in decisions
                        if (
                            decision
                            .h2_evaluable
                            and (
                                decision
                                .task_classification
                                == (
                                    TaskClassification
                                    .INTEGRATION
                                )
                            )
                        )
                    ]
                )
            ),
            "h2_by_csd_scaffold": (
                self._aggregate_h2_by_csd(
                    decisions
                )
            ),
            "source_claim_diagnostics": (
                self
                ._aggregate_claims(
                    decisions
                )
            ),
            "interventions_by_source": (
                self
                ._aggregate_intervention_sources(
                    interventions
                )
            ),
        }

    @staticmethod
    def _aggregate_overall(
        *,
        decisions,
        interventions,
    ) -> dict[str, Any]:
        outcome_counts = {
            outcome.value: 0
            for outcome
            in InterventionOutcome
        }

        for intervention in (
            interventions
        ):
            outcome_counts[
                intervention[
                    "outcome"
                ]
            ] += 1

        return {
            "runs": len(
                decisions
            ),
            "behavior_valid": sum(
                1
                for decision
                in decisions
                if decision.behavior_valid
            ),
            "task_success": sum(
                1
                for decision
                in decisions
                if (
                    decision.task_success
                    is True
                )
            ),
            "explanation_valid": sum(
                1
                for decision
                in decisions
                if (
                    decision
                    .explanation_valid
                )
            ),
            "intervention_pairs": len(
                interventions
            ),
            "intervention_outcomes": (
                outcome_counts
            ),
        }

    @staticmethod
    def _aggregate_diagnostic(
        interventions,
    ) -> dict[str, Any]:
        rows = [
            row
            for row
            in interventions
            if (
                row[
                    "task_classification"
                ]
                == (
                    TaskClassification
                    .DIAGNOSTIC
                    .value
                )
            )
        ]

        by_source: dict[
            str,
            dict[str, int],
        ] = {}

        for row in rows:
            source = row[
                "source"
            ]

            values = by_source.setdefault(
                source,
                {
                    "positive": 0,
                    "negative": 0,
                    "ambiguous": 0,
                    "unclassifiable": 0,
                    "total": 0,
                },
            )

            values[
                "total"
            ] += 1

            outcome = row[
                "outcome"
            ]

            if (
                outcome
                == InterventionOutcome
                .POSITIVE
                .value
            ):
                values[
                    "positive"
                ] += 1

            elif (
                outcome
                == InterventionOutcome
                .NEGATIVE
                .value
            ):
                values[
                    "negative"
                ] += 1

            elif (
                outcome
                == InterventionOutcome
                .SHAM_UNSTABLE_AMBIGUOUS
                .value
            ):
                values[
                    "ambiguous"
                ] += 1

            elif (
                outcome
                == InterventionOutcome
                .BEHAVIORALLY_UNCLASSIFIABLE
                .value
            ):
                values[
                    "unclassifiable"
                ] += 1

        return {
            "intervention_pairs": len(
                rows
            ),
            "by_source": (
                by_source
            ),
        }

    def _aggregate_h1(
        self,
        *,
        decisions,
        interventions,
    ) -> dict[str, Any]:
        integration = [
            decision
            for decision
            in decisions
            if (
                decision
                .task_classification
                == (
                    TaskClassification
                    .INTEGRATION
                )
                and decision.condition
                in {
                    "C1",
                    "C2",
                    "C3",
                }
            )
        ]

        by_condition: dict[
            str,
            dict[str, Any],
        ] = {}

        for condition in (
            "C1",
            "C2",
            "C3",
        ):
            rows = [
                decision
                for decision
                in integration
                if (
                    decision.condition
                    == condition
                )
            ]

            td_rows = [
                decision
                for decision
                in rows
                if (
                    decision.td_any
                    is not None
                )
            ]

            td_norm_rows = [
                decision.td_norm
                for decision
                in rows
                if (
                    decision.td_norm
                    is not None
                )
            ]

            csd_values = [
                decision
                .csd_scaffold
                for decision
                in td_rows
            ]

            by_condition[
                condition
            ] = {
                "n_runs": len(
                    rows
                ),
                "n_stable_td": len(
                    td_rows
                ),
                "td_any": (
                    self._mean_or_none(
                        [
                            float(
                                decision
                                .td_any
                            )
                            for decision
                            in td_rows
                        ]
                    )
                ),
                "mean_td_norm": (
                    self._mean_or_none(
                        td_norm_rows
                    )
                ),
                "mean_csd_scaffold": (
                    self._mean_or_none(
                        csd_values
                    )
                ),
                "median_csd_scaffold": (
                    self._median_or_none(
                        csd_values
                    )
                ),
                "max_csd_scaffold": (
                    max(
                        csd_values
                    )
                    if csd_values
                    else None
                ),
                "multi_source_positive_rate": (
                    self._mean_or_none(
                        [
                            float(
                                value
                                >= 2
                            )
                            for value
                            in csd_values
                        ]
                    )
                ),
            }

        source_td: dict[
            str,
            dict[
                str,
                float | None,
            ],
        ] = {}

        integration_rows = [
            row
            for row
            in interventions
            if (
                row[
                    "task_classification"
                ]
                == (
                    TaskClassification
                    .INTEGRATION
                    .value
                )
            )
        ]

        for condition in (
            "C1",
            "C2",
            "C3",
        ):
            values: dict[
                str,
                float | None,
            ] = {}

            for source in (
                "T",
                "H",
                "M",
                "E",
            ):
                stable = [
                    row
                    for row
                    in integration_rows
                    if (
                        row[
                            "condition"
                        ]
                        == condition
                        and row[
                            "source"
                        ]
                        == source
                        and row[
                            "outcome"
                        ]
                        in {
                            InterventionOutcome
                            .POSITIVE
                            .value,
                            InterventionOutcome
                            .NEGATIVE
                            .value,
                        }
                    )
                ]

                if not stable:
                    values[
                        source
                    ] = None

                    continue

                positive = sum(
                    1
                    for row
                    in stable
                    if (
                        row[
                            "outcome"
                        ]
                        == (
                            InterventionOutcome
                            .POSITIVE
                            .value
                        )
                    )
                )

                values[
                    source
                ] = (
                    positive
                    / len(
                        stable
                    )
                )

            source_td[
                condition
            ] = values

        return {
            "conditions": (
                by_condition
            ),
            "td_source": (
                source_td
            ),
        }

    def _aggregate_h2(
        self,
        decisions,
    ) -> dict[str, Any]:
        exact = [
            float(
                decision.exact_set
            )
            for decision
            in decisions
            if (
                decision.exact_set
                is not None
            )
        ]

        precision = [
            decision.precision
            for decision
            in decisions
            if (
                decision.precision
                is not None
            )
        ]

        recall = [
            decision.recall
            for decision
            in decisions
            if (
                decision.recall
                is not None
            )
        ]

        f1 = [
            decision.f1
            for decision
            in decisions
            if (
                decision.f1
                is not None
            )
        ]

        return {
            "n_evaluable": len(
                decisions
            ),
            "exact_set_accuracy": (
                self._mean_or_none(
                    exact
                )
            ),
            "macro_precision": (
                self._mean_or_none(
                    precision
                )
            ),
            "macro_recall": (
                self._mean_or_none(
                    recall
                )
            ),
            "macro_f1": (
                self._mean_or_none(
                    f1
                )
            ),
            "median_f1": (
                self._median_or_none(
                    f1
                )
            ),
            "n_precision_defined": len(
                precision
            ),
            "n_recall_defined": len(
                recall
            ),
            "n_f1_defined": len(
                f1
            ),
        }

    def _aggregate_h2_by_csd(
        self,
        decisions,
    ) -> dict[str, Any]:
        groups: dict[
            int,
            list[
                DecisionMetrics
            ],
        ] = {}

        for decision in decisions:
            if (
                not decision
                .h2_evaluable
            ):
                continue

            groups.setdefault(
                decision
                .csd_scaffold,
                [],
            ).append(
                decision
            )

        result: dict[
            str,
            Any,
        ] = {}

        for (
            csd,
            values,
        ) in sorted(
            groups.items()
        ):
            summary = (
                self._aggregate_h2(
                    values
                )
            )

            result[
                str(
                    csd
                )
            ] = summary

        return result

    @staticmethod
    def _aggregate_claims(
        decisions,
    ) -> dict[str, Any]:
        valid = [
            decision
            for decision
            in decisions
            if (
                decision
                .explanation_valid
            )
        ]

        reported_standard_claims = sum(
            1
            for decision
            in valid
            for source
            in decision.reported_sources
            if source
            in STANDARD_SOURCES
        )

        unverifiable_claims = sum(
            len(
                decision
                .unverifiable_sources
            )
            for decision
            in valid
        )

        unavailable_claims = sum(
            len(
                decision
                .unavailable_sources
            )
            for decision
            in valid
        )

        runs_with_unverifiable = sum(
            1
            for decision
            in valid
            if (
                decision
                .unverifiable_sources
            )
        )

        runs_with_unavailable = sum(
            1
            for decision
            in valid
            if (
                decision
                .unavailable_sources
            )
        )

        return {
            "valid_explanation_runs": len(
                valid
            ),
            "reported_standard_claims": (
                reported_standard_claims
            ),
            "unverifiable_claims": (
                unverifiable_claims
            ),
            "unavailable_claims": (
                unavailable_claims
            ),
            "runs_with_unverifiable_claims": (
                runs_with_unverifiable
            ),
            "runs_with_unavailable_claims": (
                runs_with_unavailable
            ),
            "unverifiable_claim_rate": (
                (
                    unverifiable_claims
                    / reported_standard_claims
                )
                if (
                    reported_standard_claims
                    > 0
                )
                else None
            ),
            "unavailable_claim_rate": (
                (
                    unavailable_claims
                    / reported_standard_claims
                )
                if (
                    reported_standard_claims
                    > 0
                )
                else None
            ),
            "run_unavailable_claim_rate": (
                (
                    runs_with_unavailable
                    / len(
                        valid
                    )
                )
                if valid
                else None
            ),
        }

    @staticmethod
    def _aggregate_intervention_sources(
        interventions,
    ) -> dict[str, Any]:
        result: dict[
            str,
            dict[str, int],
        ] = {}

        for row in interventions:
            source = row[
                "source"
            ]

            values = result.setdefault(
                source,
                {
                    "positive": 0,
                    "negative": 0,
                    "ambiguous": 0,
                    "unclassifiable": 0,
                    "total": 0,
                },
            )

            values[
                "total"
            ] += 1

            outcome = row[
                "outcome"
            ]

            if (
                outcome
                == InterventionOutcome
                .POSITIVE
                .value
            ):
                values[
                    "positive"
                ] += 1

            elif (
                outcome
                == InterventionOutcome
                .NEGATIVE
                .value
            ):
                values[
                    "negative"
                ] += 1

            elif (
                outcome
                == InterventionOutcome
                .SHAM_UNSTABLE_AMBIGUOUS
                .value
            ):
                values[
                    "ambiguous"
                ] += 1

            elif (
                outcome
                == InterventionOutcome
                .BEHAVIORALLY_UNCLASSIFIABLE
                .value
            ):
                values[
                    "unclassifiable"
                ] += 1

        return result

    @staticmethod
    def _mean_or_none(
        values,
    ) -> float | None:
        values = list(
            values
        )

        if not values:
            return None

        return float(
            mean(
                values
            )
        )

    @staticmethod
    def _median_or_none(
        values,
    ) -> float | None:
        values = list(
            values
        )

        if not values:
            return None

        return float(
            median(
                values
            )
        )

    @staticmethod
    def _read_json(
        path: Path,
    ) -> dict[str, Any]:
        if not path.is_file():
            raise PilotArtifactError(
                code="artifact_not_found",
                message=(
                    f"Required artifact "
                    f"{str(path)!r} "
                    "does not exist."
                ),
            )

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                value = json.load(
                    handle
                )

        except json.JSONDecodeError as exc:
            raise PilotArtifactError(
                code="invalid_artifact_json",
                message=(
                    f"Artifact "
                    f"{str(path)!r} "
                    "is not valid JSON."
                ),
            ) from exc

        if not isinstance(
            value,
            dict,
        ):
            raise PilotArtifactError(
                code="artifact_root_not_object",
                message=(
                    f"Artifact "
                    f"{str(path)!r} "
                    "must contain a JSON object."
                ),
            )

        return value

    @staticmethod
    def _required_string(
        mapping: Mapping[
            str,
            Any,
        ],
        field_name: str,
    ) -> str:
        value = mapping.get(
            field_name
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value
        ):
            raise PilotArtifactError(
                code="missing_required_field",
                message=(
                    f"Required field "
                    f"{field_name!r} "
                    "must be a "
                    "non-empty string."
                ),
            )

        return value


class PilotAnalysisWriter:
    """
    Write processed pilot datasets and a generated Markdown report.

    Raw pilot artifacts are never modified.
    """

    def write(
        self,
        *,
        analysis: PilotArtifactAnalysis,
        processed_directory: str | Path,
        report_path: str | Path,
        overwrite: bool = False,
    ) -> PilotAnalysisOutputs:
        processed = Path(
            processed_directory
        )

        report = Path(
            report_path
        )

        processed.mkdir(
            parents=True,
            exist_ok=True,
        )

        report.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        decision_path = (
            processed
            / "decision_metrics.csv"
        )

        intervention_path = (
            processed
            / "intervention_metrics.csv"
        )

        metrics_path = (
            processed
            / "pilot_metrics.json"
        )

        targets = (
            decision_path,
            intervention_path,
            metrics_path,
            report,
        )

        if not overwrite:
            existing = [
                str(
                    path
                )
                for path
                in targets
                if path.exists()
            ]

            if existing:
                raise PilotArtifactError(
                    code="analysis_output_exists",
                    message=(
                        "Analysis output already "
                        "exists: "
                        + ", ".join(
                            existing
                        )
                    ),
                )

        self._write_csv(
            decision_path,
            [
                item.to_dict()
                for item
                in (
                    analysis
                    .decision_metrics
                )
            ],
        )

        self._write_csv(
            intervention_path,
            list(
                analysis
                .intervention_metrics
            ),
        )

        self._write_json(
            metrics_path,
            analysis.to_dict(),
        )

        self._write_text(
            report,
            self._render_report(
                analysis
            ),
        )

        return PilotAnalysisOutputs(
            decision_metrics_csv=(
                str(
                    decision_path
                )
            ),
            intervention_metrics_csv=(
                str(
                    intervention_path
                )
            ),
            pilot_metrics_json=(
                str(
                    metrics_path
                )
            ),
            pilot_report_markdown=(
                str(
                    report
                )
            ),
        )

    @staticmethod
    def _write_csv(
        path: Path,
        rows,
    ) -> None:
        rows = list(
            rows
        )

        if not rows:
            with path.open(
                "w",
                encoding="utf-8",
                newline="",
            ):
                pass

            return

        fieldnames = list(
            rows[0].keys()
        )

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(
                                value,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(
                                    ",",
                                    ":",
                                ),
                            )
                            if isinstance(
                                value,
                                (
                                    list,
                                    tuple,
                                    dict,
                                ),
                            )
                            else value
                        )
                        for (
                            key,
                            value
                        )
                        in row.items()
                    }
                )

    @staticmethod
    def _write_json(
        path: Path,
        value,
    ) -> None:
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
                allow_nan=False,
            )

            handle.write(
                "\n"
            )

    @staticmethod
    def _write_text(
        path: Path,
        value: str,
    ) -> None:
        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                value
            )

    def _render_report(
        self,
        analysis: PilotArtifactAnalysis,
    ) -> str:
        overall = (
            analysis
            .aggregates[
                "overall"
            ]
        )

        diagnostic = (
            analysis
            .aggregates[
                "diagnostic_validation"
            ]
        )

        h1 = (
            analysis
            .aggregates[
                "h1_integration"
            ]
        )

        h2_all = (
            analysis
            .aggregates[
                "h2_all_evaluable"
            ]
        )

        h2_integration = (
            analysis
            .aggregates[
                "h2_integration_evaluable"
            ]
        )

        claims = (
            analysis
            .aggregates[
                "source_claim_diagnostics"
            ]
        )

        lines = [
            "# Methodological Pilot Analysis",
            "",
            (
                f"- Experiment: "
                f"`{analysis.experiment_id}`"
            ),
            (
                f"- Task suite: "
                f"`{analysis.suite_id}` "
                f"v{analysis.suite_version}"
            ),
            (
                f"- Execution plan SHA256: "
                f"`{analysis.plan_sha256}`"
            ),
            "",
            (
                "**Interpretation:** This report "
                "describes the methodological pilot. "
                "It is not a confirmatory test of "
                "H1 or H2."
            ),
            "",
            "## Integrity",
            "",
            (
                f"- Completed runs: "
                f"{overall['runs']}"
            ),
            (
                f"- Valid focal actions: "
                f"{overall['behavior_valid']}"
            ),
            (
                f"- Successful task runs: "
                f"{overall['task_success']}"
            ),
            (
                f"- Valid self-explanations: "
                f"{overall['explanation_valid']}"
            ),
            (
                f"- Intervention pairs: "
                f"{overall['intervention_pairs']}"
            ),
            "",
            "### Intervention outcomes",
            "",
            "| Outcome | Count |",
            "|---|---:|",
        ]

        for (
            outcome,
            count,
        ) in (
            overall[
                "intervention_outcomes"
            ]
            .items()
        ):
            lines.append(
                f"| {outcome} | {count} |"
            )

        lines.extend(
            [
                "",
                "## Diagnostic channel validation",
                "",
                "| Source | Positive | Negative | Ambiguous | Unclassifiable | Total |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )

        for source in (
            "T",
            "H",
            "M",
            "E",
            "P",
        ):
            values = (
                diagnostic[
                    "by_source"
                ].get(
                    source
                )
            )

            if values is None:
                continue

            lines.append(
                (
                    f"| {source} "
                    f"| {values['positive']} "
                    f"| {values['negative']} "
                    f"| {values['ambiguous']} "
                    f"| {values['unclassifiable']} "
                    f"| {values['total']} |"
                )
            )

        lines.extend(
            [
                "",
                "## Integration-task trajectory dependence",
                "",
                "| Condition | N | Stable N | TD_any | Mean TD_norm | Mean CSD_scaffold |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )

        for condition in (
            "C1",
            "C2",
            "C3",
        ):
            values = (
                h1[
                    "conditions"
                ][
                    condition
                ]
            )

            lines.append(
                (
                    f"| {condition} "
                    f"| {values['n_runs']} "
                    f"| {values['n_stable_td']} "
                    f"| {self._format_number(values['td_any'])} "
                    f"| {self._format_number(values['mean_td_norm'])} "
                    f"| {self._format_number(values['mean_csd_scaffold'])} |"
                )
            )

        lines.extend(
            [
                "",
                "## Self-explanation faithfulness",
                "",
                "### All evaluable pilot decisions",
                "",
                (
                    f"- N evaluable: "
                    f"{h2_all['n_evaluable']}"
                ),
                (
                    f"- Exact-set accuracy: "
                    f"{self._format_number(h2_all['exact_set_accuracy'])}"
                ),
                (
                    f"- Macro precision: "
                    f"{self._format_number(h2_all['macro_precision'])}"
                ),
                (
                    f"- Macro recall: "
                    f"{self._format_number(h2_all['macro_recall'])}"
                ),
                (
                    f"- Macro F1: "
                    f"{self._format_number(h2_all['macro_f1'])}"
                ),
                "",
                "### Integration-only evaluable decisions",
                "",
                (
                    f"- N evaluable: "
                    f"{h2_integration['n_evaluable']}"
                ),
                (
                    f"- Exact-set accuracy: "
                    f"{self._format_number(h2_integration['exact_set_accuracy'])}"
                ),
                (
                    f"- Macro F1: "
                    f"{self._format_number(h2_integration['macro_f1'])}"
                ),
                "",
                "## Source-claim diagnostics",
                "",
                (
                    f"- Reported standard source claims: "
                    f"{claims['reported_standard_claims']}"
                ),
                (
                    f"- Unverifiable claims: "
                    f"{claims['unverifiable_claims']}"
                ),
                (
                    f"- Unavailable-source claims: "
                    f"{claims['unavailable_claims']}"
                ),
                (
                    f"- Runs with unavailable-source claims: "
                    f"{claims['runs_with_unavailable_claims']}"
                ),
                "",
                "## Methodological disposition",
                "",
                (
                    "Pilot outputs must be reviewed "
                    "for task sensitivity, metric ceiling/floor "
                    "effects, runtime, logging completeness, "
                    "and supplemental technical diagnostics "
                    "before the final experiment is frozen."
                ),
                "",
            ]
        )

        return (
            "\n".join(
                lines
            )
        )

    @staticmethod
    def _format_number(
        value,
    ) -> str:
        if value is None:
            return "NA"

        if isinstance(
            value,
            float,
        ):
            return f"{value:.3f}"

        return str(
            value
        )