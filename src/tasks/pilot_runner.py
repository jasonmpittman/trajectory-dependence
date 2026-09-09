__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..interventions import (
    InterventionOutcome,
)
from ..model import (
    ModelAdapter,
)
from .orchestrator import (
    ExperimentalRunOrchestrator,
    ExperimentalRunResult,
)
from .pilot import (
    PilotConfiguration,
    PilotExecutionPlan,
    PilotPreflightResult,
    PilotPreflightValidator,
)
from .spec import (
    TaskSuite,
)


class PilotExecutionError(RuntimeError):
    """Defined failure during suite-level pilot execution."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message
        )

        self.code = code
        self.message = message
        self.details = deepcopy(
            details or {}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": deepcopy(
                self.details
            ),
        }


@dataclass(frozen=True)
class PilotRunSummary:
    """Concise aggregate result for one completed planned run."""

    ordinal: int

    run_id: str
    task_id: str
    condition: str
    repetition_id: int

    behavior_valid: bool
    task_success: bool | None

    normalized_action: (
        dict[str, Any] | None
    )

    explanation_valid: bool

    reported_sources: tuple[
        str,
        ...
    ]

    intervention_results: tuple[
        dict[str, Any],
        ...
    ]

    run_summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "condition": (
                self.condition
            ),
            "repetition_id": (
                self.repetition_id
            ),
            "behavior_valid": (
                self.behavior_valid
            ),
            "task_success": (
                self.task_success
            ),
            "normalized_action": (
                deepcopy(
                    self.normalized_action
                )
            ),
            "explanation_valid": (
                self.explanation_valid
            ),
            "reported_sources": list(
                self.reported_sources
            ),
            "intervention_results": [
                deepcopy(
                    result
                )
                for result
                in self.intervention_results
            ],
            "run_summary_path": (
                self.run_summary_path
            ),
        }


@dataclass(frozen=True)
class PilotExecutionSummary:
    """Aggregate methodological summary for one completed pilot."""

    experiment_id: str

    suite_id: str
    suite_version: str

    plan_sha256: str

    model_runtime_metadata: dict[
        str,
        Any,
    ]

    planned_runs: int
    completed_runs: int

    valid_behavior_runs: int
    invalid_behavior_runs: int

    successful_task_runs: int
    unsuccessful_task_runs: int
    unscored_task_runs: int

    valid_explanations: int
    invalid_explanations: int

    intervention_outcome_counts: dict[
        str,
        int,
    ]

    runs: tuple[
        PilotRunSummary,
        ...
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "model_runtime_metadata": (
                deepcopy(
                    self.model_runtime_metadata
                )
            ),
            "planned_runs": (
                self.planned_runs
            ),
            "completed_runs": (
                self.completed_runs
            ),
            "valid_behavior_runs": (
                self.valid_behavior_runs
            ),
            "invalid_behavior_runs": (
                self.invalid_behavior_runs
            ),
            "successful_task_runs": (
                self.successful_task_runs
            ),
            "unsuccessful_task_runs": (
                self.unsuccessful_task_runs
            ),
            "unscored_task_runs": (
                self.unscored_task_runs
            ),
            "valid_explanations": (
                self.valid_explanations
            ),
            "invalid_explanations": (
                self.invalid_explanations
            ),
            "intervention_outcome_counts": (
                deepcopy(
                    self.intervention_outcome_counts
                )
            ),
            "runs": [
                run.to_dict()
                for run
                in self.runs
            ],
        }


class PilotSuiteRunner:
    """
    Execute a previously validated PilotExecutionPlan.

    One already-loaded ModelAdapter is reused across every baseline,
    self-explanation, targeted replay, and sham replay.

    Execution is fail-closed:

    - preflight runs before any execution artifacts are created;
    - the experiment root may not already exist;
    - aggregate artifacts use exclusive creation;
    - a technical/orchestration exception writes a failure record;
    - completed run directories are never overwritten.
    """

    def __init__(
        self,
        *,
        preflight_validator: (
            PilotPreflightValidator
            | None
        ) = None,
        run_orchestrator: (
            ExperimentalRunOrchestrator
            | None
        ) = None,
    ) -> None:
        self.preflight_validator = (
            preflight_validator
            or PilotPreflightValidator()
        )

        self.run_orchestrator = (
            run_orchestrator
            or ExperimentalRunOrchestrator()
        )

    def execute(
        self,
        *,
        config: PilotConfiguration,
        suite: TaskSuite,
        plan: PilotExecutionPlan,
        model_adapter: ModelAdapter,
    ) -> PilotExecutionSummary:
        if not model_adapter.is_loaded:
            raise PilotExecutionError(
                code="model_not_loaded",
                message=(
                    "Pilot execution requires an "
                    "already-loaded model adapter."
                ),
            )

        # --------------------------------------------------------------
        # MODEL-FREE PREFLIGHT
        #
        # This occurs before the execution root is created.
        # --------------------------------------------------------------

        preflight = (
            self.preflight_validator
            .validate(
                config=config,
                suite=suite,
                plan=plan,
            )
        )

        runtime_metadata = (
            model_adapter
            .runtime_metadata()
            .to_dict()
        )

        self._validate_model_identity(
            config=config,
            runtime_metadata=(
                runtime_metadata
            ),
        )

        experiment_root = (
            Path(
                config.output_root
            )
            / config.experiment_id
        )

        self._prepare_experiment_root(
            experiment_root
        )

        config_path = (
            experiment_root
            / "pilot_config.json"
        )

        suite_path = (
            experiment_root
            / "task_suite.json"
        )

        plan_path = (
            experiment_root
            / "execution_plan.json"
        )

        preflight_path = (
            experiment_root
            / "preflight.json"
        )

        runtime_path = (
            experiment_root
            / "model_runtime.json"
        )

        summary_path = (
            experiment_root
            / "pilot_summary.json"
        )

        partial_summary_path = (
            experiment_root
            / "pilot_partial_summary.json"
        )

        failure_path = (
            experiment_root
            / "pilot_failure.json"
        )

        # --------------------------------------------------------------
        # IMMUTABLE EXECUTION BUNDLE
        # --------------------------------------------------------------

        self._write_json(
            config_path,
            config.to_dict(),
        )

        self._write_json(
            suite_path,
            suite.to_dict(),
        )

        self._write_json(
            plan_path,
            plan.to_dict(),
        )

        self._write_json(
            preflight_path,
            preflight.to_dict(),
        )

        self._write_json(
            runtime_path,
            runtime_metadata,
        )

        task_map = {
            task.task_id: task
            for task
            in suite.tasks
        }

        completed: list[
            PilotRunSummary
        ] = []

        current_run = None

        try:
            for run in plan.runs:
                current_run = run

                task = task_map.get(
                    run.task_id
                )

                if task is None:
                    raise PilotExecutionError(
                        code="planned_task_not_found",
                        message=(
                            f"Planned task "
                            f"{run.task_id!r} "
                            "does not exist in "
                            "the task suite."
                        ),
                    )

                action_config = (
                    run.action_config(
                        max_tokens=(
                            config
                            .action_max_tokens
                        )
                    )
                )

                explanation_config = (
                    run.explanation_config(
                        max_tokens=(
                            config
                            .explanation_max_tokens
                        )
                    )
                )

                result = (
                    self.run_orchestrator
                    .execute(
                        experiment_id=(
                            config
                            .experiment_id
                        ),
                        run_id=(
                            run.run_id
                        ),
                        task=task,
                        condition=(
                            run.condition
                        ),
                        repetition_id=(
                            run
                            .repetition_id
                        ),
                        model_adapter=(
                            model_adapter
                        ),
                        action_config=(
                            action_config
                        ),
                        explanation_config=(
                            explanation_config
                        ),
                        output_directory=(
                            run
                            .output_directory
                        ),
                    )
                )

                completed.append(
                    self._summarize_run(
                        ordinal=(
                            run.ordinal
                        ),
                        result=result,
                    )
                )

        except Exception as exc:
            partial = (
                self._build_summary(
                    config=config,
                    suite=suite,
                    plan=plan,
                    runtime_metadata=(
                        runtime_metadata
                    ),
                    completed=tuple(
                        completed
                    ),
                )
            )

            self._write_json(
                partial_summary_path,
                partial.to_dict(),
            )

            failure_payload = {
                "experiment_id": (
                    config.experiment_id
                ),
                "plan_sha256": (
                    plan.plan_sha256
                ),
                "completed_runs": len(
                    completed
                ),
                "failed_run": (
                    current_run.to_dict()
                    if (
                        current_run
                        is not None
                    )
                    else None
                ),
                "error": (
                    self._error_details(
                        exc
                    )
                ),
            }

            self._write_json(
                failure_path,
                failure_payload,
            )

            if isinstance(
                exc,
                PilotExecutionError,
            ):
                raise

            raise PilotExecutionError(
                code="pilot_execution_failure",
                message=(
                    "Pilot execution stopped after "
                    "an experimental run failed."
                ),
                details=(
                    failure_payload
                ),
            ) from exc

        summary = (
            self._build_summary(
                config=config,
                suite=suite,
                plan=plan,
                runtime_metadata=(
                    runtime_metadata
                ),
                completed=tuple(
                    completed
                ),
            )
        )

        if (
            summary.completed_runs
            != summary.planned_runs
        ):
            raise PilotExecutionError(
                code="pilot_incomplete",
                message=(
                    "Pilot execution returned without "
                    "completing every planned run."
                ),
                details={
                    "planned_runs": (
                        summary.planned_runs
                    ),
                    "completed_runs": (
                        summary.completed_runs
                    ),
                },
            )

        self._write_json(
            summary_path,
            summary.to_dict(),
        )

        return summary

    @staticmethod
    def _summarize_run(
        *,
        ordinal: int,
        result: ExperimentalRunResult,
    ) -> PilotRunSummary:
        normalized_action = None

        if (
            result.baseline_action.valid
            and result
            .baseline_action
            .parse_result
            is not None
        ):
            normalized_action = deepcopy(
                result
                .baseline_action
                .parse_result
                .normalized_action
                .payload
            )

        reported_sources: tuple[
            str,
            ...
        ] = ()

        if (
            result.self_explanation.valid
            and result
            .self_explanation
            .parse_result
            is not None
        ):
            reported_sources = tuple(
                source.value
                for source
                in (
                    result
                    .self_explanation
                    .parse_result
                    .reported_sources
                )
            )

        interventions = tuple(
            {
                "pair_id": (
                    intervention
                    .pair_id
                ),
                "source": (
                    intervention
                    .classification
                    .source
                    .value
                ),
                "outcome": (
                    intervention
                    .classification
                    .outcome
                    .value
                ),
                "targeted_changed": (
                    intervention
                    .classification
                    .targeted_changed
                ),
                "sham_changed": (
                    intervention
                    .classification
                    .sham_changed
                ),
                "targeted_events_path": (
                    intervention
                    .targeted_events_path
                ),
                "sham_events_path": (
                    intervention
                    .sham_events_path
                ),
            }
            for intervention
            in result.intervention_results
        )

        return PilotRunSummary(
            ordinal=ordinal,
            run_id=(
                result.run_id
            ),
            task_id=(
                result.task_id
            ),
            condition=(
                result.condition.value
            ),
            repetition_id=(
                result.repetition_id
            ),
            behavior_valid=(
                result.behavior_valid
            ),
            task_success=(
                result.task_success
            ),
            normalized_action=(
                normalized_action
            ),
            explanation_valid=(
                result
                .self_explanation
                .valid
            ),
            reported_sources=(
                reported_sources
            ),
            intervention_results=(
                interventions
            ),
            run_summary_path=(
                result.summary_path
            ),
        )

    @staticmethod
    def _build_summary(
        *,
        config: PilotConfiguration,
        suite: TaskSuite,
        plan: PilotExecutionPlan,
        runtime_metadata: dict[
            str,
            Any,
        ],
        completed: tuple[
            PilotRunSummary,
            ...
        ],
    ) -> PilotExecutionSummary:
        valid_behavior_runs = sum(
            1
            for run
            in completed
            if run.behavior_valid
        )

        invalid_behavior_runs = (
            len(
                completed
            )
            - valid_behavior_runs
        )

        successful_task_runs = sum(
            1
            for run
            in completed
            if run.task_success
            is True
        )

        unsuccessful_task_runs = sum(
            1
            for run
            in completed
            if run.task_success
            is False
        )

        unscored_task_runs = sum(
            1
            for run
            in completed
            if run.task_success
            is None
        )

        valid_explanations = sum(
            1
            for run
            in completed
            if run.explanation_valid
        )

        invalid_explanations = (
            len(
                completed
            )
            - valid_explanations
        )

        outcome_counts = {
            outcome.value: 0
            for outcome
            in InterventionOutcome
        }

        for run in completed:
            for intervention in (
                run.intervention_results
            ):
                outcome = intervention[
                    "outcome"
                ]

                if (
                    outcome
                    not in outcome_counts
                ):
                    outcome_counts[
                        outcome
                    ] = 0

                outcome_counts[
                    outcome
                ] += 1

        return PilotExecutionSummary(
            experiment_id=(
                config.experiment_id
            ),
            suite_id=(
                suite.suite_id
            ),
            suite_version=(
                suite.suite_version
            ),
            plan_sha256=(
                plan.plan_sha256
            ),
            model_runtime_metadata=(
                deepcopy(
                    runtime_metadata
                )
            ),
            planned_runs=len(
                plan.runs
            ),
            completed_runs=len(
                completed
            ),
            valid_behavior_runs=(
                valid_behavior_runs
            ),
            invalid_behavior_runs=(
                invalid_behavior_runs
            ),
            successful_task_runs=(
                successful_task_runs
            ),
            unsuccessful_task_runs=(
                unsuccessful_task_runs
            ),
            unscored_task_runs=(
                unscored_task_runs
            ),
            valid_explanations=(
                valid_explanations
            ),
            invalid_explanations=(
                invalid_explanations
            ),
            intervention_outcome_counts=(
                outcome_counts
            ),
            runs=completed,
        )

    @staticmethod
    def _validate_model_identity(
        *,
        config: PilotConfiguration,
        runtime_metadata: dict[
            str,
            Any,
        ],
    ) -> None:
        checks = (
            (
                "model_identifier",
                config.model.identifier,
            ),
            (
                "quantization",
                config.model.quantization,
            ),
            (
                "model_path_env",
                config
                .model
                .path_environment_variable,
            ),
        )

        for (
            runtime_field,
            expected,
        ) in checks:
            observed = (
                runtime_metadata.get(
                    runtime_field
                )
            )

            if observed != expected:
                raise PilotExecutionError(
                    code="model_identity_mismatch",
                    message=(
                        f"Runtime model field "
                        f"{runtime_field!r} does "
                        "not match pilot configuration."
                    ),
                    details={
                        "field": (
                            runtime_field
                        ),
                        "expected": (
                            expected
                        ),
                        "observed": (
                            observed
                        ),
                    },
                )

        optional_checks = (
            (
                "model_revision",
                config.model.revision,
            ),
            (
                "model_checksum",
                config.model.checksum,
            ),
        )

        for (
            runtime_field,
            expected,
        ) in optional_checks:
            if expected is None:
                continue

            observed = (
                runtime_metadata.get(
                    runtime_field
                )
            )

            if observed != expected:
                raise PilotExecutionError(
                    code="model_identity_mismatch",
                    message=(
                        f"Runtime model field "
                        f"{runtime_field!r} does "
                        "not match pilot configuration."
                    ),
                    details={
                        "field": (
                            runtime_field
                        ),
                        "expected": (
                            expected
                        ),
                        "observed": (
                            observed
                        ),
                    },
                )

    @staticmethod
    def _prepare_experiment_root(
        path: Path,
    ) -> None:
        if path.exists():
            raise PilotExecutionError(
                code="experiment_root_exists",
                message=(
                    f"Pilot experiment root "
                    f"{str(path)!r} already exists. "
                    "Pilot execution will not overwrite "
                    "prior experimental artifacts."
                ),
            )

        path.mkdir(
            parents=True,
            exist_ok=False,
        )

    @staticmethod
    def _write_json(
        path: Path,
        value: Any,
    ) -> None:
        if path.exists():
            raise PilotExecutionError(
                code="aggregate_artifact_exists",
                message=(
                    f"Aggregate artifact "
                    f"{str(path)!r} already exists."
                ),
            )

        with path.open(
            "x",
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
    def _error_details(
        error: Exception,
    ) -> dict[str, Any]:
        if hasattr(
            error,
            "to_dict",
        ):
            try:
                value = (
                    error.to_dict()
                )

                if isinstance(
                    value,
                    dict,
                ):
                    return deepcopy(
                        value
                    )

            except Exception:
                pass

        return {
            "exception_type": (
                type(error).__name__
            ),
            "message": str(
                error
            ),
        }