__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent import (
    AgentRuntime,
    FocalActionRecord,
    RunIdentity,
    SelfExplanationRecord,
    TaskEnvironmentPolicy,
)
from ..interventions import (
    InterventionClassification,
    InterventionPairValidator,
    MatchedReplayClassifier,
    MatchedReplayExecution,
    MatchedReplayExecutor,
    ValidatedInterventionPair,
)
from ..logging import (
    Condition,
    RawEventWriter,
    SourceClass,
)
from ..model import (
    InferenceConfig,
    ModelAdapter,
)
from ..snapshots import (
    PreFocalSnapshot,
    SnapshotManager,
)
from .resolution import (
    ResolvedTaskContext,
    TaskBackingState,
    TaskBackingStateBuilder,
    TaskFixtureResolver,
    TaskInterventionResolver,
)
from .spec import (
    ExperimentalTaskSpec,
)


class TaskOrchestrationError(RuntimeError):
    """Defined failure in one complete experimental run."""

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
class InterventionRunResult:
    """One resolved, replayed, and classified intervention pair."""

    pair_id: str

    validated_pair: (
        ValidatedInterventionPair
    )

    replay_execution: (
        MatchedReplayExecution
    )

    classification: (
        InterventionClassification
    )

    targeted_events_path: str
    sham_events_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "validated_pair": (
                self.validated_pair
                .to_dict()
            ),
            "replay_execution": (
                self.replay_execution
                .to_dict()
            ),
            "classification": (
                self.classification
                .to_dict()
            ),
            "targeted_events_path": (
                self.targeted_events_path
            ),
            "sham_events_path": (
                self.sham_events_path
            ),
        }


@dataclass(frozen=True)
class ExperimentalRunResult:
    """Complete result of one task × condition × repetition."""

    experiment_id: str
    run_id: str

    task_id: str
    condition: Condition
    repetition_id: int

    output_directory: str

    baseline_events_path: str
    task_spec_path: str
    resolved_context_path: str
    snapshot_path: str
    summary_path: str

    backing_state: TaskBackingState
    resolved_context: ResolvedTaskContext

    snapshot: PreFocalSnapshot

    baseline_action: FocalActionRecord
    self_explanation: SelfExplanationRecord

    behavior_valid: bool
    task_success: bool | None

    expected_action: dict[
        str,
        Any,
    ]

    intervention_results: tuple[
        InterventionRunResult,
        ...
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": (
                self.experiment_id
            ),
            "run_id": self.run_id,
            "task_id": self.task_id,
            "condition": (
                self.condition.value
            ),
            "repetition_id": (
                self.repetition_id
            ),
            "output_directory": (
                self.output_directory
            ),
            "artifacts": {
                "baseline_events": (
                    self.baseline_events_path
                ),
                "task_spec": (
                    self.task_spec_path
                ),
                "resolved_context": (
                    self.resolved_context_path
                ),
                "pre_focal_snapshot": (
                    self.snapshot_path
                ),
                "run_summary": (
                    self.summary_path
                ),
            },
            "snapshot_id": (
                self.snapshot.snapshot_id
            ),
            "snapshot_checksum": (
                self.snapshot
                .state_checksum_sha256
            ),
            "baseline_action": (
                self.baseline_action
                .to_dict()
            ),
            "self_explanation": (
                self.self_explanation
                .to_dict()
            ),
            "behavior_valid": (
                self.behavior_valid
            ),
            "task_success": (
                self.task_success
            ),
            "expected_action": (
                deepcopy(
                    self.expected_action
                )
            ),
            "intervention_results": [
                result.to_dict()
                for result
                in self.intervention_results
            ],
        }


class ExperimentalRunOrchestrator:
    """
    Execute one complete task-condition-repetition experimental run.

    Protocol ordering is fixed:

        baseline state
        -> PRE_FOCAL snapshot
        -> focal action
        -> self-explanation
        -> targeted/sham replay
        -> causal classification

    No intervention or replay is executed before self-explanation.
    """

    def __init__(
        self,
        *,
        snapshot_manager: (
            SnapshotManager | None
        ) = None,
        backing_state_builder: (
            TaskBackingStateBuilder
            | None
        ) = None,
        fixture_resolver: (
            TaskFixtureResolver
            | None
        ) = None,
        intervention_resolver: (
            TaskInterventionResolver
            | None
        ) = None,
        pair_validator: (
            InterventionPairValidator
            | None
        ) = None,
        replay_executor: (
            MatchedReplayExecutor
            | None
        ) = None,
        classifier: (
            MatchedReplayClassifier
            | None
        ) = None,
    ) -> None:
        self.snapshot_manager = (
            snapshot_manager
            or SnapshotManager()
        )

        self.backing_state_builder = (
            backing_state_builder
            or TaskBackingStateBuilder()
        )

        self.fixture_resolver = (
            fixture_resolver
            or TaskFixtureResolver()
        )

        self.intervention_resolver = (
            intervention_resolver
            or TaskInterventionResolver()
        )

        self.pair_validator = (
            pair_validator
            or InterventionPairValidator()
        )

        self.replay_executor = (
            replay_executor
            or MatchedReplayExecutor()
        )

        self.classifier = (
            classifier
            or MatchedReplayClassifier()
        )

    def execute(
        self,
        *,
        experiment_id: str,
        run_id: str,
        task: ExperimentalTaskSpec,
        condition: Condition,
        repetition_id: int,
        model_adapter: ModelAdapter,
        action_config: InferenceConfig,
        explanation_config: InferenceConfig,
        output_directory: str | Path,
    ) -> ExperimentalRunResult:
        self._validate_request(
            experiment_id=experiment_id,
            run_id=run_id,
            task=task,
            condition=condition,
            repetition_id=repetition_id,
            action_config=action_config,
            explanation_config=(
                explanation_config
            ),
        )

        run_directory = Path(
            output_directory
        )

        self._prepare_run_directory(
            run_directory
        )

        baseline_directory = (
            run_directory
            / "baseline"
        )

        state_directory = (
            run_directory
            / "state"
        )

        replay_directory = (
            run_directory
            / "replays"
        )

        baseline_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        state_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        replay_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        baseline_events_path = (
            baseline_directory
            / "events.jsonl"
        )

        runtime = AgentRuntime(
            identity=RunIdentity(
                experiment_id=(
                    experiment_id
                ),
                run_id=run_id,
                task_id=(
                    task.task_id
                ),
                task_classification=(
                    task.task_classification
                ),
                condition=condition,
                repetition_id=(
                    repetition_id
                ),
                seed=(
                    action_config.seed
                ),
            ),
            model_adapter=(
                model_adapter
            ),
            event_writer=RawEventWriter(
                baseline_events_path
            ),
        )

        runtime.start()

        runtime.begin_focal_task(
            task_prompt=(
                task.task_prompt
            ),
            action_schema=(
                task
                .focal_decision
                .schema
            ),
        )

        backing_state = (
            self.backing_state_builder
            .build(
                task=task,
                run_id=run_id,
                work_directory=(
                    state_directory
                ),
            )
        )

        resolved_context = (
            self.fixture_resolver
            .resolve(
                task=task,
                condition=condition,
                runtime=runtime,
                backing_state=(
                    backing_state
                ),
            )
        )

        prompt_element = (
            runtime.prompt_element
        )

        if prompt_element is None:
            raise TaskOrchestrationError(
                code="prompt_unavailable",
                message=(
                    "Runtime did not expose "
                    "canonical P before snapshot."
                ),
            )

        available_sources = set(
            task.available_sources[
                condition
            ]
        )

        snapshot_memory_store = (
            backing_state.memory_store
            if (
                SourceClass.M
                in available_sources
            )
            else None
        )

        snapshot_memory_namespace = (
            backing_state.memory_namespace
            if (
                SourceClass.M
                in available_sources
            )
            else None
        )

        snapshot_environment = (
            backing_state.environment
            if (
                SourceClass.E
                in available_sources
            )
            else None
        )

        snapshot = (
            self.snapshot_manager
            .capture(
                experiment_id=(
                    experiment_id
                ),
                run_id=run_id,
                task_id=(
                    task.task_id
                ),
                condition=condition,
                repetition_id=(
                    repetition_id
                ),
                sequence_position=(
                    runtime.events[-1]
                    .sequence_index
                ),
                prompt_elements=(
                    prompt_element,
                ),
                tool_elements=(
                    resolved_context
                    .tool_elements
                ),
                history_elements=(
                    resolved_context
                    .history_elements
                ),
                memory_store=(
                    snapshot_memory_store
                ),
                memory_namespace=(
                    snapshot_memory_namespace
                ),
                environment=(
                    snapshot_environment
                ),
                runtime_configuration={
                    "action_inference": (
                        action_config
                        .to_dict()
                    ),
                    "explanation_inference": (
                        explanation_config
                        .to_dict()
                    ),
                },
                task_control_state={
                    "task_version": (
                        task.task_version
                    ),
                    "focal_decision_id": (
                        task
                        .focal_decision
                        .schema
                        .focal_decision_id
                    ),
                    "available_sources": [
                        source.value
                        for source
                        in (
                            task
                            .available_sources[
                                condition
                            ]
                        )
                    ],
                    "fixture_targets": {
                        fixture_id: (
                            target.to_dict()
                        )
                        for (
                            fixture_id,
                            target
                        )
                        in (
                            resolved_context
                            .fixture_targets
                            .items()
                        )
                    },
                },
                seed_metadata={
                    "action_seed": (
                        action_config.seed
                    ),
                    "explanation_seed": (
                        explanation_config.seed
                    ),
                },
            )
        )

        environment_policy = (
            self._environment_policy(
                resolved_context
                .environment_elements
            )
        )

        # --------------------------------------------------------------
        # BASELINE FOCAL ACTION
        # --------------------------------------------------------------

        baseline_action = (
            runtime.invoke_focal_action(
                config=(
                    action_config
                ),
                tool_elements=(
                    resolved_context
                    .tool_elements
                ),
                history_elements=(
                    resolved_context
                    .history_elements
                ),
                memory_elements=(
                    resolved_context
                    .memory_elements
                ),
                environment_elements=(
                    resolved_context
                    .environment_elements
                ),
                environment_policy=(
                    environment_policy
                ),
            )
        )

        # --------------------------------------------------------------
        # SELF-EXPLANATION
        #
        # This MUST occur before any intervention or replay.
        # --------------------------------------------------------------

        self_explanation = (
            runtime
            .elicit_self_explanation(
                focal_action=(
                    baseline_action
                ),
                config=(
                    explanation_config
                ),
            )
        )

        expected_action = deepcopy(
            task
            .focal_decision
            .expected_by_condition[
                condition
            ]
        )

        (
            behavior_valid,
            task_success,
        ) = self._score_behavior(
            baseline_action=(
                baseline_action
            ),
            expected_action=(
                expected_action
            ),
        )

        # --------------------------------------------------------------
        # TARGETED + SHAM REPLAYS
        #
        # No code above this point exposes intervention results to the
        # baseline model or self-explanation invocation.
        # --------------------------------------------------------------

        intervention_results: list[
            InterventionRunResult
        ] = []

        for template in (
            task.intervention_pairs
        ):
            if (
                condition
                not in template
                .eligible_conditions
            ):
                continue

            pair = (
                self.intervention_resolver
                .resolve_pair(
                    task=task,
                    condition=condition,
                    template=template,
                    context=(
                        resolved_context
                    ),
                )
            )

            validated_pair = (
                self.pair_validator
                .validate(
                    snapshot=snapshot,
                    pair=pair,
                )
            )

            pair_directory = (
                replay_directory
                / template.pair_id
            )

            pair_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            targeted_directory = (
                pair_directory
                / "targeted"
            )

            sham_directory = (
                pair_directory
                / "sham"
            )

            targeted_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            sham_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            targeted_events_path = (
                targeted_directory
                / "events.jsonl"
            )

            sham_events_path = (
                sham_directory
                / "events.jsonl"
            )

            replay_execution = (
                self.replay_executor
                .execute_pair(
                    snapshot=snapshot,
                    baseline_action=(
                        baseline_action
                    ),
                    targeted_application=(
                        validated_pair
                        .targeted_application
                    ),
                    sham_application=(
                        validated_pair
                        .sham_application
                    ),
                    action_schema=(
                        task
                        .focal_decision
                        .schema
                    ),
                    task_classification=(
                        task
                        .task_classification
                    ),
                    model_adapter=(
                        model_adapter
                    ),
                    config=(
                        action_config
                    ),
                    targeted_event_writer=(
                        RawEventWriter(
                            targeted_events_path
                        )
                    ),
                    sham_event_writer=(
                        RawEventWriter(
                            sham_events_path
                        )
                    ),
                )
            )

            classification = (
                self.classifier
                .classify(
                    replay_execution
                )
            )

            intervention_results.append(
                InterventionRunResult(
                    pair_id=(
                        template.pair_id
                    ),
                    validated_pair=(
                        validated_pair
                    ),
                    replay_execution=(
                        replay_execution
                    ),
                    classification=(
                        classification
                    ),
                    targeted_events_path=(
                        str(
                            targeted_events_path
                        )
                    ),
                    sham_events_path=(
                        str(
                            sham_events_path
                        )
                    ),
                )
            )

        task_spec_path = (
            run_directory
            / "task_spec.json"
        )

        resolved_context_path = (
            run_directory
            / "resolved_context.json"
        )

        snapshot_path = (
            run_directory
            / "pre_focal_snapshot.json"
        )

        summary_path = (
            run_directory
            / "run_summary.json"
        )

        self._write_json(
            task_spec_path,
            task.to_dict(),
        )

        self._write_json(
            resolved_context_path,
            resolved_context.to_dict(),
        )

        self._write_json(
            snapshot_path,
            snapshot.to_dict(),
        )

        result = ExperimentalRunResult(
            experiment_id=(
                experiment_id
            ),
            run_id=run_id,
            task_id=(
                task.task_id
            ),
            condition=condition,
            repetition_id=(
                repetition_id
            ),
            output_directory=(
                str(
                    run_directory
                )
            ),
            baseline_events_path=(
                str(
                    baseline_events_path
                )
            ),
            task_spec_path=(
                str(
                    task_spec_path
                )
            ),
            resolved_context_path=(
                str(
                    resolved_context_path
                )
            ),
            snapshot_path=(
                str(
                    snapshot_path
                )
            ),
            summary_path=(
                str(
                    summary_path
                )
            ),
            backing_state=(
                backing_state
            ),
            resolved_context=(
                resolved_context
            ),
            snapshot=snapshot,
            baseline_action=(
                baseline_action
            ),
            self_explanation=(
                self_explanation
            ),
            behavior_valid=(
                behavior_valid
            ),
            task_success=(
                task_success
            ),
            expected_action=(
                expected_action
            ),
            intervention_results=tuple(
                intervention_results
            ),
        )

        self._write_json(
            summary_path,
            result.to_dict(),
        )

        return result

    @staticmethod
    def _score_behavior(
        *,
        baseline_action: FocalActionRecord,
        expected_action: dict[
            str,
            Any,
        ],
    ) -> tuple[
        bool,
        bool | None,
    ]:
        if (
            not baseline_action.valid
            or baseline_action.parse_result
            is None
        ):
            # Preserve malformed/unmappable model behavior separately
            # rather than silently calling it a valid incorrect action.
            return (
                False,
                None,
            )

        observed = (
            baseline_action
            .parse_result
            .normalized_action
            .payload
        )

        return (
            True,
            observed
            == expected_action,
        )

    @staticmethod
    def _environment_policy(
        elements,
    ) -> TaskEnvironmentPolicy | None:
        if not elements:
            return None

        return TaskEnvironmentPolicy(
            allow_environment=True,
            allowed_source_ids=frozenset(
                element.source_id
                for element
                in elements
            ),
        )

    @staticmethod
    def _prepare_run_directory(
        path: Path,
    ) -> None:
        if path.exists():
            raise TaskOrchestrationError(
                code="run_directory_exists",
                message=(
                    f"Run output directory "
                    f"{str(path)!r} already exists. "
                    "Experimental runs are append-only "
                    "and may not overwrite prior output."
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
            raise TaskOrchestrationError(
                code="artifact_already_exists",
                message=(
                    f"Artifact {str(path)!r} "
                    "already exists."
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
    def _validate_request(
        *,
        experiment_id: str,
        run_id: str,
        task: ExperimentalTaskSpec,
        condition: Condition,
        repetition_id: int,
        action_config: InferenceConfig,
        explanation_config: InferenceConfig,
    ) -> None:
        for field_name, value in (
            (
                "experiment_id",
                experiment_id,
            ),
            (
                "run_id",
                run_id,
            ),
        ):
            if (
                not isinstance(
                    value,
                    str,
                )
                or not value
            ):
                raise TaskOrchestrationError(
                    code="invalid_run_identity",
                    message=(
                        f"{field_name} must "
                        "be a non-empty string."
                    ),
                )

        if (
            condition
            not in task.conditions
        ):
            raise TaskOrchestrationError(
                code="condition_not_in_task",
                message=(
                    f"Task {task.task_id!r} "
                    f"does not define "
                    f"{condition.value}."
                ),
            )

        if (
            not isinstance(
                repetition_id,
                int,
            )
            or repetition_id < 0
        ):
            raise TaskOrchestrationError(
                code="invalid_repetition_id",
                message=(
                    "repetition_id must be "
                    "a non-negative integer."
                ),
            )

        if action_config.seed is None:
            raise TaskOrchestrationError(
                code="missing_action_seed",
                message=(
                    "Pilot focal-action inference "
                    "must use an explicit seed."
                ),
            )

        if explanation_config.seed is None:
            raise TaskOrchestrationError(
                code="missing_explanation_seed",
                message=(
                    "Pilot self-explanation inference "
                    "must use an explicit seed."
                ),
            )