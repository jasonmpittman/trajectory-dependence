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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..agent import (
    AgentRuntime,
    RunIdentity,
    ScaffoldController,
)
from ..environment import (
    DeterministicTransitionEngine,
    EnvironmentState,
    TransitionRule,
)
from ..logging import (
    Condition,
    EventType,
    RawEventWriter,
    TaskClassification,
)
from ..memory import (
    SQLiteMemoryStore,
)
from ..model import (
    InferenceConfig,
    ModelAdapter,
)
from ..normalization import (
    ActionField,
    FocalActionSchema,
)
from ..snapshots import (
    SnapshotManager,
)


SUPPLEMENTAL_STUDY_ROLE = (
    "supplemental_technical_diagnostic"
)

SUPPLEMENTAL_EXPERIMENT_ID = (
    "pilot_technical_diagnostics"
)

MEMORY_TASK_ID = (
    "supplemental_memory_write_read"
)

ENVIRONMENT_TASK_ID = (
    "supplemental_environment_transition_query"
)


class SupplementalDiagnosticError(
    RuntimeError
):
    """Defined failure in a supplemental technical diagnostic."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: dict[
            str,
            Any,
        ] | None = None,
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
class SupplementalDiagnosticResult:
    """Result of one post-pilot technical diagnostic."""

    diagnostic_id: str
    run_id: str
    task_id: str
    condition: str

    study_role: str

    post_pilot: bool
    included_in_h1: bool
    included_in_h2: bool

    action_valid: bool
    task_success: bool

    normalized_action: (
        dict[str, Any] | None
    )

    explanation_valid: bool

    reported_sources: tuple[
        str,
        ...
    ]

    technical_failure_count: int

    checks: dict[
        str,
        bool,
    ]

    passed: bool

    output_directory: str
    events_path: str
    snapshot_path: str
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnostic_id": (
                self.diagnostic_id
            ),
            "run_id": self.run_id,
            "task_id": self.task_id,
            "condition": (
                self.condition
            ),
            "study_role": (
                self.study_role
            ),
            "post_pilot": (
                self.post_pilot
            ),
            "included_in_h1": (
                self.included_in_h1
            ),
            "included_in_h2": (
                self.included_in_h2
            ),
            "action_valid": (
                self.action_valid
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
            "technical_failure_count": (
                self.technical_failure_count
            ),
            "checks": deepcopy(
                self.checks
            ),
            "passed": self.passed,
            "output_directory": (
                self.output_directory
            ),
            "events_path": (
                self.events_path
            ),
            "snapshot_path": (
                self.snapshot_path
            ),
            "summary_path": (
                self.summary_path
            ),
        }


@dataclass(frozen=True)
class SupplementalDiagnosticSuiteResult:
    """Aggregate result for the two supplemental diagnostics."""

    experiment_id: str
    study_role: str

    post_pilot: bool
    included_in_h1: bool
    included_in_h2: bool

    diagnostics: tuple[
        SupplementalDiagnosticResult,
        ...
    ]

    passed: bool

    output_root: str
    manifest_path: str
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": (
                self.experiment_id
            ),
            "study_role": (
                self.study_role
            ),
            "post_pilot": (
                self.post_pilot
            ),
            "included_in_h1": (
                self.included_in_h1
            ),
            "included_in_h2": (
                self.included_in_h2
            ),
            "diagnostics": [
                result.to_dict()
                for result
                in self.diagnostics
            ],
            "passed": self.passed,
            "output_root": (
                self.output_root
            ),
            "manifest_path": (
                self.manifest_path
            ),
            "summary_path": (
                self.summary_path
            ),
        }


class SupplementalDiagnosticRunner:
    """
    Execute post-pilot technical diagnostics outside the H1/H2 dataset.

    The runner intentionally performs no targeted/sham intervention.
    These runs test dynamic state lifecycles only.
    """

    def execute(
        self,
        *,
        model_adapter: ModelAdapter,
        action_config: InferenceConfig,
        explanation_config: InferenceConfig,
        output_root: str | Path,
    ) -> SupplementalDiagnosticSuiteResult:
        if not model_adapter.is_loaded:
            raise SupplementalDiagnosticError(
                code="model_not_loaded",
                message=(
                    "Supplemental diagnostics require "
                    "an already-loaded model adapter."
                ),
            )

        root = Path(
            output_root
        )

        if root.exists():
            raise SupplementalDiagnosticError(
                code="output_root_exists",
                message=(
                    f"Supplemental diagnostic root "
                    f"{str(root)!r} already exists. "
                    "Existing evidence will not be overwritten."
                ),
            )

        root.mkdir(
            parents=True,
            exist_ok=False,
        )

        manifest_path = (
            root
            / "diagnostic_manifest.json"
        )

        summary_path = (
            root
            / "supplemental_summary.json"
        )

        manifest = {
            "schema_version": (
                "supplemental-diagnostic-v1"
            ),
            "experiment_id": (
                SUPPLEMENTAL_EXPERIMENT_ID
            ),
            "study_role": (
                SUPPLEMENTAL_STUDY_ROLE
            ),
            "post_pilot": True,
            "included_in_h1": False,
            "included_in_h2": False,
            "purpose": (
                "Exercise dynamic persistent-memory "
                "write/read and deterministic environment "
                "transition/query lifecycles not exercised "
                "end-to-end by the six-task methodological pilot."
            ),
            "model_runtime_metadata": (
                model_adapter
                .runtime_metadata()
                .to_dict()
            ),
            "action_inference_config": (
                asdict(
                    action_config
                )
            ),
            "explanation_inference_config": (
                asdict(
                    explanation_config
                )
            ),
            "diagnostics": [
                {
                    "diagnostic_id": (
                        "memory_write_read"
                    ),
                    "task_id": (
                        MEMORY_TASK_ID
                    ),
                    "condition": (
                        Condition.C3.value
                    ),
                },
                {
                    "diagnostic_id": (
                        "environment_transition_query"
                    ),
                    "task_id": (
                        ENVIRONMENT_TASK_ID
                    ),
                    "condition": (
                        Condition.C1.value
                    ),
                },
            ],
        }

        self._write_json_exclusive(
            manifest_path,
            manifest,
        )

        memory_result = (
            self._execute_memory(
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
                    root
                    / "memory_write_read"
                ),
            )
        )

        environment_result = (
            self._execute_environment(
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
                    root
                    / "environment_transition_query"
                ),
            )
        )

        diagnostics = (
            memory_result,
            environment_result,
        )

        result = (
            SupplementalDiagnosticSuiteResult(
                experiment_id=(
                    SUPPLEMENTAL_EXPERIMENT_ID
                ),
                study_role=(
                    SUPPLEMENTAL_STUDY_ROLE
                ),
                post_pilot=True,
                included_in_h1=False,
                included_in_h2=False,
                diagnostics=(
                    diagnostics
                ),
                passed=all(
                    item.passed
                    for item
                    in diagnostics
                ),
                output_root=str(
                    root
                ),
                manifest_path=str(
                    manifest_path
                ),
                summary_path=str(
                    summary_path
                ),
            )
        )

        self._write_json_exclusive(
            summary_path,
            result.to_dict(),
        )

        return result

    def _execute_memory(
        self,
        *,
        model_adapter: ModelAdapter,
        action_config: InferenceConfig,
        explanation_config: InferenceConfig,
        output_directory: Path,
    ) -> SupplementalDiagnosticResult:
        output_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        baseline_directory = (
            output_directory
            / "baseline"
        )

        events_path = (
            baseline_directory
            / "events.jsonl"
        )

        memory_path = (
            output_directory
            / "memory.sqlite3"
        )

        snapshot_path = (
            output_directory
            / "pre_focal_snapshot.json"
        )

        summary_path = (
            output_directory
            / "run_summary.json"
        )

        identity = RunIdentity(
            experiment_id=(
                SUPPLEMENTAL_EXPERIMENT_ID
            ),
            run_id=(
                "supplemental_memory_"
                "write_read__C3__r001"
            ),
            task_id=(
                MEMORY_TASK_ID
            ),
            task_classification=(
                TaskClassification
                .DIAGNOSTIC
            ),
            condition=Condition.C3,
            repetition_id=1,
            seed=action_config.seed,
        )

        writer = RawEventWriter(
            events_path
        )

        runtime = AgentRuntime(
            identity=identity,
            model_adapter=(
                model_adapter
            ),
            event_writer=writer,
        )

        memory_store = (
            SQLiteMemoryStore(
                memory_path
            )
        )

        memory_store.initialize()

        action_schema = (
            self._action_schema(
                focal_decision_id=(
                    "memory_route"
                )
            )
        )

        task_prompt = (
            "Choose route B if persistent memory "
            "states that preferred_route is west. "
            "Otherwise choose route A."
        )

        runtime.start()

        runtime.begin_focal_task(
            task_prompt=task_prompt,
            action_schema=(
                action_schema
            ),
        )

        scaffold = ScaffoldController(
            runtime=runtime,
            memory_store=(
                memory_store
            ),
            memory_namespace=(
                "supplemental_memory"
            ),
        )

        write = scaffold.write_memory(
            memory_id=(
                "preferred_route"
            ),
            key=(
                "preferred_route"
            ),
            value="west",
            metadata={
                "study_role": (
                    SUPPLEMENTAL_STUDY_ROLE
                ),
                "diagnostic_id": (
                    "memory_write_read"
                ),
            },
        )

        read = scaffold.read_memory(
            memory_id=(
                "preferred_route"
            )
        )

        if (
            not read.found
            or read.record is None
            or read.memory_element
            is None
        ):
            raise SupplementalDiagnosticError(
                code="memory_read_failed",
                message=(
                    "Supplemental memory diagnostic "
                    "could not read the written memory."
                ),
            )

        prompt_element = (
            runtime.prompt_element
        )

        if prompt_element is None:
            raise SupplementalDiagnosticError(
                code="prompt_element_missing",
                message=(
                    "Memory diagnostic runtime "
                    "does not contain canonical P."
                ),
            )

        snapshot = (
            SnapshotManager()
            .capture(
                experiment_id=(
                    identity
                    .experiment_id
                ),
                run_id=(
                    identity.run_id
                ),
                task_id=(
                    identity.task_id
                ),
                condition=(
                    identity.condition
                ),
                repetition_id=(
                    identity
                    .repetition_id
                ),
                sequence_position=len(
                    runtime.events
                ),
                prompt_elements=(
                    prompt_element,
                ),
                memory_store=(
                    memory_store
                ),
                memory_namespace=(
                    "supplemental_memory"
                ),
                runtime_configuration={
                    "action_inference": (
                        asdict(
                            action_config
                        )
                    ),
                    "explanation_inference": (
                        asdict(
                            explanation_config
                        )
                    ),
                },
                task_control_state={
                    "study_role": (
                        SUPPLEMENTAL_STUDY_ROLE
                    ),
                    "included_in_h1": False,
                    "included_in_h2": False,
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

        self._write_json_exclusive(
            snapshot_path,
            snapshot.to_dict(),
        )

        focal = (
            scaffold.invoke_focal_action(
                config=action_config,
                memory_elements=(
                    read.memory_element,
                ),
            )
        )

        explanation = (
            runtime
            .elicit_self_explanation(
                focal_action=focal,
                config=(
                    explanation_config
                ),
            )
        )

        normalized_action = (
            self._normalized_action(
                focal
            )
        )

        expected_action = {
            "action": (
                "select_route"
            ),
            "value": "B",
        }

        technical_failures = (
            self._technical_failures(
                runtime
            )
        )

        memory_snapshot = (
            memory_store.snapshot(
                namespace=(
                    "supplemental_memory"
                )
            )
        )

        checks = {
            "memory_write_before_read": (
                write.event.sequence_index
                < read.event.sequence_index
            ),
            "record_sequence_matches_write": (
                write
                .record
                .created_sequence
                == (
                    write
                    .event
                    .sequence_index
                )
            ),
            "written_record_equals_read_record": (
                write.record
                == read.record
            ),
            "memory_context_anchored_to_read": (
                (
                    read
                    .memory_element
                    .origin_event_id
                )
                == read.event.event_id
            ),
            "memory_context_not_anchored_to_write": (
                (
                    read
                    .memory_element
                    .origin_event_id
                )
                != write.event.event_id
            ),
            "snapshot_contains_memory_record": (
                any(
                    record.memory_id
                    == "preferred_route"
                    and record.key
                    == "preferred_route"
                    and record.value
                    == "west"
                    for record
                    in (
                        memory_snapshot
                        .records
                    )
                )
            ),
            "action_valid": (
                focal.valid
            ),
            "task_success": (
                normalized_action
                == expected_action
            ),
            "explanation_valid": (
                explanation.valid
            ),
            "no_technical_failure": (
                len(
                    technical_failures
                )
                == 0
            ),
            "no_intervention_events": (
                self._count_events(
                    runtime,
                    {
                        EventType
                        .INTERVENTION,
                        EventType
                        .SHAM_INTERVENTION,
                    },
                )
                == 0
            ),
        }

        result = (
            SupplementalDiagnosticResult(
                diagnostic_id=(
                    "memory_write_read"
                ),
                run_id=(
                    identity.run_id
                ),
                task_id=(
                    identity.task_id
                ),
                condition=(
                    identity
                    .condition
                    .value
                ),
                study_role=(
                    SUPPLEMENTAL_STUDY_ROLE
                ),
                post_pilot=True,
                included_in_h1=False,
                included_in_h2=False,
                action_valid=(
                    focal.valid
                ),
                task_success=(
                    normalized_action
                    == expected_action
                ),
                normalized_action=(
                    normalized_action
                ),
                explanation_valid=(
                    explanation.valid
                ),
                reported_sources=(
                    self._reported_sources(
                        explanation
                    )
                ),
                technical_failure_count=len(
                    technical_failures
                ),
                checks=checks,
                passed=all(
                    checks.values()
                ),
                output_directory=str(
                    output_directory
                ),
                events_path=str(
                    events_path
                ),
                snapshot_path=str(
                    snapshot_path
                ),
                summary_path=str(
                    summary_path
                ),
            )
        )

        self._write_json_exclusive(
            summary_path,
            {
                **result.to_dict(),
                "expected_action": (
                    expected_action
                ),
                "memory_write": (
                    write.to_dict()
                ),
                "memory_read": (
                    read.to_dict()
                ),
                "snapshot": {
                    "snapshot_id": (
                        snapshot
                        .snapshot_id
                    ),
                    "state_checksum_sha256": (
                        snapshot
                        .state_checksum_sha256
                    ),
                },
                "focal_action": (
                    focal.to_dict()
                ),
                "self_explanation": (
                    explanation.to_dict()
                ),
            },
        )

        return result

    def _execute_environment(
        self,
        *,
        model_adapter: ModelAdapter,
        action_config: InferenceConfig,
        explanation_config: InferenceConfig,
        output_directory: Path,
    ) -> SupplementalDiagnosticResult:
        output_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        baseline_directory = (
            output_directory
            / "baseline"
        )

        events_path = (
            baseline_directory
            / "events.jsonl"
        )

        snapshot_path = (
            output_directory
            / "pre_focal_snapshot.json"
        )

        summary_path = (
            output_directory
            / "run_summary.json"
        )

        identity = RunIdentity(
            experiment_id=(
                SUPPLEMENTAL_EXPERIMENT_ID
            ),
            run_id=(
                "supplemental_environment_"
                "transition_query__C1__r001"
            ),
            task_id=(
                ENVIRONMENT_TASK_ID
            ),
            task_classification=(
                TaskClassification
                .DIAGNOSTIC
            ),
            condition=Condition.C1,
            repetition_id=1,
            seed=action_config.seed,
        )

        writer = RawEventWriter(
            events_path
        )

        runtime = AgentRuntime(
            identity=identity,
            model_adapter=(
                model_adapter
            ),
            event_writer=writer,
        )

        environment = (
            EnvironmentState(
                {
                    "door": {
                        "status": (
                            "closed"
                        )
                    }
                }
            )
        )

        transition_rule = (
            TransitionRule(
                rule_id=(
                    "open_door"
                ),
                match={
                    "action": (
                        "open_door"
                    )
                },
                path=(
                    "door",
                    "status",
                ),
                required_value=(
                    "closed"
                ),
                new_value=(
                    "opened"
                ),
                metadata={
                    "study_role": (
                        SUPPLEMENTAL_STUDY_ROLE
                    ),
                    "diagnostic_id": (
                        "environment_transition_query"
                    ),
                },
            )
        )

        transition_engine = (
            DeterministicTransitionEngine(
                (
                    transition_rule,
                )
            )
        )

        action_schema = (
            self._action_schema(
                focal_decision_id=(
                    "environment_route"
                )
            )
        )

        task_prompt = (
            "Choose route B if the current "
            "environment says door.status is opened. "
            "Otherwise choose route A."
        )

        runtime.start()

        runtime.begin_focal_task(
            task_prompt=task_prompt,
            action_schema=(
                action_schema
            ),
        )

        scaffold = ScaffoldController(
            runtime=runtime,
            environment=environment,
            transition_engine=(
                transition_engine
            ),
        )

        transition = (
            scaffold
            .transition_environment(
                action={
                    "action": (
                        "open_door"
                    )
                }
            )
        )

        observation = (
            scaffold
            .query_environment(
                path=(
                    "door",
                    "status",
                )
            )
        )

        prompt_element = (
            runtime.prompt_element
        )

        if prompt_element is None:
            raise SupplementalDiagnosticError(
                code="prompt_element_missing",
                message=(
                    "Environment diagnostic runtime "
                    "does not contain canonical P."
                ),
            )

        snapshot = (
            SnapshotManager()
            .capture(
                experiment_id=(
                    identity
                    .experiment_id
                ),
                run_id=(
                    identity.run_id
                ),
                task_id=(
                    identity.task_id
                ),
                condition=(
                    identity.condition
                ),
                repetition_id=(
                    identity
                    .repetition_id
                ),
                sequence_position=len(
                    runtime.events
                ),
                prompt_elements=(
                    prompt_element,
                ),
                environment=(
                    environment
                ),
                runtime_configuration={
                    "action_inference": (
                        asdict(
                            action_config
                        )
                    ),
                    "explanation_inference": (
                        asdict(
                            explanation_config
                        )
                    ),
                },
                task_control_state={
                    "study_role": (
                        SUPPLEMENTAL_STUDY_ROLE
                    ),
                    "included_in_h1": False,
                    "included_in_h2": False,
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

        self._write_json_exclusive(
            snapshot_path,
            snapshot.to_dict(),
        )

        focal = (
            scaffold.invoke_focal_action(
                config=action_config,
                environment_elements=(
                    observation
                    .environment_element,
                ),
            )
        )

        explanation = (
            runtime
            .elicit_self_explanation(
                focal_action=focal,
                config=(
                    explanation_config
                ),
            )
        )

        normalized_action = (
            self._normalized_action(
                focal
            )
        )

        expected_action = {
            "action": (
                "select_route"
            ),
            "value": "B",
        }

        technical_failures = (
            self._technical_failures(
                runtime
            )
        )

        final_query = environment.query(
            (
                "door",
                "status",
            )
        )

        checks = {
            "transition_before_query": (
                transition
                .event
                .sequence_index
                < (
                    observation
                    .event
                    .sequence_index
                )
            ),
            "transition_changed_state": (
                transition
                .result
                .state_changed
            ),
            "transition_before_value_closed": (
                transition
                .result
                .before
                == "closed"
            ),
            "transition_after_value_opened": (
                transition
                .result
                .after
                == "opened"
            ),
            "queried_state_is_opened": (
                observation
                .result
                .value
                == "opened"
            ),
            "environment_context_anchored_to_query": (
                (
                    observation
                    .environment_element
                    .origin_event_id
                )
                == (
                    observation
                    .event
                    .event_id
                )
            ),
            "snapshot_contains_post_transition_state": (
                final_query.value
                == "opened"
            ),
            "action_valid": (
                focal.valid
            ),
            "task_success": (
                normalized_action
                == expected_action
            ),
            "explanation_valid": (
                explanation.valid
            ),
            "no_technical_failure": (
                len(
                    technical_failures
                )
                == 0
            ),
            "no_intervention_events": (
                self._count_events(
                    runtime,
                    {
                        EventType
                        .INTERVENTION,
                        EventType
                        .SHAM_INTERVENTION,
                    },
                )
                == 0
            ),
        }

        result = (
            SupplementalDiagnosticResult(
                diagnostic_id=(
                    "environment_transition_query"
                ),
                run_id=(
                    identity.run_id
                ),
                task_id=(
                    identity.task_id
                ),
                condition=(
                    identity
                    .condition
                    .value
                ),
                study_role=(
                    SUPPLEMENTAL_STUDY_ROLE
                ),
                post_pilot=True,
                included_in_h1=False,
                included_in_h2=False,
                action_valid=(
                    focal.valid
                ),
                task_success=(
                    normalized_action
                    == expected_action
                ),
                normalized_action=(
                    normalized_action
                ),
                explanation_valid=(
                    explanation.valid
                ),
                reported_sources=(
                    self._reported_sources(
                        explanation
                    )
                ),
                technical_failure_count=len(
                    technical_failures
                ),
                checks=checks,
                passed=all(
                    checks.values()
                ),
                output_directory=str(
                    output_directory
                ),
                events_path=str(
                    events_path
                ),
                snapshot_path=str(
                    snapshot_path
                ),
                summary_path=str(
                    summary_path
                ),
            )
        )

        self._write_json_exclusive(
            summary_path,
            {
                **result.to_dict(),
                "expected_action": (
                    expected_action
                ),
                "transition_rule": (
                    transition_rule.to_dict()
                ),
                "transition": (
                    transition.to_dict()
                ),
                "environment_query": (
                    observation.to_dict()
                ),
                "snapshot": {
                    "snapshot_id": (
                        snapshot
                        .snapshot_id
                    ),
                    "state_checksum_sha256": (
                        snapshot
                        .state_checksum_sha256
                    ),
                },
                "focal_action": (
                    focal.to_dict()
                ),
                "self_explanation": (
                    explanation.to_dict()
                ),
            },
        )

        return result

    @staticmethod
    def _action_schema(
        *,
        focal_decision_id: str,
    ) -> FocalActionSchema:
        return FocalActionSchema(
            focal_decision_id=(
                focal_decision_id
            ),
            fields=(
                ActionField(
                    name="action",
                    json_type="string",
                    required=True,
                    allowed_values=(
                        "select_route",
                    ),
                ),
                ActionField(
                    name="value",
                    json_type="string",
                    required=True,
                    allowed_values=(
                        "A",
                        "B",
                    ),
                ),
            ),
            allow_additional_fields=False,
            metadata={
                "study_role": (
                    SUPPLEMENTAL_STUDY_ROLE
                ),
            },
        )

    @staticmethod
    def _normalized_action(
        focal,
    ) -> dict[str, Any] | None:
        if (
            not focal.valid
            or focal.parse_result
            is None
        ):
            return None

        return deepcopy(
            focal
            .parse_result
            .normalized_action
            .payload
        )

    @staticmethod
    def _reported_sources(
        explanation,
    ) -> tuple[
        str,
        ...
    ]:
        if (
            not explanation.valid
            or explanation.parse_result
            is None
        ):
            return ()

        return tuple(
            source.value
            for source
            in (
                explanation
                .parse_result
                .reported_sources
            )
        )

    @staticmethod
    def _technical_failures(
        runtime: AgentRuntime,
    ):
        return tuple(
            event
            for event
            in runtime.events
            if (
                event.event_type
                == (
                    EventType
                    .TECHNICAL_FAILURE
                )
            )
        )

    @staticmethod
    def _count_events(
        runtime: AgentRuntime,
        event_types,
    ) -> int:
        return sum(
            1
            for event
            in runtime.events
            if (
                event.event_type
                in event_types
            )
        )

    @staticmethod
    def _write_json_exclusive(
        path: Path,
        value,
    ) -> None:
        if path.exists():
            raise SupplementalDiagnosticError(
                code="artifact_exists",
                message=(
                    f"Supplemental artifact "
                    f"{str(path)!r} already exists."
                ),
            )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
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