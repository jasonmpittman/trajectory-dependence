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
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..agent import (
    AgentRuntime,
    RunIdentity,
)
from ..interventions import (
    InterventionPairValidator,
)
from ..logging import (
    Condition,
    RawEventWriter,
    SourceClass,
)
from ..model import (
    InferenceConfig,
    ModelAdapter,
    ModelRuntimeMetadata,
)
from ..snapshots import (
    SnapshotManager,
)
from .loader import (
    load_task_suite,
)
from .resolution import (
    TaskBackingStateBuilder,
    TaskFixtureResolver,
    TaskInterventionResolver,
)
from .spec import (
    ExperimentalTaskSpec,
    TaskSpecError,
    TaskSuite,
)


PILOT_RUN_SCHEMA_VERSION = "pilot-run-v1"
PILOT_SEED_ALGORITHM = "sha256-v1"


class PilotConfigurationError(ValueError):
    """Defined failure loading or validating pilot configuration."""

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


class PilotPreflightError(RuntimeError):
    """Defined failure during model-free pilot preflight."""

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
class PilotFileMetadata:
    """Required provenance metadata for the pilot-run configuration."""

    author: str
    date: str
    copyright: str
    credits: tuple[str, ...]
    license: str
    version: str
    maintainer: str
    status: str

    def __post_init__(self) -> None:
        for field_name, value in (
            (
                "author",
                self.author,
            ),
            (
                "date",
                self.date,
            ),
            (
                "copyright",
                self.copyright,
            ),
            (
                "license",
                self.license,
            ),
            (
                "version",
                self.version,
            ),
            (
                "maintainer",
                self.maintainer,
            ),
            (
                "status",
                self.status,
            ),
        ):
            if (
                not isinstance(
                    value,
                    str,
                )
                or not value
            ):
                raise PilotConfigurationError(
                    code="invalid_pilot_metadata",
                    message=(
                        f"Pilot metadata field "
                        f"{field_name!r} must be "
                        "a non-empty string."
                    ),
                )

        if (
            not self.credits
            or not all(
                isinstance(
                    value,
                    str,
                )
                and value
                for value
                in self.credits
            )
        ):
            raise PilotConfigurationError(
                code="invalid_pilot_metadata",
                message=(
                    "Pilot metadata credits must "
                    "contain at least one "
                    "non-empty string."
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "author": self.author,
            "date": self.date,
            "copyright": (
                self.copyright
            ),
            "credits": list(
                self.credits
            ),
            "license": self.license,
            "version": self.version,
            "maintainer": (
                self.maintainer
            ),
            "status": self.status,
        }


@dataclass(frozen=True)
class PilotSeedPolicy:
    """Deterministic task/repetition seed policy."""

    algorithm: str
    base_seed: int

    match_task_repetition_across_conditions: bool

    def __post_init__(self) -> None:
        if (
            self.algorithm
            != PILOT_SEED_ALGORITHM
        ):
            raise PilotConfigurationError(
                code="unsupported_seed_algorithm",
                message=(
                    f"Pilot seed algorithm must be "
                    f"{PILOT_SEED_ALGORITHM!r}."
                ),
            )

        if (
            not isinstance(
                self.base_seed,
                int,
            )
            or self.base_seed < 0
        ):
            raise PilotConfigurationError(
                code="invalid_base_seed",
                message=(
                    "Pilot base_seed must be "
                    "a non-negative integer."
                ),
            )

        if (
            self
            .match_task_repetition_across_conditions
            is not True
        ):
            raise PilotConfigurationError(
                code="unmatched_condition_seeds",
                message=(
                    "Pilot seed policy must match "
                    "task/repetition seeds across C0-C3."
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": (
                self.algorithm
            ),
            "base_seed": (
                self.base_seed
            ),
            "match_task_repetition_across_conditions": (
                self
                .match_task_repetition_across_conditions
            ),
        }


@dataclass(frozen=True)
class PilotModelConfig:
    """Model identity required for the pilot."""

    identifier: str
    revision: str | None
    checksum: str | None
    quantization: str

    path_environment_variable: str

    def __post_init__(self) -> None:
        for field_name, value in (
            (
                "identifier",
                self.identifier,
            ),
            (
                "quantization",
                self.quantization,
            ),
            (
                "path_environment_variable",
                self.path_environment_variable,
            ),
        ):
            if (
                not isinstance(
                    value,
                    str,
                )
                or not value
            ):
                raise PilotConfigurationError(
                    code="invalid_model_configuration",
                    message=(
                        f"Model field {field_name!r} "
                        "must be a non-empty string."
                    ),
                )

        for field_name, value in (
            (
                "revision",
                self.revision,
            ),
            (
                "checksum",
                self.checksum,
            ),
        ):
            if (
                value is not None
                and (
                    not isinstance(
                        value,
                        str,
                    )
                    or not value
                )
            ):
                raise PilotConfigurationError(
                    code="invalid_model_configuration",
                    message=(
                        f"Model field {field_name!r} "
                        "must be null or a "
                        "non-empty string."
                    ),
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": (
                self.identifier
            ),
            "revision": (
                self.revision
            ),
            "checksum": (
                self.checksum
            ),
            "quantization": (
                self.quantization
            ),
            "path_environment_variable": (
                self
                .path_environment_variable
            ),
        }


@dataclass(frozen=True)
class PilotConfiguration:
    """Validated suite-level pilot execution configuration."""

    metadata: PilotFileMetadata

    schema_version: str

    experiment_id: str
    task_suite_path: str
    output_root: str

    repetitions: int

    seed_policy: PilotSeedPolicy

    action_max_tokens: int
    explanation_max_tokens: int

    model: PilotModelConfig

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != PILOT_RUN_SCHEMA_VERSION
        ):
            raise PilotConfigurationError(
                code="unsupported_pilot_schema",
                message=(
                    f"Pilot schema must be "
                    f"{PILOT_RUN_SCHEMA_VERSION!r}."
                ),
            )

        for field_name, value in (
            (
                "experiment_id",
                self.experiment_id,
            ),
            (
                "task_suite_path",
                self.task_suite_path,
            ),
            (
                "output_root",
                self.output_root,
            ),
        ):
            if (
                not isinstance(
                    value,
                    str,
                )
                or not value
            ):
                raise PilotConfigurationError(
                    code="invalid_pilot_configuration",
                    message=(
                        f"{field_name} must be a "
                        "non-empty string."
                    ),
                )

        if (
            not isinstance(
                self.repetitions,
                int,
            )
            or self.repetitions < 1
        ):
            raise PilotConfigurationError(
                code="invalid_repetitions",
                message=(
                    "Pilot repetitions must be "
                    "an integer >= 1."
                ),
            )

        for field_name, value in (
            (
                "action_max_tokens",
                self.action_max_tokens,
            ),
            (
                "explanation_max_tokens",
                self.explanation_max_tokens,
            ),
        ):
            if (
                not isinstance(
                    value,
                    int,
                )
                or value < 1
            ):
                raise PilotConfigurationError(
                    code="invalid_max_tokens",
                    message=(
                        f"{field_name} must be "
                        "an integer >= 1."
                    ),
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": (
                self.metadata.to_dict()
            ),
            "schema_version": (
                self.schema_version
            ),
            "experiment_id": (
                self.experiment_id
            ),
            "task_suite_path": (
                self.task_suite_path
            ),
            "output_root": (
                self.output_root
            ),
            "repetitions": (
                self.repetitions
            ),
            "seed_policy": (
                self.seed_policy.to_dict()
            ),
            "action_inference": {
                "max_tokens": (
                    self.action_max_tokens
                ),
            },
            "explanation_inference": {
                "max_tokens": (
                    self
                    .explanation_max_tokens
                ),
            },
            "model": (
                self.model.to_dict()
            ),
        }


@dataclass(frozen=True)
class PilotRunPlanEntry:
    """One deterministic task × condition × repetition pilot run."""

    ordinal: int

    task_id: str
    condition: Condition
    repetition_id: int

    run_id: str

    action_seed: int
    explanation_seed: int

    output_directory: str

    def action_config(
        self,
        *,
        max_tokens: int,
    ) -> InferenceConfig:
        return InferenceConfig(
            max_tokens=max_tokens,
            seed=self.action_seed,
        )

    def explanation_config(
        self,
        *,
        max_tokens: int,
    ) -> InferenceConfig:
        return InferenceConfig(
            max_tokens=max_tokens,
            seed=self.explanation_seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": (
                self.ordinal
            ),
            "task_id": (
                self.task_id
            ),
            "condition": (
                self.condition.value
            ),
            "repetition_id": (
                self.repetition_id
            ),
            "run_id": (
                self.run_id
            ),
            "action_seed": (
                self.action_seed
            ),
            "explanation_seed": (
                self.explanation_seed
            ),
            "output_directory": (
                self.output_directory
            ),
        }


@dataclass(frozen=True)
class PilotExecutionPlan:
    """Immutable deterministic pilot execution plan."""

    experiment_id: str

    suite_id: str
    suite_version: str

    repetitions: int

    seed_algorithm: str
    base_seed: int

    runs: tuple[
        PilotRunPlanEntry,
        ...
    ]

    plan_sha256: str

    def to_dict(
        self,
        *,
        include_checksum: bool = True,
    ) -> dict[str, Any]:
        value = {
            "experiment_id": (
                self.experiment_id
            ),
            "suite_id": (
                self.suite_id
            ),
            "suite_version": (
                self.suite_version
            ),
            "repetitions": (
                self.repetitions
            ),
            "seed_algorithm": (
                self.seed_algorithm
            ),
            "base_seed": (
                self.base_seed
            ),
            "runs": [
                run.to_dict()
                for run
                in self.runs
            ],
        }

        if include_checksum:
            value[
                "plan_sha256"
            ] = self.plan_sha256

        return value


@dataclass(frozen=True)
class PilotPreflightRunResult:
    """Model-free validation result for one planned run."""

    run_id: str
    task_id: str
    condition: Condition
    repetition_id: int

    snapshot_id: str
    snapshot_checksum: str

    intervention_pair_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "condition": (
                self.condition.value
            ),
            "repetition_id": (
                self.repetition_id
            ),
            "snapshot_id": (
                self.snapshot_id
            ),
            "snapshot_checksum": (
                self.snapshot_checksum
            ),
            "intervention_pair_count": (
                self.intervention_pair_count
            ),
        }


@dataclass(frozen=True)
class PilotPreflightResult:
    """Complete model-free preflight result."""

    plan: PilotExecutionPlan

    checked_runs: int
    checked_intervention_pairs: int

    run_results: tuple[
        PilotPreflightRunResult,
        ...
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": (
                self.plan.to_dict()
            ),
            "checked_runs": (
                self.checked_runs
            ),
            "checked_intervention_pairs": (
                self
                .checked_intervention_pairs
            ),
            "run_results": [
                result.to_dict()
                for result
                in self.run_results
            ],
        }


class _PreflightModelAdapter(
    ModelAdapter
):
    """
    Loaded no-inference adapter used only for task materialization.

    Any attempt to prepare or generate model input is a preflight bug.
    """

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def runtime_metadata(
        self,
    ) -> ModelRuntimeMetadata:
        return ModelRuntimeMetadata(
            model_identifier=(
                "pilot-preflight"
            ),
            model_revision=None,
            model_checksum=None,
            quantization="none",
            backend="preflight",
            backend_version="1",
            mlx_version="none",
            python_version="preflight",
            platform="preflight",
            model_path_env="none",
            device_info={},
        )

    def prepare_input(
        self,
        model_visible_text: str,
    ):
        raise PilotPreflightError(
            code="unexpected_model_preparation",
            message=(
                "Model input preparation occurred "
                "during model-free preflight."
            ),
        )

    def generate(
        self,
        *,
        input_record,
        config,
    ):
        raise PilotPreflightError(
            code="unexpected_model_generation",
            message=(
                "Model generation occurred "
                "during model-free preflight."
            ),
        )


class PilotPlanner:
    """Build deterministic run IDs, seeds, and output paths."""

    _CONDITION_ORDER = (
        Condition.C0,
        Condition.C1,
        Condition.C2,
        Condition.C3,
    )

    def build(
        self,
        *,
        config: PilotConfiguration,
        suite: TaskSuite,
    ) -> PilotExecutionPlan:
        runs: list[
            PilotRunPlanEntry
        ] = []

        ordinal = 0

        experiment_root = (
            Path(
                config.output_root
            )
            / config.experiment_id
        )

        for task in suite.tasks:
            for repetition_id in range(
                1,
                config.repetitions
                + 1,
            ):
                action_seed = (
                    self._derive_seed(
                        base_seed=(
                            config
                            .seed_policy
                            .base_seed
                        ),
                        suite_id=(
                            suite.suite_id
                        ),
                        task_id=(
                            task.task_id
                        ),
                        repetition_id=(
                            repetition_id
                        ),
                        role="action",
                    )
                )

                explanation_seed = (
                    self._derive_seed(
                        base_seed=(
                            config
                            .seed_policy
                            .base_seed
                        ),
                        suite_id=(
                            suite.suite_id
                        ),
                        task_id=(
                            task.task_id
                        ),
                        repetition_id=(
                            repetition_id
                        ),
                        role="explanation",
                    )
                )

                for condition in (
                    self._CONDITION_ORDER
                ):
                    if (
                        condition
                        not in task.conditions
                    ):
                        continue

                    ordinal += 1

                    run_id = (
                        f"{task.task_id}"
                        f"__{condition.value}"
                        f"__r{repetition_id:03d}"
                    )

                    output_directory = (
                        experiment_root
                        / task.task_id
                        / condition.value
                        / (
                            f"r"
                            f"{repetition_id:03d}"
                        )
                    )

                    runs.append(
                        PilotRunPlanEntry(
                            ordinal=ordinal,
                            task_id=(
                                task.task_id
                            ),
                            condition=condition,
                            repetition_id=(
                                repetition_id
                            ),
                            run_id=run_id,
                            action_seed=(
                                action_seed
                            ),
                            explanation_seed=(
                                explanation_seed
                            ),
                            output_directory=(
                                str(
                                    output_directory
                                )
                            ),
                        )
                    )

        provisional = {
            "experiment_id": (
                config.experiment_id
            ),
            "suite_id": (
                suite.suite_id
            ),
            "suite_version": (
                suite.suite_version
            ),
            "repetitions": (
                config.repetitions
            ),
            "seed_algorithm": (
                config
                .seed_policy
                .algorithm
            ),
            "base_seed": (
                config
                .seed_policy
                .base_seed
            ),
            "runs": [
                run.to_dict()
                for run
                in runs
            ],
        }

        checksum = sha256(
            _canonical_json(
                provisional
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        return PilotExecutionPlan(
            experiment_id=(
                config.experiment_id
            ),
            suite_id=(
                suite.suite_id
            ),
            suite_version=(
                suite.suite_version
            ),
            repetitions=(
                config.repetitions
            ),
            seed_algorithm=(
                config
                .seed_policy
                .algorithm
            ),
            base_seed=(
                config
                .seed_policy
                .base_seed
            ),
            runs=tuple(
                runs
            ),
            plan_sha256=checksum,
        )

    @staticmethod
    def _derive_seed(
        *,
        base_seed: int,
        suite_id: str,
        task_id: str,
        repetition_id: int,
        role: str,
    ) -> int:
        """
        Derive one stable positive 31-bit seed.

        Condition is intentionally absent so C0-C3 share the same seed for
        a given task/repetition/role.
        """

        material = (
            f"{PILOT_SEED_ALGORITHM}"
            f"|{base_seed}"
            f"|{suite_id}"
            f"|{task_id}"
            f"|r{repetition_id}"
            f"|{role}"
        )

        digest = sha256(
            material.encode(
                "utf-8"
            )
        ).digest()

        value = int.from_bytes(
            digest[:4],
            byteorder="big",
            signed=False,
        )

        return (
            value
            & 0x7FFFFFFF
        )


class PilotPreflightValidator:
    """
    Validate the entire pilot plan without invoking an LLM.

    For every planned run this validates:

    - task/condition materialization;
    - backing M/E state;
    - event-backed fixture resolution;
    - PRE_FOCAL snapshot capture;
    - eligible intervention resolution;
    - targeted/sham application from identical baseline;
    - required serialized-length preservation;
    - output-path non-collision.
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

    def validate(
        self,
        *,
        config: PilotConfiguration,
        suite: TaskSuite,
        plan: PilotExecutionPlan,
    ) -> PilotPreflightResult:
        task_map = {
            task.task_id: task
            for task
            in suite.tasks
        }

        self._validate_plan_identity(
            config=config,
            suite=suite,
            plan=plan,
        )

        self._validate_output_paths(
            plan
        )

        results: list[
            PilotPreflightRunResult
        ] = []

        pair_count = 0

        adapter = (
            _PreflightModelAdapter()
        )

        with TemporaryDirectory(
            prefix=(
                "trajectory_pilot_preflight_"
            )
        ) as temporary:
            root = Path(
                temporary
            )

            for run in plan.runs:
                task = task_map.get(
                    run.task_id
                )

                if task is None:
                    raise PilotPreflightError(
                        code="planned_task_not_found",
                        message=(
                            f"Planned task "
                            f"{run.task_id!r} "
                            "does not exist in "
                            "the task suite."
                        ),
                    )

                result = self._validate_run(
                    config=config,
                    task=task,
                    run=run,
                    adapter=adapter,
                    work_directory=(
                        root
                        / (
                            f"{run.ordinal:03d}_"
                            f"{run.run_id}"
                        )
                    ),
                )

                pair_count += (
                    result
                    .intervention_pair_count
                )

                results.append(
                    result
                )

        return PilotPreflightResult(
            plan=plan,
            checked_runs=len(
                results
            ),
            checked_intervention_pairs=(
                pair_count
            ),
            run_results=tuple(
                results
            ),
        )

    def _validate_run(
        self,
        *,
        config: PilotConfiguration,
        task: ExperimentalTaskSpec,
        run: PilotRunPlanEntry,
        adapter: ModelAdapter,
        work_directory: Path,
    ) -> PilotPreflightRunResult:
        work_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        runtime = AgentRuntime(
            identity=RunIdentity(
                experiment_id=(
                    config.experiment_id
                ),
                run_id=(
                    run.run_id
                ),
                task_id=(
                    task.task_id
                ),
                task_classification=(
                    task
                    .task_classification
                ),
                condition=(
                    run.condition
                ),
                repetition_id=(
                    run.repetition_id
                ),
                seed=(
                    run.action_seed
                ),
            ),
            model_adapter=adapter,
            event_writer=RawEventWriter(
                work_directory
                / "events.jsonl"
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

        backing = (
            self.backing_state_builder
            .build(
                task=task,
                run_id=(
                    run.run_id
                ),
                work_directory=(
                    work_directory
                    / "state"
                ),
            )
        )

        context = (
            self.fixture_resolver
            .resolve(
                task=task,
                condition=(
                    run.condition
                ),
                runtime=runtime,
                backing_state=(
                    backing
                ),
            )
        )

        prompt = (
            runtime.prompt_element
        )

        if prompt is None:
            raise PilotPreflightError(
                code="preflight_prompt_unavailable",
                message=(
                    f"Run {run.run_id!r} "
                    "did not create canonical P."
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

        available_sources = set(
            task.available_sources[
                run.condition
            ]
        )

        snapshot_memory_store = (
            backing.memory_store
            if (
                SourceClass.M
                in available_sources
            )
            else None
        )

        snapshot_memory_namespace = (
            backing.memory_namespace
            if (
                SourceClass.M
                in available_sources
            )
            else None
        )

        snapshot_environment = (
            backing.environment
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
                    config.experiment_id
                ),
                run_id=(
                    run.run_id
                ),
                task_id=(
                    task.task_id
                ),
                condition=(
                    run.condition
                ),
                repetition_id=(
                    run.repetition_id
                ),
                sequence_position=(
                    runtime.events[-1]
                    .sequence_index
                ),
                prompt_elements=(
                    prompt,
                ),
                tool_elements=(
                    context.tool_elements
                ),
                history_elements=(
                    context.history_elements
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
                    "available_sources": [
                        source.value
                        for source
                        in (
                            task
                            .available_sources[
                                run.condition
                            ]
                        )
                    ],
                },
                seed_metadata={
                    "action_seed": (
                        run.action_seed
                    ),
                    "explanation_seed": (
                        run
                        .explanation_seed
                    ),
                },
            )
        )

        eligible_pairs = [
            template
            for template
            in task.intervention_pairs
            if (
                run.condition
                in template
                .eligible_conditions
            )
        ]

        for template in (
            eligible_pairs
        ):
            pair = (
                self.intervention_resolver
                .resolve_pair(
                    task=task,
                    condition=(
                        run.condition
                    ),
                    template=template,
                    context=context,
                )
            )

            validated = (
                self.pair_validator
                .validate(
                    snapshot=snapshot,
                    pair=pair,
                )
            )

            self._validate_length_requirement(
                required=(
                    template
                    .targeted
                    .length_match_required
                ),
                role="targeted",
                pair_id=(
                    template.pair_id
                ),
                run_id=(
                    run.run_id
                ),
                before=(
                    validated
                    .targeted_application
                    .diff
                    .before_serialized_chars
                ),
                after=(
                    validated
                    .targeted_application
                    .diff
                    .after_serialized_chars
                ),
            )

            self._validate_length_requirement(
                required=(
                    template
                    .sham
                    .length_match_required
                ),
                role="sham",
                pair_id=(
                    template.pair_id
                ),
                run_id=(
                    run.run_id
                ),
                before=(
                    validated
                    .sham_application
                    .diff
                    .before_serialized_chars
                ),
                after=(
                    validated
                    .sham_application
                    .diff
                    .after_serialized_chars
                ),
            )

        return PilotPreflightRunResult(
            run_id=(
                run.run_id
            ),
            task_id=(
                task.task_id
            ),
            condition=(
                run.condition
            ),
            repetition_id=(
                run.repetition_id
            ),
            snapshot_id=(
                snapshot.snapshot_id
            ),
            snapshot_checksum=(
                snapshot
                .state_checksum_sha256
            ),
            intervention_pair_count=len(
                eligible_pairs
            ),
        )

    @staticmethod
    def _validate_length_requirement(
        *,
        required: bool,
        role: str,
        pair_id: str,
        run_id: str,
        before: int,
        after: int,
    ) -> None:
        if (
            required
            and before != after
        ):
            raise PilotPreflightError(
                code="intervention_length_mismatch",
                message=(
                    f"Run {run_id!r}, pair "
                    f"{pair_id!r}, {role} "
                    "intervention requires serialized "
                    f"length matching but changed "
                    f"{before} characters to {after}."
                ),
            )

    @staticmethod
    def _validate_output_paths(
        plan: PilotExecutionPlan,
    ) -> None:
        values = [
            run.output_directory
            for run
            in plan.runs
        ]

        if (
            len(
                values
            )
            != len(
                set(
                    values
                )
            )
        ):
            raise PilotPreflightError(
                code="duplicate_output_directory",
                message=(
                    "Pilot plan contains duplicate "
                    "run output directories."
                ),
            )

        for value in values:
            if Path(
                value
            ).exists():
                raise PilotPreflightError(
                    code="planned_output_exists",
                    message=(
                        f"Planned run output "
                        f"{value!r} already exists."
                    ),
                )

    @staticmethod
    def _validate_plan_identity(
        *,
        config: PilotConfiguration,
        suite: TaskSuite,
        plan: PilotExecutionPlan,
    ) -> None:
        if (
            plan.experiment_id
            != config.experiment_id
        ):
            raise PilotPreflightError(
                code="plan_experiment_mismatch",
                message=(
                    "Pilot plan experiment_id "
                    "does not match configuration."
                ),
            )

        if (
            plan.suite_id
            != suite.suite_id
            or plan.suite_version
            != suite.suite_version
        ):
            raise PilotPreflightError(
                code="plan_suite_mismatch",
                message=(
                    "Pilot plan does not match "
                    "the loaded task suite."
                ),
            )


def load_pilot_configuration(
    path: str | Path,
) -> PilotConfiguration:
    """Load and strictly validate one pilot-run JSON file."""

    source = Path(
        path
    )

    if not source.is_file():
        raise PilotConfigurationError(
            code="pilot_config_not_found",
            message=(
                f"Pilot configuration "
                f"{str(source)!r} does not exist."
            ),
        )

    try:
        with source.open(
            "r",
            encoding="utf-8",
        ) as handle:
            raw = json.load(
                handle
            )

    except json.JSONDecodeError as exc:
        raise PilotConfigurationError(
            code="invalid_pilot_json",
            message=(
                "Pilot configuration is "
                "not valid JSON."
            ),
        ) from exc

    if not isinstance(
        raw,
        dict,
    ):
        raise PilotConfigurationError(
            code="pilot_root_not_object",
            message=(
                "Pilot configuration root "
                "must be a JSON object."
            ),
        )

    _require_fields(
        raw,
        {
            "metadata",
            "schema_version",
            "experiment_id",
            "task_suite_path",
            "output_root",
            "repetitions",
            "seed_policy",
            "action_inference",
            "explanation_inference",
            "model",
        },
        location=(
            "pilot configuration"
        ),
    )

    metadata_raw = raw[
        "metadata"
    ]

    if not isinstance(
        metadata_raw,
        dict,
    ):
        raise PilotConfigurationError(
            code="invalid_pilot_metadata",
            message=(
                "metadata must be "
                "a JSON object."
            ),
        )

    _require_fields(
        metadata_raw,
        {
            "author",
            "date",
            "copyright",
            "credits",
            "license",
            "version",
            "maintainer",
            "status",
        },
        location="metadata",
    )

    credits = metadata_raw[
        "credits"
    ]

    if not isinstance(
        credits,
        list,
    ):
        raise PilotConfigurationError(
            code="invalid_pilot_metadata",
            message=(
                "metadata.credits must "
                "be an array."
            ),
        )

    seed_raw = raw[
        "seed_policy"
    ]

    if not isinstance(
        seed_raw,
        dict,
    ):
        raise PilotConfigurationError(
            code="invalid_seed_policy",
            message=(
                "seed_policy must be "
                "a JSON object."
            ),
        )

    _require_fields(
        seed_raw,
        {
            "algorithm",
            "base_seed",
            "match_task_repetition_across_conditions",
        },
        location="seed_policy",
    )

    action_raw = raw[
        "action_inference"
    ]

    explanation_raw = raw[
        "explanation_inference"
    ]

    for (
        location,
        value,
    ) in (
        (
            "action_inference",
            action_raw,
        ),
        (
            "explanation_inference",
            explanation_raw,
        ),
    ):
        if not isinstance(
            value,
            dict,
        ):
            raise PilotConfigurationError(
                code="invalid_inference_configuration",
                message=(
                    f"{location} must be "
                    "a JSON object."
                ),
            )

        _require_fields(
            value,
            {
                "max_tokens",
            },
            location=location,
        )

    model_raw = raw[
        "model"
    ]

    if not isinstance(
        model_raw,
        dict,
    ):
        raise PilotConfigurationError(
            code="invalid_model_configuration",
            message=(
                "model must be a "
                "JSON object."
            ),
        )

    _require_fields(
        model_raw,
        {
            "identifier",
            "revision",
            "checksum",
            "quantization",
            "path_environment_variable",
        },
        location="model",
    )

    return PilotConfiguration(
        metadata=PilotFileMetadata(
            author=metadata_raw[
                "author"
            ],
            date=metadata_raw[
                "date"
            ],
            copyright=metadata_raw[
                "copyright"
            ],
            credits=tuple(
                credits
            ),
            license=metadata_raw[
                "license"
            ],
            version=metadata_raw[
                "version"
            ],
            maintainer=metadata_raw[
                "maintainer"
            ],
            status=metadata_raw[
                "status"
            ],
        ),
        schema_version=raw[
            "schema_version"
        ],
        experiment_id=raw[
            "experiment_id"
        ],
        task_suite_path=raw[
            "task_suite_path"
        ],
        output_root=raw[
            "output_root"
        ],
        repetitions=raw[
            "repetitions"
        ],
        seed_policy=PilotSeedPolicy(
            algorithm=seed_raw[
                "algorithm"
            ],
            base_seed=seed_raw[
                "base_seed"
            ],
            match_task_repetition_across_conditions=(
                seed_raw[
                    "match_task_repetition_across_conditions"
                ]
            ),
        ),
        action_max_tokens=(
            action_raw[
                "max_tokens"
            ]
        ),
        explanation_max_tokens=(
            explanation_raw[
                "max_tokens"
            ]
        ),
        model=PilotModelConfig(
            identifier=model_raw[
                "identifier"
            ],
            revision=model_raw[
                "revision"
            ],
            checksum=model_raw[
                "checksum"
            ],
            quantization=model_raw[
                "quantization"
            ],
            path_environment_variable=(
                model_raw[
                    "path_environment_variable"
                ]
            ),
        ),
    )


def load_configured_task_suite(
    config: PilotConfiguration,
) -> TaskSuite:
    """Load the task suite referenced by a validated pilot config."""

    try:
        return load_task_suite(
            config.task_suite_path
        )

    except TaskSpecError:
        raise


def _canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_fields(
    raw: dict[
        str,
        Any,
    ],
    expected: set[str],
    *,
    location: str,
) -> None:
    actual = set(
        raw.keys()
    )

    missing = (
        expected
        - actual
    )

    extra = (
        actual
        - expected
    )

    if (
        missing
        or extra
    ):
        details: list[str] = []

        if missing:
            details.append(
                "missing="
                + ",".join(
                    sorted(
                        missing
                    )
                )
            )

        if extra:
            details.append(
                "extra="
                + ",".join(
                    sorted(
                        extra
                    )
                )
            )

        raise PilotConfigurationError(
            code="pilot_schema_field_mismatch",
            message=(
                f"{location} fields do not "
                "match the schema: "
                + "; ".join(
                    details
                )
            ),
        )