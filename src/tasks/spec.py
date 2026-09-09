__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..logging import (
    Condition,
    SourceClass,
    TaskClassification,
)
from ..normalization import (
    ActionParseError,
    FocalActionSchema,
    StructuredActionParser,
)
from ..interventions import (
    InterventionOperation,
)


TASK_SPEC_SCHEMA_VERSION = "pilot-task-v1"


class TaskSpecError(ValueError):
    """Defined validation failure in an experimental task specification."""

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


class TaskFamily(str, Enum):
    """Protocol-defined task families."""

    MEMORY_DEPENDENCE = (
        "memory_dependence"
    )

    TOOL_DEPENDENCE = (
        "tool_dependence"
    )

    HISTORY_DEPENDENCE = (
        "history_dependence"
    )

    ENVIRONMENTAL_STATE_DEPENDENCE = (
        "environmental_state_dependence"
    )

    CONFLICT_INTEGRATION = (
        "conflict_integration"
    )

    MULTI_SOURCE_DEPENDENCE = (
        "multi_source_dependence"
    )


_CONDITION_SOURCE_CAPABILITIES = {
    Condition.C0: frozenset(
        {
            SourceClass.P,
        }
    ),
    Condition.C1: frozenset(
        {
            SourceClass.P,
            SourceClass.T,
            SourceClass.E,
        }
    ),
    Condition.C2: frozenset(
        {
            SourceClass.P,
            SourceClass.T,
            SourceClass.H,
            SourceClass.E,
        }
    ),
    Condition.C3: frozenset(
        {
            SourceClass.P,
            SourceClass.T,
            SourceClass.H,
            SourceClass.M,
            SourceClass.E,
        }
    ),
}


@dataclass(frozen=True)
class TaskFileMetadata:
    """Required provenance header for a task configuration file."""

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
                raise TaskSpecError(
                    code="invalid_task_metadata",
                    message=(
                        f"Task metadata field "
                        f"{field_name!r} must be a "
                        "non-empty string."
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
            raise TaskSpecError(
                code="invalid_task_metadata",
                message=(
                    "Task metadata credits must contain "
                    "at least one non-empty string."
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
class ToolFixture:
    """
    Preregistered deterministic tool interaction.

    The actual ToolResult is produced by the tool implementation at run
    time. The task specification controls only its request and state.
    """

    fixture_id: str
    tool_name: str

    request: dict[str, Any]
    state: Any

    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _require_identifier(
            self.fixture_id,
            field_name="tool fixture_id",
        )

        _require_identifier(
            self.tool_name,
            field_name="tool_name",
        )

        _require_json(
            self.request,
            field_name="tool request",
        )

        _require_json(
            self.state,
            field_name="tool state",
        )

        _require_json(
            self.metadata,
            field_name="tool metadata",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": (
                self.fixture_id
            ),
            "tool_name": (
                self.tool_name
            ),
            "request": deepcopy(
                self.request
            ),
            "state": deepcopy(
                self.state
            ),
            "metadata": deepcopy(
                self.metadata
            ),
        }


@dataclass(frozen=True)
class HistoryFixture:
    """Preregistered within-task trajectory observation."""

    fixture_id: str

    content: Any

    origin_source: (
        SourceClass | None
    )

    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _require_identifier(
            self.fixture_id,
            field_name=(
                "history fixture_id"
            ),
        )

        _require_json(
            self.content,
            field_name="history content",
        )

        _require_json(
            self.metadata,
            field_name="history metadata",
        )

        if (
            self.origin_source
            not in {
                None,
                SourceClass.T,
                SourceClass.E,
            }
        ):
            raise TaskSpecError(
                code="invalid_history_origin",
                message=(
                    "History origin_source must be "
                    "T, E, or null."
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": (
                self.fixture_id
            ),
            "content": deepcopy(
                self.content
            ),
            "origin_source": (
                self.origin_source.value
                if (
                    self.origin_source
                    is not None
                )
                else None
            ),
            "metadata": deepcopy(
                self.metadata
            ),
        }


@dataclass(frozen=True)
class MemoryFixture:
    """Preregistered persistent-memory item."""

    fixture_id: str

    key: str
    value: Any

    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _require_identifier(
            self.fixture_id,
            field_name=(
                "memory fixture_id"
            ),
        )

        _require_identifier(
            self.key,
            field_name="memory key",
        )

        _require_json(
            self.value,
            field_name="memory value",
        )

        _require_json(
            self.metadata,
            field_name="memory metadata",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": (
                self.fixture_id
            ),
            "key": self.key,
            "value": deepcopy(
                self.value
            ),
            "metadata": deepcopy(
                self.metadata
            ),
        }


@dataclass(frozen=True)
class EnvironmentObservationFixture:
    """One preregistered model-visible E observation."""

    fixture_id: str

    path: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(
            self.fixture_id,
            field_name=(
                "environment fixture_id"
            ),
        )

        if (
            not self.path
            or not all(
                isinstance(
                    value,
                    str,
                )
                and value
                for value
                in self.path
            )
        ):
            raise TaskSpecError(
                code="invalid_environment_path",
                message=(
                    "Environment observation path must "
                    "contain one or more non-empty keys."
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": (
                self.fixture_id
            ),
            "path": list(
                self.path
            ),
        }


@dataclass(frozen=True)
class EnvironmentFixture:
    """Deterministic environmental state for one task."""

    initial_state: dict[
        str,
        Any,
    ]

    observations: tuple[
        EnvironmentObservationFixture,
        ...
    ]

    def __post_init__(self) -> None:
        _require_json(
            self.initial_state,
            field_name=(
                "environment initial_state"
            ),
        )

        ids = [
            item.fixture_id
            for item
            in self.observations
        ]

        if (
            len(
                ids
            )
            != len(
                set(
                    ids
                )
            )
        ):
            raise TaskSpecError(
                code="duplicate_fixture_id",
                message=(
                    "Environment observation fixture IDs "
                    "must be unique."
                ),
            )

        for observation in self.observations:
            _read_path(
                self.initial_state,
                observation.path,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_state": deepcopy(
                self.initial_state
            ),
            "observations": [
                item.to_dict()
                for item
                in self.observations
            ],
        }


@dataclass(frozen=True)
class InterventionMutationTemplate:
    """
    Logical intervention mutation.

    Runtime source IDs are intentionally not stored here because T/H
    origin event IDs are created during each baseline run. Step 9B resolves
    fixture IDs into concrete InterventionSpec targets.
    """

    operation: InterventionOperation

    target_fixture_id: (
        str | None
    )

    target_path: (
        tuple[str, ...] | None
    )

    replacement_value: Any

    length_match_required: bool

    def __post_init__(self) -> None:
        if (
            self.target_fixture_id
            is None
            and self.target_path
            is None
        ):
            raise TaskSpecError(
                code="missing_intervention_target",
                message=(
                    "Intervention mutation requires "
                    "target_fixture_id or target_path."
                ),
            )

        if (
            self.target_fixture_id
            is not None
            and self.target_path
            is not None
        ):
            raise TaskSpecError(
                code="ambiguous_intervention_target",
                message=(
                    "Intervention mutation may not define "
                    "both target_fixture_id and target_path."
                ),
            )

        if (
            self.target_fixture_id
            is not None
        ):
            _require_identifier(
                self.target_fixture_id,
                field_name=(
                    "intervention target_fixture_id"
                ),
            )

        if (
            self.target_path
            is not None
            and (
                not self.target_path
                or not all(
                    isinstance(
                        value,
                        str,
                    )
                    and value
                    for value
                    in self.target_path
                )
            )
        ):
            raise TaskSpecError(
                code="invalid_intervention_target_path",
                message=(
                    "Intervention target_path must contain "
                    "one or more non-empty keys."
                ),
            )

        if (
            self.operation
            == InterventionOperation.REMOVE
        ):
            if (
                self.replacement_value
                is not None
            ):
                raise TaskSpecError(
                    code="remove_has_replacement",
                    message=(
                        "REMOVE intervention mutations "
                        "must not define replacement_value."
                    ),
                )

        else:
            _require_json(
                self.replacement_value,
                field_name=(
                    "intervention replacement_value"
                ),
            )

        if not isinstance(
            self.length_match_required,
            bool,
        ):
            raise TaskSpecError(
                code="invalid_length_match_flag",
                message=(
                    "length_match_required must be boolean."
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": (
                self.operation.value
            ),
            "target_fixture_id": (
                self.target_fixture_id
            ),
            "target_path": (
                list(
                    self.target_path
                )
                if (
                    self.target_path
                    is not None
                )
                else None
            ),
            "replacement_value": (
                deepcopy(
                    self.replacement_value
                )
            ),
            "length_match_required": (
                self.length_match_required
            ),
        }


@dataclass(frozen=True)
class InterventionPairTemplate:
    """One targeted/source-matched-sham task-level intervention plan."""

    pair_id: str

    source: SourceClass

    eligible_conditions: tuple[
        Condition,
        ...
    ]

    targeted: (
        InterventionMutationTemplate
    )

    sham: (
        InterventionMutationTemplate
    )

    def __post_init__(self) -> None:
        _require_identifier(
            self.pair_id,
            field_name=(
                "intervention pair_id"
            ),
        )

        if (
            self.source
            not in {
                SourceClass.P,
                SourceClass.T,
                SourceClass.H,
                SourceClass.M,
                SourceClass.E,
            }
        ):
            raise TaskSpecError(
                code="invalid_intervention_source",
                message=(
                    "Intervention pair source must be "
                    "P, T, H, M, or E."
                ),
            )

        if not self.eligible_conditions:
            raise TaskSpecError(
                code="missing_intervention_conditions",
                message=(
                    "Intervention pair requires at least "
                    "one eligible condition."
                ),
            )

        if (
            len(
                self.eligible_conditions
            )
            != len(
                set(
                    self.eligible_conditions
                )
            )
        ):
            raise TaskSpecError(
                code="duplicate_intervention_condition",
                message=(
                    "Intervention eligible_conditions "
                    "must be unique."
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": (
                self.pair_id
            ),
            "source": (
                self.source.value
            ),
            "eligible_conditions": [
                condition.value
                for condition
                in self.eligible_conditions
            ],
            "targeted": (
                self.targeted.to_dict()
            ),
            "sham": (
                self.sham.to_dict()
            ),
        }


@dataclass(frozen=True)
class FocalDecisionSpec:
    """One preregistered focal consequential decision."""

    schema: FocalActionSchema

    expected_by_condition: dict[
        Condition,
        dict[str, Any],
    ]

    def __post_init__(self) -> None:
        if not self.expected_by_condition:
            raise TaskSpecError(
                code="missing_success_criterion",
                message=(
                    "Focal decision requires an expected "
                    "normalized action for each task condition."
                ),
            )

        parser = (
            StructuredActionParser()
        )

        for (
            condition,
            expected,
        ) in (
            self.expected_by_condition
            .items()
        ):
            _require_json(
                expected,
                field_name=(
                    "expected normalized action"
                ),
            )

            try:
                parser.parse(
                    raw_text=(
                        canonical_json(
                            expected
                        )
                    ),
                    schema=self.schema,
                )

            except ActionParseError as exc:
                raise TaskSpecError(
                    code="invalid_expected_action",
                    message=(
                        f"Expected action for "
                        f"{condition.value} does not "
                        "match the focal-action schema: "
                        f"{exc.message}"
                    ),
                ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": (
                self.schema.to_dict()
            ),
            "expected_by_condition": {
                condition.value: (
                    deepcopy(
                        expected
                    )
                )
                for (
                    condition,
                    expected
                )
                in (
                    self
                    .expected_by_condition
                    .items()
                )
            },
        }


@dataclass(frozen=True)
class ExperimentalTaskSpec:
    """Validated machine-readable specification for one task."""

    task_id: str
    task_version: str

    family: TaskFamily

    task_classification: (
        TaskClassification
    )

    conditions: tuple[
        Condition,
        ...
    ]

    task_prompt: str

    focal_decision: (
        FocalDecisionSpec
    )

    available_sources: dict[
        Condition,
        tuple[
            SourceClass,
            ...
        ],
    ]

    tool_fixtures: tuple[
        ToolFixture,
        ...
    ]

    history_fixtures: tuple[
        HistoryFixture,
        ...
    ]

    memory_fixtures: tuple[
        MemoryFixture,
        ...
    ]

    environment: (
        EnvironmentFixture | None
    )

    intervention_pairs: tuple[
        InterventionPairTemplate,
        ...
    ]

    design_intent_sources: tuple[
        SourceClass,
        ...
    ]

    empirical_status: str

    notes: str | None

    def __post_init__(self) -> None:
        _require_identifier(
            self.task_id,
            field_name="task_id",
        )

        _require_identifier(
            self.task_version,
            field_name="task_version",
        )

        if (
            not isinstance(
                self.task_prompt,
                str,
            )
            or not self.task_prompt
        ):
            raise TaskSpecError(
                code="invalid_task_prompt",
                message=(
                    "task_prompt must be a "
                    "non-empty string."
                ),
            )

        if not self.conditions:
            raise TaskSpecError(
                code="missing_conditions",
                message=(
                    "Task must define at least "
                    "one condition."
                ),
            )

        if (
            len(
                self.conditions
            )
            != len(
                set(
                    self.conditions
                )
            )
        ):
            raise TaskSpecError(
                code="duplicate_condition",
                message=(
                    "Task conditions must be unique."
                ),
            )

        if (
            set(
                self.focal_decision
                .expected_by_condition
                .keys()
            )
            != set(
                self.conditions
            )
        ):
            raise TaskSpecError(
                code="success_condition_mismatch",
                message=(
                    "expected_by_condition must define "
                    "exactly the task's conditions."
                ),
            )

        if (
            set(
                self.available_sources
                .keys()
            )
            != set(
                self.conditions
            )
        ):
            raise TaskSpecError(
                code="source_condition_mismatch",
                message=(
                    "available_sources must define "
                    "exactly the task's conditions."
                ),
            )

        self._validate_sources()
        self._validate_fixtures()
        self._validate_interventions()
        self._validate_design_intent()

    def _validate_sources(
        self,
    ) -> None:
        for condition in self.conditions:
            sources = (
                self.available_sources[
                    condition
                ]
            )

            if (
                len(
                    sources
                )
                != len(
                    set(
                        sources
                    )
                )
            ):
                raise TaskSpecError(
                    code="duplicate_source",
                    message=(
                        f"available_sources for "
                        f"{condition.value} contains "
                        "duplicates."
                    ),
                )

            if (
                SourceClass.P
                not in sources
            ):
                raise TaskSpecError(
                    code="prompt_source_missing",
                    message=(
                        f"{condition.value} must "
                        "contain P."
                    ),
                )

            allowed = (
                _CONDITION_SOURCE_CAPABILITIES[
                    condition
                ]
            )

            illegal = (
                set(
                    sources
                )
                - allowed
            )

            if illegal:
                raise TaskSpecError(
                    code="condition_source_violation",
                    message=(
                        f"{condition.value} may not "
                        "expose source(s): "
                        + ", ".join(
                            sorted(
                                source.value
                                for source
                                in illegal
                            )
                        )
                    ),
                )

            if (
                SourceClass.E
                in sources
                and self.environment
                is None
            ):
                raise TaskSpecError(
                    code="environment_source_without_state",
                    message=(
                        f"{condition.value} exposes E "
                        "but the task defines no environment."
                    ),
                )

    def _validate_fixtures(
        self,
    ) -> None:
        ids: list[str] = []

        ids.extend(
            fixture.fixture_id
            for fixture
            in self.tool_fixtures
        )

        ids.extend(
            fixture.fixture_id
            for fixture
            in self.history_fixtures
        )

        ids.extend(
            fixture.fixture_id
            for fixture
            in self.memory_fixtures
        )

        if self.environment is not None:
            ids.extend(
                fixture.fixture_id
                for fixture
                in (
                    self.environment
                    .observations
                )
            )

        if (
            len(
                ids
            )
            != len(
                set(
                    ids
                )
            )
        ):
            raise TaskSpecError(
                code="duplicate_fixture_id",
                message=(
                    "Fixture IDs must be globally "
                    "unique within a task."
                ),
            )

        if (
            self.tool_fixtures
            and not any(
                SourceClass.T
                in self.available_sources[
                    condition
                ]
                for condition
                in self.conditions
            )
        ):
            raise TaskSpecError(
                code="unused_tool_fixtures",
                message=(
                    "Task defines tool fixtures but "
                    "never exposes T."
                ),
            )

        if (
            self.history_fixtures
            and not any(
                SourceClass.H
                in self.available_sources[
                    condition
                ]
                for condition
                in self.conditions
            )
        ):
            raise TaskSpecError(
                code="unused_history_fixtures",
                message=(
                    "Task defines history fixtures but "
                    "never exposes H."
                ),
            )

        if (
            self.memory_fixtures
            and not any(
                SourceClass.M
                in self.available_sources[
                    condition
                ]
                for condition
                in self.conditions
            )
        ):
            raise TaskSpecError(
                code="unused_memory_fixtures",
                message=(
                    "Task defines memory fixtures but "
                    "never exposes M."
                ),
            )

    def _validate_interventions(
        self,
    ) -> None:
        pair_ids = [
            pair.pair_id
            for pair
            in self.intervention_pairs
        ]

        if (
            len(
                pair_ids
            )
            != len(
                set(
                    pair_ids
                )
            )
        ):
            raise TaskSpecError(
                code="duplicate_intervention_pair",
                message=(
                    "Intervention pair IDs must "
                    "be unique within a task."
                ),
            )

        fixture_source_map = (
            self.fixture_source_map()
        )

        for pair in self.intervention_pairs:
            for condition in (
                pair.eligible_conditions
            ):
                if (
                    condition
                    not in self.conditions
                ):
                    raise TaskSpecError(
                        code="intervention_condition_not_in_task",
                        message=(
                            f"Pair {pair.pair_id!r} "
                            f"references condition "
                            f"{condition.value} that "
                            "the task does not run."
                        ),
                    )

                if (
                    pair.source
                    not in (
                        self.available_sources[
                            condition
                        ]
                    )
                ):
                    raise TaskSpecError(
                        code="intervention_source_unavailable",
                        message=(
                            f"Pair {pair.pair_id!r} "
                            f"targets {pair.source.value} "
                            f"under {condition.value}, "
                            "but that source is not "
                            "model-visible."
                        ),
                    )

            self._validate_mutation_target(
                pair=pair,
                mutation=(
                    pair.targeted
                ),
                role="targeted",
                fixture_source_map=(
                    fixture_source_map
                ),
            )

            self._validate_mutation_target(
                pair=pair,
                mutation=(
                    pair.sham
                ),
                role="sham",
                fixture_source_map=(
                    fixture_source_map
                ),
            )

    def _validate_mutation_target(
        self,
        *,
        pair: InterventionPairTemplate,
        mutation: InterventionMutationTemplate,
        role: str,
        fixture_source_map: dict[
            str,
            SourceClass,
        ],
    ) -> None:
        if pair.source == SourceClass.P:
            if (
                mutation.target_fixture_id
                != "prompt"
                or mutation.target_path
                is not None
            ):
                raise TaskSpecError(
                    code="invalid_prompt_intervention_target",
                    message=(
                        f"{role} P intervention must "
                        "use target_fixture_id='prompt'."
                    ),
                )

            return

        if pair.source == SourceClass.E:
            if (
                mutation.target_path
                is None
            ):
                raise TaskSpecError(
                    code="invalid_environment_intervention_target",
                    message=(
                        f"{role} E intervention "
                        "requires target_path."
                    ),
                )

            if self.environment is None:
                raise TaskSpecError(
                    code="environment_intervention_without_state",
                    message=(
                        "E intervention requires "
                        "an environment."
                    ),
                )

            _read_path(
                self.environment
                .initial_state,
                mutation.target_path,
            )

            return

        fixture_id = (
            mutation.target_fixture_id
        )

        if fixture_id is None:
            raise TaskSpecError(
                code="missing_fixture_target",
                message=(
                    f"{role} {pair.source.value} "
                    "intervention requires "
                    "target_fixture_id."
                ),
            )

        actual_source = (
            fixture_source_map.get(
                fixture_id
            )
        )

        if actual_source is None:
            raise TaskSpecError(
                code="intervention_fixture_not_found",
                message=(
                    f"Intervention fixture "
                    f"{fixture_id!r} does not exist."
                ),
            )

        if (
            actual_source
            != pair.source
        ):
            raise TaskSpecError(
                code="intervention_fixture_source_mismatch",
                message=(
                    f"Fixture {fixture_id!r} belongs "
                    f"to {actual_source.value}, not "
                    f"{pair.source.value}."
                ),
            )

    def _validate_design_intent(
        self,
    ) -> None:
        if (
            len(
                self.design_intent_sources
            )
            != len(
                set(
                    self.design_intent_sources
                )
            )
        ):
            raise TaskSpecError(
                code="duplicate_design_intent_source",
                message=(
                    "design_intent_sources must "
                    "be unique."
                ),
            )

        if (
            self.empirical_status
            != (
                "determined_only_by_intervention"
            )
        ):
            raise TaskSpecError(
                code="invalid_empirical_status",
                message=(
                    "empirical_status must be "
                    "'determined_only_by_intervention'."
                ),
            )

    def fixture_source_map(
        self,
    ) -> dict[
        str,
        SourceClass,
    ]:
        values: dict[
            str,
            SourceClass,
        ] = {}

        for fixture in (
            self.tool_fixtures
        ):
            values[
                fixture.fixture_id
            ] = SourceClass.T

        for fixture in (
            self.history_fixtures
        ):
            values[
                fixture.fixture_id
            ] = SourceClass.H

        for fixture in (
            self.memory_fixtures
        ):
            values[
                fixture.fixture_id
            ] = SourceClass.M

        if self.environment is not None:
            for fixture in (
                self.environment
                .observations
            ):
                values[
                    fixture.fixture_id
                ] = SourceClass.E

        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_version": (
                self.task_version
            ),
            "family": (
                self.family.value
            ),
            "class": (
                self.task_classification
                .value
            ),
            "conditions": [
                condition.value
                for condition
                in self.conditions
            ],
            "task_prompt": (
                self.task_prompt
            ),
            "focal_decision": (
                self.focal_decision
                .to_dict()
            ),
            "available_sources": {
                condition.value: [
                    source.value
                    for source
                    in sources
                ]
                for (
                    condition,
                    sources
                )
                in (
                    self
                    .available_sources
                    .items()
                )
            },
            "tool_fixtures": [
                fixture.to_dict()
                for fixture
                in self.tool_fixtures
            ],
            "history_fixtures": [
                fixture.to_dict()
                for fixture
                in self.history_fixtures
            ],
            "memory_fixtures": [
                fixture.to_dict()
                for fixture
                in self.memory_fixtures
            ],
            "environment": (
                self.environment.to_dict()
                if (
                    self.environment
                    is not None
                )
                else None
            ),
            "intervention_pairs": [
                pair.to_dict()
                for pair
                in self.intervention_pairs
            ],
            "causal_ground_truth": {
                "design_intent": [
                    source.value
                    for source
                    in (
                        self
                        .design_intent_sources
                    )
                ],
                "empirical_status": (
                    self.empirical_status
                ),
            },
            "notes": self.notes,
        }


@dataclass(frozen=True)
class TaskSuite:
    """Validated task-file container."""

    metadata: TaskFileMetadata

    schema_version: str

    suite_id: str
    suite_version: str

    tasks: tuple[
        ExperimentalTaskSpec,
        ...
    ]

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != TASK_SPEC_SCHEMA_VERSION
        ):
            raise TaskSpecError(
                code="unsupported_task_schema_version",
                message=(
                    f"Expected task schema version "
                    f"{TASK_SPEC_SCHEMA_VERSION!r}; "
                    f"received "
                    f"{self.schema_version!r}."
                ),
            )

        _require_identifier(
            self.suite_id,
            field_name="suite_id",
        )

        _require_identifier(
            self.suite_version,
            field_name="suite_version",
        )

        if not self.tasks:
            raise TaskSpecError(
                code="empty_task_suite",
                message=(
                    "Task suite must contain "
                    "at least one task."
                ),
            )

        ids = [
            task.task_id
            for task
            in self.tasks
        ]

        if (
            len(
                ids
            )
            != len(
                set(
                    ids
                )
            )
        ):
            raise TaskSpecError(
                code="duplicate_task_id",
                message=(
                    "Task IDs must be unique "
                    "within a suite."
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
            "suite_id": (
                self.suite_id
            ),
            "suite_version": (
                self.suite_version
            ),
            "tasks": [
                task.to_dict()
                for task
                in self.tasks
            ],
        }


def canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_identifier(
    value: Any,
    *,
    field_name: str,
) -> None:
    if (
        not isinstance(
            value,
            str,
        )
        or not value
    ):
        raise TaskSpecError(
            code="invalid_identifier",
            message=(
                f"{field_name} must be a "
                "non-empty string."
            ),
        )


def _require_json(
    value: Any,
    *,
    field_name: str,
) -> None:
    try:
        canonical_json(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TaskSpecError(
            code="non_serializable_task_value",
            message=(
                f"{field_name} must be "
                "losslessly representable "
                "as JSON."
            ),
        ) from exc


def _read_path(
    state: dict[
        str,
        Any,
    ],
    path: tuple[
        str,
        ...
    ],
) -> Any:
    current: Any = state

    for key in path:
        if (
            not isinstance(
                current,
                dict,
            )
            or key
            not in current
        ):
            raise TaskSpecError(
                code="task_environment_path_not_found",
                message=(
                    f"Environment path "
                    f"{list(path)!r} "
                    "does not exist."
                ),
            )

        current = current[
            key
        ]

    return deepcopy(
        current
    )