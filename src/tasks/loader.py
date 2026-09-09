__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from pathlib import Path
from typing import Any

from ..interventions import (
    InterventionOperation,
)
from ..logging import (
    Condition,
    SourceClass,
    TaskClassification,
)
from ..normalization import (
    ActionField,
    FocalActionSchema,
)
from .spec import (
    EnvironmentFixture,
    EnvironmentObservationFixture,
    ExperimentalTaskSpec,
    FocalDecisionSpec,
    HistoryFixture,
    InterventionMutationTemplate,
    InterventionPairTemplate,
    MemoryFixture,
    TASK_SPEC_SCHEMA_VERSION,
    TaskFamily,
    TaskFileMetadata,
    TaskSpecError,
    TaskSuite,
    ToolFixture,
)


def load_task_suite(
    path: str | Path,
) -> TaskSuite:
    """Load and validate one JSON task suite."""

    source = Path(
        path
    )

    if not source.is_file():
        raise TaskSpecError(
            code="task_file_not_found",
            message=(
                f"Task specification file "
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
        raise TaskSpecError(
            code="invalid_task_json",
            message=(
                "Task specification is not "
                "valid JSON."
            ),
        ) from exc

    if not isinstance(
        raw,
        dict,
    ):
        raise TaskSpecError(
            code="task_file_root_not_object",
            message=(
                "Task specification root "
                "must be a JSON object."
            ),
        )

    return parse_task_suite(
        raw
    )


def parse_task_suite(
    raw: dict[
        str,
        Any,
    ],
) -> TaskSuite:
    """Parse an already-loaded task-suite dictionary."""

    _require_fields(
        raw,
        {
            "metadata",
            "schema_version",
            "suite_id",
            "suite_version",
            "tasks",
        },
        location="task suite",
    )

    metadata = _parse_metadata(
        raw[
            "metadata"
        ]
    )

    schema_version = raw[
        "schema_version"
    ]

    if (
        schema_version
        != TASK_SPEC_SCHEMA_VERSION
    ):
        raise TaskSpecError(
            code="unsupported_task_schema_version",
            message=(
                f"Expected schema_version "
                f"{TASK_SPEC_SCHEMA_VERSION!r}; "
                f"received "
                f"{schema_version!r}."
            ),
        )

    task_values = raw[
        "tasks"
    ]

    if not isinstance(
        task_values,
        list,
    ):
        raise TaskSpecError(
            code="tasks_not_array",
            message=(
                "tasks must be a JSON array."
            ),
        )

    tasks = tuple(
        _parse_task(
            value
        )
        for value
        in task_values
    )

    return TaskSuite(
        metadata=metadata,
        schema_version=(
            schema_version
        ),
        suite_id=raw[
            "suite_id"
        ],
        suite_version=raw[
            "suite_version"
        ],
        tasks=tasks,
    )


def _parse_metadata(
    raw: Any,
) -> TaskFileMetadata:
    if not isinstance(
        raw,
        dict,
    ):
        raise TaskSpecError(
            code="invalid_task_metadata",
            message=(
                "metadata must be a JSON object."
            ),
        )

    _require_fields(
        raw,
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

    credits = raw[
        "credits"
    ]

    if not isinstance(
        credits,
        list,
    ):
        raise TaskSpecError(
            code="invalid_task_metadata",
            message=(
                "metadata.credits must "
                "be an array."
            ),
        )

    return TaskFileMetadata(
        author=raw[
            "author"
        ],
        date=raw[
            "date"
        ],
        copyright=raw[
            "copyright"
        ],
        credits=tuple(
            credits
        ),
        license=raw[
            "license"
        ],
        version=raw[
            "version"
        ],
        maintainer=raw[
            "maintainer"
        ],
        status=raw[
            "status"
        ],
    )


def _parse_task(
    raw: Any,
) -> ExperimentalTaskSpec:
    if not isinstance(
        raw,
        dict,
    ):
        raise TaskSpecError(
            code="task_not_object",
            message=(
                "Every task must be "
                "a JSON object."
            ),
        )

    _require_fields(
        raw,
        {
            "task_id",
            "task_version",
            "family",
            "class",
            "conditions",
            "task_prompt",
            "focal_decision",
            "available_sources",
            "fixtures",
            "intervention_pairs",
            "causal_ground_truth",
            "notes",
        },
        location=(
            f"task {raw.get('task_id')!r}"
        ),
    )

    try:
        family = TaskFamily(
            raw[
                "family"
            ]
        )

    except ValueError as exc:
        raise TaskSpecError(
            code="unknown_task_family",
            message=(
                f"Unknown task family "
                f"{raw['family']!r}."
            ),
        ) from exc

    try:
        classification = (
            TaskClassification(
                raw[
                    "class"
                ]
            )
        )

    except ValueError as exc:
        raise TaskSpecError(
            code="unknown_task_class",
            message=(
                f"Unknown task class "
                f"{raw['class']!r}."
            ),
        ) from exc

    conditions = tuple(
        _condition(
            value
        )
        for value
        in _require_list(
            raw[
                "conditions"
            ],
            field_name="conditions",
        )
    )

    focal_decision = (
        _parse_focal_decision(
            raw[
                "focal_decision"
            ]
        )
    )

    available_sources = (
        _parse_available_sources(
            raw[
                "available_sources"
            ]
        )
    )

    fixtures = raw[
        "fixtures"
    ]

    if not isinstance(
        fixtures,
        dict,
    ):
        raise TaskSpecError(
            code="fixtures_not_object",
            message=(
                "fixtures must be "
                "a JSON object."
            ),
        )

    _require_fields(
        fixtures,
        {
            "tools",
            "history",
            "memory",
            "environment",
        },
        location="fixtures",
    )

    tool_fixtures = tuple(
        _parse_tool_fixture(
            value
        )
        for value
        in _require_list(
            fixtures[
                "tools"
            ],
            field_name=(
                "fixtures.tools"
            ),
        )
    )

    history_fixtures = tuple(
        _parse_history_fixture(
            value
        )
        for value
        in _require_list(
            fixtures[
                "history"
            ],
            field_name=(
                "fixtures.history"
            ),
        )
    )

    memory_fixtures = tuple(
        _parse_memory_fixture(
            value
        )
        for value
        in _require_list(
            fixtures[
                "memory"
            ],
            field_name=(
                "fixtures.memory"
            ),
        )
    )

    environment = (
        _parse_environment_fixture(
            fixtures[
                "environment"
            ]
        )
    )

    intervention_pairs = tuple(
        _parse_intervention_pair(
            value
        )
        for value
        in _require_list(
            raw[
                "intervention_pairs"
            ],
            field_name=(
                "intervention_pairs"
            ),
        )
    )

    causal = raw[
        "causal_ground_truth"
    ]

    if not isinstance(
        causal,
        dict,
    ):
        raise TaskSpecError(
            code="causal_ground_truth_not_object",
            message=(
                "causal_ground_truth must "
                "be a JSON object."
            ),
        )

    _require_fields(
        causal,
        {
            "design_intent",
            "empirical_status",
        },
        location=(
            "causal_ground_truth"
        ),
    )

    design_intent = tuple(
        _source(
            value
        )
        for value
        in _require_list(
            causal[
                "design_intent"
            ],
            field_name=(
                "causal_ground_truth.design_intent"
            ),
        )
    )

    notes = raw[
        "notes"
    ]

    if (
        notes is not None
        and not isinstance(
            notes,
            str,
        )
    ):
        raise TaskSpecError(
            code="invalid_task_notes",
            message=(
                "notes must be a "
                "string or null."
            ),
        )

    return ExperimentalTaskSpec(
        task_id=raw[
            "task_id"
        ],
        task_version=raw[
            "task_version"
        ],
        family=family,
        task_classification=(
            classification
        ),
        conditions=conditions,
        task_prompt=raw[
            "task_prompt"
        ],
        focal_decision=(
            focal_decision
        ),
        available_sources=(
            available_sources
        ),
        tool_fixtures=(
            tool_fixtures
        ),
        history_fixtures=(
            history_fixtures
        ),
        memory_fixtures=(
            memory_fixtures
        ),
        environment=environment,
        intervention_pairs=(
            intervention_pairs
        ),
        design_intent_sources=(
            design_intent
        ),
        empirical_status=(
            causal[
                "empirical_status"
            ]
        ),
        notes=notes,
    )


def _parse_focal_decision(
    raw: Any,
) -> FocalDecisionSpec:
    if not isinstance(
        raw,
        dict,
    ):
        raise TaskSpecError(
            code="focal_decision_not_object",
            message=(
                "focal_decision must "
                "be a JSON object."
            ),
        )

    _require_fields(
        raw,
        {
            "id",
            "fields",
            "success_by_condition",
        },
        location="focal_decision",
    )

    fields_raw = _require_list(
        raw[
            "fields"
        ],
        field_name=(
            "focal_decision.fields"
        ),
    )

    fields: list[
        ActionField
    ] = []

    for value in fields_raw:
        if not isinstance(
            value,
            dict,
        ):
            raise TaskSpecError(
                code="action_field_not_object",
                message=(
                    "Each focal action field "
                    "must be a JSON object."
                ),
            )

        _require_fields(
            value,
            {
                "name",
                "json_type",
                "allowed_values",
            },
            location=(
                "focal action field"
            ),
        )

        allowed = _require_list(
            value[
                "allowed_values"
            ],
            field_name=(
                "allowed_values"
            ),
        )

        fields.append(
            ActionField(
                name=value[
                    "name"
                ],
                json_type=value[
                    "json_type"
                ],
                allowed_values=tuple(
                    allowed
                ),
            )
        )

    schema = FocalActionSchema(
        focal_decision_id=raw[
            "id"
        ],
        fields=tuple(
            fields
        ),
    )

    expected_raw = raw[
        "success_by_condition"
    ]

    if not isinstance(
        expected_raw,
        dict,
    ):
        raise TaskSpecError(
            code="success_not_object",
            message=(
                "success_by_condition must "
                "be a JSON object."
            ),
        )

    expected = {
        _condition(
            key
        ): value
        for (
            key,
            value
        )
        in expected_raw.items()
    }

    return FocalDecisionSpec(
        schema=schema,
        expected_by_condition=(
            expected
        ),
    )


def _parse_available_sources(
    raw: Any,
) -> dict[
    Condition,
    tuple[
        SourceClass,
        ...
    ],
]:
    if not isinstance(
        raw,
        dict,
    ):
        raise TaskSpecError(
            code="available_sources_not_object",
            message=(
                "available_sources must "
                "be a JSON object."
            ),
        )

    return {
        _condition(
            condition
        ): tuple(
            _source(
                value
            )
            for value
            in _require_list(
                sources,
                field_name=(
                    f"available_sources."
                    f"{condition}"
                ),
            )
        )
        for (
            condition,
            sources
        )
        in raw.items()
    }


def _parse_tool_fixture(
    raw: Any,
) -> ToolFixture:
    if not isinstance(
        raw,
        dict,
    ):
        raise TaskSpecError(
            code="tool_fixture_not_object",
            message=(
                "Tool fixture must "
                "be a JSON object."
            ),
        )

    _require_fields(
        raw,
        {
            "fixture_id",
            "tool_name",
            "request",
            "state",
            "metadata",
        },
        location="tool fixture",
    )

    return ToolFixture(
        fixture_id=raw[
            "fixture_id"
        ],
        tool_name=raw[
            "tool_name"
        ],
        request=raw[
            "request"
        ],
        state=raw[
            "state"
        ],
        metadata=raw[
            "metadata"
        ],
    )


def _parse_history_fixture(
    raw: Any,
) -> HistoryFixture:
    if not isinstance(
        raw,
        dict,
    ):
        raise TaskSpecError(
            code="history_fixture_not_object",
            message=(
                "History fixture must "
                "be a JSON object."
            ),
        )

    _require_fields(
        raw,
        {
            "fixture_id",
            "content",
            "origin_source",
            "metadata",
        },
        location="history fixture",
    )

    origin_raw = raw[
        "origin_source"
    ]

    origin_source = (
        None
        if origin_raw
        is None
        else _source(
            origin_raw
        )
    )

    return HistoryFixture(
        fixture_id=raw[
            "fixture_id"
        ],
        content=raw[
            "content"
        ],
        origin_source=(
            origin_source
        ),
        metadata=raw[
            "metadata"
        ],
    )


def _parse_memory_fixture(
    raw: Any,
) -> MemoryFixture:
    if not isinstance(
        raw,
        dict,
    ):
        raise TaskSpecError(
            code="memory_fixture_not_object",
            message=(
                "Memory fixture must "
                "be a JSON object."
            ),
        )

    _require_fields(
        raw,
        {
            "fixture_id",
            "key",
            "value",
            "metadata",
        },
        location="memory fixture",
    )

    return MemoryFixture(
        fixture_id=raw[
            "fixture_id"
        ],
        key=raw[
            "key"
        ],
        value=raw[
            "value"
        ],
        metadata=raw[
            "metadata"
        ],
    )


def _parse_environment_fixture(
    raw: Any,
) -> EnvironmentFixture | None:
    if raw is None:
        return None

    if not isinstance(
        raw,
        dict,
    ):
        raise TaskSpecError(
            code="environment_fixture_not_object",
            message=(
                "fixtures.environment must be "
                "a JSON object or null."
            ),
        )

    _require_fields(
        raw,
        {
            "initial_state",
            "observations",
        },
        location="environment fixture",
    )

    observations = tuple(
        _parse_environment_observation(
            value
        )
        for value
        in _require_list(
            raw[
                "observations"
            ],
            field_name=(
                "environment observations"
            ),
        )
    )

    return EnvironmentFixture(
        initial_state=raw[
            "initial_state"
        ],
        observations=observations,
    )


def _parse_environment_observation(
    raw: Any,
) -> EnvironmentObservationFixture:
    if not isinstance(
        raw,
        dict,
    ):
        raise TaskSpecError(
            code="environment_observation_not_object",
            message=(
                "Environment observation must "
                "be a JSON object."
            ),
        )

    _require_fields(
        raw,
        {
            "fixture_id",
            "path",
        },
        location=(
            "environment observation"
        ),
    )

    return EnvironmentObservationFixture(
        fixture_id=raw[
            "fixture_id"
        ],
        path=tuple(
            _require_list(
                raw[
                    "path"
                ],
                field_name=(
                    "environment path"
                ),
            )
        ),
    )


def _parse_intervention_pair(
    raw: Any,
) -> InterventionPairTemplate:
    if not isinstance(
        raw,
        dict,
    ):
        raise TaskSpecError(
            code="intervention_pair_not_object",
            message=(
                "Intervention pair must "
                "be a JSON object."
            ),
        )

    _require_fields(
        raw,
        {
            "pair_id",
            "source",
            "eligible_conditions",
            "targeted",
            "sham",
        },
        location="intervention pair",
    )

    return InterventionPairTemplate(
        pair_id=raw[
            "pair_id"
        ],
        source=_source(
            raw[
                "source"
            ]
        ),
        eligible_conditions=tuple(
            _condition(
                value
            )
            for value
            in _require_list(
                raw[
                    "eligible_conditions"
                ],
                field_name=(
                    "eligible_conditions"
                ),
            )
        ),
        targeted=(
            _parse_mutation(
                raw[
                    "targeted"
                ]
            )
        ),
        sham=(
            _parse_mutation(
                raw[
                    "sham"
                ]
            )
        ),
    )


def _parse_mutation(
    raw: Any,
) -> InterventionMutationTemplate:
    if not isinstance(
        raw,
        dict,
    ):
        raise TaskSpecError(
            code="intervention_mutation_not_object",
            message=(
                "Intervention mutation must "
                "be a JSON object."
            ),
        )

    _require_fields(
        raw,
        {
            "operation",
            "target_fixture_id",
            "target_path",
            "replacement_value",
            "length_match_required",
        },
        location=(
            "intervention mutation"
        ),
    )

    try:
        operation = (
            InterventionOperation(
                raw[
                    "operation"
                ]
            )
        )

    except ValueError as exc:
        raise TaskSpecError(
            code="unknown_intervention_operation",
            message=(
                f"Unknown intervention operation "
                f"{raw['operation']!r}."
            ),
        ) from exc

    target_path_raw = raw[
        "target_path"
    ]

    target_path = (
        None
        if target_path_raw
        is None
        else tuple(
            _require_list(
                target_path_raw,
                field_name=(
                    "target_path"
                ),
            )
        )
    )

    return InterventionMutationTemplate(
        operation=operation,
        target_fixture_id=(
            raw[
                "target_fixture_id"
            ]
        ),
        target_path=target_path,
        replacement_value=(
            raw[
                "replacement_value"
            ]
        ),
        length_match_required=(
            raw[
                "length_match_required"
            ]
        ),
    )


def _condition(
    value: Any,
) -> Condition:
    try:
        return Condition(
            value
        )

    except (
        ValueError,
        TypeError,
    ) as exc:
        raise TaskSpecError(
            code="unknown_condition",
            message=(
                f"Unknown condition "
                f"{value!r}."
            ),
        ) from exc


def _source(
    value: Any,
) -> SourceClass:
    try:
        return SourceClass(
            value
        )

    except (
        ValueError,
        TypeError,
    ) as exc:
        raise TaskSpecError(
            code="unknown_source",
            message=(
                f"Unknown source "
                f"{value!r}."
            ),
        ) from exc


def _require_list(
    value: Any,
    *,
    field_name: str,
) -> list[Any]:
    if not isinstance(
        value,
        list,
    ):
        raise TaskSpecError(
            code="expected_array",
            message=(
                f"{field_name} must "
                "be a JSON array."
            ),
        )

    return value


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

        raise TaskSpecError(
            code="task_schema_field_mismatch",
            message=(
                f"{location} fields do not "
                "match the schema: "
                + "; ".join(
                    details
                )
            ),
        )