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
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any

from ..logging import (
    Condition,
    ContextElement,
    SourceClass,
)
from ..memory import MemoryRecord
from ..snapshots import PreFocalSnapshot


class InterventionKind(str, Enum):
    """Experimental role of an intervention."""

    TARGETED = "targeted"
    SHAM = "sham"


class InterventionOperation(str, Enum):
    """Supported state manipulation."""

    REPLACE = "replace"
    REMOVE = "remove"


class InterventionSchemaError(ValueError):
    """Defined failure in an intervention specification."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class InterventionSpec:
    """
    One preregistered targeted or sham intervention.

    Source-specific target semantics:

    P/T/H:
        target_id = ContextElement.source_id

    M:
        target_id = MemoryRecord.memory_id

    E:
        target_path = exact nested environmental-state path

    P supports replacement only because removing current prompt P would
    violate the focal-context construct.

    E supports replacement only in this initial experiment because the
    environmental intervention is intended to alter a state variable,
    not structurally delete part of the simulated environment.
    """

    intervention_id: str
    pair_id: str

    kind: InterventionKind
    source: SourceClass
    operation: InterventionOperation

    target_id: str | None = None
    target_path: tuple[str, ...] | None = None

    replacement_value: Any = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for field_name, value in (
            (
                "intervention_id",
                self.intervention_id,
            ),
            (
                "pair_id",
                self.pair_id,
            ),
        ):
            if (
                not isinstance(
                    value,
                    str,
                )
                or not value
            ):
                raise InterventionSchemaError(
                    code="invalid_intervention_identifier",
                    message=(
                        f"{field_name} must be a non-empty string."
                    ),
                )

        if self.source == SourceClass.E:
            if self.target_id is not None:
                raise InterventionSchemaError(
                    code="invalid_environment_target",
                    message=(
                        "E interventions use target_path, "
                        "not target_id."
                    ),
                )

            if (
                self.target_path is None
                or not self.target_path
                or not all(
                    isinstance(
                        item,
                        str,
                    )
                    and item
                    for item in self.target_path
                )
            ):
                raise InterventionSchemaError(
                    code="invalid_environment_target",
                    message=(
                        "E intervention target_path must be "
                        "a non-empty tuple of string keys."
                    ),
                )

            if (
                self.operation
                != InterventionOperation.REPLACE
            ):
                raise InterventionSchemaError(
                    code="unsupported_environment_operation",
                    message=(
                        "Initial E interventions support "
                        "replacement only."
                    ),
                )

        else:
            if (
                not isinstance(
                    self.target_id,
                    str,
                )
                or not self.target_id
            ):
                raise InterventionSchemaError(
                    code="invalid_intervention_target",
                    message=(
                        f"{self.source.value} interventions require "
                        "a non-empty target_id."
                    ),
                )

            if self.target_path is not None:
                raise InterventionSchemaError(
                    code="unexpected_target_path",
                    message=(
                        "target_path is permitted only for E."
                    ),
                )

        if (
            self.source == SourceClass.P
            and self.operation
            != InterventionOperation.REPLACE
        ):
            raise InterventionSchemaError(
                code="unsupported_prompt_operation",
                message=(
                    "P interventions support replacement only."
                ),
            )

        if (
            self.operation
            == InterventionOperation.REMOVE
            and self.replacement_value
            is not None
        ):
            raise InterventionSchemaError(
                code="remove_has_replacement",
                message=(
                    "REMOVE interventions must not define "
                    "replacement_value."
                ),
            )

        if (
            self.operation
            == InterventionOperation.REPLACE
        ):
            _validate_json(
                self.replacement_value,
                field_name="replacement_value",
            )

        metadata_copy = deepcopy(
            self.metadata
        )

        _validate_json(
            metadata_copy,
            field_name="metadata",
        )

        object.__setattr__(
            self,
            "metadata",
            metadata_copy,
        )

    def target_descriptor(
        self,
    ) -> dict[str, Any]:
        if self.source == SourceClass.E:
            return {
                "target_path": list(
                    self.target_path
                    or ()
                )
            }

        return {
            "target_id": self.target_id
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": (
                self.intervention_id
            ),
            "pair_id": self.pair_id,
            "kind": self.kind.value,
            "source": self.source.value,
            "operation": (
                self.operation.value
            ),
            "target_id": self.target_id,
            "target_path": (
                list(
                    self.target_path
                )
                if self.target_path
                is not None
                else None
            ),
            "replacement_value": (
                deepcopy(
                    self.replacement_value
                )
            ),
            "metadata": deepcopy(
                self.metadata
            ),
        }


@dataclass(frozen=True)
class CounterfactualState:
    """
    Immutable experiment-controlled working state derived from a baseline
    PreFocalSnapshot.

    This is not itself the baseline snapshot and does not claim baseline
    provenance.
    """

    baseline_snapshot_id: str

    experiment_id: str
    run_id: str
    task_id: str
    condition: Condition
    repetition_id: int

    sequence_position: int

    prompt_elements: tuple[
        ContextElement,
        ...
    ]

    tool_elements: tuple[
        ContextElement,
        ...
    ]

    history_elements: tuple[
        ContextElement,
        ...
    ]

    memory_namespace: str | None
    memory_records: tuple[
        MemoryRecord,
        ...
    ]

    environment_state: (
        dict[str, Any] | None
    )

    runtime_configuration: dict[
        str,
        Any,
    ]

    task_control_state: dict[
        str,
        Any,
    ]

    seed_metadata: dict[
        str,
        Any,
    ]

    state_checksum_sha256: str

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PreFocalSnapshot,
    ) -> "CounterfactualState":
        memory_namespace = None
        memory_records: tuple[
            MemoryRecord,
            ...
        ] = ()

        if (
            snapshot.memory_snapshot
            is not None
        ):
            memory_namespace = (
                snapshot
                .memory_snapshot
                .namespace
            )

            memory_records = tuple(
                deepcopy(
                    snapshot
                    .memory_snapshot
                    .records
                )
            )

        environment_state = (
            deepcopy(
                snapshot
                .environment_snapshot
                .state
            )
            if (
                snapshot.environment_snapshot
                is not None
            )
            else None
        )

        provisional = cls(
            baseline_snapshot_id=(
                snapshot.snapshot_id
            ),
            experiment_id=(
                snapshot.experiment_id
            ),
            run_id=snapshot.run_id,
            task_id=snapshot.task_id,
            condition=snapshot.condition,
            repetition_id=(
                snapshot.repetition_id
            ),
            sequence_position=(
                snapshot.sequence_position
            ),
            prompt_elements=tuple(
                deepcopy(
                    snapshot.prompt_elements
                )
            ),
            tool_elements=tuple(
                deepcopy(
                    snapshot.tool_elements
                )
            ),
            history_elements=tuple(
                deepcopy(
                    snapshot.history_elements
                )
            ),
            memory_namespace=(
                memory_namespace
            ),
            memory_records=(
                memory_records
            ),
            environment_state=(
                environment_state
            ),
            runtime_configuration=(
                deepcopy(
                    snapshot
                    .runtime_configuration
                )
            ),
            task_control_state=(
                deepcopy(
                    snapshot
                    .task_control_state
                )
            ),
            seed_metadata=(
                deepcopy(
                    snapshot.seed_metadata
                )
            ),
            state_checksum_sha256="",
        )

        checksum = checksum_payload(
            provisional.state_payload()
        )

        return cls(
            **{
                **provisional.__dict__,
                "state_checksum_sha256": (
                    checksum
                ),
            }
        )

    def state_payload(
        self,
    ) -> dict[str, Any]:
        return {
            "baseline_snapshot_id": (
                self.baseline_snapshot_id
            ),
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
            "sequence_position": (
                self.sequence_position
            ),
            "P": [
                element.to_dict()
                for element
                in self.prompt_elements
            ],
            "T": [
                element.to_dict()
                for element
                in self.tool_elements
            ],
            "H": [
                element.to_dict()
                for element
                in self.history_elements
            ],
            "M": (
                {
                    "namespace": (
                        self.memory_namespace
                    ),
                    "records": [
                        record.to_dict()
                        for record
                        in self.memory_records
                    ],
                }
                if (
                    self.memory_namespace
                    is not None
                )
                else None
            ),
            "E": deepcopy(
                self.environment_state
            ),
            "runtime_configuration": (
                deepcopy(
                    self.runtime_configuration
                )
            ),
            "task_control_state": (
                deepcopy(
                    self.task_control_state
                )
            ),
            "seed_metadata": deepcopy(
                self.seed_metadata
            ),
        }

    def source_payload(
        self,
        source: SourceClass,
    ) -> Any:
        if source == SourceClass.P:
            return [
                element.to_dict()
                for element
                in self.prompt_elements
            ]

        if source == SourceClass.T:
            return [
                element.to_dict()
                for element
                in self.tool_elements
            ]

        if source == SourceClass.H:
            return [
                element.to_dict()
                for element
                in self.history_elements
            ]

        if source == SourceClass.M:
            if (
                self.memory_namespace
                is None
            ):
                return None

            return {
                "namespace": (
                    self.memory_namespace
                ),
                "records": [
                    record.to_dict()
                    for record
                    in self.memory_records
                ],
            }

        if source == SourceClass.E:
            return deepcopy(
                self.environment_state
            )

        raise InterventionSchemaError(
            code="unsupported_source",
            message=(
                f"Unsupported source: "
                f"{source.value!r}."
            ),
        )

    def source_checksum(
        self,
        source: SourceClass,
    ) -> str:
        return checksum_payload(
            self.source_payload(
                source
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.state_payload(),
            "state_checksum_sha256": (
                self.state_checksum_sha256
            ),
        }


@dataclass(frozen=True)
class InterventionDiff:
    """Exact before/after mutation evidence for one intervention."""

    source: SourceClass

    operation: InterventionOperation

    target: dict[str, Any]

    before: Any
    after: Any

    before_sha256: str
    after_sha256: str

    before_serialized_chars: int
    after_serialized_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "operation": (
                self.operation.value
            ),
            "target": deepcopy(
                self.target
            ),
            "before": deepcopy(
                self.before
            ),
            "after": deepcopy(
                self.after
            ),
            "before_sha256": (
                self.before_sha256
            ),
            "after_sha256": (
                self.after_sha256
            ),
            "before_serialized_chars": (
                self.before_serialized_chars
            ),
            "after_serialized_chars": (
                self.after_serialized_chars
            ),
        }


@dataclass(frozen=True)
class InterventionApplication:
    """Validated application of one intervention to baseline state."""

    spec: InterventionSpec

    baseline_snapshot_id: str

    baseline_state_checksum: str
    counterfactual_state_checksum: str

    changed_sources: tuple[
        SourceClass,
        ...
    ]

    source_checksums_before: dict[
        str,
        str,
    ]

    source_checksums_after: dict[
        str,
        str,
    ]

    diff: InterventionDiff

    counterfactual_state: CounterfactualState

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "baseline_snapshot_id": (
                self.baseline_snapshot_id
            ),
            "baseline_state_checksum": (
                self.baseline_state_checksum
            ),
            "counterfactual_state_checksum": (
                self.counterfactual_state_checksum
            ),
            "changed_sources": [
                source.value
                for source
                in self.changed_sources
            ],
            "source_checksums_before": (
                deepcopy(
                    self.source_checksums_before
                )
            ),
            "source_checksums_after": (
                deepcopy(
                    self.source_checksums_after
                )
            ),
            "diff": self.diff.to_dict(),
            "counterfactual_state": (
                self.counterfactual_state
                .to_dict()
            ),
        }


@dataclass(frozen=True)
class InterventionPair:
    """One targeted intervention and its source-matched sham."""

    pair_id: str

    targeted: InterventionSpec
    sham: InterventionSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "targeted": (
                self.targeted.to_dict()
            ),
            "sham": (
                self.sham.to_dict()
            ),
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


def checksum_payload(
    value: Any,
) -> str:
    return sha256(
        canonical_json(
            value
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def serialized_char_count(
    value: Any,
) -> int:
    return len(
        canonical_json(
            value
        )
    )


def _validate_json(
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
        raise InterventionSchemaError(
            code="non_serializable_intervention_value",
            message=(
                f"{field_name!r} must be losslessly "
                "representable as JSON."
            ),
        ) from exc