__author__ = "Jason M. Pittman"
__date__ = "August 23, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any


JSON_TYPES = frozenset(
    {
        "string",
        "integer",
        "number",
        "boolean",
        "array",
        "object",
        "null",
        "any",
    }
)


class ActionSchemaError(ValueError):
    """Defined failure in a focal-action schema."""

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
class ActionField:
    """
    Declarative field definition for one structured focal action.

    `allowed_values=None` means any value of the declared JSON type is
    permitted.

    Values are compared exactly. No semantic normalization occurs.
    """

    name: str
    json_type: str

    required: bool = True

    allowed_values: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.name,
                str,
            )
            or not self.name
        ):
            raise ActionSchemaError(
                code="invalid_field_name",
                message=(
                    "Action-field name must be a non-empty string."
                ),
            )

        if self.json_type not in JSON_TYPES:
            raise ActionSchemaError(
                code="invalid_json_type",
                message=(
                    f"Unsupported JSON type {self.json_type!r} "
                    f"for action field {self.name!r}."
                ),
            )

        if self.allowed_values is not None:
            values = tuple(
                deepcopy(
                    self.allowed_values
                )
            )

            for value in values:
                _validate_json_value(
                    value,
                    field_name=(
                        f"{self.name}.allowed_values"
                    ),
                )

                if (
                    self.json_type != "any"
                    and not value_matches_json_type(
                        value,
                        self.json_type,
                    )
                ):
                    raise ActionSchemaError(
                        code="allowed_value_type_mismatch",
                        message=(
                            f"Allowed value {value!r} does not match "
                            f"declared type {self.json_type!r} for "
                            f"field {self.name!r}."
                        ),
                    )

            object.__setattr__(
                self,
                "allowed_values",
                values,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "json_type": self.json_type,
            "required": self.required,
            "allowed_values": (
                deepcopy(
                    list(
                        self.allowed_values
                    )
                )
                if self.allowed_values
                is not None
                else None
            ),
        }


@dataclass(frozen=True)
class FocalActionSchema:
    """
    Frozen schema for one preregistered focal consequential decision.

    The schema defines syntactic/structural validity only. It does not
    encode whether the resulting action is correct or successful.
    """

    focal_decision_id: str

    fields: tuple[ActionField, ...]

    allow_additional_fields: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.focal_decision_id,
                str,
            )
            or not self.focal_decision_id
        ):
            raise ActionSchemaError(
                code="invalid_focal_decision_id",
                message=(
                    "focal_decision_id must be a non-empty string."
                ),
            )

        if not self.fields:
            raise ActionSchemaError(
                code="empty_action_schema",
                message=(
                    "A focal-action schema must define at least one "
                    "field."
                ),
            )

        names = [
            field.name
            for field in self.fields
        ]

        if len(names) != len(set(names)):
            raise ActionSchemaError(
                code="duplicate_action_field",
                message=(
                    "Focal-action schema contains duplicate field "
                    "names."
                ),
            )

        metadata_copy = deepcopy(
            self.metadata
        )

        _validate_json_value(
            metadata_copy,
            field_name="metadata",
        )

        object.__setattr__(
            self,
            "metadata",
            metadata_copy,
        )

    @property
    def required_field_names(
        self,
    ) -> frozenset[str]:
        return frozenset(
            field.name
            for field in self.fields
            if field.required
        )

    @property
    def field_names(
        self,
    ) -> frozenset[str]:
        return frozenset(
            field.name
            for field in self.fields
        )

    def field(
        self,
        name: str,
    ) -> ActionField:
        for action_field in self.fields:
            if action_field.name == name:
                return action_field

        raise ActionSchemaError(
            code="unknown_action_field",
            message=(
                f"Action schema contains no field named {name!r}."
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "focal_decision_id": (
                self.focal_decision_id
            ),
            "fields": [
                action_field.to_dict()
                for action_field in self.fields
            ],
            "allow_additional_fields": (
                self.allow_additional_fields
            ),
            "metadata": deepcopy(
                self.metadata
            ),
        }


@dataclass(frozen=True)
class NormalizedAction:
    """
    Canonical structured representation of one focal consequential action.

    `canonical_json` is the representation used for exact behavioral
    comparison where the task declares structured exact-action
    equivalence.
    """

    focal_decision_id: str

    payload: dict[str, Any]

    canonical_json: str
    canonical_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "focal_decision_id": (
                self.focal_decision_id
            ),
            "payload": deepcopy(
                self.payload
            ),
            "canonical_json": (
                self.canonical_json
            ),
            "canonical_sha256": (
                self.canonical_sha256
            ),
        }

    def behaviorally_equivalent(
        self,
        other: "NormalizedAction",
    ) -> bool:
        """
        Return exact structured-action equivalence.

        This is deliberately not semantic textual equivalence.
        """

        return (
            self.focal_decision_id
            == other.focal_decision_id
            and self.canonical_json
            == other.canonical_json
        )


def canonicalize_action(
    *,
    focal_decision_id: str,
    payload: dict[str, Any],
) -> NormalizedAction:
    """
    Create a canonical action after schema validation has succeeded.
    """

    copied = deepcopy(
        payload
    )

    _validate_json_value(
        copied,
        field_name="action_payload",
    )

    canonical = json.dumps(
        copied,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return NormalizedAction(
        focal_decision_id=(
            focal_decision_id
        ),
        payload=copied,
        canonical_json=canonical,
        canonical_sha256=sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest(),
    )


def value_matches_json_type(
    value: Any,
    json_type: str,
) -> bool:
    """
    Strict JSON-type matching.

    Python bool is deliberately excluded from integer/number because JSON
    booleans are distinct from JSON numbers.
    """

    if json_type == "any":
        return True

    if json_type == "string":
        return isinstance(
            value,
            str,
        )

    if json_type == "integer":
        return (
            isinstance(
                value,
                int,
            )
            and not isinstance(
                value,
                bool,
            )
        )

    if json_type == "number":
        return (
            isinstance(
                value,
                (int, float),
            )
            and not isinstance(
                value,
                bool,
            )
        )

    if json_type == "boolean":
        return isinstance(
            value,
            bool,
        )

    if json_type == "array":
        return isinstance(
            value,
            list,
        )

    if json_type == "object":
        return isinstance(
            value,
            dict,
        )

    if json_type == "null":
        return value is None

    return False


def _validate_json_value(
    value: Any,
    *,
    field_name: str,
) -> None:
    try:
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ActionSchemaError(
            code="non_serializable_action_value",
            message=(
                f"{field_name!r} must be losslessly "
                "representable as JSON."
            ),
        ) from exc