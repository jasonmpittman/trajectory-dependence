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
from dataclasses import dataclass
from typing import Any

from .actions import (
    FocalActionSchema,
    NormalizedAction,
    canonicalize_action,
    value_matches_json_type,
)


class ActionParseError(ValueError):
    """Defined failure parsing or validating a focal action."""

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
class ActionParseResult:
    """
    Successful parse of one raw model generation into a normalized action.
    """

    raw_text: str

    parsed_payload: dict[str, Any]

    normalized_action: NormalizedAction

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "parsed_payload": deepcopy(
                self.parsed_payload
            ),
            "normalized_action": (
                self.normalized_action.to_dict()
            ),
        }


class StructuredActionParser:
    """
    Strict JSON focal-action parser.

    The parser does not:

    - extract JSON from surrounding prose;
    - remove Markdown fences;
    - repair malformed JSON;
    - use an LLM for interpretation;
    - normalize semantic meaning.

    Leading/trailing whitespace surrounding the complete JSON document is
    permitted because JSON itself permits insignificant surrounding
    whitespace.
    """

    def parse(
        self,
        *,
        raw_text: str,
        schema: FocalActionSchema,
    ) -> ActionParseResult:
        if not isinstance(
            raw_text,
            str,
        ):
            raise ActionParseError(
                code="action_text_not_string",
                message=(
                    "Raw focal-action output must be a string."
                ),
            )

        stripped = raw_text.strip()

        if not stripped:
            raise ActionParseError(
                code="empty_action_output",
                message=(
                    "Raw focal-action output is empty."
                ),
            )

        try:
            parsed = json.loads(
                stripped
            )

        except json.JSONDecodeError as exc:
            raise ActionParseError(
                code="invalid_action_json",
                message=(
                    "Focal-action output is not one valid JSON "
                    "document."
                ),
            ) from exc

        if not isinstance(
            parsed,
            dict,
        ):
            raise ActionParseError(
                code="action_root_not_object",
                message=(
                    "Focal-action JSON root must be an object."
                ),
            )

        self._validate_fields(
            parsed=parsed,
            schema=schema,
        )

        normalized = canonicalize_action(
            focal_decision_id=(
                schema.focal_decision_id
            ),
            payload=parsed,
        )

        return ActionParseResult(
            raw_text=raw_text,
            parsed_payload=deepcopy(
                parsed
            ),
            normalized_action=normalized,
        )

    def _validate_fields(
        self,
        *,
        parsed: dict[str, Any],
        schema: FocalActionSchema,
    ) -> None:
        actual_fields = frozenset(
            parsed.keys()
        )

        missing = (
            schema.required_field_names
            - actual_fields
        )

        if missing:
            raise ActionParseError(
                code="missing_action_field",
                message=(
                    "Focal action is missing required field(s): "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                    + "."
                ),
            )

        additional = (
            actual_fields
            - schema.field_names
        )

        if (
            additional
            and not schema.allow_additional_fields
        ):
            raise ActionParseError(
                code="unexpected_action_field",
                message=(
                    "Focal action contains unexpected field(s): "
                    + ", ".join(
                        sorted(
                            additional
                        )
                    )
                    + "."
                ),
            )

        for action_field in schema.fields:
            if (
                action_field.name
                not in parsed
            ):
                continue

            value = parsed[
                action_field.name
            ]

            if not value_matches_json_type(
                value,
                action_field.json_type,
            ):
                raise ActionParseError(
                    code="action_field_type_mismatch",
                    message=(
                        f"Field {action_field.name!r} must have "
                        f"JSON type {action_field.json_type!r}."
                    ),
                )

            if (
                action_field.allowed_values
                is not None
                and value
                not in action_field.allowed_values
            ):
                raise ActionParseError(
                    code="action_field_value_not_allowed",
                    message=(
                        f"Field {action_field.name!r} contains "
                        f"disallowed value {value!r}."
                    ),
                )