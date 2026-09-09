__author__ = "Jason M. Pittman"
__date__ = "August 23, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .actions import FocalActionSchema


ACTION_PROMPT_VERSION = "focal-action-v1"


class ActionPromptError(ValueError):
    """Defined failure constructing a focal-action prompt."""

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
class FocalActionPrompt:
    """Deterministically rendered task prompt plus action contract."""

    version: str

    task_prompt: str
    action_contract: dict[str, Any]
    action_contract_json: str

    rendered_prompt: str
    rendered_prompt_sha256: str

    focal_decision_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task_prompt": self.task_prompt,
            "action_contract": self.action_contract,
            "action_contract_json": (
                self.action_contract_json
            ),
            "rendered_prompt": self.rendered_prompt,
            "rendered_prompt_sha256": (
                self.rendered_prompt_sha256
            ),
            "focal_decision_id": (
                self.focal_decision_id
            ),
        }


class FocalActionPromptBuilder:
    """Render a deterministic strict-JSON action instruction."""

    def build(
        self,
        *,
        task_prompt: str,
        schema: FocalActionSchema,
    ) -> FocalActionPrompt:
        if (
            not isinstance(task_prompt, str)
            or not task_prompt
        ):
            raise ActionPromptError(
                code="invalid_task_prompt",
                message=(
                    "task_prompt must be a non-empty string."
                ),
            )

        contract = self._build_contract(
            schema
        )

        contract_json = json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )

        instruction = (
            "Return the focal decision as exactly one JSON object "
            "matching the contract below.\n"
            "Return only the fields defined under \"properties\".\n"
            "Return no text before or after the JSON object.\n"
            "Do not use Markdown code fences.\n"
            "Do not include an explanation or commentary.\n\n"
            "Focal action contract:\n"
            f"{contract_json}"
        )

        rendered = (
            task_prompt
            + "\n\n"
            + instruction
        )

        return FocalActionPrompt(
            version=ACTION_PROMPT_VERSION,
            task_prompt=task_prompt,
            action_contract=contract,
            action_contract_json=(
                contract_json
            ),
            rendered_prompt=rendered,
            rendered_prompt_sha256=(
                sha256(
                    rendered.encode(
                        "utf-8"
                    )
                ).hexdigest()
            ),
            focal_decision_id=(
                schema.focal_decision_id
            ),
        )

    @staticmethod
    def _build_contract(
        schema: FocalActionSchema,
    ) -> dict[str, Any]:
        properties: dict[
            str,
            dict[str, Any],
        ] = {}

        required: list[str] = []

        for field in schema.fields:
            definition: dict[
                str,
                Any,
            ] = {}

            if field.json_type != "any":
                definition["type"] = (
                    field.json_type
                )

            if field.allowed_values is not None:
                definition["enum"] = list(
                    field.allowed_values
                )

            properties[
                field.name
            ] = definition

            if field.required:
                required.append(
                    field.name
                )

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": (
                schema.allow_additional_fields
            ),
        }