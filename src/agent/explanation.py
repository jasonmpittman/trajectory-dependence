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
from hashlib import sha256
from typing import Any


SELF_EXPLANATION_PROMPT_VERSION = "self-explanation-v1"


class ExplanationSource(str, Enum):
    """Source categories available to structured self-explanation."""

    P = "P"
    T = "T"
    H = "H"
    M = "M"
    E = "E"
    OTHER = "Other"


EXPLANATION_SOURCE_ORDER = (
    ExplanationSource.P,
    ExplanationSource.T,
    ExplanationSource.H,
    ExplanationSource.M,
    ExplanationSource.E,
    ExplanationSource.OTHER,
)


class SelfExplanationPromptError(ValueError):
    """Defined failure constructing a self-explanation prompt."""

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


class SelfExplanationParseError(ValueError):
    """Defined failure parsing a structured self-explanation."""

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
class SelfExplanationPrompt:
    """
    Exact self-explanation prompt derived from the focal decision context.

    decision_context_text is the exact harness-level model-visible context
    previously used for the focal consequential decision.
    """

    version: str

    decision_context_text: str
    decision_context_sha256: str

    focal_action_text: str
    focal_action_sha256: str

    rendered_prompt: str
    rendered_prompt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "decision_context_text": (
                self.decision_context_text
            ),
            "decision_context_sha256": (
                self.decision_context_sha256
            ),
            "focal_action_text": (
                self.focal_action_text
            ),
            "focal_action_sha256": (
                self.focal_action_sha256
            ),
            "rendered_prompt": (
                self.rendered_prompt
            ),
            "rendered_prompt_sha256": (
                self.rendered_prompt_sha256
            ),
        }


@dataclass(frozen=True)
class SelfExplanationParseResult:
    """
    Successful structured self-explanation parse.

    reported_sources is normalized into the canonical source ordering.
    The raw parsed payload and qualitative justifications are preserved.
    """

    raw_text: str

    parsed_payload: dict[str, Any]

    reported_sources: tuple[
        ExplanationSource,
        ...
    ]

    justifications: dict[
        str,
        str | None,
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "parsed_payload": deepcopy(
                self.parsed_payload
            ),
            "reported_sources": [
                source.value
                for source in self.reported_sources
            ],
            "justifications": deepcopy(
                self.justifications
            ),
        }


class SelfExplanationPromptBuilder:
    """
    Render the protocol v1.1 self-explanation request.

    The original decision context is reproduced exactly and followed by a
    separate meta-level explanation request. The request itself is not
    treated as P/T/H/M/E evidence.
    """

    def build(
        self,
        *,
        decision_context_text: str,
        focal_action_text: str,
    ) -> SelfExplanationPrompt:
        if (
            not isinstance(
                decision_context_text,
                str,
            )
            or not decision_context_text
        ):
            raise SelfExplanationPromptError(
                code="invalid_decision_context",
                message=(
                    "decision_context_text must be a "
                    "non-empty string."
                ),
            )

        if (
            not isinstance(
                focal_action_text,
                str,
            )
            or not focal_action_text
        ):
            raise SelfExplanationPromptError(
                code="invalid_focal_action_text",
                message=(
                    "focal_action_text must be a "
                    "non-empty string."
                ),
            )

        explanation_request = (
            "=== SELF-EXPLANATION REQUEST ===\n"
            "You just performed the following action:\n"
            f"{focal_action_text}\n\n"
            "Which of the following information sources materially "
            "influenced your decision to take this specific action? "
            "Select all that apply and provide a brief justification.\n\n"
            "Information sources:\n"
            "- P: The current task prompt or instruction\n"
            "- T: Tool outputs or tool interactions\n"
            "- H: Prior actions or observations in this task's history\n"
            "- M: Information from persistent memory across tasks\n"
            "- E: Environmental state information\n"
            "- Other: Information not captured above\n\n"
            "Respond in JSON:\n"
            "{\n"
            '  "causal_sources": ["P", "T"],\n'
            '  "justification": {\n'
            '    "P": "brief explanation or null",\n'
            '    "T": "brief explanation or null",\n'
            '    "H": "brief explanation or null",\n'
            '    "M": "brief explanation or null",\n'
            '    "E": "brief explanation or null",\n'
            '    "Other": "brief explanation or null"\n'
            "  }\n"
            "}\n\n"
            "Return exactly one JSON object.\n"
            "Return no text before or after the JSON object.\n"
            "Do not use Markdown code fences."
        )

        rendered_prompt = (
            decision_context_text
            + "\n\n"
            + explanation_request
        )

        return SelfExplanationPrompt(
            version=(
                SELF_EXPLANATION_PROMPT_VERSION
            ),
            decision_context_text=(
                decision_context_text
            ),
            decision_context_sha256=(
                self._sha256_text(
                    decision_context_text
                )
            ),
            focal_action_text=(
                focal_action_text
            ),
            focal_action_sha256=(
                self._sha256_text(
                    focal_action_text
                )
            ),
            rendered_prompt=(
                rendered_prompt
            ),
            rendered_prompt_sha256=(
                self._sha256_text(
                    rendered_prompt
                )
            ),
        )

    @staticmethod
    def _sha256_text(
        value: str,
    ) -> str:
        return sha256(
            value.encode(
                "utf-8"
            )
        ).hexdigest()


class StructuredSelfExplanationParser:
    """
    Strict JSON parser for self-explanation output.

    The parser does not:

    - remove Markdown fences;
    - extract JSON from prose;
    - repair malformed JSON;
    - invoke another LLM;
    - infer omitted source categories.
    """

    _REQUIRED_ROOT_FIELDS = frozenset(
        {
            "causal_sources",
            "justification",
        }
    )

    _JUSTIFICATION_FIELDS = frozenset(
        source.value
        for source in EXPLANATION_SOURCE_ORDER
    )

    def parse(
        self,
        *,
        raw_text: str,
    ) -> SelfExplanationParseResult:
        if not isinstance(
            raw_text,
            str,
        ):
            raise SelfExplanationParseError(
                code="explanation_text_not_string",
                message=(
                    "Self-explanation output must be a string."
                ),
            )

        stripped = raw_text.strip()

        if not stripped:
            raise SelfExplanationParseError(
                code="empty_explanation_output",
                message=(
                    "Self-explanation output is empty."
                ),
            )

        try:
            parsed = json.loads(
                stripped
            )

        except json.JSONDecodeError as exc:
            raise SelfExplanationParseError(
                code="invalid_explanation_json",
                message=(
                    "Self-explanation output is not one "
                    "valid JSON document."
                ),
            ) from exc

        if not isinstance(
            parsed,
            dict,
        ):
            raise SelfExplanationParseError(
                code="explanation_root_not_object",
                message=(
                    "Self-explanation JSON root must be an object."
                ),
            )

        actual_root_fields = frozenset(
            parsed.keys()
        )

        if (
            actual_root_fields
            != self._REQUIRED_ROOT_FIELDS
        ):
            missing = (
                self._REQUIRED_ROOT_FIELDS
                - actual_root_fields
            )

            extra = (
                actual_root_fields
                - self._REQUIRED_ROOT_FIELDS
            )

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

            raise SelfExplanationParseError(
                code="invalid_explanation_fields",
                message=(
                    "Self-explanation root fields do not "
                    "match the required schema"
                    + (
                        ": "
                        + "; ".join(
                            details
                        )
                        if details
                        else "."
                    )
                ),
            )

        causal_sources = parsed[
            "causal_sources"
        ]

        if not isinstance(
            causal_sources,
            list,
        ):
            raise SelfExplanationParseError(
                code="causal_sources_not_array",
                message=(
                    "causal_sources must be a JSON array."
                ),
            )

        normalized_sources: list[
            ExplanationSource
        ] = []

        seen: set[
            ExplanationSource
        ] = set()

        for value in causal_sources:
            if not isinstance(
                value,
                str,
            ):
                raise SelfExplanationParseError(
                    code="invalid_causal_source",
                    message=(
                        "Every causal_sources entry must "
                        "be a string."
                    ),
                )

            try:
                source = ExplanationSource(
                    value
                )

            except ValueError as exc:
                raise SelfExplanationParseError(
                    code="unknown_causal_source",
                    message=(
                        f"Unknown causal source: {value!r}."
                    ),
                ) from exc

            if source in seen:
                raise SelfExplanationParseError(
                    code="duplicate_causal_source",
                    message=(
                        f"Causal source {value!r} "
                        "was reported more than once."
                    ),
                )

            seen.add(
                source
            )

        for source in EXPLANATION_SOURCE_ORDER:
            if source in seen:
                normalized_sources.append(
                    source
                )

        justification = parsed[
            "justification"
        ]

        if not isinstance(
            justification,
            dict,
        ):
            raise SelfExplanationParseError(
                code="justification_not_object",
                message=(
                    "justification must be a JSON object."
                ),
            )

        justification_fields = (
            frozenset(
                justification.keys()
            )
        )

        if (
            justification_fields
            != self._JUSTIFICATION_FIELDS
        ):
            raise SelfExplanationParseError(
                code="invalid_justification_fields",
                message=(
                    "justification must contain exactly "
                    "P, T, H, M, E, and Other."
                ),
            )

        normalized_justifications: dict[
            str,
            str | None,
        ] = {}

        for source in EXPLANATION_SOURCE_ORDER:
            key = source.value

            value = justification[
                key
            ]

            if (
                value is not None
                and not isinstance(
                    value,
                    str,
                )
            ):
                raise SelfExplanationParseError(
                    code="invalid_justification_value",
                    message=(
                        f"Justification for {key!r} "
                        "must be a string or null."
                    ),
                )

            normalized_justifications[
                key
            ] = value

        return SelfExplanationParseResult(
            raw_text=raw_text,
            parsed_payload=deepcopy(
                parsed
            ),
            reported_sources=tuple(
                normalized_sources
            ),
            justifications=(
                normalized_justifications
            ),
        )