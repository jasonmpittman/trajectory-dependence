__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .state import (
    EnvironmentMutationResult,
    EnvironmentState,
    EnvironmentStateError,
)


class EnvironmentTransitionError(RuntimeError):
    """Defined failure during deterministic environment transition."""

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
class TransitionRule:
    """
    One explicit deterministic action-to-state transition rule.

    `match` uses exact mapping equality. No partial or semantic matching
    occurs.
    """

    rule_id: str
    match: dict[str, Any]

    path: tuple[str, ...]

    required_value: Any
    new_value: Any

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "match": deepcopy(self.match),
            "path": list(self.path),
            "required_value": deepcopy(
                self.required_value
            ),
            "new_value": deepcopy(
                self.new_value
            ),
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class EnvironmentTransitionResult:
    """Complete record of one successful deterministic transition."""

    rule_id: str
    action: dict[str, Any]

    path: tuple[str, ...]

    before: Any
    after: Any

    state_changed: bool

    mutation: EnvironmentMutationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "action": deepcopy(self.action),
            "path": list(self.path),
            "before": deepcopy(self.before),
            "after": deepcopy(self.after),
            "state_changed": self.state_changed,
            "mutation": self.mutation.to_dict(),
        }


class DeterministicTransitionEngine:
    """
    Apply exact preregistered transition rules to EnvironmentState.

    The same valid state and exact action always select the same rule and
    produce the same resulting state.
    """

    def __init__(
        self,
        rules: Sequence[TransitionRule],
    ) -> None:
        self._rules = tuple(rules)

        self._validate_rules()

    @property
    def rules(self) -> tuple[TransitionRule, ...]:
        """Return the immutable transition-rule collection."""

        return self._rules

    def apply(
        self,
        *,
        environment: EnvironmentState,
        action: Mapping[str, Any],
    ) -> EnvironmentTransitionResult:
        """Apply the unique rule exactly matching the supplied action."""

        if not isinstance(action, Mapping):
            raise EnvironmentTransitionError(
                code="invalid_action",
                message="Environment action must be a mapping.",
            )

        action_copy = deepcopy(dict(action))

        self._canonical_json(
            action_copy,
            field_name="action",
        )

        matching = [
            rule
            for rule in self._rules
            if rule.match == action_copy
        ]

        if not matching:
            raise EnvironmentTransitionError(
                code="transition_not_found",
                message=(
                    "No deterministic environment transition rule "
                    f"matches action {action_copy!r}."
                ),
            )

        if len(matching) != 1:
            raise EnvironmentTransitionError(
                code="ambiguous_transition",
                message=(
                    "More than one environment transition rule "
                    f"matches action {action_copy!r}."
                ),
            )

        rule = matching[0]

        query = environment.query(
            rule.path
        )

        if not query.found:
            raise EnvironmentTransitionError(
                code="transition_path_not_found",
                message=(
                    "Transition target path does not exist: "
                    f"{list(rule.path)!r}."
                ),
            )

        if query.value != rule.required_value:
            raise EnvironmentTransitionError(
                code="invalid_transition_state",
                message=(
                    f"Transition rule {rule.rule_id!r} requires "
                    f"{rule.required_value!r} at "
                    f"{list(rule.path)!r}, but observed "
                    f"{query.value!r}."
                ),
            )

        mutation = environment.targeted_modify(
            path=rule.path,
            value=rule.new_value,
        )

        return EnvironmentTransitionResult(
            rule_id=rule.rule_id,
            action=action_copy,
            path=rule.path,
            before=query.value,
            after=deepcopy(rule.new_value),
            state_changed=mutation.state_changed,
            mutation=mutation,
        )

    def _validate_rules(self) -> None:
        rule_ids: set[str] = set()
        action_signatures: set[str] = set()

        for rule in self._rules:
            if (
                not isinstance(rule.rule_id, str)
                or not rule.rule_id
            ):
                raise EnvironmentTransitionError(
                    code="invalid_rule_id",
                    message=(
                        "Transition rule IDs must be non-empty strings."
                    ),
                )

            if rule.rule_id in rule_ids:
                raise EnvironmentTransitionError(
                    code="duplicate_rule_id",
                    message=(
                        "Duplicate environment transition rule ID: "
                        f"{rule.rule_id!r}."
                    ),
                )

            rule_ids.add(rule.rule_id)

            if not isinstance(rule.match, dict):
                raise EnvironmentTransitionError(
                    code="invalid_rule_match",
                    message=(
                        f"Rule {rule.rule_id!r} match must be a mapping."
                    ),
                )

            if not rule.match:
                raise EnvironmentTransitionError(
                    code="invalid_rule_match",
                    message=(
                        f"Rule {rule.rule_id!r} match may not be empty."
                    ),
                )

            self._validate_path(
                rule.rule_id,
                rule.path,
            )

            match_signature = self._canonical_json(
                rule.match,
                field_name=f"{rule.rule_id}.match",
            )

            self._canonical_json(
                rule.required_value,
                field_name=(
                    f"{rule.rule_id}.required_value"
                ),
            )

            self._canonical_json(
                rule.new_value,
                field_name=f"{rule.rule_id}.new_value",
            )

            self._canonical_json(
                rule.metadata,
                field_name=f"{rule.rule_id}.metadata",
            )

            if match_signature in action_signatures:
                raise EnvironmentTransitionError(
                    code="duplicate_rule_match",
                    message=(
                        "Multiple transition rules use the same exact "
                        f"action match: {rule.match!r}."
                    ),
                )

            action_signatures.add(
                match_signature
            )

    @staticmethod
    def _validate_path(
        rule_id: str,
        path: Sequence[str],
    ) -> None:
        if (
            isinstance(path, (str, bytes))
            or not isinstance(path, Sequence)
            or len(path) == 0
            or not all(
                isinstance(item, str) and item
                for item in path
            )
        ):
            raise EnvironmentTransitionError(
                code="invalid_rule_path",
                message=(
                    f"Rule {rule_id!r} must contain a non-empty "
                    "sequence of string path elements."
                ),
            )

    @staticmethod
    def _canonical_json(
        value: Any,
        *,
        field_name: str,
    ) -> str:
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )

        except (TypeError, ValueError) as exc:
            raise EnvironmentTransitionError(
                code="non_serializable_rule",
                message=(
                    f"{field_name!r} must be losslessly "
                    "representable as JSON."
                ),
            ) from exc