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
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence


class EnvironmentStateError(RuntimeError):
    """Defined failure raised by the experimental environment subsystem."""

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
class EnvironmentQueryResult:
    """Result of an exact path query against environmental state."""

    path: tuple[str, ...]
    found: bool
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "found": self.found,
            "value": deepcopy(self.value),
        }


@dataclass(frozen=True)
class EnvironmentMutationResult:
    """Result of an explicit targeted environmental-state mutation."""

    path: tuple[str, ...]

    previous_exists: bool
    previous_value: Any
    new_value: Any

    state_changed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "previous_exists": self.previous_exists,
            "previous_value": deepcopy(self.previous_value),
            "new_value": deepcopy(self.new_value),
            "state_changed": self.state_changed,
        }


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """
    Complete integrity-checked snapshot of environmental state.

    The snapshot contains only experiment-controlled structured state.
    Runtime sequencing and other scaffold state will later be handled by
    the broader pre-focal snapshot subsystem.
    """

    state: dict[str, Any]
    checksum_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": deepcopy(self.state),
            "checksum_sha256": self.checksum_sha256,
        }


class EnvironmentState:
    """
    Owner of deterministic structured environmental state.

    Input and returned state are defensively copied so that callers cannot
    mutate the environment without using an explicit environment operation.
    """

    def __init__(
        self,
        initial_state: Mapping[str, Any],
    ) -> None:
        if not isinstance(initial_state, Mapping):
            raise EnvironmentStateError(
                code="invalid_initial_state",
                message="Environmental state must be a mapping.",
            )

        normalized = deepcopy(dict(initial_state))

        self._validate_json(
            normalized,
            field_name="initial_state",
        )

        self._state: dict[str, Any] = normalized

    def get_state(self) -> dict[str, Any]:
        """Return a defensive copy of the complete current state."""

        return deepcopy(self._state)

    def query(
        self,
        path: Sequence[str],
    ) -> EnvironmentQueryResult:
        """
        Query an exact nested state path.

        A missing path is a valid deterministic observation rather than an
        exception.
        """

        normalized_path = self._validate_path(path)

        current: Any = self._state

        for key in normalized_path:
            if not isinstance(current, Mapping) or key not in current:
                return EnvironmentQueryResult(
                    path=normalized_path,
                    found=False,
                    value=None,
                )

            current = current[key]

        return EnvironmentQueryResult(
            path=normalized_path,
            found=True,
            value=deepcopy(current),
        )

    def targeted_modify(
        self,
        *,
        path: Sequence[str],
        value: Any,
    ) -> EnvironmentMutationResult:
        """
        Modify exactly one environmental-state location.

        Parent paths must already exist. The final key may either already
        exist or be newly created.
        """

        normalized_path = self._validate_path(path)

        replacement = deepcopy(value)

        self._validate_json(
            replacement,
            field_name="value",
        )

        current: Any = self._state

        for key in normalized_path[:-1]:
            if not isinstance(current, dict) or key not in current:
                raise EnvironmentStateError(
                    code="path_not_found",
                    message=(
                        "Parent environmental path does not exist: "
                        f"{list(normalized_path)!r}"
                    ),
                )

            current = current[key]

        if not isinstance(current, dict):
            raise EnvironmentStateError(
                code="invalid_parent",
                message=(
                    "Environmental parent path is not a mapping: "
                    f"{list(normalized_path[:-1])!r}"
                ),
            )

        final_key = normalized_path[-1]

        previous_exists = final_key in current

        previous_value = (
            deepcopy(current[final_key])
            if previous_exists
            else None
        )

        state_changed = (
            not previous_exists
            or previous_value != replacement
        )

        current[final_key] = replacement

        return EnvironmentMutationResult(
            path=normalized_path,
            previous_exists=previous_exists,
            previous_value=previous_value,
            new_value=deepcopy(replacement),
            state_changed=state_changed,
        )

    def snapshot(self) -> EnvironmentSnapshot:
        """Return a complete integrity-checked snapshot."""

        state = self.get_state()

        return EnvironmentSnapshot(
            state=state,
            checksum_sha256=self._state_checksum(state),
        )

    def restore(
        self,
        snapshot: EnvironmentSnapshot,
    ) -> None:
        """Restore environmental state exactly from a validated snapshot."""

        expected_checksum = self._state_checksum(
            snapshot.state
        )

        if expected_checksum != snapshot.checksum_sha256:
            raise EnvironmentStateError(
                code="snapshot_checksum_mismatch",
                message=(
                    "Environmental snapshot checksum validation failed."
                ),
            )

        restored = deepcopy(snapshot.state)

        self._validate_json(
            restored,
            field_name="snapshot.state",
        )

        self._state = restored

    def checksum_sha256(self) -> str:
        """Return the checksum of the current environmental state."""

        return self._state_checksum(self._state)

    @staticmethod
    def _validate_path(
        path: Sequence[str],
    ) -> tuple[str, ...]:
        if (
            isinstance(path, (str, bytes))
            or not isinstance(path, Sequence)
            or len(path) == 0
            or not all(
                isinstance(item, str) and item
                for item in path
            )
        ):
            raise EnvironmentStateError(
                code="invalid_path",
                message=(
                    "Environmental path must be a non-empty sequence "
                    "of non-empty string keys."
                ),
            )

        return tuple(path)

    @staticmethod
    def _canonical_json(
        value: Any,
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
            raise EnvironmentStateError(
                code="non_serializable_state",
                message=(
                    "Environmental state must be losslessly "
                    "representable as JSON."
                ),
            ) from exc

    @classmethod
    def _validate_json(
        cls,
        value: Any,
        *,
        field_name: str,
    ) -> None:
        try:
            cls._canonical_json(value)

        except EnvironmentStateError as exc:
            raise EnvironmentStateError(
                code=exc.code,
                message=(
                    f"{field_name!r} must be losslessly "
                    "representable as JSON."
                ),
            ) from exc

    @classmethod
    def _state_checksum(
        cls,
        state: Mapping[str, Any],
    ) -> str:
        serialized = cls._canonical_json(
            dict(state)
        )

        return sha256(
            serialized.encode("utf-8")
        ).hexdigest()