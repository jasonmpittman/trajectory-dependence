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
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from ..environment import (
    EnvironmentSnapshot,
    EnvironmentState,
)
from ..logging import (
    Condition,
    ContextElement,
    SourceClass,
)
from ..memory import (
    MemorySnapshot,
    MemoryStore,
)
from .schemas import (
    PreFocalSnapshot,
    RestoredPreFocalState,
)


class SnapshotError(RuntimeError):
    """Defined failure raised by the unified snapshot subsystem."""

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


def utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


class SnapshotManager:
    """
    Capture, validate, persist, load, and restore unified pre-focal state.

    The manager does not execute model inference or interventions.
    """

    def capture(
        self,
        *,
        experiment_id: str,
        run_id: str,
        task_id: str,
        condition: Condition,
        repetition_id: int,
        sequence_position: int,
        prompt_elements: Sequence[ContextElement],
        tool_elements: Sequence[ContextElement] = (),
        history_elements: Sequence[ContextElement] = (),
        memory_store: MemoryStore | None = None,
        memory_namespace: str | None = None,
        environment: EnvironmentState | None = None,
        runtime_configuration: Mapping[str, Any] | None = None,
        task_control_state: Mapping[str, Any] | None = None,
        seed_metadata: Mapping[str, Any] | None = None,
        snapshot_id: str | None = None,
    ) -> PreFocalSnapshot:
        """
        Capture one complete pre-focal snapshot.

        The input state is defensively copied and validated before the
        snapshot is returned.
        """

        self._validate_identifier(
            "experiment_id",
            experiment_id,
        )
        self._validate_identifier(
            "run_id",
            run_id,
        )
        self._validate_identifier(
            "task_id",
            task_id,
        )

        if repetition_id < 0:
            raise SnapshotError(
                code="invalid_repetition_id",
                message=(
                    "repetition_id must be greater than or equal to zero."
                ),
            )

        if sequence_position < 0:
            raise SnapshotError(
                code="invalid_sequence_position",
                message=(
                    "sequence_position must be greater than or equal "
                    "to zero."
                ),
            )

        prompt_copy = self._clone_context_elements(
            prompt_elements
        )
        tool_copy = self._clone_context_elements(
            tool_elements
        )
        history_copy = self._clone_context_elements(
            history_elements
        )

        self._validate_source_bucket(
            prompt_copy,
            expected=SourceClass.P,
            bucket_name="prompt_elements",
        )

        self._validate_source_bucket(
            tool_copy,
            expected=SourceClass.T,
            bucket_name="tool_elements",
        )

        self._validate_source_bucket(
            history_copy,
            expected=SourceClass.H,
            bucket_name="history_elements",
        )

        if (
            memory_store is None
            and memory_namespace is not None
        ):
            raise SnapshotError(
                code="memory_configuration_mismatch",
                message=(
                    "memory_namespace was provided without a memory_store."
                ),
            )

        if (
            memory_store is not None
            and memory_namespace is None
        ):
            raise SnapshotError(
                code="memory_configuration_mismatch",
                message=(
                    "memory_store was provided without memory_namespace."
                ),
            )

        self._validate_condition_state(
            condition=condition,
            tool_elements=tool_copy,
            history_elements=history_copy,
            has_memory=memory_store is not None,
        )

        memory_snapshot = (
            memory_store.snapshot(
                namespace=memory_namespace
            )
            if (
                memory_store is not None
                and memory_namespace is not None
            )
            else None
        )

        environment_snapshot = (
            environment.snapshot()
            if environment is not None
            else None
        )

        runtime_copy = deepcopy(
            dict(
                runtime_configuration
                or {}
            )
        )

        task_control_copy = deepcopy(
            dict(
                task_control_state
                or {}
            )
        )

        seed_copy = deepcopy(
            dict(
                seed_metadata
                or {}
            )
        )

        self._validate_json(
            runtime_copy,
            field_name="runtime_configuration",
        )

        self._validate_json(
            task_control_copy,
            field_name="task_control_state",
        )

        self._validate_json(
            seed_copy,
            field_name="seed_metadata",
        )

        resolved_snapshot_id = (
            snapshot_id
            or f"snap_{uuid4().hex}"
        )

        self._validate_identifier(
            "snapshot_id",
            resolved_snapshot_id,
        )

        provisional = PreFocalSnapshot(
            snapshot_id=resolved_snapshot_id,
            experiment_id=experiment_id,
            run_id=run_id,
            task_id=task_id,
            condition=condition,
            repetition_id=repetition_id,
            sequence_position=sequence_position,
            created_at_utc=utc_timestamp(),
            prompt_elements=prompt_copy,
            tool_elements=tool_copy,
            history_elements=history_copy,
            memory_snapshot=memory_snapshot,
            environment_snapshot=environment_snapshot,
            runtime_configuration=runtime_copy,
            task_control_state=task_control_copy,
            seed_metadata=seed_copy,
            state_checksum_sha256="",
        )

        checksum = self._checksum(
            provisional.state_payload()
        )

        return PreFocalSnapshot(
            snapshot_id=provisional.snapshot_id,
            experiment_id=provisional.experiment_id,
            run_id=provisional.run_id,
            task_id=provisional.task_id,
            condition=provisional.condition,
            repetition_id=provisional.repetition_id,
            sequence_position=provisional.sequence_position,
            created_at_utc=provisional.created_at_utc,
            prompt_elements=provisional.prompt_elements,
            tool_elements=provisional.tool_elements,
            history_elements=provisional.history_elements,
            memory_snapshot=provisional.memory_snapshot,
            environment_snapshot=(
                provisional.environment_snapshot
            ),
            runtime_configuration=(
                provisional.runtime_configuration
            ),
            task_control_state=(
                provisional.task_control_state
            ),
            seed_metadata=provisional.seed_metadata,
            state_checksum_sha256=checksum,
        )

    def validate(
        self,
        snapshot: PreFocalSnapshot,
    ) -> None:
        """Validate the complete snapshot and nested state checksums."""

        expected = self._checksum(
            snapshot.state_payload()
        )

        if expected != snapshot.state_checksum_sha256:
            raise SnapshotError(
                code="snapshot_checksum_mismatch",
                message=(
                    "Unified pre-focal snapshot checksum validation "
                    "failed."
                ),
            )

        self._validate_source_bucket(
            snapshot.prompt_elements,
            expected=SourceClass.P,
            bucket_name="prompt_elements",
        )

        self._validate_source_bucket(
            snapshot.tool_elements,
            expected=SourceClass.T,
            bucket_name="tool_elements",
        )

        self._validate_source_bucket(
            snapshot.history_elements,
            expected=SourceClass.H,
            bucket_name="history_elements",
        )

        self._validate_condition_state(
            condition=snapshot.condition,
            tool_elements=snapshot.tool_elements,
            history_elements=snapshot.history_elements,
            has_memory=(
                snapshot.memory_snapshot
                is not None
            ),
        )

        if snapshot.memory_snapshot is not None:
            self._validate_memory_snapshot(
                snapshot.memory_snapshot
            )

        if snapshot.environment_snapshot is not None:
            self._validate_environment_snapshot(
                snapshot.environment_snapshot
            )

    def restore(
        self,
        snapshot: PreFocalSnapshot,
        *,
        memory_store: MemoryStore | None = None,
        environment: EnvironmentState | None = None,
    ) -> RestoredPreFocalState:
        """
        Restore the externally owned M/E components and return the
        remaining runtime state needed for future replay.

        Validation occurs before any restoration is attempted.
        """

        self.validate(snapshot)

        if (
            snapshot.memory_snapshot is not None
            and memory_store is None
        ):
            raise SnapshotError(
                code="memory_store_required",
                message=(
                    "Snapshot contains persistent memory but no "
                    "memory_store was supplied for restoration."
                ),
            )

        if (
            snapshot.environment_snapshot is not None
            and environment is None
        ):
            raise SnapshotError(
                code="environment_required",
                message=(
                    "Snapshot contains environmental state but no "
                    "EnvironmentState was supplied for restoration."
                ),
            )

        if (
            snapshot.memory_snapshot is not None
            and memory_store is not None
        ):
            memory_store.restore(
                snapshot.memory_snapshot
            )

        if (
            snapshot.environment_snapshot is not None
            and environment is not None
        ):
            environment.restore(
                snapshot.environment_snapshot
            )

        return RestoredPreFocalState(
            prompt_elements=self._clone_context_elements(
                snapshot.prompt_elements
            ),
            tool_elements=self._clone_context_elements(
                snapshot.tool_elements
            ),
            history_elements=self._clone_context_elements(
                snapshot.history_elements
            ),
            runtime_configuration=deepcopy(
                snapshot.runtime_configuration
            ),
            task_control_state=deepcopy(
                snapshot.task_control_state
            ),
            seed_metadata=deepcopy(
                snapshot.seed_metadata
            ),
            sequence_position=snapshot.sequence_position,
        )

    def save(
        self,
        snapshot: PreFocalSnapshot,
        path: Path | str,
    ) -> None:
        """
        Persist one validated snapshot as JSON.

        Existing snapshot files are never overwritten.
        """

        self.validate(snapshot)

        snapshot_path = Path(path)

        snapshot_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if snapshot_path.exists():
            raise FileExistsError(
                "Snapshot file already exists and will not be "
                f"overwritten: {snapshot_path}"
            )

        serialized = json.dumps(
            snapshot.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )

        snapshot_path.write_text(
            serialized + "\n",
            encoding="utf-8",
        )

    def load(
        self,
        path: Path | str,
    ) -> PreFocalSnapshot:
        """Load and validate one persisted snapshot."""

        snapshot_path = Path(path)

        try:
            data = json.loads(
                snapshot_path.read_text(
                    encoding="utf-8"
                )
            )

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise SnapshotError(
                code="invalid_snapshot_file",
                message=(
                    f"Unable to load snapshot file: {snapshot_path}"
                ),
            ) from exc

        try:
            snapshot = PreFocalSnapshot.from_dict(
                data
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise SnapshotError(
                code="invalid_snapshot_schema",
                message=(
                    f"Invalid snapshot structure: {snapshot_path}"
                ),
            ) from exc

        self.validate(snapshot)

        return snapshot

    def _validate_condition_state(
        self,
        *,
        condition: Condition,
        tool_elements: Sequence[ContextElement],
        history_elements: Sequence[ContextElement],
        has_memory: bool,
    ) -> None:
        if condition == Condition.C0:
            if tool_elements:
                raise SnapshotError(
                    code="condition_isolation_violation",
                    message="C0 may not contain tool state T.",
                )

            if history_elements:
                raise SnapshotError(
                    code="condition_isolation_violation",
                    message="C0 may not contain history state H.",
                )

            if has_memory:
                raise SnapshotError(
                    code="condition_isolation_violation",
                    message="C0 may not contain persistent memory M.",
                )

        elif condition == Condition.C1:
            if history_elements:
                raise SnapshotError(
                    code="condition_isolation_violation",
                    message="C1 may not contain history state H.",
                )

            if has_memory:
                raise SnapshotError(
                    code="condition_isolation_violation",
                    message="C1 may not contain persistent memory M.",
                )

        elif condition == Condition.C2:
            if has_memory:
                raise SnapshotError(
                    code="condition_isolation_violation",
                    message="C2 may not contain persistent memory M.",
                )

    @staticmethod
    def _validate_source_bucket(
        elements: Sequence[ContextElement],
        *,
        expected: SourceClass,
        bucket_name: str,
    ) -> None:
        for element in elements:
            if element.source != expected:
                raise SnapshotError(
                    code="source_bucket_mismatch",
                    message=(
                        f"{bucket_name!r} contains source "
                        f"{element.source.value!r}; expected "
                        f"{expected.value!r}."
                    ),
                )

    @staticmethod
    def _clone_context_elements(
        elements: Sequence[ContextElement],
    ) -> tuple[ContextElement, ...]:
        return tuple(
            ContextElement(
                source=element.source,
                source_id=element.source_id,
                content=deepcopy(
                    element.content
                ),
                origin_event_id=(
                    element.origin_event_id
                ),
                origin_source=(
                    element.origin_source
                ),
                created_sequence=(
                    element.created_sequence
                ),
                retrieved_sequence=(
                    element.retrieved_sequence
                ),
                metadata=deepcopy(
                    element.metadata
                ),
            )
            for element in elements
        )

    def _validate_memory_snapshot(
        self,
        snapshot: MemorySnapshot,
    ) -> None:
        payload = {
            "namespace": snapshot.namespace,
            "records": [
                record.to_dict()
                for record in snapshot.records
            ],
        }

        expected = self._checksum(
            payload
        )

        if expected != snapshot.checksum_sha256:
            raise SnapshotError(
                code="memory_snapshot_checksum_mismatch",
                message=(
                    "Nested memory snapshot checksum validation failed."
                ),
            )

    def _validate_environment_snapshot(
        self,
        snapshot: EnvironmentSnapshot,
    ) -> None:
        expected = self._checksum(
            snapshot.state
        )

        if expected != snapshot.checksum_sha256:
            raise SnapshotError(
                code="environment_snapshot_checksum_mismatch",
                message=(
                    "Nested environment snapshot checksum validation "
                    "failed."
                ),
            )

    @staticmethod
    def _validate_identifier(
        field_name: str,
        value: Any,
    ) -> None:
        if not isinstance(value, str) or not value:
            raise SnapshotError(
                code="invalid_identifier",
                message=(
                    f"{field_name!r} must be a non-empty string."
                ),
            )

    @classmethod
    def _validate_json(
        cls,
        value: Any,
        *,
        field_name: str,
    ) -> None:
        try:
            cls._canonical_json(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise SnapshotError(
                code="non_serializable_snapshot_state",
                message=(
                    f"{field_name!r} must be losslessly "
                    "representable as JSON."
                ),
            ) from exc

    @staticmethod
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

    @classmethod
    def _checksum(
        cls,
        value: Any,
    ) -> str:
        serialized = cls._canonical_json(
            value
        )

        return sha256(
            serialized.encode("utf-8")
        ).hexdigest()