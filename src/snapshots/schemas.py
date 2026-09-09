__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..environment import EnvironmentSnapshot
from ..logging import Condition, ContextElement, SourceClass
from ..memory import MemoryRecord, MemorySnapshot


def _context_element_from_dict(
    data: dict[str, Any],
) -> ContextElement:
    return ContextElement(
        source=SourceClass(data["source"]),
        source_id=data["source_id"],
        content=deepcopy(data["content"]),
        origin_event_id=data.get("origin_event_id"),
        origin_source=(
            SourceClass(data["origin_source"])
            if data.get("origin_source") is not None
            else None
        ),
        created_sequence=data.get("created_sequence"),
        retrieved_sequence=data.get("retrieved_sequence"),
        metadata=deepcopy(data.get("metadata", {})),
    )


def _memory_snapshot_from_dict(
    data: dict[str, Any],
) -> MemorySnapshot:
    records = tuple(
        MemoryRecord(
            namespace=record["namespace"],
            memory_id=record["memory_id"],
            key=record["key"],
            value=deepcopy(record["value"]),
            created_by_task=record.get("created_by_task"),
            created_sequence=record.get("created_sequence"),
            metadata=deepcopy(record.get("metadata", {})),
        )
        for record in data["records"]
    )

    return MemorySnapshot(
        namespace=data["namespace"],
        records=records,
        checksum_sha256=data["checksum_sha256"],
    )


def _environment_snapshot_from_dict(
    data: dict[str, Any],
) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        state=deepcopy(data["state"]),
        checksum_sha256=data["checksum_sha256"],
    )


@dataclass(frozen=True)
class PreFocalSnapshot:
    """
    Complete experiment-controlled state captured immediately before
    the model invocation responsible for a focal consequential decision.

    snapshot_id and created_at_utc identify the snapshot artifact.

    state_checksum_sha256 covers experimental state rather than artifact
    creation time, allowing equivalent captured state to produce the same
    state checksum.
    """

    snapshot_id: str

    experiment_id: str
    run_id: str
    task_id: str
    condition: Condition
    repetition_id: int

    sequence_position: int
    created_at_utc: str

    prompt_elements: tuple[ContextElement, ...]
    tool_elements: tuple[ContextElement, ...]
    history_elements: tuple[ContextElement, ...]

    memory_snapshot: MemorySnapshot | None
    environment_snapshot: EnvironmentSnapshot | None

    runtime_configuration: dict[str, Any]
    task_control_state: dict[str, Any]
    seed_metadata: dict[str, Any]

    state_checksum_sha256: str

    def state_payload(self) -> dict[str, Any]:
        """
        Return the state-bearing payload covered by the snapshot checksum.

        snapshot_id, created_at_utc, and the checksum itself are excluded
        because they describe the snapshot artifact rather than the
        experimental state being preserved.
        """

        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "condition": self.condition.value,
            "repetition_id": self.repetition_id,
            "sequence_position": self.sequence_position,
            "prompt_elements": [
                element.to_dict()
                for element in self.prompt_elements
            ],
            "tool_elements": [
                element.to_dict()
                for element in self.tool_elements
            ],
            "history_elements": [
                element.to_dict()
                for element in self.history_elements
            ],
            "memory_snapshot": (
                self.memory_snapshot.to_dict()
                if self.memory_snapshot is not None
                else None
            ),
            "environment_snapshot": (
                self.environment_snapshot.to_dict()
                if self.environment_snapshot is not None
                else None
            ),
            "runtime_configuration": deepcopy(
                self.runtime_configuration
            ),
            "task_control_state": deepcopy(
                self.task_control_state
            ),
            "seed_metadata": deepcopy(
                self.seed_metadata
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.state_payload()

        return {
            "snapshot_id": self.snapshot_id,
            "created_at_utc": self.created_at_utc,
            **data,
            "state_checksum_sha256": self.state_checksum_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "PreFocalSnapshot":
        memory_data = data.get(
            "memory_snapshot"
        )

        environment_data = data.get(
            "environment_snapshot"
        )

        return cls(
            snapshot_id=data["snapshot_id"],
            experiment_id=data["experiment_id"],
            run_id=data["run_id"],
            task_id=data["task_id"],
            condition=Condition(data["condition"]),
            repetition_id=data["repetition_id"],
            sequence_position=data["sequence_position"],
            created_at_utc=data["created_at_utc"],
            prompt_elements=tuple(
                _context_element_from_dict(element)
                for element in data.get(
                    "prompt_elements",
                    [],
                )
            ),
            tool_elements=tuple(
                _context_element_from_dict(element)
                for element in data.get(
                    "tool_elements",
                    [],
                )
            ),
            history_elements=tuple(
                _context_element_from_dict(element)
                for element in data.get(
                    "history_elements",
                    [],
                )
            ),
            memory_snapshot=(
                _memory_snapshot_from_dict(
                    memory_data
                )
                if memory_data is not None
                else None
            ),
            environment_snapshot=(
                _environment_snapshot_from_dict(
                    environment_data
                )
                if environment_data is not None
                else None
            ),
            runtime_configuration=deepcopy(
                data.get(
                    "runtime_configuration",
                    {},
                )
            ),
            task_control_state=deepcopy(
                data.get(
                    "task_control_state",
                    {},
                )
            ),
            seed_metadata=deepcopy(
                data.get(
                    "seed_metadata",
                    {},
                )
            ),
            state_checksum_sha256=data[
                "state_checksum_sha256"
            ],
        )


@dataclass(frozen=True)
class RestoredPreFocalState:
    """
    State reconstructed from a validated PreFocalSnapshot.

    Memory and environment are restored into their owning subsystems.
    P, T, H, runtime configuration, task state, seed metadata, and
    sequence position are returned for the future runtime to reinstate.
    """

    prompt_elements: tuple[ContextElement, ...]
    tool_elements: tuple[ContextElement, ...]
    history_elements: tuple[ContextElement, ...]

    runtime_configuration: dict[str, Any]
    task_control_state: dict[str, Any]
    seed_metadata: dict[str, Any]

    sequence_position: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_elements": [
                element.to_dict()
                for element in self.prompt_elements
            ],
            "tool_elements": [
                element.to_dict()
                for element in self.tool_elements
            ],
            "history_elements": [
                element.to_dict()
                for element in self.history_elements
            ],
            "runtime_configuration": deepcopy(
                self.runtime_configuration
            ),
            "task_control_state": deepcopy(
                self.task_control_state
            ),
            "seed_metadata": deepcopy(
                self.seed_metadata
            ),
            "sequence_position": self.sequence_position,
        }