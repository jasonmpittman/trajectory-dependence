__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .schemas import (
    Condition,
    EventType,
    ExperimentEvent,
    SourceClass,
    TaskClassification,
)


def utc_timestamp() -> str:
    """
    Return an ISO 8601 UTC timestamp.

    Wall-clock timestamps document when execution occurred. Experimental
    ordering is determined by sequence_index rather than timestamp.
    """

    return datetime.now(timezone.utc).isoformat()


class EventFactory:
    """
    Construct consistently identified and ordered events for one run.

    One EventFactory instance must correspond to exactly one experimental
    run. It owns the monotonically increasing event sequence.
    """

    def __init__(
        self,
        *,
        experiment_id: str,
        run_id: str,
        task_id: str,
        task_classification: TaskClassification,
        condition: Condition,
        repetition_id: int,
        seed: int | None,
    ) -> None:
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.task_id = task_id
        self.task_classification = task_classification
        self.condition = condition
        self.repetition_id = repetition_id
        self.seed = seed

        self._next_sequence_index = 0

    @property
    def next_sequence_index(self) -> int:
        """Return the sequence index that will be assigned next."""

        return self._next_sequence_index

    def create(
        self,
        *,
        event_type: EventType,
        source_component: str,
        raw_payload: Any,
        normalized_payload: Any = None,
        parent_event_id: str | None = None,
        snapshot_id: str | None = None,
        focal_decision_id: str | None = None,
        model_invocation_id: str | None = None,
        tool_call_id: str | None = None,
        memory_id: str | None = None,
        intervention_id: str | None = None,
        intervention_source: SourceClass | None = None,
        intervention_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentEvent:
        """
        Create the next immutable event in the run.

        Sequence indices are assigned monotonically beginning at zero.
        """

        event = ExperimentEvent(
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            task_id=self.task_id,
            task_classification=self.task_classification,
            condition=self.condition,
            repetition_id=self.repetition_id,
            seed=self.seed,
            event_id=f"evt_{uuid4().hex}",
            sequence_index=self._next_sequence_index,
            timestamp=utc_timestamp(),
            event_type=event_type,
            source_component=source_component,
            raw_payload=raw_payload,
            normalized_payload=normalized_payload,
            parent_event_id=parent_event_id,
            snapshot_id=snapshot_id,
            focal_decision_id=focal_decision_id,
            model_invocation_id=model_invocation_id,
            tool_call_id=tool_call_id,
            memory_id=memory_id,
            intervention_id=intervention_id,
            intervention_source=intervention_source,
            intervention_type=intervention_type,
            metadata=metadata or {},
        )

        self._next_sequence_index += 1

        return event