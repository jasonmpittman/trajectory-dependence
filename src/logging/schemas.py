__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.2"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Condition(str, Enum):
    """Experimental agent configuration."""

    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"


class TaskClassification(str, Enum):
    """Protocol-level task classification."""

    DIAGNOSTIC = "diagnostic"
    INTEGRATION = "integration"


class SourceClass(str, Enum):
    """
    Experimental provenance classes.

    These indicate the current experimental source role of information,
    not necessarily the mechanism through which the information was
    transported to the model.
    """

    P = "P"
    T = "T"
    H = "H"
    M = "M"
    E = "E"


class EventType(str, Enum):
    """Allowed event types in the trajectory event stream."""

    SYSTEM = "system"
    TASK = "task"
    CONTEXT_BUILD = "context_build"
    PROVENANCE_AUDIT = "provenance_audit"
    GENERATION = "generation"
    TOOL_REQUEST = "tool_request"
    TOOL_RETURN = "tool_return"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    ENVIRONMENT_QUERY = "environment_query"
    ENVIRONMENT_TRANSITION = "environment_transition"
    ACTION = "action"
    EXPLANATION = "explanation"
    SNAPSHOT = "snapshot"
    INTERVENTION = "intervention"
    SHAM_INTERVENTION = "sham_intervention"
    TECHNICAL_FAILURE = "technical_failure"



@dataclass(frozen=True)
class ContextElement:
    """
    One provenance-preserving element of model context.

    `source` records the element's current causal/source role.

    `origin_source` optionally records the class from which the
    information originally arose. For example, a tool return retained
    in within-task history can have:

        source=H
        origin_source=T

    This distinction is required to keep retained trajectory state
    separate from its original source.
    """

    source: SourceClass
    source_id: str
    content: Any

    origin_event_id: str | None = None
    origin_source: SourceClass | None = None

    created_sequence: int | None = None
    retrieved_sequence: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        data = asdict(self)

        data["source"] = self.source.value

        if self.origin_source is not None:
            data["origin_source"] = self.origin_source.value

        return data


@dataclass(frozen=True)
class ExperimentEvent:
    """
    Canonical event representation for one trajectory event.

    Events are immutable after construction. The raw event store should
    therefore represent append-only evidence rather than mutable runtime
    state.
    """

    experiment_id: str
    run_id: str
    task_id: str
    task_classification: TaskClassification
    condition: Condition
    repetition_id: int
    seed: int | None

    event_id: str
    sequence_index: int
    timestamp: str

    event_type: EventType
    source_component: str

    raw_payload: Any
    normalized_payload: Any = None

    parent_event_id: str | None = None
    snapshot_id: str | None = None
    focal_decision_id: str | None = None
    model_invocation_id: str | None = None

    tool_call_id: str | None = None
    memory_id: str | None = None

    intervention_id: str | None = None
    intervention_source: SourceClass | None = None
    intervention_type: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""

        data = asdict(self)

        data["task_classification"] = self.task_classification.value
        data["condition"] = self.condition.value
        data["event_type"] = self.event_type.value

        if self.intervention_source is not None:
            data["intervention_source"] = self.intervention_source.value

        return data