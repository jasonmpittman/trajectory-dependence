__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from uuid import uuid4

from ..environment import (
    DeterministicTransitionEngine,
    EnvironmentQueryResult,
    EnvironmentState,
    EnvironmentTransitionResult,
)
from ..logging import (
    ContextElement,
    EventType,
    ExperimentEvent,
    SourceClass,
)
from ..memory import (
    MemoryRecord,
    MemoryStore,
)
from ..model import (
    InferenceConfig,
)
from ..tools import (
    Tool,
    ToolExecutionError,
    ToolResult,
)
from .conditions import (
    get_condition_policy,
)
from .context import (
    TaskEnvironmentPolicy,
)
from .runtime import (
    AgentRuntime,
    FocalActionRecord,
)


class ScaffoldExecutionError(RuntimeError):
    """Defined failure in controlled scaffold execution."""

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
class ToolInteractionRecord:
    """Complete record of one controlled deterministic tool interaction."""

    tool_call_id: str

    request_event: ExperimentEvent
    return_event: ExperimentEvent

    result: ToolResult

    tool_element: ContextElement

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "request_event_id": (
                self.request_event.event_id
            ),
            "return_event_id": (
                self.return_event.event_id
            ),
            "result": self.result.to_dict(),
            "tool_element": (
                self.tool_element.to_dict()
            ),
        }

@dataclass(frozen=True)
class MemoryWriteObservation:
    """
    One logged persistent-memory creation.

    MEMORY_WRITE records mutation of persistent state. It deliberately
    does not create a model-visible M ContextElement. Model-visible
    persistent memory is produced only by a later MEMORY_READ.
    """

    event: ExperimentEvent
    record: MemoryRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event.event_id,
            "record": self.record.to_dict(),
        }
    
@dataclass(frozen=True)
class MemoryReadObservation:
    """One logged persistent-memory read."""

    event: ExperimentEvent

    record: MemoryRecord | None
    memory_element: ContextElement | None

    @property
    def found(self) -> bool:
        return self.record is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event.event_id,
            "found": self.found,
            "record": (
                self.record.to_dict()
                if self.record is not None
                else None
            ),
            "memory_element": (
                self.memory_element.to_dict()
                if self.memory_element is not None
                else None
            ),
        }


@dataclass(frozen=True)
class EnvironmentObservation:
    """One logged environmental-state observation."""

    event: ExperimentEvent

    result: EnvironmentQueryResult

    environment_element: ContextElement

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event.event_id,
            "result": self.result.to_dict(),
            "environment_element": (
                self.environment_element.to_dict()
            ),
        }


@dataclass(frozen=True)
class EnvironmentTransitionObservation:
    """One logged deterministic environment transition."""

    event: ExperimentEvent

    result: EnvironmentTransitionResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event.event_id,
            "result": self.result.to_dict(),
        }


class ScaffoldController:
    """
    Controlled task-scaffold execution around one AgentRuntime.

    The controller performs deterministic state acquisition. It does not
    decide which operations a task should perform; that responsibility
    belongs to the future preregistered task specification/orchestrator.

    The controller provides event-backed T, H, M, and E elements suitable
    for the existing ContextBuilder and provenance auditor.
    """

    _HISTORY_ORIGIN_SOURCE = {
        EventType.TOOL_RETURN: SourceClass.T,
        EventType.ENVIRONMENT_QUERY: SourceClass.E,
        EventType.ENVIRONMENT_TRANSITION: SourceClass.E,
    }

    _HISTORY_EVENT_TYPES = frozenset(
        {
            EventType.GENERATION,
            EventType.TOOL_REQUEST,
            EventType.TOOL_RETURN,
            EventType.ACTION,
            EventType.ENVIRONMENT_QUERY,
            EventType.ENVIRONMENT_TRANSITION,
        }
    )

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        memory_store: MemoryStore | None = None,
        memory_namespace: str | None = None,
        environment: EnvironmentState | None = None,
        transition_engine: DeterministicTransitionEngine | None = None,
    ) -> None:
        self.runtime = runtime

        self.memory_store = memory_store
        self.memory_namespace = memory_namespace

        self.environment = environment
        self.transition_engine = (
            transition_engine
        )

        if (
            self.memory_store is None
            and self.memory_namespace is not None
        ):
            raise ScaffoldExecutionError(
                code="memory_configuration_mismatch",
                message=(
                    "memory_namespace was supplied without "
                    "a memory_store."
                ),
            )

        if (
            self.memory_store is not None
            and self.memory_namespace is None
        ):
            raise ScaffoldExecutionError(
                code="memory_configuration_mismatch",
                message=(
                    "memory_store was supplied without "
                    "memory_namespace."
                ),
            )

        if (
            self.transition_engine is not None
            and self.environment is None
        ):
            raise ScaffoldExecutionError(
                code="environment_configuration_mismatch",
                message=(
                    "transition_engine was supplied without "
                    "EnvironmentState."
                ),
            )

    def execute_tool(
        self,
        *,
        tool: Tool,
        request: Mapping[str, Any],
        state: Any = None,
    ) -> ToolInteractionRecord:
        """
        Execute one deterministic T-channel tool interaction.

        C0 cannot execute tools. The tool request and return are both
        logged. The returned ContextElement is backed by TOOL_RETURN.
        """

        self._require_task()

        policy = get_condition_policy(
            self.runtime.identity.condition
        )

        if not policy.allow_tools:
            raise ScaffoldExecutionError(
                code="tool_not_allowed",
                message=(
                    f"Condition "
                    f"{self.runtime.identity.condition.value} "
                    "does not permit tool interaction."
                ),
            )

        request_payload = deepcopy(
            dict(request)
        )

        tool_call_id = (
            f"call_{uuid4().hex}"
        )

        request_event = (
            self.runtime.record_component_event(
                event_type=(
                    EventType.TOOL_REQUEST
                ),
                source_component=(
                    tool.name
                ),
                raw_payload=(
                    request_payload
                ),
                normalized_payload=(
                    deepcopy(
                        request_payload
                    )
                ),
                tool_call_id=(
                    tool_call_id
                ),
            )
        )

        try:
            result = tool.execute(
                request_payload,
                state,
            )

        except ToolExecutionError as exc:
            self.runtime.record_technical_failure(
                stage="tool_execution",
                error=exc,
            )

            raise ScaffoldExecutionError(
                code="tool_execution_failure",
                message=(
                    f"Tool {tool.name!r} failed: "
                    f"{exc.message}"
                ),
            ) from exc

        return_event = (
            self.runtime.record_component_event(
                event_type=(
                    EventType.TOOL_RETURN
                ),
                source_component=(
                    tool.name
                ),
                raw_payload=(
                    result.to_dict()
                ),
                normalized_payload=(
                    deepcopy(
                        result.normalized_result
                    )
                ),
                parent_event_id=(
                    request_event.event_id
                ),
                tool_call_id=(
                    tool_call_id
                ),
            )
        )

        tool_element = ContextElement(
            source=SourceClass.T,
            source_id=(
                f"tool_return:"
                f"{return_event.event_id}"
            ),
            content=deepcopy(
                result.normalized_result
            ),
            origin_event_id=(
                return_event.event_id
            ),
            origin_source=SourceClass.T,
            created_sequence=(
                return_event.sequence_index
            ),
        )

        return ToolInteractionRecord(
            tool_call_id=tool_call_id,
            request_event=request_event,
            return_event=return_event,
            result=result,
            tool_element=tool_element,
        )

    def retain_history(
        self,
        *,
        event_id: str,
    ) -> ContextElement:
        """
        Represent one valid prior trajectory event as H.

        The original event remains unchanged. H is a retained
        representation of that prior event.
        """

        self._require_task()

        policy = get_condition_policy(
            self.runtime.identity.condition
        )

        if not policy.allow_history:
            raise ScaffoldExecutionError(
                code="history_not_allowed",
                message=(
                    f"Condition "
                    f"{self.runtime.identity.condition.value} "
                    "does not permit trajectory history H."
                ),
            )

        origin_event = self._find_event(
            event_id
        )

        if (
            origin_event.event_type
            not in self._HISTORY_EVENT_TYPES
        ):
            raise ScaffoldExecutionError(
                code="invalid_history_event_type",
                message=(
                    f"Event type "
                    f"{origin_event.event_type.value!r} "
                    "cannot be retained as H."
                ),
            )

        origin_source = (
            self._HISTORY_ORIGIN_SOURCE.get(
                origin_event.event_type
            )
        )

        return ContextElement(
            source=SourceClass.H,
            source_id=(
                f"history:"
                f"{origin_event.event_id}"
            ),
            content=deepcopy(
                origin_event.normalized_payload
            ),
            origin_event_id=(
                origin_event.event_id
            ),
            origin_source=(
                origin_source
            ),
            created_sequence=(
                origin_event.sequence_index
            ),
            retrieved_sequence=(
                self._next_sequence_index()
            ),
        )

    def write_memory(
        self,
        *,
        memory_id: str,
        key: str,
        value: Any,
        metadata: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> MemoryWriteObservation:
        """
        Create one persistent-memory item and log MEMORY_WRITE.

        Persistent memory is available only in C3.

        A write mutates the backing memory state but does not itself
        create model-visible M. A subsequent read_memory() call must
        retrieve the record and create the event-backed M ContextElement.
        """

        self._require_task()

        policy = get_condition_policy(
            self.runtime.identity.condition
        )

        if not policy.allow_persistent_memory:
            raise ScaffoldExecutionError(
                code="memory_not_allowed",
                message=(
                    f"Condition "
                    f"{self.runtime.identity.condition.value} "
                    "does not permit persistent memory M."
                ),
            )

        if (
            self.memory_store is None
            or self.memory_namespace is None
        ):
            raise ScaffoldExecutionError(
                code="memory_not_configured",
                message=(
                    "Persistent memory has not been configured "
                    "for this scaffold controller."
                ),
            )

        created_sequence = (
            self._next_sequence_index()
        )

        metadata_value = (
            deepcopy(
                dict(
                    metadata
                )
            )
            if metadata is not None
            else {}
        )

        try:
            record = self.memory_store.write(
                namespace=(
                    self.memory_namespace
                ),
                memory_id=memory_id,
                key=key,
                value=deepcopy(
                    value
                ),
                created_by_task=(
                    self.runtime
                    .identity
                    .task_id
                ),
                created_sequence=(
                    created_sequence
                ),
                metadata=(
                    metadata_value
                ),
            )

        except Exception as exc:
            self.runtime.record_technical_failure(
                stage="memory_write",
                error=exc,
            )

            raise ScaffoldExecutionError(
                code="memory_write_failure",
                message=(
                    "Persistent-memory write failed."
                ),
            ) from exc

        raw_payload = {
            "record": (
                record.to_dict()
            ),
        }

        normalized_payload = {
            "key": record.key,
            "value": deepcopy(
                record.value
            ),
        }

        event = (
            self.runtime.record_component_event(
                event_type=(
                    EventType.MEMORY_WRITE
                ),
                source_component=(
                    "persistent_memory"
                ),
                raw_payload=(
                    raw_payload
                ),
                normalized_payload=(
                    normalized_payload
                ),
                memory_id=memory_id,
            )
        )

        if (
            event.sequence_index
            != created_sequence
        ):
            error = ScaffoldExecutionError(
                code="memory_write_sequence_mismatch",
                message=(
                    "MEMORY_WRITE event sequence does not "
                    "match the persistent-memory record's "
                    "created_sequence."
                ),
            )

            self.runtime.record_technical_failure(
                stage="memory_write",
                error=error,
            )

            raise error

        return MemoryWriteObservation(
            event=event,
            record=record,
        )

    def read_memory(
        self,
        *,
        memory_id: str,
    ) -> MemoryReadObservation:
        """
        Read one exact persistent-memory item and log MEMORY_READ.

        Only C3 permits M.
        """

        self._require_task()

        policy = get_condition_policy(
            self.runtime.identity.condition
        )

        if not policy.allow_persistent_memory:
            raise ScaffoldExecutionError(
                code="memory_not_allowed",
                message=(
                    f"Condition "
                    f"{self.runtime.identity.condition.value} "
                    "does not permit persistent memory M."
                ),
            )

        if (
            self.memory_store is None
            or self.memory_namespace is None
        ):
            raise ScaffoldExecutionError(
                code="memory_not_configured",
                message=(
                    "Persistent memory has not been configured "
                    "for this scaffold controller."
                ),
            )

        record = self.memory_store.read(
            namespace=(
                self.memory_namespace
            ),
            memory_id=memory_id,
        )

        if record is None:
            normalized_payload = {
                "found": False,
                "memory_id": memory_id,
            }

            raw_payload = deepcopy(
                normalized_payload
            )

        else:
            normalized_payload = {
                "key": record.key,
                "value": deepcopy(
                    record.value
                ),
            }

            raw_payload = {
                "found": True,
                "record": record.to_dict(),
            }

        event = (
            self.runtime.record_component_event(
                event_type=(
                    EventType.MEMORY_READ
                ),
                source_component=(
                    "persistent_memory"
                ),
                raw_payload=(
                    raw_payload
                ),
                normalized_payload=(
                    normalized_payload
                ),
                memory_id=memory_id,
            )
        )

        if record is None:
            return MemoryReadObservation(
                event=event,
                record=None,
                memory_element=None,
            )

        memory_element = ContextElement(
            source=SourceClass.M,
            source_id=(
                f"memory:{memory_id}"
            ),
            content=deepcopy(
                normalized_payload
            ),
            origin_event_id=(
                event.event_id
            ),
            origin_source=SourceClass.M,
            created_sequence=(
                event.sequence_index
            ),
            metadata={
                "memory_id": memory_id,
            },
        )

        return MemoryReadObservation(
            event=event,
            record=record,
            memory_element=(
                memory_element
            ),
        )

    def query_environment(
        self,
        *,
        path: Sequence[str],
    ) -> EnvironmentObservation:
        """
        Query deterministic environmental state and create event-backed E.

        E may exist under any condition as exogenous task state, but C0
        still cannot expose the resulting E element to the model. That
        restriction remains enforced by ContextBuilder.
        """

        self._require_task()

        if self.environment is None:
            raise ScaffoldExecutionError(
                code="environment_not_configured",
                message=(
                    "EnvironmentState has not been configured "
                    "for this scaffold controller."
                ),
            )

        result = self.environment.query(
            path
        )

        event = (
            self.runtime.record_component_event(
                event_type=(
                    EventType.ENVIRONMENT_QUERY
                ),
                source_component=(
                    "environment"
                ),
                raw_payload=(
                    result.to_dict()
                ),
                normalized_payload=(
                    result.to_dict()
                ),
            )
        )

        path_name = ".".join(
            result.path
        )

        environment_element = (
            ContextElement(
                source=SourceClass.E,
                source_id=(
                    f"environment:"
                    f"{path_name}"
                ),
                content=deepcopy(
                    result.to_dict()
                ),
                origin_event_id=(
                    event.event_id
                ),
                origin_source=(
                    SourceClass.E
                ),
                created_sequence=(
                    event.sequence_index
                ),
            )
        )

        return EnvironmentObservation(
            event=event,
            result=result,
            environment_element=(
                environment_element
            ),
        )

    def transition_environment(
        self,
        *,
        action: Mapping[str, Any],
    ) -> EnvironmentTransitionObservation:
        """
        Apply one preregistered deterministic environment transition.

        This records task dynamics. It does not automatically expose the
        transition as E or H; the task specification must explicitly choose
        whether to query the new E state or retain the transition as H.
        """

        self._require_task()

        if self.environment is None:
            raise ScaffoldExecutionError(
                code="environment_not_configured",
                message=(
                    "EnvironmentState has not been configured."
                ),
            )

        if self.transition_engine is None:
            raise ScaffoldExecutionError(
                code="transition_engine_not_configured",
                message=(
                    "DeterministicTransitionEngine has not "
                    "been configured."
                ),
            )

        action_payload = deepcopy(
            dict(action)
        )

        try:
            result = (
                self.transition_engine.apply(
                    environment=(
                        self.environment
                    ),
                    action=action_payload,
                )
            )

        except Exception as exc:
            self.runtime.record_technical_failure(
                stage=(
                    "environment_transition"
                ),
                error=exc,
            )

            raise ScaffoldExecutionError(
                code="environment_transition_failure",
                message=(
                    "Deterministic environment transition "
                    "failed."
                ),
            ) from exc

        event = (
            self.runtime.record_component_event(
                event_type=(
                    EventType.ENVIRONMENT_TRANSITION
                ),
                source_component=(
                    "environment"
                ),
                raw_payload=(
                    result.to_dict()
                ),
                normalized_payload=(
                    result.to_dict()
                ),
            )
        )

        return EnvironmentTransitionObservation(
            event=event,
            result=result,
        )

    def invoke_focal_action(
        self,
        *,
        config: InferenceConfig,
        tool_elements: Sequence[
            ContextElement
        ] = (),
        history_elements: Sequence[
            ContextElement
        ] = (),
        memory_elements: Sequence[
            ContextElement
        ] = (),
        environment_elements: Sequence[
            ContextElement
        ] = (),
    ) -> FocalActionRecord:
        """
        Commit the focal action with explicitly selected scaffold state.

        Environmental authorization is derived only from the E elements
        explicitly supplied here.
        """

        environment_policy = (
            self.environment_policy_for(
                environment_elements
            )
        )

        return self.runtime.invoke_focal_action(
            config=config,
            tool_elements=tool_elements,
            history_elements=(
                history_elements
            ),
            memory_elements=(
                memory_elements
            ),
            environment_elements=(
                environment_elements
            ),
            environment_policy=(
                environment_policy
            ),
        )

    @staticmethod
    def environment_policy_for(
        elements: Sequence[
            ContextElement
        ],
    ) -> TaskEnvironmentPolicy | None:
        """
        Build exact task-level E authorization for the supplied elements.
        """

        if not elements:
            return None

        source_ids: set[str] = set()

        for element in elements:
            if (
                element.source
                != SourceClass.E
            ):
                raise ScaffoldExecutionError(
                    code="non_environment_element",
                    message=(
                        "environment_policy_for() received "
                        "a non-E context element."
                    ),
                )

            source_ids.add(
                element.source_id
            )

        return TaskEnvironmentPolicy(
            allow_environment=True,
            allowed_source_ids=frozenset(
                source_ids
            ),
        )

    def _require_task(self) -> None:
        if not self.runtime.task_has_begun:
            raise ScaffoldExecutionError(
                code="task_not_started",
                message=(
                    "The AgentRuntime task must be started "
                    "before scaffold operations."
                ),
            )

    def _find_event(
        self,
        event_id: str,
    ) -> ExperimentEvent:
        for event in self.runtime.events:
            if event.event_id == event_id:
                return event

        raise ScaffoldExecutionError(
            code="event_not_found",
            message=(
                f"Trajectory event {event_id!r} "
                "was not found."
            ),
        )

    def _next_sequence_index(
        self,
    ) -> int:
        if not self.runtime.events:
            return 0

        return (
            self.runtime.events[-1]
            .sequence_index
            + 1
        )