__author__ = "Jason M. Pittman"
__date__ = "August 23, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Sequence

from ..logging import (
    EventType,
    ExperimentEvent,
    SourceClass,
)
from .context import StructuredContext


class ProvenanceAuditError(ValueError):
    """Defined failure during event-backed context provenance audit."""

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
class ProvenanceAuditRecord:
    """Successful provenance binding for one model-visible context element."""

    source: SourceClass
    source_id: str

    origin_event_id: str
    origin_event_type: EventType
    origin_event_sequence: int

    origin_source: SourceClass | None

    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "source_id": self.source_id,
            "origin_event_id": self.origin_event_id,
            "origin_event_type": self.origin_event_type.value,
            "origin_event_sequence": self.origin_event_sequence,
            "origin_source": (
                self.origin_source.value
                if self.origin_source is not None
                else None
            ),
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class ProvenanceAuditResult:
    """Complete successful audit for one structured context build."""

    experiment_id: str
    run_id: str
    task_id: str
    condition: str

    context_build_event_id: str
    context_build_sequence: int

    records: tuple[ProvenanceAuditRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "condition": self.condition,
            "context_build_event_id": self.context_build_event_id,
            "context_build_sequence": self.context_build_sequence,
            "records": [
                record.to_dict()
                for record in self.records
            ],
        }


class EventBackedProvenanceAuditor:
    """
    Verify model-visible context provenance against the trajectory event stream.

    The auditor does not infer provenance from text. It requires explicit
    event linkage and validates that the claimed source role is consistent
    with the actual backing event.
    """

    _DIRECT_EVENT_TYPES = {
        SourceClass.P: frozenset(
            {
                EventType.TASK,
            }
        ),
        SourceClass.T: frozenset(
            {
                EventType.TOOL_RETURN,
            }
        ),
        SourceClass.M: frozenset(
            {
                EventType.MEMORY_READ,
            }
        ),
        SourceClass.E: frozenset(
            {
                EventType.ENVIRONMENT_QUERY,
                EventType.ENVIRONMENT_TRANSITION,
            }
        ),
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

    _EVENT_ORIGIN_SOURCE = {
        EventType.TASK: SourceClass.P,
        EventType.TOOL_RETURN: SourceClass.T,
        EventType.MEMORY_READ: SourceClass.M,
        EventType.ENVIRONMENT_QUERY: SourceClass.E,
        EventType.ENVIRONMENT_TRANSITION: SourceClass.E,
    }

    def audit(
        self,
        *,
        context: StructuredContext,
        events: Sequence[ExperimentEvent],
        context_build_event_id: str,
    ) -> ProvenanceAuditResult:
        """
        Audit every context element against one logged context-build event
        and its preceding trajectory events.
        """

        event_index = self._build_event_index(
            events
        )

        try:
            context_build_event = event_index[
                context_build_event_id
            ]
        except KeyError as exc:
            raise ProvenanceAuditError(
                code="context_build_event_not_found",
                message=(
                    "Context-build event was not found in the supplied "
                    f"event stream: {context_build_event_id!r}."
                ),
            ) from exc

        if (
            context_build_event.event_type
            != EventType.CONTEXT_BUILD
        ):
            raise ProvenanceAuditError(
                code="invalid_context_build_event",
                message=(
                    f"Event {context_build_event_id!r} is not a "
                    "context_build event."
                ),
            )

        self._validate_context_anchor(
            context=context,
            context_build_event=context_build_event,
        )

        records = tuple(
            self._audit_element(
                element=element,
                event_index=event_index,
                context_build_event=context_build_event,
                context=context,
            )
            for element in context.elements
        )

        return ProvenanceAuditResult(
            experiment_id=(
                context_build_event.experiment_id
            ),
            run_id=context_build_event.run_id,
            task_id=context_build_event.task_id,
            condition=(
                context_build_event.condition.value
            ),
            context_build_event_id=(
                context_build_event.event_id
            ),
            context_build_sequence=(
                context_build_event.sequence_index
            ),
            records=records,
        )

    def _audit_element(
        self,
        *,
        element,
        event_index: dict[str, ExperimentEvent],
        context_build_event: ExperimentEvent,
        context: StructuredContext,
    ) -> ProvenanceAuditRecord:
        self._validate_source_id_convention(
            element=element,
            context=context,
        )

        if not element.origin_event_id:
            raise ProvenanceAuditError(
                code="missing_origin_event_id",
                message=(
                    f"Context element {element.source_id!r} has no "
                    "origin_event_id and cannot be event-backed."
                ),
            )

        try:
            origin_event = event_index[
                element.origin_event_id
            ]
        except KeyError as exc:
            raise ProvenanceAuditError(
                code="origin_event_not_found",
                message=(
                    f"Origin event {element.origin_event_id!r} for "
                    f"context element {element.source_id!r} was not "
                    "found."
                ),
            ) from exc

        self._validate_event_scope(
            origin_event=origin_event,
            context_build_event=context_build_event,
        )

        if (
            origin_event.sequence_index
            >= context_build_event.sequence_index
        ):
            raise ProvenanceAuditError(
                code="origin_event_not_prior",
                message=(
                    f"Origin event {origin_event.event_id!r} must occur "
                    "before the context-build event."
                ),
            )

        self._validate_event_type(
            element=element,
            origin_event=origin_event,
        )

        self._validate_origin_source(
            element=element,
            origin_event=origin_event,
        )

        self._validate_payload_binding(
            element=element,
            origin_event=origin_event,
        )

        if element.source == SourceClass.M:
            self._validate_memory_binding(
                element=element,
                origin_event=origin_event,
            )

        return ProvenanceAuditRecord(
            source=element.source,
            source_id=element.source_id,
            origin_event_id=origin_event.event_id,
            origin_event_type=origin_event.event_type,
            origin_event_sequence=(
                origin_event.sequence_index
            ),
            origin_source=element.origin_source,
            content_sha256=self._content_checksum(
                element.content
            ),
        )

    def _validate_context_anchor(
        self,
        *,
        context: StructuredContext,
        context_build_event: ExperimentEvent,
    ) -> None:
        if (
            context.task_id
            != context_build_event.task_id
        ):
            raise ProvenanceAuditError(
                code="context_task_mismatch",
                message=(
                    "Structured context task_id does not match the "
                    "context-build event."
                ),
            )

        if (
            context.condition
            != context_build_event.condition
        ):
            raise ProvenanceAuditError(
                code="context_condition_mismatch",
                message=(
                    "Structured context condition does not match the "
                    "context-build event."
                ),
            )

        if (
            context_build_event.normalized_payload
            != context.to_dict()
        ):
            raise ProvenanceAuditError(
                code="context_build_payload_mismatch",
                message=(
                    "Logged context-build normalized payload does not "
                    "exactly match the audited StructuredContext."
                ),
            )

    @staticmethod
    def _build_event_index(
        events: Sequence[ExperimentEvent],
    ) -> dict[str, ExperimentEvent]:
        index: dict[str, ExperimentEvent] = {}

        for event in events:
            if event.event_id in index:
                raise ProvenanceAuditError(
                    code="duplicate_event_id",
                    message=(
                        "Duplicate event ID encountered during "
                        f"provenance audit: {event.event_id!r}."
                    ),
                )

            index[event.event_id] = event

        return index

    @staticmethod
    def _validate_event_scope(
        *,
        origin_event: ExperimentEvent,
        context_build_event: ExperimentEvent,
    ) -> None:
        if (
            origin_event.experiment_id
            != context_build_event.experiment_id
        ):
            raise ProvenanceAuditError(
                code="cross_experiment_provenance",
                message=(
                    "Context provenance may not reference an event "
                    "from another experiment."
                ),
            )

        if (
            origin_event.run_id
            != context_build_event.run_id
        ):
            raise ProvenanceAuditError(
                code="cross_run_provenance",
                message=(
                    "Context provenance may not reference an event "
                    "from another run."
                ),
            )

        if (
            origin_event.task_id
            != context_build_event.task_id
        ):
            raise ProvenanceAuditError(
                code="cross_task_provenance",
                message=(
                    "Context provenance may not reference an event "
                    "from another task."
                ),
            )

        if (
            origin_event.condition
            != context_build_event.condition
        ):
            raise ProvenanceAuditError(
                code="cross_condition_provenance",
                message=(
                    "Context provenance may not reference an event "
                    "from another experimental condition."
                ),
            )

    def _validate_event_type(
        self,
        *,
        element,
        origin_event: ExperimentEvent,
    ) -> None:
        if element.source == SourceClass.H:
            allowed = self._HISTORY_EVENT_TYPES
        else:
            try:
                allowed = self._DIRECT_EVENT_TYPES[
                    element.source
                ]
            except KeyError as exc:
                raise ProvenanceAuditError(
                    code="unsupported_context_source",
                    message=(
                        "No provenance audit rule exists for source "
                        f"{element.source.value!r}."
                    ),
                ) from exc

        if origin_event.event_type not in allowed:
            raise ProvenanceAuditError(
                code="origin_event_type_mismatch",
                message=(
                    f"Context source {element.source.value!r} cannot "
                    f"be backed by event type "
                    f"{origin_event.event_type.value!r}."
                ),
            )

    def _validate_origin_source(
        self,
        *,
        element,
        origin_event: ExperimentEvent,
    ) -> None:
        expected = self._EVENT_ORIGIN_SOURCE.get(
            origin_event.event_type
        )

        if element.source == SourceClass.H:
            if expected is None:
                if element.origin_source is not None:
                    raise ProvenanceAuditError(
                        code="origin_source_mismatch",
                        message=(
                            f"History element {element.source_id!r} "
                            "must not claim an external origin_source "
                            "for this event type."
                        ),
                    )

            elif element.origin_source != expected:
                raise ProvenanceAuditError(
                    code="origin_source_mismatch",
                    message=(
                        f"History element {element.source_id!r} must "
                        f"record origin_source={expected.value!r}."
                    ),
                )

            return

        if (
            element.origin_source is not None
            and element.origin_source
            != element.source
        ):
            raise ProvenanceAuditError(
                code="origin_source_mismatch",
                message=(
                    f"Direct source {element.source.value!r} may "
                    "either omit origin_source or record the same "
                    "source class."
                ),
            )

    @staticmethod
    def _validate_payload_binding(
        *,
        element,
        origin_event: ExperimentEvent,
    ) -> None:
        if (
            origin_event.normalized_payload
            != element.content
        ):
            raise ProvenanceAuditError(
                code="origin_payload_mismatch",
                message=(
                    f"Context element {element.source_id!r} does not "
                    "exactly match the normalized payload of its "
                    "backing event."
                ),
            )

    @staticmethod
    def _validate_memory_binding(
        *,
        element,
        origin_event: ExperimentEvent,
    ) -> None:
        event_memory_id = origin_event.memory_id

        element_memory_id = (
            element.metadata.get(
                "memory_id"
            )
        )

        if not event_memory_id:
            raise ProvenanceAuditError(
                code="memory_event_missing_id",
                message=(
                    "A memory_read event used as model-visible M "
                    "must contain memory_id."
                ),
            )

        if (
            element_memory_id
            != event_memory_id
        ):
            raise ProvenanceAuditError(
                code="memory_id_mismatch",
                message=(
                    "Context memory_id does not match its backing "
                    "memory_read event."
                ),
            )

    @staticmethod
    def _validate_source_id_convention(
        *,
        element,
        context: StructuredContext,
    ) -> None:
        if element.source == SourceClass.P:
            expected = (
                f"prompt:{context.task_id}"
            )

            if element.source_id != expected:
                raise ProvenanceAuditError(
                    code="source_id_binding_mismatch",
                    message=(
                        "Prompt source_id must equal "
                        f"{expected!r}."
                    ),
                )

            return

        if element.source == SourceClass.T:
            if not element.origin_event_id:
                return

            expected = (
                f"tool_return:{element.origin_event_id}"
            )

            if element.source_id != expected:
                raise ProvenanceAuditError(
                    code="source_id_binding_mismatch",
                    message=(
                        "Tool-return source_id must bind directly to "
                        f"its origin event as {expected!r}."
                    ),
                )

            return

        if element.source == SourceClass.H:
            if not element.origin_event_id:
                return

            expected = (
                f"history:{element.origin_event_id}"
            )

            if element.source_id != expected:
                raise ProvenanceAuditError(
                    code="source_id_binding_mismatch",
                    message=(
                        "History source_id must bind directly to its "
                        f"origin event as {expected!r}."
                    ),
                )

            return

        if element.source == SourceClass.M:
            memory_id = element.metadata.get(
                "memory_id"
            )

            if not isinstance(
                memory_id,
                str,
            ) or not memory_id:
                raise ProvenanceAuditError(
                    code="memory_context_missing_id",
                    message=(
                        "Memory context elements must contain a "
                        "non-empty metadata['memory_id']."
                    ),
                )

            expected = (
                f"memory:{memory_id}"
            )

            if element.source_id != expected:
                raise ProvenanceAuditError(
                    code="source_id_binding_mismatch",
                    message=(
                        "Memory source_id must bind directly to its "
                        f"memory ID as {expected!r}."
                    ),
                )

            return

        if element.source == SourceClass.E:
            if not element.source_id.startswith(
                "environment:"
            ):
                raise ProvenanceAuditError(
                    code="source_id_binding_mismatch",
                    message=(
                        "Environment source_id must begin with "
                        "'environment:'."
                    ),
                )

    @staticmethod
    def _content_checksum(
        content: Any,
    ) -> str:
        if isinstance(
            content,
            str,
        ):
            serialized = content
        else:
            serialized = json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )

        return sha256(
            serialized.encode(
                "utf-8"
            )
        ).hexdigest()