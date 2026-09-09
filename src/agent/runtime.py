__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.6"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from uuid import uuid4

from ..logging import (
    Condition,
    ContextElement,
    EventFactory,
    EventType,
    ExperimentEvent,
    RawEventWriter,
    SourceClass,
    TaskClassification,
)
from ..model import (
    InferenceConfig,
    ModelAdapter,
    ModelAdapterError,
    ModelGenerationResult,
    ModelInputRecord,
)
from .context import (
    ContextBuildError,
    ContextBuilder,
    ContextSerializationError,
    ContextSerializer,
    SerializedContext,
    StructuredContext,
    TaskEnvironmentPolicy,
)
from .provenance import (
    EventBackedProvenanceAuditor,
    ProvenanceAuditError,
    ProvenanceAuditResult,
)

from ..normalization import (
    ActionParseError,
    ActionParseResult,
    FocalActionPrompt,
    FocalActionPromptBuilder,
    FocalActionSchema,
    StructuredActionParser,
)

from ..normalization import (
    ActionParseError,
    ActionParseResult,
    FocalActionPrompt,
    FocalActionPromptBuilder,
    FocalActionSchema,
    StructuredActionParser,
)

from .explanation import (
    SelfExplanationParseError,
    SelfExplanationParseResult,
    SelfExplanationPrompt,
    SelfExplanationPromptBuilder,
    StructuredSelfExplanationParser,
)

class AgentRuntimeError(RuntimeError):
    """Defined failure raised by the experimental agent runtime."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.code = code
        self.message = message
        self.details = deepcopy(
            details or {}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": deepcopy(
                self.details
            ),
        }


@dataclass(frozen=True)
class RunIdentity:
    """Immutable identity shared by every event in one experimental run."""

    experiment_id: str
    run_id: str
    task_id: str

    task_classification: TaskClassification
    condition: Condition

    repetition_id: int
    seed: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_classification": (
                self.task_classification.value
            ),
            "condition": self.condition.value,
            "repetition_id": self.repetition_id,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ModelInvocationRecord:
    """
    Complete in-memory record of one successfully audited model invocation.

    The corresponding evidence is also preserved in the append-only event
    stream.
    """

    model_invocation_id: str

    context_build_event_id: str
    provenance_audit_event_id: str
    generation_event_id: str

    structured_context: StructuredContext
    serialized_context: SerializedContext
    model_input: ModelInputRecord

    provenance_audit: ProvenanceAuditResult
    generation: ModelGenerationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_invocation_id": (
                self.model_invocation_id
            ),
            "context_build_event_id": (
                self.context_build_event_id
            ),
            "provenance_audit_event_id": (
                self.provenance_audit_event_id
            ),
            "generation_event_id": (
                self.generation_event_id
            ),
            "structured_context": (
                self.structured_context.to_dict()
            ),
            "serialized_context": {
                "model_text": (
                    self.serialized_context.model_text
                ),
                **self.serialized_context.to_audit_dict(),
            },
            "model_input": (
                self.model_input.to_dict()
            ),
            "provenance_audit": (
                self.provenance_audit.to_dict()
            ),
            "generation": (
                self.generation.to_dict()
            ),
        }

@dataclass(frozen=True)
class FocalActionRecord:
    """
    Result of one committed focal consequential decision.

    An invalid/unparseable model response is retained as an ACTION
    observation rather than converted into a technical failure.
    """

    focal_decision_id: str
    action_event_id: str

    model_invocation: ModelInvocationRecord

    valid: bool

    parse_result: (
        ActionParseResult | None
    )

    parse_error: (
        dict[str, Any] | None
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "focal_decision_id": (
                self.focal_decision_id
            ),
            "action_event_id": (
                self.action_event_id
            ),
            "model_invocation_id": (
                self.model_invocation
                .model_invocation_id
            ),
            "generation_event_id": (
                self.model_invocation
                .generation_event_id
            ),
            "valid": self.valid,
            "parse_result": (
                self.parse_result.to_dict()
                if self.parse_result
                is not None
                else None
            ),
            "parse_error": deepcopy(
                self.parse_error
            ),
        }

@dataclass(frozen=True)
class SelfExplanationRecord:
    """
    Result of one post-action structured self-explanation.

    Invalid explanation syntax is preserved as experimental behavior and
    is not converted into a technical failure.
    """

    focal_decision_id: str

    action_event_id: str
    generation_event_id: str
    explanation_event_id: str

    prompt: SelfExplanationPrompt

    generation: ModelGenerationResult

    valid: bool

    parse_result: (
        SelfExplanationParseResult | None
    )

    parse_error: (
        dict[str, Any] | None
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "focal_decision_id": (
                self.focal_decision_id
            ),
            "action_event_id": (
                self.action_event_id
            ),
            "generation_event_id": (
                self.generation_event_id
            ),
            "explanation_event_id": (
                self.explanation_event_id
            ),
            "prompt": (
                self.prompt.to_dict()
            ),
            "generation": (
                self.generation.to_dict()
            ),
            "valid": self.valid,
            "parse_result": (
                self.parse_result.to_dict()
                if self.parse_result
                is not None
                else None
            ),
            "parse_error": deepcopy(
                self.parse_error
            ),
        }

class AgentRuntime:
    """
    Experimental runtime connecting context construction, provenance audit,
    model preparation, inference, event logging, and focal-action parsing.

    This stage intentionally does not implement:

    - automatic tool execution;
    - persistent-memory policy;
    - environmental transitions;
    - self-explanation;
    - intervention/replay.
    """

    _COMPONENT_EVENT_TYPES = frozenset(
        {
            EventType.TOOL_REQUEST,
            EventType.TOOL_RETURN,
            EventType.MEMORY_READ,
            EventType.MEMORY_WRITE,
            EventType.ENVIRONMENT_QUERY,
            EventType.ENVIRONMENT_TRANSITION,
            EventType.ACTION,
            EventType.INTERVENTION,
            EventType.SHAM_INTERVENTION,
        }
    )

    def __init__(
        self,
        *,
        identity: RunIdentity,
        model_adapter: ModelAdapter,
        event_writer: RawEventWriter,
        context_builder: ContextBuilder | None = None,
        context_serializer: ContextSerializer | None = None,
        provenance_auditor: EventBackedProvenanceAuditor | None = None,
        action_prompt_builder: FocalActionPromptBuilder | None = None,
        action_parser: StructuredActionParser | None = None,
        explanation_prompt_builder: SelfExplanationPromptBuilder | None = None,
        explanation_parser: StructuredSelfExplanationParser | None = None,
    ) -> None:
        self.identity = identity
        self.model_adapter = model_adapter
        self.event_writer = event_writer

        self.context_builder = (
            context_builder
            or ContextBuilder()
        )

        self.context_serializer = (
            context_serializer
            or ContextSerializer()
        )

        self.provenance_auditor = (
            provenance_auditor
            or EventBackedProvenanceAuditor()
        )

        self.action_prompt_builder = (
            action_prompt_builder
            or FocalActionPromptBuilder()
        )

        self.action_parser = (
            action_parser
            or StructuredActionParser()
        )

        self.explanation_prompt_builder = (
            explanation_prompt_builder
            or SelfExplanationPromptBuilder()
        )

        self.explanation_parser = (
            explanation_parser
            or StructuredSelfExplanationParser()
        )

        self._validate_identity()

        self._event_factory = EventFactory(
            experiment_id=(
                self.identity.experiment_id
            ),
            run_id=self.identity.run_id,
            task_id=self.identity.task_id,
            task_classification=(
                self.identity.task_classification
            ),
            condition=self.identity.condition,
            repetition_id=(
                self.identity.repetition_id
            ),
            seed=self.identity.seed,
        )

        self._events: list[
            ExperimentEvent
        ] = []

        self._started = False

        self._task_event: (
            ExperimentEvent | None
        ) = None

        self._prompt_element: (
            ContextElement | None
        ) = None

        self._focal_action_schema: (
            FocalActionSchema | None
        ) = None

        self._focal_action_prompt: (
            FocalActionPrompt | None
        ) = None

        self._focal_action_event: (
            ExperimentEvent | None
        ) = None

        self._self_explanation_event: (
            ExperimentEvent | None
        ) = None

    @property
    def events(
        self,
    ) -> tuple[ExperimentEvent, ...]:
        """Return the current immutable event-sequence view."""

        return tuple(
            self._events
        )

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def task_has_begun(self) -> bool:
        return (
            self._task_event
            is not None
        )

    @property
    def prompt_element(
        self,
    ) -> ContextElement | None:
        """
        Return a defensive copy of the current canonical P element.

        This is used by snapshot/orchestration code without exposing
        mutable runtime internals.
        """

        return deepcopy(
            self._prompt_element
        )

    def start(self) -> ExperimentEvent:
        """
        Initialize the append-only event stream and record runtime metadata.

        The model must already be loaded. Model loading is intentionally
        outside experimental run execution.
        """

        if self._started:
            raise AgentRuntimeError(
                code="runtime_already_started",
                message=(
                    "Agent runtime has already been started."
                ),
            )

        if not self.model_adapter.is_loaded:
            raise AgentRuntimeError(
                code="model_not_loaded",
                message=(
                    "Model adapter must be loaded before the "
                    "experimental runtime starts."
                ),
            )

        self.event_writer.initialize()

        runtime_metadata = (
            self.model_adapter
            .runtime_metadata()
            .to_dict()
        )

        payload = {
            "run_identity": (
                self.identity.to_dict()
            ),
            "runtime_metadata": (
                runtime_metadata
            ),
        }

        event = self._event_factory.create(
            event_type=EventType.SYSTEM,
            source_component="agent_runtime",
            raw_payload=deepcopy(
                payload
            ),
            normalized_payload=deepcopy(
                payload
            ),
        )

        self._append_event(
            event
        )

        self._started = True

        return event

    def _record_task(
        self,
        *,
        raw_payload: Any,
        normalized_prompt: Any,
    ) -> ContextElement:
        """
        Record the single TASK event for this run and create its canonical
        prompt ContextElement P.

        raw_payload preserves task-construction metadata.

        normalized_prompt is the exact task prompt that becomes model-visible
        as P.
        """

        self._require_started()

        if self._task_event is not None:
            raise AgentRuntimeError(
                code="task_already_started",
                message=(
                    "The task has already been started for this run."
                ),
            )

        self._validate_json(
            raw_payload,
            field_name="task_raw_payload",
        )

        self._validate_json(
            normalized_prompt,
            field_name="task_normalized_prompt",
        )

        event = self._event_factory.create(
            event_type=EventType.TASK,
            source_component="task_runtime",
            raw_payload=deepcopy(
                raw_payload
            ),
            normalized_payload=deepcopy(
                normalized_prompt
            ),
        )

        self._append_event(
            event
        )

        self._task_event = event

        self._prompt_element = (
            ContextElement(
                source=SourceClass.P,
                source_id=(
                    f"prompt:{self.identity.task_id}"
                ),
                content=deepcopy(
                    normalized_prompt
                ),
                origin_event_id=(
                    event.event_id
                ),
                origin_source=SourceClass.P,
                created_sequence=(
                    event.sequence_index
                ),
            )
        )

        return self._prompt_element

    def begin_task(
        self,
        prompt_content: Any,
    ) -> ContextElement:
        """
        Begin a generic task without a focal-action contract.

        This preserves the task behavior implemented before focal-action
        integration.
        """

        return self._record_task(
            raw_payload=prompt_content,
            normalized_prompt=prompt_content,
        )

    def begin_focal_task(
        self,
        *,
        task_prompt: str,
        action_schema: FocalActionSchema,
    ) -> FocalActionPrompt:
        """
        Begin a task whose primary consequential decision uses a structured
        focal-action schema.

        The TASK event preserves both the original task and the action
        contract, while its normalized payload is the exact P exposed to the
        model.
        """

        self._require_started()

        if self._task_event is not None:
            raise AgentRuntimeError(
                code="task_already_started",
                message=(
                    "The task has already been started for this run."
                ),
            )

        try:
            focal_prompt = (
                self.action_prompt_builder.build(
                    task_prompt=task_prompt,
                    schema=action_schema,
                )
            )

        except Exception as exc:
            if hasattr(
                exc,
                "to_dict",
            ):
                details = exc.to_dict()
            else:
                details = {
                    "exception_type": (
                        type(exc).__name__
                    ),
                    "message": str(
                        exc
                    ),
                }

            raise AgentRuntimeError(
                code="focal_prompt_build_failure",
                message=(
                    "Unable to build focal-action prompt."
                ),
                details=details,
            ) from exc

        self._record_task(
            raw_payload={
                "task_prompt": task_prompt,
                "focal_action_prompt": (
                    focal_prompt.to_dict()
                ),
                "focal_action_schema": (
                    action_schema.to_dict()
                ),
            },
            normalized_prompt=(
                focal_prompt.rendered_prompt
            ),
        )

        self._focal_action_schema = (
            action_schema
        )

        self._focal_action_prompt = (
            focal_prompt
        )

        return focal_prompt

    def begin_replay_focal_task(
        self,
        *,
        prompt_content: Any,
        action_schema: FocalActionSchema,
        replay_metadata: Mapping[str, Any],
    ) -> ContextElement:
        """
        Begin a matched counterfactual replay from an already-rendered
        focal-decision prompt.

        Unlike begin_focal_task(), this method must not rebuild the focal
        prompt because doing so could alter the exact baseline/counterfactual
        P representation. The supplied prompt_content is therefore recorded
        directly as canonical P.

        replay_metadata is audit evidence only and is not model-visible.
        """

        self._require_started()

        if self._task_event is not None:
            raise AgentRuntimeError(
                code="task_already_started",
                message=(
                    "The task has already been started for this run."
                ),
            )

        self._validate_json(
            replay_metadata,
            field_name="replay_metadata",
        )

        prompt_element = self._record_task(
            raw_payload={
                "replay": True,
                "replay_metadata": deepcopy(
                    dict(
                        replay_metadata
                    )
                ),
                "focal_action_schema": (
                    action_schema.to_dict()
                ),
                "model_visible_prompt": (
                    deepcopy(
                        prompt_content
                    )
                ),
            },
            normalized_prompt=(
                deepcopy(
                    prompt_content
                )
            ),
        )

        self._focal_action_schema = (
            action_schema
        )

        self._focal_action_prompt = None

        return prompt_element

    def record_component_event(
        self,
        *,
        event_type: EventType,
        source_component: str,
        raw_payload: Any,
        normalized_payload: Any,
        parent_event_id: str | None = None,
        tool_call_id: str | None = None,
        memory_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentEvent:
        """
        Append a task-local component event prior to or between model calls.

        Higher-level tool/memory/environment orchestration will use this
        method in later stages.
        """

        self._require_task()

        if (
            event_type
            not in self._COMPONENT_EVENT_TYPES
        ):
            raise AgentRuntimeError(
                code="unsupported_component_event",
                message=(
                    f"Event type {event_type.value!r} may not be "
                    "recorded through record_component_event()."
                ),
            )

        self._validate_json(
            raw_payload,
            field_name="raw_payload",
        )

        self._validate_json(
            normalized_payload,
            field_name="normalized_payload",
        )

        metadata_value = deepcopy(
            dict(
                metadata
                or {}
            )
        )

        self._validate_json(
            metadata_value,
            field_name="metadata",
        )

        event = self._event_factory.create(
            event_type=event_type,
            source_component=source_component,
            raw_payload=deepcopy(
                raw_payload
            ),
            normalized_payload=deepcopy(
                normalized_payload
            ),
            parent_event_id=parent_event_id,
            tool_call_id=tool_call_id,
            memory_id=memory_id,
            metadata=metadata_value,
        )

        self._append_event(
            event
        )

        return event

    def record_technical_failure(
        self,
        *,
        stage: str,
        error: Exception,
        model_invocation_id: str | None = None,
    ) -> ExperimentEvent:
        """
        Record a task-local technical failure through the runtime's
        canonical TECHNICAL_FAILURE event path.

        This public wrapper allows scaffold components to preserve
        infrastructure failures without directly calling the runtime's
        private _record_failure() helper.
        """

        self._require_task()

        if (
            not isinstance(stage, str)
            or not stage
        ):
            raise AgentRuntimeError(
                code="invalid_failure_stage",
                message=(
                    "Technical-failure stage must be a "
                    "non-empty string."
                ),
            )

        return self._record_failure(
            stage=stage,
            error=error,
            model_invocation_id=(
                model_invocation_id
            ),
        )

    def invoke_model(
        self,
        *,
        config: InferenceConfig,
        tool_elements: Sequence[ContextElement] = (),
        history_elements: Sequence[ContextElement] = (),
        memory_elements: Sequence[ContextElement] = (),
        environment_elements: Sequence[ContextElement] = (),
        environment_policy: TaskEnvironmentPolicy | None = None,
    ) -> ModelInvocationRecord:
        """
        Construct, serialize, provenance-audit, and execute one model call.

        CONTEXT_BUILD is persisted before the provenance audit. Therefore,
        an audit failure remains visible in the raw event stream and is
        followed by TECHNICAL_FAILURE rather than silently disappearing.
        """

        self._require_task()

        if self._prompt_element is None:
            raise AgentRuntimeError(
                code="prompt_unavailable",
                message=(
                    "No canonical prompt element is available."
                ),
            )

        model_invocation_id = (
            f"mdl_{uuid4().hex}"
        )

        try:
            context = (
                self.context_builder.build(
                    task_id=(
                        self.identity.task_id
                    ),
                    condition=(
                        self.identity.condition
                    ),
                    prompt_elements=(
                        self._prompt_element,
                    ),
                    tool_elements=(
                        tool_elements
                    ),
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
            )

            serialized = (
                self.context_serializer.serialize(
                    context
                )
            )

            model_input = (
                self.model_adapter.prepare_input(
                    serialized.model_text
                )
            )

            self._validate_model_input_binding(
                serialized=serialized,
                model_input=model_input,
            )

        except (
            ContextBuildError,
            ContextSerializationError,
            ModelAdapterError,
        ) as exc:
            self._record_failure(
                stage="model_input_preparation",
                error=exc,
                model_invocation_id=(
                    model_invocation_id
                ),
            )

            raise AgentRuntimeError(
                code="model_input_preparation_failure",
                message=(
                    "Unable to prepare an auditable model invocation."
                ),
                details=self._error_details(
                    exc
                ),
            ) from exc

        context_payload = {
            "structured_context": (
                context.to_dict()
            ),
            "serialized_context": {
                "model_text": (
                    serialized.model_text
                ),
                **serialized.to_audit_dict(),
            },
            "model_input": (
                model_input.to_dict()
            ),
        }

        context_event = (
            self._event_factory.create(
                event_type=(
                    EventType.CONTEXT_BUILD
                ),
                source_component=(
                    "context_runtime"
                ),
                raw_payload=(
                    deepcopy(
                        context_payload
                    )
                ),
                normalized_payload=(
                    context.to_dict()
                ),
                model_invocation_id=(
                    model_invocation_id
                ),
            )
        )

        self._append_event(
            context_event
        )

        try:
            audit = (
                self.provenance_auditor.audit(
                    context=context,
                    events=self.events,
                    context_build_event_id=(
                        context_event.event_id
                    ),
                )
            )

        except ProvenanceAuditError as exc:
            self._record_failure(
                stage="provenance_audit",
                error=exc,
                model_invocation_id=(
                    model_invocation_id
                ),
            )

            raise AgentRuntimeError(
                code="provenance_audit_failure",
                message=(
                    "Model invocation failed event-backed "
                    "provenance validation."
                ),
                details=self._error_details(
                    exc
                ),
            ) from exc

        audit_event = (
            self._event_factory.create(
                event_type=(
                    EventType.PROVENANCE_AUDIT
                ),
                source_component=(
                    "provenance_auditor"
                ),
                raw_payload=(
                    audit.to_dict()
                ),
                normalized_payload=(
                    audit.to_dict()
                ),
                parent_event_id=(
                    context_event.event_id
                ),
                model_invocation_id=(
                    model_invocation_id
                ),
            )
        )

        self._append_event(
            audit_event
        )

        try:
            generation = (
                self.model_adapter.generate(
                    input_record=model_input,
                    config=config,
                )
            )

        except ModelAdapterError as exc:
            self._record_failure(
                stage="model_generation",
                error=exc,
                model_invocation_id=(
                    model_invocation_id
                ),
            )

            raise AgentRuntimeError(
                code="model_generation_failure",
                message=(
                    "Model generation failed."
                ),
                details=self._error_details(
                    exc
                ),
            ) from exc

        generation_event = (
            self._event_factory.create(
                event_type=(
                    EventType.GENERATION
                ),
                source_component=(
                    "model_adapter"
                ),
                raw_payload=(
                    generation.to_dict()
                ),
                normalized_payload=(
                    generation.raw_text
                ),
                parent_event_id=(
                    audit_event.event_id
                ),
                model_invocation_id=(
                    model_invocation_id
                ),
            )
        )

        self._append_event(
            generation_event
        )

        return ModelInvocationRecord(
            model_invocation_id=(
                model_invocation_id
            ),
            context_build_event_id=(
                context_event.event_id
            ),
            provenance_audit_event_id=(
                audit_event.event_id
            ),
            generation_event_id=(
                generation_event.event_id
            ),
            structured_context=context,
            serialized_context=(
                serialized
            ),
            model_input=model_input,
            provenance_audit=audit,
            generation=generation,
        )

    def invoke_focal_action(
        self,
        *,
        config: InferenceConfig,
        tool_elements: Sequence[ContextElement] = (),
        history_elements: Sequence[ContextElement] = (),
        memory_elements: Sequence[ContextElement] = (),
        environment_elements: Sequence[ContextElement] = (),
        environment_policy: TaskEnvironmentPolicy | None = None,
    ) -> FocalActionRecord:
        """
        Execute and commit the task's preregistered focal consequential
        decision.

        Invalid model syntax is recorded as an invalid ACTION event. It is
        experimental behavior, not a technical failure.
        """

        self._require_task()

        if self._focal_action_schema is None:
            raise AgentRuntimeError(
                code="focal_action_schema_unavailable",
                message=(
                    "begin_focal_task() must be used before "
                    "invoke_focal_action()."
                ),
            )

        if self._focal_action_event is not None:
            raise AgentRuntimeError(
                code="focal_action_already_committed",
                message=(
                    "The primary focal consequential decision has "
                    "already been committed for this task."
                ),
            )

        invocation = self.invoke_model(
            config=config,
            tool_elements=tool_elements,
            history_elements=history_elements,
            memory_elements=memory_elements,
            environment_elements=environment_elements,
            environment_policy=environment_policy,
        )

        raw_text = (
            invocation.generation.raw_text
        )

        parse_result: (
            ActionParseResult | None
        ) = None

        parse_error: (
            dict[str, Any] | None
        ) = None

        valid = False

        try:
            parse_result = (
                self.action_parser.parse(
                    raw_text=raw_text,
                    schema=(
                        self._focal_action_schema
                    ),
                )
            )

            valid = True

        except ActionParseError as exc:
            parse_error = (
                exc.to_dict()
            )

        raw_payload = {
            "focal_decision_id": (
                self._focal_action_schema
                .focal_decision_id
            ),
            "generation_event_id": (
                invocation.generation_event_id
            ),
            "raw_model_output": raw_text,
            "parse_valid": valid,
            "parse_result": (
                parse_result.to_dict()
                if parse_result is not None
                else None
            ),
            "parse_error": deepcopy(
                parse_error
            ),
        }

        normalized_payload = (
            parse_result
            .normalized_action
            .to_dict()
            if parse_result is not None
            else None
        )

        action_event = (
            self._event_factory.create(
                event_type=EventType.ACTION,
                source_component=(
                    "action_parser"
                ),
                raw_payload=(
                    raw_payload
                ),
                normalized_payload=(
                    normalized_payload
                ),
                parent_event_id=(
                    invocation.generation_event_id
                ),
                focal_decision_id=(
                    self._focal_action_schema
                    .focal_decision_id
                ),
                model_invocation_id=(
                    invocation.model_invocation_id
                ),
                metadata={
                    "parse_valid": valid,
                },
            )
        )

        self._append_event(
            action_event
        )

        self._focal_action_event = (
            action_event
        )

        return FocalActionRecord(
            focal_decision_id=(
                self._focal_action_schema
                .focal_decision_id
            ),
            action_event_id=(
                action_event.event_id
            ),
            model_invocation=invocation,
            valid=valid,
            parse_result=parse_result,
            parse_error=parse_error,
        )

    def elicit_self_explanation(
        self,
        *,
        focal_action: FocalActionRecord,
        config: InferenceConfig,
    ) -> SelfExplanationRecord:
        """
        Elicit one structured self-explanation after the focal action has
        been committed.

        The explanation sees the exact decision context previously used
        for the focal action, plus the committed focal-action output and
        the protocol-defined self-explanation request.

        No success/failure feedback, intervention result, or counterfactual
        information is supplied.
        """

        self._require_task()

        if self._focal_action_event is None:
            raise AgentRuntimeError(
                code="focal_action_not_committed",
                message=(
                    "A focal action must be committed before "
                    "self-explanation is elicited."
                ),
            )

        if (
            focal_action.action_event_id
            != self._focal_action_event.event_id
        ):
            raise AgentRuntimeError(
                code="focal_action_record_mismatch",
                message=(
                    "The supplied FocalActionRecord does not match "
                    "the focal action committed by this runtime."
                ),
            )

        if self._self_explanation_event is not None:
            raise AgentRuntimeError(
                code="self_explanation_already_committed",
                message=(
                    "A self-explanation has already been committed "
                    "for this focal action."
                ),
            )

        decision_context_text = (
            focal_action
            .model_invocation
            .serialized_context
            .model_text
        )

        focal_action_text = (
            focal_action
            .model_invocation
            .generation
            .raw_text
        )

        try:
            explanation_prompt = (
                self.explanation_prompt_builder.build(
                    decision_context_text=(
                        decision_context_text
                    ),
                    focal_action_text=(
                        focal_action_text
                    ),
                )
            )

            model_input = (
                self.model_adapter.prepare_input(
                    explanation_prompt
                    .rendered_prompt
                )
            )

            self._validate_direct_model_input_binding(
                expected_model_visible_text=(
                    explanation_prompt
                    .rendered_prompt
                ),
                model_input=model_input,
            )

        except Exception as exc:
            self._record_failure(
                stage=(
                    "self_explanation_input_preparation"
                ),
                error=exc,
                model_invocation_id=None,
            )

            raise AgentRuntimeError(
                code="self_explanation_input_failure",
                message=(
                    "Unable to prepare the self-explanation "
                    "model invocation."
                ),
                details=self._error_details(
                    exc
                ),
            ) from exc

        model_invocation_id = (
            f"mdl_{uuid4().hex}"
        )

        try:
            generation = (
                self.model_adapter.generate(
                    input_record=model_input,
                    config=config,
                )
            )

        except ModelAdapterError as exc:
            self._record_failure(
                stage=(
                    "self_explanation_generation"
                ),
                error=exc,
                model_invocation_id=(
                    model_invocation_id
                ),
            )

            raise AgentRuntimeError(
                code="self_explanation_generation_failure",
                message=(
                    "Self-explanation generation failed."
                ),
                details=self._error_details(
                    exc
                ),
            ) from exc

        generation_event = (
            self._event_factory.create(
                event_type=(
                    EventType.GENERATION
                ),
                source_component=(
                    "self_explanation"
                ),
                raw_payload=(
                    generation.to_dict()
                ),
                normalized_payload=(
                    generation.raw_text
                ),
                parent_event_id=(
                    self._focal_action_event.event_id
                ),
                focal_decision_id=(
                    focal_action
                    .focal_decision_id
                ),
                model_invocation_id=(
                    model_invocation_id
                ),
                metadata={
                    "generation_role": (
                        "self_explanation"
                    ),
                },
            )
        )

        self._append_event(
            generation_event
        )

        parse_result: (
            SelfExplanationParseResult | None
        ) = None

        parse_error: (
            dict[str, Any] | None
        ) = None

        valid = False

        try:
            parse_result = (
                self.explanation_parser.parse(
                    raw_text=(
                        generation.raw_text
                    )
                )
            )

            valid = True

        except SelfExplanationParseError as exc:
            parse_error = (
                exc.to_dict()
            )

        raw_payload = {
            "focal_decision_id": (
                focal_action
                .focal_decision_id
            ),
            "action_event_id": (
                focal_action
                .action_event_id
            ),
            "generation_event_id": (
                generation_event.event_id
            ),
            "focal_action_valid": (
                focal_action.valid
            ),
            "focal_action_raw_output": (
                focal_action_text
            ),
            "self_explanation_prompt": (
                explanation_prompt.to_dict()
            ),
            "raw_model_output": (
                generation.raw_text
            ),
            "parse_valid": valid,
            "parse_result": (
                parse_result.to_dict()
                if parse_result is not None
                else None
            ),
            "parse_error": deepcopy(
                parse_error
            ),
        }

        normalized_payload = (
            parse_result.to_dict()
            if parse_result is not None
            else None
        )

        explanation_event = (
            self._event_factory.create(
                event_type=(
                    EventType.EXPLANATION
                ),
                source_component=(
                    "self_explanation_parser"
                ),
                raw_payload=(
                    raw_payload
                ),
                normalized_payload=(
                    normalized_payload
                ),
                parent_event_id=(
                    generation_event.event_id
                ),
                focal_decision_id=(
                    focal_action
                    .focal_decision_id
                ),
                model_invocation_id=(
                    model_invocation_id
                ),
                metadata={
                    "parse_valid": valid,
                },
            )
        )

        self._append_event(
            explanation_event
        )

        self._self_explanation_event = (
            explanation_event
        )

        return SelfExplanationRecord(
            focal_decision_id=(
                focal_action
                .focal_decision_id
            ),
            action_event_id=(
                focal_action
                .action_event_id
            ),
            generation_event_id=(
                generation_event.event_id
            ),
            explanation_event_id=(
                explanation_event.event_id
            ),
            prompt=(
                explanation_prompt
            ),
            generation=(
                generation
            ),
            valid=valid,
            parse_result=(
                parse_result
            ),
            parse_error=(
                parse_error
            ),
        )

    def _validate_model_input_binding(
        self,
        *,
        serialized: SerializedContext,
        model_input: ModelInputRecord,
    ) -> None:
        if (
            model_input.model_visible_text
            != serialized.model_text
        ):
            raise ModelAdapterError(
                code="model_visible_text_mismatch",
                message=(
                    "Model adapter changed the model-visible context "
                    "during input preparation."
                ),
            )

        if len(
            model_input.messages
        ) != 1:
            raise ModelAdapterError(
                code="unexpected_message_count",
                message=(
                    "Current model adapter must prepare exactly one "
                    "user message."
                ),
            )

        message = (
            model_input.messages[0]
        )

        if (
            message.role != "user"
            or message.content
            != serialized.model_text
        ):
            raise ModelAdapterError(
                code="model_message_binding_mismatch",
                message=(
                    "Prepared chat message does not exactly preserve "
                    "the serialized model-visible context."
                ),
            )

    def _validate_direct_model_input_binding(
        self,
        *,
        expected_model_visible_text: str,
        model_input: ModelInputRecord,
    ) -> None:
        """
        Validate a direct meta-level model invocation that does not pass
        through StructuredContext/ContextSerializer.

        The self-explanation request uses this path because it reproduces
        the already-audited decision context and appends a meta-level
        explanation request rather than constructing a new P/T/H/M/E
        decision context.
        """

        if (
            model_input.model_visible_text
            != expected_model_visible_text
        ):
            raise ModelAdapterError(
                code="model_visible_text_mismatch",
                message=(
                    "Model adapter changed the direct model-visible "
                    "text during input preparation."
                ),
            )

        if len(
            model_input.messages
        ) != 1:
            raise ModelAdapterError(
                code="unexpected_message_count",
                message=(
                    "Direct model invocation must prepare exactly "
                    "one user message."
                ),
            )

        message = (
            model_input.messages[0]
        )

        if (
            message.role != "user"
            or message.content
            != expected_model_visible_text
        ):
            raise ModelAdapterError(
                code="direct_message_binding_mismatch",
                message=(
                    "Prepared direct chat message does not exactly "
                    "preserve the requested model-visible text."
                ),
            )

    def _record_failure(
        self,
        *,
        stage: str,
        error: Exception,
        model_invocation_id: str | None,
    ) -> ExperimentEvent:
        details = self._error_details(
            error
        )

        payload = {
            "stage": stage,
            "error": details,
        }

        event = self._event_factory.create(
            event_type=(
                EventType.TECHNICAL_FAILURE
            ),
            source_component="agent_runtime",
            raw_payload=deepcopy(
                payload
            ),
            normalized_payload=deepcopy(
                payload
            ),
            model_invocation_id=(
                model_invocation_id
            ),
        )

        self._append_event(
            event
        )

        return event

    def _append_event(
        self,
        event: ExperimentEvent,
    ) -> None:
        self.event_writer.append(
            event
        )

        self._events.append(
            event
        )

    def _require_started(self) -> None:
        if not self._started:
            raise AgentRuntimeError(
                code="runtime_not_started",
                message=(
                    "Agent runtime has not been started."
                ),
            )

    def _require_task(self) -> None:
        self._require_started()

        if self._task_event is None:
            raise AgentRuntimeError(
                code="task_not_started",
                message=(
                    "begin_task() or begin_focal_task() must be "
                    "called before task-local runtime operations."
                ),
            )

    def _validate_identity(self) -> None:
        for field_name, value in (
            (
                "experiment_id",
                self.identity.experiment_id,
            ),
            (
                "run_id",
                self.identity.run_id,
            ),
            (
                "task_id",
                self.identity.task_id,
            ),
        ):
            if (
                not isinstance(
                    value,
                    str,
                )
                or not value
            ):
                raise AgentRuntimeError(
                    code="invalid_run_identity",
                    message=(
                        f"{field_name} must be a non-empty string."
                    ),
                )

        if (
            not isinstance(
                self.identity.repetition_id,
                int,
            )
            or self.identity.repetition_id < 0
        ):
            raise AgentRuntimeError(
                code="invalid_run_identity",
                message=(
                    "repetition_id must be a non-negative integer."
                ),
            )

    @staticmethod
    def _error_details(
        error: Exception,
    ) -> dict[str, Any]:
        if hasattr(
            error,
            "to_dict",
        ):
            try:
                value = error.to_dict()

                if isinstance(
                    value,
                    dict,
                ):
                    return deepcopy(
                        value
                    )

            except Exception:
                pass

        return {
            "exception_type": (
                type(error).__name__
            ),
            "message": str(
                error
            ),
        }

    @staticmethod
    def _validate_json(
        value: Any,
        *,
        field_name: str,
    ) -> None:
        try:
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise AgentRuntimeError(
                code="non_serializable_runtime_payload",
                message=(
                    f"{field_name!r} must be losslessly "
                    "representable as JSON."
                ),
            ) from exc