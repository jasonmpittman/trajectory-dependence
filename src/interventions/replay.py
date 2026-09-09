__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ..agent.context import (
    TaskEnvironmentPolicy,
)
from ..agent.runtime import (
    AgentRuntime,
    FocalActionRecord,
    RunIdentity,
)
from ..logging import (
    ContextElement,
    EventType,
    ExperimentEvent,
    RawEventWriter,
    SourceClass,
    TaskClassification,
)
from ..model import (
    InferenceConfig,
    ModelAdapter,
)
from ..normalization import (
    FocalActionSchema,
)
from ..snapshots import (
    PreFocalSnapshot,
)
from .schemas import (
    CounterfactualState,
    InterventionApplication,
    InterventionKind,
    InterventionOperation,
)


class ReplayExecutionError(RuntimeError):
    """Defined failure constructing or executing a matched replay."""

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
class ReplayContext:
    """
    Exact P/T/H/M/E projection supplied to one counterfactual replay.

    These elements describe model-visible state. Before inference, T/H/M/E
    are re-materialized as replay-local events so the existing provenance
    auditor can validate them.
    """

    prompt_element: ContextElement

    tool_elements: tuple[
        ContextElement,
        ...
    ]

    history_elements: tuple[
        ContextElement,
        ...
    ]

    memory_elements: tuple[
        ContextElement,
        ...
    ]

    environment_elements: tuple[
        ContextElement,
        ...
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "P": (
                self.prompt_element.to_dict()
            ),
            "T": [
                element.to_dict()
                for element
                in self.tool_elements
            ],
            "H": [
                element.to_dict()
                for element
                in self.history_elements
            ],
            "M": [
                element.to_dict()
                for element
                in self.memory_elements
            ],
            "E": [
                element.to_dict()
                for element
                in self.environment_elements
            ],
        }


@dataclass(frozen=True)
class ReplayRecord:
    """One completed targeted or sham focal-action replay."""

    replay_run_id: str

    intervention_event_id: str

    application: InterventionApplication

    projected_context: ReplayContext

    focal_action: FocalActionRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_run_id": (
                self.replay_run_id
            ),
            "intervention_event_id": (
                self.intervention_event_id
            ),
            "application": (
                self.application.to_dict()
            ),
            "projected_context": (
                self.projected_context.to_dict()
            ),
            "focal_action": (
                self.focal_action.to_dict()
            ),
        }


@dataclass(frozen=True)
class MatchedReplayExecution:
    """
    Complete targeted/sham replay result associated with one baseline
    focal action.
    """

    baseline_snapshot_id: str

    baseline_action: FocalActionRecord

    targeted: ReplayRecord
    sham: ReplayRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_snapshot_id": (
                self.baseline_snapshot_id
            ),
            "baseline_action": (
                self.baseline_action.to_dict()
            ),
            "targeted": (
                self.targeted.to_dict()
            ),
            "sham": (
                self.sham.to_dict()
            ),
        }


class ReplayContextProjector:
    """
    Project validated counterfactual state into model-visible P/T/H/M/E.

    P/T/H are explicitly represented in PreFocalSnapshot.

    The snapshot separately preserves complete persistent memory and
    environment state. The exact M/E elements that were model-visible at
    the focal decision are therefore taken from the baseline focal
    StructuredContext and validated against the snapshot before any
    counterfactual manipulation is projected.
    """

    def project(
        self,
        *,
        snapshot: PreFocalSnapshot,
        application: InterventionApplication,
        baseline_action: FocalActionRecord,
    ) -> ReplayContext:
        if (
            application.baseline_snapshot_id
            != snapshot.snapshot_id
        ):
            raise ReplayExecutionError(
                code="replay_snapshot_mismatch",
                message=(
                    "Intervention application does not belong "
                    "to the supplied baseline snapshot."
                ),
            )

        baseline_state = (
            CounterfactualState.from_snapshot(
                snapshot
            )
        )

        self._validate_baseline_context(
            state=baseline_state,
            baseline_action=baseline_action,
        )

        state = (
            application.counterfactual_state
        )

        if len(
            state.prompt_elements
        ) != 1:
            raise ReplayExecutionError(
                code="invalid_replay_prompt_count",
                message=(
                    "Matched replay requires exactly one "
                    "canonical P element."
                ),
            )

        baseline_elements = (
            baseline_action
            .model_invocation
            .structured_context
            .elements
        )

        baseline_memory = tuple(
            deepcopy(
                element
            )
            for element
            in baseline_elements
            if (
                element.source
                == SourceClass.M
            )
        )

        baseline_environment = tuple(
            deepcopy(
                element
            )
            for element
            in baseline_elements
            if (
                element.source
                == SourceClass.E
            )
        )

        memory_elements = (
            self._project_memory_elements(
                baseline_memory,
                application,
            )
        )

        environment_elements = (
            self._project_environment_elements(
                baseline_environment,
                application,
            )
        )

        return ReplayContext(
            prompt_element=deepcopy(
                state.prompt_elements[0]
            ),
            tool_elements=tuple(
                deepcopy(
                    state.tool_elements
                )
            ),
            history_elements=tuple(
                deepcopy(
                    state.history_elements
                )
            ),
            memory_elements=(
                memory_elements
            ),
            environment_elements=(
                environment_elements
            ),
        )

    def _validate_baseline_context(
        self,
        *,
        state: CounterfactualState,
        baseline_action: FocalActionRecord,
    ) -> None:
        context = (
            baseline_action
            .model_invocation
            .structured_context
        )

        if (
            context.task_id
            != state.task_id
        ):
            raise ReplayExecutionError(
                code="baseline_task_mismatch",
                message=(
                    "Baseline focal action and snapshot "
                    "refer to different tasks."
                ),
            )

        if (
            context.condition
            != state.condition
        ):
            raise ReplayExecutionError(
                code="baseline_condition_mismatch",
                message=(
                    "Baseline focal action and snapshot "
                    "refer to different conditions."
                ),
            )

        for source, expected in (
            (
                SourceClass.P,
                state.prompt_elements,
            ),
            (
                SourceClass.T,
                state.tool_elements,
            ),
            (
                SourceClass.H,
                state.history_elements,
            ),
        ):
            observed = tuple(
                element
                for element
                in context.elements
                if (
                    element.source
                    == source
                )
            )

            if [
                item.to_dict()
                for item
                in observed
            ] != [
                item.to_dict()
                for item
                in expected
            ]:
                raise ReplayExecutionError(
                    code="baseline_context_mismatch",
                    message=(
                        "Snapshot and baseline focal context "
                        f"do not agree for source {source.value}."
                    ),
                )

        self._validate_baseline_memory(
            state=state,
            context_elements=(
                context.elements
            ),
        )

        self._validate_baseline_environment(
            state=state,
            context_elements=(
                context.elements
            ),
        )

    def _validate_baseline_memory(
        self,
        *,
        state: CounterfactualState,
        context_elements,
    ) -> None:
        records = {
            record.memory_id: record
            for record
            in state.memory_records
        }

        for element in context_elements:
            if (
                element.source
                != SourceClass.M
            ):
                continue

            memory_id = self._memory_id(
                element
            )

            if memory_id not in records:
                raise ReplayExecutionError(
                    code=(
                        "baseline_memory_projection_mismatch"
                    ),
                    message=(
                        f"Model-visible memory {memory_id!r} "
                        "does not exist in the baseline "
                        "memory snapshot."
                    ),
                )

            record = records[
                memory_id
            ]

            expected = {
                "key": record.key,
                "value": deepcopy(
                    record.value
                ),
            }

            if element.content != expected:
                raise ReplayExecutionError(
                    code=(
                        "baseline_memory_projection_mismatch"
                    ),
                    message=(
                        f"Model-visible memory {memory_id!r} "
                        "does not match the baseline "
                        "memory snapshot."
                    ),
                )

    def _validate_baseline_environment(
        self,
        *,
        state: CounterfactualState,
        context_elements,
    ) -> None:
        for element in context_elements:
            if (
                element.source
                != SourceClass.E
            ):
                continue

            content = element.content

            if (
                not isinstance(
                    content,
                    dict,
                )
                or "path"
                not in content
                or "value"
                not in content
            ):
                raise ReplayExecutionError(
                    code="invalid_environment_projection",
                    message=(
                        "Model-visible E element does not "
                        "contain the expected path/value "
                        "structure."
                    ),
                )

            path = tuple(
                content[
                    "path"
                ]
            )

            observed = (
                self._read_environment_path(
                    state.environment_state,
                    path,
                )
            )

            if observed != content[
                "value"
            ]:
                raise ReplayExecutionError(
                    code=(
                        "baseline_environment_projection_mismatch"
                    ),
                    message=(
                        "Model-visible E value does not "
                        "match the baseline environment "
                        "snapshot."
                    ),
                )

    def _project_memory_elements(
        self,
        elements: tuple[
            ContextElement,
            ...
        ],
        application: InterventionApplication,
    ) -> tuple[
        ContextElement,
        ...
    ]:
        spec = application.spec

        if (
            spec.source
            != SourceClass.M
        ):
            return tuple(
                deepcopy(
                    elements
                )
            )

        target_source_id = (
            f"memory:{spec.target_id}"
        )

        matches = [
            index
            for index, element
            in enumerate(
                elements
            )
            if (
                element.source_id
                == target_source_id
            )
        ]

        if len(matches) != 1:
            raise ReplayExecutionError(
                code="intervention_not_model_visible",
                message=(
                    "M intervention target is not represented "
                    "exactly once in the baseline focal context."
                ),
            )

        index = matches[
            0
        ]

        updated = list(
            deepcopy(
                elements
            )
        )

        if (
            spec.operation
            == InterventionOperation.REMOVE
        ):
            del updated[
                index
            ]

            return tuple(
                updated
            )

        element = updated[
            index
        ]

        if (
            not isinstance(
                element.content,
                dict,
            )
            or "key"
            not in element.content
            or "value"
            not in element.content
        ):
            raise ReplayExecutionError(
                code="invalid_memory_projection",
                message=(
                    "Model-visible M element does not contain "
                    "the expected key/value structure."
                ),
            )

        content = deepcopy(
            element.content
        )

        content[
            "value"
        ] = deepcopy(
            spec.replacement_value
        )

        updated[
            index
        ] = self._replace_content(
            element,
            content,
        )

        return tuple(
            updated
        )

    def _project_environment_elements(
        self,
        elements: tuple[
            ContextElement,
            ...
        ],
        application: InterventionApplication,
    ) -> tuple[
        ContextElement,
        ...
    ]:
        spec = application.spec

        if (
            spec.source
            != SourceClass.E
        ):
            return tuple(
                deepcopy(
                    elements
                )
            )

        target_path = tuple(
            spec.target_path
            or ()
        )

        matches: list[int] = []

        for index, element in enumerate(
            elements
        ):
            content = element.content

            if (
                isinstance(
                    content,
                    dict,
                )
                and tuple(
                    content.get(
                        "path",
                        (),
                    )
                )
                == target_path
            ):
                matches.append(
                    index
                )

        if len(matches) != 1:
            raise ReplayExecutionError(
                code="intervention_not_model_visible",
                message=(
                    "E intervention target is not represented "
                    "exactly once in the baseline focal context."
                ),
            )

        index = matches[
            0
        ]

        updated = list(
            deepcopy(
                elements
            )
        )

        element = updated[
            index
        ]

        content = deepcopy(
            element.content
        )

        content[
            "value"
        ] = deepcopy(
            spec.replacement_value
        )

        updated[
            index
        ] = self._replace_content(
            element,
            content,
        )

        return tuple(
            updated
        )

    @staticmethod
    def _replace_content(
        element: ContextElement,
        content: Any,
    ) -> ContextElement:
        return ContextElement(
            source=element.source,
            source_id=(
                element.source_id
            ),
            content=deepcopy(
                content
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

    @staticmethod
    def _memory_id(
        element: ContextElement,
    ) -> str:
        prefix = "memory:"

        if not element.source_id.startswith(
            prefix
        ):
            raise ReplayExecutionError(
                code="invalid_memory_source_id",
                message=(
                    "Model-visible M source_id does not "
                    "use the required memory:<id> format."
                ),
            )

        memory_id = (
            element.source_id[
                len(
                    prefix
                ):
            ]
        )

        if not memory_id:
            raise ReplayExecutionError(
                code="invalid_memory_source_id",
                message=(
                    "Model-visible M source_id contains "
                    "an empty memory ID."
                ),
            )

        return memory_id

    @staticmethod
    def _read_environment_path(
        state: dict[str, Any] | None,
        path: tuple[str, ...],
    ) -> Any:
        if state is None:
            raise ReplayExecutionError(
                code="environment_state_unavailable",
                message=(
                    "Baseline focal context contains E but "
                    "the snapshot has no environment state."
                ),
            )

        current: Any = (
            state
        )

        for key in path:
            if (
                not isinstance(
                    current,
                    dict,
                )
                or key
                not in current
            ):
                raise ReplayExecutionError(
                    code="environment_path_not_found",
                    message=(
                        f"Environment path {list(path)!r} "
                        "does not exist in the snapshot."
                    ),
                )

            current = current[
                key
            ]

        return deepcopy(
            current
        )


class ReplayContextMaterializer:
    """
    Create replay-local provenance events for counterfactual T/H/M/E.

    Baseline origin information is preserved as metadata. The event type
    used here describes the replay projection mechanism; it does not claim
    that the underlying external operation was physically repeated.
    """

    def materialize(
        self,
        *,
        runtime: AgentRuntime,
        context: ReplayContext,
    ) -> ReplayContext:
        tool_elements = tuple(
            self._materialize_t(
                runtime,
                element,
            )
            for element
            in context.tool_elements
        )

        history_elements = tuple(
            self._materialize_h(
                runtime,
                element,
            )
            for element
            in context.history_elements
        )

        memory_elements = tuple(
            self._materialize_m(
                runtime,
                element,
            )
            for element
            in context.memory_elements
        )

        environment_elements = tuple(
            self._materialize_e(
                runtime,
                element,
            )
            for element
            in context.environment_elements
        )

        prompt = (
            runtime.prompt_element
        )

        if prompt is None:
            raise ReplayExecutionError(
                code="replay_prompt_unavailable",
                message=(
                    "Replay runtime did not create "
                    "canonical P."
                ),
            )

        return ReplayContext(
            prompt_element=prompt,
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
        )

    def _materialize_t(
        self,
        runtime: AgentRuntime,
        baseline: ContextElement,
    ) -> ContextElement:
        event = self._projection_event(
            runtime=runtime,
            event_type=(
                EventType.TOOL_RETURN
            ),
            baseline=baseline,
        )

        return ContextElement(
            source=SourceClass.T,
            source_id=(
                f"tool_return:"
                f"{event.event_id}"
            ),
            content=deepcopy(
                baseline.content
            ),
            origin_event_id=(
                event.event_id
            ),
            origin_source=(
                SourceClass.T
            ),
            created_sequence=(
                event.sequence_index
            ),
            metadata=(
                self._projection_metadata(
                    baseline
                )
            ),
        )

    def _materialize_h(
        self,
        runtime: AgentRuntime,
        baseline: ContextElement,
    ) -> ContextElement:
        if (
            baseline.origin_source
            == SourceClass.T
        ):
            event_type = (
                EventType.TOOL_RETURN
            )

        elif (
            baseline.origin_source
            == SourceClass.E
        ):
            event_type = (
                EventType.ENVIRONMENT_QUERY
            )

        else:
            event_type = (
                EventType.ACTION
            )

        event = self._projection_event(
            runtime=runtime,
            event_type=event_type,
            baseline=baseline,
        )

        return ContextElement(
            source=SourceClass.H,
            source_id=(
                f"history:"
                f"{event.event_id}"
            ),
            content=deepcopy(
                baseline.content
            ),
            origin_event_id=(
                event.event_id
            ),
            origin_source=(
                baseline.origin_source
            ),
            created_sequence=(
                event.sequence_index
            ),
            metadata=(
                self._projection_metadata(
                    baseline
                )
            ),
        )

    def _materialize_m(
        self,
        runtime: AgentRuntime,
        baseline: ContextElement,
    ) -> ContextElement:
        memory_id = (
            ReplayContextProjector
            ._memory_id(
                baseline
            )
        )

        event = self._projection_event(
            runtime=runtime,
            event_type=(
                EventType.MEMORY_READ
            ),
            baseline=baseline,
            memory_id=memory_id,
        )

        return ContextElement(
            source=SourceClass.M,
            source_id=(
                f"memory:{memory_id}"
            ),
            content=deepcopy(
                baseline.content
            ),
            origin_event_id=(
                event.event_id
            ),
            origin_source=(
                SourceClass.M
            ),
            created_sequence=(
                event.sequence_index
            ),
            metadata={
                **self._projection_metadata(
                    baseline
                ),
                "memory_id": (
                    memory_id
                ),
            },
        )

    def _materialize_e(
        self,
        runtime: AgentRuntime,
        baseline: ContextElement,
    ) -> ContextElement:
        event = self._projection_event(
            runtime=runtime,
            event_type=(
                EventType.ENVIRONMENT_QUERY
            ),
            baseline=baseline,
        )

        return ContextElement(
            source=SourceClass.E,
            source_id=(
                f"environment:replay:"
                f"{event.event_id}"
            ),
            content=deepcopy(
                baseline.content
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
            metadata=(
                self._projection_metadata(
                    baseline
                )
            ),
        )

    def _projection_event(
        self,
        *,
        runtime: AgentRuntime,
        event_type: EventType,
        baseline: ContextElement,
        memory_id: str | None = None,
    ) -> ExperimentEvent:
        return runtime.record_component_event(
            event_type=event_type,
            source_component=(
                "matched_replay_projection"
            ),
            raw_payload={
                "replay_projection": True,
                "baseline_source_id": (
                    baseline.source_id
                ),
                "baseline_origin_event_id": (
                    baseline.origin_event_id
                ),
                "baseline_origin_source": (
                    baseline.origin_source.value
                    if (
                        baseline.origin_source
                        is not None
                    )
                    else None
                ),
                "content": deepcopy(
                    baseline.content
                ),
            },
            normalized_payload=deepcopy(
                baseline.content
            ),
            memory_id=memory_id,
            metadata=(
                self._projection_metadata(
                    baseline
                )
            ),
        )

    @staticmethod
    def _projection_metadata(
        baseline: ContextElement,
    ) -> dict[str, Any]:
        return {
            "replay_projection": True,
            "baseline_source_id": (
                baseline.source_id
            ),
            "baseline_origin_event_id": (
                baseline.origin_event_id
            ),
            "baseline_origin_source": (
                baseline.origin_source.value
                if (
                    baseline.origin_source
                    is not None
                )
                else None
            ),
        }


class MatchedReplayExecutor:
    """
    Execute targeted and sham focal replays under the exact baseline
    inference configuration.
    """

    def __init__(
        self,
        *,
        projector: ReplayContextProjector
        | None = None,
        materializer: ReplayContextMaterializer
        | None = None,
    ) -> None:
        self.projector = (
            projector
            or ReplayContextProjector()
        )

        self.materializer = (
            materializer
            or ReplayContextMaterializer()
        )

    def execute_pair(
        self,
        *,
        snapshot: PreFocalSnapshot,
        baseline_action: FocalActionRecord,
        targeted_application: InterventionApplication,
        sham_application: InterventionApplication,
        action_schema: FocalActionSchema,
        task_classification: TaskClassification,
        model_adapter: ModelAdapter,
        config: InferenceConfig,
        targeted_event_writer: RawEventWriter,
        sham_event_writer: RawEventWriter,
    ) -> MatchedReplayExecution:
        if (
            targeted_application.spec.kind
            != InterventionKind.TARGETED
        ):
            raise ReplayExecutionError(
                code="invalid_targeted_application",
                message=(
                    "targeted_application does not have "
                    "kind=targeted."
                ),
            )

        if (
            sham_application.spec.kind
            != InterventionKind.SHAM
        ):
            raise ReplayExecutionError(
                code="invalid_sham_application",
                message=(
                    "sham_application does not have "
                    "kind=sham."
                ),
            )

        if (
            targeted_application
            .baseline_snapshot_id
            != snapshot.snapshot_id
            or sham_application
            .baseline_snapshot_id
            != snapshot.snapshot_id
        ):
            raise ReplayExecutionError(
                code="pair_snapshot_mismatch",
                message=(
                    "Both replay applications must derive "
                    "from the supplied baseline snapshot."
                ),
            )

        baseline_config = (
            baseline_action
            .model_invocation
            .generation
            .inference_config
        )

        if config != baseline_config:
            raise ReplayExecutionError(
                code="inference_config_mismatch",
                message=(
                    "Matched replay inference configuration "
                    "must exactly equal the baseline focal-action "
                    "configuration."
                ),
            )

        targeted = self._execute_one(
            snapshot=snapshot,
            baseline_action=(
                baseline_action
            ),
            application=(
                targeted_application
            ),
            action_schema=action_schema,
            task_classification=(
                task_classification
            ),
            model_adapter=(
                model_adapter
            ),
            config=config,
            event_writer=(
                targeted_event_writer
            ),
        )

        sham = self._execute_one(
            snapshot=snapshot,
            baseline_action=(
                baseline_action
            ),
            application=(
                sham_application
            ),
            action_schema=action_schema,
            task_classification=(
                task_classification
            ),
            model_adapter=(
                model_adapter
            ),
            config=config,
            event_writer=(
                sham_event_writer
            ),
        )

        return MatchedReplayExecution(
            baseline_snapshot_id=(
                snapshot.snapshot_id
            ),
            baseline_action=(
                baseline_action
            ),
            targeted=targeted,
            sham=sham,
        )

    def _execute_one(
        self,
        *,
        snapshot: PreFocalSnapshot,
        baseline_action: FocalActionRecord,
        application: InterventionApplication,
        action_schema: FocalActionSchema,
        task_classification: TaskClassification,
        model_adapter: ModelAdapter,
        config: InferenceConfig,
        event_writer: RawEventWriter,
    ) -> ReplayRecord:
        projected = (
            self.projector.project(
                snapshot=snapshot,
                application=application,
                baseline_action=(
                    baseline_action
                ),
            )
        )

        state = (
            application
            .counterfactual_state
        )

        replay_run_id = (
            f"{state.run_id}"
            f"__replay__"
            f"{application.spec.kind.value}"
            f"__{application.spec.source.value}"
            f"__{application.spec.intervention_id}"
        )

        runtime = AgentRuntime(
            identity=RunIdentity(
                experiment_id=(
                    state.experiment_id
                ),
                run_id=replay_run_id,
                task_id=state.task_id,
                task_classification=(
                    task_classification
                ),
                condition=(
                    state.condition
                ),
                repetition_id=(
                    state.repetition_id
                ),
                seed=config.seed,
            ),
            model_adapter=model_adapter,
            event_writer=event_writer,
        )

        runtime.start()

        runtime.begin_replay_focal_task(
            prompt_content=(
                projected
                .prompt_element
                .content
            ),
            action_schema=(
                action_schema
            ),
            replay_metadata={
                "baseline_snapshot_id": (
                    snapshot.snapshot_id
                ),
                "baseline_run_id": (
                    state.run_id
                ),
                "replay_run_id": (
                    replay_run_id
                ),
                "intervention_id": (
                    application
                    .spec
                    .intervention_id
                ),
                "pair_id": (
                    application
                    .spec
                    .pair_id
                ),
                "kind": (
                    application
                    .spec
                    .kind
                    .value
                ),
                "source": (
                    application
                    .spec
                    .source
                    .value
                ),
                "baseline_state_checksum": (
                    application
                    .baseline_state_checksum
                ),
                "counterfactual_state_checksum": (
                    application
                    .counterfactual_state_checksum
                ),
            },
        )

        if (
            application.spec.kind
            == InterventionKind.TARGETED
        ):
            intervention_event_type = (
                EventType.INTERVENTION
            )

        else:
            intervention_event_type = (
                EventType.SHAM_INTERVENTION
            )

        intervention_event = (
            runtime.record_component_event(
                event_type=(
                    intervention_event_type
                ),
                source_component=(
                    "intervention_engine"
                ),
                raw_payload=(
                    application.to_dict()
                ),
                normalized_payload={
                    "intervention_id": (
                        application
                        .spec
                        .intervention_id
                    ),
                    "pair_id": (
                        application
                        .spec
                        .pair_id
                    ),
                    "kind": (
                        application
                        .spec
                        .kind
                        .value
                    ),
                    "source": (
                        application
                        .spec
                        .source
                        .value
                    ),
                    "changed_sources": [
                        source.value
                        for source
                        in (
                            application
                            .changed_sources
                        )
                    ],
                    "diff": (
                        application
                        .diff
                        .to_dict()
                    ),
                },
            )
        )

        materialized = (
            self.materializer.materialize(
                runtime=runtime,
                context=projected,
            )
        )

        environment_policy = (
            self._environment_policy(
                materialized
                .environment_elements
            )
        )

        action = (
            runtime.invoke_focal_action(
                config=config,
                tool_elements=(
                    materialized
                    .tool_elements
                ),
                history_elements=(
                    materialized
                    .history_elements
                ),
                memory_elements=(
                    materialized
                    .memory_elements
                ),
                environment_elements=(
                    materialized
                    .environment_elements
                ),
                environment_policy=(
                    environment_policy
                ),
            )
        )

        return ReplayRecord(
            replay_run_id=replay_run_id,
            intervention_event_id=(
                intervention_event
                .event_id
            ),
            application=application,
            projected_context=(
                materialized
            ),
            focal_action=action,
        )

    @staticmethod
    def _environment_policy(
        elements: tuple[
            ContextElement,
            ...
        ],
    ) -> TaskEnvironmentPolicy | None:
        if not elements:
            return None

        return TaskEnvironmentPolicy(
            allow_environment=True,
            allowed_source_ids=frozenset(
                element.source_id
                for element
                in elements
            ),
        )