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
from pathlib import Path
from typing import Any, Callable, Mapping

from ..agent import (
    AgentRuntime,
    ScaffoldController,
)
from ..environment import (
    EnvironmentState,
)
from ..interventions import (
    InterventionKind,
    InterventionPair,
    InterventionSpec,
)
from ..logging import (
    Condition,
    ContextElement,
    EventType,
    SourceClass,
)
from ..memory import (
    SQLiteMemoryStore,
)
from ..normalization import (
    FocalActionPromptBuilder,
)
from ..tools import (
    KeyValueLookupTool,
    Tool,
)
from .spec import (
    ExperimentalTaskSpec,
    InterventionMutationTemplate,
    InterventionPairTemplate,
)


class TaskResolutionError(RuntimeError):
    """Defined failure resolving a task specification into runtime state."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
    ) -> None:
        super().__init__(
            message
        )

        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class TaskBackingState:
    """
    Mutable backing stores associated with one experimental run.

    These stores are separate from model-visible context. Availability of
    M/E to the model remains controlled by the task condition.
    """

    memory_store: (
        SQLiteMemoryStore | None
    )

    memory_namespace: (
        str | None
    )

    environment: (
        EnvironmentState | None
    )


@dataclass(frozen=True)
class ResolvedFixtureTarget:
    """
    Runtime target corresponding to one stable task fixture ID.

    target_id is used for P/T/H/M intervention resolution.
    target_path is used for E.
    """

    fixture_id: str

    source: SourceClass

    target_id: str | None

    target_path: (
        tuple[str, ...] | None
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": (
                self.fixture_id
            ),
            "source": (
                self.source.value
            ),
            "target_id": (
                self.target_id
            ),
            "target_path": (
                list(
                    self.target_path
                )
                if (
                    self.target_path
                    is not None
                )
                else None
            ),
        }


@dataclass(frozen=True)
class ResolvedTaskContext:
    """
    Exact model-visible scaffold elements resolved for one condition.

    fixture_targets maps stable preregistered fixture IDs onto the runtime
    identifiers needed by InterventionSpec.
    """

    condition: Condition

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

    fixture_targets: dict[
        str,
        ResolvedFixtureTarget,
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": (
                self.condition.value
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
            "fixture_targets": {
                fixture_id: (
                    target.to_dict()
                )
                for (
                    fixture_id,
                    target
                )
                in (
                    self.fixture_targets
                    .items()
                )
            },
        }


class TaskBackingStateBuilder:
    """
    Construct deterministic M and E backing state for one task/run.

    Backing state may exist even when a condition does not expose it to the
    model. Model visibility is controlled separately by available_sources.
    """

    def build(
        self,
        *,
        task: ExperimentalTaskSpec,
        run_id: str,
        work_directory: str | Path,
    ) -> TaskBackingState:
        if (
            not isinstance(
                run_id,
                str,
            )
            or not run_id
        ):
            raise TaskResolutionError(
                code="invalid_run_id",
                message=(
                    "run_id must be a "
                    "non-empty string."
                ),
            )

        directory = Path(
            work_directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        memory_store: (
            SQLiteMemoryStore | None
        ) = None

        memory_namespace: (
            str | None
        ) = None

        if task.memory_fixtures:
            memory_namespace = (
                run_id
            )

            memory_store = (
                SQLiteMemoryStore(
                    directory
                    / "memory.sqlite3"
                )
            )

            memory_store.initialize()

            for fixture in (
                task.memory_fixtures
            ):
                memory_store.write(
                    namespace=(
                        memory_namespace
                    ),
                    memory_id=(
                        fixture.fixture_id
                    ),
                    key=(
                        fixture.key
                    ),
                    value=deepcopy(
                        fixture.value
                    ),
                )

        environment: (
            EnvironmentState | None
        ) = None

        if (
            task.environment
            is not None
        ):
            environment = (
                EnvironmentState(
                    deepcopy(
                        task.environment
                        .initial_state
                    )
                )
            )

        return TaskBackingState(
            memory_store=(
                memory_store
            ),
            memory_namespace=(
                memory_namespace
            ),
            environment=(
                environment
            ),
        )


class TaskFixtureResolver:
    """
    Materialize task fixtures into event-backed runtime context.

    Only sources declared model-visible under the current condition are
    materialized as P/T/H/M/E context.
    """

    def __init__(
        self,
        *,
        tool_registry: Mapping[
            str,
            Callable[
                [],
                Tool,
            ],
        ]
        | None = None,
    ) -> None:
        self.tool_registry = dict(
            tool_registry
            or {
                "key_value": (
                    KeyValueLookupTool
                ),
            }
        )

    def resolve(
        self,
        *,
        task: ExperimentalTaskSpec,
        condition: Condition,
        runtime: AgentRuntime,
        backing_state: TaskBackingState,
    ) -> ResolvedTaskContext:
        if (
            condition
            not in task.conditions
        ):
            raise TaskResolutionError(
                code="condition_not_in_task",
                message=(
                    f"Condition {condition.value} "
                    "is not part of task "
                    f"{task.task_id!r}."
                ),
            )

        if (
            runtime.identity.condition
            != condition
        ):
            raise TaskResolutionError(
                code="runtime_condition_mismatch",
                message=(
                    "Runtime condition does not match "
                    "the requested task condition."
                ),
            )

        if (
            runtime.identity.task_id
            != task.task_id
        ):
            raise TaskResolutionError(
                code="runtime_task_mismatch",
                message=(
                    "Runtime task_id does not match "
                    "the task specification."
                ),
            )

        if not runtime.task_has_begun:
            raise TaskResolutionError(
                code="task_not_started",
                message=(
                    "AgentRuntime.begin_focal_task() "
                    "must be called before fixture resolution."
                ),
            )

        available = set(
            task.available_sources[
                condition
            ]
        )

        scaffold = ScaffoldController(
            runtime=runtime,
            memory_store=(
                backing_state
                .memory_store
            ),
            memory_namespace=(
                backing_state
                .memory_namespace
            ),
            environment=(
                backing_state
                .environment
            ),
        )

        fixture_targets: dict[
            str,
            ResolvedFixtureTarget,
        ] = {}

        prompt = (
            runtime.prompt_element
        )

        if prompt is None:
            raise TaskResolutionError(
                code="prompt_unavailable",
                message=(
                    "Runtime did not expose a "
                    "canonical P element."
                ),
            )

        fixture_targets[
            "prompt"
        ] = ResolvedFixtureTarget(
            fixture_id="prompt",
            source=SourceClass.P,
            target_id=(
                prompt.source_id
            ),
            target_path=None,
        )

        tool_elements: list[
            ContextElement
        ] = []

        history_elements: list[
            ContextElement
        ] = []

        memory_elements: list[
            ContextElement
        ] = []

        environment_elements: list[
            ContextElement
        ] = []

        if (
            SourceClass.T
            in available
        ):
            for fixture in (
                task.tool_fixtures
            ):
                tool_factory = (
                    self.tool_registry.get(
                        fixture.tool_name
                    )
                )

                if (
                    tool_factory
                    is None
                ):
                    raise TaskResolutionError(
                        code="unknown_tool_fixture",
                        message=(
                            f"No deterministic tool is "
                            f"registered as "
                            f"{fixture.tool_name!r}."
                        ),
                    )

                interaction = (
                    scaffold.execute_tool(
                        tool=(
                            tool_factory()
                        ),
                        request=deepcopy(
                            fixture.request
                        ),
                        state=deepcopy(
                            fixture.state
                        ),
                    )
                )

                tool_elements.append(
                    interaction
                    .tool_element
                )

                fixture_targets[
                    fixture.fixture_id
                ] = (
                    ResolvedFixtureTarget(
                        fixture_id=(
                            fixture
                            .fixture_id
                        ),
                        source=(
                            SourceClass.T
                        ),
                        target_id=(
                            interaction
                            .tool_element
                            .source_id
                        ),
                        target_path=None,
                    )
                )

        if (
            SourceClass.H
            in available
        ):
            for fixture in (
                task.history_fixtures
            ):
                event_type = (
                    self
                    ._history_origin_event_type(
                        fixture
                        .origin_source
                    )
                )

                origin_event = (
                    runtime
                    .record_component_event(
                        event_type=(
                            event_type
                        ),
                        source_component=(
                            "task_history_fixture"
                        ),
                        raw_payload={
                            "fixture_id": (
                                fixture
                                .fixture_id
                            ),
                            "content": deepcopy(
                                fixture.content
                            ),
                            "metadata": deepcopy(
                                fixture.metadata
                            ),
                        },
                        normalized_payload=(
                            deepcopy(
                                fixture
                                .content
                            )
                        ),
                    )
                )

                history_element = (
                    scaffold.retain_history(
                        event_id=(
                            origin_event
                            .event_id
                        )
                    )
                )

                history_elements.append(
                    history_element
                )

                fixture_targets[
                    fixture.fixture_id
                ] = (
                    ResolvedFixtureTarget(
                        fixture_id=(
                            fixture
                            .fixture_id
                        ),
                        source=(
                            SourceClass.H
                        ),
                        target_id=(
                            history_element
                            .source_id
                        ),
                        target_path=None,
                    )
                )

        if (
            SourceClass.M
            in available
        ):
            if (
                backing_state
                .memory_store
                is None
                or backing_state
                .memory_namespace
                is None
            ):
                raise TaskResolutionError(
                    code="memory_backing_unavailable",
                    message=(
                        "Task condition exposes M but "
                        "persistent-memory backing state "
                        "was not constructed."
                    ),
                )

            for fixture in (
                task.memory_fixtures
            ):
                observation = (
                    scaffold.read_memory(
                        memory_id=(
                            fixture
                            .fixture_id
                        )
                    )
                )

                if (
                    not observation.found
                    or observation
                    .memory_element
                    is None
                ):
                    raise TaskResolutionError(
                        code="memory_fixture_missing",
                        message=(
                            f"Memory fixture "
                            f"{fixture.fixture_id!r} "
                            "could not be read."
                        ),
                    )

                memory_elements.append(
                    observation
                    .memory_element
                )

                # InterventionEngine targets M by memory_id,
                # not by model-visible source_id.
                fixture_targets[
                    fixture.fixture_id
                ] = (
                    ResolvedFixtureTarget(
                        fixture_id=(
                            fixture
                            .fixture_id
                        ),
                        source=(
                            SourceClass.M
                        ),
                        target_id=(
                            fixture
                            .fixture_id
                        ),
                        target_path=None,
                    )
                )

        if (
            SourceClass.E
            in available
        ):
            if (
                task.environment
                is None
                or backing_state
                .environment
                is None
            ):
                raise TaskResolutionError(
                    code="environment_backing_unavailable",
                    message=(
                        "Task condition exposes E but "
                        "environment backing state "
                        "was not constructed."
                    ),
                )

            for fixture in (
                task.environment
                .observations
            ):
                observation = (
                    scaffold.query_environment(
                        path=(
                            fixture.path
                        )
                    )
                )

                environment_elements.append(
                    observation
                    .environment_element
                )

                fixture_targets[
                    fixture.fixture_id
                ] = (
                    ResolvedFixtureTarget(
                        fixture_id=(
                            fixture
                            .fixture_id
                        ),
                        source=(
                            SourceClass.E
                        ),
                        target_id=None,
                        target_path=(
                            fixture.path
                        ),
                    )
                )

        return ResolvedTaskContext(
            condition=condition,
            tool_elements=tuple(
                tool_elements
            ),
            history_elements=tuple(
                history_elements
            ),
            memory_elements=tuple(
                memory_elements
            ),
            environment_elements=tuple(
                environment_elements
            ),
            fixture_targets=(
                fixture_targets
            ),
        )

    @staticmethod
    def _history_origin_event_type(
        origin_source: (
            SourceClass | None
        ),
    ) -> EventType:
        if (
            origin_source
            == SourceClass.T
        ):
            return (
                EventType.TOOL_RETURN
            )

        if (
            origin_source
            == SourceClass.E
        ):
            return (
                EventType
                .ENVIRONMENT_QUERY
            )

        return EventType.ACTION


class TaskInterventionResolver:
    """
    Resolve stable task-level intervention templates into concrete
    InterventionPair objects for one baseline run.
    """

    def __init__(
        self,
        *,
        focal_prompt_builder: (
            FocalActionPromptBuilder
            | None
        ) = None,
    ) -> None:
        self.focal_prompt_builder = (
            focal_prompt_builder
            or FocalActionPromptBuilder()
        )

    def resolve_pair(
        self,
        *,
        task: ExperimentalTaskSpec,
        condition: Condition,
        template: InterventionPairTemplate,
        context: ResolvedTaskContext,
    ) -> InterventionPair:
        if (
            condition
            not in template
            .eligible_conditions
        ):
            raise TaskResolutionError(
                code="intervention_not_eligible",
                message=(
                    f"Intervention pair "
                    f"{template.pair_id!r} "
                    f"is not eligible under "
                    f"{condition.value}."
                ),
            )

        if (
            context.condition
            != condition
        ):
            raise TaskResolutionError(
                code="resolved_context_condition_mismatch",
                message=(
                    "Resolved context condition "
                    "does not match intervention "
                    "resolution condition."
                ),
            )

        targeted = (
            self._resolve_mutation(
                task=task,
                template=template,
                mutation=(
                    template.targeted
                ),
                context=context,
                kind=(
                    InterventionKind
                    .TARGETED
                ),
                role="targeted",
            )
        )

        sham = (
            self._resolve_mutation(
                task=task,
                template=template,
                mutation=(
                    template.sham
                ),
                context=context,
                kind=(
                    InterventionKind.SHAM
                ),
                role="sham",
            )
        )

        return InterventionPair(
            pair_id=(
                template.pair_id
            ),
            targeted=targeted,
            sham=sham,
        )

    def _resolve_mutation(
        self,
        *,
        task: ExperimentalTaskSpec,
        template: InterventionPairTemplate,
        mutation: InterventionMutationTemplate,
        context: ResolvedTaskContext,
        kind: InterventionKind,
        role: str,
    ) -> InterventionSpec:
        source = (
            template.source
        )

        target_id: str | None = None

        target_path: (
            tuple[str, ...] | None
        ) = None

        replacement_value = deepcopy(
            mutation.replacement_value
        )

        logical_target: (
            str | list[str] | None
        ) = None

        if source == SourceClass.P:
            target = (
                context
                .fixture_targets
                .get(
                    "prompt"
                )
            )

            if (
                target is None
                or target.target_id
                is None
            ):
                raise TaskResolutionError(
                    code="prompt_target_unresolved",
                    message=(
                        "Canonical P could not be "
                        "resolved for intervention."
                    ),
                )

            if not isinstance(
                replacement_value,
                str,
            ):
                raise TaskResolutionError(
                    code="invalid_prompt_replacement",
                    message=(
                        "P intervention replacement "
                        "must be a task-prompt string."
                    ),
                )

            target_id = (
                target.target_id
            )

            logical_target = (
                "prompt"
            )

            # Critical: baseline P includes the focal-action contract.
            # Rebuild the replacement through the same prompt builder
            # rather than replacing P with the raw task prompt alone.
            replacement_value = (
                self.focal_prompt_builder
                .build(
                    task_prompt=(
                        replacement_value
                    ),
                    schema=(
                        task
                        .focal_decision
                        .schema
                    ),
                )
                .rendered_prompt
            )

        elif source == SourceClass.E:
            target_path = (
                mutation.target_path
            )

            logical_target = (
                list(
                    target_path
                    or ()
                )
            )

        else:
            fixture_id = (
                mutation
                .target_fixture_id
            )

            if fixture_id is None:
                raise TaskResolutionError(
                    code="fixture_target_missing",
                    message=(
                        f"{source.value} intervention "
                        "does not define a logical "
                        "fixture target."
                    ),
                )

            target = (
                context
                .fixture_targets
                .get(
                    fixture_id
                )
            )

            if target is None:
                raise TaskResolutionError(
                    code="fixture_target_unresolved",
                    message=(
                        f"Fixture "
                        f"{fixture_id!r} "
                        "was not materialized under "
                        f"{context.condition.value}."
                    ),
                )

            if (
                target.source
                != source
            ):
                raise TaskResolutionError(
                    code="fixture_source_mismatch",
                    message=(
                        f"Resolved fixture "
                        f"{fixture_id!r} belongs "
                        f"to {target.source.value}, "
                        f"not {source.value}."
                    ),
                )

            if (
                target.target_id
                is None
            ):
                raise TaskResolutionError(
                    code="runtime_target_missing",
                    message=(
                        f"Resolved fixture "
                        f"{fixture_id!r} does not "
                        "have a runtime target ID."
                    ),
                )

            target_id = (
                target.target_id
            )

            logical_target = (
                fixture_id
            )

        return InterventionSpec(
            intervention_id=(
                f"{task.task_id}:"
                f"{template.pair_id}:"
                f"{role}"
            ),
            pair_id=(
                template.pair_id
            ),
            kind=kind,
            source=source,
            operation=(
                mutation.operation
            ),
            target_id=(
                target_id
            ),
            target_path=(
                target_path
            ),
            replacement_value=(
                replacement_value
            ),
            metadata={
                "task_id": (
                    task.task_id
                ),
                "condition": (
                    context.condition.value
                ),
                "logical_target": (
                    logical_target
                ),
                "length_match_required": (
                    mutation
                    .length_match_required
                ),
            },
        )