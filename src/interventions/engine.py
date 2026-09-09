__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from copy import deepcopy
from dataclasses import replace
from typing import Any

from ..logging import (
    ContextElement,
    SourceClass,
)
from ..memory import (
    MemoryRecord,
)
from ..snapshots import (
    PreFocalSnapshot,
)
from .schemas import (
    CounterfactualState,
    InterventionApplication,
    InterventionDiff,
    InterventionOperation,
    InterventionSpec,
    checksum_payload,
    serialized_char_count,
)


class InterventionExecutionError(RuntimeError):
    """Defined failure applying an intervention."""

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


class InterventionEngine:
    """
    Apply one preregistered intervention to an immutable baseline snapshot.

    The baseline PreFocalSnapshot is never modified.
    """

    _SOURCE_ORDER = (
        SourceClass.P,
        SourceClass.T,
        SourceClass.H,
        SourceClass.M,
        SourceClass.E,
    )

    def apply(
        self,
        *,
        snapshot: PreFocalSnapshot,
        spec: InterventionSpec,
    ) -> InterventionApplication:
        baseline = (
            CounterfactualState
            .from_snapshot(
                snapshot
            )
        )

        source_checksums_before = (
            self._source_checksums(
                baseline
            )
        )

        (
            counterfactual,
            before_target,
            after_target,
        ) = self._apply_to_state(
            state=baseline,
            spec=spec,
        )

        source_checksums_after = (
            self._source_checksums(
                counterfactual
            )
        )

        changed_sources = tuple(
            source
            for source
            in self._SOURCE_ORDER
            if (
                source_checksums_before[
                    source.value
                ]
                != source_checksums_after[
                    source.value
                ]
            )
        )

        if not changed_sources:
            raise InterventionExecutionError(
                code="intervention_no_effect",
                message=(
                    "Intervention produced no change "
                    "in experimental state."
                ),
            )

        if (
            changed_sources
            != (
                spec.source,
            )
        ):
            raise InterventionExecutionError(
                code="intervention_source_leakage",
                message=(
                    "Intervention changed source classes "
                    "other than the intended source."
                ),
            )

        diff = InterventionDiff(
            source=spec.source,
            operation=spec.operation,
            target=(
                spec.target_descriptor()
            ),
            before=deepcopy(
                before_target
            ),
            after=deepcopy(
                after_target
            ),
            before_sha256=(
                checksum_payload(
                    before_target
                )
            ),
            after_sha256=(
                checksum_payload(
                    after_target
                )
            ),
            before_serialized_chars=(
                serialized_char_count(
                    before_target
                )
            ),
            after_serialized_chars=(
                serialized_char_count(
                    after_target
                )
            ),
        )

        return InterventionApplication(
            spec=spec,
            baseline_snapshot_id=(
                snapshot.snapshot_id
            ),
            baseline_state_checksum=(
                baseline
                .state_checksum_sha256
            ),
            counterfactual_state_checksum=(
                counterfactual
                .state_checksum_sha256
            ),
            changed_sources=(
                changed_sources
            ),
            source_checksums_before=(
                source_checksums_before
            ),
            source_checksums_after=(
                source_checksums_after
            ),
            diff=diff,
            counterfactual_state=(
                counterfactual
            ),
        )

    def _apply_to_state(
        self,
        *,
        state: CounterfactualState,
        spec: InterventionSpec,
    ) -> tuple[
        CounterfactualState,
        Any,
        Any,
    ]:
        if spec.source == SourceClass.P:
            (
                elements,
                before,
                after,
            ) = self._mutate_context_elements(
                elements=(
                    state.prompt_elements
                ),
                spec=spec,
            )

            return (
                self._rebuild_state(
                    state,
                    prompt_elements=elements,
                ),
                before,
                after,
            )

        if spec.source == SourceClass.T:
            (
                elements,
                before,
                after,
            ) = self._mutate_context_elements(
                elements=(
                    state.tool_elements
                ),
                spec=spec,
            )

            return (
                self._rebuild_state(
                    state,
                    tool_elements=elements,
                ),
                before,
                after,
            )

        if spec.source == SourceClass.H:
            (
                elements,
                before,
                after,
            ) = self._mutate_context_elements(
                elements=(
                    state.history_elements
                ),
                spec=spec,
            )

            return (
                self._rebuild_state(
                    state,
                    history_elements=elements,
                ),
                before,
                after,
            )

        if spec.source == SourceClass.M:
            (
                records,
                before,
                after,
            ) = self._mutate_memory(
                state=state,
                spec=spec,
            )

            return (
                self._rebuild_state(
                    state,
                    memory_records=records,
                ),
                before,
                after,
            )

        if spec.source == SourceClass.E:
            (
                environment_state,
                before,
                after,
            ) = self._mutate_environment(
                state=state,
                spec=spec,
            )

            return (
                self._rebuild_state(
                    state,
                    environment_state=(
                        environment_state
                    ),
                ),
                before,
                after,
            )

        raise InterventionExecutionError(
            code="unsupported_intervention_source",
            message=(
                f"Unsupported intervention source "
                f"{spec.source.value!r}."
            ),
        )

    def _mutate_context_elements(
        self,
        *,
        elements: tuple[
            ContextElement,
            ...
        ],
        spec: InterventionSpec,
    ) -> tuple[
        tuple[
            ContextElement,
            ...
        ],
        Any,
        Any,
    ]:
        matches = [
            (
                index,
                element,
            )
            for index, element
            in enumerate(
                elements
            )
            if (
                element.source_id
                == spec.target_id
            )
        ]

        if not matches:
            raise InterventionExecutionError(
                code="intervention_target_not_found",
                message=(
                    f"Context target "
                    f"{spec.target_id!r} "
                    "was not found."
                ),
            )

        if len(matches) != 1:
            raise InterventionExecutionError(
                code="intervention_target_ambiguous",
                message=(
                    f"Context target "
                    f"{spec.target_id!r} "
                    "was not unique."
                ),
            )

        index, element = (
            matches[0]
        )

        before = deepcopy(
            element.content
        )

        updated = list(
            elements
        )

        if (
            spec.operation
            == InterventionOperation.REMOVE
        ):
            del updated[
                index
            ]

            after = None

        else:
            replacement = deepcopy(
                spec.replacement_value
            )

            updated[
                index
            ] = ContextElement(
                source=element.source,
                source_id=(
                    element.source_id
                ),
                content=replacement,
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

            after = replacement

        return (
            tuple(
                updated
            ),
            before,
            after,
        )

    def _mutate_memory(
        self,
        *,
        state: CounterfactualState,
        spec: InterventionSpec,
    ) -> tuple[
        tuple[
            MemoryRecord,
            ...
        ],
        Any,
        Any,
    ]:
        if (
            state.memory_namespace
            is None
        ):
            raise InterventionExecutionError(
                code="memory_state_unavailable",
                message=(
                    "Baseline state contains no "
                    "persistent memory M."
                ),
            )

        matches = [
            (
                index,
                record,
            )
            for index, record
            in enumerate(
                state.memory_records
            )
            if (
                record.memory_id
                == spec.target_id
            )
        ]

        if not matches:
            raise InterventionExecutionError(
                code="intervention_target_not_found",
                message=(
                    f"Memory target "
                    f"{spec.target_id!r} "
                    "was not found."
                ),
            )

        if len(matches) != 1:
            raise InterventionExecutionError(
                code="intervention_target_ambiguous",
                message=(
                    f"Memory target "
                    f"{spec.target_id!r} "
                    "was not unique."
                ),
            )

        index, record = (
            matches[0]
        )

        before = deepcopy(
            record.value
        )

        updated = list(
            state.memory_records
        )

        if (
            spec.operation
            == InterventionOperation.REMOVE
        ):
            del updated[
                index
            ]

            after = None

        else:
            replacement = deepcopy(
                spec.replacement_value
            )

            updated[
                index
            ] = MemoryRecord(
                namespace=(
                    record.namespace
                ),
                memory_id=(
                    record.memory_id
                ),
                key=record.key,
                value=replacement,
                created_by_task=(
                    record.created_by_task
                ),
                created_sequence=(
                    record.created_sequence
                ),
                metadata=deepcopy(
                    record.metadata
                ),
            )

            after = replacement

        return (
            tuple(
                updated
            ),
            before,
            after,
        )

    def _mutate_environment(
        self,
        *,
        state: CounterfactualState,
        spec: InterventionSpec,
    ) -> tuple[
        dict[str, Any],
        Any,
        Any,
    ]:
        if (
            state.environment_state
            is None
        ):
            raise InterventionExecutionError(
                code="environment_state_unavailable",
                message=(
                    "Baseline state contains no "
                    "environmental state E."
                ),
            )

        environment = deepcopy(
            state.environment_state
        )

        path = (
            spec.target_path
            or ()
        )

        current: Any = (
            environment
        )

        for key in path[:-1]:
            if (
                not isinstance(
                    current,
                    dict,
                )
                or key
                not in current
            ):
                raise InterventionExecutionError(
                    code="intervention_target_not_found",
                    message=(
                        "Environmental intervention "
                        f"path {list(path)!r} "
                        "does not exist."
                    ),
                )

            current = current[
                key
            ]

        final_key = path[-1]

        if (
            not isinstance(
                current,
                dict,
            )
            or final_key
            not in current
        ):
            raise InterventionExecutionError(
                code="intervention_target_not_found",
                message=(
                    "Environmental intervention "
                    f"path {list(path)!r} "
                    "does not exist."
                ),
            )

        before = deepcopy(
            current[
                final_key
            ]
        )

        replacement = deepcopy(
            spec.replacement_value
        )

        current[
            final_key
        ] = replacement

        return (
            environment,
            before,
            replacement,
        )

    def _rebuild_state(
        self,
        state: CounterfactualState,
        **changes,
    ) -> CounterfactualState:
        provisional = replace(
            state,
            **changes,
            state_checksum_sha256="",
        )

        checksum = checksum_payload(
            provisional.state_payload()
        )

        return replace(
            provisional,
            state_checksum_sha256=(
                checksum
            ),
        )

    def _source_checksums(
        self,
        state: CounterfactualState,
    ) -> dict[str, str]:
        return {
            source.value: (
                state.source_checksum(
                    source
                )
            )
            for source
            in self._SOURCE_ORDER
        }