__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence
from hashlib import sha256

from ..logging import (
    Condition,
    ContextElement,
    SourceClass,
)
from .conditions import (
    ConditionConfigurationError,
    get_condition_policy,
    validate_condition_source_set,
)


SOURCE_ORDER = (
    SourceClass.P,
    SourceClass.T,
    SourceClass.H,
    SourceClass.M,
    SourceClass.E,
)


class ContextBuildError(ValueError):
    """Defined failure during provenance-aware context construction."""

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
class TaskEnvironmentPolicy:
    """
    Task-level authorization for explicit environmental context E.

    E is not cumulative across C0-C3. A task must explicitly authorize
    environment-derived context before it can enter a model-visible
    context in C1-C3.

    If allowed_source_ids is None, all E elements explicitly supplied by
    the task/runtime are authorized.

    If allowed_source_ids is non-None, only the named environmental
    source IDs are authorized.
    """

    allow_environment: bool
    allowed_source_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.allowed_source_ids is not None:
            normalized = frozenset(
                self.allowed_source_ids
            )

            if not all(
                isinstance(source_id, str)
                and source_id
                for source_id in normalized
            ):
                raise ContextBuildError(
                    code="invalid_environment_source_id",
                    message=(
                        "Environmental source IDs must be non-empty "
                        "strings."
                    ),
                )

            if not self.allow_environment:
                raise ContextBuildError(
                    code="inconsistent_environment_policy",
                    message=(
                        "allowed_source_ids may not be specified when "
                        "allow_environment is false."
                    ),
                )

            object.__setattr__(
                self,
                "allowed_source_ids",
                normalized,
            )

    def permits(
        self,
        source_id: str,
    ) -> bool:
        """Return whether one environmental source ID is authorized."""

        if not self.allow_environment:
            return False

        if self.allowed_source_ids is None:
            return True

        return (
            source_id
            in self.allowed_source_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_environment": self.allow_environment,
            "allowed_source_ids": (
                sorted(
                    self.allowed_source_ids
                )
                if self.allowed_source_ids
                is not None
                else None
            ),
        }


@dataclass(frozen=True)
class StructuredContext:
    """
    Provenance-preserving structured model context.

    This object represents what is authorized to become model-visible.
    It is not yet the serialized model input.
    """

    task_id: str
    condition: Condition

    elements: tuple[ContextElement, ...]

    environment_policy: TaskEnvironmentPolicy

    @property
    def source_classes(
        self,
    ) -> frozenset[SourceClass]:
        return frozenset(
            element.source
            for element in self.elements
        )

    @property
    def source_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            element.source_id
            for element in self.elements
        )

    def elements_for(
        self,
        source: SourceClass,
    ) -> tuple[ContextElement, ...]:
        """Return context elements belonging to one source class."""

        return tuple(
            element
            for element in self.elements
            if element.source == source
        )

    def to_dict(self) -> dict[str, Any]:
        source_classes = [
            source.value
            for source in SOURCE_ORDER
            if source in self.source_classes
        ]

        return {
            "task_id": self.task_id,
            "condition": self.condition.value,
            "source_classes": source_classes,
            "elements": [
                element.to_dict()
                for element in self.elements
            ],
            "environment_policy": (
                self.environment_policy.to_dict()
            ),
        }

SERIALIZATION_VERSION = "context-v1"

SECTION_HEADERS = {
    SourceClass.P: "=== CURRENT TASK ===",
    SourceClass.T: "=== TOOL INTERACTIONS ===",
    SourceClass.H: "=== CURRENT-TASK HISTORY ===",
    SourceClass.M: "=== PERSISTENT MEMORY ===",
    SourceClass.E: "=== ENVIRONMENT STATE ===",
}

ELEMENT_SEPARATOR = "\n\n---\n\n"
SECTION_SEPARATOR = "\n\n"


class ContextSerializationError(ValueError):
    """Defined failure during deterministic context serialization."""

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
class SerializedElementSpan:
    """
    Audit mapping between one serialized model-visible content span and
    its experimental provenance.

    Character offsets are zero-based and end-exclusive.

    This metadata is not added to the model-visible serialized text.
    """

    element_index: int

    source: SourceClass
    source_id: str

    origin_source: SourceClass | None
    origin_event_id: str | None

    start_char: int
    end_char: int

    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_index": self.element_index,
            "source": self.source.value,
            "source_id": self.source_id,
            "origin_source": (
                self.origin_source.value
                if self.origin_source is not None
                else None
            ),
            "origin_event_id": self.origin_event_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class SerializedContext:
    """
    Deterministic pre-model serialized context.

    `model_text` is the exact context text that may be passed to the
    future model adapter.

    `spans` is audit metadata and must never be injected into model_text.
    """

    task_id: str
    condition: Condition

    serialization_version: str

    model_text: str
    model_text_sha256: str

    spans: tuple[SerializedElementSpan, ...]

    @property
    def model_text_bytes(self) -> bytes:
        """Return the exact UTF-8 byte representation."""

        return self.model_text.encode(
            "utf-8"
        )

    def to_audit_dict(self) -> dict[str, Any]:
        """Return complete non-model-visible serialization metadata."""

        return {
            "task_id": self.task_id,
            "condition": self.condition.value,
            "serialization_version": (
                self.serialization_version
            ),
            "model_text_sha256": (
                self.model_text_sha256
            ),
            "model_text_length_chars": len(
                self.model_text
            ),
            "model_text_length_bytes": len(
                self.model_text_bytes
            ),
            "spans": [
                span.to_dict()
                for span in self.spans
            ],
        }


class ContextSerializer:
    """
    Deterministically serialize StructuredContext for model consumption.

    Serialization guarantees:

    - source-category order is P -> T -> H -> M -> E;
    - element order within a category is preserved;
    - standardized semantic source headers are model-visible;
    - experimental source IDs and origin metadata are not model-visible;
    - strings are preserved exactly;
    - structured values use deterministic sorted-key JSON;
    - identical structured context produces byte-identical model text.
    """

    def serialize(
        self,
        context: StructuredContext,
    ) -> SerializedContext:
        self._validate_context(
            context
        )

        parts: list[str] = []
        spans: list[SerializedElementSpan] = []

        cursor = 0
        element_index = 0
        first_section = True

        def append_text(
            value: str,
        ) -> None:
            nonlocal cursor

            parts.append(
                value
            )

            cursor += len(
                value
            )

        for source in SOURCE_ORDER:
            elements = context.elements_for(
                source
            )

            if not elements:
                continue

            if not first_section:
                append_text(
                    SECTION_SEPARATOR
                )

            first_section = False

            append_text(
                SECTION_HEADERS[source]
            )

            append_text(
                "\n"
            )

            for index_in_section, element in enumerate(
                elements
            ):
                if index_in_section > 0:
                    append_text(
                        ELEMENT_SEPARATOR
                    )

                serialized_content = (
                    self._serialize_content(
                        element.content
                    )
                )

                start_char = cursor

                append_text(
                    serialized_content
                )

                end_char = cursor

                spans.append(
                    SerializedElementSpan(
                        element_index=element_index,
                        source=element.source,
                        source_id=element.source_id,
                        origin_source=(
                            element.origin_source
                        ),
                        origin_event_id=(
                            element.origin_event_id
                        ),
                        start_char=start_char,
                        end_char=end_char,
                        content_sha256=sha256(
                            serialized_content.encode(
                                "utf-8"
                            )
                        ).hexdigest(),
                    )
                )

                element_index += 1

        model_text = "".join(
            parts
        )

        model_text_sha256 = sha256(
            model_text.encode(
                "utf-8"
            )
        ).hexdigest()

        return SerializedContext(
            task_id=context.task_id,
            condition=context.condition,
            serialization_version=(
                SERIALIZATION_VERSION
            ),
            model_text=model_text,
            model_text_sha256=(
                model_text_sha256
            ),
            spans=tuple(
                spans
            ),
        )

    def _validate_context(
        self,
        context: StructuredContext,
    ) -> None:
        """
        Revalidate core invariants before serialization.

        StructuredContext can technically be instantiated directly rather
        than through ContextBuilder, so the serializer must not blindly
        trust it.
        """

        if (
            not isinstance(
                context.task_id,
                str,
            )
            or not context.task_id
        ):
            raise ContextSerializationError(
                code="invalid_task_id",
                message=(
                    "Structured context must contain a non-empty "
                    "task_id."
                ),
            )

        if not context.elements:
            raise ContextSerializationError(
                code="empty_context",
                message=(
                    "Structured context contains no model-visible "
                    "elements."
                ),
            )

        prompt_elements = context.elements_for(
            SourceClass.P
        )

        if not prompt_elements:
            raise ContextSerializationError(
                code="missing_prompt",
                message=(
                    "Structured context must contain at least one "
                    "current-task prompt element P."
                ),
            )

        source_rank = {
            source: index
            for index, source in enumerate(
                SOURCE_ORDER
            )
        }

        observed_ranks = [
            source_rank[
                element.source
            ]
            for element in context.elements
        ]

        if observed_ranks != sorted(
            observed_ranks
        ):
            raise ContextSerializationError(
                code="non_deterministic_source_order",
                message=(
                    "Structured context elements are not ordered "
                    "P -> T -> H -> M -> E."
                ),
            )

        source_ids = [
            element.source_id
            for element in context.elements
        ]

        if len(
            source_ids
        ) != len(
            set(
                source_ids
            )
        ):
            raise ContextSerializationError(
                code="duplicate_source_id",
                message=(
                    "Structured context contains duplicate "
                    "source IDs."
                ),
            )

        cumulative_sources = {
            element.source
            for element in context.elements
            if element.source
            != SourceClass.E
        }

        try:
            validate_condition_source_set(
                condition=context.condition,
                sources=cumulative_sources,
            )

        except ConditionConfigurationError as exc:
            raise ContextSerializationError(
                code="condition_source_violation",
                message=exc.message,
            ) from exc

        environment_elements = (
            context.elements_for(
                SourceClass.E
            )
        )

        if environment_elements:
            if (
                context.condition
                == Condition.C0
            ):
                raise ContextSerializationError(
                    code="c0_environment_violation",
                    message=(
                        "C0 may not contain explicit "
                        "environment-derived context E."
                    ),
                )

            if not (
                context.environment_policy
                .allow_environment
            ):
                raise ContextSerializationError(
                    code="environment_not_authorized",
                    message=(
                        "Structured context contains E but its task "
                        "environment policy does not authorize E."
                    ),
                )

            unauthorized = [
                element.source_id
                for element in environment_elements
                if not (
                    context.environment_policy
                    .permits(
                        element.source_id
                    )
                )
            ]

            if unauthorized:
                raise ContextSerializationError(
                    code="environment_source_not_authorized",
                    message=(
                        "Structured context contains unauthorized "
                        "environmental source ID(s): "
                        + ", ".join(
                            sorted(
                                unauthorized
                            )
                        )
                        + "."
                    ),
                )

        for element in context.elements:
            if (
                not isinstance(
                    element.source_id,
                    str,
                )
                or not element.source_id
            ):
                raise ContextSerializationError(
                    code="invalid_source_id",
                    message=(
                        "Structured context contains an invalid "
                        "source_id."
                    ),
                )

            try:
                self._serialize_content(
                    element.content
                )

            except ContextSerializationError:
                raise

    @staticmethod
    def _serialize_content(
        content: Any,
    ) -> str:
        """
        Serialize one context element without adding provenance metadata.

        Strings remain exact strings.

        All other values use deterministic JSON with sorted object keys.
        """

        if isinstance(
            content,
            str,
        ):
            return content

        try:
            return json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ContextSerializationError(
                code="non_serializable_content",
                message=(
                    "Context content must be a string or losslessly "
                    "representable as JSON."
                ),
            ) from exc

class ContextBuilder:
    """
    Build deterministic provenance-preserving structured context.

    Bucket order is fixed:

        P -> T -> H -> M -> E

    Order within each bucket is preserved exactly as supplied by the
    caller.

    The builder does not retrieve state or silently filter prohibited
    sources. Invalid state causes an explicit ContextBuildError.
    """

    def build(
        self,
        *,
        task_id: str,
        condition: Condition | str,
        prompt_elements: Sequence[ContextElement],
        tool_elements: Sequence[ContextElement] = (),
        history_elements: Sequence[ContextElement] = (),
        memory_elements: Sequence[ContextElement] = (),
        environment_elements: Sequence[ContextElement] = (),
        environment_policy: TaskEnvironmentPolicy | None = None,
    ) -> StructuredContext:
        if not isinstance(task_id, str) or not task_id:
            raise ContextBuildError(
                code="invalid_task_id",
                message="task_id must be a non-empty string.",
            )

        policy = get_condition_policy(
            condition
        )

        prompt_copy = self._clone_elements(
            prompt_elements
        )

        tool_copy = self._clone_elements(
            tool_elements
        )

        history_copy = self._clone_elements(
            history_elements
        )

        memory_copy = self._clone_elements(
            memory_elements
        )

        environment_copy = self._clone_elements(
            environment_elements
        )

        if not prompt_copy:
            raise ContextBuildError(
                code="missing_prompt",
                message=(
                    "Every model-visible context must contain at least "
                    "one current prompt element P."
                ),
            )

        self._validate_bucket(
            prompt_copy,
            expected_source=SourceClass.P,
            bucket_name="prompt_elements",
        )

        self._validate_bucket(
            tool_copy,
            expected_source=SourceClass.T,
            bucket_name="tool_elements",
        )

        self._validate_bucket(
            history_copy,
            expected_source=SourceClass.H,
            bucket_name="history_elements",
        )

        self._validate_bucket(
            memory_copy,
            expected_source=SourceClass.M,
            bucket_name="memory_elements",
        )

        self._validate_bucket(
            environment_copy,
            expected_source=SourceClass.E,
            bucket_name="environment_elements",
        )

        cumulative_sources: set[SourceClass] = {
            SourceClass.P,
        }

        if tool_copy:
            cumulative_sources.add(
                SourceClass.T
            )

        if history_copy:
            cumulative_sources.add(
                SourceClass.H
            )

        if memory_copy:
            cumulative_sources.add(
                SourceClass.M
            )

        try:
            validate_condition_source_set(
                condition=policy.condition,
                sources=cumulative_sources,
            )
        except ConditionConfigurationError as exc:
            raise ContextBuildError(
                code="condition_source_violation",
                message=exc.message,
            ) from exc

        resolved_environment_policy = (
            environment_policy
            if environment_policy is not None
            else TaskEnvironmentPolicy(
                allow_environment=False
            )
        )

        self._validate_environment(
            condition=policy.condition,
            elements=environment_copy,
            policy=resolved_environment_policy,
        )

        combined = (
            prompt_copy
            + tool_copy
            + history_copy
            + memory_copy
            + environment_copy
        )

        self._validate_unique_source_ids(
            combined
        )

        for element in combined:
            self._validate_element_json(
                element
            )

        return StructuredContext(
            task_id=task_id,
            condition=policy.condition,
            elements=combined,
            environment_policy=(
                resolved_environment_policy
            ),
        )

    def _validate_environment(
        self,
        *,
        condition: Condition,
        elements: tuple[ContextElement, ...],
        policy: TaskEnvironmentPolicy,
    ) -> None:
        if not elements:
            return

        if condition == Condition.C0:
            raise ContextBuildError(
                code="c0_environment_violation",
                message=(
                    "C0 is stateless and may not receive explicit "
                    "environment-derived context E."
                ),
            )

        if not policy.allow_environment:
            raise ContextBuildError(
                code="environment_not_authorized",
                message=(
                    "Environmental context E was supplied but the task "
                    "does not authorize environmental exposure."
                ),
            )

        unauthorized = [
            element.source_id
            for element in elements
            if not policy.permits(
                element.source_id
            )
        ]

        if unauthorized:
            raise ContextBuildError(
                code="environment_source_not_authorized",
                message=(
                    "Task environment policy does not authorize "
                    "environmental source ID(s): "
                    + ", ".join(
                        sorted(
                            unauthorized
                        )
                    )
                    + "."
                ),
            )

    @staticmethod
    def _validate_bucket(
        elements: tuple[ContextElement, ...],
        *,
        expected_source: SourceClass,
        bucket_name: str,
    ) -> None:
        for element in elements:
            if element.source != expected_source:
                raise ContextBuildError(
                    code="source_bucket_mismatch",
                    message=(
                        f"{bucket_name!r} contains source "
                        f"{element.source.value!r}; expected "
                        f"{expected_source.value!r}."
                    ),
                )

            if (
                not isinstance(
                    element.source_id,
                    str,
                )
                or not element.source_id
            ):
                raise ContextBuildError(
                    code="invalid_source_id",
                    message=(
                        f"{bucket_name!r} contains an invalid "
                        "source_id."
                    ),
                )

    @staticmethod
    def _validate_unique_source_ids(
        elements: tuple[ContextElement, ...],
    ) -> None:
        seen: set[str] = set()

        duplicates: set[str] = set()

        for element in elements:
            if element.source_id in seen:
                duplicates.add(
                    element.source_id
                )

            seen.add(
                element.source_id
            )

        if duplicates:
            raise ContextBuildError(
                code="duplicate_source_id",
                message=(
                    "Context contains duplicate source ID(s): "
                    + ", ".join(
                        sorted(
                            duplicates
                        )
                    )
                    + "."
                ),
            )

    @staticmethod
    def _clone_elements(
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

    @staticmethod
    def _validate_element_json(
        element: ContextElement,
    ) -> None:
        try:
            json.dumps(
                element.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )

        except (TypeError, ValueError) as exc:
            raise ContextBuildError(
                code="non_serializable_context_element",
                message=(
                    "Context element "
                    f"{element.source_id!r} is not losslessly "
                    "representable as JSON."
                ),
            ) from exc