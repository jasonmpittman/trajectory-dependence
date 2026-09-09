__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

from ..logging import Condition, SourceClass


class ConditionConfigurationError(ValueError):
    """Defined failure in experimental condition configuration."""

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
class ConditionPolicy:
    """
    Immutable operational policy for one experimental condition.

    C0-C3 govern cumulative availability of P, T, H, and M.

    Environmental state E is intentionally excluded from the cumulative
    condition ladder. E is exogenous and becomes available only when the
    task specification and future context-construction policy explicitly
    permit it.
    """

    condition: Condition
    label: str

    allow_prompt: bool
    allow_tools: bool
    allow_history: bool
    allow_persistent_memory: bool

    environment_policy: str = "task_controlled"

    @property
    def cumulative_sources(self) -> frozenset[SourceClass]:
        """
        Return P/T/H/M source classes authorized by this condition.

        E is intentionally excluded because environmental availability is
        task-controlled rather than cumulative across C0-C3.
        """

        sources: set[SourceClass] = set()

        if self.allow_prompt:
            sources.add(SourceClass.P)

        if self.allow_tools:
            sources.add(SourceClass.T)

        if self.allow_history:
            sources.add(SourceClass.H)

        if self.allow_persistent_memory:
            sources.add(SourceClass.M)

        return frozenset(sources)

    @property
    def scaffold_sources(self) -> frozenset[SourceClass]:
        """Return authorized non-prompt cumulative scaffold sources."""

        return frozenset(
            source
            for source in self.cumulative_sources
            if source != SourceClass.P
        )

    def permits_cumulative_source(
        self,
        source: SourceClass,
    ) -> bool:
        """
        Return whether a P/T/H/M source is permitted by this condition.

        E is deliberately rejected here rather than returning True/False,
        because environmental availability is not a property of the
        cumulative condition ladder.
        """

        if source == SourceClass.E:
            raise ConditionConfigurationError(
                code="environment_not_cumulative",
                message=(
                    "Environmental source E is task-controlled and is "
                    "not governed by the cumulative C0-C3 source policy."
                ),
            )

        return source in self.cumulative_sources

    def to_dict(self) -> dict:
        """Return a deterministic manifest-compatible representation."""

        source_order = (
            SourceClass.P,
            SourceClass.T,
            SourceClass.H,
            SourceClass.M,
        )

        return {
            "condition": self.condition.value,
            "label": self.label,
            "allow_prompt": self.allow_prompt,
            "allow_tools": self.allow_tools,
            "allow_history": self.allow_history,
            "allow_persistent_memory": (
                self.allow_persistent_memory
            ),
            "environment_policy": self.environment_policy,
            "cumulative_sources": [
                source.value
                for source in source_order
                if source in self.cumulative_sources
            ],
        }


_CONDITION_POLICIES = {
    Condition.C0: ConditionPolicy(
        condition=Condition.C0,
        label="stateless",
        allow_prompt=True,
        allow_tools=False,
        allow_history=False,
        allow_persistent_memory=False,
    ),
    Condition.C1: ConditionPolicy(
        condition=Condition.C1,
        label="tool-enabled",
        allow_prompt=True,
        allow_tools=True,
        allow_history=False,
        allow_persistent_memory=False,
    ),
    Condition.C2: ConditionPolicy(
        condition=Condition.C2,
        label="trajectory-enabled",
        allow_prompt=True,
        allow_tools=True,
        allow_history=True,
        allow_persistent_memory=False,
    ),
    Condition.C3: ConditionPolicy(
        condition=Condition.C3,
        label="persistent-agent",
        allow_prompt=True,
        allow_tools=True,
        allow_history=True,
        allow_persistent_memory=True,
    ),
}


CONDITION_POLICIES: Mapping[
    Condition,
    ConditionPolicy,
] = MappingProxyType(
    _CONDITION_POLICIES
)


def get_condition_policy(
    condition: Condition | str,
) -> ConditionPolicy:
    """Return the immutable policy for a condition."""

    if isinstance(condition, Condition):
        normalized = condition
    else:
        try:
            normalized = Condition(condition)
        except (TypeError, ValueError) as exc:
            raise ConditionConfigurationError(
                code="unknown_condition",
                message=(
                    f"Unknown experimental condition: {condition!r}."
                ),
            ) from exc

    try:
        return CONDITION_POLICIES[
            normalized
        ]
    except KeyError as exc:
        raise ConditionConfigurationError(
            code="unknown_condition",
            message=(
                f"No condition policy exists for {normalized.value!r}."
            ),
        ) from exc


def validate_condition_source_set(
    *,
    condition: Condition | str,
    sources: Iterable[SourceClass | str],
) -> frozenset[SourceClass]:
    """
    Validate P/T/H/M source availability against a condition.

    This function validates authorization only. It does not require every
    authorized source to be present. For example, C3 permits M, but a
    particular focal decision may legitimately contain no retrieved memory.

    E is rejected because environmental-state availability must be
    validated separately against the task specification.
    """

    policy = get_condition_policy(
        condition
    )

    normalized: set[SourceClass] = set()

    for source in sources:
        if isinstance(source, SourceClass):
            normalized_source = source
        else:
            try:
                normalized_source = SourceClass(
                    source
                )
            except (TypeError, ValueError) as exc:
                raise ConditionConfigurationError(
                    code="unknown_source",
                    message=(
                        f"Unknown experimental source class: "
                        f"{source!r}."
                    ),
                ) from exc

        if normalized_source == SourceClass.E:
            raise ConditionConfigurationError(
                code="environment_requires_task_policy",
                message=(
                    "Environmental source E must be authorized by the "
                    "task/environment policy rather than the cumulative "
                    "C0-C3 condition policy."
                ),
            )

        normalized.add(
            normalized_source
        )

    prohibited = (
        normalized
        - policy.cumulative_sources
    )

    if prohibited:
        ordered = sorted(
            source.value
            for source in prohibited
        )

        raise ConditionConfigurationError(
            code="condition_source_violation",
            message=(
                f"Condition {policy.condition.value} does not permit "
                f"cumulative source(s): {', '.join(ordered)}."
            ),
        )

    return frozenset(
        normalized
    )


def validate_condition_policy_table() -> None:
    """
    Validate the complete C0-C3 policy table against protocol invariants.

    This function is intended for tests and startup validation. It does
    not alter the policy table.
    """

    expected_conditions = {
        Condition.C0,
        Condition.C1,
        Condition.C2,
        Condition.C3,
    }

    if set(
        CONDITION_POLICIES.keys()
    ) != expected_conditions:
        raise ConditionConfigurationError(
            code="incomplete_condition_table",
            message=(
                "Condition policy table must contain exactly "
                "C0, C1, C2, and C3."
            ),
        )

    expected_sources = {
        Condition.C0: frozenset(
            {
                SourceClass.P,
            }
        ),
        Condition.C1: frozenset(
            {
                SourceClass.P,
                SourceClass.T,
            }
        ),
        Condition.C2: frozenset(
            {
                SourceClass.P,
                SourceClass.T,
                SourceClass.H,
            }
        ),
        Condition.C3: frozenset(
            {
                SourceClass.P,
                SourceClass.T,
                SourceClass.H,
                SourceClass.M,
            }
        ),
    }

    for condition, expected in expected_sources.items():
        policy = CONDITION_POLICIES[
            condition
        ]

        if not policy.allow_prompt:
            raise ConditionConfigurationError(
                code="prompt_not_allowed",
                message=(
                    f"{condition.value} must permit current prompt P."
                ),
            )

        if (
            policy.cumulative_sources
            != expected
        ):
            raise ConditionConfigurationError(
                code="condition_policy_mismatch",
                message=(
                    f"{condition.value} cumulative source policy "
                    "does not match the protocol."
                ),
            )

        if (
            policy.environment_policy
            != "task_controlled"
        ):
            raise ConditionConfigurationError(
                code="invalid_environment_policy",
                message=(
                    f"{condition.value} must treat environmental "
                    "source E as task-controlled."
                ),
            )

    ordered_conditions = (
        Condition.C0,
        Condition.C1,
        Condition.C2,
        Condition.C3,
    )

    for lower, higher in zip(
        ordered_conditions,
        ordered_conditions[1:],
    ):
        lower_sources = (
            CONDITION_POLICIES[
                lower
            ].cumulative_sources
        )

        higher_sources = (
            CONDITION_POLICIES[
                higher
            ].cumulative_sources
        )

        if not lower_sources.issubset(
            higher_sources
        ):
            raise ConditionConfigurationError(
                code="non_cumulative_condition_policy",
                message=(
                    f"{higher.value} must contain every cumulative "
                    f"source permitted by {lower.value}."
                ),
            )