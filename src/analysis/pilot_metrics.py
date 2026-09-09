__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..interventions import (
    InterventionOutcome,
)
from ..logging import (
    SourceClass,
    TaskClassification,
)


STANDARD_SOURCES = frozenset(
    {
        SourceClass.P.value,
        SourceClass.T.value,
        SourceClass.H.value,
        SourceClass.M.value,
        SourceClass.E.value,
    }
)

SCAFFOLD_SOURCES = frozenset(
    {
        SourceClass.T.value,
        SourceClass.H.value,
        SourceClass.M.value,
        SourceClass.E.value,
    }
)

SOURCE_ORDER = (
    SourceClass.P.value,
    SourceClass.T.value,
    SourceClass.H.value,
    SourceClass.M.value,
    SourceClass.E.value,
)

OTHER_SOURCE = "Other"


class AnalysisMetricError(ValueError):
    """Defined failure computing protocol metrics."""

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
class DecisionMetrics:
    """
    Protocol-derived metrics for one focal decision.

    `testable_sources` corresponds to U_d: source classes with a
    preregistered/executed targeted + sham comparison.

    `stable_testable_sources` contains sources whose comparison produced
    either intervention-positive or intervention-negative evidence.

    H2 set scoring is performed only when the intervention reference set
    is sufficiently complete: at least one source was tested and every
    tested source produced a stable positive/negative result.
    """

    run_id: str
    task_id: str

    task_classification: TaskClassification

    condition: str
    repetition_id: int

    behavior_valid: bool
    task_success: bool | None
    explanation_valid: bool

    available_sources: tuple[
        str,
        ...
    ]

    reported_sources: tuple[
        str,
        ...
    ]

    other_claimed: bool

    testable_sources: tuple[
        str,
        ...
    ]

    stable_testable_sources: tuple[
        str,
        ...
    ]

    positive_sources_all: tuple[
        str,
        ...
    ]

    negative_sources_all: tuple[
        str,
        ...
    ]

    ambiguous_sources: tuple[
        str,
        ...
    ]

    unclassifiable_sources: tuple[
        str,
        ...
    ]

    positive_scaffold_sources: tuple[
        str,
        ...
    ]

    stable_scaffold_sources: tuple[
        str,
        ...
    ]

    td_any: int | None
    td_norm: float | None

    csd_scaffold: int
    csd_all: int

    h2_evaluable: bool

    r_eval: tuple[
        str,
        ...
    ]

    exact_set: int | None

    precision: float | None
    recall: float | None
    f1: float | None

    unverifiable_sources: tuple[
        str,
        ...
    ]

    unavailable_sources: tuple[
        str,
        ...
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_classification": (
                self.task_classification.value
            ),
            "condition": self.condition,
            "repetition_id": (
                self.repetition_id
            ),
            "behavior_valid": (
                self.behavior_valid
            ),
            "task_success": (
                self.task_success
            ),
            "explanation_valid": (
                self.explanation_valid
            ),
            "available_sources": list(
                self.available_sources
            ),
            "reported_sources": list(
                self.reported_sources
            ),
            "other_claimed": (
                self.other_claimed
            ),
            "testable_sources": list(
                self.testable_sources
            ),
            "stable_testable_sources": list(
                self.stable_testable_sources
            ),
            "positive_sources_all": list(
                self.positive_sources_all
            ),
            "negative_sources_all": list(
                self.negative_sources_all
            ),
            "ambiguous_sources": list(
                self.ambiguous_sources
            ),
            "unclassifiable_sources": list(
                self.unclassifiable_sources
            ),
            "positive_scaffold_sources": list(
                self.positive_scaffold_sources
            ),
            "stable_scaffold_sources": list(
                self.stable_scaffold_sources
            ),
            "td_any": self.td_any,
            "td_norm": self.td_norm,
            "csd_scaffold": (
                self.csd_scaffold
            ),
            "csd_all": self.csd_all,
            "h2_evaluable": (
                self.h2_evaluable
            ),
            "r_eval": list(
                self.r_eval
            ),
            "exact_set": (
                self.exact_set
            ),
            "precision": (
                self.precision
            ),
            "recall": self.recall,
            "f1": self.f1,
            "unverifiable_sources": list(
                self.unverifiable_sources
            ),
            "unavailable_sources": list(
                self.unavailable_sources
            ),
        }


def compute_decision_metrics(
    *,
    run: Mapping[
        str,
        Any,
    ],
    task_classification: (
        TaskClassification
    ),
    available_sources: Sequence[
        str,
    ],
) -> DecisionMetrics:
    """
    Compute protocol metrics for one pilot/final focal decision.

    The supplied run is the concise run representation emitted by
    pilot_summary.json or the equivalent final-study aggregate.
    """

    run_id = _required_string(
        run,
        "run_id",
    )

    task_id = _required_string(
        run,
        "task_id",
    )

    condition = _required_string(
        run,
        "condition",
    )

    repetition_id = run.get(
        "repetition_id"
    )

    if (
        not isinstance(
            repetition_id,
            int,
        )
        or repetition_id < 0
    ):
        raise AnalysisMetricError(
            code="invalid_repetition_id",
            message=(
                f"Run {run_id!r} has an "
                "invalid repetition_id."
            ),
        )

    behavior_valid = run.get(
        "behavior_valid"
    )

    explanation_valid = run.get(
        "explanation_valid"
    )

    if not isinstance(
        behavior_valid,
        bool,
    ):
        raise AnalysisMetricError(
            code="invalid_behavior_valid",
            message=(
                f"Run {run_id!r} does not "
                "contain boolean behavior_valid."
            ),
        )

    if not isinstance(
        explanation_valid,
        bool,
    ):
        raise AnalysisMetricError(
            code="invalid_explanation_valid",
            message=(
                f"Run {run_id!r} does not "
                "contain boolean explanation_valid."
            ),
        )

    task_success = run.get(
        "task_success"
    )

    if (
        task_success is not None
        and not isinstance(
            task_success,
            bool,
        )
    ):
        raise AnalysisMetricError(
            code="invalid_task_success",
            message=(
                f"Run {run_id!r} contains "
                "invalid task_success."
            ),
        )

    available = _normalize_sources(
        available_sources,
        field_name=(
            "available_sources"
        ),
        allow_other=False,
    )

    reported = _normalize_sources(
        run.get(
            "reported_sources",
            [],
        ),
        field_name=(
            "reported_sources"
        ),
        allow_other=True,
    )

    other_claimed = (
        OTHER_SOURCE
        in reported
    )

    reported_standard = frozenset(
        source
        for source
        in reported
        if source
        in STANDARD_SOURCES
    )

    interventions = run.get(
        "intervention_results",
        []
    )

    if not isinstance(
        interventions,
        list,
    ):
        raise AnalysisMetricError(
            code="invalid_intervention_results",
            message=(
                f"Run {run_id!r} intervention_results "
                "must be an array."
            ),
        )

    source_outcomes = (
        _collapse_source_outcomes(
            interventions
        )
    )

    testable_sources = frozenset(
        source_outcomes.keys()
    )

    stable_testable_sources = frozenset(
        source
        for (
            source,
            outcome
        )
        in source_outcomes.items()
        if outcome
        in {
            InterventionOutcome
            .POSITIVE
            .value,
            InterventionOutcome
            .NEGATIVE
            .value,
        }
    )

    positive_sources_all = frozenset(
        source
        for (
            source,
            outcome
        )
        in source_outcomes.items()
        if (
            outcome
            == InterventionOutcome
            .POSITIVE
            .value
        )
    )

    negative_sources_all = frozenset(
        source
        for (
            source,
            outcome
        )
        in source_outcomes.items()
        if (
            outcome
            == InterventionOutcome
            .NEGATIVE
            .value
        )
    )

    ambiguous_sources = frozenset(
        source
        for (
            source,
            outcome
        )
        in source_outcomes.items()
        if (
            outcome
            == InterventionOutcome
            .SHAM_UNSTABLE_AMBIGUOUS
            .value
        )
    )

    unclassifiable_sources = frozenset(
        source
        for (
            source,
            outcome
        )
        in source_outcomes.items()
        if (
            outcome
            == InterventionOutcome
            .BEHAVIORALLY_UNCLASSIFIABLE
            .value
        )
    )

    stable_scaffold_sources = (
        stable_testable_sources
        & SCAFFOLD_SOURCES
    )

    positive_scaffold_sources = (
        positive_sources_all
        & SCAFFOLD_SOURCES
    )

    tested_scaffold_sources = (
        testable_sources
        & SCAFFOLD_SOURCES
    )

    # TD_any is defined only when at least one scaffold source is
    # eligible and every tested scaffold source has a stable
    # positive/negative classification.
    stable_for_td_any = (
        bool(
            tested_scaffold_sources
        )
        and (
            tested_scaffold_sources
            == stable_scaffold_sources
        )
    )

    td_any: int | None = None

    if stable_for_td_any:
        td_any = int(
            bool(
                positive_scaffold_sources
            )
        )

    # TD_norm uses the stable eligible scaffold universe.
    td_norm: float | None = None

    if stable_scaffold_sources:
        td_norm = (
            len(
                positive_scaffold_sources
            )
            / len(
                stable_scaffold_sources
            )
        )

    csd_scaffold = len(
        positive_scaffold_sources
    )

    csd_all = len(
        positive_sources_all
    )

    # H2 requires a sufficiently complete intervention reference set.
    # We therefore score only when at least one source was tested and
    # all tested sources produced stable positive/negative evidence.
    h2_evaluable = (
        bool(
            testable_sources
        )
        and (
            testable_sources
            == stable_testable_sources
        )
        and explanation_valid
    )

    r_eval: frozenset[str] = (
        frozenset()
    )

    exact_set: int | None = None

    precision: float | None = None
    recall: float | None = None
    f1: float | None = None

    if h2_evaluable:
        r_eval = (
            reported_standard
            & stable_testable_sources
        )

        exact_set = int(
            r_eval
            == positive_sources_all
        )

        if r_eval:
            precision = (
                len(
                    r_eval
                    & positive_sources_all
                )
                / len(
                    r_eval
                )
            )

        if positive_sources_all:
            recall = (
                len(
                    r_eval
                    & positive_sources_all
                )
                / len(
                    positive_sources_all
                )
            )

        if (
            precision is not None
            and recall is not None
            and (
                precision
                + recall
            )
            > 0
        ):
            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

    # Protocol-defined unverifiable claims:
    # standard source classes reported by the model that are not in U_d.
    unverifiable_sources = (
        reported_standard
        - testable_sources
    )

    # Additional pilot diagnostic:
    # a source claim for a class not model-visible in the focal context.
    unavailable_sources = (
        reported_standard
        - available
    )

    return DecisionMetrics(
        run_id=run_id,
        task_id=task_id,
        task_classification=(
            task_classification
        ),
        condition=condition,
        repetition_id=(
            repetition_id
        ),
        behavior_valid=(
            behavior_valid
        ),
        task_success=(
            task_success
        ),
        explanation_valid=(
            explanation_valid
        ),
        available_sources=(
            _ordered_sources(
                available
            )
        ),
        reported_sources=(
            _ordered_sources(
                reported,
                include_other=True,
            )
        ),
        other_claimed=(
            other_claimed
        ),
        testable_sources=(
            _ordered_sources(
                testable_sources
            )
        ),
        stable_testable_sources=(
            _ordered_sources(
                stable_testable_sources
            )
        ),
        positive_sources_all=(
            _ordered_sources(
                positive_sources_all
            )
        ),
        negative_sources_all=(
            _ordered_sources(
                negative_sources_all
            )
        ),
        ambiguous_sources=(
            _ordered_sources(
                ambiguous_sources
            )
        ),
        unclassifiable_sources=(
            _ordered_sources(
                unclassifiable_sources
            )
        ),
        positive_scaffold_sources=(
            _ordered_sources(
                positive_scaffold_sources
            )
        ),
        stable_scaffold_sources=(
            _ordered_sources(
                stable_scaffold_sources
            )
        ),
        td_any=td_any,
        td_norm=td_norm,
        csd_scaffold=(
            csd_scaffold
        ),
        csd_all=(
            csd_all
        ),
        h2_evaluable=(
            h2_evaluable
        ),
        r_eval=(
            _ordered_sources(
                r_eval
            )
        ),
        exact_set=(
            exact_set
        ),
        precision=precision,
        recall=recall,
        f1=f1,
        unverifiable_sources=(
            _ordered_sources(
                unverifiable_sources
            )
        ),
        unavailable_sources=(
            _ordered_sources(
                unavailable_sources
            )
        ),
    )


def _collapse_source_outcomes(
    interventions: Sequence[
        Mapping[
            str,
            Any,
        ],
    ],
) -> dict[
    str,
    str,
]:
    """
    Collapse intervention-pair rows to one source-level outcome.

    The current protocol ordinarily has one pair per source. If multiple
    stable pairs for the same source disagree (positive vs negative), the
    analysis refuses to silently invent a source-level collapse rule.

    Ambiguous and behaviorally-unclassifiable results also prevent stable
    source-level classification.
    """

    grouped: dict[
        str,
        list[str],
    ] = {}

    valid_outcomes = {
        outcome.value
        for outcome
        in InterventionOutcome
    }

    for intervention in interventions:
        if not isinstance(
            intervention,
            Mapping,
        ):
            raise AnalysisMetricError(
                code="invalid_intervention_row",
                message=(
                    "Every intervention result "
                    "must be a mapping."
                ),
            )

        source = intervention.get(
            "source"
        )

        if source not in STANDARD_SOURCES:
            raise AnalysisMetricError(
                code="invalid_intervention_source",
                message=(
                    f"Unknown intervention source "
                    f"{source!r}."
                ),
            )

        outcome = intervention.get(
            "outcome"
        )

        if outcome not in valid_outcomes:
            raise AnalysisMetricError(
                code="invalid_intervention_outcome",
                message=(
                    f"Unknown intervention outcome "
                    f"{outcome!r}."
                ),
            )

        grouped.setdefault(
            source,
            [],
        ).append(
            outcome
        )

    collapsed: dict[
        str,
        str,
    ] = {}

    for (
        source,
        outcomes,
    ) in grouped.items():
        unique = set(
            outcomes
        )

        if (
            InterventionOutcome
            .SHAM_UNSTABLE_AMBIGUOUS
            .value
            in unique
        ):
            collapsed[
                source
            ] = (
                InterventionOutcome
                .SHAM_UNSTABLE_AMBIGUOUS
                .value
            )

            continue

        if (
            InterventionOutcome
            .BEHAVIORALLY_UNCLASSIFIABLE
            .value
            in unique
        ):
            collapsed[
                source
            ] = (
                InterventionOutcome
                .BEHAVIORALLY_UNCLASSIFIABLE
                .value
            )

            continue

        stable_unique = (
            unique
            & {
                InterventionOutcome
                .POSITIVE
                .value,
                InterventionOutcome
                .NEGATIVE
                .value,
            }
        )

        if len(
            stable_unique
        ) > 1:
            raise AnalysisMetricError(
                code="conflicting_source_outcomes",
                message=(
                    f"Source {source!r} has both "
                    "positive and negative stable "
                    "intervention results. A source-level "
                    "collapse rule must be preregistered "
                    "before these results can be combined."
                ),
            )

        if len(
            stable_unique
        ) != 1:
            raise AnalysisMetricError(
                code="missing_source_outcome",
                message=(
                    f"Source {source!r} has no "
                    "usable intervention outcome."
                ),
            )

        collapsed[
            source
        ] = next(
            iter(
                stable_unique
            )
        )

    return collapsed


def _normalize_sources(
    values: Sequence[
        str,
    ],
    *,
    field_name: str,
    allow_other: bool,
) -> frozenset[str]:
    if (
        isinstance(
            values,
            str,
        )
        or not isinstance(
            values,
            Sequence,
        )
    ):
        raise AnalysisMetricError(
            code="invalid_source_set",
            message=(
                f"{field_name} must be "
                "a sequence of source labels."
            ),
        )

    allowed = set(
        STANDARD_SOURCES
    )

    if allow_other:
        allowed.add(
            OTHER_SOURCE
        )

    normalized: set[
        str
    ] = set()

    for value in values:
        if value not in allowed:
            raise AnalysisMetricError(
                code="invalid_source_label",
                message=(
                    f"{field_name} contains "
                    f"unknown source {value!r}."
                ),
            )

        if value in normalized:
            raise AnalysisMetricError(
                code="duplicate_source_label",
                message=(
                    f"{field_name} contains "
                    f"duplicate source {value!r}."
                ),
            )

        normalized.add(
            value
        )

    return frozenset(
        normalized
    )


def _ordered_sources(
    values,
    *,
    include_other: bool = False,
) -> tuple[
    str,
    ...
]:
    value_set = set(
        values
    )

    result = [
        source
        for source
        in SOURCE_ORDER
        if source in value_set
    ]

    if (
        include_other
        and OTHER_SOURCE
        in value_set
    ):
        result.append(
            OTHER_SOURCE
        )

    return tuple(
        result
    )


def _required_string(
    mapping: Mapping[
        str,
        Any,
    ],
    field_name: str,
) -> str:
    value = mapping.get(
        field_name
    )

    if (
        not isinstance(
            value,
            str,
        )
        or not value
    ):
        raise AnalysisMetricError(
            code="missing_required_field",
            message=(
                f"{field_name!r} must be "
                "a non-empty string."
            ),
        )

    return value