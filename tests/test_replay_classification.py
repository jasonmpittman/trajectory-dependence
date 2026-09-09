__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from dataclasses import dataclass

import pytest

from src.interventions import (
    InterventionOutcome,
    MatchedReplayClassifier,
)
from src.logging import (
    SourceClass,
)


@dataclass
class FakeNormalizedAction:
    payload: dict


@dataclass
class FakeParseResult:
    normalized_action: (
        FakeNormalizedAction
        | None
    )


@dataclass
class FakeAction:
    valid: bool
    parse_result: (
        FakeParseResult
        | None
    )


@dataclass
class FakeSpec:
    source: SourceClass


@dataclass
class FakeApplication:
    spec: FakeSpec


@dataclass
class FakeReplay:
    application: FakeApplication
    focal_action: FakeAction


@dataclass
class FakeExecution:
    baseline_action: FakeAction
    targeted: FakeReplay
    sham: FakeReplay


def action(
    value: str,
) -> FakeAction:
    return FakeAction(
        valid=True,
        parse_result=FakeParseResult(
            normalized_action=(
                FakeNormalizedAction(
                    payload={
                        "action": (
                            "select_route"
                        ),
                        "value": value,
                    }
                )
            )
        ),
    )


def invalid_action() -> FakeAction:
    return FakeAction(
        valid=False,
        parse_result=None,
    )


def execution(
    *,
    baseline: FakeAction,
    targeted: FakeAction,
    sham: FakeAction,
    targeted_source: SourceClass = (
        SourceClass.T
    ),
    sham_source: SourceClass = (
        SourceClass.T
    ),
) -> FakeExecution:
    return FakeExecution(
        baseline_action=baseline,
        targeted=FakeReplay(
            application=(
                FakeApplication(
                    spec=FakeSpec(
                        source=(
                            targeted_source
                        )
                    )
                )
            ),
            focal_action=targeted,
        ),
        sham=FakeReplay(
            application=(
                FakeApplication(
                    spec=FakeSpec(
                        source=(
                            sham_source
                        )
                    )
                )
            ),
            focal_action=sham,
        ),
    )


def test_intervention_positive() -> None:
    result = (
        MatchedReplayClassifier()
        .classify(
            execution(
                baseline=action("B"),
                targeted=action("A"),
                sham=action("B"),
            )
        )
    )

    assert (
        result.outcome
        == InterventionOutcome.POSITIVE
    )

    assert (
        result.targeted_changed
        is True
    )

    assert (
        result.sham_changed
        is False
    )


def test_intervention_negative() -> None:
    result = (
        MatchedReplayClassifier()
        .classify(
            execution(
                baseline=action("B"),
                targeted=action("B"),
                sham=action("B"),
            )
        )
    )

    assert (
        result.outcome
        == InterventionOutcome.NEGATIVE
    )

    assert (
        result.targeted_changed
        is False
    )

    assert (
        result.sham_changed
        is False
    )


def test_sham_change_is_ambiguous_when_target_changes() -> None:
    result = (
        MatchedReplayClassifier()
        .classify(
            execution(
                baseline=action("B"),
                targeted=action("A"),
                sham=action("A"),
            )
        )
    )

    assert (
        result.outcome
        == (
            InterventionOutcome
            .SHAM_UNSTABLE_AMBIGUOUS
        )
    )

    assert (
        result.targeted_changed
        is True
    )

    assert (
        result.sham_changed
        is True
    )


def test_sham_change_is_ambiguous_when_target_stable() -> None:
    result = (
        MatchedReplayClassifier()
        .classify(
            execution(
                baseline=action("B"),
                targeted=action("B"),
                sham=action("A"),
            )
        )
    )

    assert (
        result.outcome
        == (
            InterventionOutcome
            .SHAM_UNSTABLE_AMBIGUOUS
        )
    )

    assert (
        result.targeted_changed
        is False
    )

    assert (
        result.sham_changed
        is True
    )


def test_invalid_baseline_action_is_unclassifiable() -> None:
    result = (
        MatchedReplayClassifier()
        .classify(
            execution(
                baseline=(
                    invalid_action()
                ),
                targeted=action("A"),
                sham=action("B"),
            )
        )
    )

    assert (
        result.outcome
        == (
            InterventionOutcome
            .BEHAVIORALLY_UNCLASSIFIABLE
        )
    )

    assert (
        result.targeted_changed
        is None
    )

    assert (
        result.sham_changed
        is None
    )


def test_invalid_target_action_is_unclassifiable() -> None:
    result = (
        MatchedReplayClassifier()
        .classify(
            execution(
                baseline=action("B"),
                targeted=(
                    invalid_action()
                ),
                sham=action("B"),
            )
        )
    )

    assert (
        result.outcome
        == (
            InterventionOutcome
            .BEHAVIORALLY_UNCLASSIFIABLE
        )
    )


def test_invalid_sham_action_is_unclassifiable() -> None:
    result = (
        MatchedReplayClassifier()
        .classify(
            execution(
                baseline=action("B"),
                targeted=action("A"),
                sham=(
                    invalid_action()
                ),
            )
        )
    )

    assert (
        result.outcome
        == (
            InterventionOutcome
            .BEHAVIORALLY_UNCLASSIFIABLE
        )
    )


def test_source_mismatch_is_rejected() -> None:
    with pytest.raises(
        ValueError
    ):
        (
            MatchedReplayClassifier()
            .classify(
                execution(
                    baseline=action("B"),
                    targeted=action("A"),
                    sham=action("B"),
                    targeted_source=(
                        SourceClass.T
                    ),
                    sham_source=(
                        SourceClass.H
                    ),
                )
            )
        )