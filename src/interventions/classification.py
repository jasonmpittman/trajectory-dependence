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
from enum import Enum
from typing import Any

from ..logging import (
    SourceClass,
)
from .replay import (
    MatchedReplayExecution,
)


class InterventionOutcome(str, Enum):
    """Matched targeted/sham causal classification."""

    POSITIVE = (
        "intervention_positive"
    )

    NEGATIVE = (
        "intervention_negative"
    )

    SHAM_UNSTABLE_AMBIGUOUS = (
        "sham_unstable_ambiguous"
    )

    BEHAVIORALLY_UNCLASSIFIABLE = (
        "behaviorally_unclassifiable"
    )


@dataclass(frozen=True)
class InterventionClassification:
    """Machine-readable result of one matched replay comparison."""

    source: SourceClass

    outcome: InterventionOutcome

    baseline_action: (
        dict[str, Any] | None
    )

    targeted_action: (
        dict[str, Any] | None
    )

    sham_action: (
        dict[str, Any] | None
    )

    targeted_changed: bool | None
    sham_changed: bool | None

    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": (
                self.source.value
            ),
            "outcome": (
                self.outcome.value
            ),
            "baseline_action": deepcopy(
                self.baseline_action
            ),
            "targeted_action": deepcopy(
                self.targeted_action
            ),
            "sham_action": deepcopy(
                self.sham_action
            ),
            "targeted_changed": (
                self.targeted_changed
            ),
            "sham_changed": (
                self.sham_changed
            ),
            "reason": self.reason,
        }


class MatchedReplayClassifier:
    """
    Compare baseline, targeted, and sham normalized focal actions.

    Raw generated text is never used for causal classification.
    """

    def classify(
        self,
        execution: MatchedReplayExecution,
    ) -> InterventionClassification:
        source = (
            execution
            .targeted
            .application
            .spec
            .source
        )

        sham_source = (
            execution
            .sham
            .application
            .spec
            .source
        )

        if source != sham_source:
            raise ValueError(
                "Targeted and sham replay sources do not match."
            )

        baseline = self._normalized_payload(
            execution.baseline_action
        )

        targeted = self._normalized_payload(
            execution
            .targeted
            .focal_action
        )

        sham = self._normalized_payload(
            execution
            .sham
            .focal_action
        )

        if (
            baseline is None
            or targeted is None
            or sham is None
        ):
            return InterventionClassification(
                source=source,
                outcome=(
                    InterventionOutcome
                    .BEHAVIORALLY_UNCLASSIFIABLE
                ),
                baseline_action=baseline,
                targeted_action=targeted,
                sham_action=sham,
                targeted_changed=None,
                sham_changed=None,
                reason=(
                    "At least one focal action could not "
                    "be mapped to a valid normalized action."
                ),
            )

        targeted_changed = (
            targeted
            != baseline
        )

        sham_changed = (
            sham
            != baseline
        )

        # Sham instability takes precedence because the causal attribution
        # is ambiguous whenever the source-matched control changes behavior.
        if sham_changed:
            return InterventionClassification(
                source=source,
                outcome=(
                    InterventionOutcome
                    .SHAM_UNSTABLE_AMBIGUOUS
                ),
                baseline_action=baseline,
                targeted_action=targeted,
                sham_action=sham,
                targeted_changed=(
                    targeted_changed
                ),
                sham_changed=True,
                reason=(
                    "The source-matched sham changed "
                    "normalized behavior."
                ),
            )

        if targeted_changed:
            return InterventionClassification(
                source=source,
                outcome=(
                    InterventionOutcome
                    .POSITIVE
                ),
                baseline_action=baseline,
                targeted_action=targeted,
                sham_action=sham,
                targeted_changed=True,
                sham_changed=False,
                reason=(
                    "Targeted intervention changed "
                    "normalized behavior while the "
                    "source-matched sham remained stable."
                ),
            )

        return InterventionClassification(
            source=source,
            outcome=(
                InterventionOutcome
                .NEGATIVE
            ),
            baseline_action=baseline,
            targeted_action=targeted,
            sham_action=sham,
            targeted_changed=False,
            sham_changed=False,
            reason=(
                "Neither targeted nor source-matched sham "
                "changed normalized behavior."
            ),
        )

    @staticmethod
    def _normalized_payload(
        action,
    ) -> dict[str, Any] | None:
        if (
            not action.valid
            or action.parse_result
            is None
        ):
            return None

        return deepcopy(
            action
            .parse_result
            .normalized_action
            .payload
        )