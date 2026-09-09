__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from dataclasses import dataclass

from ..snapshots import (
    PreFocalSnapshot,
)
from .engine import (
    InterventionEngine,
)
from .schemas import (
    InterventionApplication,
    InterventionKind,
    InterventionPair,
)


class InterventionValidationError(ValueError):
    """Defined failure validating a targeted/sham intervention pair."""

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
class ValidatedInterventionPair:
    """
    Targeted and sham applications independently derived from the same
    baseline snapshot.
    """

    pair: InterventionPair

    targeted_application: (
        InterventionApplication
    )

    sham_application: (
        InterventionApplication
    )

    def to_dict(self) -> dict:
        return {
            "pair": self.pair.to_dict(),
            "targeted_application": (
                self.targeted_application
                .to_dict()
            ),
            "sham_application": (
                self.sham_application
                .to_dict()
            ),
        }


class InterventionPairValidator:
    """Validate pair structure and baseline applicability."""

    def __init__(
        self,
        engine: InterventionEngine
        | None = None,
    ) -> None:
        self.engine = (
            engine
            or InterventionEngine()
        )

    def validate(
        self,
        *,
        snapshot: PreFocalSnapshot,
        pair: InterventionPair,
    ) -> ValidatedInterventionPair:
        self._validate_pair_structure(
            pair
        )

        targeted_application = (
            self.engine.apply(
                snapshot=snapshot,
                spec=pair.targeted,
            )
        )

        sham_application = (
            self.engine.apply(
                snapshot=snapshot,
                spec=pair.sham,
            )
        )

        if (
            targeted_application
            .baseline_state_checksum
            != sham_application
            .baseline_state_checksum
        ):
            raise InterventionValidationError(
                code="pair_baseline_mismatch",
                message=(
                    "Targeted and sham applications were "
                    "not derived from identical baseline state."
                ),
            )

        if (
            targeted_application
            .baseline_snapshot_id
            != sham_application
            .baseline_snapshot_id
        ):
            raise InterventionValidationError(
                code="pair_snapshot_mismatch",
                message=(
                    "Targeted and sham applications reference "
                    "different baseline snapshots."
                ),
            )

        return ValidatedInterventionPair(
            pair=pair,
            targeted_application=(
                targeted_application
            ),
            sham_application=(
                sham_application
            ),
        )

    @staticmethod
    def _validate_pair_structure(
        pair: InterventionPair,
    ) -> None:
        if (
            not isinstance(
                pair.pair_id,
                str,
            )
            or not pair.pair_id
        ):
            raise InterventionValidationError(
                code="invalid_pair_id",
                message=(
                    "Intervention pair_id must be "
                    "a non-empty string."
                ),
            )

        if (
            pair.targeted.kind
            != InterventionKind.TARGETED
        ):
            raise InterventionValidationError(
                code="invalid_targeted_kind",
                message=(
                    "Pair.targeted must have kind=targeted."
                ),
            )

        if (
            pair.sham.kind
            != InterventionKind.SHAM
        ):
            raise InterventionValidationError(
                code="invalid_sham_kind",
                message=(
                    "Pair.sham must have kind=sham."
                ),
            )

        if (
            pair.targeted.pair_id
            != pair.pair_id
            or pair.sham.pair_id
            != pair.pair_id
        ):
            raise InterventionValidationError(
                code="pair_id_mismatch",
                message=(
                    "Pair and intervention pair IDs "
                    "must match."
                ),
            )

        if (
            pair.targeted.source
            != pair.sham.source
        ):
            raise InterventionValidationError(
                code="sham_source_mismatch",
                message=(
                    "Sham intervention must manipulate "
                    "the same source class as targeted."
                ),
            )

        if (
            pair.targeted.intervention_id
            == pair.sham.intervention_id
        ):
            raise InterventionValidationError(
                code="duplicate_intervention_id",
                message=(
                    "Targeted and sham intervention IDs "
                    "must differ."
                ),
            )

        if (
            pair.targeted.to_dict()
            == pair.sham.to_dict()
        ):
            raise InterventionValidationError(
                code="sham_matches_targeted",
                message=(
                    "Sham intervention may not be "
                    "identical to the targeted intervention."
                ),
            )