__author__ = "Jason M. Pittman"
__date__ = "August 27, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class FinalTaskMatrixError(ValueError):
    """Defined failure validating the frozen final-task design matrix."""

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
class FinalTaskMatrixValidation:
    matrix_id: str

    total_tasks: int
    diagnostic_tasks: int
    integration_tasks: int

    family_counts: dict[str, int]

    real_model_candidate_screening_allowed: bool

    valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_id": self.matrix_id,
            "total_tasks": self.total_tasks,
            "diagnostic_tasks": self.diagnostic_tasks,
            "integration_tasks": self.integration_tasks,
            "family_counts": deepcopy(
                self.family_counts
            ),
            "real_model_candidate_screening_allowed": (
                self.real_model_candidate_screening_allowed
            ),
            "valid": self.valid,
        }


class FinalTaskMatrixValidator:
    REQUIRED_FAMILIES = frozenset(
        {
            "memory_dependence",
            "tool_dependence",
            "history_dependence",
            "environmental_state_dependence",
            "conflict_integration",
            "multi_source_dependence",
        }
    )

    def validate(
        self,
        matrix: Mapping[str, Any],
    ) -> FinalTaskMatrixValidation:
        matrix_id = self._required_string(
            matrix,
            "matrix_id",
        )

        families = matrix.get(
            "families"
        )

        if not isinstance(
            families,
            Mapping,
        ):
            raise FinalTaskMatrixError(
                code="families_missing",
                message=(
                    "Final task matrix must contain "
                    "a families object."
                ),
            )

        observed_families = frozenset(
            families.keys()
        )

        if observed_families != self.REQUIRED_FAMILIES:
            raise FinalTaskMatrixError(
                code="family_set_mismatch",
                message=(
                    "Final task matrix family set "
                    "does not match the six frozen families."
                ),
            )

        task_ids: set[str] = set()

        diagnostic_count = 0
        integration_count = 0

        family_counts: dict[str, int] = {}

        minimum_per_family = self._required_int(
            matrix,
            "minimum_tasks_per_family",
        )

        for family_name in sorted(
            self.REQUIRED_FAMILIES
        ):
            family = families[
                family_name
            ]

            if not isinstance(
                family,
                Mapping,
            ):
                raise FinalTaskMatrixError(
                    code="invalid_family_definition",
                    message=(
                        f"Family {family_name!r} "
                        "must be an object."
                    ),
                )

            diagnostic = self._task_ids(
                family,
                "diagnostic",
                family_name,
            )

            integration = self._task_ids(
                family,
                "integration",
                family_name,
            )

            family_total = (
                len(diagnostic)
                + len(integration)
            )

            if family_total < minimum_per_family:
                raise FinalTaskMatrixError(
                    code="family_underrepresented",
                    message=(
                        f"Family {family_name!r} "
                        f"contains only {family_total} tasks; "
                        f"minimum is {minimum_per_family}."
                    ),
                )

            for task_id in (
                diagnostic
                + integration
            ):
                if task_id in task_ids:
                    raise FinalTaskMatrixError(
                        code="duplicate_task_id",
                        message=(
                            f"Task ID {task_id!r} "
                            "appears more than once."
                        ),
                    )

                task_ids.add(
                    task_id
                )

            diagnostic_count += len(
                diagnostic
            )

            integration_count += len(
                integration
            )

            family_counts[
                family_name
            ] = family_total

        total = len(
            task_ids
        )

        expected_total = self._required_int(
            matrix,
            "target_total_tasks",
        )

        expected_diagnostic = self._required_int(
            matrix,
            "target_diagnostic_tasks",
        )

        expected_integration = self._required_int(
            matrix,
            "target_integration_tasks",
        )

        if total != expected_total:
            raise FinalTaskMatrixError(
                code="total_task_count_mismatch",
                message=(
                    f"Matrix contains {total} tasks "
                    f"but target_total_tasks is "
                    f"{expected_total}."
                ),
            )

        if (
            diagnostic_count
            != expected_diagnostic
        ):
            raise FinalTaskMatrixError(
                code="diagnostic_count_mismatch",
                message=(
                    "Diagnostic task count does not "
                    "match frozen matrix target."
                ),
            )

        if (
            integration_count
            != expected_integration
        ):
            raise FinalTaskMatrixError(
                code="integration_count_mismatch",
                message=(
                    "Integration task count does not "
                    "match frozen matrix target."
                ),
            )

        if (
            integration_count
            < (
                total
                / 2
            )
        ):
            raise FinalTaskMatrixError(
                code="insufficient_integration_coverage",
                message=(
                    "At least half of final tasks "
                    "must be integration tasks."
                ),
            )

        screening_allowed = matrix.get(
            "real_model_candidate_screening_allowed"
        )

        if screening_allowed is not False:
            raise FinalTaskMatrixError(
                code="real_model_screening_not_prohibited",
                message=(
                    "Final task construction must explicitly "
                    "prohibit real-model candidate screening."
                ),
            )

        minimum_multisource = self._required_int(
            matrix,
            "minimum_structurally_multisource_tasks",
        )

        if minimum_multisource < 4:
            raise FinalTaskMatrixError(
                code="insufficient_multisource_design_floor",
                message=(
                    "At least four tasks must be "
                    "structurally capable of multi-source "
                    "causal dispersion."
                ),
            )

        return FinalTaskMatrixValidation(
            matrix_id=matrix_id,
            total_tasks=total,
            diagnostic_tasks=(
                diagnostic_count
            ),
            integration_tasks=(
                integration_count
            ),
            family_counts=(
                family_counts
            ),
            real_model_candidate_screening_allowed=(
                screening_allowed
            ),
            valid=True,
        )

    @staticmethod
    def load(
        path: str | Path,
    ) -> dict[str, Any]:
        value = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            value,
            dict,
        ):
            raise FinalTaskMatrixError(
                code="matrix_root_not_object",
                message=(
                    "Final task matrix root "
                    "must be a JSON object."
                ),
            )

        return value

    @staticmethod
    def _task_ids(
        family,
        field_name,
        family_name,
    ) -> tuple[str, ...]:
        values = family.get(
            field_name
        )

        if not isinstance(
            values,
            list,
        ):
            raise FinalTaskMatrixError(
                code="invalid_task_id_list",
                message=(
                    f"{family_name}.{field_name} "
                    "must be an array."
                ),
            )

        result = []

        for value in values:
            if (
                not isinstance(
                    value,
                    str,
                )
                or not value
            ):
                raise FinalTaskMatrixError(
                    code="invalid_task_id",
                    message=(
                        f"{family_name}.{field_name} "
                        "contains an invalid task ID."
                    ),
                )

            result.append(
                value
            )

        return tuple(
            result
        )

    @staticmethod
    def _required_string(
        value,
        field_name,
    ):
        result = value.get(
            field_name
        )

        if (
            not isinstance(
                result,
                str,
            )
            or not result
        ):
            raise FinalTaskMatrixError(
                code="missing_required_string",
                message=(
                    f"{field_name!r} must "
                    "be a non-empty string."
                ),
            )

        return result

    @staticmethod
    def _required_int(
        value,
        field_name,
    ):
        result = value.get(
            field_name
        )

        if (
            not isinstance(
                result,
                int,
            )
            or isinstance(
                result,
                bool,
            )
            or result < 0
        ):
            raise FinalTaskMatrixError(
                code="missing_required_integer",
                message=(
                    f"{field_name!r} must "
                    "be a non-negative integer."
                ),
            )

        return result