__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .base import Tool, ToolExecutionError, ToolResult


class CalculatorTool(Tool):
    """Small deterministic arithmetic tool."""

    name = "calculator"

    def execute(
        self,
        request: Mapping[str, Any],
        state: Any = None,
    ) -> ToolResult:
        operation = request.get("operation")
        operands = request.get("operands")

        if operation not in {"add", "subtract", "multiply", "divide"}:
            raise ToolExecutionError(
                tool_name=self.name,
                code="unsupported_operation",
                message=f"Unsupported calculator operation: {operation!r}",
            )

        if not isinstance(operands, list):
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_operands",
                message="'operands' must be a list.",
            )

        values = [self._to_decimal(value) for value in operands]

        if operation in {"add", "multiply"} and len(values) < 2:
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_operand_count",
                message=f"{operation} requires at least two operands.",
            )

        if operation in {"subtract", "divide"} and len(values) != 2:
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_operand_count",
                message=f"{operation} requires exactly two operands.",
            )

        if operation == "add":
            result = sum(values, Decimal("0"))

        elif operation == "subtract":
            result = values[0] - values[1]

        elif operation == "multiply":
            result = Decimal("1")

            for value in values:
                result *= value

        else:
            if values[1] == 0:
                raise ToolExecutionError(
                    tool_name=self.name,
                    code="division_by_zero",
                    message="Division by zero is not permitted.",
                )

            result = values[0] / values[1]

        normalized_value = self._normalize_decimal(result)

        normalized_result = {
            "value": normalized_value,
        }

        return ToolResult(
            tool_name=self.name,
            request=deepcopy(dict(request)),
            raw_result=deepcopy(normalized_result),
            normalized_result=deepcopy(normalized_result),
            state_changed=False,
        )

    def _to_decimal(self, value: Any) -> Decimal:
        if isinstance(value, bool):
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_operand",
                message="Boolean values are not valid calculator operands.",
            )

        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_operand",
                message=f"Invalid calculator operand: {value!r}",
            )

    @staticmethod
    def _normalize_decimal(value: Decimal) -> str:
        normalized = value.normalize()

        if normalized == 0:
            return "0"

        if normalized == normalized.to_integral():
            return format(normalized.quantize(Decimal("1")), "f")

        return format(normalized, "f")