__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from copy import deepcopy
from typing import Any, Mapping

from .base import Tool, ToolExecutionError, ToolResult


class EnvironmentQueryTool(Tool):
    """Read a value from deterministic structured environmental state."""

    name = "environment_query"

    def execute(
        self,
        request: Mapping[str, Any],
        state: Any = None,
    ) -> ToolResult:
        path = request.get("path")

        if not self._valid_path(path):
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_path",
                message="'path' must be a non-empty list of string keys.",
            )

        if not isinstance(state, Mapping):
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_state",
                message="Environmental state must be a mapping.",
            )

        current: Any = state
        found = True

        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                found = False
                current = None
                break

            current = current[key]

        result = {
            "path": list(path),
            "found": found,
            "value": deepcopy(current) if found else None,
        }

        return ToolResult(
            tool_name=self.name,
            request=deepcopy(dict(request)),
            raw_result=deepcopy(result),
            normalized_result=deepcopy(result),
            state_changed=False,
        )

    @staticmethod
    def _valid_path(path: Any) -> bool:
        return (
            isinstance(path, list)
            and len(path) > 0
            and all(isinstance(item, str) and item for item in path)
        )