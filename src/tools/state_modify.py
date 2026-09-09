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


class StateModifyTool(Tool):
    """
    Deterministically produce a modified copy of structured state.

    Parent paths must already exist. The final key may either exist or be
    newly created.
    """

    name = "state_modify"

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

        if "value" not in request:
            raise ToolExecutionError(
                tool_name=self.name,
                code="missing_value",
                message="State modification requires a 'value'.",
            )

        if not isinstance(state, Mapping):
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_state",
                message="State modification requires mapping state.",
            )

        updated_state = deepcopy(dict(state))
        new_value = deepcopy(request["value"])

        current: Any = updated_state

        for key in path[:-1]:
            if not isinstance(current, dict) or key not in current:
                raise ToolExecutionError(
                    tool_name=self.name,
                    code="path_not_found",
                    message=f"Parent state path does not exist: {path!r}",
                )

            current = current[key]

        if not isinstance(current, dict):
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_parent",
                message=f"Parent at path {path[:-1]!r} is not a mapping.",
            )

        final_key = path[-1]

        previous_exists = final_key in current
        previous_value = (
            deepcopy(current[final_key])
            if previous_exists
            else None
        )

        state_changed = (
            not previous_exists
            or previous_value != new_value
        )

        current[final_key] = new_value

        raw_result = {
            "path": list(path),
            "previous_exists": previous_exists,
            "previous_value": previous_value,
            "new_value": deepcopy(new_value),
        }

        normalized_result = {
            "path": list(path),
            "value": deepcopy(new_value),
        }

        return ToolResult(
            tool_name=self.name,
            request=deepcopy(dict(request)),
            raw_result=raw_result,
            normalized_result=normalized_result,
            state_changed=state_changed,
            updated_state=updated_state,
        )

    @staticmethod
    def _valid_path(path: Any) -> bool:
        return (
            isinstance(path, list)
            and len(path) > 0
            and all(isinstance(item, str) and item for item in path)
        )