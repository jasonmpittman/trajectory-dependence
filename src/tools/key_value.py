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


class KeyValueLookupTool(Tool):
    """Deterministic lookup over a supplied key-value mapping."""

    name = "key_value_lookup"

    def execute(
        self,
        request: Mapping[str, Any],
        state: Any = None,
    ) -> ToolResult:
        key = request.get("key")

        if not isinstance(key, str) or not key:
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_key",
                message="'key' must be a non-empty string.",
            )

        if not isinstance(state, Mapping):
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_state",
                message="Key-value lookup state must be a mapping.",
            )

        found = key in state

        result = {
            "key": key,
            "found": found,
            "value": deepcopy(state[key]) if found else None,
        }

        return ToolResult(
            tool_name=self.name,
            request=deepcopy(dict(request)),
            raw_result=deepcopy(result),
            normalized_result=deepcopy(result),
            state_changed=False,
        )