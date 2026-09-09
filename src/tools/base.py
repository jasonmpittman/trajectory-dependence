__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


class ToolExecutionError(RuntimeError):
    """
    Deterministic tool execution failure.

    This exception represents an invalid request or a defined tool-level
    failure. It does not itself determine whether the experiment later
    classifies the event as a technical failure or task-semantic outcome.
    """

    def __init__(
        self,
        *,
        tool_name: str,
        code: str,
        message: str,
    ) -> None:
        super().__init__(message)

        self.tool_name = tool_name
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {
            "tool_name": self.tool_name,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ToolResult:
    """
    Canonical return value from a deterministic experimental tool.

    `updated_state` is populated only by tools that intentionally produce
    a state transition. Tools should not silently mutate state supplied
    by the caller.
    """

    tool_name: str
    request: dict[str, Any]

    raw_result: Any
    normalized_result: Any

    state_changed: bool = False
    updated_state: Any = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tool(ABC):
    """Minimal interface implemented by all experimental tools."""

    name: str

    @abstractmethod
    def execute(
        self,
        request: Mapping[str, Any],
        state: Any = None,
    ) -> ToolResult:
        """
        Execute a deterministic tool request.

        Implementations must not:
        - access uncontrolled external services;
        - perform hidden retries;
        - modify caller-owned state in place;
        - introduce stochastic behavior.
        """

        raise NotImplementedError