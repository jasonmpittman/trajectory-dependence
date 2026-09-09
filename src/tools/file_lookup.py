__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from .base import Tool, ToolExecutionError, ToolResult


class SyntheticFileLookupTool(Tool):
    """
    Deterministic UTF-8 file lookup confined to an experiment-controlled
    document root.
    """

    name = "synthetic_file_lookup"

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    def execute(
        self,
        request: Mapping[str, Any],
        state: Any = None,
    ) -> ToolResult:
        relative_path = request.get("path")

        if not isinstance(relative_path, str) or not relative_path:
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_path",
                message="'path' must be a non-empty string.",
            )

        candidate = (self.root / relative_path).resolve()

        if candidate != self.root and self.root not in candidate.parents:
            raise ToolExecutionError(
                tool_name=self.name,
                code="path_escape",
                message="Requested path escapes the synthetic document root.",
            )

        if not candidate.is_file():
            raise ToolExecutionError(
                tool_name=self.name,
                code="file_not_found",
                message=f"Synthetic file does not exist: {relative_path}",
            )

        raw_bytes = candidate.read_bytes()

        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(
                tool_name=self.name,
                code="invalid_encoding",
                message="Synthetic files must contain valid UTF-8 text.",
            ) from exc

        digest = sha256(raw_bytes).hexdigest()

        raw_result = {
            "path": relative_path,
            "content": content,
            "sha256": digest,
        }

        normalized_result = {
            "path": relative_path,
            "content": content,
            "sha256": digest,
        }

        return ToolResult(
            tool_name=self.name,
            request=deepcopy(dict(request)),
            raw_result=raw_result,
            normalized_result=normalized_result,
            state_changed=False,
        )