__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from pathlib import Path
from typing import Iterable

from .schemas import ExperimentEvent


class RawEventWriter:
    """
    Append-only JSONL writer for raw experimental events.

    This writer does not overwrite an existing event stream.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        create_parents: bool = True,
    ) -> None:
        self.path = Path(path)

        if create_parents:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """
        Initialize a new raw event stream.

        Raises FileExistsError if the target already exists. This prevents
        accidental destruction or replacement of experimental evidence.
        """

        if self.path.exists():
            raise FileExistsError(
                f"Raw event file already exists and will not be overwritten: "
                f"{self.path}"
            )

        self.path.touch(exist_ok=False)

    def append(self, event: ExperimentEvent) -> None:
        """
        Append one event as exactly one JSON object followed by a newline.
        """

        if not self.path.exists():
            raise FileNotFoundError(
                f"Raw event stream has not been initialized: {self.path}"
            )

        serialized = json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")

    def append_many(self, events: Iterable[ExperimentEvent]) -> None:
        """Append multiple events in iteration order."""

        for event in events:
            self.append(event)


def read_events(path: Path | str) -> list[dict]:
    """
    Load an event stream from JSONL.

    This function is intended for testing, validation, and downstream
    reconstruction. It does not modify the underlying file.
    """

    event_path = Path(path)

    events: list[dict] = []

    with event_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {event_path} at line {line_number}"
                ) from exc

            events.append(event)

    return events