__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from src.logging import ContextElement, SourceClass


def test_prompt_context_element_preserves_source() -> None:
    element = ContextElement(
        source=SourceClass.P,
        source_id="prompt:test_task",
        content="Choose a route.",
    )

    serialized = element.to_dict()

    assert serialized["source"] == "P"
    assert serialized["source_id"] == "prompt:test_task"


def test_history_can_preserve_tool_origin() -> None:
    element = ContextElement(
        source=SourceClass.H,
        source_id="history:event_004",
        content="The lookup returned threshold 7.",
        origin_event_id="event_004",
        origin_source=SourceClass.T,
        created_sequence=4,
        retrieved_sequence=9,
    )

    serialized = element.to_dict()

    assert serialized["source"] == "H"
    assert serialized["origin_source"] == "T"
    assert serialized["origin_event_id"] == "event_004"


def test_memory_remains_memory_source() -> None:
    element = ContextElement(
        source=SourceClass.M,
        source_id="memory:mem_001",
        content={"threshold": 7},
    )

    serialized = element.to_dict()

    assert serialized["source"] == "M"


def test_environment_remains_environment_source() -> None:
    element = ContextElement(
        source=SourceClass.E,
        source_id="environment:door_1",
        content={"status": "locked"},
    )

    serialized = element.to_dict()

    assert serialized["source"] == "E"