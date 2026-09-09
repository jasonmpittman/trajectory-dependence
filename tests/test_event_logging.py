__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.1"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json

import pytest

from src.logging import (
    Condition,
    EventFactory,
    EventType,
    RawEventWriter,
    TaskClassification,
    read_events,
)


def make_factory() -> EventFactory:
    return EventFactory(
        experiment_id="test_experiment",
        run_id="test_run_001",
        task_id="test_task_001",
        task_classification=TaskClassification.DIAGNOSTIC,
        condition=Condition.C1,
        repetition_id=1,
        seed=12345,
    )


def test_event_sequence_is_monotonic() -> None:
    factory = make_factory()

    first = factory.create(
        event_type=EventType.TASK,
        source_component="test",
        raw_payload={"prompt": "test"},
    )

    second = factory.create(
        event_type=EventType.GENERATION,
        source_component="test",
        raw_payload={"text": "response"},
    )

    third = factory.create(
        event_type=EventType.ACTION,
        source_component="test",
        raw_payload={"action": "choose"},
    )

    assert first.sequence_index == 0
    assert second.sequence_index == 1
    assert third.sequence_index == 2


def test_event_ids_are_unique() -> None:
    factory = make_factory()

    events = [
        factory.create(
            event_type=EventType.TASK,
            source_component="test",
            raw_payload={"index": index},
        )
        for index in range(100)
    ]

    event_ids = {event.event_id for event in events}

    assert len(event_ids) == 100


def test_raw_payload_survives_jsonl_round_trip(tmp_path) -> None:
    factory = make_factory()

    raw_payload = {
        "text": "hello",
        "nested": {
            "value": 7,
            "items": ["a", "b", "c"],
        },
        "flag": True,
    }

    event = factory.create(
        event_type=EventType.TOOL_RETURN,
        source_component="test_tool",
        raw_payload=raw_payload,
        normalized_payload={"value": 7},
    )

    event_path = tmp_path / "events.jsonl"

    writer = RawEventWriter(event_path)
    writer.initialize()
    writer.append(event)

    loaded = read_events(event_path)

    assert len(loaded) == 1
    assert loaded[0]["raw_payload"] == raw_payload
    assert loaded[0]["normalized_payload"] == {"value": 7}


def test_writer_refuses_to_overwrite_existing_stream(tmp_path) -> None:
    event_path = tmp_path / "events.jsonl"

    writer = RawEventWriter(event_path)
    writer.initialize()

    second_writer = RawEventWriter(event_path)

    with pytest.raises(FileExistsError):
        second_writer.initialize()


def test_each_jsonl_line_contains_one_complete_event(tmp_path) -> None:
    factory = make_factory()

    events = [
        factory.create(
            event_type=EventType.TASK,
            source_component="test",
            raw_payload={"index": index},
        )
        for index in range(3)
    ]

    event_path = tmp_path / "events.jsonl"

    writer = RawEventWriter(event_path)
    writer.initialize()
    writer.append_many(events)

    lines = event_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3

    parsed = [json.loads(line) for line in lines]

    assert [event["sequence_index"] for event in parsed] == [0, 1, 2]


def test_event_is_immutable() -> None:
    factory = make_factory()

    event = factory.create(
        event_type=EventType.TASK,
        source_component="test",
        raw_payload={"prompt": "test"},
    )

    with pytest.raises(Exception):
        event.sequence_index = 99