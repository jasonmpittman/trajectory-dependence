__author__ = "Jason M. Pittman"
__date__ = "August 23, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import os

import pytest

from src.agent import (
    AgentRuntime,
    RunIdentity,
)
from src.logging import (
    Condition,
    EventType,
    RawEventWriter,
    TaskClassification,
    read_events,
)
from src.model import (
    InferenceConfig,
    MLXQwenAdapter,
)


pytestmark = pytest.mark.skipif(
    os.environ.get(
        "RUN_MODEL_INTEGRATION"
    )
    != "1",
    reason=(
        "Set RUN_MODEL_INTEGRATION=1 "
        "to run local 27B model tests."
    ),
)


def test_real_qwen_logged_runtime_invocation(
    tmp_path,
) -> None:
    adapter = MLXQwenAdapter(
        model_identifier=(
            "mlx-community/"
            "Qwen3.5-27B-8bit"
        ),
        model_revision=None,
        model_checksum=None,
        quantization="8bit",
    )

    adapter.load()

    writer = RawEventWriter(
        tmp_path
        / "events.jsonl"
    )

    runtime = AgentRuntime(
        identity=RunIdentity(
            experiment_id="integration_test",
            run_id="run_001",
            task_id="runtime_smoke",
            task_classification=(
                TaskClassification.DIAGNOSTIC
            ),
            condition=Condition.C0,
            repetition_id=1,
            seed=12345,
        ),
        model_adapter=adapter,
        event_writer=writer,
    )

    runtime.start()

    runtime.begin_task(
        "Respond with exactly the following "
        "text and nothing else: "
        "RUNTIME_SMOKE_OK"
    )

    result = runtime.invoke_model(
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        )
    )

    assert (
        result.generation.raw_text.strip()
        == "RUNTIME_SMOKE_OK"
    )

    assert [
        event.event_type
        for event in runtime.events
    ] == [
        EventType.SYSTEM,
        EventType.TASK,
        EventType.CONTEXT_BUILD,
        EventType.PROVENANCE_AUDIT,
        EventType.GENERATION,
    ]

    loaded = read_events(
        writer.path
    )

    assert len(
        loaded
    ) == 5

    context_event = loaded[2]

    assert (
        context_event["event_type"]
        == "context_build"
    )

    assert (
        "rendered_prompt"
        in context_event[
            "raw_payload"
        ][
            "model_input"
        ]
    )

    assert (
        "prompt_token_ids"
        in context_event[
            "raw_payload"
        ][
            "model_input"
        ]
    )

    generation_event = loaded[4]

    assert (
        generation_event[
            "event_type"
        ]
        == "generation"
    )

    assert (
        generation_event[
            "normalized_payload"
        ].strip()
        == "RUNTIME_SMOKE_OK"
    )