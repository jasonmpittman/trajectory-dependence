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
    RawEventWriter,
    TaskClassification,
)
from src.model import (
    InferenceConfig,
    MLXQwenAdapter,
)
from src.normalization import (
    ActionField,
    FocalActionSchema,
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


def test_real_qwen_structured_focal_action(
    tmp_path,
) -> None:
    """
    Verify that the real local Qwen3.5-27B model can complete a focal
    consequential decision using the strict structured-action contract.

    This test exercises:

    - external-drive model loading;
    - AgentRuntime task initialization;
    - deterministic focal-action prompting;
    - context construction;
    - context serialization;
    - provenance audit;
    - real model generation;
    - strict action parsing;
    - normalized ACTION creation.
    """

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

    runtime = AgentRuntime(
        identity=RunIdentity(
            experiment_id=(
                "focal_action_integration"
            ),
            run_id="run_001",
            task_id=(
                "focal_action_smoke"
            ),
            task_classification=(
                TaskClassification.DIAGNOSTIC
            ),
            condition=Condition.C0,
            repetition_id=1,
            seed=12345,
        ),
        model_adapter=adapter,
        event_writer=RawEventWriter(
            tmp_path
            / "events.jsonl"
        ),
    )

    action_schema = FocalActionSchema(
        focal_decision_id="final_route",
        fields=(
            ActionField(
                name="action",
                json_type="string",
                allowed_values=(
                    "select_route",
                ),
            ),
            ActionField(
                name="value",
                json_type="string",
                allowed_values=(
                    "A",
                    "B",
                    "C",
                ),
            ),
        ),
    )

    runtime.start()

    runtime.begin_focal_task(
        task_prompt=(
            "Select route B."
        ),
        action_schema=(
            action_schema
        ),
    )

    result = (
        runtime.invoke_focal_action(
            config=InferenceConfig(
                max_tokens=64,
                seed=12345,
            )
        )
    )

    assert result.valid is True

    assert (
        result.parse_result
        is not None
    )

    assert (
        result.parse_error
        is None
    )

    assert (
        result.parse_result
        .normalized_action
        .payload
        == {
            "action": "select_route",
            "value": "B",
        }
    )

    assert (
        result.parse_result
        .normalized_action
        .focal_decision_id
        == "final_route"
    )

    action_event = (
        runtime.events[-1]
    )

    assert (
        action_event.event_type.value
        == "action"
    )

    assert (
        action_event.focal_decision_id
        == "final_route"
    )

    assert (
        action_event.metadata[
            "parse_valid"
        ]
        is True
    )

    assert (
        action_event
        .normalized_payload[
            "payload"
        ]
        == {
            "action": "select_route",
            "value": "B",
        }
    )