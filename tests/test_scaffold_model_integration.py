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
    ScaffoldController,
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
from src.tools import (
    KeyValueLookupTool,
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


def test_real_qwen_uses_c1_tool_state(
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

    runtime = AgentRuntime(
        identity=RunIdentity(
            experiment_id="integration",
            run_id="run_001",
            task_id="tool_state_smoke",
            task_classification=(
                TaskClassification.DIAGNOSTIC
            ),
            condition=Condition.C1,
            repetition_id=1,
            seed=12345,
        ),
        model_adapter=adapter,
        event_writer=RawEventWriter(
            tmp_path
            / "events.jsonl"
        ),
    )

    schema = FocalActionSchema(
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
                ),
            ),
        ),
    )

    runtime.start()

    runtime.begin_focal_task(
        task_prompt=(
            "A tool will provide the current threshold. "
            "If the tool reports threshold 7, select route B. "
            "For any other threshold, select route A."
        ),
        action_schema=schema,
    )

    scaffold = ScaffoldController(
        runtime=runtime
    )

    interaction = (
        scaffold.execute_tool(
            tool=KeyValueLookupTool(),
            request={
                "key": "threshold",
            },
            state={
                "threshold": 7,
            },
        )
    )

    result = (
        scaffold.invoke_focal_action(
            config=InferenceConfig(
                max_tokens=64,
                seed=12345,
            ),
            tool_elements=(
                interaction.tool_element,
            ),
        )
    )

    assert result.valid is True

    assert (
        result.parse_result
        is not None
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