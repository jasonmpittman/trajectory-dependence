__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
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


def test_real_qwen_structured_self_explanation(
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
            task_id=(
                "self_explanation_smoke"
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
            "Select route B."
        ),
        action_schema=schema,
    )

    action = runtime.invoke_focal_action(
        config=InferenceConfig(
            max_tokens=64,
            seed=12345,
        )
    )

    assert action.valid is True

    explanation = (
        runtime.elicit_self_explanation(
            focal_action=action,
            config=InferenceConfig(
                max_tokens=256,
                seed=12345,
            ),
        )
    )

    assert explanation.valid is True

    assert (
        explanation.parse_result
        is not None
    )

    assert all(
        source.value
        in {
            "P",
            "T",
            "H",
            "M",
            "E",
            "Other",
        }
        for source in (
            explanation
            .parse_result
            .reported_sources
        )
    )

    assert (
        runtime.events[-2].event_type
        == EventType.GENERATION
    )

    assert (
        runtime.events[-1].event_type
        == EventType.EXPLANATION
    )