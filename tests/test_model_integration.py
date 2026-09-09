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


def test_real_qwen_smoke_generation() -> None:
    adapter = MLXQwenAdapter(
        model_identifier=(
            "mlx-community/"
            "Qwen3.5-27B-8bit"
        ),

        # These may remain None in this test until the manifest
        # values are populated.
        model_revision=None,
        model_checksum=None,

        quantization="8bit",
    )

    adapter.load()

    input_record = (
        adapter.prepare_input(
            "Respond with exactly the following "
            "text and nothing else: "
            "MODEL_ADAPTER_OK"
        )
    )

    result = adapter.generate(
        input_record=input_record,
        config=InferenceConfig(
            max_tokens=32,
            seed=12345,
        ),
    )

    assert (
        result.raw_text.strip()
        == "MODEL_ADAPTER_OK"
    )

    assert (
        result.input_record
        .thinking_mode
        is False
    )

    assert (
        result.inference_config
        .decoding_strategy
        == "greedy_argmax"
    )

    assert (
        result.runtime_metadata.backend
        == "mlx-lm"
    )