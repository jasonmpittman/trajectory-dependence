__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import hashlib
import json
from pathlib import Path

import pytest

from src.model import (
    ChatMessage,
    InferenceConfig,
    ModelAdapter,
    ModelGenerationResult,
    ModelInputRecord,
    ModelRuntimeMetadata,
)
from src.tasks import (
    SupplementalDiagnosticError,
    SupplementalDiagnosticRunner,
)


VALID_MEMORY_EXPLANATION = """
{
  "causal_sources": ["P", "M"],
  "justification": {
    "P": "The task defines the route-selection rule.",
    "T": null,
    "H": null,
    "M": "Persistent memory contains the preferred route.",
    "E": null,
    "Other": null
  }
}
"""


VALID_ENVIRONMENT_EXPLANATION = """
{
  "causal_sources": ["P", "E"],
  "justification": {
    "P": "The task defines the route-selection rule.",
    "T": null,
    "H": null,
    "M": null,
    "E": "The current environment reports the door as opened.",
    "Other": null
  }
}
"""


def sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def sha256_tokens(
    values,
) -> str:
    return sha256_text(
        json.dumps(
            list(values),
            separators=(",", ":"),
        )
    )


class SequentialSupplementalAdapter(
    ModelAdapter
):
    def __init__(
        self,
        outputs,
    ):
        self.outputs = list(
            outputs
        )

    @property
    def is_loaded(self):
        return True

    def load(self):
        pass

    def runtime_metadata(
        self,
    ):
        return ModelRuntimeMetadata(
            model_identifier="synthetic",
            model_revision="synthetic",
            model_checksum="synthetic",
            quantization="test",
            backend="fake",
            backend_version="1",
            mlx_version="test",
            python_version="test",
            platform="test",
            model_path_env=(
                "TRAJECTORY_MODEL_PATH"
            ),
            device_info={},
        )

    def prepare_input(
        self,
        model_visible_text,
    ):
        rendered = (
            "<user>"
            + model_visible_text
            + "</user>"
        )

        token_ids = (
            1,
            2,
            3,
        )

        return ModelInputRecord(
            model_visible_text=(
                model_visible_text
            ),
            messages=(
                ChatMessage(
                    role="user",
                    content=(
                        model_visible_text
                    ),
                ),
            ),
            rendered_prompt=(
                rendered
            ),
            rendered_prompt_sha256=(
                sha256_text(
                    rendered
                )
            ),
            prompt_token_ids=(
                token_ids
            ),
            prompt_token_ids_sha256=(
                sha256_tokens(
                    token_ids
                )
            ),
            prompt_token_count=3,
            thinking_mode=False,
            chat_template_arguments={
                "enable_thinking": False,
            },
        )

    def generate(
        self,
        *,
        input_record,
        config,
    ):
        if not self.outputs:
            raise RuntimeError(
                "No synthetic generation remains."
            )

        text = self.outputs.pop(
            0
        )

        generated_ids = (
            10,
            11,
        )

        return ModelGenerationResult(
            raw_text=text,
            raw_text_sha256=(
                sha256_text(
                    text
                )
            ),
            generated_token_ids=(
                generated_ids
            ),
            generated_token_ids_sha256=(
                sha256_tokens(
                    generated_ids
                )
            ),
            finish_reason="stop",
            prompt_tokens=3,
            generation_tokens=2,
            prompt_tps=None,
            generation_tps=None,
            peak_memory_gb=None,
            input_record=(
                input_record
            ),
            inference_config=(
                config
            ),
            runtime_metadata=(
                self.runtime_metadata()
            ),
        )


def make_adapter():
    return SequentialSupplementalAdapter(
        [
            (
                '{"action":"select_route",'
                '"value":"B"}'
            ),
            VALID_MEMORY_EXPLANATION,
            (
                '{"action":"select_route",'
                '"value":"B"}'
            ),
            VALID_ENVIRONMENT_EXPLANATION,
        ]
    )


def execute(
    tmp_path,
):
    return (
        SupplementalDiagnosticRunner()
        .execute(
            model_adapter=(
                make_adapter()
            ),
            action_config=(
                InferenceConfig(
                    max_tokens=64,
                    seed=101,
                )
            ),
            explanation_config=(
                InferenceConfig(
                    max_tokens=256,
                    seed=102,
                )
            ),
            output_root=(
                tmp_path
                / "supplemental"
            ),
        )
    )


def test_both_diagnostics_pass(
    tmp_path,
):
    result = execute(
        tmp_path
    )

    assert (
        result.passed
        is True
    )

    assert len(
        result.diagnostics
    ) == 2

    assert all(
        item.passed
        for item
        in result.diagnostics
    )


def test_supplemental_results_are_excluded_from_h1_h2(
    tmp_path,
):
    result = execute(
        tmp_path
    )

    assert (
        result.post_pilot
        is True
    )

    assert (
        result.included_in_h1
        is False
    )

    assert (
        result.included_in_h2
        is False
    )

    for diagnostic in (
        result.diagnostics
    ):
        assert (
            diagnostic.post_pilot
            is True
        )

        assert (
            diagnostic.included_in_h1
            is False
        )

        assert (
            diagnostic.included_in_h2
            is False
        )


def test_memory_lifecycle_checks_pass(
    tmp_path,
):
    result = execute(
        tmp_path
    )

    memory = next(
        item
        for item
        in result.diagnostics
        if (
            item.diagnostic_id
            == "memory_write_read"
        )
    )

    assert (
        memory.task_success
        is True
    )

    assert (
        memory.checks[
            "memory_write_before_read"
        ]
        is True
    )

    assert (
        memory.checks[
            "record_sequence_matches_write"
        ]
        is True
    )

    assert (
        memory.checks[
            "written_record_equals_read_record"
        ]
        is True
    )

    assert (
        memory.checks[
            "memory_context_anchored_to_read"
        ]
        is True
    )

    assert (
        memory.checks[
            "memory_context_not_anchored_to_write"
        ]
        is True
    )

    assert (
        memory.checks[
            "snapshot_contains_memory_record"
        ]
        is True
    )

    assert (
        memory.checks[
            "no_technical_failure"
        ]
        is True
    )

    assert (
        memory.checks[
            "no_intervention_events"
        ]
        is True
    )


def test_environment_lifecycle_checks_pass(
    tmp_path,
):
    result = execute(
        tmp_path
    )

    environment = next(
        item
        for item
        in result.diagnostics
        if (
            item.diagnostic_id
            == "environment_transition_query"
        )
    )

    assert (
        environment.task_success
        is True
    )

    assert (
        environment.checks[
            "transition_before_query"
        ]
        is True
    )

    assert (
        environment.checks[
            "transition_changed_state"
        ]
        is True
    )

    assert (
        environment.checks[
            "transition_before_value_closed"
        ]
        is True
    )

    assert (
        environment.checks[
            "transition_after_value_opened"
        ]
        is True
    )

    assert (
        environment.checks[
            "queried_state_is_opened"
        ]
        is True
    )

    assert (
        environment.checks[
            "environment_context_anchored_to_query"
        ]
        is True
    )

    assert (
        environment.checks[
            "snapshot_contains_post_transition_state"
        ]
        is True
    )

    assert (
        environment.checks[
            "no_technical_failure"
        ]
        is True
    )

    assert (
        environment.checks[
            "no_intervention_events"
        ]
        is True
    )


def test_memory_event_stream_contains_write_before_read(
    tmp_path,
):
    result = execute(
        tmp_path
    )

    memory = next(
        item
        for item
        in result.diagnostics
        if (
            item.diagnostic_id
            == "memory_write_read"
        )
    )

    events = [
        json.loads(line)
        for line in Path(
            memory.events_path
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    types = [
        event["event_type"]
        for event in events
    ]

    assert (
        "memory_write"
        in types
    )

    assert (
        "memory_read"
        in types
    )

    assert (
        types.index(
            "memory_write"
        )
        < types.index(
            "memory_read"
        )
    )

    assert (
        "intervention"
        not in types
    )

    assert (
        "sham_intervention"
        not in types
    )


def test_environment_event_stream_contains_transition_before_query(
    tmp_path,
):
    result = execute(
        tmp_path
    )

    environment = next(
        item
        for item
        in result.diagnostics
        if (
            item.diagnostic_id
            == "environment_transition_query"
        )
    )

    events = [
        json.loads(line)
        for line in Path(
            environment.events_path
        ).read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    types = [
        event["event_type"]
        for event in events
    ]

    assert (
        "environment_transition"
        in types
    )

    assert (
        "environment_query"
        in types
    )

    assert (
        types.index(
            "environment_transition"
        )
        < types.index(
            "environment_query"
        )
    )

    assert (
        "intervention"
        not in types
    )

    assert (
        "sham_intervention"
        not in types
    )


def test_output_bundle_contains_required_artifacts(
    tmp_path,
):
    result = execute(
        tmp_path
    )

    assert (
        result.passed
        is True
    )

    root = (
        tmp_path
        / "supplemental"
    )

    assert (
        root
        / "diagnostic_manifest.json"
    ).is_file()

    assert (
        root
        / "supplemental_summary.json"
    ).is_file()

    assert (
        root
        / "memory_write_read"
        / "memory.sqlite3"
    ).is_file()

    for directory in (
        "memory_write_read",
        "environment_transition_query",
    ):
        assert (
            root
            / directory
            / "baseline"
            / "events.jsonl"
        ).is_file()

        assert (
            root
            / directory
            / "pre_focal_snapshot.json"
        ).is_file()

        assert (
            root
            / directory
            / "run_summary.json"
        ).is_file()


def test_runner_refuses_existing_root(
    tmp_path,
):
    root = (
        tmp_path
        / "supplemental"
    )

    root.mkdir()

    with pytest.raises(
        SupplementalDiagnosticError
    ) as exc_info:
        (
            SupplementalDiagnosticRunner()
            .execute(
                model_adapter=(
                    make_adapter()
                ),
                action_config=(
                    InferenceConfig(
                        max_tokens=64,
                        seed=101,
                    )
                ),
                explanation_config=(
                    InferenceConfig(
                        max_tokens=256,
                        seed=102,
                    )
                ),
                output_root=root,
            )
        )

    assert (
        exc_info.value.code
        == "output_root_exists"
    )