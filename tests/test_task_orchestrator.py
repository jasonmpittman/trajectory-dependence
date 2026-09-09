__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
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

from src.interventions import (
    InterventionOutcome,
)
from src.logging import (
    Condition,
    EventType,
)
from src.model import (
    ChatMessage,
    InferenceConfig,
    ModelAdapter,
    ModelGenerationResult,
    ModelInputRecord,
    ModelRuntimeMetadata,
)
from src.tasks import (
    ExperimentalRunOrchestrator,
    TaskOrchestrationError,
    load_task_suite,
)


PILOT_PATH = (
    Path("config")
    / "tasks"
    / "pilot-tasks.json"
)


VALID_EXPLANATION_P = """
{
  "causal_sources": ["P"],
  "justification": {
    "P": "The task prompt defined the default decision.",
    "T": null,
    "H": null,
    "M": null,
    "E": null,
    "Other": null
  }
}
"""


VALID_EXPLANATION_PT = """
{
  "causal_sources": ["P", "T"],
  "justification": {
    "P": "The task prompt defined the decision rule.",
    "T": "The threshold tool output supplied the current value.",
    "H": null,
    "M": null,
    "E": null,
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
            list(
                values
            ),
            separators=(",", ":"),
        )
    )


class SequentialOrchestratorAdapter(
    ModelAdapter
):
    def __init__(
        self,
        outputs: list[str],
    ) -> None:
        self.outputs = list(
            outputs
        )

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def runtime_metadata(
        self,
    ) -> ModelRuntimeMetadata:
        return ModelRuntimeMetadata(
            model_identifier="fake",
            model_revision="fake",
            model_checksum="fake",
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
        model_visible_text: str,
    ) -> ModelInputRecord:
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
    ) -> ModelGenerationResult:
        if not self.outputs:
            raise RuntimeError(
                "No fake generation remains."
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
            input_record=input_record,
            inference_config=config,
            runtime_metadata=(
                self.runtime_metadata()
            ),
        )


def pilot_task(
    task_id: str,
):
    suite = load_task_suite(
        PILOT_PATH
    )

    return next(
        task
        for task
        in suite.tasks
        if task.task_id == task_id
    )


def action_config() -> InferenceConfig:
    return InferenceConfig(
        max_tokens=64,
        seed=12345,
    )


def explanation_config() -> InferenceConfig:
    return InferenceConfig(
        max_tokens=256,
        seed=12345,
    )


def read_json(
    path: str | Path,
):
    with Path(
        path
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def read_events(
    path: str | Path,
):
    values = []

    with Path(
        path
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if not line.strip():
                continue

            values.append(
                json.loads(
                    line
                )
            )

    return values


def test_c0_complete_run_has_no_interventions(
    tmp_path,
) -> None:
    task = pilot_task(
        "pilot_tool_001"
    )

    adapter = (
        SequentialOrchestratorAdapter(
            [
                (
                    '{"action":"select_route",'
                    '"value":"A"}'
                ),
                VALID_EXPLANATION_P,
            ]
        )
    )

    result = (
        ExperimentalRunOrchestrator()
        .execute(
            experiment_id="pilot",
            run_id=(
                "pilot_tool_001_C0_r001"
            ),
            task=task,
            condition=Condition.C0,
            repetition_id=1,
            model_adapter=adapter,
            action_config=(
                action_config()
            ),
            explanation_config=(
                explanation_config()
            ),
            output_directory=(
                tmp_path
                / "run"
            ),
        )
    )

    assert (
        result.behavior_valid
        is True
    )

    assert (
        result.task_success
        is True
    )

    assert (
        result.intervention_results
        == ()
    )


def test_c1_tool_run_executes_baseline_explanation_and_replay(
    tmp_path,
) -> None:
    task = pilot_task(
        "pilot_tool_001"
    )

    adapter = (
        SequentialOrchestratorAdapter(
            [
                (
                    '{"action":"select_route",'
                    '"value":"B"}'
                ),
                VALID_EXPLANATION_PT,
                (
                    '{"action":"select_route",'
                    '"value":"A"}'
                ),
                (
                    '{"action":"select_route",'
                    '"value":"B"}'
                ),
            ]
        )
    )

    result = (
        ExperimentalRunOrchestrator()
        .execute(
            experiment_id="pilot",
            run_id=(
                "pilot_tool_001_C1_r001"
            ),
            task=task,
            condition=Condition.C1,
            repetition_id=1,
            model_adapter=adapter,
            action_config=(
                action_config()
            ),
            explanation_config=(
                explanation_config()
            ),
            output_directory=(
                tmp_path
                / "run"
            ),
        )
    )

    assert (
        result.task_success
        is True
    )

    assert len(
        result.intervention_results
    ) == 1

    assert (
        result.intervention_results[0]
        .classification
        .outcome
        == InterventionOutcome.POSITIVE
    )


def test_snapshot_precedes_baseline_action(
    tmp_path,
) -> None:
    task = pilot_task(
        "pilot_tool_001"
    )

    adapter = (
        SequentialOrchestratorAdapter(
            [
                (
                    '{"action":"select_route",'
                    '"value":"A"}'
                ),
                VALID_EXPLANATION_P,
            ]
        )
    )

    result = (
        ExperimentalRunOrchestrator()
        .execute(
            experiment_id="pilot",
            run_id=(
                "pilot_tool_001_C0_r001"
            ),
            task=task,
            condition=Condition.C0,
            repetition_id=1,
            model_adapter=adapter,
            action_config=(
                action_config()
            ),
            explanation_config=(
                explanation_config()
            ),
            output_directory=(
                tmp_path
                / "run"
            ),
        )
    )

    events = read_events(
        result.baseline_events_path
    )

    action_event = next(
        event
        for event
        in events
        if (
            event[
                "event_type"
            ]
            == EventType.ACTION.value
        )
    )

    assert (
        result.snapshot
        .sequence_position
        < action_event[
            "sequence_index"
        ]
    )


def test_explanation_is_after_action_in_baseline_stream(
    tmp_path,
) -> None:
    task = pilot_task(
        "pilot_tool_001"
    )

    adapter = (
        SequentialOrchestratorAdapter(
            [
                (
                    '{"action":"select_route",'
                    '"value":"A"}'
                ),
                VALID_EXPLANATION_P,
            ]
        )
    )

    result = (
        ExperimentalRunOrchestrator()
        .execute(
            experiment_id="pilot",
            run_id=(
                "pilot_tool_001_C0_r001"
            ),
            task=task,
            condition=Condition.C0,
            repetition_id=1,
            model_adapter=adapter,
            action_config=(
                action_config()
            ),
            explanation_config=(
                explanation_config()
            ),
            output_directory=(
                tmp_path
                / "run"
            ),
        )
    )

    events = read_events(
        result.baseline_events_path
    )

    action_index = next(
        event[
            "sequence_index"
        ]
        for event
        in events
        if (
            event[
                "event_type"
            ]
            == EventType.ACTION.value
        )
    )

    explanation_index = next(
        event[
            "sequence_index"
        ]
        for event
        in events
        if (
            event[
                "event_type"
            ]
            == EventType.EXPLANATION.value
        )
    )

    assert (
        action_index
        < explanation_index
    )


def test_baseline_stream_contains_no_intervention_events(
    tmp_path,
) -> None:
    task = pilot_task(
        "pilot_tool_001"
    )

    adapter = (
        SequentialOrchestratorAdapter(
            [
                (
                    '{"action":"select_route",'
                    '"value":"B"}'
                ),
                VALID_EXPLANATION_PT,
                (
                    '{"action":"select_route",'
                    '"value":"A"}'
                ),
                (
                    '{"action":"select_route",'
                    '"value":"B"}'
                ),
            ]
        )
    )

    result = (
        ExperimentalRunOrchestrator()
        .execute(
            experiment_id="pilot",
            run_id=(
                "pilot_tool_001_C1_r001"
            ),
            task=task,
            condition=Condition.C1,
            repetition_id=1,
            model_adapter=adapter,
            action_config=(
                action_config()
            ),
            explanation_config=(
                explanation_config()
            ),
            output_directory=(
                tmp_path
                / "run"
            ),
        )
    )

    event_types = {
        event[
            "event_type"
        ]
        for event
        in read_events(
            result.baseline_events_path
        )
    }

    assert (
        EventType.INTERVENTION.value
        not in event_types
    )

    assert (
        EventType.SHAM_INTERVENTION.value
        not in event_types
    )


def test_invalid_baseline_behavior_is_preserved(
    tmp_path,
) -> None:
    task = pilot_task(
        "pilot_tool_001"
    )

    adapter = (
        SequentialOrchestratorAdapter(
            [
                "I choose A.",
                VALID_EXPLANATION_P,
            ]
        )
    )

    result = (
        ExperimentalRunOrchestrator()
        .execute(
            experiment_id="pilot",
            run_id=(
                "pilot_tool_001_C0_r001"
            ),
            task=task,
            condition=Condition.C0,
            repetition_id=1,
            model_adapter=adapter,
            action_config=(
                action_config()
            ),
            explanation_config=(
                explanation_config()
            ),
            output_directory=(
                tmp_path
                / "run"
            ),
        )
    )

    assert (
        result.behavior_valid
        is False
    )

    assert (
        result.task_success
        is None
    )


def test_valid_wrong_behavior_scores_false(
    tmp_path,
) -> None:
    task = pilot_task(
        "pilot_tool_001"
    )

    adapter = (
        SequentialOrchestratorAdapter(
            [
                (
                    '{"action":"select_route",'
                    '"value":"B"}'
                ),
                VALID_EXPLANATION_P,
            ]
        )
    )

    result = (
        ExperimentalRunOrchestrator()
        .execute(
            experiment_id="pilot",
            run_id=(
                "pilot_tool_001_C0_r001"
            ),
            task=task,
            condition=Condition.C0,
            repetition_id=1,
            model_adapter=adapter,
            action_config=(
                action_config()
            ),
            explanation_config=(
                explanation_config()
            ),
            output_directory=(
                tmp_path
                / "run"
            ),
        )
    )

    assert (
        result.behavior_valid
        is True
    )

    assert (
        result.task_success
        is False
    )


def test_required_artifacts_are_written(
    tmp_path,
) -> None:
    task = pilot_task(
        "pilot_tool_001"
    )

    adapter = (
        SequentialOrchestratorAdapter(
            [
                (
                    '{"action":"select_route",'
                    '"value":"A"}'
                ),
                VALID_EXPLANATION_P,
            ]
        )
    )

    result = (
        ExperimentalRunOrchestrator()
        .execute(
            experiment_id="pilot",
            run_id=(
                "pilot_tool_001_C0_r001"
            ),
            task=task,
            condition=Condition.C0,
            repetition_id=1,
            model_adapter=adapter,
            action_config=(
                action_config()
            ),
            explanation_config=(
                explanation_config()
            ),
            output_directory=(
                tmp_path
                / "run"
            ),
        )
    )

    for value in (
        result.baseline_events_path,
        result.task_spec_path,
        result.resolved_context_path,
        result.snapshot_path,
        result.summary_path,
    ):
        assert Path(
            value
        ).is_file()


def test_summary_is_valid_json_and_matches_run(
    tmp_path,
) -> None:
    task = pilot_task(
        "pilot_tool_001"
    )

    adapter = (
        SequentialOrchestratorAdapter(
            [
                (
                    '{"action":"select_route",'
                    '"value":"A"}'
                ),
                VALID_EXPLANATION_P,
            ]
        )
    )

    result = (
        ExperimentalRunOrchestrator()
        .execute(
            experiment_id="pilot",
            run_id=(
                "pilot_tool_001_C0_r001"
            ),
            task=task,
            condition=Condition.C0,
            repetition_id=1,
            model_adapter=adapter,
            action_config=(
                action_config()
            ),
            explanation_config=(
                explanation_config()
            ),
            output_directory=(
                tmp_path
                / "run"
            ),
        )
    )

    summary = read_json(
        result.summary_path
    )

    assert (
        summary[
            "run_id"
        ]
        == result.run_id
    )

    assert (
        summary[
            "task_id"
        ]
        == "pilot_tool_001"
    )

    assert (
        summary[
            "condition"
        ]
        == "C0"
    )

    assert (
        summary[
            "task_success"
        ]
        is True
    )


def test_run_directory_may_not_be_overwritten(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "run"
    )

    path.mkdir()

    task = pilot_task(
        "pilot_tool_001"
    )

    adapter = (
        SequentialOrchestratorAdapter(
            []
        )
    )

    with pytest.raises(
        TaskOrchestrationError
    ) as exc_info:
        (
            ExperimentalRunOrchestrator()
            .execute(
                experiment_id="pilot",
                run_id=(
                    "pilot_tool_001_C0_r001"
                ),
                task=task,
                condition=Condition.C0,
                repetition_id=1,
                model_adapter=adapter,
                action_config=(
                    action_config()
                ),
                explanation_config=(
                    explanation_config()
                ),
                output_directory=path,
            )
        )

    assert (
        exc_info.value.code
        == "run_directory_exists"
    )


def test_action_seed_is_required(
    tmp_path,
) -> None:
    task = pilot_task(
        "pilot_tool_001"
    )

    config = InferenceConfig(
        max_tokens=64,
        seed=None,
    )

    with pytest.raises(
        TaskOrchestrationError
    ) as exc_info:
        (
            ExperimentalRunOrchestrator()
            .execute(
                experiment_id="pilot",
                run_id="run",
                task=task,
                condition=Condition.C0,
                repetition_id=1,
                model_adapter=(
                    SequentialOrchestratorAdapter(
                        []
                    )
                ),
                action_config=config,
                explanation_config=(
                    explanation_config()
                ),
                output_directory=(
                    tmp_path
                    / "run"
                ),
            )
        )

    assert (
        exc_info.value.code
        == "missing_action_seed"
    )