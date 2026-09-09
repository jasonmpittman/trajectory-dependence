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
from dataclasses import replace
from pathlib import Path

import pytest

from src.interventions import (
    InterventionOutcome,
)
from src.model import (
    ChatMessage,
    ModelAdapter,
    ModelGenerationResult,
    ModelInputRecord,
    ModelRuntimeMetadata,
)
from src.tasks import (
    PilotExecutionError,
    PilotModelConfig,
    PilotPlanner,
    PilotSuiteRunner,
    load_configured_task_suite,
    load_pilot_configuration,
)


VALID_EXPLANATION_P = """
{
  "causal_sources": ["P"],
  "justification": {
    "P": "The task prompt defined the default.",
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
    "P": "The task prompt defined the rule.",
    "T": "The tool supplied the threshold.",
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


class SequentialPilotAdapter(
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
                "No fake pilot generation remains."
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


def runner_fixture(
    tmp_path,
):
    config = load_pilot_configuration(
        "config/pilot-run.json"
    )

    full_suite = (
        load_configured_task_suite(
            config
        )
    )

    tool_task = next(
        task
        for task
        in full_suite.tasks
        if (
            task.task_id
            == "pilot_tool_001"
        )
    )

    suite = replace(
        full_suite,
        suite_id=(
            "pilot_runner_test"
        ),
        tasks=(
            tool_task,
        ),
    )

    config = replace(
        config,
        experiment_id=(
            "pilot_runner_test"
        ),
        output_root=str(
            tmp_path
            / "output"
        ),
        model=PilotModelConfig(
            identifier="fake",
            revision="fake",
            checksum="fake",
            quantization="test",
            path_environment_variable=(
                "TRAJECTORY_MODEL_PATH"
            ),
        ),
    )

    plan = (
        PilotPlanner()
        .build(
            config=config,
            suite=suite,
        )
    )

    # C0:
    #   baseline A
    #   explanation
    #
    # C1:
    #   baseline B
    #   explanation
    #   targeted A
    #   sham B
    #
    # C2:
    #   baseline B
    #   explanation
    #   targeted A
    #   sham B
    #
    # C3:
    #   baseline B
    #   explanation
    #   targeted A
    #   sham B
    outputs = [
        (
            '{"action":"select_route",'
            '"value":"A"}'
        ),
        VALID_EXPLANATION_P,

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

    adapter = (
        SequentialPilotAdapter(
            outputs
        )
    )

    return (
        config,
        suite,
        plan,
        adapter,
    )


def test_runner_completes_all_planned_runs(
    tmp_path,
) -> None:
    (
        config,
        suite,
        plan,
        adapter,
    ) = runner_fixture(
        tmp_path
    )

    summary = (
        PilotSuiteRunner()
        .execute(
            config=config,
            suite=suite,
            plan=plan,
            model_adapter=adapter,
        )
    )

    assert (
        summary.planned_runs
        == 4
    )

    assert (
        summary.completed_runs
        == 4
    )


def test_runner_scores_all_four_tool_runs_correctly(
    tmp_path,
) -> None:
    (
        config,
        suite,
        plan,
        adapter,
    ) = runner_fixture(
        tmp_path
    )

    summary = (
        PilotSuiteRunner()
        .execute(
            config=config,
            suite=suite,
            plan=plan,
            model_adapter=adapter,
        )
    )

    assert (
        summary.valid_behavior_runs
        == 4
    )

    assert (
        summary.successful_task_runs
        == 4
    )

    assert (
        summary.unsuccessful_task_runs
        == 0
    )

    assert (
        summary.unscored_task_runs
        == 0
    )


def test_runner_records_three_positive_tool_interventions(
    tmp_path,
) -> None:
    (
        config,
        suite,
        plan,
        adapter,
    ) = runner_fixture(
        tmp_path
    )

    summary = (
        PilotSuiteRunner()
        .execute(
            config=config,
            suite=suite,
            plan=plan,
            model_adapter=adapter,
        )
    )

    assert (
        summary
        .intervention_outcome_counts[
            InterventionOutcome
            .POSITIVE
            .value
        ]
        == 3
    )


def test_runner_records_valid_explanations(
    tmp_path,
) -> None:
    (
        config,
        suite,
        plan,
        adapter,
    ) = runner_fixture(
        tmp_path
    )

    summary = (
        PilotSuiteRunner()
        .execute(
            config=config,
            suite=suite,
            plan=plan,
            model_adapter=adapter,
        )
    )

    assert (
        summary.valid_explanations
        == 4
    )

    assert (
        summary.invalid_explanations
        == 0
    )


def test_runner_writes_immutable_aggregate_bundle(
    tmp_path,
) -> None:
    (
        config,
        suite,
        plan,
        adapter,
    ) = runner_fixture(
        tmp_path
    )

    (
        PilotSuiteRunner()
        .execute(
            config=config,
            suite=suite,
            plan=plan,
            model_adapter=adapter,
        )
    )

    root = (
        Path(
            config.output_root
        )
        / config.experiment_id
    )

    for name in (
        "pilot_config.json",
        "task_suite.json",
        "execution_plan.json",
        "preflight.json",
        "model_runtime.json",
        "pilot_summary.json",
    ):
        assert (
            root
            / name
        ).is_file()


def test_runner_refuses_existing_experiment_root(
    tmp_path,
) -> None:
    (
        config,
        suite,
        plan,
        adapter,
    ) = runner_fixture(
        tmp_path
    )

    root = (
        Path(
            config.output_root
        )
        / config.experiment_id
    )

    root.mkdir(
        parents=True,
    )

    with pytest.raises(
        PilotExecutionError
    ) as exc_info:
        (
            PilotSuiteRunner()
            .execute(
                config=config,
                suite=suite,
                plan=plan,
                model_adapter=adapter,
            )
        )

    assert (
        exc_info.value.code
        == "experiment_root_exists"
    )


def test_runner_rejects_model_identity_mismatch(
    tmp_path,
) -> None:
    (
        config,
        suite,
        plan,
        adapter,
    ) = runner_fixture(
        tmp_path
    )

    config = replace(
        config,
        model=PilotModelConfig(
            identifier="wrong-model",
            revision="fake",
            checksum="fake",
            quantization="test",
            path_environment_variable=(
                "TRAJECTORY_MODEL_PATH"
            ),
        ),
    )

    with pytest.raises(
        PilotExecutionError
    ) as exc_info:
        (
            PilotSuiteRunner()
            .execute(
                config=config,
                suite=suite,
                plan=plan,
                model_adapter=adapter,
            )
        )

    assert (
        exc_info.value.code
        == "model_identity_mismatch"
    )