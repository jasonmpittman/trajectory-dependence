__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[1]
)

if (
    str(
        REPOSITORY_ROOT
    )
    not in sys.path
):
    sys.path.insert(
        0,
        str(
            REPOSITORY_ROOT
        ),
    )


from src.model import (  # noqa: E402
    MLXQwenAdapter,
)
from src.tasks import (  # noqa: E402
    PilotPlanner,
    PilotPreflightValidator,
    PilotSuiteRunner,
    load_configured_task_suite,
    load_pilot_configuration,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or execute the trajectory-dependence pilot."
        )
    )

    mode = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate the complete pilot without loading "
            "or invoking the model."
        ),
    )

    mode.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run the complete pilot after model-free preflight."
        ),
    )

    parser.add_argument(
        "--config",
        default=(
            "config/pilot-run.json"
        ),
        help=(
            "Path to pilot-run JSON configuration."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    config = load_pilot_configuration(
        args.config
    )

    suite = load_configured_task_suite(
        config
    )

    plan = (
        PilotPlanner()
        .build(
            config=config,
            suite=suite,
        )
    )

    if args.preflight:
        result = (
            PilotPreflightValidator()
            .validate(
                config=config,
                suite=suite,
                plan=plan,
            )
        )

        print(
            json.dumps(
                {
                    "status": (
                        "preflight_passed"
                    ),
                    "experiment_id": (
                        config
                        .experiment_id
                    ),
                    "suite_id": (
                        suite.suite_id
                    ),
                    "suite_version": (
                        suite.suite_version
                    ),
                    "plan_sha256": (
                        plan.plan_sha256
                    ),
                    "planned_runs": len(
                        plan.runs
                    ),
                    "checked_runs": (
                        result
                        .checked_runs
                    ),
                    "checked_intervention_pairs": (
                        result
                        .checked_intervention_pairs
                    ),
                    "estimated_model_generations": (
                        len(
                            plan.runs
                        )
                        * 2
                        + (
                            result
                            .checked_intervention_pairs
                            * 2
                        )
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )

        return 0

    path_env = (
        config
        .model
        .path_environment_variable
    )

    model_path = os.environ.get(
        path_env
    )

    if not model_path:
        print(
            (
                f"ERROR: required environment variable "
                f"{path_env!r} is not set."
            ),
            file=sys.stderr,
        )

        return 2

    model_directory = Path(
        model_path
    )

    if not model_directory.is_dir():
        print(
            (
                f"ERROR: model path "
                f"{str(model_directory)!r} "
                "is not a directory."
            ),
            file=sys.stderr,
        )

        return 2

    adapter = MLXQwenAdapter(
        model_identifier=(
            config
            .model
            .identifier
        ),
        model_revision=(
            config
            .model
            .revision
        ),
        model_checksum=(
            config
            .model
            .checksum
        ),
        quantization=(
            config
            .model
            .quantization
        ),
    )

    adapter.load()

    summary = (
        PilotSuiteRunner()
        .execute(
            config=config,
            suite=suite,
            plan=plan,
            model_adapter=adapter,
        )
    )

    print(
        json.dumps(
            {
                "status": (
                    "pilot_completed"
                ),
                "experiment_id": (
                    summary
                    .experiment_id
                ),
                "plan_sha256": (
                    summary
                    .plan_sha256
                ),
                "planned_runs": (
                    summary
                    .planned_runs
                ),
                "completed_runs": (
                    summary
                    .completed_runs
                ),
                "valid_behavior_runs": (
                    summary
                    .valid_behavior_runs
                ),
                "invalid_behavior_runs": (
                    summary
                    .invalid_behavior_runs
                ),
                "successful_task_runs": (
                    summary
                    .successful_task_runs
                ),
                "unsuccessful_task_runs": (
                    summary
                    .unsuccessful_task_runs
                ),
                "unscored_task_runs": (
                    summary
                    .unscored_task_runs
                ),
                "valid_explanations": (
                    summary
                    .valid_explanations
                ),
                "invalid_explanations": (
                    summary
                    .invalid_explanations
                ),
                "intervention_outcome_counts": (
                    summary
                    .intervention_outcome_counts
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )