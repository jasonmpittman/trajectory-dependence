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
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPOSITORY_ROOT),
    )


from src.model import (  # noqa: E402
    InferenceConfig,
    MLXQwenAdapter,
)
from src.tasks import (  # noqa: E402
    SupplementalDiagnosticRunner,
    load_pilot_configuration,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the post-pilot dynamic-state "
            "technical diagnostics."
        )
    )

    parser.add_argument(
        "--pilot-config",
        default="config/pilot-run.json",
        help=(
            "Frozen pilot configuration used "
            "for model/runtime parameters."
        ),
    )

    parser.add_argument(
        "--output-root",
        default=(
            "artifacts/supplemental/"
            "pilot_technical_diagnostics"
        ),
        help=(
            "New output directory. Existing "
            "evidence is never overwritten."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    pilot = load_pilot_configuration(
        args.pilot_config
    )

    model_path_env = (
        pilot
        .model
        .path_environment_variable
    )

    model_path = os.environ.get(
        model_path_env
    )

    if not model_path:
        print(
            (
                "ERROR: required model "
                f"environment variable "
                f"{model_path_env!r} "
                "is not set."
            ),
            file=sys.stderr,
        )
        return 2

    if not Path(model_path).is_dir():
        print(
            (
                f"ERROR: configured model path "
                f"{model_path!r} "
                "is not a directory."
            ),
            file=sys.stderr,
        )
        return 2

    adapter = MLXQwenAdapter(
        model_identifier=(
            pilot.model.identifier
        ),
        model_revision=(
            pilot.model.revision
        ),
        model_checksum=(
            pilot.model.checksum
        ),
        quantization=(
            pilot.model.quantization
        ),
        model_path_env=(
            model_path_env
        ),
    )

    adapter.load()

    # Post-pilot diagnostic seeds. These are intentionally outside
    # the methodological pilot seed schedule and are persisted in
    # the diagnostic manifest/snapshots.
    action_config = InferenceConfig(
        max_tokens=(
            pilot.action_max_tokens
        ),
        seed=2026082501,
    )

    explanation_config = InferenceConfig(
        max_tokens=(
            pilot.explanation_max_tokens
        ),
        seed=2026082502,
    )

    result = (
        SupplementalDiagnosticRunner()
        .execute(
            model_adapter=adapter,
            action_config=(
                action_config
            ),
            explanation_config=(
                explanation_config
            ),
            output_root=(
                args.output_root
            ),
        )
    )

    print(
        json.dumps(
            {
                "status": (
                    "supplemental_diagnostics_completed"
                ),
                "passed": (
                    result.passed
                ),
                "study_role": (
                    result.study_role
                ),
                "post_pilot": (
                    result.post_pilot
                ),
                "included_in_h1": (
                    result.included_in_h1
                ),
                "included_in_h2": (
                    result.included_in_h2
                ),
                "diagnostics": [
                    {
                        "diagnostic_id": (
                            item.diagnostic_id
                        ),
                        "passed": (
                            item.passed
                        ),
                        "task_success": (
                            item.task_success
                        ),
                        "action_valid": (
                            item.action_valid
                        ),
                        "explanation_valid": (
                            item.explanation_valid
                        ),
                        "reported_sources": list(
                            item.reported_sources
                        ),
                        "technical_failure_count": (
                            item.technical_failure_count
                        ),
                    }
                    for item
                    in result.diagnostics
                ],
                "summary_path": (
                    result.summary_path
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )

    return (
        0
        if result.passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )