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


from src.analysis import (  # noqa: E402
    PilotAnalysisWriter,
    PilotArtifactAnalyzer,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce methodological pilot metrics "
            "from frozen pilot artifacts."
        )
    )

    parser.add_argument(
        "--pilot-root",
        default=(
            "artifacts/pilot/"
            "trajectory_dependence_pilot"
        ),
        help=(
            "Root directory of the completed "
            "pilot artifact bundle."
        ),
    )

    parser.add_argument(
        "--processed-dir",
        default=(
            "data/processed/pilot"
        ),
        help=(
            "Directory for processed pilot "
            "CSV/JSON outputs."
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "results/pilot-report.md"
        ),
        help=(
            "Generated Markdown pilot report."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Explicitly replace previously "
            "generated analysis outputs. "
            "Raw pilot artifacts are never modified."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    analysis = (
        PilotArtifactAnalyzer()
        .analyze(
            args.pilot_root
        )
    )

    outputs = (
        PilotAnalysisWriter()
        .write(
            analysis=analysis,
            processed_directory=(
                args.processed_dir
            ),
            report_path=(
                args.report
            ),
            overwrite=(
                args.overwrite
            ),
        )
    )

    overall = (
        analysis
        .aggregates[
            "overall"
        ]
    )

    result = {
        "status": (
            "pilot_analysis_completed"
        ),
        "experiment_id": (
            analysis.experiment_id
        ),
        "plan_sha256": (
            analysis.plan_sha256
        ),
        "runs": (
            overall[
                "runs"
            ]
        ),
        "behavior_valid": (
            overall[
                "behavior_valid"
            ]
        ),
        "task_success": (
            overall[
                "task_success"
            ]
        ),
        "explanation_valid": (
            overall[
                "explanation_valid"
            ]
        ),
        "intervention_pairs": (
            overall[
                "intervention_pairs"
            ]
        ),
        "intervention_outcomes": (
            overall[
                "intervention_outcomes"
            ]
        ),
        "outputs": (
            outputs.to_dict()
        ),
    }

    print(
        json.dumps(
            result,
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