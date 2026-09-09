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
    PilotObservabilityAnalyzer,
    PilotObservabilityWriter,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Audit pilot logging, context length, "
            "runtime, provenance, and replay integrity."
        )
    )

    parser.add_argument(
        "--pilot-root",
        default=(
            "artifacts/pilot/"
            "trajectory_dependence_pilot"
        ),
    )

    parser.add_argument(
        "--processed-dir",
        default=(
            "data/processed/pilot"
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "results/"
            "pilot-observability-report.md"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    audit = (
        PilotObservabilityAnalyzer()
        .analyze(
            args.pilot_root
        )
    )

    outputs = (
        PilotObservabilityWriter()
        .write(
            audit=audit,
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

    print(
        json.dumps(
            {
                "status": (
                    "pilot_observability_completed"
                ),
                "experiment_id": (
                    audit.experiment_id
                ),
                "plan_sha256": (
                    audit.plan_sha256
                ),
                "integrity": (
                    audit.aggregates[
                        "integrity"
                    ]
                ),
                "generation_counts": (
                    audit.aggregates[
                        "generation_counts"
                    ]
                ),
                "outputs": (
                    outputs.to_dict()
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