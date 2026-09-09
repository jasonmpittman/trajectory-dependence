__author__ = "Jason M. Pittman"
__date__ = "August 23, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import hashlib
import json
import os
from pathlib import Path


MODEL_PATH_ENV = "TRAJECTORY_MODEL_PATH"

EXCLUDED_PATH_PARTS = {
    ".cache",
}

EXCLUDED_FILENAMES = {
    ".DS_Store",
}


def sha256_file(
    path: Path,
    *,
    chunk_size: int = 16 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def included_files(
    root: Path,
) -> list[Path]:
    files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(
            root
        )

        if any(
            part in EXCLUDED_PATH_PARTS
            for part in relative.parts
        ):
            continue

        if (
            path.name
            in EXCLUDED_FILENAMES
        ):
            continue

        files.append(
            path
        )

    return sorted(
        files,
        key=lambda item: (
            item.relative_to(root).as_posix()
        ),
    )


def main() -> None:
    configured = os.environ.get(
        MODEL_PATH_ENV
    )

    if not configured:
        raise SystemExit(
            f"{MODEL_PATH_ENV} is not set."
        )

    root = Path(
        configured
    ).expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(
            f"Model path is not a directory: {root}"
        )

    records = []

    aggregate = hashlib.sha256()

    for path in included_files(
        root
    ):
        relative = (
            path.relative_to(root)
            .as_posix()
        )

        digest = sha256_file(
            path
        )

        size = path.stat().st_size

        record = {
            "path": relative,
            "size_bytes": size,
            "sha256": digest,
        }

        records.append(
            record
        )

        aggregate.update(
            (
                f"{digest}  {relative}\n"
            ).encode(
                "utf-8"
            )
        )

    output = {
        "model_path_env": (
            MODEL_PATH_ENV
        ),
        "fingerprint_scope": (
            "all_regular_files_excluding_local_cache"
        ),
        "file_count": len(
            records
        ),
        "aggregate_sha256": (
            aggregate.hexdigest()
        ),
        "files": records,
    }

    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()