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
import platform
import sys
from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load


MODEL_PATH_ENV = "TRAJECTORY_MODEL_PATH"
SMOKE_SEED = 12345

SMOKE_PROMPT = (
    "Respond with exactly the following text and "
    "nothing else: MODEL_SMOKE_OK"
)


def package_version(
    package: str,
) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def nonempty_thinking_block(
    value: str,
) -> bool:
    start_token = "<think>"
    end_token = "</think>"

    start = value.find(start_token)

    while start != -1:
        end = value.find(
            end_token,
            start + len(start_token),
        )

        if end == -1:
            return True

        body = value[
            start + len(start_token):
            end
        ]

        if body.strip():
            return True

        start = value.find(
            start_token,
            end + len(end_token),
        )

    return False


def main() -> None:
    configured = os.environ.get(
        MODEL_PATH_ENV
    )

    if not configured:
        raise SystemExit(
            f"{MODEL_PATH_ENV} is not set."
        )

    model_path = Path(
        configured
    ).expanduser().resolve()

    if not model_path.is_dir():
        raise SystemExit(
            f"Model directory not found: {model_path}"
        )

    mx.random.seed(
        SMOKE_SEED
    )

    model, tokenizer = load(
        str(model_path)
    )

    messages = [
        {
            "role": "user",
            "content": SMOKE_PROMPT,
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    if not isinstance(
        formatted_prompt,
        str,
    ):
        raise RuntimeError(
            "Chat template did not return a string."
        )

    token_ids = tokenizer.encode(
        formatted_prompt,
        add_special_tokens=False,
    )

    prompt_ends_with_open_think = (
        formatted_prompt
        .rstrip()
        .endswith("<think>")
    )

    output_text = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=32,
        verbose=False,
    )

    if not isinstance(
        output_text,
        str,
    ):
        raise RuntimeError(
            "mlx-lm generation did not return a string."
        )

    output_has_nonempty_thinking = (
        nonempty_thinking_block(
            output_text
        )
    )

    token_ids_serialized = json.dumps(
        token_ids,
        separators=(",", ":"),
    )

    report = {
        "backend": "mlx-lm",
        "model_path_env": MODEL_PATH_ENV,
        "python_version": (
            sys.version.split()[0]
        ),
        "platform": platform.platform(),
        "packages": {
            "mlx": package_version("mlx"),
            "mlx-lm": package_version(
                "mlx-lm"
            ),
            "transformers": package_version(
                "transformers"
            ),
            "huggingface-hub": (
                package_version(
                    "huggingface-hub"
                )
            ),
        },
        "inference": {
            "seed": SMOKE_SEED,
            "sampling": "argmax",
            "max_tokens": 32,
            "enable_thinking": False,
        },
        "template": {
            "formatted_prompt_chars": len(
                formatted_prompt
            ),
            "formatted_prompt_sha256": (
                sha256_text(
                    formatted_prompt
                )
            ),
            "formatted_prompt_tokens": len(
                token_ids
            ),
            "token_ids_sha256": (
                sha256_text(
                    token_ids_serialized
                )
            ),
            "prompt_ends_with_open_think": (
                prompt_ends_with_open_think
            ),
            "prompt_tail_repr": repr(
                formatted_prompt[-300:]
            ),
        },
        "generation": {
            "text": output_text,
            "text_sha256": (
                sha256_text(
                    output_text
                )
            ),
            "contains_expected_marker": (
                "MODEL_SMOKE_OK"
                in output_text
            ),
            "nonempty_thinking_block": (
                output_has_nonempty_thinking
            ),
        },
    }

    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    if prompt_ends_with_open_think:
        raise SystemExit(
            "FAIL: non-thinking template ended "
            "with an open <think> block."
        )

    if output_has_nonempty_thinking:
        raise SystemExit(
            "FAIL: generation contained a "
            "non-empty thinking block."
        )

    if "MODEL_SMOKE_OK" not in output_text:
        raise SystemExit(
            "FAIL: expected smoke marker "
            "was not generated."
        )


if __name__ == "__main__":
    main()