__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from pathlib import Path

import pytest

from src.analysis import (
    PilotObservabilityAnalyzer,
    PilotObservabilityError,
    PilotObservabilityWriter,
)


RUNTIME = {
    "model_identifier": "synthetic",
    "model_revision": "revision",
    "model_checksum": "checksum",
    "quantization": "test",
    "backend": "fake",
    "backend_version": "1",
    "mlx_version": "test",
    "python_version": "test",
    "platform": "test",
    "model_path_env": (
        "TRAJECTORY_MODEL_PATH"
    ),
    "device_info": {},
}


def write_json(
    path: Path,
    value,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def event(
    *,
    index,
    event_id,
    event_type,
    parent=None,
    raw=None,
    metadata=None,
):
    return {
        "event_id": event_id,
        "sequence_index": index,
        "event_type": event_type,
        "parent_event_id": parent,
        "model_invocation_id": (
            "mdl"
            if (
                event_type
                in {
                    "context_build",
                    "provenance_audit",
                    "generation",
                    "action",
                    "explanation",
                }
            )
            else None
        ),
        "source_component": "synthetic",
        "raw_payload": (
            raw
            if raw
            is not None
            else {}
        ),
        "normalized_payload": {},
        "metadata": (
            metadata
            if metadata
            is not None
            else {}
        ),
    }


def generation_payload(
    *,
    prompt_tokens,
    generation_tokens,
    text,
):
    return {
        "raw_text": text,
        "raw_text_sha256": (
            f"raw-{prompt_tokens}-"
            f"{generation_tokens}"
        ),
        "generated_token_ids": [
            1,
            2,
        ],
        "generated_token_ids_sha256": (
            "generated-sha"
        ),
        "finish_reason": "stop",
        "prompt_tokens": (
            prompt_tokens
        ),
        "generation_tokens": (
            generation_tokens
        ),
        "prompt_tps": 100.0,
        "generation_tps": 10.0,
        "peak_memory_gb": 4.0,
        "input_record": {
            "model_visible_text": (
                "x"
                * prompt_tokens
            ),
            "messages": [],
            "rendered_prompt": (
                "y"
                * (
                    prompt_tokens
                    + 2
                )
            ),
            "rendered_prompt_sha256": (
                "rendered-sha"
            ),
            "prompt_token_ids": [],
            "prompt_token_ids_sha256": (
                "tokens-sha"
            ),
            "prompt_token_count": (
                prompt_tokens
            ),
            "thinking_mode": False,
            "chat_template_arguments": {},
        },
        "inference_config": {
            "max_tokens": 64,
            "seed": 12345,
        },
        "runtime_metadata": RUNTIME,
    }


def write_events(
    path: Path,
    events,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for value in events:
            handle.write(
                json.dumps(
                    value,
                    sort_keys=True,
                )
            )
            handle.write(
                "\n"
            )


def make_bundle(
    tmp_path,
):
    root = (
        tmp_path
        / "pilot"
    )

    run = (
        root
        / "task"
        / "C1"
        / "r001"
    )

    write_json(
        root
        / "model_runtime.json",
        RUNTIME,
    )

    write_json(
        root
        / "pilot_summary.json",
        {
            "experiment_id": (
                "synthetic"
            ),
            "plan_sha256": (
                "plan-sha"
            ),
            "intervention_outcome_counts": {
                "intervention_positive": 1,
                "intervention_negative": 0,
                "sham_unstable_ambiguous": 0,
                "behaviorally_unclassifiable": 0,
            },
            "runs": [
                {
                    "run_id": (
                        "task__C1__r001"
                    ),
                    "task_id": "task",
                    "condition": "C1",
                    "repetition_id": 1,
                    "intervention_results": [
                        {
                            "pair_id": (
                                "tool_pair"
                            ),
                            "source": "T",
                            "outcome": (
                                "intervention_positive"
                            ),
                            "targeted_changed": True,
                            "sham_changed": False,
                        }
                    ],
                }
            ],
        },
    )

    write_json(
        run
        / "pre_focal_snapshot.json",
        {
            "snapshot_id": "snap",
            "state_checksum_sha256": (
                "checksum"
            ),
            "sequence_position": 1,
        },
    )

    write_json(
        run
        / "run_summary.json",
        {
            "snapshot_id": "snap",
            "snapshot_checksum": (
                "checksum"
            ),
            "baseline_action": {
                "action_event_id": "a1",
                "generation_event_id": "g1",
            },
            "self_explanation": {
                "action_event_id": "a1",
                "generation_event_id": "g2",
                "explanation_event_id": "e1",
            },
        },
    )

    baseline = [
        event(
            index=0,
            event_id="s",
            event_type="system",
            raw={
                "runtime_metadata": (
                    RUNTIME
                )
            },
        ),
        event(
            index=1,
            event_id="task",
            event_type="task",
        ),
        event(
            index=2,
            event_id="c1",
            event_type="context_build",
        ),
        event(
            index=3,
            event_id="p1",
            event_type="provenance_audit",
            parent="c1",
        ),
        event(
            index=4,
            event_id="g1",
            event_type="generation",
            parent="p1",
            raw=(
                generation_payload(
                    prompt_tokens=100,
                    generation_tokens=8,
                    text="action",
                )
            ),
        ),
        event(
            index=5,
            event_id="a1",
            event_type="action",
            parent="g1",
        ),
        event(
            index=6,
            event_id="g2",
            event_type="generation",
            parent="a1",
            raw=(
                generation_payload(
                    prompt_tokens=150,
                    generation_tokens=20,
                    text="explanation",
                )
            ),
            metadata={
                "generation_role": (
                    "self_explanation"
                )
            },
        ),
        event(
            index=7,
            event_id="e1",
            event_type="explanation",
            parent="g2",
        ),
    ]

    write_events(
        run
        / "baseline"
        / "events.jsonl",
        baseline,
    )

    def replay(
        intervention_type,
        *,
        prompt_tokens,
    ):
        return [
            event(
                index=0,
                event_id=(
                    f"s-{intervention_type}"
                ),
                event_type="system",
                raw={
                    "runtime_metadata": (
                        RUNTIME
                    )
                },
            ),
            event(
                index=1,
                event_id=(
                    f"task-{intervention_type}"
                ),
                event_type="task",
            ),
            event(
                index=2,
                event_id=(
                    f"i-{intervention_type}"
                ),
                event_type=(
                    intervention_type
                ),
            ),
            event(
                index=3,
                event_id=(
                    f"c-{intervention_type}"
                ),
                event_type="context_build",
            ),
            event(
                index=4,
                event_id=(
                    f"p-{intervention_type}"
                ),
                event_type="provenance_audit",
                parent=(
                    f"c-{intervention_type}"
                ),
            ),
            event(
                index=5,
                event_id=(
                    f"g-{intervention_type}"
                ),
                event_type="generation",
                parent=(
                    f"p-{intervention_type}"
                ),
                raw=(
                    generation_payload(
                        prompt_tokens=(
                            prompt_tokens
                        ),
                        generation_tokens=8,
                        text="replay",
                    )
                ),
            ),
            event(
                index=6,
                event_id=(
                    f"a-{intervention_type}"
                ),
                event_type="action",
                parent=(
                    f"g-{intervention_type}"
                ),
            ),
        ]

    write_events(
        run
        / "replays"
        / "tool_pair"
        / "targeted"
        / "events.jsonl",
        replay(
            "intervention",
            prompt_tokens=101,
        ),
    )

    write_events(
        run
        / "replays"
        / "tool_pair"
        / "sham"
        / "events.jsonl",
        replay(
            "sham_intervention",
            prompt_tokens=100,
        ),
    )

    return root


def test_audit_counts_four_generations(
    tmp_path,
):
    root = make_bundle(
        tmp_path
    )

    audit = (
        PilotObservabilityAnalyzer()
        .analyze(
            root
        )
    )

    assert len(
        audit.generations
    ) == 4

    assert (
        audit.aggregates[
            "generation_counts"
        ][
            "observed"
        ]
        == 4
    )


def test_audit_validates_event_structure(
    tmp_path,
):
    root = make_bundle(
        tmp_path
    )

    audit = (
        PilotObservabilityAnalyzer()
        .analyze(
            root
        )
    )

    run = audit.runs[
        0
    ]

    assert (
        run.logging_complete
        is True
    )

    assert (
        run.explanation_after_action
        is True
    )

    assert (
        run.snapshot_before_action
        is True
    )

    assert (
        run.replay_chains_valid
        is True
    )


def test_audit_computes_replay_token_deltas(
    tmp_path,
):
    root = make_bundle(
        tmp_path
    )

    audit = (
        PilotObservabilityAnalyzer()
        .analyze(
            root
        )
    )

    targeted = next(
        item
        for item
        in audit.generations
        if (
            item.role
            == "targeted_replay"
        )
    )

    sham = next(
        item
        for item
        in audit.generations
        if (
            item.role
            == "sham_replay"
        )
    )

    assert (
        targeted.prompt_token_delta
        == 1
    )

    assert (
        sham.prompt_token_delta
        == 0
    )


def test_audit_derives_matched_runtime(
    tmp_path,
):
    root = make_bundle(
        tmp_path
    )

    audit = (
        PilotObservabilityAnalyzer()
        .analyze(
            root
        )
    )

    matched = (
        audit.aggregates[
            "matched_unit_runtime"
        ]
    )

    assert (
        matched[
            "n_complete_units"
        ]
        == 1
    )

    assert (
        matched[
            "wall_clock_logged"
        ]
        is False
    )

    assert (
        matched[
            "mean_estimated_seconds"
        ]
        is not None
    )


def test_runtime_mismatch_is_reported(
    tmp_path,
):
    root = make_bundle(
        tmp_path
    )

    targeted = (
        root
        / "task"
        / "C1"
        / "r001"
        / "replays"
        / "tool_pair"
        / "targeted"
        / "events.jsonl"
    )

    events = [
        json.loads(
            line
        )
        for line
        in targeted.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    generation = next(
        value
        for value
        in events
        if (
            value[
                "event_type"
            ]
            == "generation"
        )
    )

    generation[
        "raw_payload"
    ][
        "runtime_metadata"
    ][
        "backend_version"
    ] = "wrong"

    write_events(
        targeted,
        events,
    )

    audit = (
        PilotObservabilityAnalyzer()
        .analyze(
            root
        )
    )

    assert (
        audit.runs[
            0
        ]
        .runtime_metadata_consistent
        is False
    )

    assert (
        audit.runs[
            0
        ]
        .logging_complete
        is False
    )


def test_writer_creates_outputs(
    tmp_path,
):
    root = make_bundle(
        tmp_path
    )

    audit = (
        PilotObservabilityAnalyzer()
        .analyze(
            root
        )
    )

    outputs = (
        PilotObservabilityWriter()
        .write(
            audit=audit,
            processed_directory=(
                tmp_path
                / "processed"
            ),
            report_path=(
                tmp_path
                / "report.md"
            ),
        )
    )

    assert Path(
        outputs.generation_csv
    ).is_file()

    assert Path(
        outputs.run_csv
    ).is_file()

    assert Path(
        outputs.summary_json
    ).is_file()

    assert Path(
        outputs.report_markdown
    ).is_file()


def test_writer_refuses_overwrite(
    tmp_path,
):
    root = make_bundle(
        tmp_path
    )

    audit = (
        PilotObservabilityAnalyzer()
        .analyze(
            root
        )
    )

    writer = (
        PilotObservabilityWriter()
    )

    writer.write(
        audit=audit,
        processed_directory=(
            tmp_path
            / "processed"
        ),
        report_path=(
            tmp_path
            / "report.md"
        ),
    )

    with pytest.raises(
        PilotObservabilityError
    ) as exc_info:
        writer.write(
            audit=audit,
            processed_directory=(
                tmp_path
                / "processed"
            ),
            report_path=(
                tmp_path
                / "report.md"
            ),
        )

    assert (
        exc_info.value.code
        == "observability_output_exists"
    )