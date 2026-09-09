__author__ = "Jason M. Pittman"
__date__ = "August 25, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from ..logging import (
    EventType,
)


class PilotObservabilityError(RuntimeError):
    """Defined failure auditing pilot execution artifacts."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
    ) -> None:
        super().__init__(
            message
        )

        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class GenerationObservability:
    """
    Logged inference/resource evidence for one model generation.

    `estimated_*_seconds` are derived from token counts and logged
    throughput. They are not wall-clock orchestration measurements.
    """

    run_id: str
    task_id: str
    condition: str
    repetition_id: int

    role: str

    pair_id: str | None
    source: str | None

    event_stream_path: str
    generation_event_id: str
    model_invocation_id: str | None

    prompt_tokens: int
    generation_tokens: int

    prompt_tps: float | None
    generation_tps: float | None

    estimated_prompt_seconds: float | None
    estimated_generation_seconds: float | None

    peak_memory_gb: float | None

    model_visible_chars: int
    rendered_prompt_chars: int

    rendered_prompt_sha256: str
    prompt_token_ids_sha256: str
    raw_text_sha256: str

    prompt_token_count_consistent: bool
    runtime_metadata_consistent: bool

    baseline_prompt_tokens: int | None
    prompt_token_delta: int | None

    baseline_model_visible_chars: int | None
    model_visible_char_delta: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "condition": self.condition,
            "repetition_id": (
                self.repetition_id
            ),
            "role": self.role,
            "pair_id": self.pair_id,
            "source": self.source,
            "event_stream_path": (
                self.event_stream_path
            ),
            "generation_event_id": (
                self.generation_event_id
            ),
            "model_invocation_id": (
                self.model_invocation_id
            ),
            "prompt_tokens": (
                self.prompt_tokens
            ),
            "generation_tokens": (
                self.generation_tokens
            ),
            "prompt_tps": self.prompt_tps,
            "generation_tps": (
                self.generation_tps
            ),
            "estimated_prompt_seconds": (
                self.estimated_prompt_seconds
            ),
            "estimated_generation_seconds": (
                self.estimated_generation_seconds
            ),
            "peak_memory_gb": (
                self.peak_memory_gb
            ),
            "model_visible_chars": (
                self.model_visible_chars
            ),
            "rendered_prompt_chars": (
                self.rendered_prompt_chars
            ),
            "rendered_prompt_sha256": (
                self.rendered_prompt_sha256
            ),
            "prompt_token_ids_sha256": (
                self.prompt_token_ids_sha256
            ),
            "raw_text_sha256": (
                self.raw_text_sha256
            ),
            "prompt_token_count_consistent": (
                self.prompt_token_count_consistent
            ),
            "runtime_metadata_consistent": (
                self.runtime_metadata_consistent
            ),
            "baseline_prompt_tokens": (
                self.baseline_prompt_tokens
            ),
            "prompt_token_delta": (
                self.prompt_token_delta
            ),
            "baseline_model_visible_chars": (
                self.baseline_model_visible_chars
            ),
            "model_visible_char_delta": (
                self.model_visible_char_delta
            ),
        }


@dataclass(frozen=True)
class RunObservability:
    """Structural/logging audit for one baseline run and its replays."""

    run_id: str
    task_id: str
    condition: str
    repetition_id: int

    baseline_event_count: int

    baseline_sequence_contiguous: bool
    baseline_focal_chain_valid: bool
    explanation_chain_valid: bool

    explanation_after_action: bool

    snapshot_before_action: bool
    snapshot_identity_valid: bool

    baseline_intervention_event_count: int
    baseline_technical_failure_count: int

    replay_streams_expected: int
    replay_streams_checked: int

    replay_sequences_contiguous: bool
    replay_chains_valid: bool
    replay_inference_configs_match_baseline: bool

    replay_technical_failure_count: int

    runtime_metadata_consistent: bool
    token_counts_consistent: bool

    logging_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "condition": self.condition,
            "repetition_id": (
                self.repetition_id
            ),
            "baseline_event_count": (
                self.baseline_event_count
            ),
            "baseline_sequence_contiguous": (
                self.baseline_sequence_contiguous
            ),
            "baseline_focal_chain_valid": (
                self.baseline_focal_chain_valid
            ),
            "explanation_chain_valid": (
                self.explanation_chain_valid
            ),
            "explanation_after_action": (
                self.explanation_after_action
            ),
            "snapshot_before_action": (
                self.snapshot_before_action
            ),
            "snapshot_identity_valid": (
                self.snapshot_identity_valid
            ),
            "baseline_intervention_event_count": (
                self.baseline_intervention_event_count
            ),
            "baseline_technical_failure_count": (
                self.baseline_technical_failure_count
            ),
            "replay_streams_expected": (
                self.replay_streams_expected
            ),
            "replay_streams_checked": (
                self.replay_streams_checked
            ),
            "replay_sequences_contiguous": (
                self.replay_sequences_contiguous
            ),
            "replay_chains_valid": (
                self.replay_chains_valid
            ),
            "replay_inference_configs_match_baseline": (
                self
                .replay_inference_configs_match_baseline
            ),
            "replay_technical_failure_count": (
                self.replay_technical_failure_count
            ),
            "runtime_metadata_consistent": (
                self.runtime_metadata_consistent
            ),
            "token_counts_consistent": (
                self.token_counts_consistent
            ),
            "logging_complete": (
                self.logging_complete
            ),
        }


@dataclass(frozen=True)
class PilotObservabilityAudit:
    """Complete execution-quality audit for one pilot bundle."""

    pilot_root: str

    experiment_id: str
    plan_sha256: str

    model_runtime_metadata: dict[
        str,
        Any,
    ]

    generations: tuple[
        GenerationObservability,
        ...
    ]

    runs: tuple[
        RunObservability,
        ...
    ]

    aggregates: dict[
        str,
        Any,
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pilot_root": self.pilot_root,
            "experiment_id": (
                self.experiment_id
            ),
            "plan_sha256": (
                self.plan_sha256
            ),
            "model_runtime_metadata": (
                deepcopy(
                    self.model_runtime_metadata
                )
            ),
            "generations": [
                item.to_dict()
                for item
                in self.generations
            ],
            "runs": [
                item.to_dict()
                for item
                in self.runs
            ],
            "aggregates": deepcopy(
                self.aggregates
            ),
        }


@dataclass(frozen=True)
class PilotObservabilityOutputs:
    """Processed artifact paths generated by the observability writer."""

    generation_csv: str
    run_csv: str
    summary_json: str
    report_markdown: str

    def to_dict(self) -> dict[str, str]:
        return {
            "generation_csv": (
                self.generation_csv
            ),
            "run_csv": self.run_csv,
            "summary_json": (
                self.summary_json
            ),
            "report_markdown": (
                self.report_markdown
            ),
        }


class PilotObservabilityAnalyzer:
    """
    Audit event-stream integrity, inference metadata, context size,
    throughput, resource use, and replay matching.

    Raw artifacts are read only.
    """

    def analyze(
        self,
        pilot_root: str | Path,
    ) -> PilotObservabilityAudit:
        root = Path(
            pilot_root
        )

        if not root.is_dir():
            raise PilotObservabilityError(
                code="pilot_root_not_found",
                message=(
                    f"Pilot root "
                    f"{str(root)!r} "
                    "does not exist."
                ),
            )

        summary = self._read_json(
            root
            / "pilot_summary.json"
        )

        runtime_metadata = (
            self._read_json(
                root
                / "model_runtime.json"
            )
        )

        experiment_id = (
            self._required_string(
                summary,
                "experiment_id",
            )
        )

        plan_sha256 = (
            self._required_string(
                summary,
                "plan_sha256",
            )
        )

        concise_runs = summary.get(
            "runs"
        )

        if not isinstance(
            concise_runs,
            list,
        ):
            raise PilotObservabilityError(
                code="invalid_pilot_summary",
                message=(
                    "pilot_summary.json "
                    "field 'runs' must "
                    "be an array."
                ),
            )

        generation_rows: list[
            GenerationObservability
        ] = []

        run_rows: list[
            RunObservability
        ] = []

        for concise in concise_runs:
            if not isinstance(
                concise,
                Mapping,
            ):
                raise PilotObservabilityError(
                    code="invalid_run_row",
                    message=(
                        "Pilot run entry must "
                        "be a mapping."
                    ),
                )

            (
                run_audit,
                run_generations,
            ) = self._audit_run(
                root=root,
                concise=concise,
                runtime_metadata=(
                    runtime_metadata
                ),
            )

            run_rows.append(
                run_audit
            )

            generation_rows.extend(
                run_generations
            )

        aggregates = self._aggregate(
            runs=run_rows,
            generations=(
                generation_rows
            ),
            summary=summary,
        )

        return PilotObservabilityAudit(
            pilot_root=str(
                root
            ),
            experiment_id=(
                experiment_id
            ),
            plan_sha256=(
                plan_sha256
            ),
            model_runtime_metadata=(
                runtime_metadata
            ),
            generations=tuple(
                generation_rows
            ),
            runs=tuple(
                run_rows
            ),
            aggregates=aggregates,
        )

    def _audit_run(
        self,
        *,
        root: Path,
        concise: Mapping[
            str,
            Any,
        ],
        runtime_metadata: dict[
            str,
            Any,
        ],
    ) -> tuple[
        RunObservability,
        list[
            GenerationObservability
        ],
    ]:
        run_id = self._required_string(
            concise,
            "run_id",
        )

        task_id = self._required_string(
            concise,
            "task_id",
        )

        condition = self._required_string(
            concise,
            "condition",
        )

        repetition_id = concise.get(
            "repetition_id"
        )

        if (
            not isinstance(
                repetition_id,
                int,
            )
            or repetition_id < 0
        ):
            raise PilotObservabilityError(
                code="invalid_repetition_id",
                message=(
                    f"Run {run_id!r} "
                    "has invalid repetition_id."
                ),
            )

        run_directory = (
            root
            / task_id
            / condition
            / (
                f"r"
                f"{repetition_id:03d}"
            )
        )

        run_summary = self._read_json(
            run_directory
            / "run_summary.json"
        )

        snapshot = self._read_json(
            run_directory
            / "pre_focal_snapshot.json"
        )

        baseline_path = (
            run_directory
            / "baseline"
            / "events.jsonl"
        )

        baseline_events = (
            self._read_events(
                baseline_path
            )
        )

        baseline_map = {
            event[
                "event_id"
            ]: event
            for event
            in baseline_events
        }

        baseline_action = (
            run_summary.get(
                "baseline_action"
            )
        )

        explanation = (
            run_summary.get(
                "self_explanation"
            )
        )

        if not isinstance(
            baseline_action,
            Mapping,
        ):
            raise PilotObservabilityError(
                code="baseline_action_missing",
                message=(
                    f"Run {run_id!r} "
                    "lacks baseline_action."
                ),
            )

        if not isinstance(
            explanation,
            Mapping,
        ):
            raise PilotObservabilityError(
                code="explanation_missing",
                message=(
                    f"Run {run_id!r} "
                    "lacks self_explanation."
                ),
            )

        action_event_id = (
            self._required_string(
                baseline_action,
                "action_event_id",
            )
        )

        focal_generation_id = (
            self._required_string(
                baseline_action,
                "generation_event_id",
            )
        )

        explanation_generation_id = (
            self._required_string(
                explanation,
                "generation_event_id",
            )
        )

        explanation_event_id = (
            self._required_string(
                explanation,
                "explanation_event_id",
            )
        )

        action_event = self._event_by_id(
            baseline_map,
            action_event_id,
            run_id=run_id,
        )

        focal_generation = (
            self._event_by_id(
                baseline_map,
                focal_generation_id,
                run_id=run_id,
            )
        )

        explanation_generation = (
            self._event_by_id(
                baseline_map,
                explanation_generation_id,
                run_id=run_id,
            )
        )

        explanation_event = (
            self._event_by_id(
                baseline_map,
                explanation_event_id,
                run_id=run_id,
            )
        )

        baseline_sequence_contiguous = (
            self._sequence_contiguous(
                baseline_events
            )
        )

        baseline_focal_chain_valid = (
            self._focal_chain_valid(
                events=baseline_map,
                generation_event=(
                    focal_generation
                ),
                action_event=(
                    action_event
                ),
            )
        )

        explanation_chain_valid = (
            (
                explanation_generation.get(
                    "event_type"
                )
                == EventType.GENERATION.value
            )
            and (
                explanation_generation.get(
                    "parent_event_id"
                )
                == action_event_id
            )
            and (
                explanation_event.get(
                    "event_type"
                )
                == EventType.EXPLANATION.value
            )
            and (
                explanation_event.get(
                    "parent_event_id"
                )
                == explanation_generation_id
            )
        )

        explanation_after_action = (
            self._sequence_index(
                explanation_event
            )
            > self._sequence_index(
                action_event
            )
        )

        snapshot_before_action = (
            self._required_int(
                snapshot,
                "sequence_position",
            )
            < self._sequence_index(
                action_event
            )
        )

        snapshot_identity_valid = (
            (
                run_summary.get(
                    "snapshot_id"
                )
                == snapshot.get(
                    "snapshot_id"
                )
            )
            and (
                run_summary.get(
                    "snapshot_checksum"
                )
                == snapshot.get(
                    "state_checksum_sha256"
                )
            )
        )

        baseline_intervention_event_count = sum(
            1
            for event
            in baseline_events
            if (
                event.get(
                    "event_type"
                )
                in {
                    EventType
                    .INTERVENTION
                    .value,
                    EventType
                    .SHAM_INTERVENTION
                    .value,
                }
            )
        )

        baseline_technical_failure_count = sum(
            1
            for event
            in baseline_events
            if (
                event.get(
                    "event_type"
                )
                == (
                    EventType
                    .TECHNICAL_FAILURE
                    .value
                )
            )
        )

        baseline_system_runtime_ok = (
            self._system_runtime_matches(
                baseline_events,
                runtime_metadata,
            )
        )

        focal_record = (
            self._generation_record(
                run_id=run_id,
                task_id=task_id,
                condition=condition,
                repetition_id=(
                    repetition_id
                ),
                role="baseline_action",
                pair_id=None,
                source=None,
                event_stream_path=(
                    baseline_path
                ),
                event=(
                    focal_generation
                ),
                expected_runtime=(
                    runtime_metadata
                ),
                baseline_prompt_tokens=None,
                baseline_model_visible_chars=None,
            )
        )

        explanation_record = (
            self._generation_record(
                run_id=run_id,
                task_id=task_id,
                condition=condition,
                repetition_id=(
                    repetition_id
                ),
                role="self_explanation",
                pair_id=None,
                source=None,
                event_stream_path=(
                    baseline_path
                ),
                event=(
                    explanation_generation
                ),
                expected_runtime=(
                    runtime_metadata
                ),
                baseline_prompt_tokens=None,
                baseline_model_visible_chars=None,
            )
        )

        baseline_config = self._generation_config(
            focal_generation
        )

        replay_streams_expected = (
            2
            * len(
                concise.get(
                    "intervention_results",
                    []
                )
            )
        )

        replay_streams_checked = 0
        replay_sequences_contiguous = True
        replay_chains_valid = True

        replay_inference_configs_match_baseline = (
            True
        )

        replay_technical_failure_count = 0
        replay_runtime_ok = True
        replay_token_counts_ok = True

        generation_rows = [
            focal_record,
            explanation_record,
        ]

        intervention_results = concise.get(
            "intervention_results",
            []
        )

        if not isinstance(
            intervention_results,
            list,
        ):
            raise PilotObservabilityError(
                code="invalid_intervention_results",
                message=(
                    f"Run {run_id!r} "
                    "intervention_results "
                    "must be an array."
                ),
            )

        for intervention in (
            intervention_results
        ):
            if not isinstance(
                intervention,
                Mapping,
            ):
                raise PilotObservabilityError(
                    code="invalid_intervention_row",
                    message=(
                        f"Run {run_id!r} "
                        "contains invalid "
                        "intervention metadata."
                    ),
                )

            pair_id = self._required_string(
                intervention,
                "pair_id",
            )

            source = self._required_string(
                intervention,
                "source",
            )

            for (
                role,
                expected_intervention_type,
            ) in (
                (
                    "targeted_replay",
                    EventType
                    .INTERVENTION
                    .value,
                ),
                (
                    "sham_replay",
                    EventType
                    .SHAM_INTERVENTION
                    .value,
                ),
            ):
                subdirectory = (
                    "targeted"
                    if (
                        role
                        == "targeted_replay"
                    )
                    else "sham"
                )

                replay_path = (
                    run_directory
                    / "replays"
                    / pair_id
                    / subdirectory
                    / "events.jsonl"
                )

                events = self._read_events(
                    replay_path
                )

                event_map = {
                    event[
                        "event_id"
                    ]: event
                    for event
                    in events
                }

                replay_streams_checked += 1

                replay_sequences_contiguous = (
                    replay_sequences_contiguous
                    and self._sequence_contiguous(
                        events
                    )
                )

                intervention_events = [
                    event
                    for event
                    in events
                    if (
                        event.get(
                            "event_type"
                        )
                        == expected_intervention_type
                    )
                ]

                wrong_intervention_type = (
                    EventType
                    .SHAM_INTERVENTION
                    .value
                    if (
                        expected_intervention_type
                        == (
                            EventType
                            .INTERVENTION
                            .value
                        )
                    )
                    else (
                        EventType
                        .INTERVENTION
                        .value
                    )
                )

                wrong_intervention_events = [
                    event
                    for event
                    in events
                    if (
                        event.get(
                            "event_type"
                        )
                        == wrong_intervention_type
                    )
                ]

                generations = [
                    event
                    for event
                    in events
                    if (
                        event.get(
                            "event_type"
                        )
                        == (
                            EventType
                            .GENERATION
                            .value
                        )
                    )
                ]

                actions = [
                    event
                    for event
                    in events
                    if (
                        event.get(
                            "event_type"
                        )
                        == (
                            EventType
                            .ACTION
                            .value
                        )
                    )
                ]

                replay_chain = (
                    len(
                        intervention_events
                    )
                    == 1
                    and not (
                        wrong_intervention_events
                    )
                    and len(
                        generations
                    )
                    == 1
                    and len(
                        actions
                    )
                    == 1
                    and self._focal_chain_valid(
                        events=event_map,
                        generation_event=(
                            generations[0]
                        ),
                        action_event=(
                            actions[0]
                        ),
                    )
                )

                replay_chains_valid = (
                    replay_chains_valid
                    and replay_chain
                )

                replay_technical_failure_count += sum(
                    1
                    for event
                    in events
                    if (
                        event.get(
                            "event_type"
                        )
                        == (
                            EventType
                            .TECHNICAL_FAILURE
                            .value
                        )
                    )
                )

                replay_runtime_ok = (
                    replay_runtime_ok
                    and self._system_runtime_matches(
                        events,
                        runtime_metadata,
                    )
                )

                if len(
                    generations
                ) != 1:
                    raise PilotObservabilityError(
                        code="replay_generation_count",
                        message=(
                            f"Replay stream "
                            f"{str(replay_path)!r} "
                            "must contain exactly "
                            "one generation."
                        ),
                    )

                replay_generation = (
                    generations[0]
                )

                replay_config = (
                    self._generation_config(
                        replay_generation
                    )
                )

                replay_inference_configs_match_baseline = (
                    replay_inference_configs_match_baseline
                    and (
                        replay_config
                        == baseline_config
                    )
                )

                record = (
                    self._generation_record(
                        run_id=run_id,
                        task_id=task_id,
                        condition=condition,
                        repetition_id=(
                            repetition_id
                        ),
                        role=role,
                        pair_id=pair_id,
                        source=source,
                        event_stream_path=(
                            replay_path
                        ),
                        event=(
                            replay_generation
                        ),
                        expected_runtime=(
                            runtime_metadata
                        ),
                        baseline_prompt_tokens=(
                            focal_record
                            .prompt_tokens
                        ),
                        baseline_model_visible_chars=(
                            focal_record
                            .model_visible_chars
                        ),
                    )
                )

                replay_runtime_ok = (
                    replay_runtime_ok
                    and record
                    .runtime_metadata_consistent
                )

                replay_token_counts_ok = (
                    replay_token_counts_ok
                    and record
                    .prompt_token_count_consistent
                )

                generation_rows.append(
                    record
                )

        runtime_metadata_consistent = (
            baseline_system_runtime_ok
            and focal_record
            .runtime_metadata_consistent
            and explanation_record
            .runtime_metadata_consistent
            and replay_runtime_ok
        )

        token_counts_consistent = (
            focal_record
            .prompt_token_count_consistent
            and explanation_record
            .prompt_token_count_consistent
            and replay_token_counts_ok
        )

        logging_complete = all(
            (
                baseline_sequence_contiguous,
                baseline_focal_chain_valid,
                explanation_chain_valid,
                explanation_after_action,
                snapshot_before_action,
                snapshot_identity_valid,
                (
                    baseline_intervention_event_count
                    == 0
                ),
                (
                    replay_streams_checked
                    == replay_streams_expected
                ),
                replay_sequences_contiguous,
                replay_chains_valid,
                replay_inference_configs_match_baseline,
                runtime_metadata_consistent,
                token_counts_consistent,
            )
        )

        return (
            RunObservability(
                run_id=run_id,
                task_id=task_id,
                condition=condition,
                repetition_id=(
                    repetition_id
                ),
                baseline_event_count=len(
                    baseline_events
                ),
                baseline_sequence_contiguous=(
                    baseline_sequence_contiguous
                ),
                baseline_focal_chain_valid=(
                    baseline_focal_chain_valid
                ),
                explanation_chain_valid=(
                    explanation_chain_valid
                ),
                explanation_after_action=(
                    explanation_after_action
                ),
                snapshot_before_action=(
                    snapshot_before_action
                ),
                snapshot_identity_valid=(
                    snapshot_identity_valid
                ),
                baseline_intervention_event_count=(
                    baseline_intervention_event_count
                ),
                baseline_technical_failure_count=(
                    baseline_technical_failure_count
                ),
                replay_streams_expected=(
                    replay_streams_expected
                ),
                replay_streams_checked=(
                    replay_streams_checked
                ),
                replay_sequences_contiguous=(
                    replay_sequences_contiguous
                ),
                replay_chains_valid=(
                    replay_chains_valid
                ),
                replay_inference_configs_match_baseline=(
                    replay_inference_configs_match_baseline
                ),
                replay_technical_failure_count=(
                    replay_technical_failure_count
                ),
                runtime_metadata_consistent=(
                    runtime_metadata_consistent
                ),
                token_counts_consistent=(
                    token_counts_consistent
                ),
                logging_complete=(
                    logging_complete
                ),
            ),
            generation_rows,
        )

    def _generation_record(
        self,
        *,
        run_id: str,
        task_id: str,
        condition: str,
        repetition_id: int,
        role: str,
        pair_id: str | None,
        source: str | None,
        event_stream_path: Path,
        event: Mapping[
            str,
            Any,
        ],
        expected_runtime: dict[
            str,
            Any,
        ],
        baseline_prompt_tokens: (
            int | None
        ),
        baseline_model_visible_chars: (
            int | None
        ),
    ) -> GenerationObservability:
        if (
            event.get(
                "event_type"
            )
            != EventType.GENERATION.value
        ):
            raise PilotObservabilityError(
                code="event_not_generation",
                message=(
                    f"Event "
                    f"{event.get('event_id')!r} "
                    "is not a generation event."
                ),
            )

        payload = event.get(
            "raw_payload"
        )

        if not isinstance(
            payload,
            Mapping,
        ):
            raise PilotObservabilityError(
                code="generation_payload_missing",
                message=(
                    "Generation raw_payload "
                    "must be a mapping."
                ),
            )

        input_record = payload.get(
            "input_record"
        )

        if not isinstance(
            input_record,
            Mapping,
        ):
            raise PilotObservabilityError(
                code="input_record_missing",
                message=(
                    "Generation payload lacks "
                    "input_record."
                ),
            )

        runtime = payload.get(
            "runtime_metadata"
        )

        if not isinstance(
            runtime,
            Mapping,
        ):
            raise PilotObservabilityError(
                code="runtime_metadata_missing",
                message=(
                    "Generation payload lacks "
                    "runtime_metadata."
                ),
            )

        prompt_tokens = self._required_int(
            payload,
            "prompt_tokens",
        )

        generation_tokens = (
            self._required_int(
                payload,
                "generation_tokens",
            )
        )

        input_prompt_tokens = (
            self._required_int(
                input_record,
                "prompt_token_count",
            )
        )

        model_visible_text = (
            self._required_string(
                input_record,
                "model_visible_text",
            )
        )

        rendered_prompt = (
            self._required_string(
                input_record,
                "rendered_prompt",
            )
        )

        prompt_tps = self._number_or_none(
            payload.get(
                "prompt_tps"
            ),
            field_name="prompt_tps",
        )

        generation_tps = (
            self._number_or_none(
                payload.get(
                    "generation_tps"
                ),
                field_name="generation_tps",
            )
        )

        peak_memory_gb = (
            self._number_or_none(
                payload.get(
                    "peak_memory_gb"
                ),
                field_name="peak_memory_gb",
            )
        )

        estimated_prompt_seconds = (
            (
                prompt_tokens
                / prompt_tps
            )
            if (
                prompt_tps
                is not None
                and prompt_tps > 0
            )
            else None
        )

        estimated_generation_seconds = (
            (
                generation_tokens
                / generation_tps
            )
            if (
                generation_tps
                is not None
                and generation_tps > 0
            )
            else None
        )

        model_visible_chars = len(
            model_visible_text
        )

        return GenerationObservability(
            run_id=run_id,
            task_id=task_id,
            condition=condition,
            repetition_id=(
                repetition_id
            ),
            role=role,
            pair_id=pair_id,
            source=source,
            event_stream_path=str(
                event_stream_path
            ),
            generation_event_id=(
                self._required_string(
                    event,
                    "event_id",
                )
            ),
            model_invocation_id=(
                event.get(
                    "model_invocation_id"
                )
            ),
            prompt_tokens=(
                prompt_tokens
            ),
            generation_tokens=(
                generation_tokens
            ),
            prompt_tps=prompt_tps,
            generation_tps=(
                generation_tps
            ),
            estimated_prompt_seconds=(
                estimated_prompt_seconds
            ),
            estimated_generation_seconds=(
                estimated_generation_seconds
            ),
            peak_memory_gb=(
                peak_memory_gb
            ),
            model_visible_chars=(
                model_visible_chars
            ),
            rendered_prompt_chars=len(
                rendered_prompt
            ),
            rendered_prompt_sha256=(
                self._required_string(
                    input_record,
                    "rendered_prompt_sha256",
                )
            ),
            prompt_token_ids_sha256=(
                self._required_string(
                    input_record,
                    "prompt_token_ids_sha256",
                )
            ),
            raw_text_sha256=(
                self._required_string(
                    payload,
                    "raw_text_sha256",
                )
            ),
            prompt_token_count_consistent=(
                prompt_tokens
                == input_prompt_tokens
            ),
            runtime_metadata_consistent=(
                dict(
                    runtime
                )
                == expected_runtime
            ),
            baseline_prompt_tokens=(
                baseline_prompt_tokens
            ),
            prompt_token_delta=(
                (
                    prompt_tokens
                    - baseline_prompt_tokens
                )
                if (
                    baseline_prompt_tokens
                    is not None
                )
                else None
            ),
            baseline_model_visible_chars=(
                baseline_model_visible_chars
            ),
            model_visible_char_delta=(
                (
                    model_visible_chars
                    - baseline_model_visible_chars
                )
                if (
                    baseline_model_visible_chars
                    is not None
                )
                else None
            ),
        )

    def _focal_chain_valid(
        self,
        *,
        events: Mapping[
            str,
            Mapping[
                str,
                Any,
            ],
        ],
        generation_event: Mapping[
            str,
            Any,
        ],
        action_event: Mapping[
            str,
            Any,
        ],
    ) -> bool:
        if (
            generation_event.get(
                "event_type"
            )
            != EventType.GENERATION.value
        ):
            return False

        if (
            action_event.get(
                "event_type"
            )
            != EventType.ACTION.value
        ):
            return False

        if (
            action_event.get(
                "parent_event_id"
            )
            != generation_event.get(
                "event_id"
            )
        ):
            return False

        audit_id = (
            generation_event.get(
                "parent_event_id"
            )
        )

        audit = events.get(
            audit_id
        )

        if (
            not isinstance(
                audit,
                Mapping,
            )
            or audit.get(
                "event_type"
            )
            != (
                EventType
                .PROVENANCE_AUDIT
                .value
            )
        ):
            return False

        context_id = audit.get(
            "parent_event_id"
        )

        context = events.get(
            context_id
        )

        return bool(
            isinstance(
                context,
                Mapping,
            )
            and (
                context.get(
                    "event_type"
                )
                == (
                    EventType
                    .CONTEXT_BUILD
                    .value
                )
            )
        )

    @staticmethod
    def _sequence_contiguous(
        events: Sequence[
            Mapping[
                str,
                Any,
            ],
        ],
    ) -> bool:
        values = [
            event.get(
                "sequence_index"
            )
            for event
            in events
        ]

        return (
            values
            == list(
                range(
                    len(
                        events
                    )
                )
            )
        )

    def _system_runtime_matches(
        self,
        events,
        expected_runtime,
    ) -> bool:
        systems = [
            event
            for event
            in events
            if (
                event.get(
                    "event_type"
                )
                == EventType.SYSTEM.value
            )
        ]

        if len(
            systems
        ) != 1:
            return False

        raw = systems[
            0
        ].get(
            "raw_payload"
        )

        if not isinstance(
            raw,
            Mapping,
        ):
            return False

        runtime = raw.get(
            "runtime_metadata"
        )

        return bool(
            isinstance(
                runtime,
                Mapping,
            )
            and dict(
                runtime
            )
            == expected_runtime
        )

    @staticmethod
    def _generation_config(
        event,
    ):
        payload = event.get(
            "raw_payload"
        )

        if not isinstance(
            payload,
            Mapping,
        ):
            raise PilotObservabilityError(
                code="generation_payload_missing",
                message=(
                    "Generation payload "
                    "is unavailable."
                ),
            )

        value = payload.get(
            "inference_config"
        )

        if not isinstance(
            value,
            Mapping,
        ):
            raise PilotObservabilityError(
                code="inference_config_missing",
                message=(
                    "Generation payload lacks "
                    "inference_config."
                ),
            )

        return deepcopy(
            dict(
                value
            )
        )

    def _aggregate(
        self,
        *,
        runs,
        generations,
        summary,
    ) -> dict[str, Any]:
        by_role: dict[
            str,
            dict[str, Any],
        ] = {}

        roles = (
            "baseline_action",
            "self_explanation",
            "targeted_replay",
            "sham_replay",
        )

        for role in roles:
            rows = [
                row
                for row
                in generations
                if row.role == role
            ]

            by_role[
                role
            ] = (
                self._generation_stats(
                    rows
                )
            )

        baseline_by_condition: dict[
            str,
            Any,
        ] = {}

        for condition in (
            "C0",
            "C1",
            "C2",
            "C3",
        ):
            rows = [
                row
                for row
                in generations
                if (
                    row.role
                    == "baseline_action"
                    and row.condition
                    == condition
                )
            ]

            baseline_by_condition[
                condition
            ] = (
                self._generation_stats(
                    rows
                )
            )

        replay_context = (
            self._replay_context_stats(
                generations
            )
        )

        intervention_count = sum(
            summary.get(
                "intervention_outcome_counts",
                {}
            ).values()
        )

        expected_generation_count = (
            (
                len(
                    runs
                )
                * 2
            )
            + (
                intervention_count
                * 2
            )
        )

        matched_unit_seconds: list[
            float
        ] = []

        baseline_map = {
            (
                row.run_id
            ): row
            for row
            in generations
            if (
                row.role
                == "baseline_action"
            )
        }

        replay_groups: dict[
            tuple[
                str,
                str,
            ],
            dict[
                str,
                GenerationObservability,
            ],
        ] = {}

        for row in generations:
            if (
                row.role
                not in {
                    "targeted_replay",
                    "sham_replay",
                }
                or row.pair_id
                is None
            ):
                continue

            replay_groups.setdefault(
                (
                    row.run_id,
                    row.pair_id,
                ),
                {},
            )[
                row.role
            ] = row

        for (
            run_id,
            _pair_id,
        ), values in (
            replay_groups.items()
        ):
            baseline = (
                baseline_map.get(
                    run_id
                )
            )

            targeted = values.get(
                "targeted_replay"
            )

            sham = values.get(
                "sham_replay"
            )

            if (
                baseline is None
                or targeted is None
                or sham is None
            ):
                continue

            components = (
                baseline,
                targeted,
                sham,
            )

            seconds: list[
                float
            ] = []

            complete = True

            for component in components:
                prompt_seconds = (
                    component
                    .estimated_prompt_seconds
                )

                generation_seconds = (
                    component
                    .estimated_generation_seconds
                )

                if (
                    prompt_seconds
                    is None
                    or generation_seconds
                    is None
                ):
                    complete = False
                    break

                seconds.append(
                    prompt_seconds
                    + generation_seconds
                )

            if complete:
                matched_unit_seconds.append(
                    sum(
                        seconds
                    )
                )

        technical_failures = sum(
            (
                run
                .baseline_technical_failure_count
                + run
                .replay_technical_failure_count
            )
            for run in runs
        )

        return {
            "integrity": {
                "runs": len(
                    runs
                ),
                "logging_complete_runs": sum(
                    1
                    for run
                    in runs
                    if run.logging_complete
                ),
                "sequence_failure_runs": sum(
                    1
                    for run
                    in runs
                    if not (
                        run
                        .baseline_sequence_contiguous
                        and run
                        .replay_sequences_contiguous
                    )
                ),
                "chain_failure_runs": sum(
                    1
                    for run
                    in runs
                    if not (
                        run
                        .baseline_focal_chain_valid
                        and run
                        .explanation_chain_valid
                        and run
                        .replay_chains_valid
                    )
                ),
                "snapshot_failure_runs": sum(
                    1
                    for run
                    in runs
                    if not (
                        run
                        .snapshot_before_action
                        and run
                        .snapshot_identity_valid
                    )
                ),
                "explanation_order_failure_runs": sum(
                    1
                    for run
                    in runs
                    if not (
                        run
                        .explanation_after_action
                    )
                ),
                "baseline_intervention_leakage_runs": sum(
                    1
                    for run
                    in runs
                    if (
                        run
                        .baseline_intervention_event_count
                        > 0
                    )
                ),
                "replay_config_mismatch_runs": sum(
                    1
                    for run
                    in runs
                    if not (
                        run
                        .replay_inference_configs_match_baseline
                    )
                ),
                "runtime_metadata_mismatch_runs": sum(
                    1
                    for run
                    in runs
                    if not (
                        run
                        .runtime_metadata_consistent
                    )
                ),
                "token_count_mismatch_runs": sum(
                    1
                    for run
                    in runs
                    if not (
                        run
                        .token_counts_consistent
                    )
                ),
                "technical_failure_events": (
                    technical_failures
                ),
                "replay_streams_checked": sum(
                    run
                    .replay_streams_checked
                    for run
                    in runs
                ),
            },
            "generation_counts": {
                "observed": len(
                    generations
                ),
                "expected": (
                    expected_generation_count
                ),
                "by_role": {
                    role: len(
                        [
                            row
                            for row
                            in generations
                            if (
                                row.role
                                == role
                            )
                        ]
                    )
                    for role
                    in roles
                },
            },
            "generation_metrics_by_role": (
                by_role
            ),
            "baseline_context_by_condition": (
                baseline_by_condition
            ),
            "replay_context_deltas": (
                replay_context
            ),
            "matched_unit_runtime": {
                "measurement": (
                    "derived_from_logged_tokens_and_tps"
                ),
                "n_complete_units": len(
                    matched_unit_seconds
                ),
                "mean_estimated_seconds": (
                    self._mean_or_none(
                        matched_unit_seconds
                    )
                ),
                "median_estimated_seconds": (
                    self._median_or_none(
                        matched_unit_seconds
                    )
                ),
                "min_estimated_seconds": (
                    min(
                        matched_unit_seconds
                    )
                    if matched_unit_seconds
                    else None
                ),
                "max_estimated_seconds": (
                    max(
                        matched_unit_seconds
                    )
                    if matched_unit_seconds
                    else None
                ),
                "wall_clock_logged": False,
            },
        }

    def _generation_stats(
        self,
        rows,
    ) -> dict[str, Any]:
        prompt_tokens = [
            row.prompt_tokens
            for row in rows
        ]

        generation_tokens = [
            row.generation_tokens
            for row in rows
        ]

        prompt_tps = [
            row.prompt_tps
            for row in rows
            if row.prompt_tps
            is not None
        ]

        generation_tps = [
            row.generation_tps
            for row in rows
            if row.generation_tps
            is not None
        ]

        peak_memory = [
            row.peak_memory_gb
            for row in rows
            if row.peak_memory_gb
            is not None
        ]

        estimated_seconds = [
            (
                row
                .estimated_prompt_seconds
                + row
                .estimated_generation_seconds
            )
            for row in rows
            if (
                row
                .estimated_prompt_seconds
                is not None
                and row
                .estimated_generation_seconds
                is not None
            )
        ]

        return {
            "n": len(
                rows
            ),
            "prompt_tokens": (
                self._summary(
                    prompt_tokens
                )
            ),
            "generation_tokens": (
                self._summary(
                    generation_tokens
                )
            ),
            "prompt_tps": (
                self._summary(
                    prompt_tps
                )
            ),
            "generation_tps": (
                self._summary(
                    generation_tps
                )
            ),
            "peak_memory_gb": (
                self._summary(
                    peak_memory
                )
            ),
            "estimated_inference_seconds": (
                self._summary(
                    estimated_seconds
                )
            ),
        }

    def _replay_context_stats(
        self,
        generations,
    ) -> dict[str, Any]:
        result: dict[
            str,
            Any,
        ] = {}

        for role in (
            "targeted_replay",
            "sham_replay",
        ):
            rows = [
                row
                for row
                in generations
                if row.role == role
            ]

            deltas = [
                row.prompt_token_delta
                for row in rows
                if (
                    row.prompt_token_delta
                    is not None
                )
            ]

            char_deltas = [
                row.model_visible_char_delta
                for row in rows
                if (
                    row.model_visible_char_delta
                    is not None
                )
            ]

            exact = sum(
                1
                for value
                in deltas
                if value == 0
            )

            by_source: dict[
                str,
                Any,
            ] = {}

            for source in (
                "P",
                "T",
                "H",
                "M",
                "E",
            ):
                source_rows = [
                    row
                    for row
                    in rows
                    if row.source == source
                ]

                if not source_rows:
                    continue

                source_deltas = [
                    row.prompt_token_delta
                    for row
                    in source_rows
                    if (
                        row.prompt_token_delta
                        is not None
                    )
                ]

                by_source[
                    source
                ] = {
                    "n": len(
                        source_rows
                    ),
                    "mean_token_delta": (
                        self._mean_or_none(
                            source_deltas
                        )
                    ),
                    "max_absolute_token_delta": (
                        max(
                            (
                                abs(
                                    value
                                )
                                for value
                                in source_deltas
                            ),
                            default=None,
                        )
                    ),
                    "exact_token_match_rate": (
                        (
                            sum(
                                1
                                for value
                                in source_deltas
                                if value == 0
                            )
                            / len(
                                source_deltas
                            )
                        )
                        if source_deltas
                        else None
                    ),
                }

            result[
                role
            ] = {
                "n": len(
                    rows
                ),
                "prompt_token_delta": (
                    self._summary(
                        deltas
                    )
                ),
                "model_visible_char_delta": (
                    self._summary(
                        char_deltas
                    )
                ),
                "exact_prompt_token_match_rate": (
                    (
                        exact
                        / len(
                            deltas
                        )
                    )
                    if deltas
                    else None
                ),
                "by_source": (
                    by_source
                ),
            }

        return result

    @staticmethod
    def _summary(
        values,
    ) -> dict[str, Any]:
        values = list(
            values
        )

        if not values:
            return {
                "n": 0,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            }

        return {
            "n": len(
                values
            ),
            "mean": float(
                mean(
                    values
                )
            ),
            "median": float(
                median(
                    values
                )
            ),
            "min": min(
                values
            ),
            "max": max(
                values
            ),
        }

    @staticmethod
    def _mean_or_none(
        values,
    ):
        values = list(
            values
        )

        if not values:
            return None

        return float(
            mean(
                values
            )
        )

    @staticmethod
    def _median_or_none(
        values,
    ):
        values = list(
            values
        )

        if not values:
            return None

        return float(
            median(
                values
            )
        )

    @staticmethod
    def _event_by_id(
        events,
        event_id,
        *,
        run_id,
    ):
        value = events.get(
            event_id
        )

        if not isinstance(
            value,
            Mapping,
        ):
            raise PilotObservabilityError(
                code="event_not_found",
                message=(
                    f"Run {run_id!r} "
                    f"does not contain event "
                    f"{event_id!r}."
                ),
            )

        return value

    @staticmethod
    def _sequence_index(
        event,
    ) -> int:
        value = event.get(
            "sequence_index"
        )

        if not isinstance(
            value,
            int,
        ):
            raise PilotObservabilityError(
                code="invalid_sequence_index",
                message=(
                    "Event sequence_index "
                    "must be an integer."
                ),
            )

        return value

    @staticmethod
    def _read_events(
        path: Path,
    ):
        if not path.is_file():
            raise PilotObservabilityError(
                code="event_stream_missing",
                message=(
                    f"Event stream "
                    f"{str(path)!r} "
                    "does not exist."
                ),
            )

        values = []

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                for line_number, line in enumerate(
                    handle,
                    start=1,
                ):
                    if not line.strip():
                        continue

                    value = json.loads(
                        line
                    )

                    if not isinstance(
                        value,
                        dict,
                    ):
                        raise PilotObservabilityError(
                            code="invalid_event",
                            message=(
                                f"Event stream "
                                f"{str(path)!r}, "
                                f"line {line_number} "
                                "does not contain "
                                "a JSON object."
                            ),
                        )

                    values.append(
                        value
                    )

        except json.JSONDecodeError as exc:
            raise PilotObservabilityError(
                code="invalid_event_json",
                message=(
                    f"Event stream "
                    f"{str(path)!r} "
                    "contains invalid JSON."
                ),
            ) from exc

        return values

    @staticmethod
    def _read_json(
        path: Path,
    ):
        if not path.is_file():
            raise PilotObservabilityError(
                code="artifact_missing",
                message=(
                    f"Artifact "
                    f"{str(path)!r} "
                    "does not exist."
                ),
            )

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                value = json.load(
                    handle
                )

        except json.JSONDecodeError as exc:
            raise PilotObservabilityError(
                code="invalid_artifact_json",
                message=(
                    f"Artifact "
                    f"{str(path)!r} "
                    "contains invalid JSON."
                ),
            ) from exc

        if not isinstance(
            value,
            dict,
        ):
            raise PilotObservabilityError(
                code="artifact_root_not_object",
                message=(
                    f"Artifact "
                    f"{str(path)!r} "
                    "must contain a JSON object."
                ),
            )

        return value

    @staticmethod
    def _required_string(
        mapping,
        field_name,
    ):
        value = mapping.get(
            field_name
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value
        ):
            raise PilotObservabilityError(
                code="missing_required_string",
                message=(
                    f"Required field "
                    f"{field_name!r} "
                    "must be a "
                    "non-empty string."
                ),
            )

        return value

    @staticmethod
    def _required_int(
        mapping,
        field_name,
    ):
        value = mapping.get(
            field_name
        )

        if not isinstance(
            value,
            int,
        ):
            raise PilotObservabilityError(
                code="missing_required_integer",
                message=(
                    f"Required field "
                    f"{field_name!r} "
                    "must be an integer."
                ),
            )

        return value

    @staticmethod
    def _number_or_none(
        value,
        *,
        field_name,
    ):
        if value is None:
            return None

        if (
            not isinstance(
                value,
                (
                    int,
                    float,
                ),
            )
            or isinstance(
                value,
                bool,
            )
        ):
            raise PilotObservabilityError(
                code="invalid_numeric_field",
                message=(
                    f"{field_name} must "
                    "be numeric or null."
                ),
            )

        return float(
            value
        )


class PilotObservabilityWriter:
    """Write processed observability datasets without touching raw logs."""

    def write(
        self,
        *,
        audit: PilotObservabilityAudit,
        processed_directory: str | Path,
        report_path: str | Path,
        overwrite: bool = False,
    ) -> PilotObservabilityOutputs:
        processed = Path(
            processed_directory
        )

        report = Path(
            report_path
        )

        processed.mkdir(
            parents=True,
            exist_ok=True,
        )

        report.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        generation_path = (
            processed
            / "generation_observability.csv"
        )

        run_path = (
            processed
            / "run_observability.csv"
        )

        summary_path = (
            processed
            / "pilot_observability.json"
        )

        targets = (
            generation_path,
            run_path,
            summary_path,
            report,
        )

        if not overwrite:
            existing = [
                str(
                    path
                )
                for path in targets
                if path.exists()
            ]

            if existing:
                raise PilotObservabilityError(
                    code="observability_output_exists",
                    message=(
                        "Observability output "
                        "already exists: "
                        + ", ".join(
                            existing
                        )
                    ),
                )

        self._write_csv(
            generation_path,
            [
                item.to_dict()
                for item
                in audit.generations
            ],
        )

        self._write_csv(
            run_path,
            [
                item.to_dict()
                for item
                in audit.runs
            ],
        )

        self._write_json(
            summary_path,
            audit.to_dict(),
        )

        self._write_text(
            report,
            self._render_report(
                audit
            ),
        )

        return PilotObservabilityOutputs(
            generation_csv=str(
                generation_path
            ),
            run_csv=str(
                run_path
            ),
            summary_json=str(
                summary_path
            ),
            report_markdown=str(
                report
            ),
        )

    @staticmethod
    def _write_csv(
        path,
        rows,
    ):
        rows = list(
            rows
        )

        if not rows:
            path.write_text(
                "",
                encoding="utf-8",
            )
            return

        fieldnames = list(
            rows[
                0
            ].keys()
        )

        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    row
                )

    @staticmethod
    def _write_json(
        path,
        value,
    ):
        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )

            handle.write(
                "\n"
            )

    @staticmethod
    def _write_text(
        path,
        value,
    ):
        path.write_text(
            value,
            encoding="utf-8",
        )

    def _render_report(
        self,
        audit,
    ):
        integrity = (
            audit
            .aggregates[
                "integrity"
            ]
        )

        counts = (
            audit
            .aggregates[
                "generation_counts"
            ]
        )

        matched = (
            audit
            .aggregates[
                "matched_unit_runtime"
            ]
        )

        targeted = (
            audit
            .aggregates[
                "replay_context_deltas"
            ][
                "targeted_replay"
            ]
        )

        sham = (
            audit
            .aggregates[
                "replay_context_deltas"
            ][
                "sham_replay"
            ]
        )

        return "\n".join(
            [
                "# Pilot Observability Audit",
                "",
                (
                    f"- Experiment: "
                    f"`{audit.experiment_id}`"
                ),
                (
                    f"- Plan SHA256: "
                    f"`{audit.plan_sha256}`"
                ),
                "",
                "## Logging integrity",
                "",
                (
                    f"- Runs audited: "
                    f"{integrity['runs']}"
                ),
                (
                    f"- Logging-complete runs: "
                    f"{integrity['logging_complete_runs']}"
                ),
                (
                    f"- Technical-failure events: "
                    f"{integrity['technical_failure_events']}"
                ),
                (
                    f"- Sequence failures: "
                    f"{integrity['sequence_failure_runs']}"
                ),
                (
                    f"- Event-chain failures: "
                    f"{integrity['chain_failure_runs']}"
                ),
                (
                    f"- Snapshot failures: "
                    f"{integrity['snapshot_failure_runs']}"
                ),
                (
                    f"- Baseline intervention leakage: "
                    f"{integrity['baseline_intervention_leakage_runs']}"
                ),
                (
                    f"- Replay configuration mismatches: "
                    f"{integrity['replay_config_mismatch_runs']}"
                ),
                (
                    f"- Runtime metadata mismatches: "
                    f"{integrity['runtime_metadata_mismatch_runs']}"
                ),
                "",
                "## Generation completeness",
                "",
                (
                    f"- Expected generations: "
                    f"{counts['expected']}"
                ),
                (
                    f"- Observed generations: "
                    f"{counts['observed']}"
                ),
                (
                    f"- Baseline actions: "
                    f"{counts['by_role']['baseline_action']}"
                ),
                (
                    f"- Self-explanations: "
                    f"{counts['by_role']['self_explanation']}"
                ),
                (
                    f"- Targeted replays: "
                    f"{counts['by_role']['targeted_replay']}"
                ),
                (
                    f"- Sham replays: "
                    f"{counts['by_role']['sham_replay']}"
                ),
                "",
                "## Context-length diagnostics",
                "",
                (
                    "- Targeted exact prompt-token "
                    "match rate: "
                    f"{self._format_number(targeted['exact_prompt_token_match_rate'])}"
                ),
                (
                    "- Sham exact prompt-token "
                    "match rate: "
                    f"{self._format_number(sham['exact_prompt_token_match_rate'])}"
                ),
                (
                    "- Targeted mean token delta: "
                    f"{self._format_number(targeted['prompt_token_delta']['mean'])}"
                ),
                (
                    "- Sham mean token delta: "
                    f"{self._format_number(sham['prompt_token_delta']['mean'])}"
                ),
                "",
                "## Runtime",
                "",
                (
                    "- Matched-unit runtime metric: "
                    "`derived_from_logged_tokens_and_tps`"
                ),
                (
                    f"- Complete matched units: "
                    f"{matched['n_complete_units']}"
                ),
                (
                    "- Mean estimated matched-unit seconds: "
                    f"{self._format_number(matched['mean_estimated_seconds'])}"
                ),
                (
                    "- Median estimated matched-unit seconds: "
                    f"{self._format_number(matched['median_estimated_seconds'])}"
                ),
                "",
                "**Important:** exact orchestration wall-clock "
                "time was not logged by the pilot runner. "
                "The timing figures above are derived from "
                "logged token counts and throughput.",
                "",
            ]
        )

    @staticmethod
    def _format_number(
        value,
    ):
        if value is None:
            return "NA"

        if isinstance(
            value,
            float,
        ):
            return f"{value:.3f}"

        return str(
            value
        )