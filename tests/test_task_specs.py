__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import json
from copy import deepcopy

import pytest

from src.logging import (
    Condition,
    SourceClass,
    TaskClassification,
)
from src.tasks import (
    TASK_SPEC_SCHEMA_VERSION,
    TaskFamily,
    TaskSpecError,
    load_task_suite,
    parse_task_suite,
)


def valid_task() -> dict:
    return {
        "task_id": "tool_diagnostic_001",
        "task_version": "0.1.0",
        "family": (
            "tool_dependence"
        ),
        "class": "diagnostic",
        "conditions": [
            "C0",
            "C1",
            "C2",
            "C3",
        ],
        "task_prompt": (
            "If a current threshold is available and "
            "equals 7, select route B. Otherwise "
            "select route A."
        ),
        "focal_decision": {
            "id": "final_route",
            "fields": [
                {
                    "name": "action",
                    "json_type": (
                        "string"
                    ),
                    "allowed_values": [
                        "select_route"
                    ],
                },
                {
                    "name": "value",
                    "json_type": (
                        "string"
                    ),
                    "allowed_values": [
                        "A",
                        "B",
                    ],
                },
            ],
            "success_by_condition": {
                "C0": {
                    "action": (
                        "select_route"
                    ),
                    "value": "A",
                },
                "C1": {
                    "action": (
                        "select_route"
                    ),
                    "value": "B",
                },
                "C2": {
                    "action": (
                        "select_route"
                    ),
                    "value": "B",
                },
                "C3": {
                    "action": (
                        "select_route"
                    ),
                    "value": "B",
                },
            },
        },
        "available_sources": {
            "C0": [
                "P"
            ],
            "C1": [
                "P",
                "T",
            ],
            "C2": [
                "P",
                "T",
            ],
            "C3": [
                "P",
                "T",
            ],
        },
        "fixtures": {
            "tools": [
                {
                    "fixture_id": (
                        "threshold_tool"
                    ),
                    "tool_name": (
                        "key_value"
                    ),
                    "request": {
                        "key": (
                            "threshold"
                        )
                    },
                    "state": {
                        "threshold": 7,
                    },
                    "metadata": {
                        "role": (
                            "decision_relevant"
                        )
                    },
                },
                {
                    "fixture_id": (
                        "label_tool"
                    ),
                    "tool_name": (
                        "key_value"
                    ),
                    "request": {
                        "key": "label"
                    },
                    "state": {
                        "label": "blue",
                    },
                    "metadata": {
                        "role": (
                            "sham"
                        )
                    },
                },
            ],
            "history": [],
            "memory": [],
            "environment": None,
        },
        "intervention_pairs": [
            {
                "pair_id": (
                    "threshold_pair"
                ),
                "source": "T",
                "eligible_conditions": [
                    "C1",
                    "C2",
                    "C3",
                ],
                "targeted": {
                    "operation": (
                        "replace"
                    ),
                    "target_fixture_id": (
                        "threshold_tool"
                    ),
                    "target_path": None,
                    "replacement_value": {
                        "threshold": 9
                    },
                    "length_match_required": (
                        True
                    ),
                },
                "sham": {
                    "operation": (
                        "replace"
                    ),
                    "target_fixture_id": (
                        "label_tool"
                    ),
                    "target_path": None,
                    "replacement_value": {
                        "label": "cyan"
                    },
                    "length_match_required": (
                        True
                    ),
                },
            }
        ],
        "causal_ground_truth": {
            "design_intent": [
                "T"
            ],
            "empirical_status": (
                "determined_only_by_intervention"
            ),
        },
        "notes": (
            "Diagnostic tool-channel task."
        ),
    }


def valid_suite() -> dict:
    return {
        "metadata": {
            "author": (
                "Jason M. Pittman"
            ),
            "date": (
                "August 24, 2026"
            ),
            "copyright": (
                "Copyright 2026"
            ),
            "credits": [
                "Jason M. Pittman"
            ],
            "license": "MIT License",
            "version": "0.1.0",
            "maintainer": (
                "Jason M. Pittman"
            ),
            "status": "Research",
        },
        "schema_version": (
            TASK_SPEC_SCHEMA_VERSION
        ),
        "suite_id": (
            "pilot_suite"
        ),
        "suite_version": "0.1.0",
        "tasks": [
            valid_task()
        ],
    }


def test_valid_suite_parses() -> None:
    suite = parse_task_suite(
        valid_suite()
    )

    assert (
        suite.suite_id
        == "pilot_suite"
    )

    assert len(
        suite.tasks
    ) == 1


def test_task_family_parses() -> None:
    task = (
        parse_task_suite(
            valid_suite()
        )
        .tasks[0]
    )

    assert (
        task.family
        == TaskFamily.TOOL_DEPENDENCE
    )

    assert (
        task.task_classification
        == TaskClassification.DIAGNOSTIC
    )


def test_condition_source_mapping_parses() -> None:
    task = (
        parse_task_suite(
            valid_suite()
        )
        .tasks[0]
    )

    assert (
        task.available_sources[
            Condition.C1
        ]
        == (
            SourceClass.P,
            SourceClass.T,
        )
    )


def test_c0_rejects_tool_source() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ][0][
        "available_sources"
    ][
        "C0"
    ] = [
        "P",
        "T",
    ]

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "condition_source_violation"
    )


def test_c1_rejects_history_source() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ][0][
        "available_sources"
    ][
        "C1"
    ] = [
        "P",
        "H",
    ]

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "condition_source_violation"
    )


def test_c2_rejects_memory_source() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ][0][
        "available_sources"
    ][
        "C2"
    ] = [
        "P",
        "M",
    ]

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "condition_source_violation"
    )


def test_c0_rejects_environment_source() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ][0][
        "available_sources"
    ][
        "C0"
    ] = [
        "P",
        "E",
    ]

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "condition_source_violation"
    )


def test_environment_source_requires_environment() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ][0][
        "available_sources"
    ][
        "C1"
    ] = [
        "P",
        "T",
        "E",
    ]

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "environment_source_without_state"
    )


def test_expected_action_must_match_schema() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ][0][
        "focal_decision"
    ][
        "success_by_condition"
    ][
        "C1"
    ][
        "value"
    ] = "Z"

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "invalid_expected_action"
    )


def test_intervention_condition_requires_visible_source() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ][0][
        "intervention_pairs"
    ][0][
        "eligible_conditions"
    ] = [
        "C0",
        "C1",
    ]

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "intervention_source_unavailable"
    )


def test_intervention_fixture_must_exist() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ][0][
        "intervention_pairs"
    ][0][
        "targeted"
    ][
        "target_fixture_id"
    ] = "missing"

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "intervention_fixture_not_found"
    )


def test_intervention_fixture_must_match_source() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ][0][
        "fixtures"
    ][
        "memory"
    ] = [
        {
            "fixture_id": "memory_1",
            "key": "route",
            "value": "B",
            "metadata": {},
        }
    ]

    raw[
        "tasks"
    ][0][
        "available_sources"
    ][
        "C3"
    ] = [
        "P",
        "T",
        "M",
    ]

    raw[
        "tasks"
    ][0][
        "intervention_pairs"
    ][0][
        "targeted"
    ][
        "target_fixture_id"
    ] = "memory_1"

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "intervention_fixture_source_mismatch"
    )


def test_fixture_ids_are_globally_unique() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ][0][
        "fixtures"
    ][
        "memory"
    ] = [
        {
            "fixture_id": (
                "threshold_tool"
            ),
            "key": "route",
            "value": "B",
            "metadata": {},
        }
    ]

    raw[
        "tasks"
    ][0][
        "available_sources"
    ][
        "C3"
    ] = [
        "P",
        "T",
        "M",
    ]

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "duplicate_fixture_id"
    )


def test_empirical_status_is_fixed() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ][0][
        "causal_ground_truth"
    ][
        "empirical_status"
    ] = (
        "T_is_causally_relevant"
    )

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "invalid_empirical_status"
    )


def test_duplicate_task_id_is_rejected() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ].append(
        deepcopy(
            raw[
                "tasks"
            ][0]
        )
    )

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "duplicate_task_id"
    )


def test_loader_reads_json_file(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "tasks.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            valid_suite(),
            handle,
        )

    suite = load_task_suite(
        path
    )

    assert (
        suite.tasks[0].task_id
        == "tool_diagnostic_001"
    )


def test_unknown_field_is_rejected() -> None:
    raw = valid_suite()

    raw[
        "tasks"
    ][0][
        "unexpected"
    ] = True

    with pytest.raises(
        TaskSpecError
    ) as exc_info:
        parse_task_suite(
            raw
        )

    assert (
        exc_info.value.code
        == "task_schema_field_mismatch"
    )