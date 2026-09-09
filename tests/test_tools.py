__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from copy import deepcopy

import pytest

from src.tools import (
    CalculatorTool,
    EnvironmentQueryTool,
    KeyValueLookupTool,
    StateModifyTool,
    SyntheticFileLookupTool,
    ToolExecutionError,
)


def test_calculator_is_deterministic() -> None:
    tool = CalculatorTool()

    request = {
        "operation": "add",
        "operands": [7, 5],
    }

    first = tool.execute(request)
    second = tool.execute(request)

    assert first.normalized_result == {"value": "12"}
    assert first.normalized_result == second.normalized_result


def test_calculator_decimal_normalization() -> None:
    tool = CalculatorTool()

    result = tool.execute(
        {
            "operation": "divide",
            "operands": [1, 4],
        }
    )

    assert result.normalized_result == {"value": "0.25"}


def test_calculator_rejects_division_by_zero() -> None:
    tool = CalculatorTool()

    with pytest.raises(ToolExecutionError) as exc_info:
        tool.execute(
            {
                "operation": "divide",
                "operands": [1, 0],
            }
        )

    assert exc_info.value.code == "division_by_zero"


def test_key_value_lookup_finds_existing_key() -> None:
    tool = KeyValueLookupTool()

    state = {
        "threshold": 7,
        "mode": "test",
    }

    result = tool.execute(
        {"key": "threshold"},
        state,
    )

    assert result.normalized_result == {
        "key": "threshold",
        "found": True,
        "value": 7,
    }


def test_key_value_lookup_missing_key_is_valid_result() -> None:
    tool = KeyValueLookupTool()

    result = tool.execute(
        {"key": "missing"},
        {"threshold": 7},
    )

    assert result.normalized_result == {
        "key": "missing",
        "found": False,
        "value": None,
    }


def test_synthetic_file_lookup_reads_controlled_file(tmp_path) -> None:
    document_root = tmp_path / "documents"
    document_root.mkdir()

    document = document_root / "example.txt"
    document.write_text("threshold = 7\n", encoding="utf-8")

    tool = SyntheticFileLookupTool(document_root)

    result = tool.execute(
        {"path": "example.txt"}
    )

    assert result.normalized_result["content"] == "threshold = 7\n"
    assert len(result.normalized_result["sha256"]) == 64


def test_synthetic_file_lookup_rejects_path_escape(tmp_path) -> None:
    document_root = tmp_path / "documents"
    document_root.mkdir()

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")

    tool = SyntheticFileLookupTool(document_root)

    with pytest.raises(ToolExecutionError) as exc_info:
        tool.execute(
            {"path": "../outside.txt"}
        )

    assert exc_info.value.code == "path_escape"


def test_environment_query_reads_nested_state() -> None:
    tool = EnvironmentQueryTool()

    state = {
        "entities": {
            "door_1": {
                "status": "locked",
            }
        }
    }

    result = tool.execute(
        {
            "path": [
                "entities",
                "door_1",
                "status",
            ]
        },
        state,
    )

    assert result.normalized_result == {
        "path": [
            "entities",
            "door_1",
            "status",
        ],
        "found": True,
        "value": "locked",
    }


def test_environment_query_does_not_modify_state() -> None:
    tool = EnvironmentQueryTool()

    state = {
        "entities": {
            "door_1": {
                "status": "locked",
            }
        }
    }

    original = deepcopy(state)

    tool.execute(
        {
            "path": [
                "entities",
                "door_1",
                "status",
            ]
        },
        state,
    )

    assert state == original


def test_state_modify_returns_modified_copy() -> None:
    tool = StateModifyTool()

    state = {
        "entities": {
            "door_1": {
                "status": "locked",
            }
        }
    }

    original = deepcopy(state)

    result = tool.execute(
        {
            "path": [
                "entities",
                "door_1",
                "status",
            ],
            "value": "unlocked",
        },
        state,
    )

    assert state == original

    assert (
        result.updated_state["entities"]["door_1"]["status"]
        == "unlocked"
    )

    assert result.state_changed is True


def test_state_modify_same_value_reports_no_change() -> None:
    tool = StateModifyTool()

    state = {
        "mode": "ready",
    }

    result = tool.execute(
        {
            "path": ["mode"],
            "value": "ready",
        },
        state,
    )

    assert result.state_changed is False
    assert result.updated_state == state


def test_tool_results_are_repeatable() -> None:
    tool = EnvironmentQueryTool()

    state = {
        "entities": {
            "container_1": {
                "open": False,
            }
        }
    }

    request = {
        "path": [
            "entities",
            "container_1",
            "open",
        ]
    }

    results = [
        tool.execute(request, state).to_dict()
        for _ in range(10)
    ]

    assert all(result == results[0] for result in results)