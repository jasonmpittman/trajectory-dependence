__author__ = "Jason M. Pittman"
__date__ = "August 23, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import math

import pytest

from src.normalization import (
    ActionField,
    ActionParseError,
    ActionSchemaError,
    FocalActionSchema,
    StructuredActionParser,
    canonicalize_action,
    value_matches_json_type,
)


def make_schema() -> FocalActionSchema:
    return FocalActionSchema(
        focal_decision_id="final_route",
        fields=(
            ActionField(
                name="action",
                json_type="string",
                allowed_values=(
                    "select_route",
                ),
            ),
            ActionField(
                name="value",
                json_type="string",
                allowed_values=(
                    "A",
                    "B",
                    "C",
                ),
            ),
        ),
    )


def test_action_field_rejects_empty_name() -> None:
    with pytest.raises(
        ActionSchemaError
    ) as exc_info:
        ActionField(
            name="",
            json_type="string",
        )

    assert (
        exc_info.value.code
        == "invalid_field_name"
    )


def test_action_field_rejects_unknown_type() -> None:
    with pytest.raises(
        ActionSchemaError
    ) as exc_info:
        ActionField(
            name="value",
            json_type="banana",
        )

    assert (
        exc_info.value.code
        == "invalid_json_type"
    )


def test_allowed_values_must_match_declared_type() -> None:
    with pytest.raises(
        ActionSchemaError
    ) as exc_info:
        ActionField(
            name="value",
            json_type="integer",
            allowed_values=(
                "not-an-integer",
            ),
        )

    assert (
        exc_info.value.code
        == "allowed_value_type_mismatch"
    )


def test_schema_requires_focal_decision_id() -> None:
    with pytest.raises(
        ActionSchemaError
    ) as exc_info:
        FocalActionSchema(
            focal_decision_id="",
            fields=(
                ActionField(
                    name="action",
                    json_type="string",
                ),
            ),
        )

    assert (
        exc_info.value.code
        == "invalid_focal_decision_id"
    )


def test_schema_requires_fields() -> None:
    with pytest.raises(
        ActionSchemaError
    ) as exc_info:
        FocalActionSchema(
            focal_decision_id="decision",
            fields=(),
        )

    assert (
        exc_info.value.code
        == "empty_action_schema"
    )


def test_schema_rejects_duplicate_fields() -> None:
    with pytest.raises(
        ActionSchemaError
    ) as exc_info:
        FocalActionSchema(
            focal_decision_id="decision",
            fields=(
                ActionField(
                    name="value",
                    json_type="string",
                ),
                ActionField(
                    name="value",
                    json_type="string",
                ),
            ),
        )

    assert (
        exc_info.value.code
        == "duplicate_action_field"
    )


@pytest.mark.parametrize(
    (
        "value",
        "json_type",
        "expected",
    ),
    [
        ("x", "string", True),
        (1, "integer", True),
        (True, "integer", False),
        (1, "number", True),
        (1.5, "number", True),
        (False, "number", False),
        (True, "boolean", True),
        ([], "array", True),
        ({}, "object", True),
        (None, "null", True),
        ("anything", "any", True),
    ],
)
def test_strict_json_type_matching(
    value,
    json_type,
    expected,
) -> None:
    assert (
        value_matches_json_type(
            value,
            json_type,
        )
        is expected
    )


def test_valid_action_parses() -> None:
    result = (
        StructuredActionParser()
        .parse(
            raw_text=(
                '{"action":"select_route","value":"B"}'
            ),
            schema=make_schema(),
        )
    )

    assert (
        result.normalized_action.payload
        == {
            "action": "select_route",
            "value": "B",
        }
    )


def test_surrounding_whitespace_is_permitted() -> None:
    result = (
        StructuredActionParser()
        .parse(
            raw_text=(
                '  \n{"action":"select_route","value":"B"}\n  '
            ),
            schema=make_schema(),
        )
    )

    assert (
        result.normalized_action.payload[
            "value"
        ]
        == "B"
    )


def test_key_order_normalizes_identically() -> None:
    parser = StructuredActionParser()
    schema = make_schema()

    first = parser.parse(
        raw_text=(
            '{"action":"select_route","value":"B"}'
        ),
        schema=schema,
    )

    second = parser.parse(
        raw_text=(
            '{"value":"B","action":"select_route"}'
        ),
        schema=schema,
    )

    assert (
        first.normalized_action.canonical_json
        == second.normalized_action.canonical_json
    )

    assert (
        first.normalized_action.canonical_sha256
        == second.normalized_action.canonical_sha256
    )

    assert (
        first.normalized_action
        .behaviorally_equivalent(
            second.normalized_action
        )
        is True
    )


def test_different_action_is_not_equivalent() -> None:
    parser = StructuredActionParser()
    schema = make_schema()

    first = parser.parse(
        raw_text=(
            '{"action":"select_route","value":"A"}'
        ),
        schema=schema,
    )

    second = parser.parse(
        raw_text=(
            '{"action":"select_route","value":"B"}'
        ),
        schema=schema,
    )

    assert (
        first.normalized_action
        .behaviorally_equivalent(
            second.normalized_action
        )
        is False
    )


def test_different_focal_decision_is_not_equivalent() -> None:
    first = canonicalize_action(
        focal_decision_id="decision_a",
        payload={
            "action": "choose",
        },
    )

    second = canonicalize_action(
        focal_decision_id="decision_b",
        payload={
            "action": "choose",
        },
    )

    assert (
        first.behaviorally_equivalent(
            second
        )
        is False
    )


def test_invalid_json_is_rejected() -> None:
    with pytest.raises(
        ActionParseError
    ) as exc_info:
        StructuredActionParser().parse(
            raw_text=(
                "{'action':'select_route'}"
            ),
            schema=make_schema(),
        )

    assert (
        exc_info.value.code
        == "invalid_action_json"
    )


def test_markdown_fence_is_rejected() -> None:
    with pytest.raises(
        ActionParseError
    ) as exc_info:
        StructuredActionParser().parse(
            raw_text=(
                "```json\n"
                '{"action":"select_route","value":"B"}'
                "\n```"
            ),
            schema=make_schema(),
        )

    assert (
        exc_info.value.code
        == "invalid_action_json"
    )


def test_prose_around_json_is_rejected() -> None:
    with pytest.raises(
        ActionParseError
    ) as exc_info:
        StructuredActionParser().parse(
            raw_text=(
                "I choose "
                '{"action":"select_route","value":"B"}'
            ),
            schema=make_schema(),
        )

    assert (
        exc_info.value.code
        == "invalid_action_json"
    )


def test_array_root_is_rejected() -> None:
    with pytest.raises(
        ActionParseError
    ) as exc_info:
        StructuredActionParser().parse(
            raw_text='["A","B"]',
            schema=make_schema(),
        )

    assert (
        exc_info.value.code
        == "action_root_not_object"
    )


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(
        ActionParseError
    ) as exc_info:
        StructuredActionParser().parse(
            raw_text=(
                '{"action":"select_route"}'
            ),
            schema=make_schema(),
        )

    assert (
        exc_info.value.code
        == "missing_action_field"
    )


def test_unexpected_field_is_rejected() -> None:
    with pytest.raises(
        ActionParseError
    ) as exc_info:
        StructuredActionParser().parse(
            raw_text=(
                '{"action":"select_route",'
                '"value":"B","reason":"because"}'
            ),
            schema=make_schema(),
        )

    assert (
        exc_info.value.code
        == "unexpected_action_field"
    )


def test_optional_field_may_be_absent() -> None:
    schema = FocalActionSchema(
        focal_decision_id="decision",
        fields=(
            ActionField(
                name="action",
                json_type="string",
            ),
            ActionField(
                name="note",
                json_type="string",
                required=False,
            ),
        ),
    )

    result = (
        StructuredActionParser()
        .parse(
            raw_text=(
                '{"action":"choose"}'
            ),
            schema=schema,
        )
    )

    assert (
        result.normalized_action.payload
        == {
            "action": "choose",
        }
    )


def test_field_type_mismatch_is_rejected() -> None:
    with pytest.raises(
        ActionParseError
    ) as exc_info:
        StructuredActionParser().parse(
            raw_text=(
                '{"action":"select_route","value":2}'
            ),
            schema=make_schema(),
        )

    assert (
        exc_info.value.code
        == "action_field_type_mismatch"
    )


def test_disallowed_value_is_rejected() -> None:
    with pytest.raises(
        ActionParseError
    ) as exc_info:
        StructuredActionParser().parse(
            raw_text=(
                '{"action":"select_route","value":"Z"}'
            ),
            schema=make_schema(),
        )

    assert (
        exc_info.value.code
        == "action_field_value_not_allowed"
    )


def test_additional_fields_can_be_explicitly_allowed() -> None:
    schema = FocalActionSchema(
        focal_decision_id="decision",
        fields=(
            ActionField(
                name="action",
                json_type="string",
            ),
        ),
        allow_additional_fields=True,
    )

    result = (
        StructuredActionParser()
        .parse(
            raw_text=(
                '{"action":"choose","extra":7}'
            ),
            schema=schema,
        )
    )

    assert (
        result.normalized_action.payload
        == {
            "action": "choose",
            "extra": 7,
        }
    )


def test_non_finite_value_is_rejected() -> None:
    with pytest.raises(
        ActionSchemaError
    ):
        canonicalize_action(
            focal_decision_id="decision",
            payload={
                "value": math.nan,
            },
        )


def test_canonical_action_is_deterministic() -> None:
    first = canonicalize_action(
        focal_decision_id="decision",
        payload={
            "z": 2,
            "a": 1,
        },
    )

    second = canonicalize_action(
        focal_decision_id="decision",
        payload={
            "a": 1,
            "z": 2,
        },
    )

    assert (
        first.canonical_json
        == second.canonical_json
    )

    assert (
        first.canonical_sha256
        == second.canonical_sha256
    )


def test_parser_preserves_raw_output() -> None:
    raw = (
        '\n{"action":"select_route","value":"A"}\n'
    )

    result = (
        StructuredActionParser()
        .parse(
            raw_text=raw,
            schema=make_schema(),
        )
    )

    assert result.raw_text == raw