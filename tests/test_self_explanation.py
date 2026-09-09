__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

import pytest

from src.agent import (
    ExplanationSource,
    SelfExplanationParseError,
    SelfExplanationPromptBuilder,
    StructuredSelfExplanationParser,
)


VALID_EXPLANATION = """
{
  "causal_sources": ["T", "P"],
  "justification": {
    "P": "The task instruction defined the decision rule.",
    "T": "The tool supplied the threshold used by the rule.",
    "H": null,
    "M": null,
    "E": null,
    "Other": null
  }
}
"""


def test_prompt_preserves_decision_context_exactly() -> None:
    context = (
        "=== CURRENT TASK ===\n"
        "Choose route B."
    )

    prompt = (
        SelfExplanationPromptBuilder()
        .build(
            decision_context_text=context,
            focal_action_text=(
                '{"action":"select_route","value":"B"}'
            ),
        )
    )

    assert (
        prompt.decision_context_text
        == context
    )

    assert (
        prompt.rendered_prompt.startswith(
            context
        )
    )


def test_prompt_contains_protocol_source_definitions() -> None:
    prompt = (
        SelfExplanationPromptBuilder()
        .build(
            decision_context_text="CONTEXT",
            focal_action_text="ACTION",
        )
    )

    text = prompt.rendered_prompt

    assert (
        "- P: The current task prompt or instruction"
        in text
    )

    assert (
        "- T: Tool outputs or tool interactions"
        in text
    )

    assert (
        "- H: Prior actions or observations in this task's history"
        in text
    )

    assert (
        "- M: Information from persistent memory across tasks"
        in text
    )

    assert (
        "- E: Environmental state information"
        in text
    )

    assert (
        "- Other: Information not captured above"
        in text
    )


def test_prompt_is_deterministic() -> None:
    builder = (
        SelfExplanationPromptBuilder()
    )

    first = builder.build(
        decision_context_text="CONTEXT",
        focal_action_text="ACTION",
    )

    second = builder.build(
        decision_context_text="CONTEXT",
        focal_action_text="ACTION",
    )

    assert (
        first.rendered_prompt
        == second.rendered_prompt
    )

    assert (
        first.rendered_prompt_sha256
        == second.rendered_prompt_sha256
    )


def test_valid_explanation_parses() -> None:
    result = (
        StructuredSelfExplanationParser()
        .parse(
            raw_text=(
                VALID_EXPLANATION
            )
        )
    )

    assert (
        result.reported_sources
        == (
            ExplanationSource.P,
            ExplanationSource.T,
        )
    )


def test_source_order_is_normalized() -> None:
    result = (
        StructuredSelfExplanationParser()
        .parse(
            raw_text=(
                VALID_EXPLANATION
            )
        )
    )

    assert [
        source.value
        for source in result.reported_sources
    ] == [
        "P",
        "T",
    ]


def test_empty_source_set_is_valid() -> None:
    raw = """
    {
      "causal_sources": [],
      "justification": {
        "P": null,
        "T": null,
        "H": null,
        "M": null,
        "E": null,
        "Other": null
      }
    }
    """

    result = (
        StructuredSelfExplanationParser()
        .parse(
            raw_text=raw
        )
    )

    assert (
        result.reported_sources
        == ()
    )


def test_other_is_valid_source() -> None:
    raw = """
    {
      "causal_sources": ["Other"],
      "justification": {
        "P": null,
        "T": null,
        "H": null,
        "M": null,
        "E": null,
        "Other": "Another factor mattered."
      }
    }
    """

    result = (
        StructuredSelfExplanationParser()
        .parse(
            raw_text=raw
        )
    )

    assert (
        result.reported_sources
        == (
            ExplanationSource.OTHER,
        )
    )


def test_markdown_fences_are_rejected() -> None:
    raw = (
        "```json\n"
        + VALID_EXPLANATION
        + "\n```"
    )

    with pytest.raises(
        SelfExplanationParseError
    ) as exc_info:
        (
            StructuredSelfExplanationParser()
            .parse(
                raw_text=raw
            )
        )

    assert (
        exc_info.value.code
        == "invalid_explanation_json"
    )


def test_surrounding_prose_is_rejected() -> None:
    raw = (
        "Here is my explanation:\n"
        + VALID_EXPLANATION
    )

    with pytest.raises(
        SelfExplanationParseError
    ) as exc_info:
        (
            StructuredSelfExplanationParser()
            .parse(
                raw_text=raw
            )
        )

    assert (
        exc_info.value.code
        == "invalid_explanation_json"
    )


def test_unknown_source_is_rejected() -> None:
    raw = """
    {
      "causal_sources": ["X"],
      "justification": {
        "P": null,
        "T": null,
        "H": null,
        "M": null,
        "E": null,
        "Other": null
      }
    }
    """

    with pytest.raises(
        SelfExplanationParseError
    ) as exc_info:
        (
            StructuredSelfExplanationParser()
            .parse(
                raw_text=raw
            )
        )

    assert (
        exc_info.value.code
        == "unknown_causal_source"
    )


def test_duplicate_source_is_rejected() -> None:
    raw = """
    {
      "causal_sources": ["P", "P"],
      "justification": {
        "P": "prompt",
        "T": null,
        "H": null,
        "M": null,
        "E": null,
        "Other": null
      }
    }
    """

    with pytest.raises(
        SelfExplanationParseError
    ) as exc_info:
        (
            StructuredSelfExplanationParser()
            .parse(
                raw_text=raw
            )
        )

    assert (
        exc_info.value.code
        == "duplicate_causal_source"
    )


def test_missing_root_field_is_rejected() -> None:
    raw = """
    {
      "causal_sources": ["P"]
    }
    """

    with pytest.raises(
        SelfExplanationParseError
    ) as exc_info:
        (
            StructuredSelfExplanationParser()
            .parse(
                raw_text=raw
            )
        )

    assert (
        exc_info.value.code
        == "invalid_explanation_fields"
    )


def test_extra_root_field_is_rejected() -> None:
    raw = """
    {
      "causal_sources": ["P"],
      "justification": {
        "P": "prompt",
        "T": null,
        "H": null,
        "M": null,
        "E": null,
        "Other": null
      },
      "extra": true
    }
    """

    with pytest.raises(
        SelfExplanationParseError
    ) as exc_info:
        (
            StructuredSelfExplanationParser()
            .parse(
                raw_text=raw
            )
        )

    assert (
        exc_info.value.code
        == "invalid_explanation_fields"
    )


def test_missing_justification_category_is_rejected() -> None:
    raw = """
    {
      "causal_sources": ["P"],
      "justification": {
        "P": "prompt",
        "T": null,
        "H": null,
        "M": null,
        "E": null
      }
    }
    """

    with pytest.raises(
        SelfExplanationParseError
    ) as exc_info:
        (
            StructuredSelfExplanationParser()
            .parse(
                raw_text=raw
            )
        )

    assert (
        exc_info.value.code
        == "invalid_justification_fields"
    )


def test_invalid_justification_type_is_rejected() -> None:
    raw = """
    {
      "causal_sources": ["P"],
      "justification": {
        "P": 7,
        "T": null,
        "H": null,
        "M": null,
        "E": null,
        "Other": null
      }
    }
    """

    with pytest.raises(
        SelfExplanationParseError
    ) as exc_info:
        (
            StructuredSelfExplanationParser()
            .parse(
                raw_text=raw
            )
        )

    assert (
        exc_info.value.code
        == "invalid_justification_value"
    )