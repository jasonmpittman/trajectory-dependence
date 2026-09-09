__author__ = "Jason M. Pittman"
__date__ = "August 22, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from copy import deepcopy
from dataclasses import replace

import pytest

from src.environment import (
    DeterministicTransitionEngine,
    EnvironmentState,
    EnvironmentStateError,
    EnvironmentTransitionError,
    TransitionRule,
)


def make_state() -> dict:
    return {
        "entities": {
            "door_1": {
                "status": "locked",
                "color": "red",
            },
            "container_1": {
                "open": False,
                "contents": [
                    "item_1",
                    "item_2",
                ],
            },
        },
        "agent_inventory": [],
        "global_flags": {
            "alarm": False,
        },
    }


def make_environment() -> EnvironmentState:
    return EnvironmentState(
        make_state()
    )


def make_transition_engine() -> DeterministicTransitionEngine:
    return DeterministicTransitionEngine(
        [
            TransitionRule(
                rule_id="unlock_door_1",
                match={
                    "action": "unlock",
                    "target": "door_1",
                },
                path=(
                    "entities",
                    "door_1",
                    "status",
                ),
                required_value="locked",
                new_value="unlocked",
            ),
            TransitionRule(
                rule_id="open_container_1",
                match={
                    "action": "open",
                    "target": "container_1",
                },
                path=(
                    "entities",
                    "container_1",
                    "open",
                ),
                required_value=False,
                new_value=True,
            ),
        ]
    )


def test_initial_state_is_defensively_copied() -> None:
    initial = make_state()

    environment = EnvironmentState(
        initial
    )

    initial["entities"]["door_1"]["status"] = "corrupted"

    result = environment.query(
        [
            "entities",
            "door_1",
            "status",
        ]
    )

    assert result.value == "locked"


def test_get_state_returns_defensive_copy() -> None:
    environment = make_environment()

    exposed = environment.get_state()

    exposed["entities"]["door_1"]["status"] = "corrupted"

    result = environment.query(
        [
            "entities",
            "door_1",
            "status",
        ]
    )

    assert result.value == "locked"


def test_query_reads_nested_state() -> None:
    environment = make_environment()

    result = environment.query(
        [
            "entities",
            "door_1",
            "status",
        ]
    )

    assert result.found is True
    assert result.value == "locked"

    assert result.path == (
        "entities",
        "door_1",
        "status",
    )


def test_query_missing_path_is_valid_result() -> None:
    environment = make_environment()

    result = environment.query(
        [
            "entities",
            "door_99",
            "status",
        ]
    )

    assert result.found is False
    assert result.value is None


def test_targeted_modify_changes_only_target() -> None:
    environment = make_environment()

    before = environment.get_state()

    mutation = environment.targeted_modify(
        path=[
            "entities",
            "door_1",
            "status",
        ],
        value="unlocked",
    )

    after = environment.get_state()

    assert mutation.previous_value == "locked"
    assert mutation.new_value == "unlocked"
    assert mutation.state_changed is True

    assert (
        after["entities"]["door_1"]["status"]
        == "unlocked"
    )

    assert (
        after["entities"]["door_1"]["color"]
        == before["entities"]["door_1"]["color"]
    )

    assert (
        after["entities"]["container_1"]
        == before["entities"]["container_1"]
    )

    assert (
        after["global_flags"]
        == before["global_flags"]
    )


def test_targeted_modify_same_value_reports_no_change() -> None:
    environment = make_environment()

    mutation = environment.targeted_modify(
        path=[
            "entities",
            "door_1",
            "status",
        ],
        value="locked",
    )

    assert mutation.state_changed is False


def test_targeted_modify_does_not_modify_external_value() -> None:
    environment = make_environment()

    replacement = {
        "items": [
            "new_item",
        ]
    }

    environment.targeted_modify(
        path=[
            "global_flags",
            "payload",
        ],
        value=replacement,
    )

    replacement["items"].append(
        "corruption"
    )

    result = environment.query(
        [
            "global_flags",
            "payload",
        ]
    )

    assert result.value == {
        "items": [
            "new_item",
        ]
    }


def test_snapshot_checksum_is_stable_for_same_state() -> None:
    environment = make_environment()

    first = environment.snapshot()
    second = environment.snapshot()

    assert (
        first.checksum_sha256
        == second.checksum_sha256
    )


def test_snapshot_restore_recovers_exact_state() -> None:
    environment = make_environment()

    before = environment.get_state()
    snapshot = environment.snapshot()

    environment.targeted_modify(
        path=[
            "entities",
            "door_1",
            "status",
        ],
        value="unlocked",
    )

    environment.targeted_modify(
        path=[
            "global_flags",
            "alarm",
        ],
        value=True,
    )

    environment.restore(snapshot)

    after = environment.get_state()

    assert after == before

    assert (
        environment.checksum_sha256()
        == snapshot.checksum_sha256
    )


def test_restore_rejects_corrupted_snapshot() -> None:
    environment = make_environment()

    snapshot = environment.snapshot()

    corrupted = replace(
        snapshot,
        checksum_sha256="0" * 64,
    )

    with pytest.raises(
        EnvironmentStateError
    ) as exc_info:
        environment.restore(
            corrupted
        )

    assert (
        exc_info.value.code
        == "snapshot_checksum_mismatch"
    )


def test_non_json_state_is_rejected() -> None:
    with pytest.raises(
        EnvironmentStateError
    ) as exc_info:
        EnvironmentState(
            {
                "invalid": {
                    1,
                    2,
                    3,
                }
            }
        )

    assert (
        exc_info.value.code
        == "non_serializable_state"
    )


def test_transition_applies_exact_rule() -> None:
    environment = make_environment()
    engine = make_transition_engine()

    result = engine.apply(
        environment=environment,
        action={
            "action": "unlock",
            "target": "door_1",
        },
    )

    assert result.rule_id == "unlock_door_1"
    assert result.before == "locked"
    assert result.after == "unlocked"
    assert result.state_changed is True

    observed = environment.query(
        [
            "entities",
            "door_1",
            "status",
        ]
    )

    assert observed.value == "unlocked"


def test_transition_rejects_invalid_current_state() -> None:
    environment = make_environment()
    engine = make_transition_engine()

    engine.apply(
        environment=environment,
        action={
            "action": "unlock",
            "target": "door_1",
        },
    )

    with pytest.raises(
        EnvironmentTransitionError
    ) as exc_info:
        engine.apply(
            environment=environment,
            action={
                "action": "unlock",
                "target": "door_1",
            },
        )

    assert (
        exc_info.value.code
        == "invalid_transition_state"
    )


def test_transition_rejects_unknown_action() -> None:
    environment = make_environment()
    engine = make_transition_engine()

    with pytest.raises(
        EnvironmentTransitionError
    ) as exc_info:
        engine.apply(
            environment=environment,
            action={
                "action": "dance",
                "target": "door_1",
            },
        )

    assert (
        exc_info.value.code
        == "transition_not_found"
    )


def test_duplicate_action_matches_are_rejected() -> None:
    with pytest.raises(
        EnvironmentTransitionError
    ) as exc_info:
        DeterministicTransitionEngine(
            [
                TransitionRule(
                    rule_id="rule_1",
                    match={
                        "action": "unlock",
                        "target": "door_1",
                    },
                    path=(
                        "entities",
                        "door_1",
                        "status",
                    ),
                    required_value="locked",
                    new_value="unlocked",
                ),
                TransitionRule(
                    rule_id="rule_2",
                    match={
                        "action": "unlock",
                        "target": "door_1",
                    },
                    path=(
                        "global_flags",
                        "alarm",
                    ),
                    required_value=False,
                    new_value=True,
                ),
            ]
        )

    assert (
        exc_info.value.code
        == "duplicate_rule_match"
    )


def test_same_state_and_action_produce_same_transition() -> None:
    first_environment = make_environment()
    second_environment = make_environment()

    engine = make_transition_engine()

    action = {
        "action": "open",
        "target": "container_1",
    }

    first_result = engine.apply(
        environment=first_environment,
        action=deepcopy(action),
    )

    second_result = engine.apply(
        environment=second_environment,
        action=deepcopy(action),
    )

    assert (
        first_result.to_dict()
        == second_result.to_dict()
    )

    assert (
        first_environment.get_state()
        == second_environment.get_state()
    )


def test_transition_does_not_modify_action_input() -> None:
    environment = make_environment()
    engine = make_transition_engine()

    action = {
        "action": "unlock",
        "target": "door_1",
    }

    original = deepcopy(action)

    engine.apply(
        environment=environment,
        action=action,
    )

    assert action == original