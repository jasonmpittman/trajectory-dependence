__author__ = "Jason M. Pittman"
__date__ = "August 24, 2026"
__copyright__ = "Copyright 2026"
__credits__ = ["Jason M. Pittman"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Jason M. Pittman"
__status__ = "Research"

from pathlib import Path

import pytest

from src.agent import (
    AgentRuntime,
    RunIdentity,
)
from src.logging import (
    Condition,
    EventType,
    RawEventWriter,
    SourceClass,
    TaskClassification,
)
from src.model import (
    ModelAdapter,
    ModelRuntimeMetadata,
)
from src.tasks import (
    TaskBackingStateBuilder,
    TaskFixtureResolver,
    TaskInterventionResolver,
    TaskResolutionError,
    load_task_suite,
)


PILOT_PATH = (
    Path("config")
    / "tasks"
    / "pilot-tasks.json"
)


class ResolutionFakeAdapter(
    ModelAdapter
):
    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def runtime_metadata(
        self,
    ) -> ModelRuntimeMetadata:
        return ModelRuntimeMetadata(
            model_identifier="fake",
            model_revision="fake",
            model_checksum="fake",
            quantization="test",
            backend="fake",
            backend_version="1",
            mlx_version="test",
            python_version="test",
            platform="test",
            model_path_env=(
                "TRAJECTORY_MODEL_PATH"
            ),
            device_info={},
        )

    def prepare_input(
        self,
        model_visible_text,
    ):
        raise AssertionError(
            "Fixture resolution must not invoke the model."
        )

    def generate(
        self,
        *,
        input_record,
        config,
    ):
        raise AssertionError(
            "Fixture resolution must not invoke the model."
        )


def pilot_task(
    task_id: str,
):
    suite = load_task_suite(
        PILOT_PATH
    )

    return next(
        task
        for task
        in suite.tasks
        if task.task_id == task_id
    )


def make_runtime(
    tmp_path,
    *,
    task_id: str,
    condition: Condition,
):
    runtime = AgentRuntime(
        identity=RunIdentity(
            experiment_id="pilot",
            run_id=(
                f"{task_id}_"
                f"{condition.value}_"
                "r001"
            ),
            task_id=task_id,
            task_classification=(
                TaskClassification.DIAGNOSTIC
            ),
            condition=condition,
            repetition_id=1,
            seed=12345,
        ),
        model_adapter=(
            ResolutionFakeAdapter()
        ),
        event_writer=RawEventWriter(
            tmp_path
            / "events.jsonl"
        ),
    )

    runtime.start()

    return runtime


def begin_and_resolve(
    tmp_path,
    *,
    task_id: str,
    condition: Condition,
):
    task = pilot_task(
        task_id
    )

    runtime = make_runtime(
        tmp_path,
        task_id=task_id,
        condition=condition,
    )

    runtime.begin_focal_task(
        task_prompt=(
            task.task_prompt
        ),
        action_schema=(
            task
            .focal_decision
            .schema
        ),
    )

    backing = (
        TaskBackingStateBuilder()
        .build(
            task=task,
            run_id=(
                runtime.identity.run_id
            ),
            work_directory=(
                tmp_path
                / "state"
            ),
        )
    )

    context = (
        TaskFixtureResolver()
        .resolve(
            task=task,
            condition=condition,
            runtime=runtime,
            backing_state=backing,
        )
    )

    return (
        task,
        runtime,
        backing,
        context,
    )


def test_c0_resolves_prompt_only(
    tmp_path,
) -> None:
    (
        _,
        _,
        _,
        context,
    ) = begin_and_resolve(
        tmp_path,
        task_id=(
            "pilot_tool_001"
        ),
        condition=Condition.C0,
    )

    assert (
        context.tool_elements
        == ()
    )

    assert (
        context.history_elements
        == ()
    )

    assert (
        context.memory_elements
        == ()
    )

    assert (
        context.environment_elements
        == ()
    )

    assert (
        context.fixture_targets[
            "prompt"
        ].source
        == SourceClass.P
    )


def test_tool_fixture_resolves_to_event_backed_t(
    tmp_path,
) -> None:
    (
        _,
        runtime,
        _,
        context,
    ) = begin_and_resolve(
        tmp_path,
        task_id=(
            "pilot_tool_001"
        ),
        condition=Condition.C1,
    )

    assert len(
        context.tool_elements
    ) == 2

    threshold_target = (
        context.fixture_targets[
            "tool_threshold"
        ]
    )

    assert (
        threshold_target.source
        == SourceClass.T
    )

    assert (
        threshold_target
        .target_id
        .startswith(
            "tool_return:"
        )
    )

    assert any(
        event.event_type
        == EventType.TOOL_RETURN
        for event
        in runtime.events
    )


def test_history_fixture_resolves_to_event_backed_h(
    tmp_path,
) -> None:
    (
        _,
        _,
        _,
        context,
    ) = begin_and_resolve(
        tmp_path,
        task_id=(
            "pilot_history_001"
        ),
        condition=Condition.C2,
    )

    assert len(
        context.history_elements
    ) == 2

    assert all(
        element.source
        == SourceClass.H
        for element
        in context.history_elements
    )

    target = (
        context.fixture_targets[
            "history_prior_branch"
        ]
    )

    assert (
        target.target_id
        .startswith(
            "history:"
        )
    )


def test_memory_backing_is_seeded_even_before_c3(
    tmp_path,
) -> None:
    task = pilot_task(
        "pilot_memory_001"
    )

    runtime = make_runtime(
        tmp_path,
        task_id=task.task_id,
        condition=Condition.C0,
    )

    backing = (
        TaskBackingStateBuilder()
        .build(
            task=task,
            run_id=(
                runtime.identity.run_id
            ),
            work_directory=(
                tmp_path
                / "state"
            ),
        )
    )

    assert (
        backing.memory_store
        is not None
    )

    assert (
        backing.memory_namespace
        == runtime.identity.run_id
    )

    record = (
        backing.memory_store
        .read(
            namespace=(
                backing
                .memory_namespace
            ),
            memory_id=(
                "memory_preferred_route"
            ),
        )
    )

    assert record is not None

    assert (
        record.value
        == "B"
    )


def test_c3_memory_resolves_to_m(
    tmp_path,
) -> None:
    (
        _,
        _,
        _,
        context,
    ) = begin_and_resolve(
        tmp_path,
        task_id=(
            "pilot_memory_001"
        ),
        condition=Condition.C3,
    )

    assert len(
        context.memory_elements
    ) == 2

    target = (
        context.fixture_targets[
            "memory_preferred_route"
        ]
    )

    assert (
        target.source
        == SourceClass.M
    )

    # InterventionEngine targets M using memory_id.
    assert (
        target.target_id
        == "memory_preferred_route"
    )


def test_environment_fixture_resolves_to_e(
    tmp_path,
) -> None:
    (
        _,
        _,
        _,
        context,
    ) = begin_and_resolve(
        tmp_path,
        task_id=(
            "pilot_environment_001"
        ),
        condition=Condition.C1,
    )

    assert len(
        context.environment_elements
    ) == 2

    status = (
        context.fixture_targets[
            "environment_door_status"
        ]
    )

    assert (
        status.source
        == SourceClass.E
    )

    assert (
        status.target_path
        == (
            "door",
            "status",
        )
    )


def test_tool_intervention_resolves_runtime_source_ids(
    tmp_path,
) -> None:
    (
        task,
        _,
        _,
        context,
    ) = begin_and_resolve(
        tmp_path,
        task_id=(
            "pilot_tool_001"
        ),
        condition=Condition.C1,
    )

    template = (
        task.intervention_pairs[
            0
        ]
    )

    pair = (
        TaskInterventionResolver()
        .resolve_pair(
            task=task,
            condition=Condition.C1,
            template=template,
            context=context,
        )
    )

    assert (
        pair.targeted.source
        == SourceClass.T
    )

    assert (
        pair.targeted.target_id
        == context.fixture_targets[
            "tool_threshold"
        ].target_id
    )

    assert (
        pair.sham.target_id
        == context.fixture_targets[
            "tool_label"
        ].target_id
    )


def test_memory_intervention_resolves_memory_id(
    tmp_path,
) -> None:
    (
        task,
        _,
        _,
        context,
    ) = begin_and_resolve(
        tmp_path,
        task_id=(
            "pilot_memory_001"
        ),
        condition=Condition.C3,
    )

    pair = (
        TaskInterventionResolver()
        .resolve_pair(
            task=task,
            condition=Condition.C3,
            template=(
                task
                .intervention_pairs[0]
            ),
            context=context,
        )
    )

    assert (
        pair.targeted.target_id
        == "memory_preferred_route"
    )

    assert (
        pair.sham.target_id
        == "memory_color"
    )


def test_environment_intervention_preserves_paths(
    tmp_path,
) -> None:
    (
        task,
        _,
        _,
        context,
    ) = begin_and_resolve(
        tmp_path,
        task_id=(
            "pilot_environment_001"
        ),
        condition=Condition.C1,
    )

    pair = (
        TaskInterventionResolver()
        .resolve_pair(
            task=task,
            condition=Condition.C1,
            template=(
                task
                .intervention_pairs[0]
            ),
            context=context,
        )
    )

    assert (
        pair.targeted.target_path
        == (
            "door",
            "status",
        )
    )

    assert (
        pair.sham.target_path
        == (
            "door",
            "color",
        )
    )


def test_prompt_intervention_rebuilds_focal_contract(
    tmp_path,
) -> None:
    (
        task,
        _,
        _,
        context,
    ) = begin_and_resolve(
        tmp_path,
        task_id=(
            "pilot_conflict_001"
        ),
        condition=Condition.C3,
    )

    template = next(
        pair
        for pair
        in task.intervention_pairs
        if (
            pair.source
            == SourceClass.P
        )
    )

    pair = (
        TaskInterventionResolver()
        .resolve_pair(
            task=task,
            condition=Condition.C3,
            template=template,
            context=context,
        )
    )

    assert isinstance(
        pair.targeted
        .replacement_value,
        str,
    )

    assert (
        "Focal action contract:"
        in pair.targeted
        .replacement_value
    )

    assert (
        "Return the focal decision as exactly one JSON object"
        in pair.targeted
        .replacement_value
    )

    assert (
        "persistent memory threshold outranks"
        in pair.targeted
        .replacement_value
    )

    assert (
        "Audit tag AX1"
        in pair.targeted
        .replacement_value
    )


def test_ineligible_intervention_condition_is_rejected(
    tmp_path,
) -> None:
    (
        task,
        _,
        _,
        context,
    ) = begin_and_resolve(
        tmp_path,
        task_id=(
            "pilot_memory_001"
        ),
        condition=Condition.C0,
    )

    with pytest.raises(
        TaskResolutionError
    ) as exc_info:
        (
            TaskInterventionResolver()
            .resolve_pair(
                task=task,
                condition=Condition.C0,
                template=(
                    task
                    .intervention_pairs[0]
                ),
                context=context,
            )
        )

    assert (
        exc_info.value.code
        == "intervention_not_eligible"
    )