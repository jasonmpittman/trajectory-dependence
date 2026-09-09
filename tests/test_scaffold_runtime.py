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

import pytest

from src.agent import (
    AgentRuntime,
    RunIdentity,
    ScaffoldController,
    ScaffoldExecutionError,
)
from src.environment import (
    DeterministicTransitionEngine,
    EnvironmentState,
    TransitionRule,
)
from src.logging import (
    Condition,
    EventType,
    RawEventWriter,
    SourceClass,
    TaskClassification,
)
from src.memory import (
    SQLiteMemoryStore,
)
from src.model import (
    ChatMessage,
    InferenceConfig,
    ModelAdapter,
    ModelGenerationResult,
    ModelInputRecord,
    ModelRuntimeMetadata,
)
from src.normalization import (
    ActionField,
    FocalActionSchema,
)
from src.tools import (
    KeyValueLookupTool,
    Tool,
    ToolExecutionError,
    ToolResult,
)


def sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def sha256_tokens(
    values,
) -> str:
    return sha256_text(
        json.dumps(
            list(values),
            separators=(",", ":"),
        )
    )


class ScaffoldFakeAdapter(
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
        model_visible_text: str,
    ) -> ModelInputRecord:
        rendered = (
            "<user>"
            + model_visible_text
            + "</user>"
        )

        ids = (
            1,
            2,
            3,
        )

        return ModelInputRecord(
            model_visible_text=(
                model_visible_text
            ),
            messages=(
                ChatMessage(
                    role="user",
                    content=(
                        model_visible_text
                    ),
                ),
            ),
            rendered_prompt=(
                rendered
            ),
            rendered_prompt_sha256=(
                sha256_text(
                    rendered
                )
            ),
            prompt_token_ids=ids,
            prompt_token_ids_sha256=(
                sha256_tokens(
                    ids
                )
            ),
            prompt_token_count=3,
            thinking_mode=False,
            chat_template_arguments={
                "enable_thinking": False,
            },
        )

    def generate(
        self,
        *,
        input_record,
        config,
    ) -> ModelGenerationResult:
        text = (
            '{"action":"select_route","value":"B"}'
        )

        generated_ids = (
            10,
            11,
        )

        return ModelGenerationResult(
            raw_text=text,
            raw_text_sha256=(
                sha256_text(
                    text
                )
            ),
            generated_token_ids=(
                generated_ids
            ),
            generated_token_ids_sha256=(
                sha256_tokens(
                    generated_ids
                )
            ),
            finish_reason="stop",
            prompt_tokens=3,
            generation_tokens=2,
            prompt_tps=None,
            generation_tps=None,
            peak_memory_gb=None,
            input_record=input_record,
            inference_config=config,
            runtime_metadata=(
                self.runtime_metadata()
            ),
        )


class FailingTool(
    Tool
):
    name = "failing_tool"

    def execute(
        self,
        request,
        state=None,
    ) -> ToolResult:
        raise ToolExecutionError(
            tool_name=self.name,
            code="intentional_failure",
            message="intentional test failure",
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


def make_runtime(
    tmp_path,
    *,
    condition: Condition,
) -> AgentRuntime:
    runtime = AgentRuntime(
        identity=RunIdentity(
            experiment_id="exp",
            run_id="run_001",
            task_id="task_001",
            task_classification=(
                TaskClassification.INTEGRATION
            ),
            condition=condition,
            repetition_id=1,
            seed=12345,
        ),
        model_adapter=(
            ScaffoldFakeAdapter()
        ),
        event_writer=RawEventWriter(
            tmp_path
            / "events.jsonl"
        ),
    )

    runtime.start()

    runtime.begin_focal_task(
        task_prompt=(
            "Use available state and "
            "select the correct route."
        ),
        action_schema=(
            make_schema()
        ),
    )

    return runtime


def make_memory_store(
    tmp_path,
) -> SQLiteMemoryStore:
    store = SQLiteMemoryStore(
        tmp_path
        / "memory.sqlite3"
    )

    store.initialize()

    store.write(
        namespace="run_001",
        memory_id="mem_001",
        key="preferred_route",
        value="B",
    )

    return store


def make_environment() -> EnvironmentState:
    return EnvironmentState(
        {
            "entities": {
                "door_1": {
                    "status": "locked",
                }
            }
        }
    )


def test_c0_rejects_tool_execution(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C0,
    )

    scaffold = ScaffoldController(
        runtime=runtime
    )

    with pytest.raises(
        ScaffoldExecutionError
    ) as exc_info:
        scaffold.execute_tool(
            tool=KeyValueLookupTool(),
            request={
                "key": "threshold",
            },
            state={
                "threshold": 7,
            },
        )

    assert (
        exc_info.value.code
        == "tool_not_allowed"
    )


def test_c1_tool_execution_logs_request_and_return(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C1,
    )

    scaffold = ScaffoldController(
        runtime=runtime
    )

    interaction = (
        scaffold.execute_tool(
            tool=KeyValueLookupTool(),
            request={
                "key": "threshold",
            },
            state={
                "threshold": 7,
            },
        )
    )

    assert (
        interaction.request_event.event_type
        == EventType.TOOL_REQUEST
    )

    assert (
        interaction.return_event.event_type
        == EventType.TOOL_RETURN
    )

    assert (
        interaction.return_event
        .parent_event_id
        == interaction.request_event.event_id
    )

    assert (
        interaction.tool_element.source
        == SourceClass.T
    )

    assert (
        interaction.tool_element
        .origin_event_id
        == interaction.return_event.event_id
    )


def test_tool_element_matches_return_payload(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C1,
    )

    scaffold = ScaffoldController(
        runtime=runtime
    )

    interaction = (
        scaffold.execute_tool(
            tool=KeyValueLookupTool(),
            request={
                "key": "threshold",
            },
            state={
                "threshold": 7,
            },
        )
    )

    assert (
        interaction.tool_element.content
        == interaction.return_event
        .normalized_payload
    )


def test_tool_failure_is_logged_as_technical_failure(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C1,
    )

    scaffold = ScaffoldController(
        runtime=runtime
    )

    with pytest.raises(
        ScaffoldExecutionError
    ) as exc_info:
        scaffold.execute_tool(
            tool=FailingTool(),
            request={
                "test": True,
            },
        )

    assert (
        exc_info.value.code
        == "tool_execution_failure"
    )

    assert (
        runtime.events[-1].event_type
        == EventType.TECHNICAL_FAILURE
    )


def test_c1_rejects_history_retention(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C1,
    )

    scaffold = ScaffoldController(
        runtime=runtime
    )

    interaction = (
        scaffold.execute_tool(
            tool=KeyValueLookupTool(),
            request={
                "key": "threshold",
            },
            state={
                "threshold": 7,
            },
        )
    )

    with pytest.raises(
        ScaffoldExecutionError
    ) as exc_info:
        scaffold.retain_history(
            event_id=(
                interaction
                .return_event
                .event_id
            )
        )

    assert (
        exc_info.value.code
        == "history_not_allowed"
    )


def test_c2_tool_return_can_be_retained_as_history(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C2,
    )

    scaffold = ScaffoldController(
        runtime=runtime
    )

    interaction = (
        scaffold.execute_tool(
            tool=KeyValueLookupTool(),
            request={
                "key": "threshold",
            },
            state={
                "threshold": 7,
            },
        )
    )

    history = (
        scaffold.retain_history(
            event_id=(
                interaction
                .return_event
                .event_id
            )
        )
    )

    assert history.source == SourceClass.H

    assert (
        history.origin_source
        == SourceClass.T
    )

    assert (
        history.content
        == interaction.return_event
        .normalized_payload
    )


def test_c2_rejects_persistent_memory(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C2,
    )

    store = make_memory_store(
        tmp_path
    )

    scaffold = ScaffoldController(
        runtime=runtime,
        memory_store=store,
        memory_namespace="run_001",
    )

    with pytest.raises(
        ScaffoldExecutionError
    ) as exc_info:
        scaffold.read_memory(
            memory_id="mem_001"
        )

    assert (
        exc_info.value.code
        == "memory_not_allowed"
    )


def test_c3_memory_read_creates_event_backed_m(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C3,
    )

    store = make_memory_store(
        tmp_path
    )

    scaffold = ScaffoldController(
        runtime=runtime,
        memory_store=store,
        memory_namespace="run_001",
    )

    observation = (
        scaffold.read_memory(
            memory_id="mem_001"
        )
    )

    assert observation.found is True

    assert (
        observation.event.event_type
        == EventType.MEMORY_READ
    )

    assert (
        observation.memory_element
        is not None
    )

    assert (
        observation.memory_element.source
        == SourceClass.M
    )

    assert (
        observation.memory_element
        .origin_event_id
        == observation.event.event_id
    )

    assert (
        observation.memory_element.content
        == observation.event
        .normalized_payload
    )


def test_missing_memory_read_is_logged_but_not_exposed_as_m(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C3,
    )

    store = make_memory_store(
        tmp_path
    )

    scaffold = ScaffoldController(
        runtime=runtime,
        memory_store=store,
        memory_namespace="run_001",
    )

    observation = (
        scaffold.read_memory(
            memory_id="missing"
        )
    )

    assert observation.found is False

    assert (
        observation.memory_element
        is None
    )

    assert (
        observation.event
        .normalized_payload[
            "found"
        ]
        is False
    )


def test_environment_query_creates_event_backed_e(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C1,
    )

    environment = (
        make_environment()
    )

    scaffold = ScaffoldController(
        runtime=runtime,
        environment=environment,
    )

    observation = (
        scaffold.query_environment(
            path=[
                "entities",
                "door_1",
                "status",
            ]
        )
    )

    assert (
        observation.event.event_type
        == EventType.ENVIRONMENT_QUERY
    )

    assert (
        observation.environment_element
        .source
        == SourceClass.E
    )

    assert (
        observation.environment_element
        .content
        == observation.event
        .normalized_payload
    )


def test_environment_policy_is_exactly_source_scoped(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C1,
    )

    scaffold = ScaffoldController(
        runtime=runtime,
        environment=make_environment(),
    )

    observation = (
        scaffold.query_environment(
            path=[
                "entities",
                "door_1",
                "status",
            ]
        )
    )

    policy = (
        scaffold.environment_policy_for(
            (
                observation
                .environment_element,
            )
        )
    )

    assert (
        policy is not None
    )

    assert (
        policy.allowed_source_ids
        == frozenset(
            {
                observation
                .environment_element
                .source_id
            }
        )
    )


def test_environment_transition_is_logged(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C2,
    )

    environment = (
        make_environment()
    )

    engine = (
        DeterministicTransitionEngine(
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
                )
            ]
        )
    )

    scaffold = ScaffoldController(
        runtime=runtime,
        environment=environment,
        transition_engine=engine,
    )

    observation = (
        scaffold.transition_environment(
            action={
                "action": "unlock",
                "target": "door_1",
            }
        )
    )

    assert (
        observation.event.event_type
        == EventType.ENVIRONMENT_TRANSITION
    )

    state = environment.query(
        [
            "entities",
            "door_1",
            "status",
        ]
    )

    assert state.value == "unlocked"


def test_c1_focal_action_can_use_tool_element(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C1,
    )

    scaffold = ScaffoldController(
        runtime=runtime
    )

    interaction = (
        scaffold.execute_tool(
            tool=KeyValueLookupTool(),
            request={
                "key": "threshold",
            },
            state={
                "threshold": 7,
            },
        )
    )

    result = (
        scaffold.invoke_focal_action(
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
            ),
            tool_elements=(
                interaction.tool_element,
            ),
        )
    )

    assert result.valid is True

    assert [
        record.source
        for record in (
            result.model_invocation
            .provenance_audit
            .records
        )
    ] == [
        SourceClass.P,
        SourceClass.T,
    ]


def test_c3_focal_action_can_combine_t_h_m_e(
    tmp_path,
) -> None:
    runtime = make_runtime(
        tmp_path,
        condition=Condition.C3,
    )

    store = make_memory_store(
        tmp_path
    )

    environment = (
        make_environment()
    )

    scaffold = ScaffoldController(
        runtime=runtime,
        memory_store=store,
        memory_namespace="run_001",
        environment=environment,
    )

    interaction = (
        scaffold.execute_tool(
            tool=KeyValueLookupTool(),
            request={
                "key": "threshold",
            },
            state={
                "threshold": 7,
            },
        )
    )

    history = (
        scaffold.retain_history(
            event_id=(
                interaction
                .request_event
                .event_id
            )
        )
    )

    memory = scaffold.read_memory(
        memory_id="mem_001"
    )

    environment_observation = (
        scaffold.query_environment(
            path=[
                "entities",
                "door_1",
                "status",
            ]
        )
    )

    assert (
        memory.memory_element
        is not None
    )

    result = (
        scaffold.invoke_focal_action(
            config=InferenceConfig(
                max_tokens=32,
                seed=12345,
            ),
            tool_elements=(
                interaction.tool_element,
            ),
            history_elements=(
                history,
            ),
            memory_elements=(
                memory.memory_element,
            ),
            environment_elements=(
                environment_observation
                .environment_element,
            ),
        )
    )

    assert result.valid is True

    assert [
        record.source
        for record in (
            result.model_invocation
            .provenance_audit
            .records
        )
    ] == [
        SourceClass.P,
        SourceClass.T,
        SourceClass.H,
        SourceClass.M,
        SourceClass.E,
    ]