#![cfg(feature = "test-utils")]
#![allow(clippy::panic_in_result_fn)]

use std::{
    convert::Infallible,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result};
use futures::StreamExt;
use rig_agent::{
    AgentBuilder,
    agent::{
        AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, InvalidToolCallAction,
        InvalidToolCallContext, ModelTurnAction, ModelTurnFinished, MultiTurnStreamItem,
        OutputMode, StreamingError, ToolCall, ToolCallAction,
    },
    completion::{Message, Prompt, PromptError},
    streaming::{StreamedAssistantContent, StreamingPrompt},
};
use rig_core::{
    completion::{AssistantContent, Usage},
    memory::{ConversationMemory, MemoryError},
    test_utils::{
        AppendFailingMemory, CountingMemory, MockCompletionModel, MockStreamEvent, MockTurn,
    },
    tool::PortableTool,
    wasm_compat::WasmBoxedFuture,
};
use rig_runtime_conformance::{
    ALL_SCENARIOS, CountingPortableTool, ScenarioId, ScenarioReport as SharedReport, scenario,
    verify_report,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

fn report(id: ScenarioId) -> SharedReport {
    SharedReport::new("rig-agent", id)
}

fn evidence<T>(report: &mut SharedReport, index: usize, actual: T) -> Result<()>
where
    T: Serialize,
{
    let observation = scenario(report.scenario_id)
        .and_then(|definition| definition.observations.get(index))
        .map(|observation| observation.description)
        .context("shared observation index must exist")?;
    report.observe(observation, actual)?;
    Ok(())
}

fn roles_valid(messages: &[Message]) -> bool {
    !messages.is_empty()
        && messages
            .windows(2)
            .all(|pair| !matches!(pair, [Message::Assistant { .. }, Message::Assistant { .. }]))
}

#[derive(Clone, Default)]
struct CountingLoadFailure(Arc<AtomicUsize>);

impl CountingLoadFailure {
    fn append_count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

impl ConversationMemory for CountingLoadFailure {
    fn load<'a>(
        &'a self,
        _conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async {
            Err(MemoryError::backend(std::io::Error::other(
                "counted load failure",
            )))
        })
    }

    fn append<'a>(
        &'a self,
        _conversation_id: &'a str,
        _messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }

    fn clear<'a>(
        &'a self,
        _conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async { Ok(()) })
    }
}

#[derive(Clone)]
struct RetryText(&'static str);

impl AgentHook for RetryText {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if event
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::Text(text) if text.text == self.0))
        {
            ModelTurnAction::retry_with_feedback("return an accepted response")
        } else {
            ModelTurnAction::continue_run()
        }
    }
}

#[derive(Clone)]
struct RepeatText(&'static str);

impl AgentHook for RepeatText {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if event
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::Text(text) if text.text == self.0))
        {
            ModelTurnAction::repeat()
        } else {
            ModelTurnAction::continue_run()
        }
    }
}

#[derive(Clone)]
struct InvalidDecision(InvalidToolCallAction);

impl AgentHook for InvalidDecision {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        _event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(self.0.clone())
    }
}

#[derive(Clone, Default)]
struct StopFirstCompletion(Arc<AtomicUsize>);

impl AgentHook for StopFirstCompletion {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            CompletionCallAction::stop("cancel before dispatch")
        } else {
            CompletionCallAction::continue_run()
        }
    }
}

struct StopBeforeModel;

impl AgentHook for StopBeforeModel {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        CompletionCallAction::stop("cancel before dispatch")
    }
}

#[derive(Clone, Default)]
struct CountCompletionCalls(Arc<AtomicUsize>);

impl AgentHook for CountCompletionCalls {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        self.0.fetch_add(1, Ordering::SeqCst);
        CompletionCallAction::continue_run()
    }
}

#[derive(Clone, Copy)]
struct StopText(&'static str);

impl AgentHook for StopText {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if event
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::Text(text) if text.text == self.0))
        {
            ModelTurnAction::stop("stop after provisional output")
        } else {
            ModelTurnAction::continue_run()
        }
    }
}

async fn model_call_budgets() -> Result<SharedReport> {
    let zero_model = MockCompletionModel::text("unused");
    let zero_agent = AgentBuilder::new(zero_model.clone()).build();
    let zero = zero_agent
        .prompt("zero")
        .max_turns(0)
        .await
        .err()
        .context("zero budget must reject")?;

    let retry_model = MockCompletionModel::new([MockTurn::text("retry"), MockTurn::text("done")]);
    let retry_agent = AgentBuilder::new(retry_model.clone())
        .add_hook(RetryText("retry"))
        .build();
    let retried = retry_agent
        .prompt("prompt")
        .max_turns(2)
        .extended_details()
        .await?;

    let mut report = report(ScenarioId::ModelCallBudgets);
    evidence(
        &mut report,
        0,
        matches!(zero, PromptError::MaxTurnsError { max_turns: 0, .. })
            && zero_model.request_count() == 0,
    )?;
    evidence(
        &mut report,
        1,
        (retry_model.request_count(), retried.completion_calls.len()),
    )?;
    Ok(report)
}

async fn canonical_transcript() -> Result<SharedReport> {
    let model = MockCompletionModel::new([MockTurn::text("retry"), MockTurn::text("kept")]);
    let agent = AgentBuilder::new(model)
        .add_hook(RepeatText("retry"))
        .build();
    let response = agent
        .prompt("prompt")
        .max_turns(2)
        .extended_details()
        .await?;
    let messages = response.messages.context("extended history")?;
    let rejected_absent = messages.iter().all(|message| {
        !matches!(message, Message::Assistant { content, .. } if content.iter().any(|item| matches!(item, AssistantContent::Text(text) if text.text == "retry")))
    });
    let mut report = report(ScenarioId::CanonicalTranscript);
    evidence(&mut report, 0, roles_valid(&messages))?;
    evidence(&mut report, 1, messages.len())?;
    evidence(&mut report, 2, rejected_absent)?;
    Ok(report)
}

async fn tool_pairing() -> Result<SharedReport> {
    let model = MockCompletionModel::new([
        MockTurn::from_contents([
            AssistantContent::tool_call(
                "first",
                "counting_portable_tool",
                serde_json::json!({"value":"first"}),
            ),
            AssistantContent::tool_call(
                "second",
                "counting_portable_tool",
                serde_json::json!({"value":"second"}),
            ),
        ])?,
        MockTurn::text("done"),
    ]);
    let tool = CountingPortableTool::default();
    let agent = AgentBuilder::new(model).tool(tool).build();
    let response = agent
        .prompt("pair")
        .max_turns(2)
        .tool_concurrency(2)
        .extended_details()
        .await?;
    let messages = response.messages.context("extended history")?;
    let ids = messages
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => Some(content),
            _ => None,
        })
        .flat_map(|content| content.iter())
        .filter_map(|content| match content {
            rig_core::message::UserContent::ToolResult(result) => Some(result.id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut report = report(ScenarioId::ToolCallResultPairing);
    evidence(&mut report, 0, ids.len())?;
    evidence(&mut report, 1, ids)?;
    Ok(report)
}

async fn usage_accounting() -> Result<SharedReport> {
    let first = Usage {
        total_tokens: 3,
        ..Usage::new()
    };
    let second = Usage {
        total_tokens: 5,
        ..Usage::new()
    };
    let model = MockCompletionModel::new([
        MockTurn::text("retry").with_usage(first),
        MockTurn::text("done").with_usage(second),
    ]);
    let agent = AgentBuilder::new(model)
        .add_hook(RetryText("retry"))
        .build();
    let response = agent
        .prompt("usage")
        .max_turns(2)
        .extended_details()
        .await?;
    let indexes = response
        .completion_calls
        .iter()
        .map(|call| call.call_index)
        .collect::<Vec<_>>();
    let duplicate_model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("duplicate-safe"),
        MockStreamEvent::final_response_with_total_tokens(11),
        MockStreamEvent::final_response_with_total_tokens(11),
    ]]);
    let duplicate_agent = AgentBuilder::new(duplicate_model).build();
    let (_, duplicate_finals, duplicate_response) =
        drain_stream(&duplicate_agent, "duplicate usage").await?;
    let duplicate_response = duplicate_response.context("duplicate-final response")?;
    let mut report = report(ScenarioId::UsageAccounting);
    evidence(
        &mut report,
        0,
        response.completion_calls.len() == 2 && response.usage.total_tokens == 8,
    )?;
    evidence(
        &mut report,
        1,
        indexes == [0, 1]
            && response
                .completion_calls
                .iter()
                .map(|call| call.usage.total_tokens)
                .collect::<Vec<_>>()
                == [3, 5],
    )?;
    evidence(
        &mut report,
        2,
        duplicate_finals == 1
            && duplicate_response.usage.total_tokens == 11
            && duplicate_response.completion_calls.len() == 1,
    )?;
    Ok(report)
}

async fn invalid_run(
    action: InvalidToolCallAction,
    requested_name: &str,
    with_tool: bool,
) -> Result<(Result<String, PromptError>, usize, usize)> {
    let model = MockCompletionModel::new([
        MockTurn::tool_call("invalid", requested_name, serde_json::json!({"value":"x"})),
        MockTurn::text("recovered"),
    ]);
    let tool = CountingPortableTool::default();
    let request = if with_tool {
        AgentBuilder::new(model.clone())
            .tool(tool.clone())
            .add_hook(InvalidDecision(action))
            .build()
            .prompt("invalid")
            .max_turns(2)
            .max_invalid_tool_call_retries(1)
            .await
    } else {
        AgentBuilder::new(model.clone())
            .add_hook(InvalidDecision(action))
            .build()
            .prompt("invalid")
            .max_turns(2)
            .max_invalid_tool_call_retries(1)
            .await
    };
    Ok((request, tool.calls().len(), model.request_count()))
}

async fn invalid_tool_recovery() -> Result<SharedReport> {
    let fail = invalid_run(InvalidToolCallAction::fail(), "missing", false).await?;
    let retry = invalid_run(
        InvalidToolCallAction::retry("use an advertised tool"),
        "missing",
        false,
    )
    .await?;
    let repair = invalid_run(
        InvalidToolCallAction::repair("counting_portable_tool"),
        "COUNTING_PORTABLE_TOOL",
        true,
    )
    .await?;
    let skip = invalid_run(InvalidToolCallAction::skip("skip"), "missing", false).await?;
    let stop = invalid_run(InvalidToolCallAction::stop("stop"), "missing", false).await?;
    let distinct = matches!(fail.0, Err(PromptError::UnknownToolCall { .. }))
        && matches!(&retry.0, Ok(value) if value == "recovered")
        && matches!(&repair.0, Ok(value) if value == "recovered")
        && matches!(&skip.0, Ok(value) if value == "recovered")
        && matches!(stop.0, Err(PromptError::PromptCancelled { .. }))
        && retry.2 == 2
        && repair.1 == 1;
    let suppressed = fail.1 == 0 && retry.1 == 0 && skip.1 == 0 && stop.1 == 0;
    let mut report = report(ScenarioId::InvalidToolRecovery);
    evidence(&mut report, 0, distinct)?;
    evidence(&mut report, 1, suppressed)?;
    Ok(report)
}

async fn response_retry_rollback() -> Result<SharedReport> {
    let feedback_model =
        MockCompletionModel::new([MockTurn::text("reject"), MockTurn::text("accepted")]);
    let preparation = CountCompletionCalls::default();
    let feedback_agent = AgentBuilder::new(feedback_model.clone())
        .add_hook(RetryText("reject"))
        .add_hook(preparation.clone())
        .build();
    let feedback_response = feedback_agent
        .prompt("prompt")
        .max_turns(2)
        .extended_details()
        .await?;
    let feedback_requests = feedback_model.requests();
    let feedback_messages = feedback_response
        .messages
        .context("feedback retry history")?;
    let tool_free = feedback_requests
        .iter()
        .all(|request| request.tools.is_empty());
    let feedback = feedback_requests.get(1).is_some_and(|request| {
        request.chat_history.len() > 1
            && serde_json::to_string(&request.chat_history)
                .is_ok_and(|history| history.contains("return an accepted response"))
    });

    let tool_model = MockCompletionModel::new([
        MockTurn::from_contents([
            AssistantContent::text("reject"),
            AssistantContent::tool_call(
                "real",
                "counting_portable_tool",
                serde_json::json!({"value":"executed"}),
            ),
        ])?,
        MockTurn::text("tool continuation"),
    ]);
    let tool = CountingPortableTool::default();
    let tool_agent = AgentBuilder::new(tool_model.clone())
        .tool(tool.clone())
        .add_hook(RetryText("reject"))
        .build();
    let tool_result = tool_agent.prompt("tool-bearing").max_turns(2).await;

    let repeated_model =
        MockCompletionModel::new([MockTurn::text("reject"), MockTurn::text("reject")]);
    let repeated_agent = AgentBuilder::new(repeated_model.clone())
        .add_hook(RetryText("reject"))
        .build();
    let repeated = repeated_agent.prompt("repeat policy").max_turns(2).await;

    let rollback_model =
        MockCompletionModel::new([MockTurn::text("reject"), MockTurn::text("accepted")]);
    let rollback_agent = AgentBuilder::new(rollback_model)
        .add_hook(RepeatText("reject"))
        .build();
    let rollback_response = rollback_agent
        .prompt("prompt")
        .max_turns(2)
        .extended_details()
        .await?;
    let rollback_messages = rollback_response
        .messages
        .context("rollback retry history")?;
    let rejected_absent = !serde_json::to_string(&rollback_messages)?.contains("reject");
    let mut report = report(ScenarioId::ResponseRetryRollback);
    evidence(&mut report, 0, tool_free && feedback_requests.len() == 2)?;
    evidence(
        &mut report,
        1,
        matches!(tool_result, Err(PromptError::PromptCancelled { .. }))
            && tool.calls().is_empty()
            && tool_model.request_count() == 1,
    )?;
    evidence(
        &mut report,
        2,
        feedback && feedback_messages.len() == 4 && preparation.0.load(Ordering::SeqCst) == 2,
    )?;
    evidence(
        &mut report,
        3,
        repeated.is_err() && repeated_model.request_count() == 2,
    )?;
    evidence(
        &mut report,
        4,
        feedback_requests.len() == 2 && feedback_response.completion_calls.len() == 2,
    )?;
    evidence(&mut report, 5, rejected_absent)?;
    Ok(report)
}

async fn stop_and_cancellation() -> Result<SharedReport> {
    let model = MockCompletionModel::text("agent remains usable");
    let agent = AgentBuilder::new(model.clone())
        .add_hook(StopFirstCompletion::default())
        .build();
    let error = agent
        .prompt("cancel")
        .await
        .err()
        .context("hook cancellation must stop")?;
    let diagnostic = matches!(
        &error,
        PromptError::PromptCancelled { chat_history, reason }
            if chat_history == &[Message::user("cancel")] && reason == "cancel before dispatch"
    );
    let canceled_request_count = model.request_count();
    let cancellation_memory = CountingMemory::default();
    let cancellation_model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("provisional cancellation"),
        MockStreamEvent::final_response_with_total_tokens(1),
    ]]);
    let cancellation_agent = AgentBuilder::new(cancellation_model)
        .memory(cancellation_memory.clone())
        .conversation("cancelled-stream")
        .build();
    let mut stream = cancellation_agent.stream_prompt("cancel stream").await;
    while let Some(item) = stream.next().await {
        if matches!(
            item,
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::Text(_)
            ))
        ) {
            break;
        }
    }
    drop(stream);
    tokio::task::yield_now().await;
    let mut report = report(ScenarioId::StopAndCancellation);
    evidence(&mut report, 0, diagnostic && canceled_request_count == 0)?;
    evidence(&mut report, 1, cancellation_memory.append_count() == 0)?;
    evidence(&mut report, 2, diagnostic)?;
    let retention = scenario(ScenarioId::StopAndCancellation)
        .and_then(|definition| definition.observations.get(3))
        .map(|observation| observation.description)
        .context("classic retention boundary")?;
    report.observe_not_applicable(
        retention,
        "the classic runtime returns terminal state directly and has no retained ECS entity or cleanup tick",
    )?;
    Ok(report)
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Answer {
    answer: String,
}

async fn structured_output() -> Result<SharedReport> {
    let native_model = MockCompletionModel::text(r#"{"answer":"native"}"#);
    let native = AgentBuilder::new(native_model.clone())
        .output_schema::<Answer>()
        .output_mode(OutputMode::Native)
        .build()
        .prompt("native")
        .await?;
    let tool_model = MockCompletionModel::new([MockTurn::tool_call(
        "output",
        "final_result_1",
        serde_json::json!({"answer":"tool"}),
    )]);
    let collision = PortableNamedTool::<1>;
    let tool = AgentBuilder::new(tool_model.clone())
        .tool(collision)
        .output_schema::<Answer>()
        .output_mode(OutputMode::Tool)
        .build()
        .prompt("tool")
        .await?;
    let prompted_model = MockCompletionModel::text(r#"{"answer":"prompted"}"#);
    let prompted = AgentBuilder::new(prompted_model.clone())
        .output_schema::<Answer>()
        .output_mode(OutputMode::Prompted)
        .build()
        .prompt("prompted")
        .await?;
    let auto_native_model = MockCompletionModel::text(r#"{"answer":"auto-native"}"#);
    let auto_native = AgentBuilder::new(auto_native_model.clone())
        .output_schema::<Answer>()
        .build()
        .prompt("auto native")
        .await?;
    let auto_tool_model = MockCompletionModel::new([MockTurn::tool_call(
        "auto-output",
        "final_result_1",
        serde_json::json!({"answer":"auto-tool"}),
    )]);
    let auto_tool = AgentBuilder::new(auto_tool_model.clone())
        .tool(PortableNamedTool::<2>)
        .output_schema::<Answer>()
        .build()
        .prompt("auto tool")
        .await?;
    let recovery_model =
        MockCompletionModel::new([MockTurn::text("invalid-one"), MockTurn::text("invalid-two")]);
    let recovery = AgentBuilder::new(recovery_model.clone())
        .output_schema::<Answer>()
        .output_mode(OutputMode::Tool)
        .build()
        .prompt("bounded recovery")
        .max_turns(2)
        .await?;
    let native_constraint = native_model
        .requests()
        .first()
        .is_some_and(|request| request.output_schema.is_some());
    let tool_names = tool_model
        .requests()
        .first()
        .map(|request| {
            request
                .tools
                .iter()
                .map(|tool| tool.name.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let prompted_constraint = prompted_model
        .requests()
        .first()
        .is_some_and(|request| request.output_schema.is_none());
    let auto_native_constraint = auto_native_model
        .requests()
        .first()
        .is_some_and(|request| request.output_schema.is_some() && request.tools.is_empty());
    let auto_tool_constraint = auto_tool_model.requests().first().is_some_and(|request| {
        request.output_schema.is_none()
            && request.tools.iter().any(|tool| tool.name == "final_result")
            && request
                .tools
                .iter()
                .any(|tool| tool.name == "final_result_1")
    });
    let mut report = report(ScenarioId::StructuredOutput);
    evidence(
        &mut report,
        0,
        native.contains("native")
            && tool.contains("tool")
            && prompted.contains("prompted")
            && auto_native.contains("auto-native")
            && auto_tool.contains("auto-tool")
            && auto_native_constraint
            && auto_tool_constraint,
    )?;
    evidence(
        &mut report,
        1,
        native_constraint
            && prompted_constraint
            && tool_names.iter().any(|name| name == "final_result")
            && tool_names.iter().any(|name| name == "final_result_1"),
    )?;
    evidence(
        &mut report,
        2,
        recovery == "invalid-two" && recovery_model.request_count() == 2,
    )?;
    Ok(report)
}

#[derive(Clone, Copy)]
struct PortableNamedTool<const N: usize>;

#[derive(Deserialize)]
struct NoArgs {}

impl<const N: usize> PortableTool for PortableNamedTool<N> {
    const NAME: &'static str = "final_result";
    type Args = NoArgs;
    type Output = String;
    type Error = Infallible;

    fn description(&self) -> String {
        "collision tool".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type":"object"})
    }

    async fn call(&self, _arguments: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(format!("collision-{N}"))
    }
}

async fn memory() -> Result<SharedReport> {
    let model = MockCompletionModel::text("done");
    let memory = CountingMemory::default();
    let agent = AgentBuilder::new(model.clone())
        .memory(memory.clone())
        .conversation("thread")
        .build();
    let response = agent.prompt("prompt").extended_details().await?;

    let multi_memory = CountingMemory::default();
    let multi_model = MockCompletionModel::new([
        MockTurn::tool_call(
            "memory-tool",
            "counting_portable_tool",
            serde_json::json!({"value":"remembered"}),
        ),
        MockTurn::text("multi-step done"),
    ]);
    let multi_agent = AgentBuilder::new(multi_model)
        .tool(CountingPortableTool::default())
        .memory(multi_memory.clone())
        .conversation("multi-step")
        .build();
    let multi_result = multi_agent.prompt("multi").max_turns(2).await?;

    let stopped_memory = CountingMemory::default();
    let stopped_model = MockCompletionModel::text("unused");
    let stopped_agent = AgentBuilder::new(stopped_model)
        .memory(stopped_memory.clone())
        .conversation("thread")
        .add_hook(StopBeforeModel)
        .build();
    let _ = stopped_agent.prompt("stop").await;
    let failed_memory = CountingLoadFailure::default();
    let failed_model = MockCompletionModel::text("unused");
    let load_failure = AgentBuilder::new(failed_model.clone())
        .memory(failed_memory.clone())
        .conversation("thread")
        .build()
        .prompt("load")
        .await;
    let append_result = AgentBuilder::new(MockCompletionModel::text("append survives"))
        .memory(AppendFailingMemory::new("classic append failure"))
        .conversation("thread")
        .build()
        .prompt("append")
        .await;
    let mut report = report(ScenarioId::Memory);
    evidence(
        &mut report,
        0,
        memory.load_count() == 1
            && model
                .requests()
                .first()
                .is_some_and(|request| request.chat_history.len() == 1),
    )?;
    evidence(
        &mut report,
        1,
        memory.append_count() == 1
            && response
                .messages
                .as_ref()
                .is_some_and(|messages| messages.len() == 2),
    )?;
    evidence(
        &mut report,
        2,
        multi_result == "multi-step done" && multi_memory.append_count() == 1,
    )?;
    evidence(
        &mut report,
        3,
        matches!(load_failure, Err(PromptError::MemoryError(_)))
            && failed_model.request_count() == 0,
    )?;
    evidence(
        &mut report,
        4,
        matches!(append_result, Ok(value) if value == "append survives"),
    )?;
    evidence(
        &mut report,
        5,
        stopped_memory.append_count() == 0 && failed_memory.append_count() == 0,
    )?;
    Ok(report)
}

async fn drain_stream(
    agent: &rig_agent::Agent<MockCompletionModel>,
    prompt: &str,
) -> Result<(Vec<String>, usize, Option<rig_agent::agent::PromptResponse>)> {
    let mut stream = agent.stream_prompt(prompt).await;
    let mut deltas = Vec::new();
    let mut provider_finals = 0;
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        match item? {
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text)) => {
                deltas.push(text.text);
            }
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Final(_)) => {
                provider_finals += 1;
            }
            MultiTurnStreamItem::FinalResponse(response) => final_response = Some(response),
            _ => {}
        }
    }
    Ok((deltas, provider_finals, final_response))
}

async fn blocking_streaming_parity() -> Result<SharedReport> {
    let blocking_model = MockCompletionModel::new([MockTurn::text("same").with_usage(Usage {
        total_tokens: 4,
        ..Usage::new()
    })]);
    let blocking = AgentBuilder::new(blocking_model)
        .build()
        .prompt("prompt")
        .extended_details()
        .await?;
    let streaming_model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("sa"),
        MockStreamEvent::text("me"),
        MockStreamEvent::final_response_with_total_tokens(4),
    ]]);
    let streaming_agent = AgentBuilder::new(streaming_model).build();
    let (_, _, streaming) = drain_stream(&streaming_agent, "prompt").await?;
    let streaming = streaming.context("stream final response")?;
    let parity = blocking.output == streaming.output
        && blocking.messages == streaming.messages
        && blocking.usage == streaming.usage
        && blocking.completion_calls == streaming.completion_calls;
    let blocking_error = AgentBuilder::new(MockCompletionModel::new([MockTurn::error(
        "parity failure",
    )]))
    .build()
    .prompt("error")
    .await;
    let streaming_error_agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([[
        MockStreamEvent::error("parity failure"),
    ]]))
    .build();
    let mut error_stream = streaming_error_agent.stream_prompt("error").await;
    let mut streaming_error = None;
    let mut streaming_final = false;
    while let Some(item) = error_stream.next().await {
        match item {
            Err(error) => streaming_error = Some(error),
            Ok(MultiTurnStreamItem::FinalResponse(_)) => streaming_final = true,
            _ => {}
        }
    }
    let mut report = report(ScenarioId::BlockingStreamingParity);
    evidence(&mut report, 0, parity)?;
    evidence(
        &mut report,
        1,
        matches!(blocking_error, Err(PromptError::CompletionError(_)))
            && streaming_error
                .as_ref()
                .is_some_and(|error| error.to_string().contains("parity failure"))
            && !streaming_final,
    )?;
    Ok(report)
}

async fn provider_final_exposure() -> Result<SharedReport> {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_total_tokens(1),
    ]]);
    let agent = AgentBuilder::new(model).build();
    let (_, finals, final_response) = drain_stream(&agent, "prompt").await?;

    let failing_model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("early"),
        MockStreamEvent::final_response_with_total_tokens(1),
        MockStreamEvent::error("late"),
    ]]);
    let failing_agent = AgentBuilder::new(failing_model).build();
    let mut stream = failing_agent.stream_prompt("prompt").await;
    let mut saw_run_final = false;
    let mut saw_error = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_run_final = true,
            Err(_) => saw_error = true,
            _ => {}
        }
    }
    let mut report = report(ScenarioId::ProviderFinalExposure);
    evidence(&mut report, 0, finals == 1 && final_response.is_some())?;
    evidence(&mut report, 1, saw_error && !saw_run_final)?;
    Ok(report)
}

async fn provisional_streaming() -> Result<SharedReport> {
    let model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("retry"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
        [
            MockStreamEvent::text("accepted"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
    ]);
    let agent = AgentBuilder::new(model)
        .add_hook(RepeatText("retry"))
        .build();
    let mut stream = agent.stream_prompt("prompt").max_turns(2).await;
    let mut saw_retry_delta = false;
    let mut saw_retry_marker = false;
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        match item? {
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text)) => {
                saw_retry_delta |= text.text == "retry";
            }
            MultiTurnStreamItem::ModelTurnRetried { .. } => saw_retry_marker = true,
            MultiTurnStreamItem::FinalResponse(response) => final_response = Some(response),
            _ => {}
        }
    }
    let final_response = final_response.context("final response")?;
    let history = final_response.messages.context("stream history")?;

    let rejected_agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("provider-rejected"),
        MockStreamEvent::error("reject after delta"),
    ]]))
    .build();
    let mut rejected_stream = rejected_agent.stream_prompt("reject").await;
    let mut rejected_delta = false;
    let mut rejected_error = false;
    let mut rejected_final = false;
    while let Some(item) = rejected_stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text))) => {
                rejected_delta |= text.text == "provider-rejected";
            }
            Ok(MultiTurnStreamItem::FinalResponse(_)) => rejected_final = true,
            Err(_) => rejected_error = true,
            _ => {}
        }
    }

    let stopped_agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("stop-me"),
        MockStreamEvent::final_response_with_total_tokens(1),
    ]]))
    .add_hook(StopText("stop-me"))
    .build();
    let mut stopped_stream = stopped_agent.stream_prompt("stop").await;
    let mut stopped_delta = false;
    let mut stopped_history_clean = false;
    let mut stopped_final = false;
    while let Some(item) = stopped_stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text))) => {
                stopped_delta |= text.text == "stop-me";
            }
            Ok(MultiTurnStreamItem::FinalResponse(_)) => stopped_final = true,
            Err(error) => {
                stopped_history_clean = match error {
                    StreamingError::Prompt(error) => matches!(
                        error.as_ref(),
                        PromptError::PromptCancelled { chat_history, .. }
                            if chat_history == &[Message::user("stop")]
                    ),
                    StreamingError::Completion(_) => false,
                };
            }
            _ => {}
        }
    }

    let cancellation_memory = CountingMemory::default();
    let cancellation_agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("cancel-me"),
        MockStreamEvent::final_response_with_total_tokens(1),
    ]]))
    .memory(cancellation_memory.clone())
    .conversation("provisional-cancel")
    .build();
    let mut cancellation_stream = cancellation_agent.stream_prompt("cancel").await;
    let cancellation_delta = matches!(
        cancellation_stream.next().await,
        Some(Ok(MultiTurnStreamItem::StreamAssistantItem(
            StreamedAssistantContent::Text(_)
        )))
    );
    drop(cancellation_stream);
    tokio::task::yield_now().await;
    let mut report = report(ScenarioId::ProvisionalStreaming);
    evidence(&mut report, 0, saw_retry_delta)?;
    evidence(
        &mut report,
        1,
        saw_retry_marker && !serde_json::to_string(&history)?.contains("retry"),
    )?;
    evidence(
        &mut report,
        2,
        rejected_delta && rejected_error && !rejected_final,
    )?;
    evidence(
        &mut report,
        3,
        stopped_delta && stopped_history_clean && !stopped_final,
    )?;
    evidence(
        &mut report,
        4,
        cancellation_delta && cancellation_memory.append_count() == 0,
    )?;
    Ok(report)
}

struct SkipAllTools;

impl AgentHook for SkipAllTools {
    async fn on_tool_call(&self, _ctx: &HookContext, _event: ToolCall<'_>) -> ToolCallAction {
        ToolCallAction::skip("suppressed")
    }
}

async fn tool_suppression() -> Result<SharedReport> {
    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "call",
            "counting_portable_tool",
            serde_json::json!({"value":"no"}),
        ),
        MockTurn::text("done"),
    ]);
    let tool = CountingPortableTool::default();
    let agent = AgentBuilder::new(model)
        .tool(tool.clone())
        .add_hook(SkipAllTools)
        .build();
    let response = agent.prompt("skip").max_turns(2).await?;

    let invalid = invalid_run(InvalidToolCallAction::skip("invalid"), "missing", false).await?;
    let stopped = invalid_run(InvalidToolCallAction::stop("stop"), "missing", false).await?;

    let output_model = MockCompletionModel::new([MockTurn::from_contents([
        AssistantContent::tool_call(
            "peer",
            "counting_portable_tool",
            serde_json::json!({"value":"must-not-run"}),
        ),
        AssistantContent::tool_call(
            "output",
            "final_result",
            serde_json::json!({"answer":"done"}),
        ),
    ])?]);
    let output_tool = CountingPortableTool::default();
    let output = AgentBuilder::new(output_model)
        .tool(output_tool.clone())
        .output_schema::<Answer>()
        .output_mode(OutputMode::Tool)
        .build()
        .prompt("finalize")
        .await?;

    let cancellation_model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::tool_call(
            "cancelled-call",
            "counting_portable_tool",
            serde_json::json!({"value":"must-not-run"}),
        ),
        MockStreamEvent::final_response_with_total_tokens(1),
    ]]);
    let cancellation_tool = CountingPortableTool::default();
    let cancellation_agent = AgentBuilder::new(cancellation_model)
        .tool(cancellation_tool.clone())
        .build();
    let mut cancellation_stream = cancellation_agent.stream_prompt("cancel tool").await;
    let mut saw_tool_call = false;
    while let Some(item) = cancellation_stream.next().await {
        if matches!(
            item,
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ToolCall { .. }
            ))
        ) {
            saw_tool_call = true;
            break;
        }
    }
    drop(cancellation_stream);
    tokio::task::yield_now().await;
    let mut report = report(ScenarioId::ToolSuppression);
    evidence(&mut report, 0, invalid.1 == 0)?;
    evidence(
        &mut report,
        1,
        tool.calls().is_empty() && response == "done",
    )?;
    evidence(
        &mut report,
        2,
        output.contains("done") && output_tool.calls().is_empty(),
    )?;
    evidence(&mut report, 3, stopped.1 == 0)?;
    evidence(
        &mut report,
        4,
        saw_tool_call && cancellation_tool.calls().is_empty(),
    )?;
    Ok(report)
}

#[derive(Clone)]
struct ConcurrencyProbe {
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
    completed: Arc<Mutex<Vec<String>>>,
}

#[derive(Deserialize)]
struct ProbeArgs {
    value: String,
    delay_ms: u64,
}

impl PortableTool for ConcurrencyProbe {
    const NAME: &'static str = "classic_concurrency_probe";
    type Args = ProbeArgs;
    type Output = serde_json::Value;
    type Error = Infallible;

    fn description(&self) -> String {
        "classic concurrency probe".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type":"object"})
    }

    async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(arguments.delay_ms)).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        self.completed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(arguments.value.clone());
        Ok(serde_json::json!({"value":arguments.value}))
    }
}

async fn concurrency() -> Result<SharedReport> {
    let model = MockCompletionModel::new([
        MockTurn::from_contents([
            AssistantContent::tool_call(
                "slow",
                "classic_concurrency_probe",
                serde_json::json!({"value":"slow","delay_ms":20}),
            ),
            AssistantContent::tool_call(
                "fast",
                "classic_concurrency_probe",
                serde_json::json!({"value":"fast","delay_ms":1}),
            ),
        ])?,
        MockTurn::text("done"),
    ]);
    let probe = ConcurrencyProbe {
        active: Arc::new(AtomicUsize::new(0)),
        max_active: Arc::new(AtomicUsize::new(0)),
        completed: Arc::new(Mutex::new(Vec::new())),
    };
    let agent = AgentBuilder::new(model).tool(probe.clone()).build();
    let response = agent
        .prompt("parallel")
        .max_turns(2)
        .tool_concurrency(2)
        .extended_details()
        .await?;
    let messages = response.messages.context("history")?;
    let ids = messages
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => Some(content),
            _ => None,
        })
        .flat_map(|content| content.iter())
        .filter_map(|content| match content {
            rig_core::message::UserContent::ToolResult(result) => Some(result.id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let completion_order = probe
        .completed
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    let mut report = report(ScenarioId::Concurrency);
    evidence(&mut report, 0, probe.max_active.load(Ordering::SeqCst) <= 2)?;
    evidence(&mut report, 1, ids)?;
    evidence(&mut report, 2, completion_order)?;
    Ok(report)
}

async fn stale_result_handling() -> Result<SharedReport> {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("early"),
        MockStreamEvent::final_response_with_total_tokens(1),
        MockStreamEvent::final_response_with_total_tokens(1),
        MockStreamEvent::error("late"),
    ]]);
    let agent = AgentBuilder::new(model).build();
    let mut stream = agent.stream_prompt("prompt").await;
    let mut provider_finals = 0;
    let mut run_finals = 0;
    let mut errors = 0;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Final(_))) => {
                provider_finals += 1;
            }
            Ok(MultiTurnStreamItem::FinalResponse(_)) => run_finals += 1,
            Err(_) => errors += 1,
            _ => {}
        }
    }
    let duplicate_agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("duplicate-safe"),
        MockStreamEvent::final_response_with_total_tokens(2),
        MockStreamEvent::final_response_with_total_tokens(2),
    ]]))
    .build();
    let (_, duplicate_finals, duplicate_response) =
        drain_stream(&duplicate_agent, "duplicate").await?;

    let cancellation_memory = CountingMemory::default();
    let cancellation_agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("cancelled-late"),
        MockStreamEvent::final_response_with_total_tokens(1),
    ]]))
    .memory(cancellation_memory.clone())
    .conversation("stale-cancel")
    .build();
    let mut cancellation_stream = cancellation_agent.stream_prompt("cancel").await;
    let _ = cancellation_stream.next().await;
    drop(cancellation_stream);
    tokio::task::yield_now().await;
    let mut report = report(ScenarioId::StaleResultHandling);
    evidence(
        &mut report,
        0,
        duplicate_finals == 1
            && duplicate_response.is_some_and(|response| {
                response.usage.total_tokens == 2 && response.completion_calls.len() == 1
            }),
    )?;
    evidence(
        &mut report,
        1,
        provider_finals == 0 && run_finals == 0 && errors == 1,
    )?;
    let superseded = scenario(ScenarioId::StaleResultHandling)
        .and_then(|definition| definition.observations.get(2))
        .map(|observation| observation.description)
        .context("classic superseded-ingress boundary")?;
    report.observe_not_applicable(
        superseded,
        "the classic runtime directly owns and drops each provider future, so it exposes no superseded external ingress surface",
    )?;
    evidence(&mut report, 3, cancellation_memory.append_count() == 0)?;
    let identity = scenario(ScenarioId::StaleResultHandling)
        .and_then(|definition| definition.observations.get(4))
        .map(|observation| observation.description)
        .context("classic external identity boundary")?;
    report.observe_not_applicable(
        identity,
        "the classic runtime exposes no world, tenant, generation, or external effect-ingress identity header",
    )?;
    Ok(report)
}

#[tokio::test]
async fn every_shared_scenario_passes_the_classic_runtime() -> Result<()> {
    let reports = [
        model_call_budgets().await?,
        canonical_transcript().await?,
        tool_pairing().await?,
        usage_accounting().await?,
        invalid_tool_recovery().await?,
        response_retry_rollback().await?,
        stop_and_cancellation().await?,
        structured_output().await?,
        memory().await?,
        blocking_streaming_parity().await?,
        provider_final_exposure().await?,
        provisional_streaming().await?,
        tool_suppression().await?,
        concurrency().await?,
        stale_result_handling().await?,
    ];
    assert_eq!(reports.len(), ALL_SCENARIOS.len());
    for report in reports {
        verify_report(&report)?;
    }
    Ok(())
}
