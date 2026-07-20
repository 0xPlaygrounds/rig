#![allow(clippy::panic_in_result_fn)]

use std::{
    collections::BTreeSet,
    convert::Infallible,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result};
use rig_core::{
    completion::{
        AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
        Message, Usage,
    },
    memory::{ConversationMemory, MemoryError},
    streaming::StreamingCompletionResponse,
    test_utils::{
        AppendFailingMemory, CountingMemory, MockCompletionModel, MockResponse, MockStreamEvent,
        MockTurn,
    },
    tool::{PortableDynamicTool, PortableTool, ToolOutput},
    wasm_compat::WasmBoxedFuture,
};
use rig_ecs::{
    AgentSpec, EcsModelExt, EffectCompletion, EffectIngress, EffectRejectionReason,
    InvalidToolPolicy, LocalRuntime, MemoryEffectOutput, ModelEffectOutput, OutputMode,
    ProvisionalDelta, ResponseRetryPolicy, RunEvent, RuntimeConfig, RuntimeError, StreamingMode,
    StructuredOutputPolicy, TenantId, TerminalReason,
};
use rig_runtime_conformance::{
    ALL_SCENARIOS, CountingPortableTool, ScenarioId, ScenarioReport, scenario, verify_report,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone)]
struct DelayedModel {
    inner: MockCompletionModel,
    delay: Duration,
}

impl CompletionModel for DelayedModel {
    type Response = MockResponse;
    type StreamingResponse = MockResponse;
    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>) -> Self {
        Self {
            inner: MockCompletionModel::default(),
            delay: Duration::ZERO,
        }
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        tokio::time::sleep(self.delay).await;
        self.inner.completion(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        tokio::time::sleep(self.delay).await;
        self.inner.stream(request).await
    }
}

fn test_runtime() -> Result<LocalRuntime> {
    Ok(LocalRuntime::with_config(RuntimeConfig {
        effect_timeout: Duration::from_secs(2),
        max_schedule_passes: 2_048,
        terminal_retention_ticks: 2,
        ..RuntimeConfig::default()
    })?)
}

fn report(id: ScenarioId) -> ScenarioReport {
    ScenarioReport::new("rig-ecs", id)
}

fn evidence<T>(report: &mut ScenarioReport, index: usize, actual: T) -> Result<()>
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

fn role_sequence_valid(messages: &[Message]) -> bool {
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

async fn model_call_budgets() -> Result<ScenarioReport> {
    let zero_model = MockCompletionModel::text("unused");
    let mut zero_runtime = test_runtime()?;
    let zero_agent = zero_runtime.spawn_agent(
        zero_model
            .clone()
            .into_ecs_agent_builder()
            .max_model_calls(0)
            .build(),
    )?;
    let zero_error = zero_runtime
        .run(zero_agent, "zero")
        .await
        .err()
        .context("zero must reject before I/O")?;

    let retry_model = MockCompletionModel::new([MockTurn::text(""), MockTurn::text("accepted")]);
    let mut retry_runtime = test_runtime()?;
    let retry_agent = retry_runtime.spawn_agent(
        retry_model
            .clone()
            .into_ecs_agent_builder()
            .max_model_calls(2)
            .response_retry_policy(ResponseRetryPolicy::RejectEmpty { max_retries: 1 })
            .build(),
    )?;
    let retry_result = retry_runtime.run(retry_agent, "retry").await?;

    let mut report = report(ScenarioId::ModelCallBudgets);
    evidence(
        &mut report,
        0,
        matches!(zero_error, RuntimeError::ModelCallBudgetExhausted)
            && zero_model.request_count() == 0,
    )?;
    evidence(
        &mut report,
        1,
        (retry_model.request_count(), retry_result.model_calls.len()),
    )?;
    Ok(report)
}

async fn canonical_transcript() -> Result<ScenarioReport> {
    let model = MockCompletionModel::new([MockTurn::text(""), MockTurn::text("kept")]);
    let mut runtime = test_runtime()?;
    let agent = runtime.spawn_agent(
        model
            .into_ecs_agent_builder()
            .response_retry_policy(ResponseRetryPolicy::RejectEmpty { max_retries: 1 })
            .build(),
    )?;
    let result = runtime.run(agent, "prompt").await?;

    let rejected_absent = result
        .transcript
        .iter()
        .all(|message| !matches!(message, Message::Assistant { content, .. } if content.iter().any(|item| matches!(item, AssistantContent::Text(text) if text.text.is_empty()))));
    let mut report = report(ScenarioId::CanonicalTranscript);
    evidence(&mut report, 0, role_sequence_valid(&result.transcript))?;
    evidence(&mut report, 1, result.transcript.len())?;
    evidence(&mut report, 2, rejected_absent)?;
    Ok(report)
}

async fn tool_pairing() -> Result<ScenarioReport> {
    let first = AssistantContent::tool_call(
        "first",
        "counting_portable_tool",
        serde_json::json!({"value":"first"}),
    );
    let second = AssistantContent::tool_call(
        "second",
        "counting_portable_tool",
        serde_json::json!({"value":"second"}),
    );
    let model = MockCompletionModel::new([
        MockTurn::from_contents([first, second])?,
        MockTurn::text("done"),
    ]);
    let tool = CountingPortableTool::default();
    let mut runtime = test_runtime()?;
    let agent = runtime.spawn_agent(model.into_ecs_agent_builder().build())?;
    runtime.install_tool(agent, tool)?;
    let result = runtime.run(agent, "pair").await?;
    let result_ids = match result.transcript.get(2) {
        Some(Message::User { content }) => content
            .iter()
            .filter_map(|content| match content {
                rig_core::message::UserContent::ToolResult(result) => Some(result.id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    let mut report = report(ScenarioId::ToolCallResultPairing);
    evidence(&mut report, 0, result_ids.len())?;
    evidence(&mut report, 1, result_ids)?;
    Ok(report)
}

async fn usage_accounting() -> Result<ScenarioReport> {
    let usage = Usage {
        total_tokens: 7,
        ..Usage::new()
    };
    let model = DelayedModel {
        inner: MockCompletionModel::new([MockTurn::text("unused")]),
        delay: Duration::from_millis(50),
    };
    let mut runtime = test_runtime()?;
    let agent = runtime.spawn_agent(model.into_ecs_agent_builder().build())?;
    let handle = runtime.start_run(agent, "usage")?;
    let _ = runtime.step(handle).await?;
    let header = runtime
        .active_effect_headers(handle)?
        .first()
        .copied()
        .context("model operation must be active")?;
    for _ in 0..2 {
        runtime.ingest(EffectIngress::Completion(EffectCompletion::Model {
            header,
            result: Ok(ModelEffectOutput {
                choice: vec![AssistantContent::text("done")],
                usage,
                message_id: Some("usage-message".to_string()),
                raw_final: None,
            }),
        }))?;
    }
    let result = runtime.drive_to_terminal(handle).await?;
    let billed_model = MockCompletionModel::new([
        MockTurn::text("").with_usage(Usage {
            total_tokens: 3,
            ..Usage::new()
        }),
        MockTurn::text("accepted").with_usage(Usage {
            total_tokens: 5,
            ..Usage::new()
        }),
    ]);
    let mut billed_runtime = test_runtime()?;
    let billed_agent = billed_runtime.spawn_agent(
        billed_model
            .into_ecs_agent_builder()
            .response_retry_policy(ResponseRetryPolicy::RejectEmpty { max_retries: 1 })
            .max_model_calls(2)
            .build(),
    )?;
    let billed = billed_runtime.run(billed_agent, "billed retry").await?;
    let mut report = report(ScenarioId::UsageAccounting);
    evidence(&mut report, 0, result.model_calls.len() == 1)?;
    evidence(
        &mut report,
        1,
        billed.model_calls.len() == 2
            && billed
                .model_calls
                .iter()
                .filter(|call| call.accepted)
                .count()
                == 1
            && billed.usage.total_tokens == 8,
    )?;
    evidence(
        &mut report,
        2,
        result
            .events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    RunEvent::EffectRejected(EffectRejectionReason::Duplicate)
                )
            })
            .count()
            == 1
            && result.usage.total_tokens == 7,
    )?;
    Ok(report)
}

async fn run_invalid_policy(
    policy: InvalidToolPolicy,
    name: &str,
    install: bool,
) -> Result<(TerminalReason, usize, usize)> {
    let model = MockCompletionModel::new([
        MockTurn::tool_call("invalid", name, serde_json::json!({"value":"x"})),
        MockTurn::text("recovered"),
    ]);
    let tool = CountingPortableTool::default();
    let mut runtime = test_runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_ecs_agent_builder()
            .invalid_tool_policy(policy)
            .build(),
    )?;
    if install {
        runtime.install_tool(agent, tool.clone())?;
    }
    let handle = runtime.start_run(agent, "invalid")?;
    let result = runtime.drive_to_terminal(handle).await?;
    Ok((
        result.terminal_reason,
        tool.calls().len(),
        model.request_count(),
    ))
}

async fn invalid_tool_recovery() -> Result<ScenarioReport> {
    let fail = run_invalid_policy(InvalidToolPolicy::Fail, "missing", false).await?;
    let retry = run_invalid_policy(
        InvalidToolPolicy::Retry { max_retries: 1 },
        "missing",
        false,
    )
    .await?;
    let repair =
        run_invalid_policy(InvalidToolPolicy::Repair, "COUNTING_PORTABLE_TOOL", true).await?;
    let skip = run_invalid_policy(InvalidToolPolicy::Skip, "missing", false).await?;
    let stop = run_invalid_policy(InvalidToolPolicy::Stop, "missing", false).await?;
    let distinguishable = matches!(fail.0, TerminalReason::Failed { .. })
        && matches!(retry.0, TerminalReason::Completed)
        && matches!(repair.0, TerminalReason::Completed)
        && matches!(skip.0, TerminalReason::Completed)
        && matches!(stop.0, TerminalReason::Stopped)
        && retry.2 == 2
        && repair.1 == 1;
    let suppressed_never_executed = fail.1 == 0 && retry.1 == 0 && skip.1 == 0 && stop.1 == 0;
    let mut report = report(ScenarioId::InvalidToolRecovery);
    evidence(&mut report, 0, distinguishable)?;
    evidence(&mut report, 1, suppressed_never_executed)?;
    Ok(report)
}

async fn response_retry_rollback() -> Result<ScenarioReport> {
    let model = MockCompletionModel::new([MockTurn::text(""), MockTurn::text("accepted")]);
    let mut runtime = test_runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_ecs_agent_builder()
            .response_retry_policy(ResponseRetryPolicy::RejectEmpty { max_retries: 1 })
            .max_model_calls(2)
            .build(),
    )?;
    let result = runtime.run(agent, "prompt").await?;

    let requests = model.requests();
    let corrective_feedback = requests
        .get(1)
        .is_some_and(|request| request.chat_history.len() > 1);

    let tool_model = MockCompletionModel::new([
        MockTurn::tool_call(
            "real",
            "counting_portable_tool",
            serde_json::json!({"value":"executed"}),
        ),
        MockTurn::text("tool continuation"),
    ]);
    let tool = CountingPortableTool::default();
    let mut tool_runtime = test_runtime()?;
    let tool_agent = tool_runtime.spawn_agent(
        tool_model
            .clone()
            .into_ecs_agent_builder()
            .response_retry_policy(ResponseRetryPolicy::RejectEmpty { max_retries: 1 })
            .max_model_calls(2)
            .build(),
    )?;
    tool_runtime.install_tool(tool_agent, tool.clone())?;
    let tool_result = tool_runtime.run(tool_agent, "tool-bearing").await?;

    let repeated_model = MockCompletionModel::new([MockTurn::text(""), MockTurn::text("")]);
    let mut repeated_runtime = test_runtime()?;
    let repeated_agent = repeated_runtime.spawn_agent(
        repeated_model
            .clone()
            .into_ecs_agent_builder()
            .response_retry_policy(ResponseRetryPolicy::RejectEmpty { max_retries: 1 })
            .max_model_calls(2)
            .build(),
    )?;
    let repeated_handle = repeated_runtime.start_run(repeated_agent, "repeat policy")?;
    let repeated = repeated_runtime.drive_to_terminal(repeated_handle).await?;
    let rejected_absent = result.transcript.len() == 2
        && result.text.as_deref() == Some("accepted")
        && result
            .model_calls
            .iter()
            .filter(|record| record.accepted)
            .count()
            == 1;
    let mut report = report(ScenarioId::ResponseRetryRollback);
    evidence(&mut report, 0, requests.len() == 2)?;
    evidence(
        &mut report,
        1,
        tool_result.text.as_deref() == Some("tool continuation")
            && tool.calls().len() == 1
            && tool_model.request_count() == 2,
    )?;
    evidence(
        &mut report,
        2,
        corrective_feedback
            && requests.get(1).is_some_and(|request| {
                serde_json::to_string(&request.chat_history)
                    .is_ok_and(|history| history.contains("previous response was empty"))
            }),
    )?;
    evidence(
        &mut report,
        3,
        matches!(repeated.terminal_reason, TerminalReason::Failed { .. })
            && repeated_model.request_count() == 2,
    )?;
    evidence(
        &mut report,
        4,
        requests.len() == 2 && result.model_calls.len() == 2,
    )?;
    evidence(&mut report, 5, rejected_absent)?;
    Ok(report)
}

async fn stop_and_cancellation() -> Result<ScenarioReport> {
    let stopped = run_invalid_policy(InvalidToolPolicy::Stop, "missing", false).await?;
    let model = MockCompletionModel::text("late");
    let mut runtime = test_runtime()?;
    let agent = runtime.spawn_agent(model.into_ecs_agent_builder().build())?;
    let handle = runtime.start_run(agent, "cancel")?;
    let _ = runtime.step(handle).await?;
    let header = runtime
        .active_effect_headers(handle)?
        .first()
        .copied()
        .context("active model header")?;
    runtime.cancel(handle)?;
    runtime.ingest(EffectIngress::Delta {
        header,
        sequence: 0,
        delta: ProvisionalDelta::Text("too-late".to_string()),
    })?;
    let result = runtime.drive_to_terminal(handle).await?;
    let terminal_blocks_commit = matches!(result.terminal_reason, TerminalReason::Cancelled)
        && result.transcript == [Message::user("cancel")]
        && result
            .events
            .iter()
            .any(|event| matches!(event, RunEvent::EffectRejected(EffectRejectionReason::Late)));
    let before_cleanup = runtime.finish_run(handle).is_ok();
    // Stepping the run's own handle would renew its observation lease, so the
    // retention window is aged through a second run's schedule passes.
    let pump_agent = runtime.spawn_agent(
        MockCompletionModel::text("pump")
            .into_ecs_agent_builder()
            .build(),
    )?;
    let pump = runtime.start_run(pump_agent, "pump")?;
    let _ = runtime.drive_to_terminal(pump).await?;
    let _ = runtime.step(pump).await;
    let _ = runtime.step(pump).await;
    let after_cleanup = matches!(runtime.finish_run(handle), Err(RuntimeError::UnknownRun(_)));
    let mut report = report(ScenarioId::StopAndCancellation);
    evidence(
        &mut report,
        0,
        matches!(stopped.0, TerminalReason::Stopped) && stopped.1 == 0,
    )?;
    evidence(&mut report, 1, terminal_blocks_commit)?;
    evidence(
        &mut report,
        2,
        result
            .events
            .iter()
            .any(|event| matches!(event, RunEvent::Terminal(_))),
    )?;
    evidence(&mut report, 3, before_cleanup && after_cleanup)?;
    Ok(report)
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Answer {
    answer: String,
}

fn synthetic_name(schema: &schemars::Schema) -> String {
    let digest = Sha256::digest(schema.as_value().to_string().as_bytes());
    let prefix = digest
        .iter()
        .take(4)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("__rig_output_{prefix}")
}

async fn structured_output() -> Result<ScenarioReport> {
    let schema = schemars::schema_for!(Answer);
    let base_name = synthetic_name(&schema);
    let collision_name = format!("{base_name}_");
    let model = MockCompletionModel::new([MockTurn::tool_call(
        "output",
        &collision_name,
        serde_json::json!({"answer":"ok"}),
    )]);
    let mut runtime = test_runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_ecs_agent_builder()
            .structured_output::<Answer>(StructuredOutputPolicy {
                mode: OutputMode::Tool,
                max_retries: 1,
                best_effort: false,
            })
            .build(),
    )?;
    runtime.install_dynamic_tool(
        agent,
        PortableDynamicTool::new(
            base_name.clone(),
            "collision",
            serde_json::json!({"type":"object"}),
            |_| Box::pin(async { Ok(ToolOutput::text("must not execute")) }),
        ),
    )?;
    let result = runtime.run(agent, "structured").await?;
    let request = model.requests().into_iter().next().context("request")?;

    let prompted_model = MockCompletionModel::text(r#"{"answer":"prompted"}"#);
    let mut prompted_runtime = test_runtime()?;
    let prompted_agent = prompted_runtime.spawn_agent(
        prompted_model
            .clone()
            .into_ecs_agent_builder()
            .structured_output::<Answer>(StructuredOutputPolicy {
                mode: OutputMode::Prompted,
                max_retries: 0,
                best_effort: false,
            })
            .build(),
    )?;
    let prompted = prompted_runtime.run(prompted_agent, "prompted").await?;
    let prompted_request = prompted_model.requests();

    let native_model = MockCompletionModel::text(r#"{"answer":"native"}"#);
    let mut native_runtime = test_runtime()?;
    let native_agent = native_runtime.spawn_agent(
        native_model
            .clone()
            .into_ecs_agent_builder()
            .structured_output::<Answer>(StructuredOutputPolicy {
                mode: OutputMode::Native,
                max_retries: 0,
                best_effort: false,
            })
            .build(),
    )?;
    let native = native_runtime.run(native_agent, "native").await?;

    let auto_native_model = MockCompletionModel::text(r#"{"answer":"auto-native"}"#);
    let mut auto_native_runtime = test_runtime()?;
    let auto_native_agent = auto_native_runtime.spawn_agent(
        auto_native_model
            .clone()
            .into_ecs_agent_builder()
            .structured_output::<Answer>(StructuredOutputPolicy::default())
            .build(),
    )?;
    let auto_native = auto_native_runtime
        .run(auto_native_agent, "auto native")
        .await?;

    let auto_tool_model = MockCompletionModel::new([MockTurn::tool_call(
        "auto-output",
        &base_name,
        serde_json::json!({"answer":"auto-tool"}),
    )]);
    let mut auto_tool_runtime = test_runtime()?;
    let auto_tool_agent = auto_tool_runtime.spawn_agent(
        auto_tool_model
            .clone()
            .into_ecs_agent_builder()
            .structured_output::<Answer>(StructuredOutputPolicy::default())
            .build(),
    )?;
    auto_tool_runtime.install_tool(auto_tool_agent, CountingPortableTool::default())?;
    let auto_tool = auto_tool_runtime.run(auto_tool_agent, "auto tool").await?;

    let recovery_model =
        MockCompletionModel::new([MockTurn::text("invalid-one"), MockTurn::text("invalid-two")]);
    let mut recovery_runtime = test_runtime()?;
    let recovery_agent = recovery_runtime.spawn_agent(
        recovery_model
            .clone()
            .into_ecs_agent_builder()
            .max_model_calls(2)
            .structured_output::<Answer>(StructuredOutputPolicy {
                mode: OutputMode::Native,
                max_retries: 1,
                best_effort: true,
            })
            .build(),
    )?;
    let recovery = recovery_runtime
        .run(recovery_agent, "bounded recovery")
        .await?;

    let modes_work = result.structured_output == Some(serde_json::json!({"answer":"ok"}))
        && prompted.structured_output == Some(serde_json::json!({"answer":"prompted"}))
        && native.structured_output == Some(serde_json::json!({"answer":"native"}))
        && auto_native.structured_output == Some(serde_json::json!({"answer":"auto-native"}))
        && auto_tool.structured_output == Some(serde_json::json!({"answer":"auto-tool"}))
        && prompted_request
            .first()
            .is_some_and(|request| request.output_schema.is_none())
        && native_model
            .requests()
            .first()
            .is_some_and(|request| request.output_schema.is_some())
        && auto_native_model
            .requests()
            .first()
            .is_some_and(|request| request.output_schema.is_some() && request.tools.is_empty())
        && auto_tool_model.requests().first().is_some_and(|request| {
            request.output_schema.is_none()
                && request
                    .tools
                    .iter()
                    .any(|tool| tool.name == "counting_portable_tool")
                && request.tools.iter().any(|tool| tool.name == base_name)
        });
    let mut report = report(ScenarioId::StructuredOutput);
    evidence(&mut report, 0, modes_work)?;
    evidence(
        &mut report,
        1,
        request.tools.iter().any(|tool| tool.name == collision_name)
            && request.tools.iter().any(|tool| tool.name == base_name),
    )?;
    evidence(
        &mut report,
        2,
        recovery.text.as_deref() == Some("invalid-two")
            && recovery.structured_output.is_none()
            && recovery_model.request_count() == 2,
    )?;
    Ok(report)
}

async fn memory() -> Result<ScenarioReport> {
    let model = MockCompletionModel::text("done");
    let memory = CountingMemory::default();
    let mut runtime = test_runtime()?;
    let tenant_id = TenantId::new();
    let memory_id = runtime.register_memory(tenant_id, memory.clone());
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_ecs_agent_builder()
            .tenant(tenant_id)
            .memory(memory_id, "thread")
            .build(),
    )?;
    let result = runtime.run(agent, "prompt").await?;

    let multi_memory = CountingMemory::default();
    let multi_model = MockCompletionModel::new([
        MockTurn::tool_call(
            "memory-tool",
            "counting_portable_tool",
            serde_json::json!({"value":"remembered"}),
        ),
        MockTurn::text("multi-step done"),
    ]);
    let mut multi_runtime = test_runtime()?;
    let multi_tenant = TenantId::new();
    let multi_memory_id = multi_runtime.register_memory(multi_tenant, multi_memory.clone());
    let multi_agent = multi_runtime.spawn_agent(
        multi_model
            .into_ecs_agent_builder()
            .tenant(multi_tenant)
            .memory(multi_memory_id, "multi-step")
            .max_model_calls(2)
            .build(),
    )?;
    multi_runtime.install_tool(multi_agent, CountingPortableTool::default())?;
    let multi_result = multi_runtime.run(multi_agent, "multi").await?;

    let stopped_memory = CountingMemory::default();
    let stopped_model =
        MockCompletionModel::new([MockTurn::tool_call("bad", "missing", serde_json::json!({}))]);
    let mut stopped_runtime = test_runtime()?;
    let stopped_tenant_id = TenantId::new();
    let stopped_memory_id =
        stopped_runtime.register_memory(stopped_tenant_id, stopped_memory.clone());
    let stopped_agent = stopped_runtime.spawn_agent(
        stopped_model
            .into_ecs_agent_builder()
            .tenant(stopped_tenant_id)
            .memory(stopped_memory_id, "thread")
            .invalid_tool_policy(InvalidToolPolicy::Stop)
            .build(),
    )?;
    let handle = stopped_runtime.start_run(stopped_agent, "stop")?;
    let stopped = stopped_runtime.drive_to_terminal(handle).await?;

    let mut failing_runtime = test_runtime()?;
    let failing_tenant_id = TenantId::new();
    let failing_memory = CountingLoadFailure::default();
    let failing_memory_id =
        failing_runtime.register_memory(failing_tenant_id, failing_memory.clone());
    let failing_model = MockCompletionModel::text("unused");
    let failing_agent = failing_runtime.spawn_agent(
        failing_model
            .clone()
            .into_ecs_agent_builder()
            .tenant(failing_tenant_id)
            .memory(failing_memory_id, "thread")
            .build(),
    )?;
    let load_failure = failing_runtime.run(failing_agent, "load").await;

    let mut append_runtime = test_runtime()?;
    let append_tenant_id = TenantId::new();
    let append_memory = append_runtime.register_memory(
        append_tenant_id,
        AppendFailingMemory::new("ECS append failure"),
    );
    let append_agent = append_runtime.spawn_agent(
        MockCompletionModel::text("append survives")
            .into_ecs_agent_builder()
            .tenant(append_tenant_id)
            .memory(append_memory, "thread")
            .build(),
    )?;
    let append_result = append_runtime.run(append_agent, "append").await?;
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
        memory.append_count() == 1 && result.transcript.len() == 2,
    )?;
    evidence(
        &mut report,
        2,
        multi_result.text.as_deref() == Some("multi-step done") && multi_memory.append_count() == 1,
    )?;
    evidence(
        &mut report,
        3,
        matches!(load_failure, Err(RuntimeError::MemoryEffect(_)))
            && failing_model.request_count() == 0,
    )?;
    evidence(
        &mut report,
        4,
        append_result.text.as_deref() == Some("append survives")
            && append_result
                .failure_diagnostic
                .as_deref()
                .is_some_and(|message| message.contains("append failed")),
    )?;
    evidence(
        &mut report,
        5,
        matches!(stopped.terminal_reason, TerminalReason::Stopped)
            && stopped_memory.append_count() == 0
            && failing_memory.append_count() == 0,
    )?;
    Ok(report)
}

async fn blocking_streaming_parity() -> Result<ScenarioReport> {
    let blocking_model = MockCompletionModel::new([MockTurn::text("same").with_usage(Usage {
        total_tokens: 4,
        ..Usage::new()
    })]);
    let streaming_model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("sa"),
        MockStreamEvent::text("me"),
        MockStreamEvent::final_response_with_total_tokens(4),
    ]]);
    let mut blocking_runtime = test_runtime()?;
    let blocking_agent =
        blocking_runtime.spawn_agent(blocking_model.into_ecs_agent_builder().build())?;
    let blocking = blocking_runtime
        .run_blocking(blocking_agent, "prompt")
        .await?;
    let mut streaming_runtime = test_runtime()?;
    let streaming_agent =
        streaming_runtime.spawn_agent(streaming_model.into_ecs_agent_builder().build())?;
    let streaming = streaming_runtime
        .run_streaming::<MockResponse>(streaming_agent, "prompt")
        .await?
        .result;
    let parity = blocking.text == streaming.text
        && blocking.transcript == streaming.transcript
        && blocking.usage == streaming.usage
        && blocking.terminal_reason == streaming.terminal_reason;
    let mut blocking_error_runtime = test_runtime()?;
    let blocking_error_agent = blocking_error_runtime.spawn_agent(
        MockCompletionModel::new([MockTurn::error("parity failure")])
            .into_ecs_agent_builder()
            .build(),
    )?;
    let blocking_error = blocking_error_runtime
        .run_blocking(blocking_error_agent, "error")
        .await;
    let mut streaming_error_runtime = test_runtime()?;
    let streaming_error_agent = streaming_error_runtime.spawn_agent(
        MockCompletionModel::from_stream_turns([[MockStreamEvent::error("parity failure")]])
            .into_ecs_agent_builder()
            .streaming(StreamingMode::Streaming)
            .build(),
    )?;
    let streaming_error = streaming_error_runtime
        .run_streaming::<MockResponse>(streaming_error_agent, "error")
        .await;
    let mut report = report(ScenarioId::BlockingStreamingParity);
    evidence(&mut report, 0, parity)?;
    evidence(
        &mut report,
        1,
        matches!(blocking_error, Err(RuntimeError::ModelEffect(_)))
            && matches!(streaming_error, Err(RuntimeError::ModelEffect(_))),
    )?;
    Ok(report)
}

async fn provider_final_exposure() -> Result<ScenarioReport> {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("ok"),
        MockStreamEvent::final_response_with_total_tokens(1),
    ]]);
    let mut runtime = test_runtime()?;
    let agent = runtime.spawn_agent(model.into_ecs_agent_builder().build())?;
    let result = runtime
        .run_streaming::<MockResponse>(agent, "stream")
        .await?;
    let typed = result
        .events
        .iter()
        .any(|event| matches!(event, rig_ecs::StreamingRunEvent::ProviderFinal { .. }));

    let failing = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("early"),
        MockStreamEvent::final_response_with_total_tokens(1),
        MockStreamEvent::error("late"),
    ]]);
    let mut failing_runtime = test_runtime()?;
    let failing_agent = failing_runtime.spawn_agent(
        failing
            .into_ecs_agent_builder()
            .streaming(StreamingMode::Streaming)
            .build(),
    )?;
    let handle = failing_runtime.start_run(failing_agent, "stream")?;
    let failure = failing_runtime.drive_to_terminal(handle).await?;
    let mut report = report(ScenarioId::ProviderFinalExposure);
    evidence(&mut report, 0, typed)?;
    evidence(
        &mut report,
        1,
        failure.raw_final::<MockResponse>().is_none()
            && matches!(failure.terminal_reason, TerminalReason::Failed { .. }),
    )?;
    Ok(report)
}

async fn provisional_streaming() -> Result<ScenarioReport> {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("provisional"),
        MockStreamEvent::error("reject"),
    ]]);
    let mut runtime = test_runtime()?;
    let agent = runtime.spawn_agent(
        model
            .into_ecs_agent_builder()
            .streaming(StreamingMode::Streaming)
            .build(),
    )?;
    let handle = runtime.start_run(agent, "prompt")?;
    let result = runtime.drive_to_terminal(handle).await?;
    let observed = result.events.iter().any(|event| {
        let RunEvent::Provisional { delta, .. } = event else {
            return false;
        };
        matches!(delta.as_ref(), ProvisionalDelta::Text(text) if text == "provisional")
    });

    let retry_model = MockCompletionModel::from_stream_turns([
        [
            MockStreamEvent::text("not-json"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
        [
            MockStreamEvent::text(r#"{"answer":"accepted"}"#),
            MockStreamEvent::final_response_with_total_tokens(1),
        ],
    ]);
    let mut retry_runtime = test_runtime()?;
    let retry_agent = retry_runtime.spawn_agent(
        retry_model
            .into_ecs_agent_builder()
            .streaming(StreamingMode::Streaming)
            .max_model_calls(2)
            .structured_output::<Answer>(StructuredOutputPolicy {
                mode: OutputMode::Native,
                max_retries: 1,
                best_effort: false,
            })
            .build(),
    )?;
    let retried = retry_runtime
        .run_streaming::<MockResponse>(retry_agent, "retry")
        .await?
        .result;
    let retry_observed = retried.events.iter().any(|event| {
        matches!(
            event,
            RunEvent::Provisional { delta, .. }
                if matches!(delta.as_ref(), ProvisionalDelta::Text(text) if text == "not-json")
        )
    });

    let stop_model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("stop-provisional"),
        MockStreamEvent::tool_call("invalid", "missing", serde_json::json!({})),
        MockStreamEvent::final_response_with_total_tokens(1),
    ]]);
    let mut stop_runtime = test_runtime()?;
    let stop_agent = stop_runtime.spawn_agent(
        stop_model
            .into_ecs_agent_builder()
            .streaming(StreamingMode::Streaming)
            .invalid_tool_policy(InvalidToolPolicy::Stop)
            .build(),
    )?;
    let stop_handle = stop_runtime.start_run(stop_agent, "stop")?;
    let stopped = stop_runtime.drive_to_terminal(stop_handle).await?;
    let stop_observed = stopped.events.iter().any(|event| {
        matches!(
            event,
            RunEvent::Provisional { delta, .. }
                if matches!(delta.as_ref(), ProvisionalDelta::Text(text) if text == "stop-provisional")
        )
    });

    let cancel_model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("cancel-provisional"),
        MockStreamEvent::final_response_with_total_tokens(1),
    ]]);
    let mut cancel_runtime = test_runtime()?;
    let cancel_agent = cancel_runtime.spawn_agent(
        cancel_model
            .into_ecs_agent_builder()
            .streaming(StreamingMode::Streaming)
            .build(),
    )?;
    let mut cancel_stream =
        cancel_runtime.start_streaming::<MockResponse>(cancel_agent, "cancel")?;
    let mut cancel_observed = false;
    while let Some(event) = cancel_stream.next_event().await? {
        if matches!(
            event,
            rig_ecs::StreamingRunEvent::Runtime(event)
                if matches!(*event, RunEvent::Provisional { .. })
        ) {
            cancel_observed = true;
            break;
        }
    }
    let cancelled = cancel_stream.cancel()?;
    let mut report = report(ScenarioId::ProvisionalStreaming);
    evidence(&mut report, 0, observed)?;
    evidence(
        &mut report,
        1,
        retry_observed
            && !serde_json::to_string(&retried.transcript)?.contains("not-json")
            && retried.text.as_deref() == Some(r#"{"answer":"accepted"}"#),
    )?;
    evidence(
        &mut report,
        2,
        result.transcript == [Message::user("prompt")],
    )?;
    evidence(
        &mut report,
        3,
        stop_observed
            && matches!(stopped.terminal_reason, TerminalReason::Stopped)
            && stopped.transcript == [Message::user("stop")],
    )?;
    evidence(
        &mut report,
        4,
        cancel_observed
            && matches!(cancelled.terminal_reason, TerminalReason::Cancelled)
            && cancelled.transcript == [Message::user("cancel")],
    )?;
    Ok(report)
}

async fn tool_suppression() -> Result<ScenarioReport> {
    let skipped = run_invalid_policy(InvalidToolPolicy::Skip, "missing", false).await?;
    let invalid_peer_model = MockCompletionModel::new([
        MockTurn::from_contents([
            AssistantContent::tool_call("invalid-peer", "missing", serde_json::json!({})),
            AssistantContent::tool_call(
                "valid-peer",
                "counting_portable_tool",
                serde_json::json!({"value":"valid"}),
            ),
        ])?,
        MockTurn::text("continued"),
    ]);
    let invalid_peer_tool = CountingPortableTool::default();
    let mut invalid_peer_runtime = test_runtime()?;
    let invalid_peer_agent = invalid_peer_runtime.spawn_agent(
        invalid_peer_model
            .into_ecs_agent_builder()
            .invalid_tool_policy(InvalidToolPolicy::Skip)
            .max_model_calls(2)
            .build(),
    )?;
    invalid_peer_runtime.install_tool(invalid_peer_agent, invalid_peer_tool.clone())?;
    let invalid_peer_result = invalid_peer_runtime
        .run(invalid_peer_agent, "invalid peer")
        .await?;

    let schema = schemars::schema_for!(Answer);
    let output_name = synthetic_name(&schema);
    let model = MockCompletionModel::new([MockTurn::from_contents([
        AssistantContent::tool_call(
            "peer",
            "counting_portable_tool",
            serde_json::json!({"value":"must-not-run"}),
        ),
        AssistantContent::tool_call("output", output_name, serde_json::json!({"answer":"done"})),
    ])?]);
    let tool = CountingPortableTool::default();
    let mut runtime = test_runtime()?;
    let agent = runtime.spawn_agent(
        model
            .into_ecs_agent_builder()
            .structured_output::<Answer>(StructuredOutputPolicy {
                mode: OutputMode::Tool,
                max_retries: 0,
                best_effort: false,
            })
            .build(),
    )?;
    runtime.install_tool(agent, tool.clone())?;
    let result = runtime.run(agent, "finalize").await?;

    let stopped = run_invalid_policy(InvalidToolPolicy::Stop, "missing", false).await?;
    let cancellation_tool = CountingPortableTool::default();
    let cancellation_model = DelayedModel {
        inner: MockCompletionModel::new([MockTurn::tool_call(
            "cancelled",
            "counting_portable_tool",
            serde_json::json!({"value":"must-not-run"}),
        )]),
        delay: Duration::from_millis(100),
    };
    let mut cancellation_runtime = test_runtime()?;
    let cancellation_agent =
        cancellation_runtime.spawn_agent(cancellation_model.into_ecs_agent_builder().build())?;
    cancellation_runtime.install_tool(cancellation_agent, cancellation_tool.clone())?;
    let cancellation_handle = cancellation_runtime.start_run(cancellation_agent, "cancel tool")?;
    let _ = cancellation_runtime.step(cancellation_handle).await?;
    cancellation_runtime.cancel(cancellation_handle)?;
    let cancelled = cancellation_runtime
        .drive_to_terminal(cancellation_handle)
        .await?;

    let mut report = report(ScenarioId::ToolSuppression);
    evidence(
        &mut report,
        0,
        invalid_peer_tool.calls().len() == 1
            && invalid_peer_result.text.as_deref() == Some("continued")
            && invalid_peer_result.events.iter().any(|event| {
                matches!(event, RunEvent::ToolSuppressed { tool_call_id } if tool_call_id == "invalid-peer")
            }),
    )?;
    evidence(&mut report, 1, skipped.1 == 0)?;
    evidence(
        &mut report,
        2,
        tool.calls().is_empty()
            && result
                .events
                .iter()
                .any(|event| matches!(event, RunEvent::ToolSuppressed { tool_call_id } if tool_call_id == "peer")),
    )?;
    evidence(&mut report, 3, stopped.1 == 0)?;
    evidence(
        &mut report,
        4,
        matches!(cancelled.terminal_reason, TerminalReason::Cancelled)
            && cancellation_tool.calls().is_empty(),
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
    const NAME: &'static str = "concurrency_probe";
    type Args = ProbeArgs;
    type Output = serde_json::Value;
    type Error = Infallible;

    fn description(&self) -> String {
        "bounded concurrency probe".to_string()
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
        Ok(serde_json::json!({"value": arguments.value}))
    }
}

async fn concurrency() -> Result<ScenarioReport> {
    let model = MockCompletionModel::new([
        MockTurn::from_contents([
            AssistantContent::tool_call(
                "slow",
                "concurrency_probe",
                serde_json::json!({"value":"slow","delay_ms":20}),
            ),
            AssistantContent::tool_call(
                "fast",
                "concurrency_probe",
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
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        max_tool_calls: 2,
        effect_timeout: Duration::from_secs(2),
        ..RuntimeConfig::default()
    })?;
    let agent = runtime.spawn_agent(model.into_ecs_agent_builder().build())?;
    runtime.install_tool(agent, probe.clone())?;
    let result = runtime.run(agent, "parallel").await?;
    let ids = match result.transcript.get(2) {
        Some(Message::User { content }) => content
            .iter()
            .filter_map(|content| match content {
                rig_core::message::UserContent::ToolResult(result) => Some(result.id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };
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

async fn stale_result_handling() -> Result<ScenarioReport> {
    let model = DelayedModel {
        inner: MockCompletionModel::text("done"),
        delay: Duration::from_millis(20),
    };
    let mut runtime = test_runtime()?;
    let tenant_id = TenantId::new();
    let model_id = runtime.register_model(tenant_id, model);
    let agent = runtime.spawn_agent_spec(AgentSpec::new(model_id, tenant_id))?;
    let handle = runtime.start_run(agent, "hostile")?;
    let _ = runtime.step(handle).await?;
    let header = runtime
        .active_effect_headers(handle)?
        .first()
        .copied()
        .context("active header")?;
    let mut wrong_runtime = header;
    wrong_runtime.runtime_id = rig_ecs::RuntimeId::new();
    let mut wrong_tenant = header;
    wrong_tenant.tenant_id = TenantId::new();
    let mut wrong_generation = header;
    wrong_generation.generation = header.generation.next();
    let mut wrong_correlation = header;
    wrong_correlation.correlation_id = rig_ecs::CorrelationId::new();
    let mut wrong_authorization = header;
    wrong_authorization.capability_id = Some(rig_ecs::CapabilityId::new());
    for hostile in [
        wrong_runtime,
        wrong_tenant,
        wrong_generation,
        wrong_correlation,
        wrong_authorization,
    ] {
        runtime.ingest(EffectIngress::Delta {
            header: hostile,
            sequence: 0,
            delta: ProvisionalDelta::Text("hostile".to_string()),
        })?;
    }
    runtime.ingest(EffectIngress::Delta {
        header,
        sequence: 0,
        delta: ProvisionalDelta::Text("once".to_string()),
    })?;
    runtime.ingest(EffectIngress::Delta {
        header,
        sequence: 0,
        delta: ProvisionalDelta::Text("once".to_string()),
    })?;
    runtime.ingest(EffectIngress::Completion(EffectCompletion::Memory {
        header,
        result: Ok(MemoryEffectOutput::Appended),
    }))?;
    let result = runtime.drive_to_terminal(handle).await?;
    let reasons = runtime
        .effect_rejections()
        .iter()
        .map(|rejection| rejection.reason)
        .collect::<BTreeSet<_>>();
    let wrong_payload_count = runtime
        .effect_rejections()
        .iter()
        .filter(|rejection| {
            matches!(
                rejection.reason,
                EffectRejectionReason::WrongAuthorization | EffectRejectionReason::WrongPayload
            )
        })
        .count();

    let cancel_model = MockCompletionModel::text("late");
    let mut cancel_runtime = test_runtime()?;
    let cancel_agent = cancel_runtime.spawn_agent(cancel_model.into_ecs_agent_builder().build())?;
    let cancel_handle = cancel_runtime.start_run(cancel_agent, "cancel")?;
    let _ = cancel_runtime.step(cancel_handle).await?;
    let cancel_header = cancel_runtime
        .active_effect_headers(cancel_handle)?
        .first()
        .copied()
        .context("cancel header")?;
    cancel_runtime.cancel(cancel_handle)?;
    cancel_runtime.ingest(EffectIngress::Delta {
        header: cancel_header,
        sequence: 0,
        delta: ProvisionalDelta::Text("late".to_string()),
    })?;
    let canceled = cancel_runtime.drive_to_terminal(cancel_handle).await?;

    let superseded_model = DelayedModel {
        inner: MockCompletionModel::new([MockTurn::text("provider-late")]),
        delay: Duration::from_millis(200),
    };
    let mut superseded_runtime = test_runtime()?;
    let superseded_agent = superseded_runtime.spawn_agent(
        superseded_model
            .into_ecs_agent_builder()
            .response_retry_policy(ResponseRetryPolicy::RejectEmpty { max_retries: 1 })
            .max_model_calls(2)
            .build(),
    )?;
    let superseded_handle = superseded_runtime.start_run(superseded_agent, "superseded")?;
    let _ = superseded_runtime.step(superseded_handle).await?;
    let first_header = superseded_runtime
        .active_effect_headers(superseded_handle)?
        .first()
        .copied()
        .context("first retry operation")?;
    superseded_runtime.ingest(EffectIngress::Completion(EffectCompletion::Model {
        header: first_header,
        result: Ok(ModelEffectOutput {
            choice: vec![AssistantContent::text("")],
            usage: Usage::new(),
            message_id: None,
            raw_final: None,
        }),
    }))?;
    let _ = superseded_runtime.step(superseded_handle).await?;
    let second_header = superseded_runtime
        .active_effect_headers(superseded_handle)?
        .into_iter()
        .find(|header| header.operation_id != first_header.operation_id)
        .context("retried model operation")?;
    superseded_runtime.ingest(EffectIngress::Completion(EffectCompletion::Model {
        header: first_header,
        result: Ok(ModelEffectOutput {
            choice: vec![AssistantContent::text("superseded content")],
            usage: Usage::new(),
            message_id: None,
            raw_final: None,
        }),
    }))?;
    superseded_runtime.ingest(EffectIngress::Completion(EffectCompletion::Model {
        header: second_header,
        result: Ok(ModelEffectOutput {
            choice: vec![AssistantContent::text("accepted retry")],
            usage: Usage::new(),
            message_id: None,
            raw_final: None,
        }),
    }))?;
    let superseded = superseded_runtime
        .drive_to_terminal(superseded_handle)
        .await?;
    let superseded_rejected = superseded_runtime
        .effect_rejections()
        .iter()
        .any(|rejection| {
            rejection.header.operation_id == first_header.operation_id
                && rejection.reason == EffectRejectionReason::Duplicate
        });
    let all_boundaries_rejected = [
        EffectRejectionReason::ForeignRuntime,
        EffectRejectionReason::WrongTenant,
        EffectRejectionReason::WrongGeneration,
        EffectRejectionReason::WrongCorrelation,
        EffectRejectionReason::WrongAuthorization,
    ]
    .iter()
    .all(|reason| reasons.contains(reason));
    let mut report = report(ScenarioId::StaleResultHandling);
    evidence(
        &mut report,
        0,
        reasons.contains(&EffectRejectionReason::Duplicate),
    )?;
    evidence(
        &mut report,
        1,
        reasons.contains(&EffectRejectionReason::WrongCorrelation)
            && wrong_payload_count >= 2
            && result.text.as_deref() == Some("done"),
    )?;
    evidence(
        &mut report,
        2,
        superseded_rejected
            && superseded.text.as_deref() == Some("accepted retry")
            && !serde_json::to_string(&superseded.transcript)?.contains("superseded content"),
    )?;
    evidence(
        &mut report,
        3,
        matches!(canceled.terminal_reason, TerminalReason::Cancelled)
            && canceled.transcript == [Message::user("cancel")],
    )?;
    evidence(&mut report, 4, all_boundaries_rejected)?;
    Ok(report)
}

#[tokio::test]
async fn every_shared_scenario_passes_the_ecs_runtime() -> Result<()> {
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
