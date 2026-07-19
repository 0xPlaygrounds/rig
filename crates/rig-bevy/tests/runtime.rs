// This integration harness deliberately panics inside a provider double to
// verify task-failure containment and mutates JSON snapshots by stable keys to
// exercise malformed persistence input. Those operations are confined to test
// construction rather than production runtime paths.
#![allow(clippy::indexing_slicing, clippy::panic, clippy::panic_in_result_fn)]

use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result};
use rig_bevy::{
    AgentSpec, BevyModelExt, BindingIdentity, CapabilityKind, CapabilityNode, ContentVisibility,
    CorrelationId, EffectCompletion, EffectHeader, EffectIngress, EffectRejectionReason,
    Generation, GrantNode, HostedRuntime, InvalidToolPolicy, LocalRunResult, LocalRuntime,
    ModelEffectError, OperationId, OutputMode, ProtectedSnapshot, ProvisionalDelta, RebindRegistry,
    ResponseRetryPolicy, RunEvent, RunHandle, RunId, RunPhase, RunStepStatus, RuntimeConfig,
    RuntimeError, SnapshotContentPolicy, SnapshotError, SnapshotProtector, StreamingMode,
    StreamingRunEvent, StructuredOutputPolicy, TenantId, TerminalReason, ToolEffectOutput,
};
use rig_core::{
    OneOrMany,
    completion::{
        AssistantContent, CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
        Message, Usage,
    },
    memory::{ConversationMemory, MemoryError},
    message::{ToolCall, ToolChoice, ToolFunction, ToolResult, ToolResultContent, UserContent},
    streaming::StreamingCompletionResponse,
    test_utils::{CountingMemory, MockCompletionModel, MockResponse, MockStreamEvent, MockTurn},
    tool::{PortableDynamicTool, PortableTool, ToolOutput},
    vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex, request::Filter},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};
use rig_runtime_conformance::{
    CountingPortableTool, PortableEmbeddingFixture, portable_dynamic_fixture,
    portable_fixture_output,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

#[derive(Clone)]
struct DifferentMock(MockCompletionModel);

impl CompletionModel for DifferentMock {
    type Response = MockResponse;
    type StreamingResponse = MockResponse;
    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>) -> Self {
        Self(MockCompletionModel::default())
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        self.0.completion(request).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        self.0.stream(request).await
    }
}

#[derive(Clone)]
struct SlowMock {
    inner: MockCompletionModel,
    delay: Duration,
}

#[derive(Clone)]
struct PanicModel;

impl CompletionModel for PanicModel {
    type Response = MockResponse;
    type StreamingResponse = MockResponse;
    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>) -> Self {
        Self
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        panic!("test model panic")
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        panic!("test model panic")
    }
}

impl CompletionModel for SlowMock {
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

#[derive(Clone)]
struct GateTool {
    permits: Arc<Semaphore>,
    calls: Arc<AtomicUsize>,
}

#[derive(Deserialize)]
struct GateArgs {
    value: String,
}

impl PortableTool for GateTool {
    const NAME: &'static str = "gate_tool";
    type Args = GateArgs;
    type Output = String;
    type Error = Infallible;

    fn description(&self) -> String {
        "test-only gated portable tool".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "string"}}
        })
    }

    async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Ok(permit) = Arc::clone(&self.permits).acquire_owned().await {
            permit.forget();
        }
        Ok(arguments.value)
    }
}

#[derive(Clone, Copy)]
struct OrderedDelayTool;

#[derive(Deserialize)]
struct OrderedDelayArgs {
    value: String,
    delay_ms: u64,
}

impl PortableTool for OrderedDelayTool {
    const NAME: &'static str = "ordered_delay_tool";
    type Args = OrderedDelayArgs;
    type Output = String;
    type Error = Infallible;

    fn description(&self) -> String {
        "return a value after a test-controlled delay".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["value", "delay_ms"],
            "properties": {
                "value": {"type": "string"},
                "delay_ms": {"type": "integer"}
            }
        })
    }

    async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
        tokio::time::sleep(Duration::from_millis(arguments.delay_ms)).await;
        Ok(arguments.value)
    }
}

fn runtime() -> Result<LocalRuntime> {
    Ok(LocalRuntime::with_config(RuntimeConfig {
        effect_timeout: Duration::from_secs(2),
        terminal_retention_ticks: 2,
        ..RuntimeConfig::default()
    })?)
}

#[derive(Clone, Default)]
struct FixtureVectorStore {
    calls: Arc<AtomicUsize>,
}

impl VectorStoreIndex for FixtureVectorStore {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T>(
        &self,
        _request: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>
    where
        T: for<'a> Deserialize<'a> + WasmCompatSend,
    {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let document = serde_json::from_value(serde_json::json!({
            "body": "retrieved through a correlated store effect"
        }))?;
        Ok(vec![(0.99, "doc-1".to_string(), document)])
    }

    async fn top_n_ids(
        &self,
        _request: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(vec![(0.99, "doc-1".to_string())])
    }
}

#[tokio::test]
async fn local_blocking_returns_canonical_text_and_concrete_raw_final() -> Result<()> {
    let model = MockCompletionModel::new([MockTurn::text("hello").with_usage(Usage {
        total_tokens: 3,
        ..Usage::new()
    })]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .tenant(TenantId::new())
            .build(),
    )?;

    let result = runtime.run_blocking(agent, "hi").await?;

    assert_eq!(result.text.as_deref(), Some("hello"));
    assert_eq!(result.usage.total_tokens, 3);
    assert!(result.raw_final::<MockResponse>().is_some());
    assert_eq!(model.request_count(), 1);
    Ok(())
}

#[tokio::test]
async fn bevy_requests_preserve_effective_preamble_for_opt_in_telemetry() -> Result<()> {
    for record_content in [false, true] {
        let model = MockCompletionModel::text("done");
        let mut runtime = runtime()?;
        let agent = runtime.spawn_agent(
            model
                .clone()
                .into_bevy_agent_builder()
                .preamble("effective system policy")
                .record_telemetry_content(record_content)
                .build(),
        )?;

        let _ = runtime.run(agent, "prompt").await?;
        let request = model.requests().pop().context("recorded model request")?;
        assert_eq!(request.preamble.as_deref(), Some("effective system policy"));
        assert_eq!(request.record_telemetry_content, record_content);
    }
    Ok(())
}

#[tokio::test]
async fn bevy_agent_request_controls_reach_the_provider_request() -> Result<()> {
    let model = MockCompletionModel::text("done");
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .temperature(0.35)
            .max_tokens(321)
            .build(),
    )?;

    let _ = runtime.run(agent, "request controls").await?;
    let request = model.requests().pop().context("recorded model request")?;

    assert_eq!(request.temperature, Some(0.35));
    assert_eq!(request.max_tokens, Some(321));
    Ok(())
}

#[tokio::test]
async fn zero_model_budget_rejects_before_provider_dispatch() -> Result<()> {
    let model = MockCompletionModel::text("must not run");
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .max_model_calls(0)
            .build(),
    )?;

    let error = runtime
        .run(agent, "hi")
        .await
        .err()
        .context("zero model-call budget must fail")?;

    assert!(matches!(error, RuntimeError::ModelCallBudgetExhausted));
    assert_eq!(model.request_count(), 0);
    Ok(())
}

#[tokio::test]
async fn portable_tool_effect_commits_call_and_result_before_continuation() -> Result<()> {
    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "call-1",
            "counting_portable_tool",
            serde_json::json!({"value": "owned"}),
        ),
        MockTurn::text("done"),
    ]);
    let tool = CountingPortableTool::default();
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().build())?;
    runtime.install_tool(agent, tool.clone())?;

    let result = runtime.run(agent, "use tool").await?;

    assert_eq!(tool.calls(), ["owned"]);
    assert_eq!(result.text.as_deref(), Some("done"));
    assert_eq!(result.model_calls.len(), 2);
    assert!(matches!(
        result.transcript.get(1),
        Some(Message::Assistant { content, .. })
            if matches!(content.first_ref(), AssistantContent::ToolCall(call) if call.id == "call-1")
    ));
    assert!(matches!(
        result.transcript.get(2),
        Some(Message::User { .. })
    ));
    Ok(())
}

#[tokio::test]
async fn portable_embedding_and_dynamic_tools_preserve_rich_outputs_in_bevy() -> Result<()> {
    let embedding_name = <PortableEmbeddingFixture as PortableTool>::NAME;
    let embedding_model = MockCompletionModel::new([
        MockTurn::tool_call(
            "embedding-success",
            embedding_name,
            serde_json::json!({"value": "ok"}),
        ),
        MockTurn::tool_call(
            "embedding-failure",
            embedding_name,
            serde_json::json!({"value": "ignored", "fail": true}),
        ),
        MockTurn::text("embedding done"),
    ]);
    let embedding_tool = PortableEmbeddingFixture::new("shared");
    let portable_schema = rig_core::embeddings::ToolSchema::try_from(&embedding_tool)?;
    let mut embedding_runtime = runtime()?;
    let embedding_agent =
        embedding_runtime.spawn_agent(embedding_model.clone().into_bevy_agent_builder().build())?;
    embedding_runtime.install_tool(embedding_agent, embedding_tool)?;

    let embedding_result = embedding_runtime
        .run(embedding_agent, "use portable embedding tool twice")
        .await?;
    let first_request = embedding_model
        .requests()
        .into_iter()
        .next()
        .context("embedding model request")?;
    let definition = first_request
        .tools
        .iter()
        .find(|definition| definition.name == portable_schema.name)
        .context("portable embedding definition")?;
    assert_eq!(definition.description, "shared portable embedding fixture");
    assert_eq!(
        definition.parameters,
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "fail": {"type": "boolean"}
            },
            "required": ["value"]
        })
    );

    let embedding_outputs = embedding_result
        .transcript
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => content.iter().find_map(|content| match content {
                UserContent::ToolResult(result) => {
                    Some(ToolOutput::content(result.content.clone()))
                }
                _ => None,
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        embedding_outputs,
        [
            portable_fixture_output("shared:ok"),
            portable_fixture_output("portable failure")
        ]
    );

    let dynamic = portable_dynamic_fixture();
    let dynamic_model = MockCompletionModel::new([
        MockTurn::tool_call(
            "dynamic-success",
            "portable_runtime_name",
            serde_json::json!({"value": "ok"}),
        ),
        MockTurn::tool_call(
            "dynamic-failure",
            "portable_runtime_name",
            serde_json::json!({"value": "ignored", "fail": true}),
        ),
        MockTurn::text("dynamic done"),
    ]);
    let mut dynamic_runtime = runtime()?;
    let dynamic_agent =
        dynamic_runtime.spawn_agent(dynamic_model.into_bevy_agent_builder().build())?;
    dynamic_runtime.install_dynamic_tool(dynamic_agent, dynamic)?;
    let dynamic_result = dynamic_runtime
        .run(dynamic_agent, "use dynamic tool")
        .await?;
    let dynamic_outputs = dynamic_result
        .transcript
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => content.iter().find_map(|content| match content {
                UserContent::ToolResult(result) => {
                    Some(ToolOutput::content(result.content.clone()))
                }
                _ => None,
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        dynamic_outputs,
        [
            portable_fixture_output("dynamic:ok"),
            portable_fixture_output("portable dynamic failure")
        ]
    );
    Ok(())
}

#[derive(Debug, PartialEq)]
struct NormalizedRun {
    text: Option<String>,
    transcript: serde_json::Value,
    usage: Usage,
    accepted_calls: Vec<bool>,
    events: Vec<String>,
}

fn normalize_run(result: &LocalRunResult) -> Result<NormalizedRun> {
    let events = result
        .events
        .iter()
        .map(|event| match event {
            RunEvent::ModelDispatched(_) => "model:dispatched".to_string(),
            RunEvent::Provisional { delta, .. } => format!(
                "model:delta:{}",
                match delta.as_ref() {
                    ProvisionalDelta::Text(_) => "text",
                    ProvisionalDelta::ToolCall(_) => "tool-call",
                    ProvisionalDelta::ToolCallDelta { .. } => "tool-call-delta",
                    ProvisionalDelta::Reasoning(_) => "reasoning",
                    ProvisionalDelta::ReasoningDelta { .. } => "reasoning-delta",
                    ProvisionalDelta::Unknown(_) => "unknown",
                    _ => "future-delta",
                }
            ),
            RunEvent::ProviderFinal { .. } => "model:provider-final".to_string(),
            RunEvent::ToolDispatched { tool_call_id, .. } => {
                format!("tool:dispatched:{tool_call_id}")
            }
            RunEvent::ToolCommitted { tool_call_id, .. } => {
                format!("tool:committed:{tool_call_id}")
            }
            RunEvent::ResponseRetried => "response:retried".to_string(),
            RunEvent::ToolSuppressed { tool_call_id } => {
                format!("tool:suppressed:{tool_call_id}")
            }
            RunEvent::Terminal(reason) => format!("terminal:{reason:?}"),
            RunEvent::EffectRejected(reason) => format!("effect:rejected:{reason:?}"),
            _ => "runtime:future-event".to_string(),
        })
        .collect();
    Ok(NormalizedRun {
        text: result.text.clone(),
        transcript: serde_json::to_value(&result.transcript)?,
        usage: result.usage,
        accepted_calls: result
            .model_calls
            .iter()
            .map(|call| call.accepted)
            .collect(),
        events,
    })
}

async fn run_with_agent_and_run_insertion_order(
    reverse: bool,
) -> Result<(NormalizedRun, NormalizedRun)> {
    let tool_turn = MockTurn::from_contents([
        AssistantContent::tool_call(
            "slow-call",
            OrderedDelayTool::NAME,
            serde_json::json!({"value": "slow", "delay_ms": 30}),
        ),
        AssistantContent::tool_call(
            "fast-call",
            OrderedDelayTool::NAME,
            serde_json::json!({"value": "fast", "delay_ms": 1}),
        ),
    ])?;
    let tool_model = MockCompletionModel::new([tool_turn, MockTurn::text("tool batch done")]);
    let plain_model = MockCompletionModel::text("plain done");
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        max_effects: 4,
        max_model_calls: 2,
        max_tool_calls: 2,
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    })?;

    let (tool_agent, plain_agent) = if reverse {
        let plain = runtime.spawn_agent(plain_model.into_bevy_agent_builder().build())?;
        let tool = runtime.spawn_agent(
            tool_model
                .into_bevy_agent_builder()
                .max_model_calls(2)
                .build(),
        )?;
        (tool, plain)
    } else {
        let tool = runtime.spawn_agent(
            tool_model
                .into_bevy_agent_builder()
                .max_model_calls(2)
                .build(),
        )?;
        let plain = runtime.spawn_agent(plain_model.into_bevy_agent_builder().build())?;
        (tool, plain)
    };
    runtime.install_tool(tool_agent, OrderedDelayTool)?;

    let (tool_handle, plain_handle) = if reverse {
        let plain = runtime.start_run(plain_agent, "plain prompt")?;
        let tool = runtime.start_run(tool_agent, "tool prompt")?;
        (tool, plain)
    } else {
        let tool = runtime.start_run(tool_agent, "tool prompt")?;
        let plain = runtime.start_run(plain_agent, "plain prompt")?;
        (tool, plain)
    };

    let (tool_result, plain_result) = if reverse {
        let plain = runtime.drive_to_terminal(plain_handle).await?;
        let tool = runtime.drive_to_terminal(tool_handle).await?;
        (tool, plain)
    } else {
        let tool = runtime.drive_to_terminal(tool_handle).await?;
        let plain = runtime.drive_to_terminal(plain_handle).await?;
        (tool, plain)
    };
    Ok((normalize_run(&tool_result)?, normalize_run(&plain_result)?))
}

#[tokio::test]
async fn shuffled_entity_insertion_preserves_deterministic_run_results() -> Result<()> {
    let forward = run_with_agent_and_run_insertion_order(false).await?;
    let reversed = run_with_agent_and_run_insertion_order(true).await?;

    assert_eq!(forward, reversed);
    assert_eq!(forward.0.text.as_deref(), Some("tool batch done"));
    let transcript = forward.0.transcript.to_string();
    assert!(transcript.find("slow-call") < transcript.find("fast-call"));
    assert!(transcript.find("slow") < transcript.find("fast"));
    Ok(())
}

#[tokio::test]
async fn vector_store_adapter_uses_store_topology_and_correlated_effects() -> Result<()> {
    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "store-call",
            "search_vector_store",
            serde_json::json!({"query": "runtime split", "samples": 1}),
        ),
        MockTurn::text("grounded answer"),
    ]);
    let store = FixtureVectorStore::default();
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().build())?;
    let grant = runtime.install_vector_store(agent, store.clone())?;

    let result = runtime.run(agent, "retrieve context").await?;

    assert_eq!(store.calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.text.as_deref(), Some("grounded answer"));
    assert_eq!(
        runtime.capability_kind(grant.capability_id)?,
        CapabilityKind::Store
    );
    assert!(matches!(
        result.transcript.get(2),
        Some(Message::User { .. })
    ));
    Ok(())
}

#[tokio::test]
async fn memory_loads_before_model_and_appends_only_successful_commits() -> Result<()> {
    let model = MockCompletionModel::text("remembered");
    let memory = CountingMemory::default();
    let mut runtime = runtime()?;
    let tenant_id = TenantId::new();
    let memory_id = runtime.register_memory(tenant_id, memory.clone());
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .tenant(tenant_id)
            .memory(memory_id, "conversation")
            .build(),
    )?;

    let result = runtime.run(agent, "hello").await?;

    assert_eq!(memory.load_count(), 1);
    assert_eq!(memory.append_count(), 1);
    let first_request = model
        .requests()
        .into_iter()
        .next()
        .context("memory-backed model request")?;
    assert_eq!(first_request.chat_history.len(), 1);
    assert_eq!(result.transcript.len(), 2);
    Ok(())
}

#[tokio::test]
async fn later_stream_error_keeps_deltas_provisional_and_suppresses_raw_final() -> Result<()> {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("early"),
        MockStreamEvent::final_response_with_total_tokens(2),
        MockStreamEvent::error("late failure"),
    ]]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .into_bevy_agent_builder()
            .streaming(StreamingMode::Streaming)
            .build(),
    )?;
    let handle = runtime.start_run(agent, "stream")?;
    // The public stepping path preserves the failed terminal result for inspection.
    let outcome = runtime.drive_to_terminal(handle).await?;

    assert!(matches!(
        outcome.terminal_reason,
        TerminalReason::Failed { .. }
    ));
    assert!(outcome.raw_final::<MockResponse>().is_none());
    assert!(
        outcome
            .events
            .iter()
            .any(|event| matches!(event, rig_bevy::RunEvent::Provisional { .. }))
    );
    assert_eq!(outcome.transcript, [Message::user("stream")]);
    Ok(())
}

#[tokio::test]
async fn live_stream_preserves_identical_chunks_before_typed_final() -> Result<()> {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("same"),
        MockStreamEvent::text("same"),
        MockStreamEvent::final_response_with_total_tokens(2),
    ]]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().build())?;
    let mut stream = runtime.start_streaming::<MockResponse>(agent, "stream")?;
    let mut sequence = Vec::new();

    while let Some(event) = stream.next_event().await? {
        match event {
            StreamingRunEvent::Runtime(event) => match *event {
                RunEvent::Provisional { delta, .. } => {
                    if matches!(delta.as_ref(), ProvisionalDelta::Text(text) if text == "same") {
                        sequence.push("delta");
                    }
                }
                RunEvent::Terminal(_) => sequence.push("terminal"),
                _ => {}
            },
            StreamingRunEvent::ProviderFinal { .. } => sequence.push("provider-final"),
            _ => {}
        }
    }
    let result = stream.finish()?;

    assert_eq!(sequence, ["delta", "delta", "provider-final", "terminal"]);
    assert_eq!(result.text.as_deref(), Some("samesame"));
    Ok(())
}

#[tokio::test]
async fn legal_stream_without_provider_final_omits_typed_final_event() -> Result<()> {
    let model = MockCompletionModel::from_stream_turns([[MockStreamEvent::text("text only")]]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().build())?;

    let streamed = runtime
        .run_streaming::<MockResponse>(agent, "no raw final")
        .await?;

    assert_eq!(streamed.result.text.as_deref(), Some("text only"));
    assert!(streamed.result.raw_final::<MockResponse>().is_none());
    assert!(
        !streamed
            .events
            .iter()
            .any(|event| matches!(event, StreamingRunEvent::ProviderFinal { .. }))
    );
    Ok(())
}

#[tokio::test]
async fn streaming_convenience_collector_enforces_event_capacity() -> Result<()> {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("one"),
        MockStreamEvent::text("two"),
        MockStreamEvent::text("three"),
        MockStreamEvent::text("four"),
        MockStreamEvent::final_response_with_total_tokens(4),
    ]]);
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        ingress_capacity: 1,
        event_capacity: 3,
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    })?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().build())?;

    let error = runtime
        .run_streaming::<MockResponse>(agent, "bounded collection")
        .await
        .err()
        .context("collector must reject unbounded accumulation")?;

    assert!(matches!(
        error,
        RuntimeError::EventCollectionLimit { capacity: 3 }
    ));
    Ok(())
}

#[tokio::test]
async fn long_stream_resets_consecutive_pass_bound_for_local_and_hosted_drivers() -> Result<()> {
    fn stream_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("a"),
            MockStreamEvent::text("b"),
            MockStreamEvent::text("c"),
            MockStreamEvent::text("d"),
            MockStreamEvent::text("e"),
            MockStreamEvent::text("f"),
            MockStreamEvent::final_response_with_total_tokens(6),
        ]])
    }

    let config = RuntimeConfig {
        ingress_capacity: 1,
        max_schedule_passes: 2,
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    };
    let mut local = LocalRuntime::with_config(config.clone())?;
    let local_agent = local.spawn_agent(
        stream_model()
            .into_bevy_agent_builder()
            .streaming(StreamingMode::Streaming)
            .build(),
    )?;
    let local_result = local.run(local_agent, "local stream").await?;
    assert_eq!(local_result.text.as_deref(), Some("abcdef"));

    let mut hosted_local = LocalRuntime::with_config(config)?;
    let hosted_agent = hosted_local.spawn_agent(
        stream_model()
            .into_bevy_agent_builder()
            .streaming(StreamingMode::Streaming)
            .build(),
    )?;
    let hosted = HostedRuntime::new(hosted_local);
    let handle = hosted.start_run(hosted_agent, "hosted stream").await?;
    let hosted_result = hosted.drive_to_terminal(handle).await?;
    assert_eq!(hosted_result.text.as_deref(), Some("abcdef"));
    Ok(())
}

#[tokio::test]
async fn dropping_live_stream_cancels_before_provider_dispatch() -> Result<()> {
    let inner = MockCompletionModel::text("must not run");
    let model = SlowMock {
        inner: inner.clone(),
        delay: Duration::from_secs(1),
    };
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().build())?;
    let stream = runtime.start_streaming::<MockResponse>(agent, "drop immediately")?;
    let handle = stream.handle();

    drop(stream);

    let result = runtime.finish_run(handle)?;
    assert!(matches!(result.terminal_reason, TerminalReason::Cancelled));
    assert_eq!(result.transcript, [Message::user("drop immediately")]);
    assert!(runtime.active_effect_headers(handle)?.is_empty());
    assert_eq!(inner.request_count(), 0);
    Ok(())
}

#[tokio::test]
async fn dropping_live_stream_after_provisional_delta_prevents_commit() -> Result<()> {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("provisional"),
        MockStreamEvent::final_response_with_total_tokens(1),
    ]]);
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        ingress_capacity: 1,
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    })?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().build())?;
    let mut stream = runtime.start_streaming::<MockResponse>(agent, "drop after delta")?;
    let handle = stream.handle();

    loop {
        let event = stream
            .next_event()
            .await?
            .context("stream must publish a provisional delta")?;
        if matches!(
            event,
            StreamingRunEvent::Runtime(event)
                if matches!(*event, RunEvent::Provisional { .. })
        ) {
            break;
        }
    }
    drop(stream);

    let result = runtime.finish_run(handle)?;
    assert!(matches!(result.terminal_reason, TerminalReason::Cancelled));
    assert_eq!(result.transcript, [Message::user("drop after delta")]);
    assert!(result.raw_final::<MockResponse>().is_none());
    assert!(runtime.active_effect_headers(handle)?.is_empty());
    Ok(())
}

#[tokio::test]
async fn typed_stream_reports_provider_final_type_mismatch() -> Result<()> {
    let model = MockCompletionModel::from_stream_turns([[
        MockStreamEvent::text("done"),
        MockStreamEvent::final_response_with_total_tokens(1),
    ]]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().build())?;

    let error = runtime
        .run_streaming::<String>(agent, "stream")
        .await
        .err()
        .context("wrong provider-final type must be explicit")?;

    assert!(matches!(error, RuntimeError::RawFinalTypeMismatch { .. }));
    Ok(())
}

#[tokio::test]
async fn provider_usage_overflow_fails_without_panicking_or_wrapping() -> Result<()> {
    let model = MockCompletionModel::new([
        MockTurn::text("").with_usage(Usage {
            total_tokens: u64::MAX,
            ..Usage::new()
        }),
        MockTurn::text("would overflow").with_usage(Usage {
            total_tokens: 1,
            ..Usage::new()
        }),
    ]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .into_bevy_agent_builder()
            .response_retry_policy(ResponseRetryPolicy::RejectEmpty { max_retries: 1 })
            .build(),
    )?;
    let handle = runtime.start_run(agent, "usage")?;

    let result = runtime.drive_to_terminal(handle).await?;

    assert!(matches!(
        result.terminal_reason,
        TerminalReason::Failed { ref code } if code == "usage_overflow"
    ));
    assert_eq!(result.usage.total_tokens, u64::MAX);
    assert_eq!(result.model_calls.len(), 1);
    Ok(())
}

#[tokio::test]
async fn wrong_tool_payload_does_not_consume_the_legitimate_completion() -> Result<()> {
    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "real-call",
            "gate_tool",
            serde_json::json!({"value":"real"}),
        ),
        MockTurn::text("done"),
    ]);
    let permits = Arc::new(Semaphore::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let tool = GateTool {
        permits: Arc::clone(&permits),
        calls: Arc::clone(&calls),
    };
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().build())?;
    runtime.install_tool(agent, tool)?;
    let handle = runtime.start_run(agent, "tool")?;

    let _ = runtime.step(handle).await?;
    runtime.wait_for_effect().await?;
    let _ = runtime.step(handle).await?;
    let header = runtime
        .active_effect_headers(handle)?
        .into_iter()
        .next()
        .context("tool operation must remain active")?;
    runtime.ingest(EffectIngress::Completion(EffectCompletion::Tool {
        header,
        tool_call_id: "forged-call".to_string(),
        order: 0,
        result: ToolEffectOutput::Success(ToolOutput::text("forged")),
    }))?;
    let _ = runtime.step(handle).await?;

    assert_eq!(runtime.active_effect_headers(handle)?, [header]);
    assert!(runtime.effect_rejections().iter().any(|rejection| {
        rejection.header.operation_id == header.operation_id
            && rejection.reason == EffectRejectionReason::WrongPayload
    }));

    permits.add_permits(1);
    let result = runtime.drive_to_terminal(handle).await?;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.text.as_deref(), Some("done"));
    assert!(!serde_json::to_string(&result.transcript)?.contains("forged"));
    Ok(())
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
struct StructuredAnswer {
    answer: String,
}

fn assert_no_orphan_tool_use(messages: &[Message]) {
    let answered = messages
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
        .collect::<std::collections::BTreeSet<_>>();
    for content in messages.iter().filter_map(|message| match message {
        Message::Assistant { content, .. } => Some(content),
        _ => None,
    }) {
        for item in content.iter() {
            if let AssistantContent::ToolCall(call) = item {
                assert!(
                    answered.contains(call.id.as_str()),
                    "assistant tool call has no matching result"
                );
            }
        }
    }
}

fn synthetic_output_tool_name<T>(real_tool_names: &[&str]) -> String
where
    T: schemars::JsonSchema,
{
    let schema = schemars::schema_for!(T);
    let mut hasher = Sha256::new();
    hasher.update(schema.as_value().to_string().as_bytes());
    let prefix = hasher
        .finalize()
        .iter()
        .take(4)
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let mut candidate = format!("__rig_output_{prefix}");
    while real_tool_names.contains(&candidate.as_str()) {
        candidate.push('_');
    }
    candidate
}

#[tokio::test]
async fn tool_choice_none_degrades_tool_output_mode_to_native() -> Result<()> {
    let model = MockCompletionModel::text(r#"{"answer":"native"}"#);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .tool_choice(ToolChoice::None)
            .structured_output::<StructuredAnswer>(StructuredOutputPolicy {
                mode: OutputMode::Tool,
                max_retries: 0,
                best_effort: false,
            })
            .build(),
    )?;
    runtime.install_tool(agent, CountingPortableTool::default())?;

    let result = runtime.run(agent, "none").await?;
    let request = model.requests().pop().context("model request")?;

    assert!(request.output_schema.is_some());
    assert!(request.tools.is_empty());
    assert_eq!(
        result.structured_output,
        Some(serde_json::json!({"answer":"native"}))
    );
    Ok(())
}

#[tokio::test]
async fn specific_real_tool_degrades_tool_output_mode_to_native() -> Result<()> {
    let model = MockCompletionModel::text(r#"{"answer":"native"}"#);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["counting_portable_tool".to_string()],
            })
            .structured_output::<StructuredAnswer>(StructuredOutputPolicy {
                mode: OutputMode::Tool,
                max_retries: 0,
                best_effort: false,
            })
            .build(),
    )?;
    runtime.install_tool(agent, CountingPortableTool::default())?;

    let _ = runtime.run(agent, "specific real").await?;
    let request = model.requests().pop().context("model request")?;

    assert!(request.output_schema.is_some());
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, "counting_portable_tool");
    Ok(())
}

#[tokio::test]
async fn invalid_specific_and_unsatisfied_required_choices_fail_before_io() -> Result<()> {
    for choice in [
        ToolChoice::Specific {
            function_names: Vec::new(),
        },
        ToolChoice::Specific {
            function_names: vec!["unknown".to_string()],
        },
        ToolChoice::Required,
    ] {
        let model = MockCompletionModel::text("must not run");
        let mut runtime = runtime()?;
        let agent = runtime.spawn_agent(
            model
                .clone()
                .into_bevy_agent_builder()
                .tool_choice(choice)
                .build(),
        )?;
        let error = runtime
            .run(agent, "invalid choice")
            .await
            .err()
            .context("invalid choice must fail")?;
        assert!(matches!(
            error,
            RuntimeError::RunFailed { ref code, .. } if code == "invalid_tool_choice"
        ));
        assert_eq!(model.request_count(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn specific_synthetic_output_tool_stays_pinned_across_retry() -> Result<()> {
    let output_name = synthetic_output_tool_name::<StructuredAnswer>(&[]);
    let model = MockCompletionModel::new([
        MockTurn::tool_call("invalid", output_name.clone(), serde_json::json!({})),
        MockTurn::tool_call(
            "valid",
            output_name.clone(),
            serde_json::json!({"answer":"tool"}),
        ),
    ]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .tool_choice(ToolChoice::Specific {
                function_names: vec![output_name.clone()],
            })
            .structured_output::<StructuredAnswer>(StructuredOutputPolicy {
                mode: OutputMode::Tool,
                max_retries: 1,
                best_effort: false,
            })
            .build(),
    )?;

    let result = runtime.run(agent, "specific output").await?;

    assert_eq!(
        result.structured_output,
        Some(serde_json::json!({"answer":"tool"}))
    );
    assert_eq!(model.request_count(), 2);
    assert_eq!(result.text.as_deref(), Some(r#"{"answer":"tool"}"#));
    assert_no_orphan_tool_use(&result.transcript);
    assert!(
        model
            .requests()
            .iter()
            .all(|request| { request.tools.len() == 1 && request.tools[0].name == output_name })
    );
    Ok(())
}

#[tokio::test]
async fn required_choice_is_satisfied_by_the_synthetic_output_tool() -> Result<()> {
    let output_name = synthetic_output_tool_name::<StructuredAnswer>(&[]);
    let model = MockCompletionModel::new([MockTurn::tool_call(
        "required",
        output_name.clone(),
        serde_json::json!({"answer":"required"}),
    )]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .tool_choice(ToolChoice::Required)
            .structured_output::<StructuredAnswer>(StructuredOutputPolicy {
                mode: OutputMode::Tool,
                max_retries: 0,
                best_effort: false,
            })
            .build(),
    )?;

    let result = runtime.run(agent, "required output").await?;

    assert_eq!(
        result.structured_output,
        Some(serde_json::json!({"answer":"required"}))
    );
    assert_eq!(result.text.as_deref(), Some(r#"{"answer":"required"}"#));
    assert_no_orphan_tool_use(&result.transcript);
    let request = model.requests().pop().context("model request")?;
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, output_name);
    Ok(())
}

#[tokio::test]
async fn structured_output_validates_and_best_effort_commits_last_response() -> Result<()> {
    let policy = StructuredOutputPolicy {
        mode: OutputMode::Native,
        max_retries: 0,
        best_effort: true,
    };
    let model = MockCompletionModel::text("not-json");
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .into_bevy_agent_builder()
            .structured_output::<StructuredAnswer>(policy)
            .build(),
    )?;

    let result = runtime.run(agent, "answer").await?;

    assert_eq!(result.text.as_deref(), Some("not-json"));
    assert!(result.structured_output.is_none());
    assert_eq!(result.transcript.len(), 2);
    Ok(())
}

#[tokio::test]
async fn structured_output_honors_refs_combinators_and_constraints() -> Result<()> {
    let schema = serde_json::from_value(serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {
            "answer": {
                "type": "string",
                "minLength": 5,
                "pattern": "^[a-z]+$"
            }
        },
        "type": "object",
        "required": ["answer"],
        "properties": {
            "answer": {
                "anyOf": [
                    {"$ref": "#/$defs/answer"},
                    {"const": "fallback"}
                ]
            }
        },
        "additionalProperties": false
    }))?;
    let model = MockCompletionModel::new([
        MockTurn::text(r#"{"answer":"x"}"#),
        MockTurn::text(r#"{"answer":"valid"}"#),
    ]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .structured_output_raw(
                schema,
                StructuredOutputPolicy {
                    mode: OutputMode::Native,
                    max_retries: 1,
                    best_effort: false,
                },
            )
            .build(),
    )?;

    let result = runtime.run(agent, "schema").await?;

    assert_eq!(model.request_count(), 2);
    assert_eq!(
        result.structured_output,
        Some(serde_json::json!({"answer":"valid"}))
    );
    Ok(())
}

#[tokio::test]
async fn tool_structured_mode_rejects_textual_json() -> Result<()> {
    let model = MockCompletionModel::new([
        MockTurn::text(r#"{"answer":"first"}"#),
        MockTurn::text(r#"{"answer":"second"}"#),
    ]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .structured_output::<StructuredAnswer>(StructuredOutputPolicy {
                mode: OutputMode::Tool,
                max_retries: 1,
                best_effort: false,
            })
            .build(),
    )?;

    let error = runtime
        .run(agent, "tool-only")
        .await
        .err()
        .context("tool mode must require the synthetic output tool")?;

    assert!(matches!(error, RuntimeError::StructuredOutput(_)));
    assert_eq!(model.request_count(), 2);
    Ok(())
}

#[tokio::test]
async fn exact_inflight_tool_version_survives_replacement() -> Result<()> {
    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "old-call",
            "counting_portable_tool",
            serde_json::json!({"value": "old"}),
        ),
        MockTurn::text("done"),
    ]);
    let old = CountingPortableTool::default();
    let new = CountingPortableTool::default();
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().build())?;
    let grant = runtime.install_tool(agent, old.clone())?;
    let handle = runtime.start_run(agent, "go")?;

    // First pass creates the immutable turn snapshot and dispatches the model.
    let _ = runtime.step(handle).await?;
    runtime.replace_tool(grant.grant_id, new.clone())?;
    let result = runtime.drive_to_terminal(handle).await?;

    assert!(matches!(result.terminal_reason, TerminalReason::Completed));
    assert_eq!(old.calls(), ["old"]);
    assert!(new.calls().is_empty());
    Ok(())
}

#[test]
fn revoked_grants_cannot_fork_the_immutable_replacement_chain() -> Result<()> {
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        MockCompletionModel::text("unused")
            .into_bevy_agent_builder()
            .build(),
    )?;
    let original = runtime.install_tool(agent, CountingPortableTool::default())?;
    let replacement = runtime.replace_tool(original.grant_id, CountingPortableTool::default())?;
    let topology_after_replacement = runtime.world().entities().len();

    let repeated = runtime
        .replace_tool(original.grant_id, CountingPortableTool::default())
        .err()
        .context("an already-revoked predecessor must not fork the version chain")?;
    assert!(matches!(
        repeated,
        RuntimeError::RevokedGrant(id) if id == original.grant_id
    ));
    assert_eq!(replacement.revision, 2);
    assert_eq!(runtime.world().entities().len(), topology_after_replacement);

    runtime.revoke_grant(replacement.grant_id)?;
    let revoked = runtime
        .replace_tool(replacement.grant_id, CountingPortableTool::default())
        .err()
        .context("an explicitly revoked grant must not authorize replacement")?;
    assert!(matches!(
        revoked,
        RuntimeError::RevokedGrant(id) if id == replacement.grant_id
    ));
    assert_eq!(runtime.world().entities().len(), topology_after_replacement);
    Ok(())
}

#[test]
fn duplicate_active_tool_names_are_rejected_atomically() -> Result<()> {
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        MockCompletionModel::text("unused")
            .into_bevy_agent_builder()
            .build(),
    )?;
    runtime.install_tool(agent, CountingPortableTool::default())?;
    let topology_before = runtime.world().entities().len();

    let error = runtime
        .install_tool(agent, CountingPortableTool::default())
        .err()
        .context("one agent's provider-facing tool namespace must be unique")?;
    assert!(matches!(
        error,
        RuntimeError::DuplicateToolName { agent_id, .. } if agent_id == agent
    ));
    assert_eq!(runtime.world().entities().len(), topology_before);
    Ok(())
}

#[tokio::test]
async fn memory_bindings_cannot_cross_tenants() -> Result<()> {
    let owner = TenantId::new();
    let attacker = TenantId::new();
    let memory = CountingMemory::default();
    let mut runtime = runtime()?;
    let memory_id = runtime.register_memory(owner, memory.clone());

    let definition = MockCompletionModel::text("unused")
        .into_bevy_agent_builder()
        .tenant(attacker)
        .memory(memory_id, "cross-tenant")
        .build();
    let rejected_model_id = definition.model_id();
    let error = runtime
        .spawn_agent(definition)
        .err()
        .context("cross-tenant memory construction must fail")?;

    assert!(matches!(error, RuntimeError::TenantMismatch { .. }));
    let leaked_model = runtime
        .spawn_agent_spec(AgentSpec::new(rejected_model_id, attacker))
        .err()
        .context("failed agent construction must not leave a registered model")?;
    assert!(matches!(
        leaked_model,
        RuntimeError::UnknownModel(id) if id == rejected_model_id
    ));
    assert_eq!(memory.load_count(), 0);
    assert_eq!(memory.append_count(), 0);
    Ok(())
}

#[tokio::test]
async fn ingress_events_rejections_and_cancellation_are_bounded() -> Result<()> {
    let model = SlowMock {
        inner: MockCompletionModel::text("late"),
        delay: Duration::from_secs(1),
    };
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        ingress_capacity: 1,
        event_capacity: 1,
        rejection_capacity: 2,
        effect_timeout: Duration::from_secs(2),
        ..RuntimeConfig::default()
    })?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().build())?;
    let handle = runtime.start_run(agent, "bounds")?;
    let mut events = runtime.subscribe(handle)?;
    let _ = runtime.step(handle).await?;
    let header = runtime
        .active_effect_headers(handle)?
        .into_iter()
        .next()
        .context("model operation")?;

    runtime.ingest(EffectIngress::Delta {
        header,
        sequence: 0,
        delta: ProvisionalDelta::Text("first".to_string()),
    })?;
    assert!(matches!(
        runtime.ingest(EffectIngress::Delta {
            header,
            sequence: 1,
            delta: ProvisionalDelta::Text("second".to_string()),
        }),
        Err(RuntimeError::IngressFull)
    ));
    let _ = runtime.step(handle).await?;
    assert!(matches!(
        events.try_recv(),
        Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_))
    ));

    for sequence in 1..=3 {
        let mut hostile = header;
        hostile.runtime_id = rig_bevy::RuntimeId::new();
        runtime.ingest(EffectIngress::Delta {
            header: hostile,
            sequence,
            delta: ProvisionalDelta::Text("hostile".to_string()),
        })?;
        let _ = runtime.step(handle).await?;
    }
    assert_eq!(runtime.effect_rejections().len(), 2);

    runtime.cancel(handle)?;
    let result = runtime.drive_to_terminal(handle).await?;
    assert!(matches!(result.terminal_reason, TerminalReason::Cancelled));
    assert!(runtime.active_effect_headers(handle)?.is_empty());
    Ok(())
}

#[tokio::test]
async fn tenant_validation_precedes_run_lifecycle_validation() -> Result<()> {
    let owner = TenantId::new();
    let attacker = TenantId::new();
    let model = SlowMock {
        inner: MockCompletionModel::text("late"),
        delay: Duration::from_secs(1),
    };
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().tenant(owner).build())?;
    let handle = runtime.start_run(agent, "tenant boundary")?;
    let _ = runtime.step(handle).await?;
    let header = runtime
        .active_effect_headers(handle)?
        .into_iter()
        .next()
        .context("model operation")?;
    let mut hostile_header = header;
    hostile_header.tenant_id = attacker;

    runtime.ingest(EffectIngress::Delta {
        header: hostile_header,
        sequence: 0,
        delta: ProvisionalDelta::Text("active probe".to_string()),
    })?;
    let _ = runtime.step(handle).await?;
    assert_eq!(
        runtime.effect_rejections().last().map(|entry| entry.reason),
        Some(EffectRejectionReason::WrongTenant)
    );

    let hostile_handle = RunHandle {
        runtime_id: handle.runtime_id,
        run_id: handle.run_id,
        generation: Generation(handle.generation.0.saturating_add(1)),
        tenant_id: attacker,
    };
    assert!(matches!(
        runtime.cancel(hostile_handle),
        Err(RuntimeError::TenantMismatch { .. })
    ));

    runtime.cancel(handle)?;
    let result = runtime.drive_to_terminal(handle).await?;
    assert!(matches!(result.terminal_reason, TerminalReason::Cancelled));
    runtime.ingest(EffectIngress::Delta {
        header: hostile_header,
        sequence: 1,
        delta: ProvisionalDelta::Text("terminal probe".to_string()),
    })?;
    assert!(matches!(
        runtime.step(handle).await?,
        RunStepStatus::Terminal
    ));
    assert_eq!(
        runtime.effect_rejections().last().map(|entry| entry.reason),
        Some(EffectRejectionReason::WrongTenant)
    );
    assert!(matches!(
        runtime.finish_run(hostile_handle),
        Err(RuntimeError::TenantMismatch { .. })
    ));
    Ok(())
}

#[tokio::test]
async fn hosted_driver_waits_for_slow_effect_without_burning_schedule_passes() -> Result<()> {
    let model = SlowMock {
        inner: MockCompletionModel::text("slow success"),
        delay: Duration::from_millis(75),
    };
    let mut local = LocalRuntime::with_config(RuntimeConfig {
        max_schedule_passes: 8,
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    })?;
    let agent = local.spawn_agent(model.into_bevy_agent_builder().build())?;
    let hosted = HostedRuntime::new(local);
    let handle = hosted.start_run(agent, "slow").await?;

    let result = hosted.drive_to_terminal(handle).await?;

    assert!(matches!(result.terminal_reason, TerminalReason::Completed));
    assert_eq!(result.text.as_deref(), Some("slow success"));
    Ok(())
}

#[tokio::test]
async fn effect_queue_backpressure_waits_across_three_runs_without_livelock() -> Result<()> {
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        effect_queue_capacity: 1,
        max_effects: 1,
        max_model_calls: 1,
        max_schedule_passes: 4,
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    })?;
    let slow_agent = runtime.spawn_agent(
        SlowMock {
            inner: MockCompletionModel::text("first"),
            delay: Duration::from_millis(80),
        }
        .into_bevy_agent_builder()
        .build(),
    )?;
    let middle_agent = runtime.spawn_agent(
        MockCompletionModel::text("second")
            .into_bevy_agent_builder()
            .build(),
    )?;
    let target_agent = runtime.spawn_agent(
        MockCompletionModel::text("third")
            .into_bevy_agent_builder()
            .build(),
    )?;
    let slow = runtime.start_run(slow_agent, "slow")?;
    let _ = runtime.step(slow).await?;
    let middle = runtime.start_run(middle_agent, "middle")?;
    let target = runtime.start_run(target_agent, "target")?;

    let target_result = tokio::time::timeout(
        Duration::from_millis(500),
        runtime.drive_to_terminal(target),
    )
    .await
    .context("a capacity-waiting run must wake as earlier effects drain")??;

    let slow_result = runtime.drive_to_terminal(slow).await?;
    let middle_result = runtime.drive_to_terminal(middle).await?;
    assert_eq!(target_result.text.as_deref(), Some("third"));
    assert_eq!(slow_result.text.as_deref(), Some("first"));
    assert_eq!(middle_result.text.as_deref(), Some("second"));
    Ok(())
}

#[tokio::test]
async fn panicking_model_cannot_hang_local_driver() -> Result<()> {
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        effect_timeout: Duration::from_millis(25),
        ..RuntimeConfig::default()
    })?;
    let agent = runtime.spawn_agent(PanicModel.into_bevy_agent_builder().build())?;

    let outcome = tokio::time::timeout(Duration::from_millis(250), runtime.run(agent, "panic"))
        .await
        .context("driver must remain bounded after an effect task panics")?;
    let error = outcome.err().context("panicking model must fail the run")?;

    assert!(matches!(
        error,
        RuntimeError::RunFailed { ref code, .. } if code == "effect_wait"
    ));
    Ok(())
}

#[tokio::test]
async fn foreign_ingress_cannot_extend_a_missing_target_effect_deadline() -> Result<()> {
    let foreign_events = (0..20_000)
        .map(|index| MockStreamEvent::text(index.to_string()))
        .chain(std::iter::once(
            MockStreamEvent::final_response_with_total_tokens(20_000),
        ))
        .collect::<Vec<_>>();
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        ingress_capacity: 1,
        max_effects: 2,
        max_model_calls: 2,
        effect_timeout: Duration::from_millis(25),
        ..RuntimeConfig::default()
    })?;
    let foreign_agent = runtime.spawn_agent(
        MockCompletionModel::from_stream_turns([foreign_events])
            .into_bevy_agent_builder()
            .streaming(StreamingMode::Streaming)
            .build(),
    )?;
    let target_agent = runtime.spawn_agent(PanicModel.into_bevy_agent_builder().build())?;
    let _foreign = runtime.start_run(foreign_agent, "foreign stream")?;
    let target = runtime.start_run(target_agent, "missing completion")?;

    let outcome = tokio::time::timeout(
        Duration::from_millis(250),
        runtime.drive_to_terminal(target),
    )
    .await
    .context("foreign ingress must not postpone the target effect deadline")?;
    let result = outcome?;

    assert!(matches!(
        result.terminal_reason,
        TerminalReason::Failed { .. }
    ));
    Ok(())
}

#[tokio::test]
async fn dropping_hidden_handle_run_future_cancels_and_cleans_up() -> Result<()> {
    let config = RuntimeConfig {
        effect_timeout: Duration::from_secs(1),
        terminal_retention_ticks: 1,
        ..RuntimeConfig::default()
    };
    let tenant_id = TenantId::new();
    let slow_inner = MockCompletionModel::text("must not commit");
    let fast_model = MockCompletionModel::text("fast");
    let mut runtime = LocalRuntime::with_config(config)?;
    let slow_model_id = runtime.register_persistable_model(
        tenant_id,
        BindingIdentity::new("slow-model", "v1"),
        SlowMock {
            inner: slow_inner.clone(),
            delay: Duration::from_millis(200),
        },
    );
    let fast_model_id = runtime.register_persistable_model(
        tenant_id,
        BindingIdentity::new("fast-model", "v1"),
        fast_model.clone(),
    );
    let slow_agent = runtime.spawn_agent_spec(AgentSpec::new(slow_model_id, tenant_id))?;
    let fast_agent = runtime.spawn_agent_spec(AgentSpec::new(fast_model_id, tenant_id))?;

    assert!(
        tokio::time::timeout(
            Duration::from_millis(20),
            runtime.run(slow_agent, "abandoned prompt")
        )
        .await
        .is_err()
    );
    let _ = runtime.run(fast_agent, "retained prompt").await?;
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert_eq!(slow_inner.request_count(), 0);

    let protector = XorProtector(0x37);
    let snapshot = runtime
        .protected_snapshot_with_policy(&protector, SnapshotContentPolicy::CanonicalRunState)?;
    let plaintext = protector.unprotect(&snapshot.payload)?;
    let domain: serde_json::Value = serde_json::from_slice(&plaintext)?;
    let runs = domain
        .get("runs")
        .and_then(serde_json::Value::as_array)
        .context("snapshot runs")?;
    assert_eq!(runs.len(), 1);
    assert!(!String::from_utf8(plaintext)?.contains("abandoned prompt"));
    Ok(())
}

#[tokio::test]
async fn hosted_runs_use_run_scoped_progress_and_wakeups() -> Result<()> {
    let mut local = LocalRuntime::with_config(RuntimeConfig {
        max_schedule_passes: 64,
        effect_timeout: Duration::from_millis(300),
        ..RuntimeConfig::default()
    })?;
    let slow_agent = local.spawn_agent(
        SlowMock {
            inner: MockCompletionModel::text("slow"),
            delay: Duration::from_millis(80),
        }
        .into_bevy_agent_builder()
        .build(),
    )?;
    let fast_agent = local.spawn_agent(
        SlowMock {
            inner: MockCompletionModel::text("fast"),
            delay: Duration::from_millis(10),
        }
        .into_bevy_agent_builder()
        .build(),
    )?;
    let hosted = HostedRuntime::new(local);
    let slow = hosted.start_run(slow_agent, "slow").await?;
    let fast = hosted.start_run(fast_agent, "fast").await?;

    let (slow_result, fast_result) = tokio::time::timeout(Duration::from_millis(200), async {
        tokio::join!(
            hosted.drive_to_terminal(slow),
            hosted.drive_to_terminal(fast)
        )
    })
    .await
    .context("one run's completion must not strand another run's waiter")?;

    assert_eq!(slow_result?.text.as_deref(), Some("slow"));
    assert_eq!(fast_result?.text.as_deref(), Some("fast"));
    Ok(())
}

#[tokio::test]
async fn hosted_driver_drains_foreign_ingress_that_blocks_its_own_completion() -> Result<()> {
    let mut local = LocalRuntime::with_config(RuntimeConfig {
        ingress_capacity: 1,
        max_effects: 2,
        max_model_calls: 2,
        effect_timeout: Duration::from_millis(500),
        ..RuntimeConfig::default()
    })?;
    let early_agent = local.spawn_agent(
        SlowMock {
            inner: MockCompletionModel::text("early"),
            delay: Duration::from_millis(5),
        }
        .into_bevy_agent_builder()
        .build(),
    )?;
    let target_agent = local.spawn_agent(
        SlowMock {
            inner: MockCompletionModel::text("target"),
            delay: Duration::from_millis(40),
        }
        .into_bevy_agent_builder()
        .build(),
    )?;
    let hosted = HostedRuntime::new(local);
    let early = hosted.start_run(early_agent, "early").await?;
    let target = hosted.start_run(target_agent, "target").await?;

    let target_result =
        tokio::time::timeout(Duration::from_millis(300), hosted.drive_to_terminal(target))
            .await
            .context("foreign ingress must wake the target driver to free the shared channel")??;

    assert_eq!(target_result.text.as_deref(), Some("target"));
    assert_eq!(
        hosted.finish_run(early).await?.text.as_deref(),
        Some("early")
    );
    Ok(())
}

#[tokio::test]
async fn hosted_cancellation_wakes_a_run_waiting_for_effect_queue_capacity() -> Result<()> {
    let mut local = LocalRuntime::with_config(RuntimeConfig {
        effect_queue_capacity: 1,
        max_effects: 1,
        max_model_calls: 1,
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    })?;
    let blocking_agent = local.spawn_agent(
        SlowMock {
            inner: MockCompletionModel::text("blocking"),
            delay: Duration::from_millis(500),
        }
        .into_bevy_agent_builder()
        .build(),
    )?;
    let waiting_agent = local.spawn_agent(
        MockCompletionModel::text("must not dispatch")
            .into_bevy_agent_builder()
            .build(),
    )?;
    let hosted = HostedRuntime::new(local);
    let blocking = hosted.start_run(blocking_agent, "blocking").await?;
    let _ = hosted.step(blocking).await?;
    let waiting = hosted.start_run(waiting_agent, "waiting").await?;
    let driver = {
        let hosted = hosted.clone();
        tokio::spawn(async move { hosted.drive_to_terminal(waiting).await })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;

    hosted.cancel(waiting).await?;
    let result = tokio::time::timeout(Duration::from_millis(200), driver)
        .await
        .context("cancellation must wake the capacity waiter")???;

    assert!(matches!(result.terminal_reason, TerminalReason::Cancelled));
    hosted.cancel(blocking).await?;
    let _ = hosted.drive_to_terminal(blocking).await?;
    Ok(())
}

#[tokio::test]
async fn hosted_deferred_effect_wakes_when_a_semaphore_owner_is_cancelled() -> Result<()> {
    let mut local = LocalRuntime::with_config(RuntimeConfig {
        effect_queue_capacity: 8,
        max_effects: 1,
        max_model_calls: 1,
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    })?;
    let blocking_agent = local.spawn_agent(
        SlowMock {
            inner: MockCompletionModel::text("blocking"),
            delay: Duration::from_millis(500),
        }
        .into_bevy_agent_builder()
        .build(),
    )?;
    let target_agent = local.spawn_agent(
        MockCompletionModel::text("target")
            .into_bevy_agent_builder()
            .build(),
    )?;
    let hosted = HostedRuntime::new(local);
    let blocking = hosted.start_run(blocking_agent, "blocking").await?;
    let _ = hosted.step(blocking).await?;
    let target = hosted.start_run(target_agent, "deferred").await?;
    let _ = hosted.step(target).await?;
    let driver = {
        let hosted = hosted.clone();
        tokio::spawn(async move { hosted.drive_to_terminal(target).await })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;

    hosted.cancel(blocking).await?;
    let blocker_result = hosted.drive_to_terminal(blocking).await?;
    assert!(matches!(
        blocker_result.terminal_reason,
        TerminalReason::Cancelled
    ));
    let target_result = tokio::time::timeout(Duration::from_millis(200), driver)
        .await
        .context("deferred effect must wake when execution capacity is released")???;

    assert!(matches!(
        target_result.terminal_reason,
        TerminalReason::Completed
    ));
    assert_eq!(target_result.text.as_deref(), Some("target"));
    Ok(())
}

#[tokio::test]
async fn dropping_hosted_driver_future_leaves_the_run_resumable() -> Result<()> {
    let mut local = LocalRuntime::with_config(RuntimeConfig {
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    })?;
    let agent = local.spawn_agent(
        SlowMock {
            inner: MockCompletionModel::text("resumed"),
            delay: Duration::from_millis(80),
        }
        .into_bevy_agent_builder()
        .build(),
    )?;
    let hosted = HostedRuntime::new(local);
    let handle = hosted.start_run(agent, "resume after drop").await?;
    let driver = {
        let hosted = hosted.clone();
        tokio::spawn(async move { hosted.drive_to_terminal(handle).await })
    };
    tokio::time::sleep(Duration::from_millis(10)).await;
    driver.abort();
    let _ = driver.await;

    let result = hosted.drive_to_terminal(handle).await?;

    assert!(matches!(result.terminal_reason, TerminalReason::Completed));
    assert_eq!(result.text.as_deref(), Some("resumed"));
    Ok(())
}

#[tokio::test]
async fn hosted_runtime_rejects_a_second_driver_for_the_same_handle() -> Result<()> {
    let mut local = runtime()?;
    let agent = local.spawn_agent(
        SlowMock {
            inner: MockCompletionModel::text("single owner"),
            delay: Duration::from_millis(40),
        }
        .into_bevy_agent_builder()
        .build(),
    )?;
    let hosted = HostedRuntime::new(local);
    let handle = hosted.start_run(agent, "one handle").await?;

    let (left, right) = tokio::join!(
        hosted.drive_to_terminal(handle),
        hosted.drive_to_terminal(handle)
    );
    let outcomes = [left, right];
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| outcome
                .as_ref()
                .is_ok_and(|result| result.text.as_deref() == Some("single owner")))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, Err(RuntimeError::RunAlreadyDriven(id)) if *id == handle.run_id))
            .count(),
        1
    );
    Ok(())
}

struct XorProtector(u8);

impl SnapshotProtector for XorProtector {
    fn protector_id(&self) -> &str {
        "test-xor-v1"
    }

    fn protect(&self, plaintext: &[u8]) -> Result<Vec<u8>, SnapshotError> {
        Ok(plaintext.iter().map(|byte| byte ^ self.0).collect())
    }

    fn unprotect(&self, protected: &[u8]) -> Result<Vec<u8>, SnapshotError> {
        Ok(protected.iter().map(|byte| byte ^ self.0).collect())
    }
}

fn reprotect_domain(
    snapshot: &ProtectedSnapshot,
    protector: &dyn SnapshotProtector,
    domain: &serde_json::Value,
) -> Result<ProtectedSnapshot> {
    let payload = protector.protect(&serde_json::to_vec(domain)?)?;
    Ok(ProtectedSnapshot {
        version: snapshot.version,
        protector_id: snapshot.protector_id.clone(),
        protected_digest: Sha256::digest(&payload).into(),
        payload,
    })
}

#[tokio::test]
async fn canonical_snapshot_validation_rejects_impossible_transcript_and_accounting() -> Result<()>
{
    let tenant_id = TenantId::new();
    let identity = BindingIdentity::new("snapshot-validation-model", "v1");
    let model = MockCompletionModel::new([MockTurn::text("done").with_usage(Usage {
        total_tokens: 3,
        ..Usage::new()
    })]);
    let mut runtime = runtime()?;
    let model_id = runtime.register_persistable_model(tenant_id, identity, model.clone());
    let agent = runtime.spawn_agent_spec(AgentSpec::new(model_id, tenant_id))?;
    let _ = runtime.run(agent, "snapshot validation").await?;
    let protector = XorProtector(0x61);
    let snapshot = runtime
        .protected_snapshot_with_policy(&protector, SnapshotContentPolicy::CanonicalRunState)?;
    let base: serde_json::Value = serde_json::from_slice(&protector.unprotect(&snapshot.payload)?)?;
    let mut invalid_domains = Vec::new();

    let mut boundary = base.clone();
    let boundary_run = boundary
        .get_mut("runs")
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|runs| runs.first_mut())
        .context("boundary run")?;
    let message_count = boundary_run
        .pointer("/transcript/messages")
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
        .context("transcript messages")?;
    boundary_run["transcript"]["new_messages_start"] = serde_json::json!(message_count + 1);
    invalid_domains.push(boundary);

    let mut duplicate_operation = base.clone();
    let duplicate_accounting = duplicate_operation
        .get_mut("runs")
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|runs| runs.first_mut())
        .and_then(|run| run.get_mut("accounting"))
        .context("duplicate accounting")?;
    let duplicate_call = duplicate_accounting
        .get("model_calls")
        .and_then(serde_json::Value::as_array)
        .and_then(|calls| calls.first())
        .cloned()
        .context("model call")?;
    duplicate_accounting["model_calls"]
        .as_array_mut()
        .context("model calls array")?
        .push(duplicate_call);
    duplicate_accounting["model_calls_dispatched"] = serde_json::json!(2);
    invalid_domains.push(duplicate_operation);

    let mut impossible_dispatch_count = base.clone();
    impossible_dispatch_count["runs"][0]["accounting"]["model_calls_dispatched"] =
        serde_json::json!(0);
    invalid_domains.push(impossible_dispatch_count);

    let mut over_budget_dispatch_count = base.clone();
    over_budget_dispatch_count["runs"][0]["accounting"]["model_calls_dispatched"] =
        serde_json::json!(9);
    invalid_domains.push(over_budget_dispatch_count);

    let mut out_of_order_observation = base.clone();
    out_of_order_observation["runs"][0]["terminal"]["terminal_tick"] = serde_json::json!(1);
    out_of_order_observation["runs"][0]["observation"]["observed_tick"] = serde_json::json!(0);
    invalid_domains.push(out_of_order_observation);

    let tool_call = Message::Assistant {
        id: None,
        content: OneOrMany::one(AssistantContent::tool_call(
            "snapshot-call",
            "snapshot-tool",
            serde_json::json!({}),
        )),
    };
    let mut orphan_call = base.clone();
    orphan_call["runs"][0]["transcript"]["messages"] =
        serde_json::to_value([Message::user("prompt"), tool_call.clone()])?;
    invalid_domains.push(orphan_call);

    let mut unmatched_result = base.clone();
    unmatched_result["runs"][0]["transcript"]["messages"] = serde_json::to_value([
        Message::user("prompt"),
        tool_call.clone(),
        Message::tool_result("different-call", "result"),
    ])?;
    invalid_domains.push(unmatched_result);

    let Message::User { content } = Message::tool_result("snapshot-call", "result") else {
        anyhow::bail!("tool-result constructor returned the wrong role");
    };
    let result = content.first();
    let duplicate_result = Message::User {
        content: OneOrMany::many([result.clone(), result])?,
    };
    let mut duplicate_pairing = base.clone();
    duplicate_pairing["runs"][0]["transcript"]["messages"] =
        serde_json::to_value([Message::user("prompt"), tool_call, duplicate_result])?;
    invalid_domains.push(duplicate_pairing);

    let mut duplicate_assistant_role = base.clone();
    duplicate_assistant_role["runs"][0]["transcript"]["messages"] = serde_json::to_value([
        Message::user("prompt"),
        Message::assistant("first"),
        Message::assistant("second"),
    ])?;
    invalid_domains.push(duplicate_assistant_role);

    let mut usage_mismatch = base;
    usage_mismatch["runs"][0]["accounting"]["usage"]["total_tokens"] = serde_json::json!(4);
    invalid_domains.push(usage_mismatch);

    for domain in invalid_domains {
        let tampered = reprotect_domain(&snapshot, &protector, &domain)?;
        let mut bindings = RebindRegistry::new();
        bindings.bind_model(model_id, tenant_id, identity, model.clone());
        let error =
            LocalRuntime::restore(RuntimeConfig::default(), &tampered, &protector, bindings)
                .err()
                .context("impossible canonical state must be rejected")?;
        assert!(matches!(
            error,
            RuntimeError::Snapshot(SnapshotError::InvalidRelationship(_))
        ));
    }
    Ok(())
}

#[test]
fn canonical_snapshot_validation_binds_run_policy_state_to_its_agent() -> Result<()> {
    let tenant_id = TenantId::new();
    let identity = BindingIdentity::new("snapshot-policy-model", "v1");
    let model = MockCompletionModel::text("unused");
    let policy = StructuredOutputPolicy {
        mode: OutputMode::Native,
        max_retries: 1,
        best_effort: false,
    };
    let mut runtime = runtime()?;
    let model_id = runtime.register_persistable_model(tenant_id, identity, model.clone());
    let agent = runtime.spawn_agent_spec(
        AgentSpec::new(model_id, tenant_id)
            .structured_output::<StructuredAnswer>(policy)
            .response_retry_policy(ResponseRetryPolicy::RejectEmpty { max_retries: 1 })
            .invalid_tool_policy(rig_bevy::InvalidToolPolicy::Retry { max_retries: 1 }),
    )?;
    let _ = runtime.start_run(agent, "snapshot policy")?;
    let protector = XorProtector(0x64);
    let snapshot = runtime
        .protected_snapshot_with_policy(&protector, SnapshotContentPolicy::CanonicalRunState)?;
    let base: serde_json::Value = serde_json::from_slice(&protector.unprotect(&snapshot.payload)?)?;
    let mut invalid_domains = Vec::new();

    let mut missing_structured = base.clone();
    missing_structured["runs"][0]["structured"] = serde_json::Value::Null;
    invalid_domains.push(missing_structured);

    let mut added_to_unstructured = base.clone();
    added_to_unstructured["agents"][0]["spec"]["structured_output"] = serde_json::Value::Null;
    invalid_domains.push(added_to_unstructured);

    let mut changed_schema = base.clone();
    changed_schema["runs"][0]["structured"]["schema"] = serde_json::json!({"type":"string"});
    invalid_domains.push(changed_schema);

    let mut changed_policy = base.clone();
    changed_policy["runs"][0]["structured"]["policy"]["best_effort"] =
        serde_json::Value::Bool(true);
    invalid_domains.push(changed_policy);

    let mut excessive_structured_retries = base.clone();
    excessive_structured_retries["runs"][0]["structured"]["retries"] = serde_json::json!(2);
    invalid_domains.push(excessive_structured_retries);

    let mut excessive_response_retries = base.clone();
    excessive_response_retries["runs"][0]["response_retries"] = serde_json::json!(2);
    invalid_domains.push(excessive_response_retries);

    let mut excessive_invalid_tool_retries = base.clone();
    excessive_invalid_tool_retries["runs"][0]["invalid_tool_retries"] = serde_json::json!(2);
    invalid_domains.push(excessive_invalid_tool_retries);

    let mut inconsistent_resolution = base;
    inconsistent_resolution["runs"][0]["structured"]["output_tool_name"] =
        serde_json::json!("forged_output_tool");
    invalid_domains.push(inconsistent_resolution);

    for domain in invalid_domains {
        let tampered = reprotect_domain(&snapshot, &protector, &domain)?;
        let mut bindings = RebindRegistry::new();
        bindings.bind_model(model_id, tenant_id, identity, model.clone());
        let error =
            LocalRuntime::restore(RuntimeConfig::default(), &tampered, &protector, bindings)
                .err()
                .context("tampered run policy state must be rejected")?;
        assert!(matches!(
            error,
            RuntimeError::Snapshot(SnapshotError::InvalidRelationship(_))
        ));
    }
    Ok(())
}

#[tokio::test]
async fn canonical_snapshot_validation_rejects_impossible_memory_lifecycle_flags() -> Result<()> {
    let tenant_id = TenantId::new();
    let model_identity = BindingIdentity::new("snapshot-memory-model", "v1");
    let memory_identity = BindingIdentity::new("snapshot-memory", "v1");
    let model = MockCompletionModel::text("done");
    let memory = CountingMemory::default();
    let mut runtime = runtime()?;
    let model_id = runtime.register_persistable_model(tenant_id, model_identity, model.clone());
    let memory_id = runtime.register_persistable_memory(tenant_id, memory_identity, memory.clone());
    let agent = runtime.spawn_agent_spec(
        AgentSpec::new(model_id, tenant_id).memory(memory_id, "snapshot-memory-thread"),
    )?;
    let _ = runtime.run(agent, "persist memory").await?;
    let protector = XorProtector(0x65);
    let snapshot = runtime
        .protected_snapshot_with_policy(&protector, SnapshotContentPolicy::CanonicalRunState)?;
    let base: serde_json::Value = serde_json::from_slice(&protector.unprotect(&snapshot.payload)?)?;

    let mut loaded_false = base.clone();
    loaded_false["runs"][0]["memory"]["loaded"] = serde_json::Value::Bool(false);

    let mut ready_but_appended = base;
    ready_but_appended["runs"][0]["phase"] = serde_json::to_value(RunPhase::ReadyModel)?;
    ready_but_appended["runs"][0]["terminal"] = serde_json::Value::Null;
    ready_but_appended["runs"][0]["observation"] = serde_json::Value::Null;

    for domain in [loaded_false, ready_but_appended] {
        let tampered = reprotect_domain(&snapshot, &protector, &domain)?;
        let mut bindings = RebindRegistry::new();
        bindings.bind_model(model_id, tenant_id, model_identity, model.clone());
        bindings.bind_memory(memory_id, tenant_id, memory_identity, memory.clone());
        let error =
            LocalRuntime::restore(RuntimeConfig::default(), &tampered, &protector, bindings)
                .err()
                .context("tampered memory lifecycle state must be rejected")?;
        assert!(matches!(
            error,
            RuntimeError::Snapshot(SnapshotError::InvalidRelationship(_))
        ));
    }
    Ok(())
}

#[tokio::test]
async fn canonical_snapshot_validation_rejects_invalid_structured_values() -> Result<()> {
    let tenant_id = TenantId::new();
    let identity = BindingIdentity::new("snapshot-structured-value-model", "v1");
    let model = MockCompletionModel::text(r#"{"answer":"valid"}"#);
    let mut runtime = runtime()?;
    let model_id = runtime.register_persistable_model(tenant_id, identity, model.clone());
    let agent = runtime.spawn_agent_spec(
        AgentSpec::new(model_id, tenant_id).structured_output::<StructuredAnswer>(
            StructuredOutputPolicy {
                mode: OutputMode::Native,
                max_retries: 0,
                best_effort: false,
            },
        ),
    )?;
    let _ = runtime.run(agent, "persist structured value").await?;
    let protector = XorProtector(0x66);
    let snapshot = runtime
        .protected_snapshot_with_policy(&protector, SnapshotContentPolicy::CanonicalRunState)?;
    let base: serde_json::Value = serde_json::from_slice(&protector.unprotect(&snapshot.payload)?)?;

    let mut schema_invalid = base.clone();
    schema_invalid["runs"][0]["structured"]["value"] = serde_json::json!({"answer": 7});

    let mut value_before_terminal = base;
    value_before_terminal["runs"][0]["phase"] = serde_json::to_value(RunPhase::ReadyModel)?;
    value_before_terminal["runs"][0]["terminal"] = serde_json::Value::Null;
    value_before_terminal["runs"][0]["observation"] = serde_json::Value::Null;

    for domain in [schema_invalid, value_before_terminal] {
        let tampered = reprotect_domain(&snapshot, &protector, &domain)?;
        let mut bindings = RebindRegistry::new();
        bindings.bind_model(model_id, tenant_id, identity, model.clone());
        let error =
            LocalRuntime::restore(RuntimeConfig::default(), &tampered, &protector, bindings)
                .err()
                .context("tampered structured-output value must be rejected")?;
        assert!(matches!(
            error,
            RuntimeError::Snapshot(SnapshotError::InvalidRelationship(_))
        ));
    }
    Ok(())
}

#[test]
fn snapshot_validation_rejects_duplicate_active_tool_names() -> Result<()> {
    let tenant_id = TenantId::new();
    let model_identity = BindingIdentity::new("duplicate-name-model", "v1");
    let tool_identity = BindingIdentity::new("duplicate-name-tool", "v1");
    let model = MockCompletionModel::text("unused");
    let tool = CountingPortableTool::default();
    let mut runtime = runtime()?;
    let model_id = runtime.register_persistable_model(tenant_id, model_identity, model.clone());
    let agent = runtime.spawn_agent_spec(AgentSpec::new(model_id, tenant_id))?;
    let grant = runtime.install_persistable_tool(agent, tool_identity, tool.clone())?;
    let protector = XorProtector(0x62);
    let snapshot = runtime.protected_snapshot(&protector)?;
    let mut domain: serde_json::Value =
        serde_json::from_slice(&protector.unprotect(&snapshot.payload)?)?;
    let grants = domain
        .get_mut("grants")
        .and_then(serde_json::Value::as_array_mut)
        .context("snapshot grants")?;
    let mut duplicate = grants.first().cloned().context("installed grant")?;
    duplicate["id"] = serde_json::to_value(rig_bevy::GrantId::new())?;
    grants.push(duplicate);
    let tampered = reprotect_domain(&snapshot, &protector, &domain)?;
    let mut bindings = RebindRegistry::new();
    bindings.bind_model(model_id, tenant_id, model_identity, model);
    bindings.bind_tool(grant.capability_id, tool_identity, tool);

    let error = LocalRuntime::restore(RuntimeConfig::default(), &tampered, &protector, bindings)
        .err()
        .context("ambiguous active tool names must be rejected during restore")?;
    assert!(matches!(
        error,
        RuntimeError::Snapshot(SnapshotError::InvalidRelationship(_))
    ));
    Ok(())
}

#[test]
fn unchanged_domain_snapshot_bytes_are_deterministic() -> Result<()> {
    let tenant_id = TenantId::new();
    let model_identity = BindingIdentity::new("deterministic-model", "v1");
    let tool_identity = BindingIdentity::new("deterministic-tool", "v1");
    let mut runtime = runtime()?;
    let model_id = runtime.register_persistable_model(
        tenant_id,
        model_identity,
        MockCompletionModel::text("unused"),
    );
    let agent = runtime.spawn_agent_spec(AgentSpec::new(model_id, tenant_id))?;
    runtime.install_persistable_tool(agent, tool_identity, CountingPortableTool::default())?;
    let protector = XorProtector(0x63);

    let first = runtime.protected_snapshot(&protector)?;
    let second = runtime.protected_snapshot(&protector)?;

    assert_eq!(
        protector.unprotect(&first.payload)?,
        protector.unprotect(&second.payload)?
    );
    Ok(())
}

#[tokio::test]
async fn snapshot_policies_control_content_and_restore_requires_exact_rebinding() -> Result<()> {
    let model = MockCompletionModel::text("secret-answer");
    let mut runtime = runtime()?;
    let tenant_id = TenantId::new();
    let binding_identity = BindingIdentity::new("test-mock-model", "v1/config-a");
    let model_id = runtime.register_persistable_model(tenant_id, binding_identity, model.clone());
    let _unused_model = runtime.register_model(
        tenant_id,
        DifferentMock(MockCompletionModel::text("unused")),
    );
    let agent = runtime.spawn_agent_spec(
        AgentSpec::new(model_id, tenant_id)
            .temperature(0.42)
            .max_tokens(777),
    )?;
    let handle = runtime.start_run(agent, "secret-prompt")?;
    let result = runtime.drive_to_terminal(handle).await?;
    let protector = XorProtector(0xA5);

    let metadata_snapshot = runtime.protected_snapshot(&protector)?;
    let metadata_plaintext = protector.unprotect(&metadata_snapshot.payload)?;
    let metadata_text = String::from_utf8(metadata_plaintext)?;
    assert!(!metadata_text.contains("secret-prompt"));
    assert!(!metadata_text.contains("secret-answer"));

    let snapshot = runtime
        .protected_snapshot_with_policy(&protector, SnapshotContentPolicy::CanonicalRunState)?;
    let canonical_plaintext = protector.unprotect(&snapshot.payload)?;
    // The canonical policy persists run content; the protector must still
    // obscure it in the stored payload bytes.
    assert_ne!(snapshot.payload, canonical_plaintext);
    let canonical_text = String::from_utf8(canonical_plaintext)?;
    assert!(canonical_text.contains("secret-prompt"));
    assert!(canonical_text.contains("secret-answer"));

    let missing = LocalRuntime::restore(
        RuntimeConfig::default(),
        &snapshot,
        &protector,
        RebindRegistry::new(),
    )
    .err()
    .context("exact model binding must be mandatory")?;
    assert!(matches!(
        missing,
        RuntimeError::Snapshot(SnapshotError::MissingModel(_))
    ));

    let mut mismatched_bindings = RebindRegistry::new();
    mismatched_bindings.bind_model(
        model_id,
        tenant_id,
        BindingIdentity::new("test-mock-model", "v2/incompatible"),
        DifferentMock(MockCompletionModel::text("wrong implementation")),
    );
    let mismatch = LocalRuntime::restore(
        RuntimeConfig::default(),
        &snapshot,
        &protector,
        mismatched_bindings,
    )
    .err()
    .context("concrete model mismatch must be rejected")?;
    assert!(matches!(
        mismatch,
        RuntimeError::Snapshot(SnapshotError::BindingMismatch { kind: "model", .. })
    ));

    let mut corrupted = snapshot.clone();
    if let Some(byte) = corrupted.payload.first_mut() {
        *byte ^= 0x01;
    }
    let corruption = LocalRuntime::restore(
        RuntimeConfig::default(),
        &corrupted,
        &protector,
        RebindRegistry::new(),
    )
    .err()
    .context("corrupted protected payload must be rejected")?;
    assert!(matches!(
        corruption,
        RuntimeError::Snapshot(SnapshotError::Integrity)
    ));

    let plaintext = protector.unprotect(&snapshot.payload)?;
    let mut domain: serde_json::Value = serde_json::from_slice(&plaintext)?;
    assert!(
        domain
            .get("agents")
            .and_then(serde_json::Value::as_array)
            .and_then(|agents| agents.first())
            .and_then(|agent| agent.get("spec"))
            .and_then(|spec| spec.get("additional_params"))
            .is_some_and(serde_json::Value::is_null)
    );
    let persisted_spec = domain
        .get("agents")
        .and_then(serde_json::Value::as_array)
        .and_then(|agents| agents.first())
        .and_then(|agent| agent.get("spec"))
        .context("persisted agent spec")?;
    assert_eq!(
        persisted_spec.get("temperature"),
        Some(&serde_json::json!(0.42))
    );
    assert_eq!(
        persisted_spec.get("max_tokens"),
        Some(&serde_json::json!(777))
    );
    let first_run = domain
        .get_mut("runs")
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|runs| runs.first_mut())
        .and_then(serde_json::Value::as_object_mut)
        .context("snapshot should contain one run record")?;
    first_run.insert(
        "tenant_id".to_string(),
        serde_json::to_value(TenantId::new())?,
    );
    let tampered_payload = protector.protect(&serde_json::to_vec(&domain)?)?;
    let tampered = ProtectedSnapshot {
        version: snapshot.version,
        protector_id: snapshot.protector_id.clone(),
        protected_digest: Sha256::digest(&tampered_payload).into(),
        payload: tampered_payload,
    };
    let mut tampered_bindings = RebindRegistry::new();
    tampered_bindings.bind_model(model_id, tenant_id, binding_identity, model.clone());
    let relationship_error = LocalRuntime::restore(
        RuntimeConfig::default(),
        &tampered,
        &protector,
        tampered_bindings,
    )
    .err()
    .context("cross-tenant restored topology must be rejected")?;
    assert!(matches!(
        relationship_error,
        RuntimeError::Snapshot(SnapshotError::InvalidRelationship(_))
    ));

    let mut bindings = RebindRegistry::new();
    bindings.bind_model(model_id, tenant_id, binding_identity, model);
    let mut restored =
        LocalRuntime::restore(RuntimeConfig::default(), &snapshot, &protector, bindings)?;
    assert_ne!(restored.id(), runtime.id());
    let restored_handle = restored.run_handle(result.run_id, tenant_id)?;
    let restored_result = restored.finish_run(restored_handle)?;
    assert_eq!(restored_result.text.as_deref(), Some("secret-answer"));
    let redacted = restored.explain(restored_handle, ContentVisibility::Redacted)?;
    assert!(redacted.canonical_transcript.is_none());
    Ok(())
}

#[test]
fn snapshots_reject_behavior_bearing_config_the_policy_cannot_preserve() -> Result<()> {
    let tenant_id = TenantId::new();
    let model_identity = BindingIdentity::new("test-mock-model", "snapshot-config");

    let mut preamble_runtime = runtime()?;
    let model_id = preamble_runtime.register_persistable_model(
        tenant_id,
        model_identity,
        MockCompletionModel::text("unused"),
    );
    let preamble_agent = preamble_runtime
        .spawn_agent_spec(AgentSpec::new(model_id, tenant_id).preamble("required safety policy"))?;
    let preamble_error = preamble_runtime
        .protected_snapshot(&XorProtector(0x31))
        .err()
        .context("metadata-only snapshot must not drop a preamble")?;
    assert!(matches!(
        preamble_error,
        SnapshotError::MetadataOnlyPreamble(id) if id == preamble_agent
    ));

    let mut params_runtime = runtime()?;
    let model_id = params_runtime.register_persistable_model(
        tenant_id,
        model_identity,
        MockCompletionModel::text("unused"),
    );
    let params_agent = params_runtime.spawn_agent_spec(
        AgentSpec::new(model_id, tenant_id)
            .additional_params(serde_json::json!({"api_key":"must-never-persist"})),
    )?;
    for policy in [
        SnapshotContentPolicy::MetadataOnly,
        SnapshotContentPolicy::CanonicalRunState,
    ] {
        let error = params_runtime
            .protected_snapshot_with_policy(&XorProtector(0x32), policy)
            .err()
            .context("provider parameters must make snapshotting explicit")?;
        assert!(matches!(
            error,
            SnapshotError::NonPersistableProviderParameters(id) if id == params_agent
        ));
    }
    Ok(())
}

#[test]
fn metadata_snapshot_rejects_memory_topology_instead_of_degrading_it() -> Result<()> {
    let tenant_id = TenantId::new();
    let model_identity = BindingIdentity::new("test-mock-model", "memory-snapshot");
    let memory_identity = BindingIdentity::new("test-memory", "memory-snapshot");
    let mut runtime = runtime()?;
    let model_id = runtime.register_persistable_model(
        tenant_id,
        model_identity,
        MockCompletionModel::text("unused"),
    );
    let memory_id =
        runtime.register_persistable_memory(tenant_id, memory_identity, CountingMemory::default());
    let agent = runtime.spawn_agent_spec(
        AgentSpec::new(model_id, tenant_id).memory(memory_id, "private-conversation"),
    )?;

    let error = runtime
        .protected_snapshot(&XorProtector(0x33))
        .err()
        .context("metadata-only snapshot must not drop memory behavior")?;

    assert!(matches!(
        error,
        SnapshotError::MetadataOnlyMemory(id) if id == agent
    ));
    Ok(())
}

#[tokio::test]
async fn persistable_store_replacement_cleanup_roundtrips() -> Result<()> {
    let tenant_id = TenantId::new();
    let model_identity = BindingIdentity::new("test-mock-model", "store-snapshot");
    let old_identity = BindingIdentity::new("fixture-store", "v1");
    let new_identity = BindingIdentity::new("fixture-store", "v2");
    let model = MockCompletionModel::text("done");
    let mut runtime = runtime()?;
    let model_id = runtime.register_persistable_model(tenant_id, model_identity, model.clone());
    let agent = runtime.spawn_agent_spec(AgentSpec::new(model_id, tenant_id))?;
    let old = runtime.install_persistable_vector_store(
        agent,
        old_identity,
        FixtureVectorStore::default(),
    )?;

    let kind_error = runtime
        .replace_tool(old.grant_id, CountingPortableTool::default())
        .err()
        .context("a generic tool replacement must not preserve a false store kind")?;
    assert!(matches!(
        kind_error,
        RuntimeError::CapabilityKindMismatch {
            capability_id,
            expected: CapabilityKind::Tool,
            actual: CapabilityKind::Store,
        } if capability_id == old.capability_id
    ));

    let identity_error = runtime
        .replace_vector_store(old.grant_id, FixtureVectorStore::default())
        .err()
        .context("persistable replacement must require a new identity")?;
    assert!(matches!(
        identity_error,
        RuntimeError::PersistenceIdentityRequired(id) if id == old.capability_id
    ));

    let replacement_store = FixtureVectorStore::default();
    let replacement = runtime.replace_persistable_vector_store(
        old.grant_id,
        new_identity,
        replacement_store.clone(),
    )?;
    let _ = runtime.run(agent, "cleanup retired store").await?;
    assert!(matches!(
        runtime.capability_kind(old.capability_id),
        Err(RuntimeError::UnknownCapability(id)) if id == old.capability_id
    ));
    assert!(matches!(
        runtime.tool_grant(old.grant_id),
        Err(RuntimeError::UnknownGrant(id)) if id == old.grant_id
    ));

    let protector = XorProtector(0x34);
    let snapshot = runtime.protected_snapshot(&protector)?;
    let mut wrong_kind_bindings = RebindRegistry::new();
    wrong_kind_bindings.bind_model(model_id, tenant_id, model_identity, model.clone());
    wrong_kind_bindings.bind_tool(
        replacement.capability_id,
        new_identity,
        replacement_store.clone(),
    );
    let wrong_kind = LocalRuntime::restore(
        RuntimeConfig::default(),
        &snapshot,
        &protector,
        wrong_kind_bindings,
    )
    .err()
    .context("a store snapshot must reject a generic-tool rebinding")?;
    assert!(matches!(
        wrong_kind,
        RuntimeError::Snapshot(SnapshotError::BindingMismatch {
            kind: "capability",
            ..
        })
    ));

    let mut bindings = RebindRegistry::new();
    bindings.bind_model(model_id, tenant_id, model_identity, model);
    bindings.bind_vector_store(replacement.capability_id, new_identity, replacement_store);
    let restored =
        LocalRuntime::restore(RuntimeConfig::default(), &snapshot, &protector, bindings)?;
    assert_eq!(
        restored.capability_kind(replacement.capability_id)?,
        CapabilityKind::Store
    );
    Ok(())
}

#[test]
fn persistable_dynamic_tool_roundtrips() -> Result<()> {
    let tenant_id = TenantId::new();
    let model_identity = BindingIdentity::new("test-mock-model", "dynamic-snapshot");
    let tool_identity = BindingIdentity::new("dynamic-echo", "v1");
    let dynamic = PortableDynamicTool::new(
        "dynamic_echo",
        "echo dynamic input",
        serde_json::json!({"type":"object"}),
        |_arguments| Box::pin(async { Ok(ToolOutput::text("dynamic")) }),
    );
    let model = MockCompletionModel::text("unused");
    let mut runtime = runtime()?;
    let model_id = runtime.register_persistable_model(tenant_id, model_identity, model.clone());
    let agent = runtime.spawn_agent_spec(AgentSpec::new(model_id, tenant_id))?;
    let grant = runtime.install_persistable_dynamic_tool(agent, tool_identity, dynamic.clone())?;
    let protector = XorProtector(0x36);
    let snapshot = runtime.protected_snapshot(&protector)?;
    let mut bindings = RebindRegistry::new();
    bindings.bind_model(model_id, tenant_id, model_identity, model);
    bindings.bind_dynamic_tool(grant.capability_id, tool_identity, dynamic);

    let restored =
        LocalRuntime::restore(RuntimeConfig::default(), &snapshot, &protector, bindings)?;

    assert_eq!(
        restored.capability_kind(grant.capability_id)?,
        CapabilityKind::Tool
    );
    Ok(())
}

#[tokio::test]
async fn restored_runtime_tick_preserves_terminal_retention_age() -> Result<()> {
    let config = RuntimeConfig {
        terminal_retention_ticks: 2,
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    };
    let tenant_id = TenantId::new();
    let model_identity = BindingIdentity::new("test-mock-model", "tick-snapshot");
    let model = MockCompletionModel::text("done");
    let mut runtime = LocalRuntime::with_config(config.clone())?;
    let model_id = runtime.register_persistable_model(tenant_id, model_identity, model.clone());
    let agent = runtime.spawn_agent_spec(AgentSpec::new(model_id, tenant_id))?;
    let handle = runtime.start_run(agent, "tick")?;

    loop {
        match runtime.step(handle).await? {
            RunStepStatus::Terminal => break,
            RunStepStatus::Progressed | RunStepStatus::EffectProgressed => {}
            RunStepStatus::Quiescent => runtime.wait_for_effect().await?,
        }
    }
    // Every Terminal return renews the observed lease, so active polling far
    // beyond the retention window never loses the run.
    for _ in 0..12 {
        assert!(matches!(
            runtime.step(handle).await?,
            RunStepStatus::Terminal
        ));
    }
    let result = runtime.finish_run(handle)?;
    let protector = XorProtector(0x35);
    let snapshot = runtime
        .protected_snapshot_with_policy(&protector, SnapshotContentPolicy::CanonicalRunState)?;
    let mut bindings = RebindRegistry::new();
    bindings.bind_model(model_id, tenant_id, model_identity, model);
    let mut restored = LocalRuntime::restore(config, &snapshot, &protector, bindings)?;
    let restored_handle = restored.run_handle(result.run_id, tenant_id)?;
    assert!(matches!(
        restored.step(restored_handle).await?,
        RunStepStatus::Terminal
    ));

    // Once its own handle stops being stepped, the observed run ages from its
    // preserved last observation while another handle pumps the schedule.
    let pump = restored.start_run(agent, "pump")?;
    let _ = restored.drive_to_terminal(pump).await?;
    for _ in 0..4 {
        let _ = restored.step(pump).await;
    }
    assert!(matches!(
        restored.run_handle(result.run_id, tenant_id),
        Err(RuntimeError::UnknownRun(id)) if id == result.run_id
    ));
    Ok(())
}

#[test]
fn snapshots_reject_ephemeral_binding_claims() -> Result<()> {
    let mut runtime = runtime()?;
    let _agent = runtime.spawn_agent(
        MockCompletionModel::text("unused")
            .into_bevy_agent_builder()
            .build(),
    )?;

    let error = runtime
        .protected_snapshot(&XorProtector(0x5A))
        .err()
        .context("ephemeral binding must not claim restart compatibility")?;

    assert!(matches!(
        error,
        SnapshotError::MissingBindingIdentity { kind: "model", .. }
    ));
    Ok(())
}

#[tokio::test]
async fn debug_surfaces_are_redacted_while_typed_provider_details_remain_inspectable() -> Result<()>
{
    let secret_prompt = "prompt-secret-4b6579";
    let secret_output = "output-secret-536563";
    let secret_provider_body = r#"{"error":"provider-secret-50726f"}"#;
    let tenant_id = TenantId::new();
    let spec = AgentSpec::new(rig_bevy::ModelId::new(), tenant_id)
        .preamble(secret_prompt)
        .additional_params(serde_json::json!({"api_key": secret_provider_body}));
    let rendered_spec = format!("{spec:?}");
    assert!(!rendered_spec.contains(secret_prompt));
    assert!(!rendered_spec.contains(secret_provider_body));
    assert!(!rendered_spec.contains(&tenant_id.to_string()));

    let correlation_id = CorrelationId::new();
    let header = EffectHeader {
        runtime_id: rig_bevy::RuntimeId::new(),
        run_id: RunId::new(),
        operation_id: OperationId::new(),
        generation: Generation::default(),
        correlation_id,
        tenant_id,
        capability_id: None,
        grant_id: None,
        capability_revision: None,
    };
    let rendered_header = format!("{header:?}");
    assert!(!rendered_header.contains(&tenant_id.to_string()));
    assert!(!rendered_header.contains(&correlation_id.to_string()));

    let capability_id = rig_bevy::CapabilityId::new();
    let grant_id = rig_bevy::GrantId::new();
    let agent_id = rig_bevy::AgentId::new();
    let run_id = RunId::new();
    let nodes = [
        format!(
            "{:?}",
            CapabilityNode {
                id: capability_id,
                tenant_id,
                kind: CapabilityKind::Tool,
                definition: None,
                revision: 1,
                retired: false,
            }
        ),
        format!(
            "{:?}",
            GrantNode {
                id: grant_id,
                agent_id,
                capability_id,
                tenant_id,
                revoked: false,
            }
        ),
        format!(
            "{:?}",
            RunHandle {
                runtime_id: rig_bevy::RuntimeId::new(),
                run_id,
                generation: Generation::default(),
                tenant_id,
            }
        ),
    ];
    assert!(
        nodes
            .iter()
            .all(|debug| !debug.contains(&tenant_id.to_string()))
    );
    let tenant_error = RuntimeError::TenantMismatch {
        expected: tenant_id,
        actual: TenantId::new(),
    };
    assert!(!format!("{tenant_error}").contains(&tenant_id.to_string()));
    assert!(!format!("{tenant_error:?}").contains(&tenant_id.to_string()));

    let provider =
        ModelEffectError::Provider(CompletionError::from_provider_body(secret_provider_body));
    assert_eq!(
        provider.provider_response_body(),
        Some(secret_provider_body)
    );
    assert!(!format!("{provider}").contains(secret_provider_body));
    assert!(!format!("{provider:?}").contains(secret_provider_body));
    let runtime_error = RuntimeError::ModelEffect(Arc::new(provider));
    assert_eq!(
        runtime_error.provider_response_json()?,
        Some(serde_json::json!({"error":"provider-secret-50726f"}))
    );
    assert!(!format!("{runtime_error:?}").contains(secret_provider_body));

    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        MockCompletionModel::text(secret_output)
            .into_bevy_agent_builder()
            .preamble(secret_prompt)
            .build(),
    )?;
    let result = runtime.run(agent, secret_prompt).await?;
    let rendered_result = format!("{result:?}");
    assert!(!rendered_result.contains(secret_prompt));
    assert!(!rendered_result.contains(secret_output));
    Ok(())
}

#[tokio::test]
async fn hosted_driver_allows_cancellation_between_schedule_passes() -> Result<()> {
    let model = MockCompletionModel::text("must not commit");
    let mut local = runtime()?;
    let agent = local.spawn_agent(model.into_bevy_agent_builder().build())?;
    let hosted = HostedRuntime::new(local);
    let handle = hosted.start_run(agent, "cancel hosted").await?;

    let _ = hosted.step(handle).await?;
    hosted.cancel(handle).await?;
    let result = hosted.drive_to_terminal(handle).await?;

    assert!(matches!(result.terminal_reason, TerminalReason::Cancelled));
    assert_eq!(result.transcript, [Message::user("cancel hosted")]);
    assert!(result.raw_final::<MockResponse>().is_none());
    Ok(())
}

#[tokio::test]
async fn hosted_provider_diagnostic_is_typed_only_by_name_and_contains_no_content() -> Result<()> {
    let model = MockCompletionModel::text("sensitive provider content");
    let mut local = runtime()?;
    let agent = local.spawn_agent(model.into_bevy_agent_builder().build())?;
    let hosted = HostedRuntime::new(local);
    let handle = hosted.start_run(agent, "sensitive prompt").await?;
    let result = hosted.drive_to_terminal(handle).await?;
    let diagnostic = hosted
        .provider_diagnostic(handle)
        .await?
        .context("hosted provider diagnostic")?;

    assert!(diagnostic.provider_type.contains("MockResponse"));
    let rendered = format!("{diagnostic:?}");
    assert!(!rendered.contains("sensitive provider content"));
    assert!(!rendered.contains("sensitive prompt"));
    assert_eq!(result.text.as_deref(), Some("sensitive provider content"));
    Ok(())
}

#[tokio::test]
async fn tool_mode_best_effort_commits_without_orphan_tool_calls() -> Result<()> {
    let output_name = synthetic_output_tool_name::<StructuredAnswer>(&[]);
    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "first",
            output_name.clone(),
            serde_json::json!({"wrong": 1}),
        ),
        MockTurn::from_contents([
            AssistantContent::text("nearly"),
            AssistantContent::ToolCall(ToolCall::new(
                "peer".to_string(),
                ToolFunction::new("unrelated_tool".to_string(), serde_json::json!({})),
            )),
            AssistantContent::ToolCall(ToolCall::new(
                "second".to_string(),
                ToolFunction::new(output_name.clone(), serde_json::json!({"still": 2})),
            )),
        ])?,
    ]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .tool_choice(ToolChoice::Specific {
                function_names: vec![output_name.clone()],
            })
            .structured_output::<StructuredAnswer>(StructuredOutputPolicy {
                mode: OutputMode::Tool,
                max_retries: 1,
                best_effort: true,
            })
            .build(),
    )?;

    let result = runtime.run(agent, "best effort").await?;

    assert_eq!(model.request_count(), 2);
    assert!(matches!(result.terminal_reason, TerminalReason::Completed));
    assert!(result.structured_output.is_none());
    assert_no_orphan_tool_use(&result.transcript);
    let last_assistant = result
        .transcript
        .iter()
        .rev()
        .find_map(|message| match message {
            Message::Assistant { content, .. } => Some(content),
            _ => None,
        })
        .context("best-effort assistant turn")?;
    assert!(
        last_assistant
            .iter()
            .all(|item| !matches!(item, AssistantContent::ToolCall(_)))
    );
    assert_eq!(result.text.as_deref(), Some(r#"nearly{"still":2}"#));
    Ok(())
}

#[tokio::test]
async fn tool_mode_best_effort_snapshot_restores_and_rebinds_handles() -> Result<()> {
    let output_name = synthetic_output_tool_name::<StructuredAnswer>(&[]);
    let model = MockCompletionModel::new([MockTurn::tool_call(
        "only",
        output_name.clone(),
        serde_json::json!({"invalid": true}),
    )]);
    let tenant_id = TenantId::new();
    let binding_identity = BindingIdentity::new("test-mock-model", "best-effort/v1");
    let mut runtime = runtime()?;
    let model_id = runtime.register_persistable_model(tenant_id, binding_identity, model.clone());
    let agent = runtime.spawn_agent_spec(
        AgentSpec::new(model_id, tenant_id)
            .tool_choice(ToolChoice::Specific {
                function_names: vec![output_name.clone()],
            })
            .structured_output::<StructuredAnswer>(StructuredOutputPolicy {
                mode: OutputMode::Tool,
                max_retries: 0,
                best_effort: true,
            }),
    )?;
    let result = runtime.run(agent, "persist best effort").await?;
    assert!(matches!(result.terminal_reason, TerminalReason::Completed));

    let protector = XorProtector(0x5A);
    let snapshot = runtime
        .protected_snapshot_with_policy(&protector, SnapshotContentPolicy::CanonicalRunState)?;
    let mut bindings = RebindRegistry::new();
    bindings.bind_model(
        model_id,
        tenant_id,
        BindingIdentity::new("test-mock-model", "best-effort/v1"),
        model,
    );
    let mut restored =
        LocalRuntime::restore(RuntimeConfig::default(), &snapshot, &protector, bindings)?;

    let wrong_tenant = restored
        .run_handle(result.run_id, TenantId::new())
        .err()
        .context("foreign tenant must not receive a restored handle")?;
    assert!(matches!(wrong_tenant, RuntimeError::TenantMismatch { .. }));
    let unknown = restored
        .run_handle(RunId::new(), tenant_id)
        .err()
        .context("unknown runs must not receive a handle")?;
    assert!(matches!(unknown, RuntimeError::UnknownRun(_)));

    let restored_handle = restored.run_handle(result.run_id, tenant_id)?;
    let restored_result = restored.finish_run(restored_handle)?;
    assert_no_orphan_tool_use(&restored_result.transcript);
    assert_eq!(restored_result.transcript.len(), result.transcript.len());
    assert_eq!(restored_result.text, result.text);
    Ok(())
}

#[tokio::test]
async fn memory_after_tool_mode_best_effort_yields_a_valid_next_run() -> Result<()> {
    let output_name = synthetic_output_tool_name::<StructuredAnswer>(&[]);
    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "first",
            output_name.clone(),
            serde_json::json!({"wrong": 1}),
        ),
        MockTurn::text(r#"{"answer":"valid"}"#),
    ]);
    let memory = CountingMemory::default();
    let mut runtime = runtime()?;
    let tenant_id = TenantId::new();
    let memory_id = runtime.register_memory(tenant_id, memory.clone());
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .tenant(tenant_id)
            .memory(memory_id, "best-effort")
            .tool_choice(ToolChoice::Specific {
                function_names: vec![output_name.clone()],
            })
            .structured_output::<StructuredAnswer>(StructuredOutputPolicy {
                mode: OutputMode::Tool,
                max_retries: 0,
                best_effort: true,
            })
            .build(),
    )?;

    let first = runtime.run(agent, "first question").await?;
    assert!(matches!(first.terminal_reason, TerminalReason::Completed));
    let second = runtime.run(agent, "second question").await?;
    assert!(matches!(second.terminal_reason, TerminalReason::Completed));

    let second_request = model.requests().pop().context("second model request")?;
    assert!(second_request.chat_history.len() > 1);
    assert!(second_request.chat_history.iter().all(|message| {
        match message {
            Message::Assistant { content, .. } => content
                .iter()
                .all(|item| !matches!(item, AssistantContent::ToolCall(_))),
            _ => true,
        }
    }));
    Ok(())
}

#[tokio::test]
async fn duplicate_tool_call_identities_fail_before_any_tool_executes() -> Result<()> {
    let model = MockCompletionModel::new([MockTurn::from_contents([
        AssistantContent::ToolCall(ToolCall::new(
            "dup".to_string(),
            ToolFunction::new(
                "counting_portable_tool".to_string(),
                serde_json::json!({"value": "a"}),
            ),
        )),
        AssistantContent::ToolCall(ToolCall::new(
            "dup".to_string(),
            ToolFunction::new(
                "counting_portable_tool".to_string(),
                serde_json::json!({"value": "b"}),
            ),
        )),
    ])?]);
    let tool = CountingPortableTool::default();
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(model.into_bevy_agent_builder().build())?;
    runtime.install_tool(agent, tool.clone())?;

    let error = runtime
        .run(agent, "dup")
        .await
        .err()
        .context("duplicate tool call identities must fail the run")?;

    assert!(matches!(
        error,
        RuntimeError::RunFailed { ref code, .. } if code == "duplicate_tool_call"
    ));
    assert!(tool.calls().is_empty());
    Ok(())
}

#[tokio::test]
async fn recovery_feedback_clears_after_an_accepted_response() -> Result<()> {
    let model = MockCompletionModel::new([
        MockTurn::text(""),
        MockTurn::tool_call(
            "call-1",
            "counting_portable_tool",
            serde_json::json!({"value": "recovered"}),
        ),
        MockTurn::text("final answer"),
    ]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .response_retry_policy(ResponseRetryPolicy::RejectEmpty { max_retries: 1 })
            .build(),
    )?;
    runtime.install_tool(agent, CountingPortableTool::default())?;

    let result = runtime.run(agent, "question").await?;

    assert_eq!(result.text.as_deref(), Some("final answer"));
    let requests = model.requests();
    assert_eq!(requests.len(), 3);
    let contains_feedback = |request: &CompletionRequest| {
        request.chat_history.iter().any(|message| match message {
            Message::User { content } => content.iter().any(|item| {
                matches!(
                    item,
                    UserContent::Text(text) if text.text.contains("previous response was empty")
                )
            }),
            _ => false,
        })
    };
    assert!(contains_feedback(&requests[1]));
    assert!(!contains_feedback(&requests[2]));
    Ok(())
}

#[tokio::test]
async fn local_deferred_effect_wakes_when_a_semaphore_owner_is_cancelled() -> Result<()> {
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        max_effects: 1,
        max_model_calls: 1,
        effect_timeout: Duration::from_secs(5),
        ..RuntimeConfig::default()
    })?;
    let blocking_agent = runtime.spawn_agent(
        SlowMock {
            inner: MockCompletionModel::text("blocking"),
            delay: Duration::from_secs(30),
        }
        .into_bevy_agent_builder()
        .build(),
    )?;
    let target_agent = runtime.spawn_agent(
        MockCompletionModel::text("target")
            .into_bevy_agent_builder()
            .build(),
    )?;
    let blocking = runtime.start_run(blocking_agent, "blocking")?;
    let _ = runtime.step(blocking).await?;
    let target = runtime.start_run(target_agent, "deferred")?;
    let _ = runtime.step(target).await?;
    runtime.cancel(blocking)?;

    let result = tokio::time::timeout(
        Duration::from_millis(500),
        runtime.drive_to_terminal(target),
    )
    .await
    .context("deferred local run must wake when execution capacity is released")??;

    assert!(matches!(result.terminal_reason, TerminalReason::Completed));
    assert_eq!(result.text.as_deref(), Some("target"));
    let blocker = runtime.finish_run(blocking)?;
    assert!(matches!(blocker.terminal_reason, TerminalReason::Cancelled));
    Ok(())
}

#[tokio::test]
async fn local_streaming_deferred_effect_wakes_when_a_semaphore_owner_is_cancelled() -> Result<()> {
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        max_effects: 1,
        max_model_calls: 1,
        effect_timeout: Duration::from_secs(5),
        ..RuntimeConfig::default()
    })?;
    let blocking_agent = runtime.spawn_agent(
        SlowMock {
            inner: MockCompletionModel::text("blocking"),
            delay: Duration::from_secs(30),
        }
        .into_bevy_agent_builder()
        .build(),
    )?;
    let target_agent = runtime.spawn_agent(
        MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("target"),
            MockStreamEvent::final_response_with_total_tokens(1),
        ]])
        .into_bevy_agent_builder()
        .streaming(StreamingMode::Streaming)
        .build(),
    )?;
    let blocking = runtime.start_run(blocking_agent, "blocking")?;
    let _ = runtime.step(blocking).await?;
    runtime.cancel(blocking)?;

    let mut stream = runtime.start_streaming::<MockResponse>(target_agent, "deferred")?;
    tokio::time::timeout(Duration::from_millis(500), async {
        while stream.next_event().await?.is_some() {}
        Ok::<_, RuntimeError>(())
    })
    .await
    .context("deferred streaming run must wake when execution capacity is released")??;
    let result = stream.finish()?;

    assert_eq!(result.text.as_deref(), Some("target"));
    Ok(())
}

#[tokio::test]
async fn abandoned_terminal_runs_are_cleaned_after_unobserved_retention() -> Result<()> {
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        terminal_retention_ticks: 64,
        unobserved_terminal_retention_ticks: 4,
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    })?;
    let abandoned_agent = runtime.spawn_agent(
        SlowMock {
            inner: MockCompletionModel::text("abandoned"),
            delay: Duration::from_secs(30),
        }
        .into_bevy_agent_builder()
        .build(),
    )?;
    let pump_agent = runtime.spawn_agent(
        MockCompletionModel::text("pump")
            .into_bevy_agent_builder()
            .build(),
    )?;
    // The abandoned run goes terminal through cancellation processed during the
    // pump run's schedule passes — its own handle is never stepped, so it stays
    // unobserved and ages on the unobserved clock.
    let abandoned = runtime.start_run(abandoned_agent, "abandoned")?;
    runtime.cancel(abandoned)?;
    let pump = runtime.start_run(pump_agent, "pump")?;

    let mut cleaned = false;
    for _ in 0..32 {
        match runtime.step(pump).await {
            Ok(_) => {}
            Err(RuntimeError::UnknownRun(_)) => break,
            Err(error) => return Err(error.into()),
        }
        // Existence probe only: `finish_run` would itself observe the run.
        if matches!(
            runtime.run_handle(abandoned.run_id, abandoned.tenant_id),
            Err(RuntimeError::UnknownRun(_))
        ) {
            cleaned = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }

    assert!(
        cleaned,
        "unobserved terminal run must be cleaned by retention"
    );
    Ok(())
}

#[derive(Clone)]
struct FixedHistoryMemory(Vec<Message>);

impl ConversationMemory for FixedHistoryMemory {
    fn load<'a>(
        &'a self,
        _conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        let messages = self.0.clone();
        Box::pin(async move { Ok(messages) })
    }

    fn append<'a>(
        &'a self,
        _conversation_id: &'a str,
        _messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move { Ok(()) })
    }

    fn clear<'a>(
        &'a self,
        _conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move { Ok(()) })
    }
}

#[tokio::test]
async fn non_canonical_memory_history_fails_before_any_model_call() -> Result<()> {
    let bad_histories = [
        // A sliding-window cut that orphaned a tool result.
        vec![
            Message::user("window start"),
            Message::tool_result("orphan", "2"),
        ],
        // A summarizer that emitted consecutive assistant messages.
        vec![
            Message::user("question"),
            Message::assistant("summary part one"),
            Message::assistant("summary part two"),
        ],
    ];
    for history in bad_histories {
        let model = MockCompletionModel::text("never called");
        let memory = FixedHistoryMemory(history);
        let mut runtime = runtime()?;
        let tenant_id = TenantId::new();
        let memory_id = runtime.register_memory(tenant_id, memory);
        let agent = runtime.spawn_agent(
            model
                .clone()
                .into_bevy_agent_builder()
                .tenant(tenant_id)
                .memory(memory_id, "windowed")
                .build(),
        )?;

        let error = runtime
            .run(agent, "prompt")
            .await
            .err()
            .context("non-canonical loaded history must fail the run")?;

        assert!(matches!(
            error,
            RuntimeError::RunFailed { ref code, .. } if code == "memory_history"
        ));
        assert_eq!(model.request_count(), 0);
    }
    Ok(())
}

#[tokio::test]
async fn assistant_role_prompts_are_rejected_at_start() -> Result<()> {
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        MockCompletionModel::text("unused")
            .into_bevy_agent_builder()
            .build(),
    )?;

    let error = runtime
        .start_run(agent, Message::assistant("prefill"))
        .err()
        .context("assistant prompts must be rejected before run creation")?;

    assert!(matches!(error, RuntimeError::InvalidPrompt { .. }));
    Ok(())
}

#[tokio::test]
async fn memory_history_failure_still_snapshots_and_restores() -> Result<()> {
    let model = MockCompletionModel::text("never called");
    let memory = FixedHistoryMemory(vec![Message::tool_result("orphan", "2")]);
    let tenant_id = TenantId::new();
    let model_identity = BindingIdentity::new("test-mock-model", "memory-history/v1");
    let memory_identity = BindingIdentity::new("fixed-history-memory", "memory-history/v1");
    let mut runtime = runtime()?;
    let model_id = runtime.register_persistable_model(tenant_id, model_identity, model.clone());
    let memory_id = runtime.register_persistable_memory(tenant_id, memory_identity, memory.clone());
    let agent = runtime
        .spawn_agent_spec(AgentSpec::new(model_id, tenant_id).memory(memory_id, "windowed"))?;
    let handle = runtime.start_run(agent, "prompt")?;
    let result = runtime.drive_to_terminal(handle).await?;
    assert!(matches!(
        result.terminal_reason,
        TerminalReason::Failed { ref code } if code == "memory_history"
    ));
    assert_eq!(model.request_count(), 0);

    // The rejected history never entered the transcript, so the failed run
    // must round-trip through snapshot and restore.
    let protector = XorProtector(0x66);
    let snapshot = runtime
        .protected_snapshot_with_policy(&protector, SnapshotContentPolicy::CanonicalRunState)?;
    let mut bindings = RebindRegistry::new();
    bindings.bind_model(
        model_id,
        tenant_id,
        BindingIdentity::new("test-mock-model", "memory-history/v1"),
        model,
    );
    bindings.bind_memory(
        memory_id,
        tenant_id,
        BindingIdentity::new("fixed-history-memory", "memory-history/v1"),
        memory,
    );
    let mut restored =
        LocalRuntime::restore(RuntimeConfig::default(), &snapshot, &protector, bindings)?;
    let restored_handle = restored.run_handle(handle.run_id, tenant_id)?;
    let restored_result = restored.finish_run(restored_handle)?;
    assert!(matches!(
        restored_result.terminal_reason,
        TerminalReason::Failed { ref code } if code == "memory_history"
    ));
    Ok(())
}

#[tokio::test]
async fn multi_turn_tool_runs_commit_under_tail_validation() -> Result<()> {
    let model = MockCompletionModel::new([
        MockTurn::tool_call(
            "a",
            "counting_portable_tool",
            serde_json::json!({"value": "1"}),
        ),
        MockTurn::tool_call(
            "b",
            "counting_portable_tool",
            serde_json::json!({"value": "2"}),
        ),
        MockTurn::text("done"),
    ]);
    let tool = CountingPortableTool::default();
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(model.clone().into_bevy_agent_builder().build())?;
    runtime.install_tool(agent, tool.clone())?;

    let result = runtime.run(agent, "multi turn").await?;

    assert!(matches!(result.terminal_reason, TerminalReason::Completed));
    assert_eq!(result.text.as_deref(), Some("done"));
    assert_eq!(model.request_count(), 3);
    assert_eq!(tool.calls(), ["1", "2"]);
    assert_eq!(result.transcript.len(), 6);
    assert_no_orphan_tool_use(&result.transcript);
    Ok(())
}

#[tokio::test]
async fn duplicate_tool_call_identities_recover_under_retry_policy() -> Result<()> {
    let model = MockCompletionModel::new([
        MockTurn::from_contents([
            AssistantContent::ToolCall(ToolCall::new(
                "dup".to_string(),
                ToolFunction::new(
                    "counting_portable_tool".to_string(),
                    serde_json::json!({"value": "a"}),
                ),
            )),
            AssistantContent::ToolCall(ToolCall::new(
                "dup".to_string(),
                ToolFunction::new(
                    "counting_portable_tool".to_string(),
                    serde_json::json!({"value": "b"}),
                ),
            )),
        ])?,
        MockTurn::text("clean answer"),
    ]);
    let tool = CountingPortableTool::default();
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .invalid_tool_policy(InvalidToolPolicy::Retry { max_retries: 1 })
            .build(),
    )?;
    runtime.install_tool(agent, tool.clone())?;

    let result = runtime.run(agent, "dup retry").await?;

    assert!(matches!(result.terminal_reason, TerminalReason::Completed));
    assert_eq!(result.text.as_deref(), Some("clean answer"));
    assert_eq!(model.request_count(), 2);
    assert!(tool.calls().is_empty());
    assert!(
        result
            .events
            .iter()
            .any(|event| matches!(event, RunEvent::ResponseRetried))
    );
    Ok(())
}

#[tokio::test]
async fn duplicate_tool_call_identities_deduplicate_under_skip_policy() -> Result<()> {
    let model = MockCompletionModel::new([
        MockTurn::from_contents([
            AssistantContent::ToolCall(ToolCall::new(
                "dup".to_string(),
                ToolFunction::new(
                    "counting_portable_tool".to_string(),
                    serde_json::json!({"value": "first"}),
                ),
            )),
            AssistantContent::ToolCall(ToolCall::new(
                "dup".to_string(),
                ToolFunction::new(
                    "counting_portable_tool".to_string(),
                    serde_json::json!({"value": "second"}),
                ),
            )),
        ])?,
        MockTurn::text("after tools"),
    ]);
    let tool = CountingPortableTool::default();
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .invalid_tool_policy(InvalidToolPolicy::Skip)
            .build(),
    )?;
    runtime.install_tool(agent, tool.clone())?;

    let result = runtime.run(agent, "dup skip").await?;

    assert!(matches!(result.terminal_reason, TerminalReason::Completed));
    assert_eq!(tool.calls(), ["first"]);
    assert_no_orphan_tool_use(&result.transcript);
    assert!(
        result
            .events
            .iter()
            .any(|event| matches!(event, RunEvent::ToolSuppressed { .. }))
    );
    Ok(())
}

#[tokio::test]
async fn restored_unobserved_terminal_runs_get_a_fresh_retention_lease() -> Result<()> {
    let tenant_id = TenantId::new();
    let abandoned_identity = BindingIdentity::new("slow-model", "lease/v1");
    let pump_identity = BindingIdentity::new("pump-model", "lease/v1");
    let abandoned_model = SlowMock {
        inner: MockCompletionModel::text("abandoned"),
        delay: Duration::from_secs(30),
    };
    let pump_model = MockCompletionModel::text("pump");
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        terminal_retention_ticks: 1_024,
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    })?;
    let abandoned_model_id =
        runtime.register_persistable_model(tenant_id, abandoned_identity, abandoned_model.clone());
    let pump_model_id =
        runtime.register_persistable_model(tenant_id, pump_identity, pump_model.clone());
    let abandoned_agent =
        runtime.spawn_agent_spec(AgentSpec::new(abandoned_model_id, tenant_id))?;
    let pump_agent = runtime.spawn_agent_spec(AgentSpec::new(pump_model_id, tenant_id))?;

    // Cancel the abandoned run and let the pump run's drive process it; the
    // abandoned handle itself is never stepped, so the run stays unobserved.
    let abandoned = runtime.start_run(abandoned_agent, "abandoned")?;
    runtime.cancel(abandoned)?;
    let pump = runtime.start_run(pump_agent, "pump")?;
    let _ = runtime.drive_to_terminal(pump).await?;
    // Age the unobserved run well past the restore-side window below.
    for _ in 0..30 {
        assert!(matches!(runtime.step(pump).await?, RunStepStatus::Terminal));
    }

    let protector = XorProtector(0x21);
    let snapshot = runtime
        .protected_snapshot_with_policy(&protector, SnapshotContentPolicy::CanonicalRunState)?;
    let mut bindings = RebindRegistry::new();
    bindings.bind_model(
        abandoned_model_id,
        tenant_id,
        BindingIdentity::new("slow-model", "lease/v1"),
        abandoned_model,
    );
    bindings.bind_model(
        pump_model_id,
        tenant_id,
        BindingIdentity::new("pump-model", "lease/v1"),
        pump_model,
    );
    // The restored window (16) is far below the preserved age (~30): without a
    // fresh retention lease, the first restored schedule pass would clean the
    // run this snapshot exists to preserve.
    let mut restored = LocalRuntime::restore(
        RuntimeConfig {
            unobserved_terminal_retention_ticks: 16,
            effect_timeout: Duration::from_secs(1),
            ..RuntimeConfig::default()
        },
        &snapshot,
        &protector,
        bindings,
    )?;
    let restored_pump = restored.start_run(pump_agent, "restored pump")?;
    let _ = restored.drive_to_terminal(restored_pump).await?;

    let restored_handle = restored.run_handle(abandoned.run_id, tenant_id)?;
    let restored_result = restored.finish_run(restored_handle)?;
    assert!(matches!(
        restored_result.terminal_reason,
        TerminalReason::Cancelled
    ));
    Ok(())
}

#[tokio::test]
async fn retired_tools_stop_advertising_while_references_drain() -> Result<()> {
    let model = MockCompletionModel::new([MockTurn::text("first"), MockTurn::text("second")]);
    let tool = CountingPortableTool::default();
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(model.clone().into_bevy_agent_builder().build())?;
    let grant = runtime.install_tool(agent, tool)?;

    let _ = runtime.run(agent, "before retirement").await?;
    let first_request = model
        .requests()
        .first()
        .cloned()
        .context("first model request")?;
    assert_eq!(first_request.tools.len(), 1);

    runtime.retire_tool(grant.grant_id)?;
    let _ = runtime.run(agent, "after retirement").await?;
    let second_request = model.requests().pop().context("second model request")?;
    assert!(second_request.tools.is_empty());
    // With no live turn referencing it, cleanup removes the retired
    // capability entirely.
    assert!(matches!(
        runtime.capability_kind(grant.capability_id),
        Err(RuntimeError::UnknownCapability(_))
    ));
    Ok(())
}

#[tokio::test]
async fn polling_terminal_steps_renew_the_observed_retention_lease() -> Result<()> {
    let mut runtime = LocalRuntime::with_config(RuntimeConfig {
        terminal_retention_ticks: 4,
        effect_timeout: Duration::from_secs(1),
        ..RuntimeConfig::default()
    })?;
    let polled_agent = runtime.spawn_agent(
        MockCompletionModel::text("polled")
            .into_bevy_agent_builder()
            .build(),
    )?;
    let pump_agent = runtime.spawn_agent(
        MockCompletionModel::text("pump")
            .into_bevy_agent_builder()
            .build(),
    )?;
    let polled = runtime.start_run(polled_agent, "polled")?;
    let _ = runtime.drive_to_terminal(polled).await?;
    let pump = runtime.start_run(pump_agent, "pump")?;

    // Interleaved polling far past the 4-tick window: every Terminal return
    // renews the lease, so the actively polled run is never cleaned even
    // while the pump run's passes advance the tick.
    for _ in 0..12 {
        assert!(matches!(
            runtime.step(polled).await?,
            RunStepStatus::Terminal
        ));
        let _ = runtime.step(pump).await;
    }
    let result = runtime.finish_run(polled)?;
    assert!(matches!(result.terminal_reason, TerminalReason::Completed));

    // Once polling stops, the run ages out from its last observation.
    for _ in 0..8 {
        let _ = runtime.step(pump).await;
    }
    assert!(matches!(
        runtime.finish_run(polled),
        Err(RuntimeError::UnknownRun(_))
    ));
    Ok(())
}

#[tokio::test]
async fn structured_output_turns_pass_through_the_duplicate_identity_ladder() -> Result<()> {
    let output_name = synthetic_output_tool_name::<StructuredAnswer>(&[]);
    let duplicate_turn = || {
        MockTurn::from_contents([
            AssistantContent::ToolCall(ToolCall::new(
                "dup".to_string(),
                ToolFunction::new(output_name.clone(), serde_json::json!({"answer": "first"})),
            )),
            AssistantContent::ToolCall(ToolCall::new(
                "dup".to_string(),
                ToolFunction::new(output_name.clone(), serde_json::json!({"answer": "second"})),
            )),
        ])
    };

    // Default policy: the duplicated identity fails the run even though the
    // turn carries the structured-output call.
    let model = MockCompletionModel::new([duplicate_turn()?]);
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .tool_choice(ToolChoice::Specific {
                function_names: vec![output_name.clone()],
            })
            .structured_output::<StructuredAnswer>(StructuredOutputPolicy {
                mode: OutputMode::Tool,
                max_retries: 0,
                best_effort: false,
            })
            .build(),
    )?;
    let error = runtime
        .run(agent, "dup structured")
        .await
        .err()
        .context("duplicate identities in a structured turn must fail")?;
    assert!(matches!(
        error,
        RuntimeError::RunFailed { ref code, .. } if code == "duplicate_tool_call"
    ));

    // Retry policy: the same defect recovers on a clean second response.
    let retry_model = MockCompletionModel::new([
        duplicate_turn()?,
        MockTurn::tool_call(
            "valid",
            output_name.clone(),
            serde_json::json!({"answer": "recovered"}),
        ),
    ]);
    let mut retry_runtime = LocalRuntime::new()?;
    let agent = retry_runtime.spawn_agent(
        retry_model
            .clone()
            .into_bevy_agent_builder()
            .tool_choice(ToolChoice::Specific {
                function_names: vec![output_name.clone()],
            })
            .invalid_tool_policy(InvalidToolPolicy::Retry { max_retries: 1 })
            .structured_output::<StructuredAnswer>(StructuredOutputPolicy {
                mode: OutputMode::Tool,
                max_retries: 0,
                best_effort: false,
            })
            .build(),
    )?;
    let result = retry_runtime.run(agent, "dup structured retry").await?;
    assert_eq!(
        result.structured_output,
        Some(serde_json::json!({"answer": "recovered"}))
    );
    assert_eq!(retry_model.request_count(), 2);
    Ok(())
}

#[tokio::test]
async fn mixed_tool_result_and_text_histories_are_accepted() -> Result<()> {
    let history = vec![
        Message::user("what is 1+1"),
        Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                "c1".to_string(),
                ToolFunction::new("calculator".to_string(), serde_json::json!({"sum": "1+1"})),
            ))),
        },
        Message::User {
            content: OneOrMany::many(vec![
                UserContent::ToolResult(ToolResult {
                    id: "c1".to_string(),
                    call_id: None,
                    content: OneOrMany::one(ToolResultContent::text("2")),
                }),
                UserContent::text("thanks, and please be brief"),
            ])?,
        },
        Message::assistant("It is 2."),
    ];
    let model = MockCompletionModel::text("answered");
    let memory = FixedHistoryMemory(history);
    let mut runtime = runtime()?;
    let tenant_id = TenantId::new();
    let memory_id = runtime.register_memory(tenant_id, memory);
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .tenant(tenant_id)
            .memory(memory_id, "classic-history")
            .build(),
    )?;

    let result = runtime.run(agent, "next question").await?;

    assert!(matches!(result.terminal_reason, TerminalReason::Completed));
    assert_eq!(model.request_count(), 1);
    let request = model.requests().pop().context("model request")?;
    assert_eq!(request.chat_history.len(), 5);
    Ok(())
}

#[tokio::test]
async fn duplicate_and_invalid_name_retries_share_one_budget() -> Result<()> {
    let model = MockCompletionModel::new([
        MockTurn::from_contents([
            AssistantContent::ToolCall(ToolCall::new(
                "dup".to_string(),
                ToolFunction::new(
                    "counting_portable_tool".to_string(),
                    serde_json::json!({"value": "a"}),
                ),
            )),
            AssistantContent::ToolCall(ToolCall::new(
                "dup".to_string(),
                ToolFunction::new(
                    "counting_portable_tool".to_string(),
                    serde_json::json!({"value": "b"}),
                ),
            )),
        ])?,
        MockTurn::tool_call("solo", "unadvertised_tool", serde_json::json!({})),
    ]);
    let tool = CountingPortableTool::default();
    let mut runtime = runtime()?;
    let agent = runtime.spawn_agent(
        model
            .clone()
            .into_bevy_agent_builder()
            .invalid_tool_policy(InvalidToolPolicy::Retry { max_retries: 1 })
            .build(),
    )?;
    runtime.install_tool(agent, tool.clone())?;

    let error = runtime
        .run(agent, "shared budget")
        .await
        .err()
        .context("second defect must exhaust the shared retry budget")?;

    assert!(matches!(
        error,
        RuntimeError::RunFailed { ref code, .. } if code == "invalid_tool_retry_exhausted"
    ));
    assert_eq!(model.request_count(), 2);
    assert!(tool.calls().is_empty());
    Ok(())
}
