//! AgentSession integration tests against the scripted Mock provider:
//! blocking runs, tool loops, invalid-call recovery, policy-surfaced
//! decision points, and suspend/resume round-trips.

use std::sync::Arc;

use rig::OneOrMany;
use rig::agent::AgentConfig;
use rig::completion::{CompletionResponse, FinishReason, Usage};
use rig::message::{AssistantContent, ToolCall, UserContent};
use rig::provider::{MockScript, ProviderConfig, Runtime};
use rig::session::{AgentSession, SessionEvent, SessionPolicy};
use rig_agent::agent::hook::{InvalidToolCallAction, ModelTurnAction};
use rig_agent::agent::prepare::ToolCatalog;
use rig_core::completion::ToolDefinition;

fn usage(total: u64) -> Usage {
    let mut usage = Usage::new();
    usage.total_tokens = total;
    usage
}

fn text_response(text: &str, total_tokens: u64) -> CompletionResponse {
    CompletionResponse::new(
        OneOrMany::one(AssistantContent::text(text)),
        usage(total_tokens),
        "mock",
    )
    .with_message_id("msg_1")
    .with_finish_reason(FinishReason::Stop)
}

fn tool_call_response(id: &str, name: &str, args: serde_json::Value) -> CompletionResponse {
    CompletionResponse::new(
        OneOrMany::one(AssistantContent::tool_call(id, name, args)),
        usage(3),
        "mock",
    )
    .with_finish_reason(FinishReason::ToolCalls)
}

fn adder_catalog() -> ToolCatalog {
    ToolCatalog::new(vec![ToolDefinition {
        name: "add".to_string(),
        description: "Adds two numbers".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"]
        }),
    }])
}

fn mock_provider(script: MockScript) -> ProviderConfig {
    ProviderConfig::Mock(script)
}

fn session(script: MockScript) -> AgentSession {
    AgentSession::new(
        AgentConfig::new(),
        mock_provider(script),
        Arc::new(Runtime::new()),
        "hello",
    )
}

fn tool_result_for(call: &ToolCall, content: &str) -> UserContent {
    UserContent::tool_result(
        call.id.clone(),
        OneOrMany::one(rig::message::ToolResultContent::text(content)),
    )
}

#[tokio::test]
async fn tool_less_run_returns_scripted_output() {
    let script = MockScript::from_responses(vec![text_response("scripted answer", 5)]);
    let done = session(script).run().await.expect("run should succeed");
    assert_eq!(done.output, "scripted answer");
    assert_eq!(done.usage.total_tokens, 5);
}

#[tokio::test]
async fn tool_loop_round_trips_results_and_aggregates_usage() {
    let script = MockScript::from_responses(vec![
        tool_call_response("call_1", "add", serde_json::json!({"a": 1, "b": 2})),
        text_response("the sum is 3", 7),
    ]);
    let mut session = session(script).with_tools(adder_catalog());
    session.config.max_turns = Some(3);
    // Rebuild the run budget: config is read at construction time.
    let mut session = AgentSession::new(
        session.config.clone(),
        session.provider.clone(),
        Arc::new(Runtime::new()),
        "hello",
    )
    .with_tools(adder_catalog());

    let calls = match session.advance().await.expect("first advance") {
        SessionEvent::ToolCallsReady(calls) => calls,
        other => panic!("expected ToolCallsReady, got {other:?}"),
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool_call.function.name, "add");
    assert!(calls[0].preresolved_result.is_none());

    let results = vec![tool_result_for(&calls[0].tool_call, "3")];
    session
        .provide_tool_results(results)
        .expect("results should be accepted");

    let done = match session.advance().await.expect("second advance") {
        SessionEvent::Done(done) => done,
        other => panic!("expected Done, got {other:?}"),
    };
    assert_eq!(done.output, "the sum is 3");
    // Usage aggregates across both model calls (3 + 7).
    assert_eq!(done.usage.total_tokens, 10);
    assert_eq!(session.run_state().completion_calls().len(), 2);
}

#[tokio::test]
async fn invalid_tool_call_skip_flows_through_preresolved_results() {
    let script = MockScript::from_responses(vec![
        tool_call_response("call_1", "bogus", serde_json::json!({})),
        text_response("recovered", 2),
    ]);
    let mut config = AgentConfig::new();
    config.max_turns = Some(3);
    let mut session = AgentSession::new(
        config,
        mock_provider(script),
        Arc::new(Runtime::new()),
        "hello",
    )
    .with_tools(adder_catalog());

    let context = match session.advance().await.expect("first advance") {
        SessionEvent::InvalidToolCall(context) => context,
        other => panic!("expected InvalidToolCall, got {other:?}"),
    };
    assert_eq!(context.tool_name, "bogus");
    assert!(context.available_tools.contains(&"add".to_string()));

    session
        .resolve_invalid(InvalidToolCallAction::skip("not available"))
        .expect("skip should resolve");

    // The skipped call still flows through CallTools with a preresolved
    // result the host must return verbatim.
    let calls = match session.advance().await.expect("post-skip advance") {
        SessionEvent::ToolCallsReady(calls) => calls,
        other => panic!("expected ToolCallsReady, got {other:?}"),
    };
    assert_eq!(calls.len(), 1);
    let preresolved = calls[0]
        .preresolved_result
        .clone()
        .expect("skipped call should carry a preresolved result");
    session
        .provide_tool_results(vec![preresolved])
        .expect("preresolved result should be accepted");

    let done = match session.advance().await.expect("final advance") {
        SessionEvent::Done(done) => done,
        other => panic!("expected Done, got {other:?}"),
    };
    assert_eq!(done.output, "recovered");
}

#[tokio::test]
async fn surfaced_turns_can_be_retried_with_feedback() {
    let script = MockScript::from_responses(vec![
        text_response("RETRY", 1),
        text_response("final answer", 2),
    ]);
    let mut config = AgentConfig::new();
    config.max_turns = Some(2);
    let mut session = AgentSession::new(
        config,
        mock_provider(script),
        Arc::new(Runtime::new()),
        "hello",
    )
    .with_policy(SessionPolicy {
        surface_model_turns: true,
        surface_completion_calls: false,
    });

    let content = match session.advance().await.expect("first advance") {
        SessionEvent::TurnFinished { content, .. } => content,
        other => panic!("expected TurnFinished, got {other:?}"),
    };
    let text = match content.first() {
        AssistantContent::Text(text) => text.text,
        other => panic!("expected text content, got {other:?}"),
    };
    assert_eq!(text, "RETRY");
    // Observation parity: the full provider response is reachable.
    assert_eq!(
        session
            .last_response()
            .and_then(|response| response.message_id.as_deref()),
        Some("msg_1")
    );

    session
        .reply_turn(ModelTurnAction::retry_with_feedback("Answer fully."))
        .expect("retry should be accepted");

    let done = loop {
        match session.advance().await.expect("advance after retry") {
            SessionEvent::TurnFinished { .. } => {
                session
                    .reply_turn(ModelTurnAction::Continue)
                    .expect("accept the retried turn");
            }
            SessionEvent::Done(done) => break done,
            other => panic!("unexpected event {other:?}"),
        }
    };
    assert_eq!(done.output, "final answer");
}

#[tokio::test]
async fn suspend_and_resume_mid_tools_round_trips() {
    let script = MockScript::from_responses(vec![
        tool_call_response("call_1", "add", serde_json::json!({"a": 2, "b": 2})),
        text_response("resumed: 4", 4),
    ]);
    let mut config = AgentConfig::new();
    config.max_turns = Some(3);
    let mut session = AgentSession::new(
        config.clone(),
        mock_provider(script.clone()),
        Arc::new(Runtime::new()),
        "hello",
    )
    .with_tools(adder_catalog());

    let calls = match session.advance().await.expect("first advance") {
        SessionEvent::ToolCallsReady(calls) => calls,
        other => panic!("expected ToolCallsReady, got {other:?}"),
    };

    // Suspend: the run state (including pending calls) is plain serde data.
    let serialized =
        serde_json::to_string(session.run_state()).expect("run state should serialize");
    drop(session);

    let run = serde_json::from_str(&serialized).expect("run state should deserialize");
    // The mock script's clone shares its cursor, so the resumed session
    // continues from the recorded position (turn 2 next).
    let mut resumed =
        AgentSession::resume(config, mock_provider(script), Arc::new(Runtime::new()), run)
            .with_tools(adder_catalog());

    // next_step is idempotent for pending tools: the same calls re-surface.
    let resumed_calls = match resumed.advance().await.expect("resumed advance") {
        SessionEvent::ToolCallsReady(calls) => calls,
        other => panic!("expected ToolCallsReady after resume, got {other:?}"),
    };
    assert_eq!(resumed_calls[0].tool_call.id, calls[0].tool_call.id);

    resumed
        .provide_tool_results(vec![tool_result_for(&resumed_calls[0].tool_call, "4")])
        .expect("results should be accepted");
    let done = match resumed.advance().await.expect("final advance") {
        SessionEvent::Done(done) => done,
        other => panic!("expected Done, got {other:?}"),
    };
    assert_eq!(done.output, "resumed: 4");
}

#[tokio::test]
async fn run_refuses_executable_tools() {
    let script = MockScript::from_responses(vec![text_response("unused", 1)]);
    let session = session(script).with_tools(adder_catalog());
    let error = session.run().await.expect_err("run should refuse tools");
    assert!(error.to_string().contains("cannot execute tools"));
}
