//! AgentStream integration tests against the scripted Mock provider:
//! delta ordering, the announce-before-execute tool contract, and
//! terminal accounting.

use std::sync::Arc;

use rig::OneOrMany;
use rig::agent::AgentConfig;
use rig::completion::{CompletionResponse, FinishReason, Usage};
use rig::message::{AssistantContent, ToolCall, UserContent};
use rig::provider::{MockScript, ProviderConfig, Runtime};
use rig::session::SessionPolicy;
use rig::stream::{AgentStream, AgentStreamItem};
use rig_agent::agent::prepare::ToolCatalog;
use rig_core::completion::ToolDefinition;
use rig_core::streaming::StreamedAssistantContent;

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

fn tool_result_for(call: &ToolCall, content: &str) -> UserContent {
    UserContent::tool_result(
        call.id.clone(),
        OneOrMany::one(rig::message::ToolResultContent::text(content)),
    )
}

#[tokio::test]
async fn streamed_text_arrives_as_deltas_then_final() {
    let script = MockScript::from_responses(vec![text_response("hello world", 5)]).with_streams(
        vec![vec![
            StreamedAssistantContent::text("hello "),
            StreamedAssistantContent::text("world"),
        ]],
    );
    let mut stream = AgentStream::new(
        AgentConfig::new(),
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "hi",
    );

    let mut deltas = String::new();
    let mut final_output = None;
    while let Some(item) = stream.next_item().await {
        match item.expect("stream should not fail") {
            AgentStreamItem::Assistant(StreamedAssistantContent::Text(text)) => {
                deltas.push_str(&text.text);
            }
            AgentStreamItem::Final(response) => final_output = Some(response.output),
            _ => {}
        }
    }
    assert_eq!(deltas, "hello world");
    assert_eq!(final_output.as_deref(), Some("hello world"));
    assert_eq!(stream.usage().total_tokens, 5);
}

#[tokio::test]
async fn tool_calls_are_announced_before_tool_calls_ready() {
    let script = MockScript::from_responses(vec![
        tool_call_response("call_1", "add", serde_json::json!({"a": 1, "b": 2})),
        text_response("the sum is 3", 7),
    ]);
    let mut config = AgentConfig::new();
    config.max_turns = Some(3);
    let mut stream = AgentStream::new(
        config,
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "add 1 and 2",
    )
    .with_tools(adder_catalog());

    let mut saw_announce = false;
    let mut committed = 0usize;
    let mut final_output = None;
    while let Some(item) = stream.next_item().await {
        match item.expect("stream should not fail") {
            AgentStreamItem::Assistant(StreamedAssistantContent::ToolCall {
                tool_call, ..
            }) => {
                assert_eq!(tool_call.function.name, "add");
                saw_announce = true;
            }
            AgentStreamItem::ToolCallsReady(calls) => {
                // Announce-before-execute: the complete call item precedes
                // the execution request.
                assert!(saw_announce, "ToolCallsReady arrived unannounced");
                assert_eq!(calls.len(), 1);
                let results = vec![tool_result_for(&calls[0].tool_call, "3")];
                stream
                    .provide_tool_results(results)
                    .expect("results should be accepted");
            }
            AgentStreamItem::ToolExecutionCommitted { tool_call, .. } => {
                assert_eq!(tool_call.function.name, "add");
                committed += 1;
            }
            AgentStreamItem::Final(response) => final_output = Some(response.output),
            _ => {}
        }
    }
    assert_eq!(committed, 1);
    assert_eq!(final_output.as_deref(), Some("the sum is 3"));
    // Usage aggregates across both model calls (3 + 7).
    assert_eq!(stream.usage().total_tokens, 10);
}

#[tokio::test]
async fn policy_surfaced_turns_pause_the_stream() {
    let script = MockScript::from_responses(vec![text_response("answer", 2)]);
    let mut stream = AgentStream::new(
        AgentConfig::new(),
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "hi",
    )
    .with_policy(SessionPolicy {
        surface_model_turns: true,
        surface_completion_calls: true,
    });

    let mut saw_before_call = false;
    let mut saw_turn_finished = false;
    let mut final_output = None;
    while let Some(item) = stream.next_item().await {
        match item.expect("stream should not fail") {
            AgentStreamItem::BeforeModelCall { turn, .. } => {
                assert_eq!(turn, 1);
                saw_before_call = true;
                stream
                    .reply_before_call(
                        rig_agent::agent::hook::CompletionCallAction::Continue,
                    )
                    .await
                    .expect("continue should be accepted");
            }
            AgentStreamItem::TurnFinished { content, .. } => {
                saw_turn_finished = true;
                assert!(matches!(content.first(), AssistantContent::Text(_)));
                stream
                    .reply_turn(rig_agent::agent::hook::ModelTurnAction::Continue)
                    .expect("accept should be accepted");
            }
            AgentStreamItem::Final(response) => final_output = Some(response.output),
            _ => {}
        }
    }
    assert!(saw_before_call);
    assert!(saw_turn_finished);
    assert_eq!(final_output.as_deref(), Some("answer"));
}
