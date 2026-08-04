//! AgentStream integration tests against the scripted Mock provider:
//! delta ordering, the announce-before-execute tool contract, and
//! terminal accounting.

use std::sync::Arc;

use rig::OneOrMany;
use rig::agent::AgentConfig;
use rig::completion::{CompletionResponse, Document, FinishReason, Usage};
use rig::message::{AssistantContent, ToolCall, UserContent};
use rig::provider::{MockScript, MockStreamError, ProviderConfig, Runtime};
use rig::session::SessionPolicy;
use rig::stream::{AgentStream, AgentStreamItem};
use rig_agent::agent::hook::{CompletionCallAction, InvalidToolCallAction, RequestPatch};
use rig_agent::agent::prepare::ToolCatalog;
use rig_agent::agent::run::{PendingToolCall, ToolResultSubmission};
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

fn submission_for(call: &PendingToolCall, result: UserContent) -> ToolResultSubmission {
    ToolResultSubmission::new(
        call.internal_call_id.clone().expect("durable internal id"),
        result,
    )
}

fn retry_patch() -> RequestPatch {
    RequestPatch::new()
        .temperature(0.25)
        .active_tools(["add"])
        .additional_params(serde_json::json!({"retained": true, "shared": "old"}))
        .history([rig::message::Message::user("retry history")])
        .context(Document {
            id: "retry-context".to_string(),
            text: "retained once".to_string(),
            additional_props: Default::default(),
        })
}

fn assert_retry_patch(request: &rig::completion::CompletionRequest) {
    assert_eq!(request.temperature, Some(0.25));
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.tools[0].name, "add");
    assert_eq!(request.documents.len(), 1);
    assert_eq!(request.documents[0].id, "retry-context");
    assert_eq!(
        request.chat_history.iter().next(),
        Some(&rig::message::Message::user("retry history"))
    );
}

#[tokio::test]
async fn streamed_text_arrives_as_deltas_then_final() {
    let script =
        MockScript::from_responses(vec![text_response("hello world", 5)]).with_streams(vec![vec![
            StreamedAssistantContent::text("hello "),
            StreamedAssistantContent::text("world"),
        ]]);
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
                let results = vec![submission_for(
                    &calls[0],
                    tool_result_for(&calls[0].tool_call, "3"),
                )];
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
async fn transient_provider_error_recovers_on_next_poll() {
    let script = MockScript::from_responses(vec![
        text_response("unused", 0),
        text_response("recovered", 5),
    ])
    .with_errors(vec![Some("boom".to_string())]);
    let mut stream = AgentStream::new(
        AgentConfig::new().with_max_turns(2),
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "hi",
    );

    let error = stream
        .next_item()
        .await
        .expect("stream should not end")
        .expect_err("first poll must surface the provider error");
    assert!(error.to_string().contains("boom"));

    // The run returned to the pre-call state: polling again spends the second
    // configured attempt and reissues the same logical turn.
    let mut final_output = None;
    while let Some(item) = stream.next_item().await {
        if let AgentStreamItem::Final(response) = item.expect("no further errors") {
            final_output = Some(response.output);
        }
    }
    assert_eq!(final_output.as_deref(), Some("recovered"));
}

#[tokio::test]
async fn stream_open_failure_retains_and_merges_the_exact_request_patch() {
    let script = MockScript::from_responses(vec![
        text_response("unused", 0),
        text_response("recovered", 5),
    ])
    .with_errors(vec![Some("boom".to_string())]);
    let probe = script.clone();
    let mut stream = AgentStream::new(
        AgentConfig::new().with_max_turns(2),
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "hi",
    )
    .with_tools(adder_catalog());
    stream.patch_next_turn(retry_patch());

    stream
        .next_item()
        .await
        .expect("stream should not end")
        .expect_err("the first provider open must fail");
    stream.patch_next_turn(
        RequestPatch::new()
            .max_tokens(77)
            .additional_params(serde_json::json!({"later": true, "shared": "new"})),
    );

    while let Some(item) = stream.next_item().await {
        item.expect("retry should succeed");
    }
    let requests = probe.requests();
    assert_eq!(requests.len(), 2);
    assert_retry_patch(&requests[0]);
    assert_retry_patch(&requests[1]);
    assert_eq!(requests[0].max_tokens, None);
    assert_eq!(requests[1].max_tokens, Some(77));
    assert_eq!(
        requests[1].additional_params,
        Some(serde_json::json!({
            "retained": true,
            "later": true,
            "shared": "new"
        }))
    );
}

#[tokio::test]
async fn cancelled_stream_open_reissues_the_exact_answered_attempt() {
    let script = MockScript::from_responses(vec![
        text_response("unused", 0),
        text_response("recovered", 5),
    ])
    .with_pending(vec![true, false]);
    let probe = script.clone();
    let mut stream = AgentStream::new(
        AgentConfig::new().with_max_turns(2),
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "hi",
    )
    .with_tools(adder_catalog())
    .with_policy(SessionPolicy {
        surface_completion_calls: true,
        ..SessionPolicy::default()
    });

    assert!(matches!(
        stream.next_item().await,
        Some(Ok(AgentStreamItem::BeforeModelCall { .. }))
    ));
    tokio::time::timeout(
        std::time::Duration::from_millis(20),
        stream.reply_before_call(CompletionCallAction::Patch(retry_patch())),
    )
    .await
    .expect_err("the scripted stream-open operation remains pending");
    assert_eq!(probe.calls(), 1);
    assert_eq!(stream.run_state().messages().len(), 1);
    assert!(stream.run_state().completion_calls().is_empty());

    stream.patch_next_turn(
        RequestPatch::new()
            .max_tokens(77)
            .additional_params(serde_json::json!({"later": true, "shared": "new"})),
    );
    let mut saw_replayed_gate = false;
    let mut final_output = None;
    while let Some(item) = stream.next_item().await {
        match item.expect("cancelled open retries") {
            AgentStreamItem::BeforeModelCall { .. } => saw_replayed_gate = true,
            AgentStreamItem::Final(response) => final_output = Some(response.output),
            _ => {}
        }
    }
    assert!(
        !saw_replayed_gate,
        "answered BeforeModelCall must not replay"
    );
    assert_eq!(final_output.as_deref(), Some("recovered"));
    assert_eq!(stream.run_state().turn(), 1);

    let requests = probe.requests();
    assert_eq!(requests.len(), 2);
    assert_retry_patch(&requests[0]);
    assert_retry_patch(&requests[1]);
    assert_eq!(requests[1].max_tokens, Some(77));
    assert_eq!(
        requests[1].additional_params,
        Some(serde_json::json!({
            "retained": true,
            "later": true,
            "shared": "new"
        }))
    );
}

#[tokio::test]
async fn cancelled_next_item_reissues_the_exact_unanswered_attempt() {
    let script = MockScript::from_responses(vec![
        text_response("unused", 0),
        text_response("recovered", 5),
    ])
    .with_pending(vec![true, false]);
    let probe = script.clone();
    let mut stream = AgentStream::new(
        AgentConfig::new().with_max_turns(2),
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "hi",
    )
    .with_tools(adder_catalog());
    stream.patch_next_turn(retry_patch());

    tokio::time::timeout(std::time::Duration::from_millis(20), stream.next_item())
        .await
        .expect_err("the scripted stream-open operation remains pending");
    assert_eq!(probe.calls(), 1);
    assert_eq!(stream.run_state().messages().len(), 1);
    assert!(stream.run_state().completion_calls().is_empty());

    stream.patch_next_turn(RequestPatch::new().max_tokens(77));
    let mut final_output = None;
    while let Some(item) = stream.next_item().await {
        if let AgentStreamItem::Final(response) = item.expect("cancelled open retries") {
            final_output = Some(response.output);
        }
    }
    assert_eq!(final_output.as_deref(), Some("recovered"));
    assert_eq!(stream.run_state().turn(), 1);

    let requests = probe.requests();
    assert_eq!(requests.len(), 2);
    assert_retry_patch(&requests[0]);
    assert_retry_patch(&requests[1]);
    assert_eq!(requests[1].max_tokens, Some(77));
}

#[tokio::test]
async fn stream_preparation_failure_is_retryable_and_retains_the_patch() {
    let script = MockScript::from_responses(vec![text_response("recovered", 5)]);
    let probe = script.clone();
    let mut stream = AgentStream::new(
        AgentConfig::new().with_max_turns(2),
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "hi",
    );
    stream.patch_next_turn(retry_patch());

    let error = stream
        .next_item()
        .await
        .expect("stream should not end")
        .expect_err("active_tools must reject a tool absent from the catalog");
    assert!(error.to_string().contains("active_tools"));
    assert_eq!(probe.calls(), 0, "preparation fails before provider IO");

    stream.tools = adder_catalog();
    while let Some(item) = stream.next_item().await {
        item.expect("corrected retry should succeed");
    }
    let requests = probe.requests();
    assert_eq!(requests.len(), 1);
    assert_retry_patch(&requests[0]);
}

#[tokio::test]
async fn failed_stream_open_does_not_commit_a_provisional_output_tool() {
    let output = tool_call_response(
        "out_1",
        "final_result",
        serde_json::json!({"answer": "recovered"}),
    );
    let script = MockScript::from_responses(vec![text_response("unused", 0), output])
        .with_errors(vec![Some("boom".to_string())]);
    let mut config = AgentConfig::new().with_max_turns(2);
    config.output_schema = Some(rig_core::schemars::json_schema!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"]
    }));
    config.output_mode = rig_agent::agent::run::OutputMode::Tool;
    let mut stream = AgentStream::new(
        config,
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "hi",
    );

    stream
        .next_item()
        .await
        .expect("stream should not end")
        .expect_err("the first provider open must fail");
    assert_eq!(stream.run_state().output_tool_name(), None);

    let mut final_output = None;
    while let Some(item) = stream.next_item().await {
        if let AgentStreamItem::Final(response) = item.expect("retry should succeed") {
            final_output = Some(response.output);
        }
    }
    assert_eq!(final_output.as_deref(), Some(r#"{"answer":"recovered"}"#));
    assert_eq!(stream.run_state().output_tool_name(), Some("final_result"));
}

#[tokio::test]
async fn close_turn_after_partial_text_rolls_back_and_retries_cleanly() {
    let script = MockScript::from_responses(vec![
        text_response("unused", 9),
        text_response("recovered", 5),
    ])
    .with_streams(vec![vec![StreamedAssistantContent::text("provisional")]]);
    let mut stream = AgentStream::new(
        AgentConfig::new().with_max_turns(2),
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "hi",
    );

    assert!(matches!(
        stream.next_item().await,
        Some(Ok(AgentStreamItem::Assistant(
            StreamedAssistantContent::Text(_)
        )))
    ));
    stream.close_turn();
    assert_eq!(stream.usage().total_tokens, 0);
    assert!(stream.run_state().completion_calls().is_empty());
    assert_eq!(stream.run_state().messages().len(), 1);

    let mut final_output = None;
    while let Some(item) = stream.next_item().await {
        if let AgentStreamItem::Final(response) = item.expect("retry should succeed") {
            final_output = Some(response.output);
        }
    }
    assert_eq!(final_output.as_deref(), Some("recovered"));
    assert_eq!(stream.usage().total_tokens, 5);
    assert_eq!(stream.run_state().completion_calls().len(), 1);
}

#[tokio::test]
async fn midstream_error_discards_partial_text_and_tool_then_retries_once() {
    let script = MockScript::from_responses(vec![
        tool_call_response("call_1", "add", serde_json::json!({"a": 1, "b": 2})),
        text_response("recovered", 5),
    ])
    .with_streams(vec![vec![
        StreamedAssistantContent::text("provisional"),
        StreamedAssistantContent::ToolCall {
            tool_call: ToolCall::new(
                "call_1".to_string(),
                rig::message::ToolFunction::new(
                    "add".to_string(),
                    serde_json::json!({"a": 1, "b": 2}),
                ),
            ),
            internal_call_id: "rig_call_1".to_string(),
        },
    ]])
    .with_stream_errors(vec![Some(MockStreamError::new(2, "midstream boom"))]);
    let probe = script.clone();
    let mut stream = AgentStream::new(
        AgentConfig::new().with_max_turns(2),
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "hi",
    )
    .with_tools(adder_catalog());

    assert!(matches!(
        stream.next_item().await,
        Some(Ok(AgentStreamItem::Assistant(
            StreamedAssistantContent::Text(_)
        )))
    ));
    let error = stream
        .next_item()
        .await
        .expect("stream should not end")
        .expect_err("the injected midstream error must surface");
    assert!(error.to_string().contains("midstream boom"));
    assert_eq!(stream.usage().total_tokens, 0);
    assert!(stream.run_state().completion_calls().is_empty());
    assert_eq!(stream.run_state().messages().len(), 1);

    let mut saw_tool_gate = false;
    let mut final_output = None;
    while let Some(item) = stream.next_item().await {
        match item.expect("retry should succeed without repeating the error") {
            AgentStreamItem::ToolCallsReady(_) => saw_tool_gate = true,
            AgentStreamItem::Final(response) => final_output = Some(response.output),
            _ => {}
        }
    }
    assert!(!saw_tool_gate, "the failed attempt's tool must not survive");
    assert_eq!(final_output.as_deref(), Some("recovered"));
    assert_eq!(probe.calls(), 2);
    assert_eq!(stream.usage().total_tokens, 5);
}

async fn invalid_tool_recovery_does_not_commit_before_provider_eof(action: InvalidToolCallAction) {
    let invalid = tool_call_response("call_unknown", "unknown", serde_json::json!({"value": 1}));
    let script = MockScript::from_responses(vec![invalid, text_response("recovered", 5)])
        .with_stream_errors(vec![Some(MockStreamError::new(
            2,
            "invalid recovery drain failed",
        ))]);
    let probe = script.clone();
    let mut config = AgentConfig::new().with_max_turns(2);
    config.max_invalid_tool_call_retries = 1;
    let mut stream = AgentStream::new(
        config,
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "call a tool",
    )
    .with_tools(adder_catalog());

    assert!(matches!(
        stream.next_item().await,
        Some(Ok(AgentStreamItem::InvalidToolCall(_)))
    ));
    let error = stream
        .resolve_invalid(action)
        .await
        .expect_err("the provider error after the invalid call must surface");
    assert!(error.to_string().contains("invalid recovery drain failed"));

    // Retry/skip recovery is provisional until the abandoned provider stream
    // reaches checked EOF. The later provider error therefore leaves no
    // usage, history, completion record, or synthetic skip observation.
    assert_eq!(stream.usage().total_tokens, 0);
    assert_eq!(stream.run_state().messages().len(), 1);
    assert!(stream.run_state().completion_calls().is_empty());
    assert!(stream.last_response().is_none());

    let mut final_output = None;
    let mut saw_abandoned_user_item = false;
    let mut saw_abandoned_retry_item = false;
    while let Some(item) = stream.next_item().await {
        match item.expect("the rolled-back attempt should retry cleanly") {
            AgentStreamItem::User(_) => saw_abandoned_user_item = true,
            AgentStreamItem::ModelTurnRetried { .. } => saw_abandoned_retry_item = true,
            AgentStreamItem::Final(response) => final_output = Some(response.output),
            _ => {}
        }
    }
    assert!(!saw_abandoned_user_item);
    assert!(!saw_abandoned_retry_item);
    assert_eq!(final_output.as_deref(), Some("recovered"));
    assert_eq!(probe.calls(), 2);
    assert_eq!(stream.usage().total_tokens, 5);
    assert_eq!(stream.run_state().completion_calls().len(), 1);
}

#[tokio::test]
async fn invalid_tool_retry_rolls_back_when_the_abandoned_stream_later_fails() {
    invalid_tool_recovery_does_not_commit_before_provider_eof(InvalidToolCallAction::retry(
        "use add instead",
    ))
    .await;
}

#[tokio::test]
async fn invalid_tool_skip_rolls_back_when_the_abandoned_stream_later_fails() {
    invalid_tool_recovery_does_not_commit_before_provider_eof(InvalidToolCallAction::skip(
        "blocked by policy",
    ))
    .await;
}

#[tokio::test]
async fn provider_final_followed_by_error_commits_no_provisional_usage() {
    let provisional_final = rig_core::streaming::StreamFinal::new("mock", usage(9));
    let script = MockScript::from_responses(vec![
        text_response("unused", 9),
        text_response("recovered", 5),
    ])
    .with_streams(vec![vec![
        StreamedAssistantContent::text("provisional"),
        StreamedAssistantContent::Final(provisional_final),
    ]])
    .with_stream_errors(vec![Some(MockStreamError::new(2, "after-final boom"))]);
    let mut stream = AgentStream::new(
        AgentConfig::new().with_max_turns(2),
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "hi",
    );

    assert!(stream.next_item().await.is_some());
    assert!(stream.next_item().await.is_some());
    assert_eq!(stream.usage().total_tokens, 0);
    assert!(stream.run_state().completion_calls().is_empty());
    let error = stream
        .next_item()
        .await
        .expect("stream should not end")
        .expect_err("the post-final error must surface");
    assert!(error.to_string().contains("after-final boom"));
    assert_eq!(stream.usage().total_tokens, 0);
    assert!(stream.run_state().completion_calls().is_empty());
    assert!(
        stream.last_response().is_none(),
        "a failed attempt cannot leave provisional terminal metadata"
    );

    while let Some(item) = stream.next_item().await {
        item.expect("retry should succeed");
    }
    assert_eq!(stream.usage().total_tokens, 5);
    assert_eq!(stream.run_state().completion_calls().len(), 1);
}

#[tokio::test]
async fn eof_only_turn_does_not_inherit_an_earlier_message_id() {
    let first_final = rig_core::streaming::StreamFinal::new("mock", usage(3))
        .with_message_id("msg_first".to_string());
    let script = MockScript::from_responses(vec![tool_call_response(
        "call_1",
        "add",
        serde_json::json!({"a": 1, "b": 2}),
    )])
    .with_streams(vec![
        vec![
            StreamedAssistantContent::ToolCall {
                tool_call: ToolCall::new(
                    "call_1".to_string(),
                    rig::message::ToolFunction::new(
                        "add".to_string(),
                        serde_json::json!({"a": 1, "b": 2}),
                    ),
                ),
                internal_call_id: "rig_call_1".to_string(),
            },
            StreamedAssistantContent::Final(first_final),
        ],
        // There is deliberately no paired response/final record for turn 2.
        vec![StreamedAssistantContent::text("done")],
    ]);
    let mut config = AgentConfig::new();
    config.max_turns = Some(2);
    let mut stream = AgentStream::new(
        config,
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "hi",
    )
    .with_tools(adder_catalog())
    .with_policy(SessionPolicy {
        surface_model_turns: true,
        ..SessionPolicy::default()
    });

    let mut message_ids = Vec::new();
    while let Some(item) = stream.next_item().await {
        match item.expect("two-turn stream should succeed") {
            AgentStreamItem::TurnFinished { message_id, .. } => {
                message_ids.push(message_id);
                stream
                    .reply_turn(rig_agent::agent::hook::ModelTurnAction::Continue)
                    .expect("accept turn");
            }
            AgentStreamItem::ToolCallsReady(calls) => {
                stream
                    .provide_tool_results(vec![submission_for(
                        &calls[0],
                        tool_result_for(&calls[0].tool_call, "3"),
                    )])
                    .expect("provide tool result");
            }
            _ => {}
        }
    }

    assert_eq!(message_ids, vec![Some("msg_first".to_string()), None]);
    assert!(
        stream.last_response().is_none(),
        "a successful EOF-only turn must clear prior-turn terminal metadata"
    );
}

#[tokio::test]
async fn turn_finished_usage_is_per_turn_not_run_aggregate() {
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
    .with_tools(adder_catalog())
    .with_policy(SessionPolicy {
        surface_model_turns: true,
        surface_completion_calls: false,
        ..SessionPolicy::default()
    });

    let mut turn_usages = Vec::new();
    while let Some(item) = stream.next_item().await {
        match item.expect("stream should not fail") {
            AgentStreamItem::TurnFinished { usage, .. } => {
                turn_usages.push(usage.total_tokens);
                stream
                    .reply_turn(rig_agent::agent::hook::ModelTurnAction::Continue)
                    .expect("accept the turn");
            }
            AgentStreamItem::ToolCallsReady(calls) => {
                let results = vec![submission_for(
                    &calls[0],
                    tool_result_for(&calls[0].tool_call, "3"),
                )];
                stream
                    .provide_tool_results(results)
                    .expect("results should be accepted");
            }
            _ => {}
        }
    }
    // Each TurnFinished reports that turn's usage (3, then 7), never the
    // run aggregate (3, then 10).
    assert_eq!(turn_usages, vec![3, 7]);
    assert_eq!(stream.usage().total_tokens, 10);
}

#[tokio::test]
async fn duplicate_tool_call_ids_surface_each_result_once() {
    // Two tool calls sharing one provider id in a single turn.
    let dup_turn = CompletionResponse::new(
        OneOrMany::many(vec![
            AssistantContent::tool_call("call_1", "add", serde_json::json!({"a": 1, "b": 2})),
            AssistantContent::tool_call("call_1", "add", serde_json::json!({"a": 3, "b": 4})),
        ])
        .expect("two items"),
        usage(3),
        "mock",
    )
    .with_finish_reason(FinishReason::ToolCalls);
    let script = MockScript::from_responses(vec![dup_turn, text_response("done", 1)]);
    let mut config = AgentConfig::new();
    config.max_turns = Some(3);
    let mut stream = AgentStream::new(
        config,
        ProviderConfig::Mock(script),
        Arc::new(Runtime::new()),
        "add twice",
    )
    .with_tools(adder_catalog());

    let mut ready_internal_ids = Vec::new();
    let mut surfaced_results = Vec::new();
    while let Some(item) = stream.next_item().await {
        match item.expect("stream should not fail") {
            AgentStreamItem::ToolCallsReady(calls) => {
                assert_eq!(calls.len(), 2);
                ready_internal_ids = calls
                    .iter()
                    .map(|call| call.internal_call_id.clone().expect("durable internal id"))
                    .collect();
                // Reverse submission order: Rig identity, not the duplicated
                // provider ID or vector position, must restore call order.
                let results = vec![
                    submission_for(&calls[1], tool_result_for(&calls[1].tool_call, "7")),
                    submission_for(&calls[0], tool_result_for(&calls[0].tool_call, "3")),
                ];
                stream
                    .provide_tool_results(results)
                    .expect("results should be accepted");
            }
            AgentStreamItem::User(rig_core::streaming::StreamedUserContent::ToolResult {
                tool_result,
                internal_call_id,
            }) => {
                let text = match tool_result.content.first() {
                    rig::message::ToolResultContent::Text(text) => text.text,
                    other => panic!("expected text result, got {other:?}"),
                };
                surfaced_results.push((internal_call_id, text));
            }
            _ => {}
        }
    }
    // Multiset consumption: each result surfaces exactly once, paired
    // positionally — not the first result duplicated and the second dropped.
    assert_eq!(
        surfaced_results,
        vec![
            (ready_internal_ids[0].clone(), "3".to_string()),
            (ready_internal_ids[1].clone(), "7".to_string()),
        ]
    );
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
        ..SessionPolicy::default()
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
                    .reply_before_call(rig_agent::agent::hook::CompletionCallAction::Continue)
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
