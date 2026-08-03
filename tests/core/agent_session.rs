//! AgentSession integration tests against the scripted Mock provider:
//! blocking runs, tool loops, invalid-call recovery, policy-surfaced
//! decision points, and suspend/resume round-trips.

use std::sync::Arc;

use rig::OneOrMany;
use rig::agent::AgentConfig;
use rig::completion::{CompletionResponse, Document, FinishReason, Usage};
use rig::message::{AssistantContent, ToolCall, UserContent};
use rig::provider::{MockScript, ProviderConfig, Runtime};
use rig::session::{AgentSession, SessionEvent, SessionPolicy};
use rig_agent::agent::hook::RequestPatch;
use rig_agent::agent::hook::{CompletionCallAction, InvalidToolCallAction, ModelTurnAction};
use rig_agent::agent::prepare::ToolCatalog;
use rig_agent::agent::run::{PendingToolCall, ToolResultSubmission};
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

fn retrying_session(script: MockScript) -> AgentSession {
    AgentSession::new(
        AgentConfig::new().with_max_turns(2),
        mock_provider(script),
        Arc::new(Runtime::new()),
        "hello",
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

    let results = vec![submission_for(
        &calls[0],
        tool_result_for(&calls[0].tool_call, "3"),
    )];
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
async fn duplicate_provider_ids_use_submission_identity_when_results_arrive_reversed() {
    let duplicate_turn = CompletionResponse::new(
        OneOrMany::many(vec![
            AssistantContent::tool_call("duplicate", "add", serde_json::json!({"a": 1, "b": 2})),
            AssistantContent::tool_call("duplicate", "add", serde_json::json!({"a": 3, "b": 4})),
        ])
        .expect("two calls"),
        usage(3),
        "mock",
    )
    .with_finish_reason(FinishReason::ToolCalls);
    let mut config = AgentConfig::new();
    config.max_turns = Some(2);
    let mut session = AgentSession::new(
        config,
        mock_provider(MockScript::from_responses(vec![
            duplicate_turn,
            text_response("done", 1),
        ])),
        Arc::new(Runtime::new()),
        "add twice",
    )
    .with_tools(adder_catalog())
    .with_policy(SessionPolicy {
        surface_tool_results: true,
        ..SessionPolicy::default()
    });

    let calls = match session.advance().await.expect("tool calls") {
        SessionEvent::ToolCallsReady(calls) => calls,
        other => panic!("expected ToolCallsReady, got {other:?}"),
    };
    let internal_ids = calls
        .iter()
        .map(|call| call.internal_call_id.clone().expect("durable internal id"))
        .collect::<Vec<_>>();
    assert_ne!(internal_ids[0], internal_ids[1]);
    session
        .provide_tool_results(vec![
            submission_for(&calls[1], tool_result_for(&calls[1].tool_call, "7")),
            submission_for(&calls[0], tool_result_for(&calls[0].tool_call, "3")),
        ])
        .expect("reverse-ordered submissions are accepted");

    for (expected_internal_id, expected_text) in internal_ids.iter().zip(["3", "7"]) {
        let (internal_call_id, result) = match session.advance().await.expect("result gate") {
            SessionEvent::ToolResultReady {
                internal_call_id,
                result,
                ..
            } => (internal_call_id, result),
            other => panic!("expected ToolResultReady, got {other:?}"),
        };
        assert_eq!(&internal_call_id, expected_internal_id);
        let rendered = serde_json::to_string(&result).expect("result serializes");
        assert!(rendered.contains(expected_text), "got {rendered}");
        session
            .reply_tool_result(rig_agent::agent::hook::ToolResultAction::keep())
            .expect("keep result");
    }

    assert!(matches!(
        session.advance().await.expect("final turn"),
        SessionEvent::Done(_)
    ));
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
        .provide_tool_results(vec![submission_for(&calls[0], preresolved)])
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
        ..SessionPolicy::default()
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
        .provide_tool_results(vec![submission_for(
            &resumed_calls[0],
            tool_result_for(&resumed_calls[0].tool_call, "4"),
        )])
        .expect("results should be accepted");
    let done = match resumed.advance().await.expect("final advance") {
        SessionEvent::Done(done) => done,
        other => panic!("expected Done, got {other:?}"),
    };
    assert_eq!(done.output, "resumed: 4");
}

#[tokio::test]
async fn resume_reconstitutes_pending_invalid_tool_call() {
    let script = MockScript::from_responses(vec![
        tool_call_response("call_1", "bogus", serde_json::json!({})),
        text_response("recovered", 2),
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

    match session.advance().await.expect("first advance") {
        SessionEvent::InvalidToolCall(context) => assert_eq!(context.tool_name, "bogus"),
        other => panic!("expected InvalidToolCall, got {other:?}"),
    }

    // Suspend on the invalid-call decision itself.
    let serialized =
        serde_json::to_string(session.run_state()).expect("run state should serialize");
    drop(session);

    let run = serde_json::from_str(&serialized).expect("run state should deserialize");
    let mut resumed =
        AgentSession::resume(config, mock_provider(script), Arc::new(Runtime::new()), run)
            .with_tools(adder_catalog());

    // The pending decision is re-derived: advance re-surfaces the invalid
    // call and resolve_invalid answers it, instead of bricking the run.
    let context = match resumed.advance().await.expect("resumed advance") {
        SessionEvent::InvalidToolCall(context) => context,
        other => panic!("expected InvalidToolCall after resume, got {other:?}"),
    };
    assert_eq!(context.tool_name, "bogus");
    resumed
        .resolve_invalid(InvalidToolCallAction::skip("not available"))
        .expect("skip should resolve after resume");

    let calls = match resumed.advance().await.expect("post-skip advance") {
        SessionEvent::ToolCallsReady(calls) => calls,
        other => panic!("expected ToolCallsReady, got {other:?}"),
    };
    let preresolved = calls[0]
        .preresolved_result
        .clone()
        .expect("skipped call should carry a preresolved result");
    resumed
        .provide_tool_results(vec![submission_for(&calls[0], preresolved)])
        .expect("preresolved result should be accepted");
    let done = match resumed.advance().await.expect("final advance") {
        SessionEvent::Done(done) => done,
        other => panic!("expected Done, got {other:?}"),
    };
    assert_eq!(done.output, "recovered");
}

#[tokio::test]
async fn resume_reconstitutes_post_repair_turn_verdict_before_tools() {
    let script = MockScript::from_responses(vec![
        tool_call_response("call_1", "bogus", serde_json::json!({"a": 2, "b": 2})),
        text_response("recovered: 4", 2),
    ]);
    let mut config = AgentConfig::new();
    config.max_turns = Some(3);
    let policy = SessionPolicy {
        surface_model_turns: true,
        ..SessionPolicy::default()
    };
    let mut session = AgentSession::new(
        config.clone(),
        mock_provider(script.clone()),
        Arc::new(Runtime::new()),
        "hello",
    )
    .with_tools(adder_catalog())
    .with_policy(policy);

    assert!(matches!(
        session.advance().await.expect("surface invalid call"),
        SessionEvent::InvalidToolCall(_)
    ));
    session
        .resolve_invalid(InvalidToolCallAction::repair("add"))
        .expect("repair should produce an accepted turn");

    // Suspend in the post-resolution AcceptedModelTurn window, before the
    // host has observed or answered TurnFinished.
    let serialized =
        serde_json::to_string(session.run_state()).expect("run state should serialize");
    let run = serde_json::from_str(&serialized).expect("run state should deserialize");
    let mut resumed = AgentSession::resume(
        config.clone(),
        mock_provider(script.clone()),
        Arc::new(Runtime::new()),
        run,
    )
    .with_tools(adder_catalog())
    .with_policy(policy);

    match resumed.advance().await.expect("resurface accepted turn") {
        SessionEvent::TurnFinished { content, .. } => {
            assert!(content.iter().any(|item| {
                matches!(
                    item,
                    AssistantContent::ToolCall(call) if call.function.name == "add"
                )
            }));
        }
        other => panic!("expected TurnFinished before tools after resume, got {other:?}"),
    }
    resumed
        .reply_turn(ModelTurnAction::Continue)
        .expect("accept repaired turn");

    // Checkpoint again after answering Continue but before advancing. The
    // durable verdict state must prevent a duplicate TurnFinished on resume.
    let serialized =
        serde_json::to_string(resumed.run_state()).expect("continued run should serialize");
    let run = serde_json::from_str(&serialized).expect("continued run should deserialize");
    let mut continued =
        AgentSession::resume(config, mock_provider(script), Arc::new(Runtime::new()), run)
            .with_tools(adder_catalog())
            .with_policy(SessionPolicy {
                surface_model_turns: true,
                ..SessionPolicy::default()
            });

    let calls = match continued.advance().await.expect("advance to tools") {
        SessionEvent::ToolCallsReady(calls) => calls,
        other => panic!("expected ToolCallsReady without a duplicate verdict, got {other:?}"),
    };
    assert_eq!(calls[0].tool_call.function.name, "add");
}

#[tokio::test]
async fn transient_provider_error_recovers_on_next_advance() {
    let script = MockScript::from_responses(vec![
        text_response("unused", 0),
        text_response("recovered", 5),
    ])
    .with_errors(vec![Some("boom".to_string())]);
    let mut session = retrying_session(script);

    let error = session
        .advance()
        .await
        .expect_err("first advance must surface the provider error");
    assert!(error.to_string().contains("boom"));

    // The run returned to the pre-call state: the next advance spends the
    // second configured attempt and re-issues the same logical turn.
    let done = match session.advance().await.expect("second advance") {
        SessionEvent::Done(done) => done,
        other => panic!("expected Done, got {other:?}"),
    };
    assert_eq!(done.output, "recovered");
}

#[tokio::test]
async fn provider_failure_retains_and_merges_the_exact_request_patch() {
    let script = MockScript::from_responses(vec![
        text_response("unused", 0),
        text_response("recovered", 5),
    ])
    .with_errors(vec![Some("boom".to_string())]);
    let probe = script.clone();
    let mut session = retrying_session(script).with_tools(adder_catalog());
    session.patch_next_turn(retry_patch());

    session
        .advance()
        .await
        .expect_err("the first provider call must fail");
    session.patch_next_turn(
        RequestPatch::new()
            .max_tokens(77)
            .additional_params(serde_json::json!({"later": true, "shared": "new"})),
    );

    let done = match session.advance().await.expect("retry should succeed") {
        SessionEvent::Done(done) => done,
        other => panic!("expected Done, got {other:?}"),
    };
    assert_eq!(done.output, "recovered");

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
async fn cancelled_provider_future_reissues_the_exact_answered_attempt() {
    let script = MockScript::from_responses(vec![
        text_response("unused", 0),
        text_response("recovered", 5),
    ])
    .with_pending(vec![true, false]);
    let probe = script.clone();
    let mut session = retrying_session(script)
        .with_tools(adder_catalog())
        .with_policy(SessionPolicy {
            surface_completion_calls: true,
            ..SessionPolicy::default()
        });

    let event = session.advance().await.expect("before-call event");
    assert!(matches!(event, SessionEvent::BeforeModelCall { .. }));
    session
        .reply_before_call(CompletionCallAction::Patch(retry_patch()))
        .expect("patch decision");

    tokio::time::timeout(std::time::Duration::from_millis(20), session.advance())
        .await
        .expect_err("the scripted provider operation remains pending");
    assert_eq!(probe.calls(), 1);
    assert_eq!(session.run_state().messages().len(), 1);
    assert_eq!(session.run_state().completion_calls().len(), 0);

    session.patch_next_turn(
        RequestPatch::new()
            .max_tokens(77)
            .additional_params(serde_json::json!({"later": true, "shared": "new"})),
    );
    let done = match session.advance().await.expect("cancelled call retries") {
        SessionEvent::Done(done) => done,
        other => panic!("answered BeforeModelCall must not replay, got {other:?}"),
    };
    assert_eq!(done.output, "recovered");
    assert_eq!(session.run_state().turn(), 1);

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
async fn preparation_failure_is_retryable_and_retains_the_patch() {
    let script = MockScript::from_responses(vec![text_response("recovered", 5)]);
    let probe = script.clone();
    let mut session = retrying_session(script);
    session.patch_next_turn(retry_patch());

    let error = session
        .advance()
        .await
        .expect_err("active_tools must reject a tool absent from the catalog");
    assert!(error.to_string().contains("active_tools"));
    assert_eq!(probe.calls(), 0, "preparation fails before provider IO");

    session.tools = adder_catalog();
    let done = match session
        .advance()
        .await
        .expect("corrected retry should succeed")
    {
        SessionEvent::Done(done) => done,
        other => panic!("expected Done, got {other:?}"),
    };
    assert_eq!(done.output, "recovered");
    let requests = probe.requests();
    assert_eq!(requests.len(), 1);
    assert_retry_patch(&requests[0]);
}

#[tokio::test]
async fn failed_attempt_does_not_commit_a_provisional_output_tool() {
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
    let mut session = AgentSession::new(
        config,
        mock_provider(script),
        Arc::new(Runtime::new()),
        "hello",
    );

    session
        .advance()
        .await
        .expect_err("the first provider call must fail");
    assert_eq!(session.run_state().output_tool_name(), None);

    let done = match session.advance().await.expect("retry should succeed") {
        SessionEvent::Done(done) => done,
        other => panic!("expected Done, got {other:?}"),
    };
    assert_eq!(done.output, r#"{"answer":"recovered"}"#);
    assert_eq!(session.run_state().output_tool_name(), Some("final_result"));
}

#[tokio::test]
async fn resume_recovers_run_serialized_awaiting_model() {
    use rig_agent::agent::run::{AgentRun, AgentRunStep};

    // Hand-drive a run into AwaitingModel (a provider call in flight) and
    // serialize it there — what a crash between CallModel and its response
    // leaves behind.
    let mut run = AgentRun::new("hello").max_turns(2);
    assert!(matches!(
        run.next_step().expect("next_step"),
        AgentRunStep::CallModel { .. }
    ));
    let serialized = serde_json::to_string(&run).expect("run should serialize");

    let run: AgentRun = serde_json::from_str(&serialized).expect("run should deserialize");
    let script = MockScript::from_responses(vec![text_response("resumed", 1)]);
    let mut resumed = AgentSession::resume(
        AgentConfig::new(),
        mock_provider(script),
        Arc::new(Runtime::new()),
        run,
    );

    // resume() abandons the in-flight call so advance re-issues it.
    let done = match resumed.advance().await.expect("resumed advance") {
        SessionEvent::Done(done) => done,
        other => panic!("expected Done, got {other:?}"),
    };
    assert_eq!(done.output, "resumed");
}

#[tokio::test]
async fn unanswered_before_model_call_is_a_protocol_violation() {
    let script = MockScript::from_responses(vec![text_response("answer", 1)]);
    let mut session = session(script).with_policy(SessionPolicy {
        surface_model_turns: false,
        surface_completion_calls: true,
        ..SessionPolicy::default()
    });

    match session.advance().await.expect("first advance") {
        SessionEvent::BeforeModelCall { turn, .. } => assert_eq!(turn, 1),
        other => panic!("expected BeforeModelCall, got {other:?}"),
    }

    // Advancing without answering is a protocol violation (matching the
    // stream driver), not a silent auto-continue.
    let error = session
        .advance()
        .await
        .expect_err("advance must reject an unanswered BeforeModelCall");
    assert!(error.to_string().contains("reply_before_call"));

    // Answering afterwards still works.
    session
        .reply_before_call(rig_agent::agent::hook::CompletionCallAction::Continue)
        .expect("continue should be accepted");
    let done = match session.advance().await.expect("post-reply advance") {
        SessionEvent::Done(done) => done,
        other => panic!("expected Done, got {other:?}"),
    };
    assert_eq!(done.output, "answer");
}

#[derive(serde::Deserialize, rig::schemars::JsonSchema)]
struct ExtractedNumber {
    n: u32,
}

#[tokio::test]
async fn extract_retry_history_starts_with_the_original_prompt() {
    // Attempt 1 burns the run's inner output-retry (two non-JSON turns),
    // finalizes best-effort, and fails deserialization; attempt 2 succeeds.
    let script = MockScript::from_responses(vec![
        text_response("not json", 1),
        text_response("still not json", 1),
        text_response(r#"{"n": 7}"#, 1),
    ]);
    let value: ExtractedNumber = rig::extract::extract(
        AgentConfig::new(),
        mock_provider(script.clone()),
        Arc::new(Runtime::new()),
        "extract the number",
        1,
    )
    .await
    .expect("extraction should succeed on the retry");
    assert_eq!(value.n, 7);

    // The retry request's history must open with the original user prompt,
    // not the previous assistant output (strict providers reject histories
    // that start with an assistant message).
    let requests = script.requests();
    assert_eq!(requests.len(), 3);
    let first_conversation_message = requests[2]
        .chat_history
        .iter()
        .find(|message| !matches!(message, rig::message::Message::System { .. }))
        .expect("retry request should carry conversation messages");
    assert_eq!(
        *first_conversation_message,
        rig::message::Message::user("extract the number")
    );
}

#[tokio::test]
async fn extract_with_usage_sums_usage_across_failed_and_successful_attempts() {
    // Attempt 1 fails deserialization after a billed response (5 tokens);
    // attempt 2 succeeds (7 tokens). The outcome reports the sum.
    let script = MockScript::from_responses(vec![
        text_response("not json", 5),
        text_response(r#"{"n": 7}"#, 7),
    ]);
    let outcome = rig::extract::extract_with_usage::<ExtractedNumber>(
        AgentConfig::new(),
        mock_provider(script),
        Arc::new(Runtime::new()),
        "extract the number",
        1,
    )
    .await
    .expect("extraction should succeed on the retry");
    assert_eq!(outcome.value.n, 7);
    assert_eq!(outcome.usage.total_tokens, 12);
}

#[tokio::test]
async fn extract_native_parses_json_wrapped_in_prose() {
    // Native mode: no synthetic output tool; the model's final text carries
    // the JSON wrapped in prose, and the balanced-JSON fallback finds it.
    let script = MockScript::from_responses(vec![text_response(
        r#"Sure! Here is the result: {"n": 42} — hope that helps."#,
        9,
    )]);
    let outcome = rig::extract::extract_native::<ExtractedNumber>(
        AgentConfig::new(),
        mock_provider(script.clone()),
        Arc::new(Runtime::new()),
        "extract the number",
        0,
    )
    .await
    .expect("native extraction should succeed");
    assert_eq!(outcome.value.n, 42);
    assert_eq!(outcome.usage.total_tokens, 9);

    // The request carried the schema natively (no synthetic output tool).
    let requests = script.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].tools.is_empty());
    assert!(requests[0].output_schema.is_some());
}

#[tokio::test]
async fn run_refuses_executable_tools() {
    let script = MockScript::from_responses(vec![text_response("unused", 1)]);
    let session = session(script).with_tools(adder_catalog());
    let error = session.run().await.expect_err("run should refuse tools");
    assert!(error.to_string().contains("cannot execute tools"));
}
