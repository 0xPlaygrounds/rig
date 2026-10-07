use super::*;
use rig_core::message::ToolChoice;
use rig_core::message::{StopReason, ToolFunction, ToolResultContent};
use serde_json::json;

#[test]
fn initial_prompt_is_rewritable_only_before_the_run_starts() {
    let mut run = AgentRun::new("original").max_turns(2);
    assert!(run.initial_prompt().is_some());
    run.rewrite_initial_prompt("rewritten")
        .expect("rewrite before start");

    let step = run.next_step().expect("first step");
    let AgentRunStep::CallModel {
        prompt, history, ..
    } = step
    else {
        panic!("expected CallModel");
    };
    assert_eq!(prompt, Message::user("rewritten"));
    assert!(history.is_empty());

    // Once the run has started, the prompt is committed.
    assert!(run.initial_prompt().is_none());
    assert!(run.rewrite_initial_prompt("too late").is_err());
}

#[test]
fn input_chat_history_reflects_the_configured_history() {
    let run = AgentRun::new("p");
    assert!(run.input_chat_history().is_empty());
    let run = AgentRun::new("p").with_history(vec![Message::user("earlier")]);
    assert_eq!(run.input_chat_history(), [Message::user("earlier")]);
}

fn entry(kind: &str, turn: usize, value: serde_json::Value) -> RunEntry {
    RunEntry {
        kind: kind.to_string(),
        turn,
        value,
    }
}

#[test]
fn entries_round_trip_through_serialization_in_append_order() {
    let mut run = AgentRun::new("p");
    run.append_entry(entry("approval", 1, json!({"tool": "add"})));
    run.append_entry(entry("counter", 1, json!(1)));
    run.append_entry(entry("counter", 2, json!(2)));

    let serialized = serde_json::to_string(&run).expect("run serializes");
    let restored: AgentRun = serde_json::from_str(&serialized).expect("run deserializes");
    assert_eq!(restored.entries(), run.entries());
    assert_eq!(
        restored.entries_of("counter").count(),
        2,
        "kind filter sees both counter snapshots"
    );
    // Last-wins: the snapshot pattern reads the most recent entry.
    assert_eq!(
        restored.last_entry_of("counter"),
        Some(&entry("counter", 2, json!(2)))
    );
    assert_eq!(restored.last_entry_of("absent"), None);

    // A cloned ("forked") run carries the entries verbatim.
    assert_eq!(restored.entries(), run.entries());
}

#[test]
fn runs_serialize_empty_entries_explicitly() {
    let run = AgentRun::new("p");
    let value = serde_json::to_value(&run).expect("serializes");
    assert_eq!(
        value["entries"],
        json!([]),
        "present, and empty, when empty"
    );
    let restored: AgentRun = serde_json::from_value(value).expect("deserializes");
    assert!(restored.entries().is_empty());
}

/// The step and outcome types round-trip: a host caching an in-flight
/// step in serializable state (a saved world) can restore it.
#[test]
fn run_step_and_outcome_round_trip_through_serde() {
    let step = AgentRunStep::CallModel {
        prompt: Message::user("hi"),
        history: vec![Message::assistant("prior")],
        turn: 1,
    };
    let json = serde_json::to_string(&step).expect("serialize step");
    let restored: AgentRunStep = serde_json::from_str(&json).expect("deserialize step");
    let AgentRunStep::CallModel { turn, .. } = restored else {
        panic!("wrong variant");
    };
    assert_eq!(turn, 1);

    let outcome = ModelTurnOutcome::Continue {
        response_hook_suppressed: true,
    };
    let json = serde_json::to_string(&outcome).expect("serialize outcome");
    assert!(matches!(
        serde_json::from_str::<ModelTurnOutcome>(&json).expect("deserialize outcome"),
        ModelTurnOutcome::Continue {
            response_hook_suppressed: true
        }
    ));
}

fn tool_names(names: &[&str]) -> std::collections::BTreeSet<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

/// The policy of a turn advertising `names` under the default tool choice.
fn policy(names: &[&str]) -> TurnPolicy {
    TurnPolicy::new(tool_names(names), None, None).expect("policy")
}

fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
    Usage::new()
        .input_tokens(input_tokens)
        .output_tokens(output_tokens)
        .total_tokens(input_tokens + output_tokens)
}

/// The provider document of a turn these tests build by hand: there is no
/// provider behind it, so the document names the test as its origin.
fn hand_raw() -> serde_json::Value {
    json!({"origin": "hand-built test turn"})
}

fn text_turn(text: &str) -> ModelTurn {
    text_turn_with_raw(text, hand_raw())
}

fn text_turn_with_raw(text: &str, raw: serde_json::Value) -> ModelTurn {
    ModelTurn::new(
        rig_core::message::AssistantMessage::default(),
        vec![AssistantContent::text(text)],
        Usage::default(),
        policy(&["add"]),
        raw,
    )
}

fn tool_call(id: &str, name: &str) -> AssistantContent {
    // The provider-boundary shape: a non-empty wire id becomes both the
    // durable id and the provider correlator; an empty wire id takes the
    // deterministic `tool-0` handle (`provider` records the absence).
    AssistantContent::ToolCall(ToolCall::from_wire(
        id,
        ToolFunction::new(
            rig_core::message::ToolName::new(name.to_string()).expect("tool name"),
            json!({"x": 1}),
        ),
    ))
}

fn tool_call_turn(id: &str, name: &str) -> ModelTurn {
    tool_call_turn_with_raw(id, name, hand_raw())
}

fn tool_call_turn_with_raw(id: &str, name: &str, raw: serde_json::Value) -> ModelTurn {
    ModelTurn::new(
        rig_core::message::AssistantMessage::default(),
        vec![tool_call(id, name)],
        Usage::default(),
        policy(&["add"]),
        raw,
    )
}

fn tool_result(id: &str, output: &str) -> UserContent {
    // Every result in these tests answers a call to the `add` tool; the
    // executed tool's name is required data on a result.
    UserContent::tool_result(
        rig_core::message::CallId::from_wire(id),
        rig_core::message::ToolName::new("add").expect("tool name"),
        vec![ToolResultContent::text(output)],
    )
}

fn expect_call_model(run: &mut AgentRun) -> (Message, Vec<Message>, usize) {
    match run.next_step().expect("next_step should succeed") {
        AgentRunStep::CallModel {
            prompt,
            history,
            turn,
        } => (prompt, history, turn),
        step => panic!("expected CallModel, got {step:?}"),
    }
}

fn expect_call_tools(run: &mut AgentRun) -> Vec<PendingToolCall> {
    match run.next_step().expect("next_step should succeed") {
        AgentRunStep::CallTools { calls } => calls,
        step => panic!("expected CallTools, got {step:?}"),
    }
}

fn expect_done(run: &mut AgentRun) -> PromptResponse {
    match run.next_step().expect("next_step should succeed") {
        AgentRunStep::Done(response) => response,
        step => panic!("expected Done, got {step:?}"),
    }
}

fn expect_continue(outcome: ModelTurnOutcome) -> bool {
    match outcome {
        ModelTurnOutcome::Continue {
            response_hook_suppressed,
        } => response_hook_suppressed,
        outcome => panic!("expected Continue, got {outcome:?}"),
    }
}

fn expect_needs_resolution(outcome: ModelTurnOutcome) -> InvalidToolCallContext {
    match outcome {
        ModelTurnOutcome::NeedsResolution(context) => context,
        outcome => panic!("expected NeedsResolution, got {outcome:?}"),
    }
}

#[test]
fn repeated_model_turn_reuses_prompt_without_recording_rejected_response() {
    let first_usage = usage(10, 3);
    let second_usage = usage(7, 2);
    let mut run = AgentRun::new("question").max_turns(2);

    let (first_prompt, first_history, first_turn) = expect_call_model(&mut run);
    assert_eq!(first_prompt, Message::user("question"));
    assert!(first_history.is_empty());
    assert_eq!(first_turn, 1);
    expect_continue(
        run.model_response(text_turn("rejected").with_usage_for_test(first_usage))
            .expect("first response"),
    );

    run.retry_model_turn(RetryRequest::Repeat)
        .expect("repeat should be accepted");
    let (second_prompt, second_history, second_turn) = expect_call_model(&mut run);
    assert_eq!(second_prompt, Message::user("question"));
    assert!(second_history.is_empty());
    assert_eq!(second_turn, 2);
    assert_eq!(run.messages(), &[Message::user("question")]);

    expect_continue(
        run.model_response(text_turn("accepted").with_usage_for_test(second_usage))
            .expect("second response"),
    );
    let response = expect_done(&mut run);
    assert_eq!(response.output(), "accepted");
    assert_eq!(response.usage, first_usage + second_usage);
    assert_eq!(response.completion_calls.len(), 2);
    let messages = response.messages;
    assert_eq!(messages.len(), 2);
    assert!(!format!("{messages:?}").contains("rejected"));
}

#[test]
fn model_turn_retry_rejects_tool_calls_without_advancing_to_execution() {
    let mut run = AgentRun::new("add things").max_turns(2);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("call_1", "add"))
            .expect("tool response"),
    );
    let err = run
        .retry_model_turn(RetryRequest::Feedback("do not call tools".to_string()))
        .expect_err("tool-bearing retries must fail closed");

    let PromptError::Cancelled {
        chat_history,
        reason,
    } = err
    else {
        panic!("tool-bearing retry should return Cancelled");
    };
    assert!(reason.contains("tool-bearing model turns"));
    assert!(reason.contains("tool-call hooks"));
    assert_eq!(chat_history, vec![Message::user("add things")]);
    assert!(run.next_step().is_err(), "failed run cannot execute tools");
}

#[test]
fn tool_roundtrip_threads_history_and_usage() {
    let mut run = AgentRun::new("add things").max_turns(2);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("call_1", "add").with_usage_for_test(usage(10, 5)))
            .expect("model_response should succeed"),
    );

    let calls = expect_call_tools(&mut run);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool_call.function.name, "add");
    assert!(calls[0].preresolved_result.is_none());

    run.tool_results(vec![tool_result("call_1", "2")])
        .expect("tool_results should succeed");

    let (prompt, history, turn) = expect_call_model(&mut run);
    assert_eq!(turn, 2);
    // The tool-result user message becomes the new prompt; the assistant
    // turn is part of the history.
    assert!(matches!(prompt, Message::User { .. }));
    assert_eq!(history.len(), 2);

    expect_continue(
        run.model_response(text_turn("the answer is 2").with_usage_for_test(usage(20, 7)))
            .expect("model_response should succeed"),
    );

    let response = expect_done(&mut run);
    assert_eq!(response.output(), "the answer is 2");
    assert_eq!(response.usage, usage(30, 12));
    assert_eq!(response.completion_calls.len(), 2);
    assert_eq!(response.completion_calls[0].call_index, 0);
    assert_eq!(response.completion_calls[0].usage, usage(10, 5));
    assert_eq!(response.completion_calls[1].usage, usage(20, 7));
    // prompt, assistant tool call, tool result, final assistant text
    assert_eq!(response.messages.len(), 4);
}

/// A turn that calls `add` and ends with `stop` at `finish`.
fn ended_call_turn(stop: StopReason, finish: FinishReason) -> ModelTurn {
    ModelTurn::new(
        AssistantMessage::default().with_stop(stop),
        vec![
            AssistantContent::text("checking"),
            tool_call("call_1", "add"),
        ],
        Usage::default(),
        policy(&["add"]),
        hand_raw(),
    )
    .with_finish_reason(Some(finish))
}

/// A turn that ends in an error runs none of its tool calls: the run ends
/// with the turn's stop reason, and the turn stays in the run's messages.
#[test]
fn a_failed_turn_runs_no_tools_and_ends_the_run() {
    let mut run = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut run);
    let reason = "Provider finish_reason: pause_turn_unmapped";
    expect_continue(
        run.model_response(ended_call_turn(
            StopReason::Error(reason.to_owned()),
            FinishReason::Other("pause_turn_unmapped".to_owned()),
        ))
        .expect("model_response should succeed"),
    );
    let error = run.next_step().expect_err("the failed turn ends the run");
    assert!(
        matches!(&error, PromptError::Provider(ProviderError::Response(message))
            if message.contains("none of its tool calls ran") && message.contains(reason)),
        "{error}"
    );
    let kept = run.messages().last().expect("the failed turn is kept");
    assert!(
        matches!(kept, Message::Assistant(turn)
            if turn.stop.as_ref().is_some_and(StopReason::is_failure)
                && turn.tool_calls().count() == 1),
        "{kept:?}"
    );
}

/// A turn the token limit ended is finished, so its complete call runs.
#[test]
fn a_length_turn_with_a_complete_call_runs_it() {
    let mut run = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(ended_call_turn(StopReason::Length, FinishReason::Length))
            .expect("model_response should succeed"),
    );
    let calls = expect_call_tools(&mut run);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool_call.function.name, "add");
}

#[test]
fn invalid_tool_call_fail_returns_unknown_tool_call() {
    let mut run = AgentRun::new("call something");

    expect_call_model(&mut run);
    let context = expect_needs_resolution(
        run.model_response(tool_call_turn("call_1", "unknown"))
            .expect("model_response should succeed"),
    );
    assert_eq!(context.tool_name, "unknown");
    assert_eq!(context.available_tools, vec!["add".to_string()]);
    assert!(!context.is_streaming);
    // Diagnostic history includes the rejected assistant turn.
    assert_eq!(context.chat_history.len(), 2);

    let err = run
        .resolve_invalid_tool_call(InvalidToolCallAction::fail())
        .expect_err("fail action should error");
    assert!(matches!(
        err,
        PromptError::UnknownToolCall { tool_name, .. } if tool_name == "unknown"
    ));
}

#[test]
fn empty_tool_results_cancel_the_run() {
    let mut run = AgentRun::new("call something").max_turns(2);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("call_1", "add"))
            .expect("model_response should succeed"),
    );
    expect_call_tools(&mut run);

    let err = run
        .tool_results(Vec::new())
        .expect_err("empty results should cancel");
    assert!(matches!(
        err,
        PromptError::Cancelled { reason, .. }
            if reason.contains("tool execution produced no tool results")
    ));
}

#[test]
fn out_of_protocol_calls_are_rejected_without_corrupting_state() {
    let mut run = AgentRun::new("hello");

    let err = run
        .tool_results(vec![tool_result("call_1", "x")])
        .expect_err("no CallTools pending");
    assert!(matches!(err, PromptError::Cancelled { .. }));

    // The run is still drivable after a rejected out-of-protocol call.
    expect_call_model(&mut run);
    let err = run
        .next_step()
        .expect_err("model response is pending, next_step must be rejected");
    assert!(matches!(err, PromptError::Cancelled { .. }));
    expect_continue(
        run.model_response(text_turn("hi"))
            .expect("model_response should still succeed"),
    );
    assert_eq!(expect_done(&mut run).output(), "hi");
}

#[test]
fn model_response_rejected_after_streamed_completion_call_record() {
    let mut run = AgentRun::new("hello");
    expect_call_model(&mut run);
    run.record_streamed_completion_call(
        Usage::default(),
        ResponseIdentity::default(),
        None,
        hand_raw(),
    )
    .expect("record should succeed");

    let err = run
        .model_response(text_turn("hi"))
        .expect_err("mixed streamed/non-streamed ingestion must be rejected");
    assert!(matches!(err, PromptError::Cancelled { .. }));
    // No duplicate completion call was appended.
    assert_eq!(run.completion_calls().len(), 1);
}

#[test]
fn done_step_is_idempotent() {
    let mut run = AgentRun::new("hello");
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(text_turn("hi"))
            .expect("model_response should succeed"),
    );
    assert_eq!(expect_done(&mut run).output(), "hi");
    assert_eq!(expect_done(&mut run).output(), "hi");
}

#[test]
fn serialized_run_alone_carries_pending_tool_calls() {
    let mut run = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("call_1", "add"))
            .expect("model_response should succeed"),
    );
    expect_call_tools(&mut run);

    // A fresh process receives only the serialized run: the pending tool
    // calls must be recoverable from the state itself.
    let serialized = serde_json::to_string(&run).expect("mid-run state should serialize");
    drop(run);
    let mut resumed: AgentRun =
        serde_json::from_str(&serialized).expect("mid-run state should deserialize");

    let calls = expect_call_tools(&mut resumed);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool_call.function.name, "add");
    // Re-emission is idempotent while results are pending.
    let calls_again = expect_call_tools(&mut resumed);
    assert_eq!(calls_again[0].tool_call.id, calls[0].tool_call.id);

    // Answer using only IDs learned from the re-emitted step.
    let results = calls
        .iter()
        .map(|call| {
            UserContent::tool_result(
                call.tool_call.id.clone(),
                call.tool_call.function.name.clone(),
                vec![ToolResultContent::text("2")],
            )
        })
        .collect::<Vec<_>>();
    resumed
        .tool_results(results)
        .expect("tool_results should succeed");
    expect_call_model(&mut resumed);
    expect_continue(
        resumed
            .model_response(text_turn("done"))
            .expect("model_response should succeed"),
    );
    assert_eq!(expect_done(&mut resumed).output(), "done");
}

#[test]
fn tool_results_validates_against_pending_calls() {
    let drive_to_pending_tools = || {
        let mut run = AgentRun::new("add things").max_turns(2);
        expect_call_model(&mut run);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add"))
                .expect("model_response should succeed"),
        );
        expect_call_tools(&mut run);
        run
    };

    // A result for an unknown call ID is rejected without corrupting the run.
    let mut run = drive_to_pending_tools();
    let err = run
        .tool_results(vec![tool_result("call_unknown", "2")])
        .expect_err("unknown tool call id must be rejected");
    assert!(matches!(err, PromptError::Cancelled { .. }));
    run.tool_results(vec![tool_result("call_1", "2")])
        .expect("valid results should still be accepted after a rejection");

    // Leaving a pending call unanswered is rejected.
    let mut run = drive_to_pending_tools();
    let err = run
        .tool_results(vec![tool_result("call_1", "2"), tool_result("call_1", "3")])
        .expect_err("answering one call twice must be rejected");
    assert!(matches!(err, PromptError::Cancelled { .. }));

    // Non-tool-result content is rejected.
    let mut run = drive_to_pending_tools();
    let err = run
        .tool_results(vec![UserContent::text("not a tool result")])
        .expect_err("non-tool-result content must be rejected");
    assert!(matches!(err, PromptError::Cancelled { .. }));
}

#[test]
fn agent_run_deserializes_suspended_state() {
    // A suspended run persisted mid-`ExecutingTools` restores and resumes:
    // the recorded call's usage loads, the pending tool call is re-issued,
    // and the run advances to the next model call after results arrive.
    let mut suspended = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut suspended);
    expect_continue(
        suspended
            .model_response(tool_call_turn("call_1", "add"))
            .expect("model_response should succeed"),
    );
    expect_call_tools(&mut suspended);
    let fixture = serde_json::to_string(&suspended).expect("suspended run should serialize");

    let bare_ids = fixture.replace(r#""id":{"provider":"call_1"}"#, r#""id":"call_1""#);
    assert_ne!(bare_ids, fixture);
    assert!(
        serde_json::from_str::<AgentRun>(&bare_ids).is_err(),
        "bare IDs must not be silently reinterpreted"
    );

    let mut restored: AgentRun =
        serde_json::from_str(&fixture).expect("suspended run should deserialize");
    assert_eq!(restored.completion_calls()[0].usage, Usage::default());

    let calls = expect_call_tools(&mut restored);
    assert_eq!(calls.len(), 1);
    // The call's identity is its own id.
    assert_eq!(
        calls[0].tool_call.id,
        rig_core::message::CallId::from_wire("call_1")
    );
    restored
        .tool_results(vec![tool_result("call_1", "2")])
        .expect("tool_results should succeed");
    expect_call_model(&mut restored);
}

#[test]
fn serde_round_trip_at_exhausted_budget_preserves_boundary() {
    let mut run = AgentRun::new("add things").max_turns(1);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("call_1", "add"))
            .expect("model_response should succeed"),
    );
    expect_call_tools(&mut run);
    run.tool_results(vec![tool_result("call_1", "2")])
        .expect("tool_results should succeed");

    let serialized = serde_json::to_string(&run).expect("exhausted run should serialize");
    let mut restored: AgentRun =
        serde_json::from_str(&serialized).expect("exhausted run should deserialize");
    assert_eq!(restored.completion_calls().len(), 1);
    let err = restored
        .next_step()
        .expect_err("restored run must not emit a second model call");
    assert!(matches!(err, PromptError::MaxTurns { max_turns: 1, .. }));
    assert_eq!(restored.turn(), 1);
}

#[test]
fn serde_round_trip_mid_run_resumes_identically() {
    let drive_to_pending_tools = || {
        let mut run = AgentRun::new("add things").max_turns(2);
        expect_call_model(&mut run);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add").with_usage_for_test(usage(10, 5)))
                .expect("model_response should succeed"),
        );
        expect_call_tools(&mut run);
        run
    };

    let finish = |mut run: AgentRun| {
        run.tool_results(vec![tool_result("call_1", "2")])
            .expect("tool_results should succeed");
        expect_call_model(&mut run);
        expect_continue(
            run.model_response(text_turn("done").with_usage_for_test(usage(3, 4)))
                .expect("model_response should succeed"),
        );
        expect_done(&mut run)
    };

    let uninterrupted = finish(drive_to_pending_tools());

    let suspended = drive_to_pending_tools();
    let serialized = serde_json::to_string(&suspended).expect("mid-run state should serialize");
    // The pending call's id is part of the persisted state: a resumed
    // process keeps the id its consumers already saw.
    assert!(
        serialized.contains("\"call_1\""),
        "the pending call persists its id: {serialized}"
    );
    let restored: AgentRun =
        serde_json::from_str(&serialized).expect("mid-run state should deserialize");
    let resumed = finish(restored);

    assert_eq!(resumed.output(), uninterrupted.output());
    assert_eq!(resumed.usage, uninterrupted.usage);
    assert_eq!(resumed.completion_calls, uninterrupted.completion_calls);
    // Direct value comparison: with `additional_params` a named field,
    // a restored message is identical to the live one — no serialized-form
    // detour that would hide a round-trip divergence.
    assert_eq!(resumed.messages, uninterrupted.messages);
}

#[test]
fn pending_invalid_tool_call_survives_serde_round_trip() {
    let mut run = AgentRun::new("call something");
    expect_call_model(&mut run);
    let context = expect_needs_resolution(
        run.model_response(tool_call_turn("call_1", "unknown"))
            .expect("model_response should succeed"),
    );

    let serialized = serde_json::to_string(&run).expect("state should serialize");
    let restored: AgentRun = serde_json::from_str(&serialized).expect("state should deserialize");
    let restored_context = restored
        .pending_invalid_tool_call()
        .expect("pending resolution should survive serialization");
    assert_eq!(restored_context.tool_name, context.tool_name);
    assert_eq!(
        restored_context.chat_history.len(),
        context.chat_history.len()
    );
    assert_eq!(restored_context.tool_call_id, context.tool_call_id);
    assert_eq!(
        context
            .tool_call_id
            .as_ref()
            .and_then(|id| id.provider().map(|provider| provider.as_str())),
        Some("call_1")
    );
    assert_eq!(restored_context.tool_call_id, context.tool_call_id);
}

/// Every assistant tool call in `messages` must have a matching user tool
/// result — an unanswered tool_use is rejected by providers on replay.
fn assert_no_orphan_tool_use(messages: &[Message]) {
    let mut pending = Vec::new();
    for message in messages {
        match message {
            Message::Assistant(rig_core::message::AssistantMessage { content, .. }) => {
                pending.extend(content.iter().filter_map(|item| match item {
                    AssistantContent::ToolCall(call) => Some(&call.id),
                    _ => None,
                }));
            }
            Message::User { content } => {
                for item in content {
                    if let UserContent::ToolResult(result) = item {
                        let index = pending
                            .iter()
                            .position(|id| *id == &result.call)
                            .expect("result must answer a preceding pending call");
                        pending.remove(index);
                    }
                }
            }
            Message::System { .. } => {}
        }
    }
    assert!(
        pending.is_empty(),
        "unanswered tool call occurrences: {pending:?}"
    );
}

/// A turn calling the output tool with arguments that are not a JSON object.
fn malformed_output_tool_turn() -> ModelTurn {
    ModelTurn::new(
        rig_core::message::AssistantMessage::default(),
        vec![AssistantContent::ToolCall(ToolCall::from_wire(
            "call_1",
            ToolFunction::parse(
                rig_core::message::ToolName::new("final_result").expect("tool name"),
                "{\"x\":",
            ),
        ))],
        Usage::default(),
        TurnPolicy::new(tool_names(&["add"]), None, Some("final_result".to_string()))
            .expect("policy"),
        hand_raw(),
    )
}

#[test]
fn a_malformed_output_tool_call_is_not_the_runs_output() {
    let mut run = AgentRun::new("summarize").with_output_tool_name("final_result");
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(malformed_output_tool_turn())
            .expect("model_response should succeed"),
    );
    let error = run
        .next_step()
        .expect_err("malformed output arguments fail the run");
    assert!(error.to_string().contains("not a JSON object"), "{error}");
}

#[test]
fn tool_mode_reprompts_when_output_args_are_not_a_json_object() {
    let mut run = AgentRun::new("summarize")
        .max_turns(2)
        .with_output_tool_name("final_result")
        .with_output_validation(None, 1);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(malformed_output_tool_turn())
            .expect("model_response should succeed"),
    );

    let (prompt, mut history, turn) = expect_call_model(&mut run);
    assert_eq!(turn, 2);
    let prompt_json = serde_json::to_string(&prompt).expect("prompt should serialize");
    assert!(prompt_json.contains("not a JSON object"), "{prompt_json}");
    history.push(prompt);
    assert_no_orphan_tool_use(&history);
    assert!(!run.is_done());
}

/// A run pinned to an output tool whose turn policy names none (a resumed
/// runner without a schema prepares Native turns) does not ask the model to
/// call that tool: the turn did not advertise it.
#[test]
fn a_text_answer_is_not_reprompted_for_an_output_tool_the_turn_did_not_advertise() {
    let mut run = AgentRun::new("summarize")
        .max_turns(2)
        .with_output_tool_name("final_result")
        .with_output_validation(None, 1);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(text_turn("plain prose"))
            .expect("model_response should succeed"),
    );
    let response = expect_done(&mut run);
    assert_eq!(response.output(), "plain prose");
}

/// A skipped call to the pinned output tool on a turn whose policy does not
/// name it is an ordinary skipped call, not the run's answer.
#[test]
fn a_skipped_call_to_an_unadvertised_output_tool_does_not_finalize_the_run() {
    let mut run = AgentRun::new("summarize")
        .max_turns(2)
        .with_output_tool_name("final_result");
    expect_call_model(&mut run);
    let context = expect_needs_resolution(
        run.model_response(tool_call_turn("c1", "final_result"))
            .expect("model_response should succeed"),
    );
    assert_eq!(context.reason, InvalidToolCallReason::UnknownTool);
    expect_continue(
        run.resolve_invalid_tool_call(InvalidToolCallAction::skip("not this turn"))
            .expect("the skip is accepted"),
    );
    let calls = expect_call_tools(&mut run);
    assert_eq!(calls.len(), 1);
    assert!(calls[0].preresolved_result.is_some());
    assert!(!run.is_done());
}

impl ModelTurn {
    fn with_usage_for_test(mut self, usage: Usage) -> Self {
        self.usage = usage;
        self
    }
}

/// Durable human-in-the-loop: the run is serialized while tool calls are
/// pending, reconstructed from JSON (as a separate process / request would),
/// and only then does the human decision land — approve one call, deny the
/// other. The resumed-from-bytes run accepts those results and continues to
/// completion, proving approval can happen out-of-process / arbitrarily later.
/// This is the state-machine foundation for `examples/agent_with_durable_approval`.
#[test]
fn durable_human_in_the_loop_approval_survives_serialize_resume() {
    let mut run = AgentRun::new("pay two invoices").max_turns(3);
    let (_, _, turn) = expect_call_model(&mut run);
    assert_eq!(turn, 1);

    // Turn 1: the model emits two tool calls.
    let two_calls = vec![tool_call("c1", "add"), tool_call("c2", "add")];
    let outcome = run
        .model_response(ModelTurn::new(
            rig_core::message::AssistantMessage::default(),
            two_calls,
            Usage::default(),
            policy(&["add"]),
            hand_raw(),
        ))
        .expect("model_response");
    expect_continue(outcome);

    // CallTools is now pending. Serialize the run (a durable checkpoint) and
    // reconstruct it from the bytes — nothing live crosses this boundary.
    let checkpoint = serde_json::to_string(&run).expect("serialize suspended run");
    let mut resumed: AgentRun = serde_json::from_str(&checkpoint).expect("deserialize run");

    // The resumed run re-emits the pending calls purely from its own state.
    let calls = expect_call_tools(&mut resumed);
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls[0]
            .tool_call
            .id
            .provider()
            .map(|provider| provider.as_str()),
        Some("c1")
    );
    assert_eq!(
        calls[1]
            .tool_call
            .id
            .provider()
            .map(|provider| provider.as_str()),
        Some("c2")
    );

    // The human decision lands only after the resume: approve c1 (real
    // result), deny c2 (the reason becomes the tool result the model sees).
    resumed
        .tool_results(vec![
            tool_result("c1", "approved-result"),
            tool_result("c2", "denied by reviewer: second payment not authorized"),
        ])
        .expect("tool_results on the resumed run");

    // Both decisions are recorded in the resumed run's persisted state.
    let after = serde_json::to_string(&resumed).expect("serialize resumed run");
    assert!(
        after.contains("approved-result"),
        "the approved call's result must be in the resumed run state"
    );
    assert!(
        after.contains("denied by reviewer: second payment not authorized"),
        "the denied call's reason must be in the resumed run state"
    );

    // Turn 2: the model wraps up; the run completes from the resumed state.
    let (_, _, turn2) = expect_call_model(&mut resumed);
    assert_eq!(turn2, 2);
    expect_continue(
        resumed
            .model_response(text_turn("done"))
            .expect("model_response 2"),
    );
    let response = expect_done(&mut resumed);
    assert_eq!(response.output(), "done");
}

// ---------------------------------------------------------------------
// Raw provider response capture (always on), at the state-machine layer:
// the drivers hand `AgentRun` the payload they read off the provider
// response (blocking) or the stream terminal (streamed); the run must
// record it per call, and persisted run state must carry it across a
// suspend/resume boundary. There is no absent payload: a turn is built from
// the document that produced it, and a record without one is refused on load.
// ---------------------------------------------------------------------

fn raw_payload(attempt: &str) -> serde_json::Value {
    json!({
        "id": format!("resp-{attempt}"),
        "provider_only": attempt,
    })
}

/// A suspended run's recorded payloads survive the serialize/resume
/// boundary intact — a resumed process sees exactly what the live one
/// recorded.
#[test]
fn recorded_raw_survives_serde_round_trip() {
    let raw = raw_payload("suspended");
    let mut run = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn_with_raw("call_1", "add", raw.clone()))
            .expect("model_response should succeed"),
    );
    expect_call_tools(&mut run);

    let serialized = serde_json::to_string(&run).expect("mid-run state should serialize");
    let restored: AgentRun =
        serde_json::from_str(&serialized).expect("mid-run state should deserialize");
    assert_eq!(restored.completion_calls().len(), 1);
    assert_eq!(restored.completion_calls()[0].raw, raw);
    assert_eq!(restored.completion_calls(), run.completion_calls());
}

/// `ModelTurn` and `CompletionCall` carry `raw` through their serde round
/// trips, and a record without the key is refused rather than loaded with
/// a payload invented.
#[test]
fn raw_round_trips_and_a_missing_key_is_refused() {
    let raw = raw_payload("turn");
    let turn = text_turn_with_raw("hi", raw.clone());
    let value = serde_json::to_value(&turn).expect("turn should serialize");
    assert_eq!(value["raw"], raw);
    let restored: ModelTurn =
        serde_json::from_value(value.clone()).expect("turn should deserialize");
    assert_eq!(restored.raw, raw);

    let mut without_raw = value;
    without_raw
        .as_object_mut()
        .expect("turn serializes as an object")
        .shift_remove("raw")
        .expect("the raw key was present");
    let error = serde_json::from_value::<ModelTurn>(without_raw)
        .expect_err("a turn without a raw key is refused");
    assert!(error.to_string().contains("raw"), "{error}");

    let call = CompletionCall::new(0, usage(1, 2), raw_payload("call"));
    let mut value = serde_json::to_value(&call).expect("call should serialize");
    assert_eq!(value["raw"], raw_payload("call"));
    let restored: CompletionCall =
        serde_json::from_value(value.clone()).expect("call should deserialize");
    assert_eq!(restored, call);
    value
        .as_object_mut()
        .expect("call serializes as an object")
        .shift_remove("raw")
        .expect("the raw key was present");
    let error = serde_json::from_value::<CompletionCall>(value)
        .expect_err("a call without a raw key is refused");
    assert!(error.to_string().contains("raw"), "{error}");
}

/// The persisted envelope is versioned: a run written under another format
/// number is refused by name, and an unknown key is refused rather than
/// ignored, so state from another rig is never loaded with defaults
/// filled in.
#[test]
fn a_run_of_another_format_or_with_an_unknown_key_is_refused() {
    let run = AgentRun::new("hello");
    let mut value = serde_json::to_value(&run).expect("run should serialize");
    assert_eq!(value["format"], RUN_FORMAT);

    value["format"] = serde_json::json!(RUN_FORMAT + 1);
    let error =
        serde_json::from_value::<AgentRun>(value.clone()).expect_err("another format is refused");
    assert_eq!(
        error.to_string(),
        format!(
            "resume refused: the run is format {}, this rig reads format {RUN_FORMAT}",
            RUN_FORMAT + 1
        )
    );

    value["format"] = serde_json::json!(RUN_FORMAT);
    value["retired_field"] = serde_json::json!(true);
    let error = serde_json::from_value::<AgentRun>(value).expect_err("an unknown key is refused");
    assert!(error.to_string().contains("retired_field"), "{error}");
}

fn assistant(content: Vec<AssistantContent>) -> Message {
    Message::Assistant(rig_core::message::AssistantMessage::new(content))
}

#[test]
fn with_validated_history_gates_construction() {
    let bad = vec![
        assistant(vec![AssistantContent::text("a")]),
        assistant(vec![AssistantContent::text("b")]),
    ];
    assert!(AgentRun::new("x").with_validated_history(bad).is_err());
    assert!(
        AgentRun::new("x")
            .with_validated_history(vec![Message::user("ok")])
            .is_ok()
    );
}

/// The model a run last asked survives a suspension: a resumed run's
/// selection hook is shown it as `previous_model`, as a fresh run's is.
#[test]
fn serde_round_trip_keeps_the_previous_model() {
    let mut run = AgentRun::new("add things").max_turns(3);
    assert!(run.previous_model().is_none());
    run.set_previous_model(rig_core::completion::ModelRef::new("fast"));
    let state = serde_json::to_string(&run).expect("the run serializes");
    let restored: AgentRun = serde_json::from_str(&state).expect("the run restores");
    assert_eq!(
        restored.previous_model().map(|model| model.as_str()),
        Some("fast")
    );
}

/// rig#2322: a reasoning-only turn the provider cut short fails the run
/// *before* anything is committed — the reasoning is not history.
#[test]
fn a_truncated_reasoning_only_turn_commits_nothing() {
    let mut run = AgentRun::new("solve this");
    let _ = expect_call_model(&mut run);
    let turn = ModelTurn::new(
        rig_core::message::AssistantMessage::default(),
        vec![AssistantContent::Reasoning(
            rig_core::message::Reasoning::new("thinking, never answering"),
        )],
        Usage::default(),
        policy(&["add"]),
        hand_raw(),
    )
    .with_finish_reason(Some(FinishReason::Length));
    // The turn is accepted into resolution; the run fails when it is read.
    expect_continue(
        run.model_response(turn)
            .expect("the turn is taken into resolution"),
    );
    let error = run
        .next_step()
        .expect_err("an answerless truncated turn fails the run");
    assert!(format!("{error:?}").contains("Length"), "{error:?}");
    assert!(
        run.messages()
            .iter()
            .all(|message| !matches!(message, Message::Assistant(_))),
        "the reasoning-only turn is not history: {:?}",
        run.messages()
    );
}

/// An empty choice, or a retry with no tool calls to answer, builds no
/// message, so nothing empty is appended to history.
#[test]
fn transcript_helpers_build_no_message_from_nothing() {
    assert_eq!(
        transcript::assistant_message(rig_core::message::AssistantMessage::default(), Vec::new()),
        None
    );
    assert_eq!(
        transcript::assistant_turn(rig_core::message::AssistantMessage::default(), Vec::new()),
        None
    );
    assert_eq!(
        transcript::invalid_tool_retry_user_message(
            &[AssistantContent::text("no calls here")],
            &rig_core::message::CallId::from_wire("call_1"),
            "feedback",
        ),
        None
    );
}

/// A turn calling `add` with arguments that are not a JSON object.
fn malformed_call_turn(id: &str) -> ModelTurn {
    ModelTurn::new(
        rig_core::message::AssistantMessage::default(),
        vec![AssistantContent::ToolCall(ToolCall::from_wire(
            id,
            ToolFunction::parse(
                rig_core::message::ToolName::new("add").expect("tool name"),
                "{\"x\":",
            ),
        ))],
        Usage::default(),
        policy(&["add"]),
        hand_raw(),
    )
}

/// Drive one model turn through its tool step, answering every call.
fn tool_step(run: &mut AgentRun, turn: ModelTurn) -> Result<(), PromptError> {
    expect_call_model(run);
    expect_continue(run.model_response(turn)?);
    let calls = match run.next_step()? {
        AgentRunStep::CallTools { calls } => calls,
        step => {
            return Err(PromptError::cancelled(
                Vec::new(),
                format!("expected CallTools, got {step:?}"),
            ));
        }
    };
    run.tool_results(
        calls
            .iter()
            .map(|call| {
                UserContent::ToolResult(
                    call.tool_call
                        .error_result(vec![ToolResultContent::text("answered")]),
                )
            })
            .collect(),
    )
}

#[test]
fn malformed_turns_past_the_limit_fail_the_run() {
    let mut run = AgentRun::new("go")
        .max_turns(10)
        .max_consecutive_malformed_tool_calls(2);
    tool_step(&mut run, malformed_call_turn("c1")).expect("first malformed turn");
    tool_step(&mut run, malformed_call_turn("c2")).expect("second malformed turn");
    let error = tool_step(&mut run, malformed_call_turn("c3")).expect_err("past the limit");
    let message = error.to_string();
    assert!(
        message.contains("tool `add` was called with arguments that are not a JSON object on 3 consecutive turns, more than the 2 retries allowed: "),
        "{message}"
    );
    assert!(run.next_step().is_err(), "the run stays failed");
}

#[test]
fn a_well_formed_tool_step_resets_the_malformed_count() {
    let mut run = AgentRun::new("go")
        .max_turns(10)
        .max_consecutive_malformed_tool_calls(1);
    tool_step(&mut run, malformed_call_turn("c1")).expect("malformed");
    tool_step(&mut run, tool_call_turn("c2", "add")).expect("well formed");
    tool_step(&mut run, malformed_call_turn("c3")).expect("malformed after the reset");
    tool_step(&mut run, malformed_call_turn("c4")).expect_err("past the limit");
}

#[test]
fn without_a_limit_malformed_turns_are_answered_until_max_turns() {
    let mut run = AgentRun::new("go").max_turns(4);
    for id in ["c1", "c2", "c3", "c4"] {
        tool_step(&mut run, malformed_call_turn(id)).expect("answered");
    }
    assert!(
        run.next_step().is_err(),
        "the turn budget, not a malformed limit, ends the run"
    );
}

#[test]
fn ignore_does_not_lift_the_malformed_limit() {
    let mut run = AgentRun::new("go")
        .max_turns(10)
        .max_consecutive_malformed_tool_calls(0)
        .with_unhandled_invalid_tool_call(UnhandledInvalidToolCall::Ignore);
    let error = tool_step(&mut run, malformed_call_turn("c1")).expect_err("past the limit");
    assert!(
        error
            .to_string()
            .contains("more than the 0 retries allowed"),
        "{error}"
    );
}

#[test]
fn the_malformed_count_survives_serde() {
    let mut run = AgentRun::new("go")
        .max_turns(10)
        .max_consecutive_malformed_tool_calls(1);
    tool_step(&mut run, malformed_call_turn("c1")).expect("malformed");
    let saved = serde_json::to_string(&run).expect("serialize");
    let mut restored: AgentRun = serde_json::from_str(&saved).expect("deserialize");
    tool_step(&mut restored, malformed_call_turn("c2")).expect_err("past the limit");
}

#[test]
fn a_malformed_call_has_a_context_only_while_its_tools_are_pending() {
    let mut run = AgentRun::new("go").max_turns(2);
    let turn = malformed_call_turn("c1");
    let AssistantContent::ToolCall(call) = turn.choice[0].clone() else {
        panic!("a tool call");
    };
    assert!(run.malformed_tool_call_context(&call, false).is_none());
    expect_call_model(&mut run);
    expect_continue(run.model_response(turn).expect("model_response"));
    let calls = expect_call_tools(&mut run);
    let context = run
        .malformed_tool_call_context(&calls[0].tool_call, true)
        .expect("a malformed pending call has a context");
    assert_eq!(context.tool_name, "add");
    assert_eq!(context.args.as_deref(), Some("{\"x\":"));
    assert_eq!(context.available_tools, ["add"]);
    assert_eq!(context.allowed_tools, ["add"]);
    assert!(context.is_streaming);
    assert!(matches!(
        context.reason,
        InvalidToolCallReason::MalformedArguments { .. }
    ));
    let AssistantContent::ToolCall(parsed) = tool_call("c2", "add") else {
        panic!("a tool call");
    };
    assert!(run.malformed_tool_call_context(&parsed, false).is_none());
}

#[test]
fn invalid_call_contexts_name_why_the_name_was_rejected() {
    let mut run = AgentRun::new("go");
    expect_call_model(&mut run);
    let context = expect_needs_resolution(
        run.model_response(tool_call_turn("c1", "unknown"))
            .expect("model_response"),
    );
    assert_eq!(context.reason, InvalidToolCallReason::UnknownTool);

    let mut run = AgentRun::new("go");
    expect_call_model(&mut run);
    let context = expect_needs_resolution(
        run.model_response(ModelTurn::new(
            rig_core::message::AssistantMessage::default(),
            vec![tool_call("c1", "sub")],
            Usage::default(),
            TurnPolicy::new(
                tool_names(&["add", "sub"]),
                Some(ToolChoice::Specific {
                    function_names: vec![
                        rig_core::message::ToolName::new("add").expect("tool name"),
                    ],
                }),
                None,
            )
            .expect("policy"),
            hand_raw(),
        ))
        .expect("model_response"),
    );
    assert_eq!(
        context.reason,
        InvalidToolCallReason::DisallowedByToolChoice
    );
}

/// A turn calling `name` under a policy that advertises `add` and reserves
/// `output` as the output tool.
fn output_policy_turn(id: &str, name: &str, output: &str) -> ModelTurn {
    ModelTurn::new(
        rig_core::message::AssistantMessage::default(),
        vec![tool_call(id, name)],
        Usage::default(),
        TurnPolicy::new(tool_names(&["add"]), None, Some(output.to_string())).expect("policy"),
        hand_raw(),
    )
}

#[test]
fn the_first_turn_policy_pins_the_output_tool() {
    let mut run = AgentRun::new("go").max_turns(3);
    assert_eq!(run.output_tool_name(), None);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(output_policy_turn("c1", "add", "final_result"))
            .expect("model_response"),
    );
    assert_eq!(run.output_tool_name(), Some("final_result"));
    expect_call_tools(&mut run);
    run.tool_results(vec![tool_result("c1", "2")])
        .expect("tool results");

    // A later turn naming another output tool never unpins the first.
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(output_policy_turn("c2", "add", "other_result"))
            .expect("model_response"),
    );
    assert_eq!(run.output_tool_name(), Some("final_result"));
    expect_call_tools(&mut run);
    run.tool_results(vec![tool_result("c2", "2")])
        .expect("tool results");

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(output_policy_turn("c3", "final_result", "final_result"))
            .expect("model_response"),
    );
    let AgentRunStep::Done(response) = run.next_step().expect("next_step") else {
        panic!("the pinned output tool's call finalizes the run");
    };
    assert_eq!(response.output(), r#"{"x":1}"#);
}

/// A driver that retries a provider error before any response re-prepares
/// with no pinned name; the turn that does answer pins it, and its output
/// call is intercepted as the answer.
#[test]
fn a_provider_error_retry_still_pins_and_intercepts_the_output_tool() {
    let spec = RunSpec {
        output_schema: Some(json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        })),
        max_turns: Some(2),
        ..RunSpec::new()
    };
    let tools = vec![ToolDefinition {
        name: rig_core::message::ToolName::new("add").expect("tool name"),
        description: "adds".to_string(),
        parameters: json!({"type": "object"}),
    }];
    let mut run = AgentRun::from_spec(&spec, "go", None);
    let (_, history, _) = expect_call_model(&mut run);

    let prepare = |run: &AgentRun| {
        prepare_request(
            &spec,
            &Default::default(),
            &history,
            tools.clone(),
            run.output_tool_name(),
            None,
        )
        .expect("prepared")
    };
    // The first attempt fails at the provider: no response reaches the run.
    let failed = prepare(&run);
    assert_eq!(run.output_tool_name(), None);

    let prepared = prepare(&run);
    assert_eq!(prepared.policy, failed.policy);
    let output_tool = prepared
        .policy
        .output_tool()
        .expect("Tool output mode")
        .to_string();
    expect_continue(
        run.model_response(ModelTurn::new(
            rig_core::message::AssistantMessage::default(),
            vec![tool_call("c1", &output_tool)],
            Usage::default(),
            prepared.policy.clone(),
            hand_raw(),
        ))
        .expect("model_response"),
    );
    assert_eq!(run.output_tool_name(), Some(output_tool.as_str()));
    let AgentRunStep::Done(response) = run.next_step().expect("next_step") else {
        panic!("the output tool call is the answer");
    };
    assert_eq!(response.output(), r#"{"x":1}"#);
}

#[test]
fn a_format_2_run_is_refused_by_name() {
    let run = AgentRun::new("go");
    let mut value = serde_json::to_value(&run).expect("serialize");
    value["format"] = json!(2);
    let error = serde_json::from_value::<AgentRun>(value).expect_err("format 2 is refused");
    assert!(
        error
            .to_string()
            .contains("the run is format 2, this rig reads format 3"),
        "{error}"
    );
}
