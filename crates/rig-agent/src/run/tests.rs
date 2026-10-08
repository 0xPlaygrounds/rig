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

/// The call behind an open pending call, whatever its kind.
fn pending(call: &PendingToolCall) -> &ToolCall {
    match call {
        PendingToolCall::Execute(call) => call.tool_call(),
        PendingToolCall::Malformed(call) => call.tool_call(),
    }
}

fn exec_call(call: PendingToolCall) -> ExecCall {
    match call {
        PendingToolCall::Execute(call) => call,
        other => panic!("expected an executable call, got {other:?}"),
    }
}

fn success(output: &str) -> rig_core::tool::ToolResult {
    rig_core::tool::ToolResult::success(rig_core::tool::ToolOutput::text(output))
}

/// Answer an open call: an executable one with `output`, a malformed one
/// with the run's default feedback.
fn answer_open(call: PendingToolCall, output: &str) -> ToolAnswer {
    match call {
        PendingToolCall::Execute(call) => call.answer(success(output)),
        PendingToolCall::Malformed(call) => call.answer(None),
    }
}

/// Answer every open call of the pending `CallTools` step with `output`.
fn answer_calls(run: &mut AgentRun, output: &str) -> Result<(), PromptError> {
    let calls = expect_call_tools(run);
    run.answer_all(calls.into_iter().map(|call| answer_open(call, output)))
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
    assert_eq!(chat_history.into_vec(), vec![Message::user("add things")]);
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
    assert!(matches!(&calls[0], PendingToolCall::Execute(call) if call.name() == "add"));

    answer_calls(&mut run, "2").expect("the answer should be accepted");

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
    assert_eq!(pending(&calls[0]).function.name, "add");
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
fn no_answers_leave_the_calls_open() {
    let mut run = AgentRun::new("call something").max_turns(2);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("call_1", "add"))
            .expect("model_response should succeed"),
    );
    expect_call_tools(&mut run);

    run.answer_all([]).expect("no answers commit nothing");
    assert_eq!(expect_call_tools(&mut run).len(), 1);
}

/// An executable call from another run at the `CallTools` step.
fn call_from_another_run(id: &str) -> ExecCall {
    let mut other = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut other);
    expect_continue(
        other
            .model_response(tool_call_turn(id, "add"))
            .expect("model_response should succeed"),
    );
    exec_call(expect_call_tools(&mut other).remove(0))
}

#[test]
fn out_of_protocol_calls_are_rejected_without_corrupting_state() {
    let mut run = AgentRun::new("hello");

    let err = run
        .answer(call_from_another_run("call_1").answer(success("x")))
        .expect_err("no CallTools pending");
    assert!(matches!(err, PromptError::Cancelled { .. }));

    // The run is still drivable after a rejected out-of-protocol call, and
    // next_step while a model response is pending re-issues the call.
    let issued = expect_call_model(&mut run);
    assert_eq!(expect_call_model(&mut run), issued);
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
    assert_eq!(pending(&calls[0]).function.name, "add");
    // Re-emission is idempotent while results are pending.
    let calls_again = expect_call_tools(&mut resumed);
    assert_eq!(pending(&calls_again[0]).id, pending(&calls[0]).id);

    // Answer using only the calls of the re-emitted step.
    resumed
        .answer_all(calls.into_iter().map(|call| answer_open(call, "2")))
        .expect("the answers should be accepted");
    expect_call_model(&mut resumed);
    expect_continue(
        resumed
            .model_response(text_turn("done"))
            .expect("model_response should succeed"),
    );
    assert_eq!(expect_done(&mut resumed).output(), "done");
}

#[test]
fn answers_validate_against_pending_calls() {
    let drive_to_pending_tools = || {
        let mut run = AgentRun::new("add things").max_turns(2);
        expect_call_model(&mut run);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add"))
                .expect("model_response should succeed"),
        );
        let call = exec_call(expect_call_tools(&mut run).remove(0));
        (run, call)
    };

    // An answer to another batch's call is rejected without corrupting the run.
    let (mut run, call) = drive_to_pending_tools();
    let err = run
        .answer(call_from_another_run("call_unknown").answer(success("2")))
        .expect_err("an answer for another call must be rejected");
    assert!(matches!(err, PromptError::Cancelled { .. }));
    run.answer(call.answer(success("2")))
        .expect("valid answers should still be accepted after a rejection");

    // Answering one call twice is rejected, and nothing is committed.
    let (mut run, call) = drive_to_pending_tools();
    let err = run
        .answer_all([call.clone().answer(success("2")), call.answer(success("3"))])
        .expect_err("answering one call twice must be rejected");
    assert!(matches!(err, PromptError::Cancelled { .. }));
    assert_eq!(expect_call_tools(&mut run).len(), 1);
}

/// A host that keeps its call queue apart from the run, and restores a run
/// one turn ahead, holds a call whose id the provider reused: the stale
/// answer is refused and nothing is committed.
#[test]
fn an_answer_from_an_earlier_turn_is_refused_when_ids_repeat() {
    let mut run = AgentRun::new("add things").max_turns(3);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("c1", "add"))
            .expect("model_response should succeed"),
    );
    let first = exec_call(expect_call_tools(&mut run).remove(0));
    let saved: ExecCall =
        serde_json::from_str(&serde_json::to_string(&first).expect("serialize the call"))
            .expect("deserialize the call");
    run.answer(first.answer(success("turn 1")))
        .expect("the turn-1 answer should be accepted");

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("c1", "add"))
            .expect("model_response should succeed"),
    );
    expect_call_tools(&mut run);
    let messages = run.full_history().len();
    let err = run
        .answer(saved.answer(success("stale")))
        .expect_err("an answer issued for turn 1 must not close turn 2's call");
    assert!(
        matches!(&err, PromptError::Cancelled { reason, .. } if reason.contains("turn 1")),
        "{err:?}"
    );
    assert_eq!(run.full_history().len(), messages, "nothing is committed");

    let current = exec_call(expect_call_tools(&mut run).remove(0));
    run.answer(current.answer(success("turn 2")))
        .expect("the turn-2 answer should be accepted");
    expect_call_model(&mut run);
}

#[test]
fn projection_start_reprojects_only_a_pending_tool_batch() {
    let mut run = AgentRun::new("add things").max_turns(2);
    assert_eq!(run.projection_start(), run.messages().len());
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("call_1", "add"))
            .expect("model_response should succeed"),
    );
    let calls = expect_call_tools(&mut run);
    let start = run.projection_start();
    assert!(
        matches!(&run.messages()[start..], [Message::Assistant(_)]),
        "a pending batch starts at its assistant message: {:?}",
        run.messages()
    );
    run.answer_all(calls.into_iter().map(|call| answer_open(call, "2")))
        .expect("the answers should be accepted");
    assert_eq!(run.projection_start(), run.messages().len());
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
        pending(&calls[0]).id,
        rig_core::message::CallId::from_wire("call_1")
    );
    answer_calls(&mut restored, "2").expect("the answer should be accepted");
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
    answer_calls(&mut run, "2").expect("the answer should be accepted");

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
        answer_calls(&mut run, "2").expect("the answer should be accepted");
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
    // The skip answered the call: the run asks the model again.
    expect_call_model(&mut run);
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
        pending(&calls[0])
            .id
            .provider()
            .map(|provider| provider.as_str()),
        Some("c1")
    );
    assert_eq!(
        pending(&calls[1])
            .id
            .provider()
            .map(|provider| provider.as_str()),
        Some("c2")
    );

    // The human decision lands only after the resume: approve c1 (real
    // result), deny c2 (the reason becomes the skipped result the model sees).
    let mut calls = calls.into_iter().map(exec_call);
    let (Some(approved), Some(denied)) = (calls.next(), calls.next()) else {
        panic!("two executable calls");
    };
    resumed
        .answer_all([
            approved.answer(success("approved-result")),
            denied.answer(rig_core::tool::ToolResult::skipped(
                "denied by reviewer: second payment not authorized",
            )),
        ])
        .expect("answers on the resumed run");

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
            return Err(run.cancel_error(format!("expected CallTools, got {step:?}")));
        }
    };
    run.answer_all(calls.into_iter().map(|call| answer_open(call, "answered")))
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
fn a_malformed_call_has_a_context_naming_its_tools() {
    let mut run = AgentRun::new("go").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(malformed_call_turn("c1"))
            .expect("model_response"),
    );
    let calls = expect_call_tools(&mut run);
    let [PendingToolCall::Malformed(call)] = calls.as_slice() else {
        panic!("one malformed call: {calls:?}");
    };
    let context = run.malformed_context(call, true);
    assert_eq!(context.tool_name, "add");
    assert_eq!(context.args.as_deref(), Some("{\"x\":"));
    assert_eq!(context.available_tools, ["add"]);
    assert_eq!(context.allowed_tools, ["add"]);
    assert!(context.is_streaming);
    assert!(matches!(
        context.reason,
        InvalidToolCallReason::MalformedArguments { .. }
    ));
}

#[test]
fn a_malformed_call_context_history_ends_at_the_turn_carrying_the_call() {
    let mut run = AgentRun::new("go").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(malformed_call_turn("c1"))
            .expect("model_response"),
    );
    let Some(PendingToolCall::Malformed(call)) = expect_call_tools(&mut run).pop() else {
        panic!("the call is malformed");
    };
    let context = run.malformed_context(&call, false);
    let Some(Message::Assistant(AssistantMessage { content, .. })) = context.chat_history.last()
    else {
        panic!(
            "the context history ends with the assistant turn: {:?}",
            context.chat_history
        );
    };
    assert!(
        content
            .iter()
            .any(|item| matches!(item, AssistantContent::ToolCall(item) if item.id == *call.id())),
        "the last turn carries the malformed call"
    );
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
    answer_calls(&mut run, "2").expect("tool results");

    // A later turn naming another output tool never unpins the first.
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(output_policy_turn("c2", "add", "other_result"))
            .expect("model_response"),
    );
    assert_eq!(run.output_tool_name(), Some("final_result"));
    expect_call_tools(&mut run);
    answer_calls(&mut run, "2").expect("tool results");

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
            .contains("the run is format 2, this rig reads format 4"),
        "{error}"
    );
}

#[test]
fn a_format_3_run_is_refused_by_name() {
    // Format 3 kept in-flight flags beside the state; a format-3 envelope is
    // refused rather than resumed with them dropped.
    let run = AgentRun::new("go");
    let mut value = serde_json::to_value(&run).expect("serialize");
    value["format"] = json!(3);
    let error = serde_json::from_value::<AgentRun>(value).expect_err("format 3 is refused");
    assert!(
        error
            .to_string()
            .contains("the run is format 3, this rig reads format 4"),
        "{error}"
    );
}

/// Transcript well-formedness has one meaning: whatever a run commits to
/// its own history must be accepted by `with_validated_history`, the
/// documented way to resume a run in another process (a `/reload`).
fn assert_resumable(history: Vec<Message>) {
    let result = AgentRun::new("continue").with_validated_history(history.clone());
    assert!(
        result.is_ok(),
        "a history the run produced is refused on resume: {:?}\nhistory: {history:#?}",
        result.err()
    );
}

/// A tool-bearing turn that fails is kept in the run's messages (for
/// display) and its calls are never answered; resuming that history
/// must still be accepted.
#[test]
fn a_history_with_a_failed_tool_turn_can_be_resumed() {
    let mut run = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(ended_call_turn(
            StopReason::Error("boom".to_owned()),
            FinishReason::Other("boom".to_owned()),
        ))
        .expect("model_response should succeed"),
    );
    run.next_step().expect_err("the failed turn ends the run");
    assert_resumable(run.full_history());
}

/// The run answers duplicate call ids one result per occurrence; the
/// history it commits must be resumable.
#[test]
fn a_history_with_duplicate_call_ids_can_be_resumed() {
    let mut run = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(ModelTurn::new(
            AssistantMessage::default(),
            vec![tool_call("call_1", "add"), tool_call("call_1", "add")],
            Usage::default(),
            policy(&["add"]),
            hand_raw(),
        ))
        .expect("model_response should succeed"),
    );
    let calls = expect_call_tools(&mut run);
    assert_eq!(calls.len(), 2);
    run.answer_all(calls.into_iter().map(|call| answer_open(call, "2")))
        .expect("the run accepts one answer per occurrence");
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(text_turn("done"))
            .expect("model_response should succeed"),
    );
    let response = expect_done(&mut run);
    assert_resumable(response.messages);
}

/// The host is told to run tools, then the process reloads before results
/// arrive: the run's cancel error carries its history, and that history
/// must be resumable.
#[test]
fn a_history_cancelled_while_tools_run_can_be_resumed() {
    let mut run = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("call_1", "add"))
            .expect("model_response should succeed"),
    );
    expect_call_tools(&mut run);
    let PromptError::Cancelled { chat_history, .. } = run.cancel_error("reload") else {
        panic!("cancel_error returns Cancelled");
    };
    assert_resumable(chat_history.into_vec());
}

/// The guard for the one pairing rule: at every step of these drives, the
/// history the run would hand a resuming process is a canonical transcript,
/// and so is every history an error carries.
#[test]
fn every_step_of_a_run_leaves_a_canonical_full_history() {
    fn check(run: &AgentRun) {
        let history = run.full_history();
        assert_eq!(
            validate_canonical(&history),
            Ok(()),
            "history: {history:#?}"
        );
    }
    fn check_error(error: PromptError) {
        let history = match error {
            PromptError::Cancelled { chat_history, .. } => chat_history.into_vec(),
            PromptError::MaxTurns { chat_history, .. } => chat_history,
            error => panic!("expected an error carrying history, got {error:?}"),
        };
        assert_eq!(
            validate_canonical(&history),
            Ok(()),
            "history: {history:#?}"
        );
    }
    let to_tools = |calls: Vec<AssistantContent>| {
        let mut run = AgentRun::new("add things").max_turns(2);
        check(&run);
        expect_call_model(&mut run);
        check(&run);
        expect_continue(
            run.model_response(ModelTurn::new(
                AssistantMessage::default(),
                calls,
                Usage::default(),
                policy(&["add"]),
                hand_raw(),
            ))
            .expect("model_response should succeed"),
        );
        check(&run);
        expect_call_tools(&mut run);
        check(&run);
        run
    };
    let pair = || {
        vec![
            tool_call("call_1", "add"),
            tool_call("call_1", "add"),
            tool_call("call_2", "add"),
        ]
    };

    // A round trip with a repeated id, to the end of the budget.
    let mut run = to_tools(pair());
    answer_calls(&mut run, "1").expect("one answer per occurrence is accepted");
    check(&run);
    expect_call_model(&mut run);
    check(&run);
    expect_continue(
        run.model_response(tool_call_turn("call_3", "add"))
            .expect("model_response should succeed"),
    );
    expect_call_tools(&mut run);
    check(&run);
    check_error(run.cancel_error("reload"));
    answer_calls(&mut run, "4").expect("the answer is accepted");
    check_error(run.next_step().expect_err("the budget is spent"));

    // A rejected answer while tools run carries a closed history.
    let mut run = to_tools(pair());
    check_error(
        run.answer(call_from_another_run("call_9").answer(success("2")))
            .expect_err("the answer is rejected"),
    );
    check(&run);

    // A failed tool-bearing turn stays in history, answered by nothing.
    let mut run = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(ended_call_turn(
            StopReason::Aborted("stopped".to_owned()),
            FinishReason::Other("stopped".to_owned()),
        ))
        .expect("model_response should succeed"),
    );
    run.next_step().expect_err("the failed turn ends the run");
    check(&run);

    // A skipped call's pre-resolved result closes it.
    let mut run = AgentRun::new("summarize")
        .max_turns(2)
        .with_output_tool_name("final_result");
    expect_call_model(&mut run);
    expect_needs_resolution(
        run.model_response(tool_call_turn("c1", "final_result"))
            .expect("model_response should succeed"),
    );
    check(&run);
    expect_continue(
        run.resolve_invalid_tool_call(InvalidToolCallAction::skip("not this turn"))
            .expect("the skip is accepted"),
    );
    // The skip answered the whole turn: it is committed and the run moves on.
    expect_call_model(&mut run);
    check(&run);
}

/// While tools run, `messages` is the raw view and `full_history` closes the
/// pending calls exactly as replay would answer them, so the first request
/// after a resume reads the same as one sent from the live run.
#[test]
fn full_history_closes_pending_calls_as_replay_does() {
    let mut run = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(ModelTurn::new(
            AssistantMessage::default(),
            vec![tool_call("call_1", "add"), tool_call("call_1", "add")],
            Usage::default(),
            policy(&["add"]),
            hand_raw(),
        ))
        .expect("model_response should succeed"),
    );
    let calls = expect_call_tools(&mut run);
    assert_eq!(run.messages().len(), 2);
    let history = run.full_history();
    assert_eq!(history.len(), 3);
    assert_eq!(history[..2], run.messages()[..]);
    assert_eq!(
        history[2],
        rig_core::transcript::close_pending(calls.iter().map(pending))
    );
    assert_eq!(
        rig_core::transcript::repair(run.messages().to_vec()).messages,
        history
    );
}

// Interrupted-step proofs: a run persisted while a step is in flight (the
// state a host serializes when a process restart, such as a coding agent's
// /reload, lands mid-turn) must resume, and a cancellation mid-step must
// leave a canonical history.

#[test]
fn run_persisted_mid_model_call_resumes_by_reissuing_the_call() {
    // The host emitted CallModel and started the request; the process died
    // before the response arrived. The persisted run must re-issue the same
    // model call after a restart rather than dead-end.
    let mut run = AgentRun::new("hi").max_turns(2);
    let (prompt, history, turn) = expect_call_model(&mut run);
    assert_eq!(turn, 1);

    let persisted = serde_json::to_string(&run).expect("in-flight run should serialize");
    let mut restored: AgentRun =
        serde_json::from_str(&persisted).expect("in-flight run should deserialize");

    let step = restored.next_step();
    let Ok(AgentRunStep::CallModel {
        prompt: resumed_prompt,
        history: resumed_history,
        turn: resumed_turn,
    }) = step
    else {
        panic!("a run persisted mid model call must re-issue CallModel on resume, got {step:?}");
    };
    assert_eq!(resumed_prompt, prompt);
    assert_eq!(resumed_history, history);
    assert_eq!(
        resumed_turn, turn,
        "the interrupted call must not consume a turn"
    );
}

#[test]
fn run_persisted_mid_stream_resumes_by_reissuing_the_call() {
    // Streamed variant: the driver recorded the stream's terminal usage, then
    // the process died before the assembled turn was fed. The provider
    // connection is gone, so the only way forward is to re-issue the call.
    let mut run = AgentRun::new("hi").max_turns(2);
    let (prompt, _, turn) = expect_call_model(&mut run);
    run.record_streamed_completion_call(
        usage(10, 5),
        ResponseIdentity::default(),
        None,
        hand_raw(),
    )
    .expect("record should succeed");

    let persisted = serde_json::to_string(&run).expect("mid-stream run should serialize");
    let mut restored: AgentRun =
        serde_json::from_str(&persisted).expect("mid-stream run should deserialize");

    let step = restored.next_step();
    let Ok(AgentRunStep::CallModel {
        prompt: resumed_prompt,
        turn: resumed_turn,
        ..
    }) = step
    else {
        panic!("a run persisted mid stream must re-issue CallModel on resume, got {step:?}");
    };
    assert_eq!(resumed_prompt, prompt);
    assert_eq!(resumed_turn, turn);
}

#[test]
fn cancel_mid_tool_batch_yields_canonical_history() {
    // A sans-IO host (Ctrl-C or /reload while tools run) cancels via
    // cancel_error. PromptError::Cancelled documents its history as
    // canonical; it must not end in an unanswered tool call.
    let mut run = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("call_1", "add"))
            .expect("model_response should succeed"),
    );
    expect_call_tools(&mut run);

    let PromptError::Cancelled { chat_history, .. } = run.cancel_error("interrupted") else {
        panic!("cancel_error must build a Cancelled error");
    };
    assert_eq!(
        rig_core::transcript::validate_canonical(&chat_history),
        Ok(()),
        "cancelled mid-batch history must be canonical: {chat_history:?}"
    );
}

fn two_call_turn() -> ModelTurn {
    ModelTurn::new(
        rig_core::message::AssistantMessage::default(),
        vec![tool_call("call_1", "add"), tool_call("call_2", "add")],
        Usage::default(),
        policy(&["add"]),
        hand_raw(),
    )
}

fn drive_to_two_pending_calls() -> AgentRun {
    let mut run = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(two_call_turn())
            .expect("model_response should succeed"),
    );
    assert_eq!(expect_call_tools(&mut run).len(), 2);
    run
}

fn call_ids(calls: &[PendingToolCall]) -> Vec<String> {
    calls
        .iter()
        .map(|call| pending(call).id.to_string())
        .collect()
}

#[test]
fn a_run_persisted_mid_batch_resumes_with_only_the_unanswered_calls() {
    let mut run = drive_to_two_pending_calls();
    let second = expect_call_tools(&mut run).remove(1);
    run.answer(answer_open(second, "3"))
        .expect("one answer is accepted on its own");

    let persisted = serde_json::to_string(&run).expect("mid-batch run should serialize");
    let mut restored: AgentRun =
        serde_json::from_str(&persisted).expect("mid-batch run should deserialize");

    let pending = expect_call_tools(&mut restored);
    assert_eq!(call_ids(&pending), ["call_1"]);
    // Idempotent: asking again re-emits the same remainder.
    assert_eq!(call_ids(&expect_call_tools(&mut restored)), ["call_1"]);

    answer_calls(&mut restored, "2").expect("the last answer completes the batch");
    let (prompt, _, turn) = expect_call_model(&mut restored);
    assert_eq!(turn, 2);
    // The results enter history in call order, not arrival order.
    assert_eq!(
        prompt,
        Message::User {
            content: vec![tool_result("call_1", "2"), tool_result("call_2", "3")]
        }
    );
    assert_eq!(validate_canonical(&restored.full_history()), Ok(()));
}

#[test]
fn reissuing_a_pending_model_call_consumes_no_turn() {
    let mut run = AgentRun::new("hi").max_turns(1);
    let issued = expect_call_model(&mut run);
    assert_eq!(expect_call_model(&mut run), issued);
    assert_eq!(run.turn(), 1);
    expect_continue(
        run.model_response(text_turn("hello"))
            .expect("the single budgeted turn is still answerable"),
    );
    assert_eq!(expect_done(&mut run).output(), "hello");
}

#[test]
fn a_reissued_stream_may_record_its_own_completion_call() {
    let mut run = AgentRun::new("hi").max_turns(1);
    expect_call_model(&mut run);
    let record = |run: &mut AgentRun| {
        run.record_streamed_completion_call(
            usage(10, 5),
            ResponseIdentity::default(),
            None,
            hand_raw(),
        )
    };
    record(&mut run).expect("the first attempt records");
    record(&mut run).expect_err("one attempt records once");
    expect_call_model(&mut run);
    record(&mut run).expect("the re-issued attempt records");
    // The abandoned attempt stays billed.
    assert_eq!(run.completion_calls().len(), 2);
    let mut billed = usage(10, 5);
    billed += usage(10, 5);
    assert_eq!(run.usage(), billed);
}

#[test]
fn a_call_is_answered_once() {
    let mut run = drive_to_two_pending_calls();
    let first = exec_call(expect_call_tools(&mut run).remove(0));
    run.answer(first.clone().answer(success("2")))
        .expect("one answer is accepted");
    let err = run
        .answer(first.answer(success("2")))
        .expect_err("a call is answered once");
    assert!(err.to_string().contains("already answered"), "{err}");
    assert_eq!(call_ids(&expect_call_tools(&mut run)), ["call_2"]);
}

#[test]
fn cancel_mid_batch_keeps_the_answered_results_and_aborts_the_rest() {
    let mut run = drive_to_two_pending_calls();
    let first = expect_call_tools(&mut run).remove(0);
    run.answer(answer_open(first, "2"))
        .expect("one answer is accepted");
    let PromptError::Cancelled { chat_history, .. } = run.cancel_error("interrupted") else {
        panic!("cancel_error must build a Cancelled error");
    };
    assert_eq!(validate_canonical(&chat_history), Ok(()));
    let Some(Message::User { content }) = chat_history.last() else {
        panic!("the batch must be closed: {chat_history:?}");
    };
    let [answered, UserContent::ToolResult(aborted)] = content.as_slice() else {
        panic!("one result per call: {content:?}");
    };
    assert_eq!(answered, &tool_result("call_1", "2"));
    assert!(aborted.is_error);
    assert_eq!(aborted.call.to_string(), "call_2");
    // A cancellation reports; the run is still mid-batch.
    assert_eq!(call_ids(&expect_call_tools(&mut run)), ["call_2"]);
}

#[test]
fn canonical_history_round_trips_and_deserializing_validates() {
    let mut run = drive_to_two_pending_calls();
    let history = run.canonical_history();
    let json = serde_json::to_value(&history).expect("history serializes");
    let back: CanonicalHistory = serde_json::from_value(json).expect("canonical history loads");
    assert_eq!(back, history);
    assert_eq!(back.into_iter().collect::<Vec<_>>(), history.into_vec());

    let open = run.messages().to_vec();
    assert!(CanonicalHistory::validate(open.clone()).is_err());
    let json = serde_json::to_value(&open).expect("history serializes");
    serde_json::from_value::<CanonicalHistory>(json).expect_err("an open tool call is refused");
    answer_calls(&mut run, "2").expect("the batch is still answerable");
}

/// A malformed-call limit fails the run before its turn enters history, so
/// a later cancellation's history is canonical.
#[test]
fn a_failed_malformed_limit_leaves_a_canonical_cancel_history() {
    let mut run = AgentRun::new("go")
        .max_turns(10)
        .max_consecutive_malformed_tool_calls(0);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(malformed_call_turn("c1"))
            .expect("model_response"),
    );
    run.next_step().expect_err("past the limit");
    let PromptError::Cancelled { chat_history, .. } = run.cancel_error("after failure") else {
        panic!("cancel_error builds a Cancelled error");
    };
    assert_eq!(chat_history.into_vec(), vec![Message::user("go")]);
}

/// A failed turn's calls stay in history for display and, owing no result,
/// stay as they are in any later cancellation's history.
#[test]
fn a_failed_turn_leaves_a_canonical_cancel_history() {
    let mut run = AgentRun::new("add things").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(ended_call_turn(
            StopReason::Error("x".to_owned()),
            FinishReason::Other("x".to_owned()),
        ))
        .expect("model_response"),
    );
    run.next_step().expect_err("failed turn");
    let error = run
        .next_step()
        .expect_err("protocol violation on a failed run");
    let PromptError::Cancelled { chat_history, .. } = error else {
        panic!("a protocol violation cancels, got {error:?}");
    };
    assert_eq!(
        validate_canonical(&chat_history),
        Ok(()),
        "{chat_history:?}"
    );
    assert_eq!(chat_history.into_vec(), run.full_history());
}

/// Every history the crate builds round-trips through serde, even from an
/// unchecked input history with an open call.
#[test]
fn a_canonical_history_from_an_unchecked_input_history_round_trips() {
    let open = vec![
        Message::user("q"),
        Message::Assistant(rig_core::message::AssistantMessage::new(vec![tool_call(
            "x", "add",
        )])),
    ];
    let run = AgentRun::new("hi").with_history(open);
    let history = run.canonical_history();
    assert_eq!(validate_canonical(&history), Ok(()), "{history:?}");
    let json = serde_json::to_value(&history).expect("serialize");
    let restored: CanonicalHistory = serde_json::from_value(json).expect("deserialize");
    assert_eq!(restored, history);
    let Message::User { content } = &history[2] else {
        panic!("the open call is answered before the prompt: {history:?}");
    };
    assert_eq!(content.len(), 2, "{content:?}");
}

/// A cut-off `write_file` call: the provider stopped at the token limit in
/// the middle of the `content` string.
fn truncated_write_turn(id: &str) -> ModelTurn {
    ModelTurn::new(
        AssistantMessage::default().with_stop(StopReason::Length),
        vec![AssistantContent::ToolCall(ToolCall::from_wire(
            id,
            ToolFunction::parse(
                rig_core::message::ToolName::new("write_file").expect("tool name"),
                r#"{"path":"a.rs","content":"fn ma"#,
            ),
        ))],
        Usage::default(),
        policy(&["write_file"]),
        hand_raw(),
    )
    .with_finish_reason(Some(FinishReason::Length))
}

/// Persist a run mid-step and resume it from bytes, as a host does across a
/// process restart.
fn persist_and_resume(run: AgentRun) -> AgentRun {
    let bytes = serde_json::to_vec(&run).expect("a pending run serializes");
    drop(run);
    serde_json::from_slice(&bytes).expect("a pending run deserializes")
}

/// The calls a driver that follows the `CallTools` contract executes: every
/// [`PendingToolCall::Execute`], with the call's arguments.
fn calls_a_driver_executes(calls: &[PendingToolCall]) -> Vec<(String, serde_json::Value)> {
    calls
        .iter()
        .filter_map(|call| match call {
            PendingToolCall::Execute(call) => Some((call.name().to_string(), call.arguments())),
            _ => None,
        })
        .collect()
}

/// A forged executable call: the step's first call re-tagged `Execute` in
/// its serialized form, as a host could write it to disk.
fn forge_exec_call(step: &AgentRunStep) -> ExecCall {
    let mut value = serde_json::to_value(step).expect("the step serializes");
    let call = value["CallTools"]["calls"][0]
        .as_object_mut()
        .and_then(|call| call.shift_remove("Malformed"))
        .expect("the first call is malformed");
    value["CallTools"]["calls"][0] = json!({ "Execute": call });
    match serde_json::from_value(value).expect("the forged step deserializes") {
        AgentRunStep::CallTools { mut calls } => exec_call(calls.remove(0)),
        step => panic!("expected CallTools, got {step:?}"),
    }
}

/// A host resumes a run whose last turn was cut off mid tool call. The run
/// knows the call's arguments are malformed, so the `CallTools` step must not
/// offer it as executable work: a host that runs every executable call
/// would write the salvaged prefix `fn ma` to `a.rs`.
#[test]
fn a_resumed_truncated_call_is_not_offered_as_executable() {
    let mut run = AgentRun::new("write a.rs").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(truncated_write_turn("c1"))
            .expect("model_response should succeed"),
    );
    let mut resumed = persist_and_resume(run);
    let calls = expect_call_tools(&mut resumed);
    assert_eq!(calls.len(), 1);
    let PendingToolCall::Malformed(call) = &calls[0] else {
        panic!("the truncated call is malformed: {calls:?}");
    };
    assert_eq!(call.raw_arguments(), r#"{"path":"a.rs","content":"fn ma"#);

    let executed = calls_a_driver_executes(&calls);
    assert!(
        executed.is_empty(),
        "CallTools offered a malformed call as executable work: {executed:?}"
    );
}

/// The run cannot be told a malformed call executed successfully. A
/// `MalformedCall` has no executed answer (see its `compile_fail` doctest),
/// and an `ExecCall` forged from its serialized form is refused at the run.
#[test]
fn an_executed_answer_for_a_malformed_call_is_rejected() {
    let mut run = AgentRun::new("write a.rs").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(truncated_write_turn("c1"))
            .expect("model_response should succeed"),
    );
    let mut resumed = persist_and_resume(run);
    let step = resumed.next_step().expect("the CallTools step");
    let messages = resumed.messages().len();

    let forged = forge_exec_call(&step);
    let error = resumed
        .answer(forged.answer(success("wrote 5 bytes to a.rs")))
        .expect_err("an executed answer for a malformed call is refused");
    assert!(error.to_string().contains("protocol violation"), "{error}");
    assert_eq!(resumed.messages().len(), messages, "nothing was committed");
    assert!(matches!(
        expect_call_tools(&mut resumed).as_slice(),
        [PendingToolCall::Malformed(_)]
    ));
}

/// A call the invalid-call hook skipped (here: forbidden by the active tool
/// choice, though the tool exists) is answered by the run. It is never
/// offered as a `CallTools` call, and a forged executed answer is refused.
#[test]
fn a_skipped_call_is_never_offered_and_cannot_be_answered() {
    let mut run = AgentRun::new("look around").max_turns(2);
    expect_call_model(&mut run);
    let context = expect_needs_resolution(
        run.model_response(ModelTurn::new(
            AssistantMessage::default(),
            vec![tool_call("c1", "shell")],
            Usage::default(),
            TurnPolicy::new(
                tool_names(&["read", "shell"]),
                Some(ToolChoice::Specific {
                    function_names: vec![
                        rig_core::message::ToolName::new("read").expect("tool name"),
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
    expect_continue(
        run.resolve_invalid_tool_call(InvalidToolCallAction::skip("read-only turn"))
            .expect("the skip is accepted"),
    );
    let mut resumed = persist_and_resume(run);
    // The skip answered the whole turn: the run goes back to the model.
    expect_call_model(&mut resumed);
    let Some(Message::User { content }) = resumed.messages().last().cloned() else {
        panic!("the skip result was committed");
    };
    assert!(matches!(
        content.as_slice(),
        [UserContent::ToolResult(result)] if result.is_error
    ));

    let forged: ExecCall = serde_json::from_value(json!({
        "turn": 1,
        "index": 0,
        "tool_call": serde_json::to_value(match tool_call("c1", "shell") {
            AssistantContent::ToolCall(call) => call,
            _ => panic!("a tool call"),
        })
        .expect("the call serializes"),
    }))
    .expect("the forged call deserializes");
    let messages = resumed.messages().len();
    resumed
        .answer(forged.answer(success("rm -rf target: done")))
        .expect_err("an executed result for a skipped call is refused");
    assert_eq!(resumed.messages().len(), messages, "nothing was committed");
}

/// A batch whose answers include one the run refuses stores none of them.
#[test]
fn answer_all_stores_nothing_when_one_answer_is_refused() {
    let mut run = AgentRun::new("add twice").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(run.model_response(two_call_turn()).expect("model_response"));
    let first = exec_call(expect_call_tools(&mut run).remove(0));
    run.answer_all([
        first.answer(success("first")),
        call_from_another_run("c9").answer(success("stray")),
    ])
    .expect_err("an answer for another call is refused");
    assert_eq!(expect_call_tools(&mut run).len(), 2, "no answer was stored");
}

/// A malformed call answered with `Stop` cancels the run with the history
/// before the batch.
#[test]
fn a_malformed_call_answered_with_stop_cancels_the_run() {
    let mut run = AgentRun::new("write a.rs").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(truncated_write_turn("c1"))
            .expect("model_response should succeed"),
    );
    let Some(PendingToolCall::Malformed(call)) = expect_call_tools(&mut run).pop() else {
        panic!("one malformed call");
    };
    let history = run.full_history();
    let error = run
        .answer(call.answer(Some(InvalidToolCallAction::Stop {
            reason: "stop".to_string(),
        })))
        .expect_err("stop cancels the run");
    assert!(matches!(
        error,
        PromptError::Cancelled { chat_history, .. } if *chat_history == history[..]
    ));
    assert!(run.next_step().is_err(), "the run has failed");
}

/// A malformed call's `Stop` applied with an executed sibling's answer in the
/// same `answer_all`: the run fails with that result kept, in its history
/// and the error's, not closed as unanswered.
#[test]
fn a_malformed_stop_keeps_an_answer_applied_with_it() {
    let mut run = AgentRun::new("add").max_turns(2);
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(ModelTurn::new(
            AssistantMessage::default(),
            vec![
                tool_call("c1", "add"),
                AssistantContent::ToolCall(ToolCall::from_wire(
                    "c2",
                    ToolFunction::parse(
                        rig_core::message::ToolName::new("add").expect("tool name"),
                        "{\"x\":",
                    ),
                )),
            ],
            Usage::default(),
            policy(&["add"]),
            hand_raw(),
        ))
        .expect("model_response"),
    );
    let mut calls = expect_call_tools(&mut run);
    let Some(PendingToolCall::Malformed(malformed)) = calls.pop() else {
        panic!("the second call is malformed");
    };
    let executed = exec_call(calls.remove(0));
    let error = run
        .answer_all([
            executed.answer(success("executed")),
            malformed.answer(Some(InvalidToolCallAction::Stop {
                reason: "stop".to_string(),
            })),
        ])
        .expect_err("stop cancels the run");
    let PromptError::Cancelled { chat_history, .. } = error else {
        panic!("a cancellation, got {error:?}");
    };
    let kept = |history: &[Message]| match history.last() {
        Some(Message::User { content }) => matches!(
            content.first(),
            Some(UserContent::ToolResult(result)) if result.content.iter().any(|part| matches!(
                part,
                rig_core::message::ToolResultContent::Text(text) if text.text == "executed"
            ))
        ),
        _ => false,
    };
    assert!(kept(&chat_history), "the error keeps it: {chat_history:?}");
    assert!(
        kept(run.messages()),
        "the run keeps it: {:?}",
        run.messages()
    );
    assert!(run.next_step().is_err(), "the run has failed");
}
