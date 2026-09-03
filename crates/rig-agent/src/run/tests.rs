use super::*;
use rig_core::message::{ToolFunction, ToolResultContent};
use rig_core::streaming::BlockId;
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
fn runs_serialized_without_entries_still_deserialize() {
    let run = AgentRun::new("p");
    let mut value = serde_json::to_value(&run).expect("serializes");
    let object = value.as_object_mut().expect("object");
    assert!(!object.contains_key("entries"), "absent when empty");
    let restored: AgentRun = serde_json::from_value(value).expect("deserializes");
    assert!(restored.entries().is_empty());
}

#[test]
fn entries_never_reach_the_protocol_steps() {
    // The no-context-leakage guarantee at the protocol level: two
    // identical runs, one carrying entries, emit identical CallModel
    // steps — entries are storage, not context.
    let mut plain = AgentRun::new("p").max_turns(2);
    let mut logged = AgentRun::new("p").max_turns(2);
    logged.append_entry(entry("state", 0, json!({"visible": "never"})));

    let plain_step = plain.next_step().expect("step");
    let logged_step = logged.next_step().expect("step");
    let (
        AgentRunStep::CallModel {
            prompt: plain_prompt,
            history: plain_history,
            turn: plain_turn,
        },
        AgentRunStep::CallModel {
            prompt: logged_prompt,
            history: logged_history,
            turn: logged_turn,
        },
    ) = (plain_step, logged_step)
    else {
        panic!("both runs emit CallModel first");
    };
    assert_eq!(plain_prompt, logged_prompt);
    assert_eq!(plain_history, logged_history);
    assert_eq!(plain_turn, logged_turn);
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

#[test]
fn from_response_matches_hand_assembly_field_for_field() {
    let resp = CompletionResponse::new(
        vec![AssistantContent::text("hi"), tool_call("call_1", "add")],
        usage(11, 7),
        "openai",
    )
    .with_message_id("msg_1".to_string())
    .with_response_id("chatcmpl_1".to_string())
    .with_provider_request_id("req_1".to_string())
    .with_finish_reason(FinishReason::ToolCalls)
    .with_raw(json!({"provider": "payload"}));

    let executable = tool_names(&["add"]);
    let allowed = tool_names(&["add", "final_output"]);
    let turn = ModelTurn::from_response_parts(&resp, executable.clone(), allowed.clone());

    let expected = ModelTurn::new(
        resp.message_id.clone(),
        resp.choice.clone(),
        resp.usage,
        executable,
        allowed,
    )
    .with_identity(resp.response_id.clone(), resp.provider_request_id.clone())
    .with_finish_reason(resp.finish_reason())
    .with_raw(resp.raw.clone());

    assert_eq!(turn.message_id, expected.message_id);
    assert_eq!(turn.response_id, expected.response_id);
    assert_eq!(turn.provider_request_id, expected.provider_request_id);
    assert_eq!(turn.choice, expected.choice);
    assert_eq!(turn.usage, expected.usage);
    assert_eq!(turn.executable_tool_names, expected.executable_tool_names);
    assert_eq!(turn.allowed_tool_names, expected.allowed_tool_names);
    assert_eq!(turn.finish_reason, expected.finish_reason);
    assert_eq!(turn.raw, expected.raw);
}

fn tool_names(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
    Usage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
        ..Usage::new()
    }
}

fn text_turn(text: &str) -> ModelTurn {
    ModelTurn::new(
        None,
        vec![AssistantContent::text(text)],
        Usage::new(),
        tool_names(&["add"]),
        tool_names(&["add"]),
    )
}

fn tool_call(id: &str, name: &str) -> AssistantContent {
    // The provider-boundary shape: a non-empty wire id becomes both the
    // durable id and the provider correlator; an empty wire id mints a
    // fresh unique handle (`provider` records the absence).
    AssistantContent::ToolCall(ToolCall::from_wire(
        id,
        ToolFunction::new(name.to_string(), json!({"x": 1})),
    ))
}

fn tool_call_turn(id: &str, name: &str) -> ModelTurn {
    ModelTurn::new(
        None,
        vec![tool_call(id, name)],
        Usage::new(),
        tool_names(&["add"]),
        tool_names(&["add"]),
    )
}

fn tool_result(id: &str, output: &str) -> UserContent {
    // Every result in these tests answers a call to the `add` tool; the
    // executed tool's name is required data on a result.
    UserContent::tool_result(id, "add", vec![ToolResultContent::text(output)])
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
fn text_only_run_completes_in_one_turn() {
    let mut run = AgentRun::new("hello");

    let (prompt, history, turn) = expect_call_model(&mut run);
    assert_eq!(prompt, Message::user("hello"));
    assert!(history.is_empty());
    assert_eq!(turn, 1);

    let suppressed = expect_continue(
        run.model_response(text_turn("hi there"))
            .expect("model_response should succeed"),
    );
    assert!(!suppressed);

    let response = expect_done(&mut run);
    assert_eq!(response.output, "hi there");
    let messages = response.messages.expect("messages should be recorded");
    assert_eq!(messages.len(), 2);
    assert!(run.is_done());
}

#[test]
fn input_history_prefixes_request_history() {
    let mut run = AgentRun::new("question")
        .with_history(vec![Message::user("earlier"), Message::assistant("reply")]);

    let (_, history, _) = expect_call_model(&mut run);
    assert_eq!(
        history,
        vec![Message::user("earlier"), Message::assistant("reply")]
    );

    expect_continue(
        run.model_response(text_turn("answer"))
            .expect("model_response should succeed"),
    );
    let response = expect_done(&mut run);
    // Returned messages exclude the input history.
    assert_eq!(
        response
            .messages
            .expect("messages should be recorded")
            .len(),
        2
    );
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
    assert_eq!(response.output, "accepted");
    assert_eq!(response.usage, first_usage + second_usage);
    assert_eq!(response.completion_calls.len(), 2);
    let messages = response.messages.expect("response history");
    assert_eq!(messages.len(), 2);
    assert!(!format!("{messages:?}").contains("rejected"));
}

#[test]
fn feedback_retry_records_rejected_response_and_corrective_prompt() {
    let mut run = AgentRun::new("question").max_turns(2);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(text_turn("rejected"))
            .expect("first response"),
    );
    run.retry_model_turn(RetryRequest::Feedback("try another approach".to_string()))
        .expect("feedback retry should be accepted");

    let (prompt, history, turn) = expect_call_model(&mut run);
    assert_eq!(prompt, Message::user("try another approach"));
    assert_eq!(turn, 2);
    assert_eq!(
        history,
        vec![Message::user("question"), Message::assistant("rejected")]
    );
}

#[test]
fn repeated_model_turn_consumes_existing_max_turns_budget() {
    let mut run = AgentRun::new("question");

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(text_turn("rejected"))
            .expect("first response"),
    );
    run.retry_model_turn(RetryRequest::Repeat)
        .expect("state transition itself should succeed");

    let err = run.next_step().expect_err("second call must exceed budget");
    assert!(matches!(
        err,
        PromptError::MaxTurnsError { max_turns: 1, .. }
    ));
    assert_eq!(run.completion_calls().len(), 1);
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

    let PromptError::PromptCancelled {
        chat_history,
        reason,
    } = err
    else {
        panic!("tool-bearing retry should return PromptCancelled");
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
    assert_eq!(response.output, "the answer is 2");
    assert_eq!(response.usage, usage(30, 12));
    assert_eq!(response.completion_calls.len(), 2);
    assert_eq!(response.completion_calls[0].call_index, 0);
    assert_eq!(response.completion_calls[0].usage, usage(10, 5));
    assert_eq!(response.completion_calls[1].usage, usage(20, 7));
    // prompt, assistant tool call, tool result, final assistant text
    assert_eq!(
        response
            .messages
            .expect("messages should be recorded")
            .len(),
        4
    );
}

#[test]
fn parallel_tool_calls_surface_in_emission_order() {
    let mut run = AgentRun::new("do both").max_turns(2);

    expect_call_model(&mut run);
    let turn = ModelTurn::new(
        None,
        vec![tool_call("call_1", "add"), tool_call("call_2", "add")],
        Usage::new(),
        tool_names(&["add"]),
        tool_names(&["add"]),
    );
    expect_continue(
        run.model_response(turn)
            .expect("model_response should succeed"),
    );

    let calls = expect_call_tools(&mut run);
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].tool_call.id, "call_1");
    assert_eq!(calls[1].tool_call.id, "call_2");

    // Results fed out of order still land in one user message.
    run.tool_results(vec![tool_result("call_2", "b"), tool_result("call_1", "a")])
        .expect("tool_results should succeed");
    let messages = run.messages();
    assert!(matches!(
        messages.last(),
        Some(Message::User { content }) if content.len() == 2
    ));
}

#[test]
fn max_turns_zero_rejects_initial_model_call() {
    let mut run = AgentRun::new("do not call").max_turns(0);

    let err = run
        .next_step()
        .expect_err("zero budget should emit no call");
    assert!(matches!(
        err,
        PromptError::MaxTurnsError { max_turns: 0, .. }
    ));
    assert_eq!(run.turn(), 0);
}

#[test]
fn new_implicitly_allows_one_model_call_and_rejects_tool_continuation() {
    let mut run = AgentRun::new("add things");

    let (_, _, turn) = expect_call_model(&mut run);
    assert_eq!(turn, 1);
    expect_continue(
        run.model_response(tool_call_turn("call_1", "add"))
            .expect("model_response should succeed"),
    );
    expect_call_tools(&mut run);
    run.tool_results(vec![tool_result("call_1", "2")])
        .expect("tool_results should succeed");

    let err = run
        .next_step()
        .expect_err("second model call should exceed budget");
    assert!(matches!(
        err,
        PromptError::MaxTurnsError { max_turns: 1, .. }
    ));
    assert_eq!(run.turn(), 1);
}

#[test]
fn max_turns_n_allows_exactly_n_model_calls() {
    let mut run = AgentRun::new("loop").max_turns(3);

    for (expected_turn, call_id) in [(1, "call_1"), (2, "call_2"), (3, "call_3")] {
        let (_, _, turn) = expect_call_model(&mut run);
        assert_eq!(turn, expected_turn);
        expect_continue(
            run.model_response(tool_call_turn(call_id, "add"))
                .expect("model_response should succeed"),
        );
        expect_call_tools(&mut run);
        run.tool_results(vec![tool_result(call_id, "0")])
            .expect("tool_results should succeed");
    }

    let err = run
        .next_step()
        .expect_err("fourth model call should exceed budget");
    assert!(matches!(
        err,
        PromptError::MaxTurnsError { max_turns: 3, .. }
    ));
    assert_eq!(run.turn(), 3);
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
fn invalid_tool_call_stop_leaves_run_terminal() {
    let mut run = AgentRun::new("call something");

    expect_call_model(&mut run);
    expect_needs_resolution(
        run.model_response(tool_call_turn("call_1", "unknown"))
            .expect("model_response should succeed"),
    );
    let err = run
        .resolve_invalid_tool_call(InvalidToolCallAction::stop("operator stop"))
        .expect_err("stop should cancel the run");
    assert!(matches!(
        err,
        PromptError::PromptCancelled { reason, .. } if reason == "operator stop"
    ));

    let err = run
        .next_step()
        .expect_err("a stopped run must remain terminal");
    assert!(matches!(
        err,
        PromptError::PromptCancelled { reason, .. }
            if reason.contains("next_step called after the run already failed")
    ));
}

#[test]
fn invalid_tool_call_retry_rolls_back_with_feedback() {
    let mut run = AgentRun::new("call something")
        .max_turns(2)
        .max_invalid_tool_call_retries(1);

    expect_call_model(&mut run);
    expect_needs_resolution(
        run.model_response(tool_call_turn("call_1", "unknown"))
            .expect("model_response should succeed"),
    );
    let outcome = run
        .resolve_invalid_tool_call(InvalidToolCallAction::retry("use add instead"))
        .expect("retry should be accepted");
    assert!(matches!(outcome, ModelTurnOutcome::TurnRetried));

    // The rolled-back turn appended the assistant message and feedback.
    assert_eq!(run.messages().len(), 3);
    let (prompt, _, turn) = expect_call_model(&mut run);
    assert_eq!(turn, 2);
    assert!(matches!(
        prompt,
        Message::User { ref content }
            if matches!(content.first(), Some(UserContent::ToolResult(_)))
    ));

    // Budget of one: a second retry fails with UnknownToolCall.
    expect_needs_resolution(
        run.model_response(tool_call_turn("call_2", "unknown"))
            .expect("model_response should succeed"),
    );
    let err = run
        .resolve_invalid_tool_call(InvalidToolCallAction::retry("again"))
        .expect_err("budget exhausted");
    assert!(matches!(err, PromptError::UnknownToolCall { .. }));
}

#[test]
fn invalid_tool_call_retry_cannot_emit_call_past_total_budget() {
    let mut run = AgentRun::new("call something")
        .max_turns(1)
        .max_invalid_tool_call_retries(1);

    expect_call_model(&mut run);
    expect_needs_resolution(
        run.model_response(tool_call_turn("call_1", "unknown"))
            .expect("model_response should succeed"),
    );
    let outcome = run
        .resolve_invalid_tool_call(InvalidToolCallAction::retry("use add instead"))
        .expect("retry resolution should be accepted");
    assert!(matches!(outcome, ModelTurnOutcome::TurnRetried));
    assert_eq!(run.completion_calls().len(), 1);

    let err = run
        .next_step()
        .expect_err("retry must not emit a second model call");
    assert!(matches!(
        err,
        PromptError::MaxTurnsError { max_turns: 1, .. }
    ));
    assert_eq!(run.turn(), 1);
}

#[test]
fn invalid_tool_call_repair_renames_and_suppresses_response_hook() {
    let mut run = AgentRun::new("call something").max_turns(2);

    expect_call_model(&mut run);
    expect_needs_resolution(
        run.model_response(tool_call_turn("call_1", "default_api"))
            .expect("model_response should succeed"),
    );
    let suppressed = expect_continue(
        run.resolve_invalid_tool_call(InvalidToolCallAction::repair("add"))
            .expect("repair should be accepted"),
    );
    assert!(suppressed);

    let calls = expect_call_tools(&mut run);
    assert_eq!(calls[0].tool_call.function.name, "add");
    assert!(calls[0].preresolved_result.is_none());
}

#[test]
fn invalid_tool_call_repair_to_disallowed_name_fails() {
    let mut run = AgentRun::new("call something");

    expect_call_model(&mut run);
    expect_needs_resolution(
        run.model_response(tool_call_turn("call_1", "unknown"))
            .expect("model_response should succeed"),
    );
    let err = run
        .resolve_invalid_tool_call(InvalidToolCallAction::repair("also_unknown"))
        .expect_err("repair to disallowed name should fail");
    assert!(matches!(
        err,
        PromptError::UnknownToolCall { tool_name, .. } if tool_name == "also_unknown"
    ));
}

#[test]
fn invalid_tool_call_skip_suppresses_all_peer_executions() {
    let mut run = AgentRun::new("call things").max_turns(2);

    expect_call_model(&mut run);
    let turn = ModelTurn::new(
        None,
        vec![tool_call("call_1", "unknown"), tool_call("call_2", "add")],
        Usage::new(),
        tool_names(&["add"]),
        tool_names(&["add"]),
    );
    expect_needs_resolution(
        run.model_response(turn)
            .expect("model_response should succeed"),
    );
    let suppressed = expect_continue(
        run.resolve_invalid_tool_call(InvalidToolCallAction::skip("not available"))
            .expect("skip should be accepted"),
    );
    assert!(suppressed);

    let calls = expect_call_tools(&mut run);
    assert_eq!(calls.len(), 2);
    // Both the skipped call and its valid peer carry preresolved results.
    assert!(calls.iter().all(|call| call.preresolved_result.is_some()));
}

/// Two ID-LESS calls (older ollama daemons issue no tool-call id) must
/// not collide in the skipped map: each mints its own unique correlation
/// handle at the provider boundary, so skipping the invalid one leaves
/// the valid peer with its own "not executed" result, and each call reads
/// back its OWN preresolved result — position, not id, is the key.
#[test]
fn id_less_calls_keep_distinct_skip_results() {
    let mut run = AgentRun::new("call things").max_turns(2);

    expect_call_model(&mut run);
    let turn = ModelTurn::new(
        None,
        vec![tool_call("", "unknown"), tool_call("", "add")],
        Usage::new(),
        tool_names(&["add"]),
        tool_names(&["add"]),
    );
    expect_needs_resolution(
        run.model_response(turn)
            .expect("model_response should succeed"),
    );
    let suppressed = expect_continue(
        run.resolve_invalid_tool_call(InvalidToolCallAction::skip("not available"))
            .expect("skip should be accepted"),
    );
    assert!(suppressed);

    let calls = expect_call_tools(&mut run);
    assert_eq!(calls.len(), 2);
    assert_ne!(
        calls[0].tool_call.id, calls[1].tool_call.id,
        "id-less calls mint distinct correlation handles, never a shared sentinel"
    );
    assert!(
        calls
            .iter()
            .all(|call| !call.tool_call.id.is_empty() && call.tool_call.provider.is_none()),
        "minted handles are non-empty and record the provider's absence"
    );
    let results: Vec<String> = calls
        .iter()
        .map(|call| match call.preresolved_result.as_ref() {
            Some(rig_core::message::UserContent::ToolResult(result)) => result
                .content
                .iter()
                .filter_map(|content| match content {
                    rig_core::message::ToolResultContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect(),
            _ => panic!("both calls carry preresolved results"),
        })
        .collect();
    assert_eq!(
        results[0], "not available",
        "the skipped call reads its own feedback"
    );
    assert_eq!(
        results[1], TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER,
        "the peer reads its own synthetic result, not the skipped one\'s"
    );
}

#[test]
fn skip_under_tool_choice_none_fails() {
    let mut run = AgentRun::new("call something").with_tool_choice(ToolChoice::None);

    expect_call_model(&mut run);
    expect_needs_resolution(
        run.model_response(ModelTurn::new(
            None,
            vec![tool_call("call_1", "add")],
            Usage::new(),
            tool_names(&["add"]),
            BTreeSet::new(),
        ))
        .expect("model_response should succeed"),
    );
    let err = run
        .resolve_invalid_tool_call(InvalidToolCallAction::skip("nope"))
        .expect_err("skip under ToolChoice::None should fail");
    assert!(matches!(err, PromptError::UnknownToolCall { .. }));
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
        PromptError::PromptCancelled { reason, .. }
            if reason.contains("tool execution produced no tool results")
    ));
}

#[test]
fn out_of_protocol_calls_are_rejected_without_corrupting_state() {
    let mut run = AgentRun::new("hello");

    let err = run
        .tool_results(vec![tool_result("call_1", "x")])
        .expect_err("no CallTools pending");
    assert!(matches!(err, PromptError::PromptCancelled { .. }));

    // The run is still drivable after a rejected out-of-protocol call.
    expect_call_model(&mut run);
    let err = run
        .next_step()
        .expect_err("model response is pending, next_step must be rejected");
    assert!(matches!(err, PromptError::PromptCancelled { .. }));
    expect_continue(
        run.model_response(text_turn("hi"))
            .expect("model_response should still succeed"),
    );
    assert_eq!(expect_done(&mut run).output, "hi");
}

#[test]
fn model_response_rejected_after_streamed_completion_call_record() {
    let mut run = AgentRun::new("hello");
    expect_call_model(&mut run);
    run.record_streamed_completion_call(
        Usage::new(),
        ResponseIdentity::default(),
        None,
        serde_json::Value::Null,
    )
    .expect("record should succeed");

    let err = run
        .model_response(text_turn("hi"))
        .expect_err("mixed streamed/non-streamed ingestion must be rejected");
    assert!(matches!(err, PromptError::PromptCancelled { .. }));
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
    assert_eq!(expect_done(&mut run).output, "hi");
    assert_eq!(expect_done(&mut run).output, "hi");
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
        .map(|call| tool_result(&call.tool_call.id, "2"))
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
    assert_eq!(expect_done(&mut resumed).output, "done");
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
    assert!(matches!(err, PromptError::PromptCancelled { .. }));
    run.tool_results(vec![tool_result("call_1", "2")])
        .expect("valid results should still be accepted after a rejection");

    // Leaving a pending call unanswered is rejected.
    let mut run = drive_to_pending_tools();
    let err = run
        .tool_results(vec![tool_result("call_1", "2"), tool_result("call_1", "3")])
        .expect_err("answering one call twice must be rejected");
    assert!(matches!(err, PromptError::PromptCancelled { .. }));

    // Non-tool-result content is rejected.
    let mut run = drive_to_pending_tools();
    let err = run
        .tool_results(vec![UserContent::text("not a tool result")])
        .expect_err("non-tool-result content must be rejected");
    assert!(matches!(err, PromptError::PromptCancelled { .. }));
}

#[test]
fn agent_run_deserializes_suspended_state() {
    // A suspended run persisted mid-`ExecutingTools` restores and resumes:
    // the recorded call's usage loads, the pending tool call is re-issued,
    // and the run advances to the next model call after results arrive.
    let fixture = r#"{"max_turns":2,"max_invalid_tool_call_retries":0,"tool_choice":null,"chat_history":null,"new_messages":[{"role":"user","content":[{"type":"text","text":"add things"}]},{"role":"assistant","id":null,"content":[{"type":"toolcall","id":"call_1","function":{"name":"add","arguments":{"x":1}},"signature":null,"additional_params":null}]}],"current_turn":1,"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"cached_input_tokens":0,"cache_creation_input_tokens":0,"tool_use_prompt_tokens":0,"reasoning_tokens":0},"completion_calls":[{"call_index":0,"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0,"cached_input_tokens":0,"cache_creation_input_tokens":0,"tool_use_prompt_tokens":0,"reasoning_tokens":0}}],"completion_call_index":1,"invalid_tool_call_retries":0,"rollback_pending":false,"streamed_completion_call_recorded":false,"state":{"ExecutingTools":[{"tool_call":{"id":"call_1","function":{"name":"add","arguments":{"x":1}},"signature":null,"additional_params":null},"preresolved_result":null,"block_id":"wire:call_1"}]}}"#;

    let mut restored: AgentRun =
        serde_json::from_str(fixture).expect("suspended run should deserialize");
    assert_eq!(restored.completion_calls()[0].usage, Usage::new());

    let calls = expect_call_tools(&mut restored);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].block_id, BlockId::wire("call_1"));
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
    assert!(matches!(
        err,
        PromptError::MaxTurnsError { max_turns: 1, .. }
    ));
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
    // The pending call's block id is part of the persisted state, not a
    // default a decoder fills in: a resumed process keeps the id its
    // consumers already saw.
    assert!(
        serialized.contains("\"block_id\""),
        "the pending call persists its block id: {serialized}"
    );
    let restored: AgentRun =
        serde_json::from_str(&serialized).expect("mid-run state should deserialize");
    let resumed = finish(restored);

    assert_eq!(resumed.output, uninterrupted.output);
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
    // A buffered turn's call is keyed by its durable id, live and resumed
    // alike — the same key its pending call would carry.
    assert_eq!(context.block_id, Some(BlockId::wire("call_1")));
    assert_eq!(restored_context.block_id, context.block_id);
}

/// A turn calling `name`, advertising it as an allowed-but-not-executable
/// tool (the shape Tool output mode produces — see #1928).
fn output_tool_turn(id: &str, name: &str) -> ModelTurn {
    ModelTurn::new(
        None,
        vec![tool_call(id, name)],
        Usage::new(),
        tool_names(&["add"]),
        tool_names(&["add", name]),
    )
}

fn output_tool_turn_with_args(id: &str, name: &str, arguments: serde_json::Value) -> ModelTurn {
    ModelTurn::new(
        None,
        vec![AssistantContent::ToolCall(ToolCall::from_wire(
            id,
            ToolFunction::new(name.to_string(), arguments),
        ))],
        Usage::new(),
        tool_names(&["add"]),
        tool_names(&["add", name]),
    )
}

/// Every assistant tool call in `messages` must have a matching user tool
/// result — an unanswered tool_use is rejected by providers on replay.
fn assert_no_orphan_tool_use(messages: &[Message]) {
    let mut answered = BTreeSet::new();
    for message in messages {
        if let Message::User { content } = message {
            for item in content.iter() {
                if let UserContent::ToolResult(result) = item {
                    answered.insert(result.call.to_string());
                }
            }
        }
    }
    for message in messages {
        if let Message::Assistant { content, .. } = message {
            for item in content.iter() {
                if let AssistantContent::ToolCall(call) = item {
                    assert!(
                        answered.contains(call.id.as_str()),
                        "assistant tool_call {:?} has no matching tool_result in history",
                        call.id
                    );
                }
            }
        }
    }
}

#[test]
fn output_tool_call_finalizes_run_with_arguments() {
    let mut run = AgentRun::new("summarize").with_output_tool_name("final_result");

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(output_tool_turn("call_1", "final_result"))
            .expect("model_response should succeed"),
    );

    // The output tool is not executed; its arguments become the run output.
    let response = expect_done(&mut run);
    assert_eq!(response.output, r#"{"x":1}"#);
    assert!(run.is_done());

    // The finalizing turn is persisted as assistant text, not as the raw
    // output-tool call, so the saved history has no dangling tool_use.
    let messages = response.messages.expect("messages should be recorded");
    assert_no_orphan_tool_use(&messages);
    assert!(matches!(
        messages.last(),
        Some(Message::Assistant { content, .. })
            if assistant_text_from_choice(content) == r#"{"x":1}"#
    ));
}

#[test]
fn scalar_output_tool_call_is_serialized_as_reparseable_json() {
    let mut run = AgentRun::new("summarize").with_output_tool_name("final_result");

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(output_tool_turn_with_args(
            "call_1",
            "final_result",
            json!("complete"),
        ))
        .expect("model_response should succeed"),
    );

    let response = expect_done(&mut run);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&response.output)
            .expect("scalar output must remain valid JSON"),
        json!("complete")
    );
    assert_eq!(response.output, r#""complete""#);

    let messages = response.messages.expect("messages should be recorded");
    assert_no_orphan_tool_use(&messages);
    assert!(matches!(
        messages.last(),
        Some(Message::Assistant { content, .. })
            if assistant_text_from_choice(content) == r#""complete""#
    ));
}

#[test]
fn output_tool_call_wins_over_sibling_real_tool_calls() {
    let mut run = AgentRun::new("do it")
        .max_turns(2)
        .with_output_tool_name("final_result");

    expect_call_model(&mut run);
    // The model emits a real tool call *and* the output tool in one turn;
    // the output-tool intercept wins and the real call is never executed.
    let turn = ModelTurn::new(
        None,
        vec![
            tool_call("call_1", "add"),
            tool_call("call_2", "final_result"),
        ],
        Usage::new(),
        tool_names(&["add"]),
        tool_names(&["add", "final_result"]),
    );
    expect_continue(
        run.model_response(turn)
            .expect("model_response should succeed"),
    );

    let response = expect_done(&mut run);
    assert_eq!(response.output, r#"{"x":1}"#);
    assert!(run.is_done());

    // Both the sibling `add` call and the output-tool call are dropped from
    // the persisted assistant message, leaving no unanswered tool_use.
    let messages = response.messages.expect("messages should be recorded");
    assert_no_orphan_tool_use(&messages);
    assert!(
        messages.iter().all(|message| match message {
            Message::Assistant { content, .. } => !content
                .iter()
                .any(|item| matches!(item, AssistantContent::ToolCall(_))),
            _ => true,
        }),
        "no assistant tool calls should survive in the finalized history"
    );
}

#[test]
fn real_tool_calls_still_execute_when_output_tool_unused() {
    // With an output tool configured but only real tools called, the run
    // proceeds to tool execution as normal (the intercept must not fire).
    let mut run = AgentRun::new("add things")
        .max_turns(2)
        .with_output_tool_name("final_result");

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("call_1", "add"))
            .expect("model_response should succeed"),
    );

    let calls = expect_call_tools(&mut run);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool_call.function.name, "add");
}

fn required_field_schema(field: &str) -> serde_json::Value {
    json!({
        "type": "object",
        "required": [field],
        "properties": { field: { "type": "string" } },
    })
}

#[test]
fn tool_mode_reprompts_when_output_tool_not_called() {
    // #1928: in Tool mode the model finalized with plain text instead of
    // calling the output tool, so the run re-prompts (within budget).
    let mut run = AgentRun::new("summarize")
        .max_turns(2)
        .with_output_tool_name("final_result")
        .with_output_validation(Some(required_field_schema("summary")), 1);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(text_turn("here is the answer"))
            .expect("model_response should succeed"),
    );

    // Instead of finalizing, the run emits a second CallModel with corrective
    // feedback naming the output tool.
    let (prompt, _history, turn) = expect_call_model(&mut run);
    assert_eq!(turn, 2);
    let prompt_json = serde_json::to_string(&prompt).expect("prompt should serialize");
    assert!(
        prompt_json.contains("final_result"),
        "re-prompt feedback should name the output tool: {prompt_json}"
    );
    assert!(!run.is_done());
}

#[test]
fn tool_mode_reprompts_when_output_args_missing_required_fields() {
    // #1928: the output tool was called but its arguments omit a required
    // field, so the run re-prompts rather than finalizing invalid output.
    let mut run = AgentRun::new("summarize")
        .max_turns(2)
        .with_output_tool_name("final_result")
        // `output_tool_turn` calls with args {"x":1}; require a different key.
        .with_output_validation(Some(required_field_schema("summary")), 1);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(output_tool_turn("call_1", "final_result"))
            .expect("model_response should succeed"),
    );

    let (_prompt, _history, turn) = expect_call_model(&mut run);
    assert_eq!(turn, 2);
    assert!(!run.is_done());
}

#[test]
fn tool_mode_accepts_valid_json_text_without_reprompting() {
    // The model returned valid structured output as plain text instead of an
    // output-tool call — accept it rather than wasting a turn re-prompting.
    let mut run = AgentRun::new("summarize")
        .max_turns(3)
        .with_output_tool_name("final_result")
        .with_output_validation(Some(required_field_schema("summary")), 1);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(text_turn(r#"{"summary":"all good"}"#))
            .expect("model_response should succeed"),
    );

    let response = expect_done(&mut run);
    assert_eq!(response.output, r#"{"summary":"all good"}"#);
    assert!(run.is_done());
}

#[test]
fn tool_mode_finalizes_best_effort_when_model_call_budget_exhausted() {
    let mut run = AgentRun::new("summarize")
        .max_turns(1)
        .with_output_tool_name("final_result")
        .with_output_validation(Some(required_field_schema("summary")), 1);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(text_turn("invalid output"))
            .expect("model_response should succeed"),
    );

    let response = expect_done(&mut run);
    assert_eq!(response.output, "invalid output");
    assert_eq!(run.turn(), 1);
}

#[test]
fn tool_mode_finalizes_best_effort_when_output_retry_budget_exhausted() {
    // With no retry budget, invalid output finalizes best-effort (the caller
    // validates) rather than looping — and history stays free of orphan
    // tool_use.
    let mut run = AgentRun::new("summarize")
        .max_turns(3)
        .with_output_tool_name("final_result")
        .with_output_validation(Some(required_field_schema("summary")), 0);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(output_tool_turn("call_1", "final_result"))
            .expect("model_response should succeed"),
    );

    let response = expect_done(&mut run);
    assert_eq!(response.output, r#"{"x":1}"#);
    let messages = response.messages.expect("messages should be recorded");
    assert_no_orphan_tool_use(&messages);
}

#[test]
fn set_output_tool_name_is_idempotent_and_only_fills_when_unset() {
    // A pre-set name (e.g. via `with_output_tool_name`) is never overwritten,
    // keeping a resumed run deterministic.
    let mut run = AgentRun::new("x").with_output_tool_name("first");
    run.set_output_tool_name(Some("second".to_string()));
    run.set_output_tool_name(None);
    assert_eq!(run.output_tool_name.as_deref(), Some("first"));

    // When unset, the first non-None value fills it.
    let mut run = AgentRun::new("x");
    run.set_output_tool_name(None);
    assert_eq!(run.output_tool_name, None);
    run.set_output_tool_name(Some("filled".to_string()));
    assert_eq!(run.output_tool_name.as_deref(), Some("filled"));
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
            None,
            two_calls,
            Usage::new(),
            tool_names(&["add"]),
            tool_names(&["add"]),
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
    assert_eq!(calls[0].tool_call.id, "c1");
    assert_eq!(calls[1].tool_call.id, "c2");

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
    assert_eq!(response.output, "done");
}

// ---------------------------------------------------------------------
// Raw provider response capture (always on), at the state-machine layer:
// the drivers hand `AgentRun` the payload they read off the provider
// response (blocking) or the stream terminal (streamed); the run must
// record it per call, and persisted run state must carry it across a
// suspend/resume boundary — while state written before the field existed
// still loads. A `Value::Null` here means the turn was built without a provider
// response behind it (hand-built, or persisted before the field existed),
// never that capture was declined.
// ---------------------------------------------------------------------

fn raw_payload(attempt: &str) -> serde_json::Value {
    json!({
        "id": format!("resp-{attempt}"),
        "provider_only": attempt,
    })
}

#[test]
fn model_turn_raw_is_recorded_on_the_completion_call() {
    let first = raw_payload("turn-1");
    let second = raw_payload("turn-2");
    let mut run = AgentRun::new("add things").max_turns(2);

    expect_call_model(&mut run);
    expect_continue(
        run.model_response(tool_call_turn("call_1", "add").with_raw(first.clone()))
            .expect("model_response should succeed"),
    );
    expect_call_tools(&mut run);
    run.tool_results(vec![tool_result("call_1", "2")])
        .expect("tool_results should succeed");
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(text_turn("done").with_raw(second.clone()))
            .expect("model_response should succeed"),
    );

    let response = expect_done(&mut run);
    let raws: Vec<_> = response
        .completion_calls
        .iter()
        .map(|call| call.raw.clone())
        .collect();
    assert_eq!(
        raws,
        [first, second],
        "each call carries its own turn's payload"
    );
}

/// A `ModelTurn` built without `with_raw` has no provider response behind
/// it, so its record carries `Value::Null` — the only way a record ends up
/// without a payload.
#[test]
fn model_turn_without_raw_records_null() {
    let mut run = AgentRun::new("hello");
    expect_call_model(&mut run);
    expect_continue(
        run.model_response(text_turn("hi"))
            .expect("model_response should succeed"),
    );
    assert_eq!(run.completion_calls()[0].raw, serde_json::Value::Null);
}

#[test]
fn streamed_completion_call_record_carries_raw() {
    let raw = raw_payload("streamed");
    let mut run = AgentRun::new("hello");
    expect_call_model(&mut run);
    let call = run
        .record_streamed_completion_call(
            usage(3, 4),
            ResponseIdentity::default(),
            None,
            raw.clone(),
        )
        .expect("record should succeed");
    assert_eq!(call.raw, raw);
    assert_eq!(run.completion_calls()[0].raw, raw);

    let mut run = AgentRun::new("hello");
    expect_call_model(&mut run);
    let call = run
        .record_streamed_completion_call(
            usage(3, 4),
            ResponseIdentity::default(),
            None,
            serde_json::Value::Null,
        )
        .expect("record should succeed");
    assert_eq!(
        call.raw,
        serde_json::Value::Null,
        "a terminal with no payload behind it records Value::Null"
    );
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
        run.model_response(tool_call_turn("call_1", "add").with_raw(raw.clone()))
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

/// `ModelTurn` carries `raw` through its own serde round trip, and a
/// turn serialized before the field existed (no `raw` key) still loads
/// with `raw` as `Value::Null`.
#[test]
fn model_turn_raw_round_trips_and_missing_key_loads_as_null() {
    let raw = raw_payload("turn");
    let turn = text_turn("hi").with_raw(raw.clone());

    let value = serde_json::to_value(&turn).expect("turn should serialize");
    assert_eq!(value["raw"], raw);
    let restored: ModelTurn =
        serde_json::from_value(value.clone()).expect("turn should deserialize");
    assert_eq!(restored.raw, raw);

    let mut without_raw = value;
    without_raw
        .as_object_mut()
        .expect("turn serializes as an object")
        .remove("raw")
        .expect("the raw key was present");
    let legacy: ModelTurn =
        serde_json::from_value(without_raw).expect("a turn without a raw key still loads");
    assert_eq!(legacy.raw, serde_json::Value::Null);
    assert_eq!(legacy.choice, turn.choice);
}

/// The same for a persisted `CompletionCall`: `raw` is skipped when `Value::Null`
/// (state written before the field is byte-identical), and a record
/// without the key loads with `raw` as `Value::Null`.
#[test]
fn completion_call_raw_round_trips_and_missing_key_loads_as_null() {
    let raw = raw_payload("call");
    let call = CompletionCall::new(0, usage(1, 2)).with_raw(raw.clone());

    let value = serde_json::to_value(&call).expect("call should serialize");
    assert_eq!(value["raw"], raw);
    let restored: CompletionCall = serde_json::from_value(value).expect("call should deserialize");
    assert_eq!(restored, call);

    let unset =
        serde_json::to_value(CompletionCall::new(0, usage(1, 2))).expect("call should serialize");
    assert!(
        unset.get("raw").is_none(),
        "a Value::Null raw is not written, so pre-field state is unchanged"
    );
    let legacy: CompletionCall =
        serde_json::from_value(unset).expect("a call without a raw key still loads");
    assert_eq!(legacy.raw, serde_json::Value::Null);
}

#[test]
fn from_spec_matches_the_builder_chain() {
    let spec = RunSpec {
        max_turns: Some(4),
        max_invalid_tool_call_retries: 2,
        tool_choice: Some(ToolChoice::Auto),
        ..RunSpec::new()
    };
    let via_spec = AgentRun::from_spec(&spec, "hi", None);
    let via_chain = AgentRun::new("hi")
        .max_turns(4)
        .max_invalid_tool_call_retries(2)
        .with_output_validation(None, RunSpec::DEFAULT_OUTPUT_RETRIES)
        .with_tool_choice(ToolChoice::Auto);
    assert_eq!(
        serde_json::to_value(&via_spec).expect("serialize"),
        serde_json::to_value(&via_chain).expect("serialize")
    );
}

fn assistant(content: Vec<AssistantContent>) -> Message {
    Message::Assistant { id: None, content }
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
