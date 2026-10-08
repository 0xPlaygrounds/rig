use super::super::policy::InvalidToolCallAction;
use super::super::response::PromptError;
use super::super::{AgentRun, AgentRunStep, PendingToolCall, ToolAnswer};
use super::*;
use rig_core::completion::{CompletionResponse, Usage};
use rig_core::message::ToolFunction;
use rig_core::streaming::Transcript;
use serde_json::json;

/// Answer an executable call with the success `2`.
fn answer_two(call: PendingToolCall) -> ToolAnswer {
    let PendingToolCall::Execute(call) = call else {
        panic!("expected an executable call, got {call:?}");
    };
    call.answer(rig_core::tool::ToolResult::success(
        rig_core::tool::ToolOutput::text("2"),
    ))
}

fn add_policy() -> TurnPolicy {
    TurnPolicy::new(["add".to_string()].into(), None, None).expect("policy")
}

fn assembler() -> StreamedTurnAssembler {
    StreamedTurnAssembler::new(add_policy())
}

fn tool_call(id: &str, name: &str) -> ToolCall {
    ToolCall::from_wire(
        id,
        ToolFunction::new(
            ToolName::new(name.to_string()).expect("tool name"),
            json!({"x": 1}),
        ),
    )
}

fn event(value: serde_json::Value) -> serde_json::Value {
    json!({"item": "event", "value": value})
}

fn text(part: u32, text: &str) -> Vec<serde_json::Value> {
    vec![
        event(json!({"event": "start", "part": part, "kind": "text"})),
        event(json!({"event": "text", "part": part, "text": text})),
        event(json!({"event": "end", "part": part, "content": AssistantContent::text(text)})),
    ]
}

/// A call as a wire that sends it whole streams it: its start naming the
/// tool, its arguments, its end.
fn call(part: u32, call: &ToolCall) -> Vec<serde_json::Value> {
    vec![
        event(json!({
            "event": "start",
            "part": part,
            "kind": "tool_call",
            "name": call.function.name,
        })),
        event(json!({
            "event": "arguments",
            "part": part,
            "json": call.function.arguments_value().to_string(),
        })),
        event(json!({
            "event": "end",
            "part": part,
            "content": AssistantContent::ToolCall(call.clone()),
        })),
    ]
}

/// The items of a stream, in order.
fn items(events: impl IntoIterator<Item = Vec<serde_json::Value>>) -> Vec<Item<StreamEvent>> {
    Transcript::parse_prefix(serde_json::Value::Array(
        events.into_iter().flatten().collect(),
    ))
    .expect("a stream in order")
    .into_items()
}

fn ingest_all(asm: &mut StreamedTurnAssembler, items: &[Item<StreamEvent>]) {
    for item in items {
        asm.ingest(item).expect("ingest");
    }
}

fn response(choice: Vec<AssistantContent>) -> CompletionResponse {
    CompletionResponse::new(
        choice,
        Usage::default(),
        rig_core::message::Origin::new("test.api", "mock", ""),
        json!({}),
    )
}

/// A mid-stream assembler survives a serde round trip: feeding the rest
/// of the stream to the restored assembler produces the same turn as an
/// uninterrupted run — a saved world can resume a streamed turn.
#[test]
fn assembler_round_trips_mid_stream() {
    let add = tool_call("tc1", "add");
    let items = items([text(0, "thinking "), call(1, &add)]);

    let mut uninterrupted = assembler();
    ingest_all(&mut uninterrupted, &items);

    let mut first_half = assembler();
    ingest_all(&mut first_half, &items[..4]);
    let json = serde_json::to_string(&first_half).expect("serialize");
    drop(first_half);
    let mut restored: StreamedTurnAssembler = serde_json::from_str(&json).expect("deserialize");
    ingest_all(&mut restored, &items[4..]);

    let response = response(vec![
        AssistantContent::text("thinking "),
        AssistantContent::ToolCall(add),
    ]);
    let direct = uninterrupted.finish(&response);
    let resumed = restored.finish(&response);
    assert_eq!(resumed.choice, direct.choice);
    assert_eq!(resumed.policy, direct.policy);
}

/// The decode-outcome contract, as a total matrix: every unknown payload
/// has exactly one of two outcomes — excluded-and-counted (one warning at
/// turn end), or excluded-quiet (provider-native unmodeled) — and no shape
/// is silent. `expected` is a wildcard-free match, so a new shape class
/// cannot compile without a mandated outcome, and the coverage assert
/// below fails until it also has a fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ShapeClass {
    WellFormedText,
    UnknownKeyedText,
    TaggedText,
    TaggedRigBlock,
    MalformedParamsText,
    /// A provider-native frame that happens to carry a string `text`
    /// key (e.g. an annotation event). Not rig content: it stays quiet.
    ProviderNativeTextCarrying,
    ProviderNativeUnmodeled,
}

#[derive(Debug, PartialEq)]
enum ExpectedOutcome {
    /// The payload is rig assistant content: excluding it from assembly
    /// loses transcript content, so the assembler counts it.
    ExcludedAndCounted,
    /// A provider-native shape rig does not model: excluded quietly.
    ExcludedQuiet,
}

/// The matrix's outcome column. No wildcard arm — the compiler is the
/// missing-cell error.
///
/// Every row reaches the assembler as an `Unknown` payload — the stream
/// vocabulary is typed, so a decoder emits text as a text event and only
/// unmodeled frames travel as `Unknown`. What the matrix pins is the
/// classification of what *does* arrive there: a tagged rig block (a
/// replayed assistant item, text included) is counted; a text carrier
/// whose params were malformed is counted; a bare or unknown-tagged object
/// is a provider-native shape and stays quiet.
fn expected(shape: ShapeClass) -> ExpectedOutcome {
    match shape {
        ShapeClass::TaggedText | ShapeClass::TaggedRigBlock | ShapeClass::MalformedParamsText => {
            ExpectedOutcome::ExcludedAndCounted
        }
        ShapeClass::WellFormedText
        | ShapeClass::UnknownKeyedText
        | ShapeClass::ProviderNativeTextCarrying
        | ShapeClass::ProviderNativeUnmodeled => ExpectedOutcome::ExcludedQuiet,
    }
}

/// The matrix's fixture rows. Every shape class appears at least once
/// (pinned by the coverage assert in the test); classes with several
/// wire spellings carry one fixture per spelling.
fn decode_matrix_cases() -> Vec<(ShapeClass, serde_json::Value)> {
    vec![
        (ShapeClass::WellFormedText, json!({"text": "hi"})),
        (
            ShapeClass::UnknownKeyedText,
            json!({"text": "hi", "citations": ["stray"], "future": 1}),
        ),
        (
            ShapeClass::TaggedText,
            json!({"type": "text", "text": "hi"}),
        ),
        (
            ShapeClass::TaggedRigBlock,
            json!({"type": "toolcall", "id": {"provider": "call_1"},
                   "function": {"name": "add", "arguments": {}}}),
        ),
        (
            ShapeClass::TaggedRigBlock,
            json!({"type": "image", "data": {"type": "base64", "value": "aGk="}}),
        ),
        (
            ShapeClass::MalformedParamsText,
            json!({"text": "hi", "native": []}),
        ),
        (
            ShapeClass::MalformedParamsText,
            json!({"type": "text", "text": "hi", "native": []}),
        ),
        (
            ShapeClass::ProviderNativeUnmodeled,
            json!({"type": "web_search_call", "id": "ws_1"}),
        ),
        (
            ShapeClass::ProviderNativeTextCarrying,
            json!({"type": "output_text.annotation", "text": "hi"}),
        ),
        (ShapeClass::ProviderNativeUnmodeled, json!({"text": 42})),
    ]
}

#[test]
fn decode_outcome_matrix_is_total_and_no_shape_is_silent() {
    let cases = decode_matrix_cases();
    // Vacuity floor: an emptied fixture table must fail loudly, not
    // pass by checking nothing.
    assert!(!cases.is_empty(), "decode_matrix_cases returned no rows");
    // Coverage: every shape class has at least one fixture. Extend
    // `witnesses` (and `decode_matrix_cases`) when adding a variant —
    // `expected` already refuses to compile without a classification.
    let witnesses = [
        ShapeClass::WellFormedText,
        ShapeClass::UnknownKeyedText,
        ShapeClass::TaggedText,
        ShapeClass::TaggedRigBlock,
        ShapeClass::MalformedParamsText,
        ShapeClass::ProviderNativeTextCarrying,
        ShapeClass::ProviderNativeUnmodeled,
    ];
    for shape in witnesses {
        assert!(
            cases.iter().any(|(case_shape, _)| *case_shape == shape),
            "no fixture for {shape:?} — add a row to decode_matrix_cases"
        );
    }

    for (shape, payload) in cases {
        let item = Item::Unknown(payload.clone().into());
        let mut asm = assembler();
        asm.ingest(&item).expect("ingest");
        assert_eq!(asm.aggregated_text(), "", "{shape:?}: {payload}");
        match expected(shape) {
            ExpectedOutcome::ExcludedAndCounted => {
                assert_eq!(
                    asm.excluded_assistant_content(),
                    1,
                    "{shape:?} loses assistant content and must be counted: {payload}"
                );
            }
            ExpectedOutcome::ExcludedQuiet => {
                assert_eq!(
                    asm.excluded_assistant_content(),
                    0,
                    "{shape:?} is provider-native and must stay quiet: {payload}"
                );
            }
        }
    }
}

/// An ignored call stays out of the turn across a checkpoint.
#[test]
fn an_ignored_call_stays_out_of_the_turn_after_a_checkpoint() {
    let multiply = tool_call("c1", "multiply");
    let mut asm = assembler();
    surface_invalid(&mut asm, &items([call(0, &multiply)]));
    assert!(
        asm.resolve_pending_invalid(&StreamedResolution::Ignored)
            .is_empty()
    );
    let asm: StreamedTurnAssembler =
        serde_json::from_str(&serde_json::to_string(&asm).unwrap()).unwrap();
    let turn = asm.finish(&response(vec![AssistantContent::ToolCall(multiply)]));
    assert!(turn.choice.is_empty(), "{:?}", turn.choice);
}

/// Nothing is ingested while an invalid call awaits its resolution.
#[test]
fn ingest_refuses_items_while_an_invalid_call_awaits_resolution() {
    let mut asm = assembler();
    let stream = items([call(0, &tool_call("c1", "multiply")), text(1, "after")]);
    surface_invalid(&mut asm, &stream[..3]);
    assert!(asm.ingest(&stream[3]).is_err());
}

/// Ingest `stream`, returning what its last item asks of the driver.
fn asm_ingest_to(
    asm: &mut StreamedTurnAssembler,
    stream: &[Item<StreamEvent>],
) -> Vec<StreamedTurnEvent> {
    let (last, before) = stream.split_last().expect("a stream");
    ingest_all(asm, before);
    asm.ingest(last).expect("ingest should succeed")
}

fn expect_invalid(events: Vec<StreamedTurnEvent>) -> StreamedInvalidToolCall {
    match events.into_iter().next() {
        Some(StreamedTurnEvent::InvalidToolCall(invalid)) => invalid,
        other => panic!("expected InvalidToolCall, got {other:?}"),
    }
}

/// Ingest a stream whose last item is an invalid call's end, and return
/// the call surfaced for resolution.
fn surface_invalid(
    asm: &mut StreamedTurnAssembler,
    stream: &[Item<StreamEvent>],
) -> StreamedInvalidToolCall {
    expect_invalid(asm_ingest_to(asm, stream))
}

#[test]
fn streamed_run_completes_a_tool_roundtrip() {
    let mut run = AgentRun::new("add things").max_turns(2);

    // Turn 1: the model streams one tool call.
    let AgentRunStep::CallModel { .. } = run.next_step().expect("next_step") else {
        panic!("expected CallModel");
    };
    let add = tool_call("tc_1", "add");
    let mut asm = assembler();
    ingest_all(&mut asm, &items([call(0, &add)]));
    let usage = Usage::new()
        .input_tokens(5)
        .output_tokens(7)
        .total_tokens(12);
    run.record_streamed_completion_call(
        usage,
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::json!({}),
    )
    .expect("record should succeed");
    let turn = asm.finish(&response(vec![AssistantContent::ToolCall(add.clone())]));
    run.streamed_turn(turn)
        .expect("streamed_turn should succeed");

    let AgentRunStep::CallTools { calls } = run.next_step().expect("next_step") else {
        panic!("expected CallTools");
    };
    assert_eq!(calls.len(), 1);
    assert!(matches!(&calls[0], PendingToolCall::Execute(call) if *call.id() == add.id));
    run.answer_all(calls.into_iter().map(answer_two))
        .expect("the answer should be accepted");

    // Turn 2: plain text finishes the run.
    let AgentRunStep::CallModel { .. } = run.next_step().expect("next_step") else {
        panic!("expected CallModel");
    };
    let asm = assembler();
    run.record_streamed_completion_call(
        Usage::default(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::json!({}),
    )
    .expect("record should succeed");
    run.streamed_turn(asm.finish(&response(vec![AssistantContent::text("done")])))
        .expect("streamed_turn should succeed");

    let AgentRunStep::Done(response) = run.next_step().expect("next_step") else {
        panic!("expected Done");
    };
    assert_eq!(response.output(), "done");
    assert_eq!(response.usage, usage);
    assert_eq!(response.completion_calls.len(), 2);
    assert_eq!(response.completion_calls[0].usage, usage);
    assert_eq!(response.completion_calls[1].usage, Usage::default());
    // prompt, assistant tool call, tool result, final assistant text
    assert_eq!(response.messages.len(), 4);
}

#[test]
fn streamed_invalid_tool_call_stop_leaves_run_terminal() {
    let mut run = AgentRun::new("use the tool");
    run.next_step().expect("next_step");

    let mut asm = assembler();
    let invalid = surface_invalid(
        &mut asm,
        &items([call(0, &tool_call("tc_1", "default_api"))]),
    );
    let partial = asm.partial_turn(&response(vec![]));

    let err = run
        .resolve_streamed_invalid_tool_call(
            &partial,
            &invalid,
            InvalidToolCallAction::stop("operator stop"),
        )
        .expect_err("stop should cancel the run");
    assert!(matches!(
        err,
        PromptError::Cancelled { reason, .. } if reason == "operator stop"
    ));

    let err = run
        .next_step()
        .expect_err("a stopped streamed run must remain terminal");
    assert!(matches!(
        err,
        PromptError::Cancelled { reason, .. }
            if reason.contains("next_step called after the run already failed")
    ));
}

#[test]
fn streamed_turn_rejects_unknown_tool_calls_fail_fast() {
    let mut run = AgentRun::new("use the tool");
    run.next_step().expect("next_step");
    record_terminal(&mut run);

    let turn = StreamedTurn {
        head: AssistantMessage::default(),
        choice: vec![AssistantContent::ToolCall(tool_call("tc_1", "unknown"))],
        policy: add_policy(),
        finish_reason: None,
    };
    let err = run
        .streamed_turn(turn)
        .expect_err("unknown tool should fail fast");
    assert!(matches!(
        err,
        PromptError::UnknownToolCall { tool_name, .. } if tool_name == "unknown"
    ));
}

#[test]
fn streamed_completion_call_record_requires_a_model_call() {
    // A fresh run has emitted no CallModel: recording must be rejected
    // even though the machine is in its initial PreparingRequest state.
    let mut run = AgentRun::new("hello");
    let err = run
        .record_streamed_completion_call(
            Usage::default(),
            rig_core::completion::ResponseIdentity::default(),
            None,
            serde_json::json!({}),
        )
        .expect_err("recording before any model call must be rejected");
    assert!(matches!(err, PromptError::Cancelled { .. }));

    // The run stays drivable.
    run.next_step().expect("next_step should still succeed");
    run.record_streamed_completion_call(
        Usage::default(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::json!({}),
    )
    .expect("recording during a pending model call succeeds");
}

/// Record the turn's completion call the way a driver does from the
/// stream's response, for tests that feed a hand-assembled turn.
fn record_terminal(run: &mut AgentRun) {
    run.record_streamed_completion_call(
        Usage::default(),
        Default::default(),
        None,
        json!({"origin": "hand-built terminal"}),
    )
    .expect("the turn's completion call records while the model call is pending");
}

/// The payload of a streamed turn lives on the response only the driver
/// sees, so a turn fed before the driver recorded its completion call is a
/// protocol violation: nothing is recorded on the run's behalf.
#[test]
fn streamed_turn_without_a_recorded_completion_call_is_a_protocol_violation() {
    let mut run = AgentRun::new("hello");
    run.next_step().expect("next_step");

    let asm = assembler();
    let err = run
        .streamed_turn(asm.finish(&response(vec![AssistantContent::text("done")])))
        .expect_err("a turn without its completion call recorded is refused");
    assert!(
        matches!(&err, PromptError::Cancelled { reason, .. }
            if reason.contains("record_streamed_completion_call")),
        "{err:?}"
    );
    assert!(run.completion_calls().is_empty());
}

#[test]
fn streamed_completion_call_is_recorded_once_per_turn() {
    let mut run = AgentRun::new("hello");
    run.next_step().expect("next_step");

    run.record_streamed_completion_call(
        Usage::default(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::json!({}),
    )
    .expect("first record succeeds");
    let err = run
        .record_streamed_completion_call(
            Usage::default(),
            rig_core::completion::ResponseIdentity::default(),
            None,
            serde_json::json!({}),
        )
        .expect_err("second record for the same turn must be rejected");
    assert!(matches!(err, PromptError::Cancelled { .. }));
    assert_eq!(run.completion_calls().len(), 1);
}

#[test]
fn streamed_run_serde_round_trips_while_tools_pend() {
    let mut run = AgentRun::new("add things").max_turns(2);
    run.next_step().expect("next_step");

    let add = tool_call("tc_1", "add");
    let mut asm = assembler();
    ingest_all(&mut asm, &items([call(0, &add)]));
    run.record_streamed_completion_call(
        Usage::default(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::json!({}),
    )
    .expect("record should succeed");
    run.streamed_turn(asm.finish(&response(vec![AssistantContent::ToolCall(add)])))
        .expect("streamed_turn should succeed");
    run.next_step().expect("CallTools step");

    let serialized = serde_json::to_string(&run).expect("serialize mid-run");
    let mut restored: AgentRun = serde_json::from_str(&serialized).expect("deserialize mid-run");
    let AgentRunStep::CallTools { calls } = restored.next_step().expect("CallTools step") else {
        panic!("expected CallTools");
    };
    restored
        .answer_all(calls.into_iter().map(answer_two))
        .expect("the answer should be accepted");
    assert!(matches!(
        restored.next_step().expect("next turn"),
        AgentRunStep::CallModel { turn: 2, .. }
    ));
}

#[test]
fn typed_namespaces_survive_pending_tool_checkpoints_and_completed_turn_reuse() {
    for reverse in [false, true] {
        for after_call_tools in [false, true] {
            let mut run = AgentRun::new("do both twice").max_turns(3);
            run.next_step().unwrap();
            for turn in 0..2 {
                let generated = ToolCall::new(
                    CallId::from_wire(""),
                    ToolFunction::new(ToolName::new("add").expect("tool name"), json!({"x": 1})),
                );
                let explicit = tool_call(&generated.id.wire(), "add");
                let mut calls = [generated, explicit];
                if reverse {
                    calls.reverse();
                }
                let mut asm = assembler();
                ingest_all(&mut asm, &items([call(0, &calls[0]), call(1, &calls[1])]));
                let choice = calls
                    .iter()
                    .cloned()
                    .map(AssistantContent::ToolCall)
                    .collect::<Vec<_>>();
                record_terminal(&mut run);
                run.streamed_turn(asm.finish(&response(choice))).unwrap();
                if after_call_tools {
                    run.next_step().unwrap();
                }
                let mut restored: AgentRun =
                    serde_json::from_value(serde_json::to_value(&run).unwrap()).unwrap();
                let AgentRunStep::CallTools {
                    calls: restored_calls,
                } = restored.next_step().unwrap()
                else {
                    panic!("pending tools");
                };
                let AgentRunStep::CallTools { calls: run_calls } = run.next_step().unwrap() else {
                    panic!("pending tools");
                };
                for (index, call) in restored_calls.iter().enumerate() {
                    assert!(
                        matches!(call, PendingToolCall::Execute(call) if *call.tool_call() == calls[index])
                    );
                }
                let before = serde_json::to_value(&restored).unwrap();
                assert!(
                    restored
                        .answer_all([
                            answer_two(restored_calls[0].clone()),
                            answer_two(restored_calls[0].clone()),
                        ])
                        .is_err()
                );
                assert_eq!(
                    before,
                    serde_json::to_value(&restored).unwrap(),
                    "duplicate namespace answer must not consume state"
                );
                restored
                    .answer_all(restored_calls.into_iter().rev().map(answer_two))
                    .unwrap();
                run.answer_all(run_calls.into_iter().rev().map(answer_two))
                    .unwrap();
                assert!(
                    matches!(restored.next_step().unwrap(), AgentRunStep::CallModel { turn: next, .. } if next == turn + 2)
                );
                run.next_step().unwrap();
                assert_eq!(
                    serde_json::to_value(&restored).unwrap(),
                    serde_json::to_value(&run).unwrap()
                );
                run = restored;
            }
        }
    }
}

/// A partial turn that produced nothing is no assistant message.
#[test]
fn an_empty_partial_turn_is_no_assistant_message() {
    let turn = PartialStreamedTurn {
        head: Default::default(),
        content: vec![AssistantContent::text("")],
        pending_tool_calls: Vec::new(),
    };
    assert_eq!(turn.assistant_message(None), None);
}
