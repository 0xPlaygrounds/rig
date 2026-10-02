use super::super::policy::InvalidToolCallAction;
use super::super::response::PromptError;
use super::super::{AgentRun, AgentRunStep};
use super::*;
use rig_core::completion::{CompletionResponse, Usage};
use rig_core::message::{ToolResultContent, UserContent};
use rig_core::streaming::Transcript;
use serde_json::json;

fn tool_names(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

fn assembler() -> StreamedTurnAssembler {
    StreamedTurnAssembler::new(tool_names(&["add"]), tool_names(&["add"]))
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

fn reasoning(part: u32, fragments: &[&str]) -> Vec<serde_json::Value> {
    let mut events = vec![event(
        json!({"event": "start", "part": part, "kind": "reasoning"}),
    )];
    events.extend(
        fragments
            .iter()
            .map(|text| event(json!({"event": "reasoning", "part": part, "text": text}))),
    );
    events
}

/// A call as every wire streams it: its start, its arguments, its end.
fn call(part: u32, call: &ToolCall) -> Vec<serde_json::Value> {
    vec![
        event(json!({"event": "start", "part": part, "kind": "tool_call"})),
        event(json!({
            "event": "arguments",
            "part": part,
            "json": call.function.arguments.to_string(),
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
    CompletionResponse::new(choice, Usage::default(), "mock", json!({}))
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
    let direct = uninterrupted.finish(Some("msg".to_string()), &response);
    let resumed = restored.finish(Some("msg".to_string()), &response);
    assert_eq!(resumed.choice, direct.choice);
    assert_eq!(resumed.executable_tool_names, direct.executable_tool_names);
    assert_eq!(resumed.allowed_tool_names, direct.allowed_tool_names);
}

#[test]
fn text_accumulates_and_emits() {
    let mut asm = assembler();
    let items = items([vec![
        event(json!({"event": "start", "part": 0, "kind": "text"})),
        event(json!({"event": "text", "part": 0, "text": "hel"})),
        event(json!({"event": "text", "part": 0, "text": "lo"})),
    ]]);
    for item in &items {
        let events = asm.ingest(item).expect("ingest should succeed");
        assert!(matches!(
            events.as_slice(),
            [StreamedTurnEvent::EmitIngested]
        ));
    }
    assert_eq!(asm.aggregated_text(), "hello");
}

#[test]
fn unknown_item_emits_to_consumer_without_touching_accumulation() {
    let mut asm = assembler();
    ingest_all(&mut asm, &items([text(0, "answer")]));

    let events = asm
        .ingest(&Item::Unknown(
            json!({ "type": "web_search_call", "id": "ws_1" }).into(),
        ))
        .expect("ingest unknown should succeed");

    // The unmodeled item is forwarded to the consumer ...
    assert!(matches!(
        events.as_slice(),
        [StreamedTurnEvent::EmitIngested]
    ));
    // ... but perturbs no accumulation state used to build the assistant message.
    assert_eq!(asm.aggregated_text(), "answer");
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
            json!({"type": "toolcall", "id": {"provider": {"call_id": "call_1"}},
                   "function": {"name": "add", "arguments": {}}}),
        ),
        (
            ShapeClass::TaggedRigBlock,
            json!({"type": "image", "data": {"type": "base64", "value": "aGk="}}),
        ),
        (
            ShapeClass::MalformedParamsText,
            json!({"text": "hi", "additional_params": []}),
        ),
        (
            ShapeClass::MalformedParamsText,
            json!({"type": "text", "text": "hi", "additional_params": []}),
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

/// Fragments are provisional; only the end releases a validated call.
#[test]
fn a_call_streams_fragments_before_its_end_validates_it() {
    let add = tool_call("tc1", "add");
    let mut asm = assembler();
    let events: Vec<_> = items([call(0, &add)])
        .iter()
        .map(|item| asm.ingest(item).expect("ingest"))
        .collect();
    assert!(matches!(
        events[0].as_slice(),
        [StreamedTurnEvent::EmitToolCallFragment]
    ));
    assert!(matches!(
        events[1].as_slice(),
        [StreamedTurnEvent::EmitToolCallFragment]
    ));
    assert!(matches!(
        events[2].as_slice(),
        [StreamedTurnEvent::EmitToolCall { call }] if *call == add
    ));
}

/// Reasoning accumulates per part, so interleaved parts stay apart.
#[test]
fn aggregated_reasoning_is_scoped_to_its_part() {
    let mut asm = assembler();
    ingest_all(
        &mut asm,
        &items([reasoning(0, &["a", "b"]), reasoning(1, &["c"])]),
    );
    assert_eq!(asm.aggregated_reasoning(0), Some("ab"));
    assert_eq!(asm.aggregated_reasoning(1), Some("c"));
    assert_eq!(asm.aggregated_reasoning(2), None);
}

/// The turn is the response's choice in start order: reasoning is not
/// regrouped ahead of text, an ignored call is left out, and a repaired
/// call carries its new name.
#[test]
fn finish_keeps_start_order_without_ignored_calls_and_with_repaired_names() {
    let mut asm = assembler();
    let ignored = tool_call("tc_ignored", "multiply");
    let repaired = tool_call("tc_repaired", "default_api");
    let stream = items([text(0, "hi"), call(1, &ignored), call(2, &repaired)]);
    expect_invalid(asm_ingest_to(&mut asm, &stream[..6]));
    asm.resolve_pending_invalid(&StreamedResolution::Ignored);
    let replayed = expect_invalid(asm_ingest_to(&mut asm, &stream[6..]));
    assert_eq!(replayed.tool_call, repaired);
    let released = asm.resolve_pending_invalid(&StreamedResolution::Repaired {
        tool_name: "add".to_owned(),
    });
    assert!(matches!(
        released.as_slice(),
        [StreamedTurnEvent::EmitToolCall { call }] if call.function.name == "add"
    ));

    let reasoning = AssistantContent::Reasoning(Reasoning::new("later").sealed("mock"));
    let turn = asm.finish(
        None,
        &response(vec![
            AssistantContent::text("hi"),
            AssistantContent::ToolCall(ignored),
            AssistantContent::ToolCall(repaired.clone()),
            reasoning.clone(),
        ]),
    );
    let mut renamed = repaired;
    renamed.function.name = ToolName::new("add").expect("tool name");
    assert_eq!(
        turn.choice,
        vec![
            AssistantContent::text("hi"),
            AssistantContent::ToolCall(renamed),
            reasoning,
        ]
    );
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
    let turn = asm.finish(None, &response(vec![AssistantContent::ToolCall(multiply)]));
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
    let usage = Usage {
        input_tokens: Some(5),
        output_tokens: Some(7),
        total_tokens: Some(12),
        ..Usage::default()
    };
    run.record_streamed_completion_call(
        usage,
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::json!({}),
    )
    .expect("record should succeed");
    let turn = asm.finish(
        Some("msg_1".to_string()),
        &response(vec![AssistantContent::ToolCall(add.clone())]),
    );
    run.streamed_turn(turn)
        .expect("streamed_turn should succeed");

    let AgentRunStep::CallTools { calls } = run.next_step().expect("next_step") else {
        panic!("expected CallTools");
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].tool_call.id, add.id);
    run.tool_results(vec![UserContent::tool_result(
        CallId::from_wire("tc_1"),
        ToolName::new("add").expect("tool name"),
        vec![ToolResultContent::text("2")],
    )])
    .expect("tool_results should succeed");

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
    run.streamed_turn(asm.finish(None, &response(vec![AssistantContent::text("done")])))
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
fn streamed_invalid_tool_call_retry_rolls_back_with_partial_turn() {
    let mut run = AgentRun::new("use the tool")
        .max_turns(2)
        .max_invalid_tool_call_retries(1);
    run.next_step().expect("next_step");

    let mut asm = assembler();
    let invalid = surface_invalid(
        &mut asm,
        &items([
            text(0, "thinking "),
            call(1, &tool_call("tc_1", "default_api")),
        ]),
    );
    let partial = asm.partial_turn(Some("msg_1".to_string()), &[]);
    assert_eq!(partial.text.as_deref(), Some("thinking "));

    let context = run.streamed_invalid_tool_call_context(&partial, &invalid);
    assert!(context.is_streaming);
    assert_eq!(context.tool_name, "default_api");
    assert_eq!(context.tool_call_id, Some(CallId::from_wire("tc_1")));

    let resolution = run
        .resolve_streamed_invalid_tool_call(
            &partial,
            &invalid,
            InvalidToolCallAction::retry("use add instead"),
        )
        .expect("retry should be accepted");
    assert!(matches!(
        resolution,
        StreamedResolution::TurnAbandoned {
            skipped_tool_result: None
        }
    ));
    asm.resolve_pending_invalid(&resolution);

    // Usage from the drained stream is recorded after the rollback.
    run.record_streamed_completion_call(
        Usage::default(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::json!({}),
    )
    .expect("record after rollback should succeed");

    // The rollback appended the partial assistant turn and feedback.
    assert_eq!(run.messages().len(), 3);
    let AgentRunStep::CallModel { turn, .. } = run.next_step().expect("next_step") else {
        panic!("expected CallModel retry");
    };
    assert_eq!(turn, 2);
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
    let partial = asm.partial_turn(Some("msg_1".to_string()), &[]);

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
fn streamed_invalid_tool_call_retry_cannot_emit_call_past_total_budget() {
    let mut run = AgentRun::new("use the tool")
        .max_turns(1)
        .max_invalid_tool_call_retries(1);
    run.next_step().expect("initial model call");

    let mut asm = assembler();
    let invalid = surface_invalid(
        &mut asm,
        &items([call(0, &tool_call("tc_1", "default_api"))]),
    );
    let partial = asm.partial_turn(Some("msg_1".to_string()), &[]);
    let resolution = run
        .resolve_streamed_invalid_tool_call(
            &partial,
            &invalid,
            InvalidToolCallAction::retry("use add instead"),
        )
        .expect("retry resolution should be accepted");
    assert!(matches!(
        resolution,
        StreamedResolution::TurnAbandoned {
            skipped_tool_result: None
        }
    ));
    run.record_streamed_completion_call(
        Usage::default(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::json!({}),
    )
    .expect("completion call should be recorded");
    assert_eq!(run.completion_calls().len(), 1);

    let err = run
        .next_step()
        .expect_err("retry must not emit a second model call");
    assert!(matches!(err, PromptError::MaxTurns { max_turns: 1, .. }));
    assert_eq!(run.turn(), 1);
}

#[test]
fn streamed_invalid_tool_call_skip_returns_synthetic_result() {
    let mut run = AgentRun::new("use the tool").max_turns(2);
    run.next_step().expect("next_step");

    let mut asm = assembler();
    let invalid = surface_invalid(
        &mut asm,
        &items([call(0, &tool_call("tc_1", "default_api"))]),
    );
    let partial = asm.partial_turn(None, &[]);

    let resolution = run
        .resolve_streamed_invalid_tool_call(
            &partial,
            &invalid,
            InvalidToolCallAction::skip("not available"),
        )
        .expect("skip should be accepted");
    let StreamedResolution::TurnAbandoned {
        skipped_tool_result: Some(tool_result),
    } = &resolution
    else {
        panic!("expected skipped tool result");
    };
    assert_eq!(
        tool_result
            .call
            .provider()
            .map(|provider| provider.call_id.as_str()),
        Some("tc_1")
    );
}

#[test]
fn streamed_invalid_tool_call_repair_releases_the_renamed_call() {
    let mut run = AgentRun::new("use the tool").max_turns(2);
    run.next_step().expect("next_step");

    let mut asm = assembler();
    let invalid = surface_invalid(
        &mut asm,
        &items([call(0, &tool_call("tc_1", "default_api"))]),
    );
    assert_eq!(invalid.args.as_deref(), Some("{\"x\":1}"));

    let partial = asm.partial_turn(None, &[]);
    let resolution = run
        .resolve_streamed_invalid_tool_call(
            &partial,
            &invalid,
            InvalidToolCallAction::repair("add"),
        )
        .expect("repair should be accepted");
    assert!(matches!(
        resolution,
        StreamedResolution::Repaired { ref tool_name } if tool_name == "add"
    ));

    let events = asm.resolve_pending_invalid(&resolution);
    let [StreamedTurnEvent::EmitToolCall { call }] = events.as_slice() else {
        panic!("expected the repaired call, got {events:?}");
    };
    assert_eq!(call.function.name, "add");
    assert_eq!(call.function.arguments, json!({"x": 1}));
    assert_eq!(call.id, CallId::from_wire("tc_1"));
}

#[test]
fn streamed_turn_rejects_unknown_tool_calls_fail_fast() {
    let mut run = AgentRun::new("use the tool");
    run.next_step().expect("next_step");
    record_terminal(&mut run);

    let turn = StreamedTurn {
        message_id: None,
        choice: vec![AssistantContent::ToolCall(tool_call("tc_1", "unknown"))],
        executable_tool_names: tool_names(&["add"]),
        allowed_tool_names: tool_names(&["add"]),
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
        .streamed_turn(asm.finish(None, &response(vec![AssistantContent::text("done")])))
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
    run.streamed_turn(asm.finish(None, &response(vec![AssistantContent::ToolCall(add)])))
        .expect("streamed_turn should succeed");
    run.next_step().expect("CallTools step");

    let serialized = serde_json::to_string(&run).expect("serialize mid-run");
    let mut restored: AgentRun = serde_json::from_str(&serialized).expect("deserialize mid-run");
    restored
        .tool_results(vec![UserContent::tool_result(
            CallId::from_wire("tc_1"),
            ToolName::new("add").expect("tool name"),
            vec![ToolResultContent::text("2")],
        )])
        .expect("tool_results should succeed");
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
                run.streamed_turn(asm.finish(None, &response(choice)))
                    .unwrap();
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
                run.next_step().unwrap();
                for (index, call) in restored_calls.iter().enumerate() {
                    assert_eq!(call.tool_call, calls[index]);
                }
                let results = calls
                    .iter()
                    .rev()
                    .map(|call| {
                        UserContent::tool_result(
                            call.id.clone(),
                            ToolName::new("add").expect("tool name"),
                            vec![ToolResultContent::text("2")],
                        )
                    })
                    .collect::<Vec<_>>();
                let before = serde_json::to_value(&restored).unwrap();
                assert!(
                    restored
                        .tool_results(vec![results[0].clone(), results[0].clone()])
                        .is_err()
                );
                assert_eq!(
                    before,
                    serde_json::to_value(&restored).unwrap(),
                    "duplicate namespace answer must not consume state"
                );
                restored.tool_results(results.clone()).unwrap();
                run.tool_results(results).unwrap();
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

/// The partial turn's reasoning is what the stream folded so far, sealed
/// to its issuer.
#[test]
fn a_partial_turn_carries_the_ended_reasoning_sealed_to_its_issuer() {
    let asm = assembler();
    let sealed = Reasoning::new("because").sealed("mock");
    let partial = asm.partial_turn(
        None,
        &[
            AssistantContent::Reasoning(sealed.clone()),
            AssistantContent::text("ignored: text is the assembler's"),
        ],
    );
    assert_eq!(partial.reasoning, vec![sealed]);
    assert_eq!(partial.text, None);
}

/// A partial turn that produced nothing is no assistant message.
#[test]
fn an_empty_partial_turn_is_no_assistant_message() {
    let turn = PartialStreamedTurn {
        message_id: None,
        text: Some(String::new()),
        reasoning: Vec::new(),
        pending_tool_calls: Vec::new(),
    };
    assert_eq!(turn.assistant_message(None), None);
}
