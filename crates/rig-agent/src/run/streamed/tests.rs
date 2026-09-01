use super::super::policy::InvalidToolCallAction;
use super::super::response::PromptError;
use super::super::{AgentRun, AgentRunStep};
use super::*;
use rig_core::message::{Text, ToolResultContent, UserContent};
use rig_core::test_utils::mock_final;
use serde_json::json;

fn tool_names(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

/// A mid-stream assembler survives a serde round trip: feeding the rest
/// of the stream to the restored assembler produces the same turn as an
/// uninterrupted run — a saved world can resume a streamed turn.
#[test]
fn assembler_round_trips_mid_stream() {
    let items = [
        text_item("thinking "),
        name_delta("tc1", "add"),
        args_delta("tc1", "{\"x\":"),
        args_delta("tc1", "1}"),
        tool_call_item("tc1", "add"),
    ];

    let mut uninterrupted = assembler();
    for item in &items {
        uninterrupted.ingest(item).expect("ingest");
    }

    let mut first_half = assembler();
    for item in &items[..2] {
        first_half.ingest(item).expect("ingest");
    }
    let json = serde_json::to_string(&first_half).expect("serialize");
    drop(first_half);
    let mut restored: StreamedTurnAssembler = serde_json::from_str(&json).expect("deserialize");
    for item in &items[2..] {
        restored.ingest(item).expect("ingest");
    }

    let final_choice = vec![AssistantContent::ToolCall(tool_call("tc1", "add"))];
    let direct = uninterrupted.finish(Some("msg".to_string()), &final_choice);
    let resumed = restored.finish(Some("msg".to_string()), &final_choice);
    assert_eq!(resumed.choice, direct.choice);
    assert_eq!(resumed.internal_call_ids, direct.internal_call_ids);
    assert_eq!(resumed.executable_tool_names, direct.executable_tool_names);
    assert_eq!(resumed.allowed_tool_names, direct.allowed_tool_names);
}

fn assembler() -> StreamedTurnAssembler {
    StreamedTurnAssembler::new(tool_names(&["add"]), tool_names(&["add"]))
}

fn text_item(text: &str) -> StreamedAssistantContent {
    StreamedAssistantContent::Text(Text::new(text.to_string()))
}

/// Deterministic test-only internal id derived from a wire id, so a
/// delta and its completed call correlate without threading state.
fn iid_for(id: &str) -> InternalCallId {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in id.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    InternalCallId::from_raw(hash | 1).expect("non-zero")
}

fn tool_call(id: &str, name: &str) -> ToolCall {
    // The provider-boundary shape: the wire id becomes both the durable
    // id and the provider correlator.
    ToolCall::from_wire(id, ToolFunction::new(name.to_string(), json!({"x": 1})))
}

fn tool_call_item(id: &str, name: &str) -> StreamedAssistantContent {
    StreamedAssistantContent::ToolCall {
        tool_call: tool_call(id, name),
        internal_call_id: iid_for(id),
    }
}

fn final_item() -> StreamedAssistantContent {
    StreamedAssistantContent::Final(mock_final(Usage::new()))
}

fn name_delta(id: &str, name: &str) -> StreamedAssistantContent {
    StreamedAssistantContent::ToolCallDelta {
        internal_call_id: iid_for(id),
        content: ToolCallDeltaContent::Name(name.to_string()),
    }
}

fn args_delta(id: &str, arguments: &str) -> StreamedAssistantContent {
    StreamedAssistantContent::ToolCallDelta {
        internal_call_id: iid_for(id),
        content: ToolCallDeltaContent::Delta(arguments.to_string()),
    }
}

#[test]
fn text_accumulates_and_emits() {
    let mut asm = assembler();
    let events = asm
        .ingest(&text_item("hel"))
        .expect("ingest should succeed");
    assert!(matches!(
        events.as_slice(),
        [StreamedTurnEvent::EmitIngested]
    ));
    asm.ingest(&text_item("lo")).expect("ingest should succeed");
    assert_eq!(asm.aggregated_text(), "hello");
}

#[test]
fn unknown_item_emits_to_consumer_without_touching_accumulation() {
    let mut asm = assembler();
    asm.ingest(&text_item("answer"))
        .expect("ingest text should succeed");

    let events = asm
        .ingest(&StreamedAssistantContent::Unknown(
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

/// The decode-outcome contract, as a total matrix: every stream-item
/// payload has exactly one of three outcomes — assembled,
/// excluded-and-counted (one warning at turn end), or excluded-quiet
/// (provider-native unmodeled) — and no shape is silent. `expected` is a
/// wildcard-free match, so a new shape class cannot compile without a
/// mandated outcome, and the coverage assert below fails until it also
/// has a fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ShapeClass {
    WellFormedText,
    UnknownKeyedText,
    TaggedText,
    TaggedRigBlock,
    MalformedParamsText,
    /// A provider-native frame that happens to carry a string `text`
    /// key (e.g. an annotation event). Tolerance folds its text into
    /// the message — the documented noise tradeoff: never losing real
    /// text outranks occasionally ingesting a frame's caption.
    ProviderNativeTextCarrying,
    ProviderNativeUnmodeled,
}

#[derive(Debug, PartialEq)]
enum ExpectedOutcome {
    Assembled { text: &'static str },
    ExcludedAndCounted,
    ExcludedQuiet,
}

/// The matrix's outcome column. No wildcard arm — the compiler is the
/// missing-cell error.
fn expected(shape: ShapeClass) -> ExpectedOutcome {
    match shape {
        ShapeClass::WellFormedText
        | ShapeClass::UnknownKeyedText
        | ShapeClass::TaggedText
        | ShapeClass::ProviderNativeTextCarrying => ExpectedOutcome::Assembled { text: "hi" },
        ShapeClass::TaggedRigBlock | ShapeClass::MalformedParamsText => {
            ExpectedOutcome::ExcludedAndCounted
        }
        ShapeClass::ProviderNativeUnmodeled => ExpectedOutcome::ExcludedQuiet,
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
            json!({"type": "toolcall", "id": "call_1",
                   "function": {"name": "add", "arguments": {}}}),
        ),
        (
            ShapeClass::TaggedRigBlock,
            json!({"type": "reasoning", "id": null, "content": []}),
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
        let item = serde_json::from_value::<StreamedAssistantContent>(payload.clone())
            .expect("stream-item decode is tolerant and must not fail");
        let mut asm = assembler();
        match expected(shape) {
            ExpectedOutcome::Assembled { text } => {
                assert!(
                    matches!(&item, StreamedAssistantContent::Text(t) if t.text == text),
                    "{shape:?} must decode as stream text: {payload}"
                );
                asm.ingest(&item).expect("ingest");
                assert_eq!(asm.aggregated_text(), text, "{shape:?}: {payload}");
                assert_eq!(
                    asm.excluded_assistant_content(),
                    0,
                    "{shape:?} must not count as excluded: {payload}"
                );
            }
            ExpectedOutcome::ExcludedAndCounted => {
                assert!(
                    matches!(&item, StreamedAssistantContent::Unknown(_)),
                    "{shape:?} must decode Unknown: {payload}"
                );
                asm.ingest(&item).expect("ingest");
                assert_eq!(asm.aggregated_text(), "", "{shape:?}: {payload}");
                assert_eq!(
                    asm.excluded_assistant_content(),
                    1,
                    "{shape:?} loses assistant content and must be counted: {payload}"
                );
            }
            ExpectedOutcome::ExcludedQuiet => {
                assert!(
                    matches!(&item, StreamedAssistantContent::Unknown(_)),
                    "{shape:?} must decode Unknown: {payload}"
                );
                asm.ingest(&item).expect("ingest");
                assert_eq!(asm.aggregated_text(), "", "{shape:?}: {payload}");
                assert_eq!(
                    asm.excluded_assistant_content(),
                    0,
                    "{shape:?} is provider-native and must stay quiet: {payload}"
                );
            }
        }
    }
}
#[test]
fn choice_text_items_judge_annotation_by_presence() {
    // `AdditionalParams` is non-empty by construction — an empty carrier
    // is unrepresentable (`try_from_value(json!({}))` yields `None`) —
    // so plain `is_some()` is the whole annotation rule and live and
    // restored classification agree by type.
    let unannotated = AssistantContent::Text(Text {
        text: String::new(),
        additional_params: rig_core::message::AdditionalParams::try_from_value(json!({}))
            .expect("object params"),
    });
    assert!(assistant_text_items_from_choice(&[unannotated]).is_empty());

    // A genuinely annotated empty block is content and survives.
    let annotated = AssistantContent::Text(Text {
        text: String::new(),
        additional_params: rig_core::message::AdditionalParams::try_from_value(
            json!({"citations": [1]}),
        )
        .expect("object params"),
    });
    assert_eq!(assistant_text_items_from_choice(&[annotated]).len(), 1);
}

#[test]
fn argument_deltas_buffer_until_name_validates() {
    let mut asm = assembler();

    let events = asm
        .ingest(&args_delta("tc_1", "{\"x\""))
        .expect("ingest should succeed");
    assert!(events.is_empty(), "arguments must buffer before the name");

    let events = asm
        .ingest(&name_delta("tc_1", "add"))
        .expect("ingest should succeed");
    let contents: Vec<_> = events
        .iter()
        .map(|event| match event {
            StreamedTurnEvent::EmitToolCallDelta { content, .. } => content.clone(),
            other => panic!("expected EmitToolCallDelta, got {other:?}"),
        })
        .collect();
    assert_eq!(
        contents,
        vec![
            ToolCallDeltaContent::Name("add".to_string()),
            ToolCallDeltaContent::Delta("{\"x\"".to_string()),
        ]
    );

    // Subsequent argument deltas now pass straight through.
    let events = asm
        .ingest(&args_delta("tc_1", ":1}"))
        .expect("ingest should succeed");
    assert_eq!(events.len(), 1);
}

#[test]
fn buffered_arguments_without_validated_name_error_at_final() {
    let mut asm = assembler();
    asm.ingest(&args_delta("tc_1", "{\"x\":1}"))
        .expect("ingest should succeed");

    assert!(asm.pending_delta_error().is_some());
    assert!(asm.ingest(&final_item()).is_err());
}

#[test]
fn finish_orders_reasoning_text_then_tool_calls() {
    let mut asm = assembler();
    asm.ingest(&StreamedAssistantContent::ReasoningDelta {
        id: "corr_1".to_string(),
        provider_id: Some("rs_1".to_string()),
        reasoning: "think".to_string(),
    })
    .expect("ingest should succeed");
    asm.ingest(&tool_call_item("tc_1", "add"))
        .expect("ingest should succeed");

    // Provider aggregation order differs deliberately.
    let final_choice = vec![
        AssistantContent::text("answer"),
        AssistantContent::ToolCall(tool_call("tc_1", "add")),
    ];

    let turn = asm.finish(Some("msg_1".to_string()), &final_choice);
    let kinds: Vec<&'static str> = turn
        .choice
        .iter()
        .map(|item| match item {
            AssistantContent::Reasoning(_) => "reasoning",
            AssistantContent::Text(_) => "text",
            AssistantContent::ToolCall(_) => "tool_call",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, vec!["reasoning", "text", "tool_call"]);
}

fn reasoning_delta(
    correlator: &str,
    provider_id: Option<&str>,
    text: &str,
) -> StreamedAssistantContent {
    StreamedAssistantContent::ReasoningDelta {
        id: correlator.to_string(),
        provider_id: provider_id.map(str::to_string),
        reasoning: text.to_string(),
    }
}

fn completed_reasoning(
    correlator: &str,
    provider_id: Option<&str>,
    text: &str,
    signature: Option<&str>,
) -> StreamedAssistantContent {
    let mut reasoning = Reasoning::new_with_signature(text, signature.map(str::to_string));
    if let Some(provider_id) = provider_id {
        reasoning = reasoning.with_id(provider_id.to_string());
    }
    StreamedAssistantContent::Reasoning {
        reasoning,
        id: correlator.to_string(),
    }
}

fn assembled_reasoning_of(asm: &StreamedTurnAssembler) -> Vec<Reasoning> {
    asm.partial_turn(None).reasoning
}

#[test]
fn aggregated_reasoning_delta_is_scoped_to_each_interleaved_part() {
    let mut asm = assembler();
    asm.ingest(&reasoning_delta("corr_a", None, "first "))
        .expect("ingest");
    assert_eq!(asm.aggregated_reasoning("corr_a"), Some("first "));

    asm.ingest(&reasoning_delta("corr_b", Some("rs_b"), "second"))
        .expect("ingest");
    assert_eq!(asm.aggregated_reasoning("corr_b"), Some("second"));

    asm.ingest(&reasoning_delta("corr_a", Some("rs_a"), "part"))
        .expect("ingest");
    assert_eq!(asm.aggregated_reasoning("corr_a"), Some("first part"));
    assert_eq!(asm.aggregated_reasoning("corr_b"), Some("second"));
    assert_eq!(asm.aggregated_reasoning("missing"), None);

    let reasoning = assembled_reasoning_of(&asm);
    assert_eq!(reasoning[0].id.as_deref(), Some("rs_a"));
    assert_eq!(reasoning[1].id.as_deref(), Some("rs_b"));
}

#[test]
fn aggregated_reasoning_delta_uses_a_new_pending_part_after_completion() {
    let mut asm = assembler();
    asm.ingest(&reasoning_delta("corr_a", Some("rs_a"), "old"))
        .expect("ingest");
    asm.ingest(&completed_reasoning(
        "corr_a",
        Some("rs_a"),
        "old",
        Some("sig"),
    ))
    .expect("ingest");
    assert_eq!(asm.aggregated_reasoning("corr_a"), None);

    asm.ingest(&reasoning_delta("corr_a", Some("rs_new"), "new"))
        .expect("ingest");
    assert_eq!(asm.aggregated_reasoning("corr_a"), Some("new"));
}

#[test]
fn interleaved_delta_parts_stay_distinct_in_arrival_order() {
    let mut asm = assembler();
    asm.ingest(&reasoning_delta("corr_a", None, "first "))
        .expect("ingest");
    asm.ingest(&reasoning_delta("corr_a", None, "part"))
        .expect("ingest");
    asm.ingest(&tool_call_item("tc_1", "add")).expect("ingest");
    asm.ingest(&reasoning_delta("corr_b", None, "second part"))
        .expect("ingest");

    let reasoning = assembled_reasoning_of(&asm);
    assert_eq!(
        reasoning.len(),
        2,
        "two parts must not merge: {reasoning:?}"
    );
    assert!(matches!(
        reasoning[0].content.first(),
        Some(rig_core::message::ReasoningContent::Text { text, .. }) if text == "first part"
    ));
    assert!(matches!(
        reasoning[1].content.first(),
        Some(rig_core::message::ReasoningContent::Text { text, .. }) if text == "second part"
    ));
}

#[test]
fn delta_only_part_survives_alongside_a_completed_block() {
    // The openrouter shape: visible chain-of-thought streams as deltas
    // whose synthesized end stays silent, while an encrypted block
    // arrives completed. Both must reach history, deltas first.
    let mut asm = assembler();
    asm.ingest(&reasoning_delta("corr_cot", None, "visible thoughts"))
        .expect("ingest");
    asm.ingest(&completed_reasoning(
        "corr_enc",
        Some("rd_1"),
        "encrypted payload",
        Some("sig"),
    ))
    .expect("ingest");

    let reasoning = assembled_reasoning_of(&asm);
    assert_eq!(
        reasoning.len(),
        2,
        "the visible chain of thought must not be dropped: {reasoning:?}"
    );
    assert!(matches!(
        reasoning[0].content.first(),
        Some(rig_core::message::ReasoningContent::Text { text, .. })
            if text == "visible thoughts"
    ));
    assert_eq!(reasoning[0].id, None);
    assert_eq!(reasoning[1].id.as_deref(), Some("rd_1"));
}

/// A later completion restating the SAME correlator is the same part's
/// authoritative whole (the unsigned-close-then-signed-restatement
/// shape): it replaces the completed slot, never appends a duplicate.
#[test]
fn a_same_correlator_completion_replaces_the_completed_part() {
    let mut asm = assembler();
    asm.ingest(&reasoning_delta("corr_a", None, "think"))
        .expect("ingest");
    asm.ingest(&completed_reasoning("corr_a", None, "think", None))
        .expect("ingest");
    asm.ingest(&completed_reasoning("corr_a", None, "think", Some("sig")))
        .expect("ingest");

    let reasoning = assembled_reasoning_of(&asm);
    assert_eq!(
        reasoning.len(),
        1,
        "one part per correlator, signed restatement replaces: {reasoning:?}"
    );
    assert!(matches!(
        reasoning[0].content.first(),
        Some(rig_core::message::ReasoningContent::Text { text, signature: Some(sig) })
            if text == "think" && sig == "sig"
    ));
}

/// Same shape with a provider id: the exact-correlator match must win
/// BEFORE the shared-provider-id extend fallback, or the signed
/// restatement doubles its own text.
#[test]
fn a_same_correlator_completion_with_a_provider_id_does_not_double_extend() {
    let mut asm = assembler();
    asm.ingest(&reasoning_delta("corr_a", Some("rs_1"), "think"))
        .expect("ingest");
    asm.ingest(&completed_reasoning("corr_a", Some("rs_1"), "think", None))
        .expect("ingest");
    asm.ingest(&completed_reasoning(
        "corr_a",
        Some("rs_1"),
        "think",
        Some("sig"),
    ))
    .expect("ingest");

    let reasoning = assembled_reasoning_of(&asm);
    assert_eq!(reasoning.len(), 1, "{reasoning:?}");
    assert_eq!(
        reasoning[0].content.len(),
        1,
        "the restatement must replace, not extend: {reasoning:?}"
    );
}

#[test]
fn completed_block_supersedes_its_deltas_by_correlator() {
    let mut asm = assembler();
    asm.ingest(&reasoning_delta("corr_a", None, "streamed text"))
        .expect("ingest");
    asm.ingest(&completed_reasoning(
        "corr_a",
        None,
        "streamed text",
        Some("sig_1"),
    ))
    .expect("ingest");

    let reasoning = assembled_reasoning_of(&asm);
    assert_eq!(
        reasoning.len(),
        1,
        "the completed block replaces its own deltas: {reasoning:?}"
    );
    assert!(matches!(
        reasoning[0].content.first(),
        Some(rig_core::message::ReasoningContent::Text { text, signature: Some(sig) })
            if text == "streamed text" && sig == "sig_1"
    ));
}

#[test]
fn completed_block_supersedes_its_deltas_by_provider_id() {
    let mut asm = assembler();
    asm.ingest(&reasoning_delta("corr_a", Some("rs_1"), "streamed text"))
        .expect("ingest");
    // A completed restatement whose correlator does not match (e.g. a
    // whole-block event minted its own) still supersedes via the
    // durable provider handle.
    asm.ingest(&completed_reasoning(
        "corr_other",
        Some("rs_1"),
        "restated text",
        None,
    ))
    .expect("ingest");

    let reasoning = assembled_reasoning_of(&asm);
    assert_eq!(reasoning.len(), 1, "{reasoning:?}");
    assert!(matches!(
        reasoning[0].content.first(),
        Some(rig_core::message::ReasoningContent::Text { text, .. }) if text == "restated text"
    ));
}

#[test]
fn completed_blocks_sharing_a_provider_id_extend_one_part() {
    let mut asm = assembler();
    asm.ingest(&completed_reasoning(
        "corr_1",
        Some("rs_1"),
        "step-1",
        Some("sig-1"),
    ))
    .expect("ingest");
    asm.ingest(&completed_reasoning(
        "corr_2",
        Some("rs_1"),
        "step-2",
        Some("sig-2"),
    ))
    .expect("ingest");
    asm.ingest(&completed_reasoning("corr_3", Some("rs_2"), "other", None))
        .expect("ingest");

    let reasoning = assembled_reasoning_of(&asm);
    assert_eq!(reasoning.len(), 2, "{reasoning:?}");
    assert_eq!(reasoning[0].id.as_deref(), Some("rs_1"));
    assert_eq!(reasoning[0].content.len(), 2);
    assert_eq!(reasoning[1].id.as_deref(), Some("rs_2"));
}

#[test]
fn completed_blocks_without_ids_stay_separate_parts() {
    let mut asm = assembler();
    asm.ingest(&completed_reasoning("corr_1", None, "first", None))
        .expect("ingest");
    asm.ingest(&completed_reasoning("corr_2", None, "second", None))
        .expect("ingest");

    let reasoning = assembled_reasoning_of(&asm);
    assert_eq!(
        reasoning.len(),
        2,
        "id-less blocks never merge: {reasoning:?}"
    );
}

#[test]
fn each_delta_part_keeps_its_own_provider_id() {
    let mut asm = assembler();
    asm.ingest(&reasoning_delta("corr_a", Some("rs_a"), "alpha"))
        .expect("ingest");
    asm.ingest(&reasoning_delta("corr_b", Some("rs_b"), "beta"))
        .expect("ingest");

    let reasoning = assembled_reasoning_of(&asm);
    assert_eq!(reasoning.len(), 2, "{reasoning:?}");
    assert_eq!(reasoning[0].id.as_deref(), Some("rs_a"));
    assert_eq!(reasoning[1].id.as_deref(), Some("rs_b"));
}

#[test]
fn canonical_choice_and_partial_turn_agree_on_multi_part_reasoning() {
    let mut asm = assembler();
    asm.ingest(&reasoning_delta("corr_a", None, "visible"))
        .expect("ingest");
    asm.ingest(&completed_reasoning(
        "corr_b",
        Some("rd_1"),
        "enc",
        Some("sig"),
    ))
    .expect("ingest");

    let partial = asm.partial_turn(None).reasoning;
    let final_choice = vec![AssistantContent::text("")];
    let turn = asm.finish(None, &final_choice);
    let finished: Vec<Reasoning> = turn
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => Some(reasoning.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(partial, finished, "partial and finished assembly agree");
    assert_eq!(finished.len(), 2);
}

#[test]
fn finish_passes_raw_choice_through_for_plain_text_turns() {
    let mut asm = assembler();
    asm.ingest(&text_item("hi")).expect("ingest should succeed");

    let final_choice = vec![AssistantContent::text("hi")];
    let turn = asm.finish(None, &final_choice);
    assert_eq!(
        serde_json::to_value(&turn.choice).expect("serialize"),
        serde_json::to_value(&final_choice).expect("serialize"),
    );
}

fn expect_invalid(events: Vec<StreamedTurnEvent>) -> StreamedInvalidToolCall {
    match events.into_iter().next() {
        Some(StreamedTurnEvent::InvalidToolCall(invalid)) => *invalid,
        other => panic!("expected InvalidToolCall, got {other:?}"),
    }
}

#[test]
fn streamed_run_completes_a_tool_roundtrip() {
    let mut run = AgentRun::new("add things").max_turns(2);

    // Turn 1: the model streams one tool call.
    let AgentRunStep::CallModel { .. } = run.next_step().expect("next_step") else {
        panic!("expected CallModel");
    };
    let mut asm = assembler();
    assert!(
        asm.ingest(&tool_call_item("tc_1", "add"))
            .expect("ingest should succeed")
            .is_empty()
    );
    let usage = Usage {
        input_tokens: 5,
        output_tokens: 7,
        total_tokens: 12,
        ..Usage::new()
    };
    run.record_streamed_completion_call(
        usage,
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::Value::Null,
    )
    .expect("record should succeed");
    let final_choice = vec![AssistantContent::ToolCall(tool_call("tc_1", "add"))];
    run.streamed_turn(asm.finish(Some("msg_1".to_string()), &final_choice))
        .expect("streamed_turn should succeed");

    let AgentRunStep::CallTools { calls } = run.next_step().expect("next_step") else {
        panic!("expected CallTools");
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].internal_call_id, Some(iid_for("tc_1")));
    run.tool_results(vec![UserContent::tool_result(
        "tc_1",
        "add",
        vec![ToolResultContent::text("2")],
    )])
    .expect("tool_results should succeed");

    // Turn 2: plain text finishes the run.
    let AgentRunStep::CallModel { .. } = run.next_step().expect("next_step") else {
        panic!("expected CallModel");
    };
    let asm = assembler();
    run.record_streamed_completion_call(
        Usage::new(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::Value::Null,
    )
    .expect("record should succeed");
    let final_choice = vec![AssistantContent::text("done")];
    run.streamed_turn(asm.finish(None, &final_choice))
        .expect("streamed_turn should succeed");

    let AgentRunStep::Done(response) = run.next_step().expect("next_step") else {
        panic!("expected Done");
    };
    assert_eq!(response.output, "done");
    assert_eq!(response.usage, usage);
    assert_eq!(response.completion_calls.len(), 2);
    assert_eq!(response.completion_calls[0].usage, usage);
    assert_eq!(response.completion_calls[1].usage, Usage::new());
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
fn streamed_invalid_tool_call_retry_rolls_back_with_partial_turn() {
    let mut run = AgentRun::new("use the tool")
        .max_turns(2)
        .max_invalid_tool_call_retries(1);
    run.next_step().expect("next_step");

    let mut asm = assembler();
    asm.ingest(&text_item("thinking ")).expect("ingest");
    let invalid = expect_invalid(
        asm.ingest(&tool_call_item("tc_1", "default_api"))
            .expect("ingest should succeed"),
    );
    let partial = asm.partial_turn(Some("msg_1".to_string()));
    assert_eq!(partial.text.as_deref(), Some("thinking "));

    let context = run.streamed_invalid_tool_call_context(&partial, &invalid);
    assert!(context.is_streaming);
    assert_eq!(context.tool_name, "default_api");
    assert_eq!(context.internal_call_id, Some(iid_for("tc_1")));

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
        Usage::new(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::Value::Null,
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
    let invalid = expect_invalid(
        asm.ingest(&tool_call_item("tc_1", "default_api"))
            .expect("ingest should succeed"),
    );
    let partial = asm.partial_turn(Some("msg_1".to_string()));

    let err = run
        .resolve_streamed_invalid_tool_call(
            &partial,
            &invalid,
            InvalidToolCallAction::stop("operator stop"),
        )
        .expect_err("stop should cancel the run");
    assert!(matches!(
        err,
        PromptError::PromptCancelled { reason, .. } if reason == "operator stop"
    ));

    let err = run
        .next_step()
        .expect_err("a stopped streamed run must remain terminal");
    assert!(matches!(
        err,
        PromptError::PromptCancelled { reason, .. }
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
    let invalid = expect_invalid(
        asm.ingest(&tool_call_item("tc_1", "default_api"))
            .expect("ingest should succeed"),
    );
    let partial = asm.partial_turn(Some("msg_1".to_string()));
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
        Usage::new(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::Value::Null,
    )
    .expect("completion call should be recorded");
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
fn streamed_invalid_tool_call_skip_returns_synthetic_result() {
    let mut run = AgentRun::new("use the tool").max_turns(2);
    run.next_step().expect("next_step");

    let mut asm = assembler();
    let invalid = expect_invalid(
        asm.ingest(&tool_call_item("tc_1", "default_api"))
            .expect("ingest should succeed"),
    );
    let partial = asm.partial_turn(None);

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
    assert_eq!(tool_result.call, "tc_1");
}

#[test]
fn streamed_invalid_name_delta_repair_replays_buffered_arguments() {
    let mut run = AgentRun::new("use the tool").max_turns(2);
    run.next_step().expect("next_step");

    let mut asm = assembler();
    asm.ingest(&args_delta("tc_1", "{\"x\":1}"))
        .expect("ingest should succeed");
    let invalid = expect_invalid(
        asm.ingest(&name_delta("tc_1", "default_api"))
            .expect("ingest should succeed"),
    );
    assert_eq!(invalid.args.as_deref(), Some("{\"x\":1}"));

    let partial = asm.partial_turn(None);
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
    let contents: Vec<_> = events
        .iter()
        .map(|event| match event {
            StreamedTurnEvent::EmitToolCallDelta { content, .. } => content.clone(),
            other => panic!("expected EmitToolCallDelta, got {other:?}"),
        })
        .collect();
    assert_eq!(
        contents,
        vec![
            ToolCallDeltaContent::Name("add".to_string()),
            ToolCallDeltaContent::Delta("{\"x\":1}".to_string()),
        ]
    );
}

#[test]
fn streamed_turn_rejects_unknown_tool_calls_fail_fast() {
    let mut run = AgentRun::new("use the tool");
    run.next_step().expect("next_step");

    let turn = StreamedTurn {
        message_id: None,
        choice: vec![AssistantContent::ToolCall(tool_call("tc_1", "unknown"))],
        executable_tool_names: tool_names(&["add"]),
        allowed_tool_names: tool_names(&["add"]),
        internal_call_ids: Vec::new(),
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
            Usage::new(),
            rig_core::completion::ResponseIdentity::default(),
            None,
            serde_json::Value::Null,
        )
        .expect_err("recording before any model call must be rejected");
    assert!(matches!(err, PromptError::PromptCancelled { .. }));

    // The run stays drivable.
    run.next_step().expect("next_step should still succeed");
    run.record_streamed_completion_call(
        Usage::new(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::Value::Null,
    )
    .expect("recording during a pending model call succeeds");
}

#[test]
fn duplicate_tool_call_ids_keep_distinct_internal_ids_through_the_run() {
    let mut run = AgentRun::new("do both").max_turns(2);
    run.next_step().expect("next_step");

    let mut asm = assembler();
    asm.ingest(&StreamedAssistantContent::ToolCall {
        tool_call: tool_call("tc_1", "add"),
        internal_call_id: iid_for("a"),
    })
    .expect("ingest should succeed");
    asm.ingest(&StreamedAssistantContent::ToolCall {
        tool_call: tool_call("tc_1", "add"),
        internal_call_id: iid_for("b"),
    })
    .expect("ingest should succeed");
    run.record_streamed_completion_call(
        Usage::new(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::Value::Null,
    )
    .expect("record should succeed");

    let final_choice = vec![
        AssistantContent::ToolCall(tool_call("tc_1", "add")),
        AssistantContent::ToolCall(tool_call("tc_1", "add")),
    ];
    run.streamed_turn(asm.finish(None, &final_choice))
        .expect("streamed_turn should succeed");

    // The internal IDs survive in the run state itself: a serde round
    // trip must keep both calls distinguishable.
    let serialized = serde_json::to_string(&run).expect("serialize");
    let mut restored: AgentRun = serde_json::from_str(&serialized).expect("deserialize");
    let AgentRunStep::CallTools { calls } = restored.next_step().expect("next_step") else {
        panic!("expected CallTools");
    };
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].internal_call_id, Some(iid_for("a")));
    assert_eq!(calls[1].internal_call_id, Some(iid_for("b")));
}

#[test]
fn streamed_turn_records_the_completion_call_when_the_driver_did_not() {
    let mut run = AgentRun::new("hello");
    run.next_step().expect("next_step");

    let asm = assembler();
    let final_choice = vec![AssistantContent::text("done")];
    run.streamed_turn(asm.finish(None, &final_choice))
        .expect("streamed_turn should succeed");

    // Exactly one CompletionCall per model call, even without an explicit
    // record; usage is simply unreported.
    assert_eq!(run.completion_calls().len(), 1);
    assert_eq!(run.completion_calls()[0].usage, Usage::new());
}

#[test]
fn streamed_completion_call_is_recorded_once_per_turn() {
    let mut run = AgentRun::new("hello");
    run.next_step().expect("next_step");

    run.record_streamed_completion_call(
        Usage::new(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::Value::Null,
    )
    .expect("first record succeeds");
    let err = run
        .record_streamed_completion_call(
            Usage::new(),
            rig_core::completion::ResponseIdentity::default(),
            None,
            serde_json::Value::Null,
        )
        .expect_err("second record for the same turn must be rejected");
    assert!(matches!(err, PromptError::PromptCancelled { .. }));
    assert_eq!(run.completion_calls().len(), 1);
}

#[test]
fn streamed_run_serde_round_trips_while_tools_pend() {
    let mut run = AgentRun::new("add things").max_turns(2);
    run.next_step().expect("next_step");

    let mut asm = assembler();
    asm.ingest(&tool_call_item("tc_1", "add"))
        .expect("ingest should succeed");
    run.record_streamed_completion_call(
        Usage::new(),
        rig_core::completion::ResponseIdentity::default(),
        None,
        serde_json::Value::Null,
    )
    .expect("record should succeed");
    let final_choice = vec![AssistantContent::ToolCall(tool_call("tc_1", "add"))];
    run.streamed_turn(asm.finish(None, &final_choice))
        .expect("streamed_turn should succeed");
    run.next_step().expect("CallTools step");

    let serialized = serde_json::to_string(&run).expect("serialize mid-run");
    let mut restored: AgentRun = serde_json::from_str(&serialized).expect("deserialize mid-run");
    restored
        .tool_results(vec![UserContent::tool_result(
            "tc_1",
            "add",
            vec![ToolResultContent::text("2")],
        )])
        .expect("tool_results should succeed");
    assert!(matches!(
        restored.next_step().expect("next turn"),
        AgentRunStep::CallModel { turn: 2, .. }
    ));
}
