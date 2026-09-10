use super::*;

/// Test-side key syntax: legacy minted renderings decode to minted
/// keys; anything else is wire-derived.
fn pid(id: &str) -> StreamPartId {
    use crate::streaming::MintKind;
    for (namespace, kind) in [
        ("reasoning-", MintKind::Reasoning),
        ("block-", MintKind::Block),
        ("output-", MintKind::Output),
        ("tool-", MintKind::Tool),
        ("text-", MintKind::Text),
    ] {
        if let Some(rest) = id.strip_prefix(namespace)
            && let Ok(index) = rest.parse::<u64>()
        {
            return kind.for_wire_index(index);
        }
    }
    StreamPartId::wire(id)
}

/// Provider handle matching the key syntax: wire-shaped ids carry
/// themselves; minted renderings carry none.
fn wid(id: &str) -> Option<WireId> {
    pid(id).wire_str().and_then(|_| WireId::new(id))
}

fn full(id: &str, content: ReasoningContent) -> Reasoning {
    Reasoning {
        id: wid(id).map(|id| id.into_string()),
        content: vec![content],
    }
}

fn summary(text: &str) -> ReasoningContent {
    ReasoningContent::Summary(text.to_owned())
}

fn reasoning_text(text: &str) -> ReasoningContent {
    ReasoningContent::Text {
        text: text.to_owned(),
        signature: None,
    }
}

/// Text of every reasoning part, flattened in part order.
fn reasoning_texts(parts: &[AssistantContent]) -> Vec<String> {
    parts
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Reasoning(reasoning) => Some(reasoning.content.iter()),
            _ => None,
        })
        .flatten()
        .map(|content| match content {
            ReasoningContent::Summary(text) => text.clone(),
            ReasoningContent::Text { text, .. } => text.clone(),
            ReasoningContent::Encrypted(data) => data.clone(),
            ReasoningContent::Redacted { data } => data.clone(),
        })
        .collect()
}

fn end(id: &str, mode: UnparseableToolInput) -> ToolInputEnd {
    ToolInputEnd::new(pid(id), mode)
}

fn call_named(id: &str, name: &str) -> ToolCall {
    ToolCall::from_wire(
        id,
        crate::message::ToolFunction {
            name: name.to_owned(),
            arguments: serde_json::json!({}),
        },
    )
}

// --- reasoning lifecycle ---

/// An end's authoritative restatement supersedes the open part's delta
/// accumulation (pydantic-ai `_replace_part` semantics).
#[test]
fn an_end_restatement_replaces_its_delta_accumulation() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.reasoning_delta(&pid("rs_1"), wid("rs_1").as_ref(), "partial ");
    accumulator.reasoning_delta(&pid("rs_1"), wid("rs_1").as_ref(), "thought");
    accumulator.reasoning_end(
        &pid("rs_1"),
        Some(full("rs_1", reasoning_text("the complete chain"))),
        None,
    );

    let parts = accumulator.finish();
    assert_eq!(reasoning_texts(&parts), vec!["the complete chain"]);
    assert_eq!(parts.len(), 1);
}

/// A restatement that doesn't restate the durable handle must not erase
/// the one established at part-open; a restatement that does carries
/// its own.
#[test]
fn an_id_less_restatement_keeps_the_open_parts_provider_handle() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.reasoning_delta(&pid("rs_1"), wid("rs_1").as_ref(), "partial");
    let completed = accumulator
        .reasoning_end(
            &pid("rs_1"),
            Some(Reasoning {
                id: None,
                content: vec![reasoning_text("the complete chain")],
            }),
            None,
        )
        .expect("open part completes");
    assert_eq!(
        completed.id.as_deref(),
        Some("rs_1"),
        "an absent restatement id falls back to the handle set at open"
    );

    let mut accumulator = PartsAccumulator::new();
    accumulator.reasoning_delta(&pid("rs_2"), wid("rs_2").as_ref(), "partial");
    let completed = accumulator
        .reasoning_end(
            &pid("rs_2"),
            Some(full("rs_other", reasoning_text("restated"))),
            None,
        )
        .expect("open part completes");
    assert_eq!(
        completed.id.as_deref(),
        Some("rs_other"),
        "a restated handle wins over the one set at open"
    );
}

/// An open part keeps collapsing across interleaved output: with no end
/// synthesized, later deltas and the restatement extend/replace the
/// SAME part (the wire-key Responses shape).
#[test]
fn an_open_part_collapses_across_interleaved_output() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.reasoning_delta(&pid("rs_1"), wid("rs_1").as_ref(), "thinking");
    accumulator.tool_call(
        &pid("call_1"),
        call_named("call_1", "probe"),
        InternalCallId::new(),
    );
    accumulator.reasoning_end(
        &pid("rs_1"),
        Some(full("rs_1", reasoning_text("full reasoning"))),
        None,
    );

    let parts = accumulator.finish();
    assert_eq!(reasoning_texts(&parts), vec!["full reasoning"]);
    assert_eq!(
        parts.len(),
        2,
        "reasoning replaced in place beside the call"
    );
    assert!(matches!(
        parts.first(),
        Some(AssistantContent::Reasoning(_))
    ));
}

/// Two signature-bearing segments under a per-stream constant key
/// (gemini REST/Interactions, cohere, ollama all key reasoning by one
/// minted constant): the second signature must land on a DISTINCT part.
/// Signatures cannot merge — overwriting the first would replay with
/// only the last signature and fail `MISSING_THOUGHT_SIGNATURE`.
#[test]
fn a_second_signature_under_a_constant_key_records_a_distinct_part() {
    let mut accumulator = PartsAccumulator::new();
    let key = pid("reasoning-0");
    accumulator.reasoning_delta(&key, None, "thought A");
    accumulator.reasoning_end(&key, None, Some("sig_A".to_string()));
    // A signature-only end for the already-finished constant key: the
    // wire signed a second segment with nothing new streamed to sign.
    accumulator.reasoning_end(&key, None, Some("sig_B".to_string()));

    let parts = accumulator.finish();
    let signed: Vec<(String, Option<String>)> = parts
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Reasoning(reasoning) => Some(reasoning),
            _ => None,
        })
        .flat_map(|reasoning| {
            reasoning.content.iter().map(|content| match content {
                ReasoningContent::Text { text, signature } => (text.clone(), signature.clone()),
                other => (format!("{other:?}"), None),
            })
        })
        .collect();
    assert_eq!(
        signed,
        vec![
            ("thought A".to_string(), Some("sig_A".to_string())),
            (String::new(), Some("sig_B".to_string())),
        ],
        "both signatures survive, each on its own slot in its own part"
    );
    let reasoning_parts = parts
        .iter()
        .filter(|part| matches!(part, AssistantContent::Reasoning(_)))
        .count();
    assert_eq!(reasoning_parts, 2, "the second signature is a sibling part");
}

/// A synthesized end splits a constant-key wire's blocks: a delta after
/// the end opens a NEW part (key reuse), never merging backwards.
#[test]
fn a_delta_after_the_end_opens_a_fresh_part() {
    let mut accumulator = PartsAccumulator::new();
    let key = pid("reasoning-0");
    accumulator.reasoning_delta(&key, None, "A");
    accumulator.reasoning_end(&key, None, None);
    accumulator.text_delta("visible");
    accumulator.reasoning_delta(&key, None, "B");

    let parts = accumulator.finish();
    assert_eq!(reasoning_texts(&parts), vec!["A", "B"]);
    assert!(matches!(
        parts.get(1),
        Some(AssistantContent::Text(text)) if text.text == "visible"
    ));
}

/// Same-key whole blocks after the entity finished are siblings: every
/// part survives, in arrival order (the Responses multi-part item).
#[test]
fn same_key_whole_blocks_are_siblings_and_all_survive() {
    let mut accumulator = PartsAccumulator::new();
    for content in [
        summary("s1"),
        summary("s2"),
        reasoning_text("visible"),
        ReasoningContent::Encrypted("enc".to_owned()),
    ] {
        accumulator.reasoning_end(&pid("rs_1"), Some(full("rs_1", content)), None);
    }

    let parts = accumulator.finish();
    assert_eq!(reasoning_texts(&parts), vec!["s1", "s2", "visible", "enc"]);
}

/// Deltas then the item's multi-part done block: the first restatement
/// supersedes the delta accumulation, the rest append as siblings.
#[test]
fn deltas_then_sibling_whole_blocks_keep_each_part_once() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.reasoning_delta(&pid("rs_1"), wid("rs_1").as_ref(), "s1");
    for content in [
        summary("s1"),
        summary("s2"),
        ReasoningContent::Encrypted("enc".to_owned()),
    ] {
        accumulator.reasoning_end(&pid("rs_1"), Some(full("rs_1", content)), None);
    }

    let parts = accumulator.finish();
    assert_eq!(reasoning_texts(&parts), vec!["s1", "s2", "enc"]);
}

#[test]
fn distinct_keys_never_interact() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.reasoning_delta(&pid("rs_1"), wid("rs_1").as_ref(), "first item deltas");
    accumulator.reasoning_end(
        &pid("rs_2"),
        Some(full("rs_2", reasoning_text("a different item"))),
        None,
    );

    let parts = accumulator.finish();
    assert_eq!(
        reasoning_texts(&parts),
        vec!["first item deltas", "a different item"]
    );
}

/// A trailing signature-only end signs the block that HOLDS the
/// chain-of-thought — wherever its arrival position was — never an
/// empty sibling (#2258 B4).
#[test]
fn a_trailing_signature_signs_the_finished_block() {
    let mut accumulator = PartsAccumulator::new();
    let key = pid("reasoning-0");
    accumulator.reasoning_delta(&key, None, "the chain");
    accumulator.reasoning_end(&key, None, None);
    accumulator.text_delta("answer");
    accumulator.reasoning_end(&key, None, Some("sig".to_owned()));

    let parts = accumulator.finish();
    assert_eq!(parts.len(), 2);
    assert!(matches!(
        parts.first(),
        Some(AssistantContent::Reasoning(reasoning))
            if matches!(reasoning.content.first(), Some(ReasoningContent::Text { text, signature })
                if text == "the chain" && signature.as_deref() == Some("sig"))
    ));
}

/// A signature closing an open block signs the accumulated deltas.
#[test]
fn a_signature_end_signs_the_open_block() {
    let mut accumulator = PartsAccumulator::new();
    let key = pid("reasoning-0");
    accumulator.reasoning_delta(&key, None, "the chain");
    accumulator.reasoning_end(&key, None, Some("sig".to_owned()));

    let parts = accumulator.finish();
    assert_eq!(parts.len(), 1);
    assert!(matches!(
        parts.first(),
        Some(AssistantContent::Reasoning(reasoning))
            if matches!(reasoning.content.first(), Some(ReasoningContent::Text { signature, .. })
                if signature.as_deref() == Some("sig"))
    ));
}

/// A signature with nothing streamed records a signature-only part —
/// replay-required provider state survives.
#[test]
fn a_signature_only_stream_records_a_signature_only_part() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.reasoning_end(&pid("reasoning-0"), None, Some("sig".to_owned()));
    let parts = accumulator.finish();
    assert_eq!(parts.len(), 1);
}

/// A bare repeated end is a no-op: idempotence belongs to the entity.
#[test]
fn repeated_bare_ends_are_no_ops() {
    let mut accumulator = PartsAccumulator::new();
    let key = pid("reasoning-0");
    accumulator.reasoning_delta(&key, None, "A");
    assert!(accumulator.reasoning_end(&key, None, None).is_some());
    assert!(accumulator.reasoning_end(&key, None, None).is_none());
    assert!(accumulator.reasoning_end(&key, None, None).is_none());
    assert_eq!(accumulator.finish().len(), 1);
}

// --- text lifecycle ---

#[test]
fn text_and_reasoning_interleave_in_arrival_order() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.text_delta("intro");
    accumulator.reasoning_end(&pid("rs_1"), Some(full("rs_1", summary("thinking"))), None);
    accumulator.text_start(&pid("msg_2"), None);
    accumulator.text_delta("out");
    accumulator.text_delta("ro");

    let parts = accumulator.finish();
    assert_eq!(parts.len(), 3);
    assert!(matches!(
        parts.first(),
        Some(AssistantContent::Text(text)) if text.text == "intro"
    ));
    assert!(matches!(parts.get(1), Some(AssistantContent::Reasoning(_))));
    assert!(matches!(
        parts.get(2),
        Some(AssistantContent::Text(text)) if text.text == "outro"
    ));
}

#[test]
fn distinct_text_start_ids_open_distinct_parts() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.text_start(&pid("msg_1"), None);
    accumulator.text_delta("first");
    accumulator.text_start(&pid("msg_2"), None);
    accumulator.text_delta("second");

    let parts = accumulator.finish();
    let texts: Vec<&str> = parts
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, vec!["first", "second"]);
}

/// A `TextStart` whose key was already seen reactivates that block
/// across interleaved output instead of opening a duplicate part.
#[test]
fn a_seen_text_start_id_reactivates_its_block() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.text_start(&pid("msg_1"), None);
    accumulator.text_delta("collapsing ");
    accumulator.reasoning_end(&pid("rs_1"), Some(full("rs_1", summary("thinking"))), None);
    accumulator.text_start(&pid("msg_1"), None);
    accumulator.text_delta("text");

    let parts = accumulator.finish();
    assert_eq!(parts.len(), 2, "one text part, one reasoning part");
    assert!(matches!(
        parts.first(),
        Some(AssistantContent::Text(text)) if text.text == "collapsing text"
    ));
}

/// A `TextStart` that never receives content leaves no empty part.
#[test]
fn a_content_less_text_start_leaves_no_empty_part() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.text_start(&pid("msg_1"), None);
    accumulator.reasoning_end(&pid("rs_1"), Some(full("rs_1", summary("thinking"))), None);

    let parts = accumulator.finish();
    assert_eq!(parts.len(), 1);
    assert!(matches!(
        parts.first(),
        Some(AssistantContent::Reasoning(_))
    ));
}

/// An explicit `TextEnd` closes the block: later bare deltas open a
/// fresh part.
#[test]
fn text_end_closes_the_block_for_bare_deltas() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.text_start(&pid("msg_1"), None);
    accumulator.text_delta("first");
    accumulator.text_end(&pid("msg_1"));
    accumulator.text_delta("second");

    let parts = accumulator.finish();
    let texts: Vec<&str> = parts
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, vec!["first", "second"]);
}

#[test]
fn finish_on_an_empty_stream_yields_no_parts() {
    let mut accumulator = PartsAccumulator::new();
    let parts = accumulator.finish();
    // Was one fabricated empty-text part, which the old non-empty content
    // type forced and which was indistinguishable on the wire from a real
    // empty text block.
    assert!(parts.is_empty());
}

// --- tool-call lifecycle (the settled semantics) ---

#[test]
fn fragments_assemble_into_a_completed_tool_call_with_a_stable_internal_id() {
    let mut accumulator = PartsAccumulator::new();
    let first = accumulator.tool_name_delta(&pid("call_1"), "get_weather");
    let second = accumulator.tool_args_delta(&pid("call_1"), "{\"location\":");
    accumulator.tool_args_delta(&pid("call_1"), "\"Paris\"}");
    assert_eq!(first, second);

    let (tool_call, internal_call_id) = accumulator
        .tool_input_end(end("call_1", UnparseableToolInput::Drop))
        .expect("no error")
        .expect("call must finalize");
    assert_eq!(internal_call_id, first, "internal id is minted at start");
    assert_eq!(tool_call.id, "call_1");
    assert_eq!(tool_call.function.name, "get_weather");
    assert!(accumulator.saw_tool_call());
}

#[test]
fn an_empty_name_fragment_does_not_erase_an_established_name() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("call_1"), "get_weather");
    accumulator.tool_name_delta(&pid("call_1"), "");
    accumulator.tool_args_delta(&pid("call_1"), "{}");

    let (tool_call, _) = accumulator
        .tool_input_end(end("call_1", UnparseableToolInput::Drop))
        .expect("no error")
        .expect("call must finalize under the established name");
    assert_eq!(tool_call.function.name, "get_weather");
}

#[test]
fn a_call_with_no_streamed_arguments_is_a_parameterless_invocation() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("call_1"), "ping");
    let (tool_call, _) = accumulator
        .tool_input_end(end("call_1", UnparseableToolInput::Drop))
        .expect("no error")
        .expect("parameterless calls are preserved");
    assert_eq!(tool_call.function.arguments, serde_json::json!({}));
}

#[test]
fn drop_mode_drops_partial_arguments_and_nameless_calls() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("call_1"), "ping");
    accumulator.tool_args_delta(&pid("call_1"), "{\"x\":");
    assert!(
        accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Drop))
            .expect("no error")
            .is_none()
    );
    accumulator.tool_args_delta(&pid("call_2"), "{\"y\":1}");
    assert!(
        accumulator
            .tool_input_end(end("call_2", UnparseableToolInput::Drop))
            .expect("no error")
            .is_none(),
        "a call whose name never arrived is dropped"
    );
    assert!(!accumulator.saw_tool_call());
}

#[test]
fn error_mode_surfaces_malformed_input_as_an_error() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("call_1"), "get_weather");
    accumulator.tool_args_delta(&pid("call_1"), "{\"location\": not-json");
    let err = accumulator
        .tool_input_end(end("call_1", UnparseableToolInput::Error))
        .expect_err("malformed complete input must error");
    assert!(err.to_string().contains("get_weather"));
}

#[test]
fn keep_mode_leaves_the_call_open_for_later_fragments() {
    let mut accumulator = PartsAccumulator::new();
    let internal = accumulator.tool_name_delta(&pid("call_1"), "search");
    accumulator.tool_args_delta(&pid("call_1"), "{\"q\":\"ru");
    assert!(
        accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Keep))
            .expect("no error")
            .is_none()
    );
    accumulator.tool_args_delta(&pid("call_1"), "st\"}");
    let (tool_call, internal_after) = accumulator
        .tool_input_end(end("call_1", UnparseableToolInput::Drop))
        .expect("no error")
        .expect("extended call finalizes");
    assert_eq!(internal_after, internal);
    assert_eq!(
        tool_call.function.arguments,
        serde_json::json!({"q": "rust"})
    );
}

#[test]
fn authoritative_end_fields_supersede_assembly() {
    let mut accumulator = PartsAccumulator::new();
    let internal = accumulator.tool_name_delta(&pid("fc_1"), "provisional");
    accumulator.tool_args_delta(&pid("fc_1"), "{\"partial\":");

    let mut done = end("fc_1", UnparseableToolInput::Drop);
    done.name = Some("final_name".to_owned());
    done.arguments = Some(serde_json::json!({"x": 1}));
    done.call_id = Some("call_abc".to_owned());
    let (tool_call, internal_after) = accumulator
        .tool_input_end(done)
        .expect("no error")
        .expect("authoritative payload finalizes");
    assert_eq!(internal_after, internal);
    // The authoritative correlator drives rig's id; the wire-derived
    // assembly key rides along as the provider item id.
    assert_eq!(tool_call.id, "call_abc");
    let provider = tool_call.provider.as_ref().expect("provider ids are kept");
    assert_eq!(provider.call_id, "call_abc");
    assert_eq!(provider.item_id.as_deref(), Some("fc_1"));
    assert_eq!(tool_call.function.name, "final_name");
}

#[test]
fn an_end_with_no_open_call_creates_the_call_from_its_payload() {
    let mut accumulator = PartsAccumulator::new();
    let mut done = end("fc_1", UnparseableToolInput::Drop);
    done.name = Some("add".to_owned());
    done.arguments = Some(serde_json::json!({"x": 2}));
    let (tool_call, _) = accumulator
        .tool_input_end(done)
        .expect("no error")
        .expect("whole done items finalize");
    assert_eq!(tool_call.function.name, "add");
}

/// 84a43e9e #1, closed structurally: a REPEATED end for a finished
/// entity finalizes nothing — even when it carries the authoritative
/// name/arguments payload that used to bypass the guard and duplicate
/// the call. The finished-set is populated by every route.
#[test]
fn a_repeated_end_with_an_authoritative_payload_is_a_no_op() {
    let mut accumulator = PartsAccumulator::new();
    let mut done = end("fc_1", UnparseableToolInput::Drop);
    done.name = Some("add".to_owned());
    done.arguments = Some(serde_json::json!({"x": 2}));
    accumulator
        .tool_input_end(done.clone())
        .expect("no error")
        .expect("finalizes once");
    assert!(
        accumulator
            .tool_input_end(done)
            .expect("no error")
            .is_none(),
        "a repeated authoritative end must not duplicate the call"
    );
    assert_eq!(
        accumulator
            .finish()
            .iter()
            .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
            .count(),
        1
    );
}

/// A drop is a finalization: a truncated call dismissed under `Drop`
/// must not be resurrected by a later payload-bearing end for the same
/// key — that end is the same repeated-end defect the finished-set
/// guard exists for, arriving after a drop instead of a success.
#[test]
fn a_dropped_truncated_call_is_not_resurrected_by_a_payload_bearing_end() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("call_1"), "get_weather");
    accumulator.tool_args_delta(&pid("call_1"), "{\"loc\":");
    assert!(
        accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Drop))
            .expect("no error")
            .is_none(),
        "truncated arguments drop"
    );
    let mut retry = end("call_1", UnparseableToolInput::Drop);
    retry.name = Some("get_weather".to_owned());
    retry.arguments = Some(serde_json::json!({"loc": "Paris"}));
    assert!(
        accumulator
            .tool_input_end(retry)
            .expect("no error")
            .is_none(),
        "the dropped entity is finished; a later payload must not fabricate a phantom call"
    );
    assert!(!accumulator.saw_tool_call());
}

/// Same for the nameless drop: the entity finished when it was
/// dismissed, so a later end supplying the missing name is repeated,
/// not completing.
#[test]
fn a_dropped_nameless_call_is_not_resurrected_by_a_named_end() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_args_delta(&pid("call_1"), "{\"y\":1}");
    assert!(
        accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Drop))
            .expect("no error")
            .is_none(),
        "nameless call drops"
    );
    let mut retry = end("call_1", UnparseableToolInput::Drop);
    retry.name = Some("late_name".to_owned());
    retry.arguments = Some(serde_json::json!({"y": 1}));
    assert!(
        accumulator
            .tool_input_end(retry)
            .expect("no error")
            .is_none(),
        "the dropped entity is finished; a late name must not fabricate a phantom call"
    );
    assert!(!accumulator.saw_tool_call());
}

#[test]
fn a_stale_end_for_a_finalized_key_is_a_no_op() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("call_1"), "ping");
    accumulator
        .tool_input_end(end("call_1", UnparseableToolInput::Drop))
        .expect("no error")
        .expect("finalizes");
    assert!(
        accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Drop))
            .expect("no error")
            .is_none()
    );
}

/// The accumulation bound applies to the FIRST fragment too: a single
/// oversized fragment trips overflow (and thereby the
/// unparseable-input policy at finalization) exactly like the same
/// payload split across fragments.
#[test]
fn an_oversized_first_fragment_trips_the_accumulation_bound() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("call_1"), "big");
    let oversized = "x".repeat(MAX_TOOL_INPUT_BYTES + 1);
    accumulator.tool_args_delta(&pid("call_1"), &oversized);
    assert!(
        accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Drop))
            .expect("no error")
            .is_none(),
        "overflow forces the unparseable path; the call must not finalize"
    );
    assert!(!accumulator.saw_tool_call());
}

#[test]
fn null_placeholder_is_replaced_by_following_json_fragments() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("call_123"), "web_search");
    accumulator.tool_args_delta(&pid("call_123"), "null");
    accumulator.tool_args_delta(&pid("call_123"), "{\"query\": \"META");
    accumulator.tool_args_delta(&pid("call_123"), " Platforms news\"}");
    let (tool_call, _) = accumulator
        .tool_input_end(end("call_123", UnparseableToolInput::Drop))
        .expect("no error")
        .expect("call must finalize");
    assert_eq!(
        tool_call.function.arguments,
        serde_json::json!({"query": "META Platforms news"})
    );
}

#[test]
fn parallel_calls_assemble_independently() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("call_a"), "get_weather");
    accumulator.tool_name_delta(&pid("call_b"), "get_time");
    accumulator.tool_args_delta(&pid("call_a"), "{\"location\":\"Paris\"}");
    accumulator.tool_args_delta(&pid("call_b"), "{\"zone\":\"UTC\"}");
    let (call_b, _) = accumulator
        .tool_input_end(end("call_b", UnparseableToolInput::Drop))
        .expect("no error")
        .expect("finalizes");
    let (call_a, _) = accumulator
        .tool_input_end(end("call_a", UnparseableToolInput::Drop))
        .expect("no error")
        .expect("finalizes");
    assert_eq!(
        call_b.function.arguments,
        serde_json::json!({"zone": "UTC"})
    );
    assert_eq!(
        call_a.function.arguments,
        serde_json::json!({"location": "Paris"})
    );
}

#[test]
fn minted_keys_keep_id_less_parallel_calls_distinct() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("tool-0"), "get_weather");
    accumulator.tool_name_delta(&pid("tool-1"), "get_time");
    accumulator.tool_args_delta(&pid("tool-0"), "{\"city\":");
    accumulator.tool_args_delta(&pid("tool-1"), "{\"zone\":");
    accumulator.tool_args_delta(&pid("tool-0"), "\"Tokyo\"}");
    accumulator.tool_args_delta(&pid("tool-1"), "\"UTC\"}");
    let (first, _) = accumulator
        .tool_input_end(end("tool-0", UnparseableToolInput::Drop))
        .expect("no error")
        .expect("finalizes");
    let (second, _) = accumulator
        .tool_input_end(end("tool-1", UnparseableToolInput::Drop))
        .expect("no error")
        .expect("finalizes");
    // Id-less calls mint distinct correlation handles and record that
    // the provider issued nothing — never a shared empty sentinel.
    assert!(!first.id.as_str().is_empty());
    assert!(!second.id.as_str().is_empty());
    assert_ne!(first.id, second.id);
    assert!(first.provider.is_none());
    assert!(second.provider.is_none());
    assert_eq!(
        first.function.arguments,
        serde_json::json!({"city": "Tokyo"})
    );
    assert_eq!(
        second.function.arguments,
        serde_json::json!({"zone": "UTC"})
    );
}

#[test]
fn finish_discards_calls_still_open_at_stream_end() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("call_1"), "ping");
    accumulator.tool_args_delta(&pid("call_1"), "{\"x\":");
    let parts = accumulator.finish();
    // Discarding the only (incomplete) call leaves the turn with nothing,
    // which is now representable instead of being padded with an empty
    // text part.
    assert!(parts.is_empty());
    assert!(!accumulator.saw_tool_call());
}

#[test]
fn the_tool_id_override_supersedes_the_assembly_key() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("tool-0"), "ping");
    let mut done = end("tool-0", UnparseableToolInput::Drop);
    done.tool_id = WireId::new("call_late");
    let (tool_call, _) = accumulator
        .tool_input_end(done)
        .expect("no error")
        .expect("finalizes");
    assert_eq!(tool_call.id, "call_late");
}

#[test]
fn a_full_call_adopts_the_internal_id_its_deltas_published() {
    let mut accumulator = PartsAccumulator::new();
    let published = accumulator.tool_name_delta(&pid("tc1"), "add");
    accumulator.tool_args_delta(&pid("tc1"), "{\"x\":1}");

    let adopted =
        accumulator.tool_call(&pid("tc1"), call_named("tc1", "add"), InternalCallId::new());
    assert_eq!(adopted, published);
    assert_eq!(
        accumulator
            .finish()
            .iter()
            .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
            .count(),
        1
    );
}

#[test]
fn an_end_after_a_full_call_for_the_same_key_is_a_no_op() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("tc1"), "add");
    accumulator.tool_call(&pid("tc1"), call_named("tc1", "add"), InternalCallId::new());
    let mut done = end("tc1", UnparseableToolInput::Drop);
    done.name = Some("add".to_owned());
    done.arguments = Some(serde_json::json!({"x": 1}));
    assert!(
        accumulator
            .tool_input_end(done)
            .expect("no error")
            .is_none()
    );
    assert_eq!(
        accumulator
            .finish()
            .iter()
            .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
            .count(),
        1
    );
}

#[test]
fn fragments_reusing_a_finished_key_open_a_fresh_call() {
    let mut accumulator = PartsAccumulator::new();
    let first = accumulator.tool_name_delta(&pid("tc1"), "add");
    accumulator.tool_call(&pid("tc1"), call_named("tc1", "add"), InternalCallId::new());
    let second = accumulator.tool_name_delta(&pid("tc1"), "subtract");
    assert_ne!(second, first);
    accumulator.tool_args_delta(&pid("tc1"), "{\"y\":2}");
    let (tool_call, internal) = accumulator
        .tool_input_end(end("tc1", UnparseableToolInput::Drop))
        .expect("no error")
        .expect("the reused key's call finalizes");
    assert_eq!(internal, second);
    assert_eq!(tool_call.function.name, "subtract");
}

#[test]
fn a_wire_restatement_adopts_the_single_open_minted_assembly() {
    let mut accumulator = PartsAccumulator::new();
    let published = accumulator.tool_name_delta(&pid("tool-0"), "add");
    accumulator.tool_args_delta(&pid("tool-0"), "{\"x\":1}");
    // A genuine restatement: same tool, arguments covering the
    // streamed fragments.
    let restated = ToolCall::from_wire(
        "call_late",
        crate::message::ToolFunction {
            name: "add".to_owned(),
            arguments: serde_json::json!({"x": 1}),
        },
    );
    let adopted = accumulator.tool_call(&pid("call_late"), restated, InternalCallId::new());
    assert_eq!(adopted, published);
    assert!(
        accumulator
            .tool_input_end(end("tool-0", UnparseableToolInput::Drop))
            .expect("no error")
            .is_none()
    );
}

/// A gateway that fragments arguments WITHOUT ever sending a name
/// delta leaves the assembly nameless (`ensure_open_tool_input`
/// records `""`). An empty name is no evidence against the
/// restatement — demanding name equality published the whole call
/// under a fresh minted id (breaking delta correlation) and left the
/// assembly to finalize as a duplicate.
#[test]
fn a_wire_restatement_adopts_a_nameless_args_only_assembly() {
    let mut accumulator = PartsAccumulator::new();
    let published = accumulator.tool_args_delta(&pid("tool-0"), "{\"x\":1}");
    let restated = ToolCall::from_wire(
        "call_late",
        crate::message::ToolFunction {
            name: "add".to_owned(),
            arguments: serde_json::json!({"x": 1}),
        },
    );
    let adopted = accumulator.tool_call(&pid("call_late"), restated, InternalCallId::new());
    assert_eq!(adopted, published);
    assert!(
        accumulator
            .tool_input_end(end("tool-0", UnparseableToolInput::Drop))
            .expect("no error")
            .is_none(),
        "the adopted assembly must not finalize a second time"
    );
}

/// A buffer still holding the literal `null` placeholder (the gateway
/// shape `tool_args_delta` documents) streamed no real arguments:
/// any restatement covers it vacuously. Strict subset comparison
/// rejected it (`{...} != null`) and published a duplicate.
#[test]
fn a_wire_restatement_adopts_an_assembly_holding_the_null_placeholder() {
    let mut accumulator = PartsAccumulator::new();
    let published = accumulator.tool_name_delta(&pid("tool-0"), "add");
    accumulator.tool_args_delta(&pid("tool-0"), "null");
    let restated = ToolCall::from_wire(
        "call_late",
        crate::message::ToolFunction {
            name: "add".to_owned(),
            arguments: serde_json::json!({"x": 1}),
        },
    );
    let adopted = accumulator.tool_call(&pid("call_late"), restated, InternalCallId::new());
    assert_eq!(adopted, published);
    assert!(
        accumulator
            .tool_input_end(end("tool-0", UnparseableToolInput::Drop))
            .expect("no error")
            .is_none()
    );
}

/// A wire-keyed whole call must not steal an open minted assembly it
/// doesn't evidently restate: a different tool name (or arguments that
/// don't cover the streamed fragments) is an UNRELATED call. The
/// assembly stays open for its own end — its streamed arguments were
/// silently lost when adoption keyed on cardinality alone.
#[test]
fn an_unrelated_wire_keyed_call_does_not_steal_the_open_assembly() {
    let mut accumulator = PartsAccumulator::new();
    let published = accumulator.tool_name_delta(&pid("tool-0"), "get_weather");
    accumulator.tool_args_delta(&pid("tool-0"), "{\"city\":\"Paris\"}");

    let unrelated = ToolCall::from_wire(
        "call_other",
        crate::message::ToolFunction {
            name: "get_time".to_owned(),
            arguments: serde_json::json!({"zone": "UTC"}),
        },
    );
    let fresh = InternalCallId::new();
    let adopted = accumulator.tool_call(&pid("call_other"), unrelated, fresh);
    assert_eq!(adopted, fresh, "an unrelated call has nothing to adopt");

    // The assembly still finalizes from its own end, streamed args intact.
    let (weather, internal) = accumulator
        .tool_input_end(end("tool-0", UnparseableToolInput::Error))
        .expect("no error")
        .expect("the open assembly must finalize");
    assert_eq!(internal, published);
    assert_eq!(weather.function.name, "get_weather");
    assert_eq!(
        weather.function.arguments,
        serde_json::json!({"city": "Paris"})
    );

    let calls = accumulator
        .finish()
        .iter()
        .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
        .count();
    assert_eq!(calls, 2, "both calls survive as distinct parts");
}

/// Same-name adoption still demands the arguments cover the fragments:
/// a same-tool call whose args contradict what streamed is a second
/// invocation of that tool, not a restatement.
#[test]
fn a_same_name_call_with_foreign_arguments_does_not_adopt() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("tool-0"), "add");
    accumulator.tool_args_delta(&pid("tool-0"), "{\"x\":1}");
    let second_invocation = ToolCall::from_wire(
        "call_other",
        crate::message::ToolFunction {
            name: "add".to_owned(),
            arguments: serde_json::json!({"x": 99}),
        },
    );
    let fresh = InternalCallId::new();
    let adopted = accumulator.tool_call(&pid("call_other"), second_invocation, fresh);
    assert_eq!(adopted, fresh);
    assert!(
        accumulator
            .tool_input_end(end("tool-0", UnparseableToolInput::Error))
            .expect("no error")
            .is_some(),
        "the assembly still finalizes separately"
    );
}

#[test]
fn a_wire_restatement_never_guesses_between_two_minted_assemblies() {
    let mut accumulator = PartsAccumulator::new();
    let first = accumulator.tool_name_delta(&pid("tool-0"), "add");
    let second = accumulator.tool_name_delta(&pid("tool-1"), "subtract");
    let minted = InternalCallId::new();
    let published =
        accumulator.tool_call(&pid("call_late"), call_named("call_late", "add"), minted);
    assert_eq!(published, minted);
    assert_ne!(published, first);
    assert_ne!(published, second);
}

#[test]
fn oversized_tool_input_truncates_and_finalizes_through_policy() {
    let mut accumulator = PartsAccumulator::new();
    accumulator.tool_name_delta(&pid("call_1"), "bulk");
    let chunk = "x".repeat(1024 * 1024);
    accumulator.tool_args_delta(&pid("call_1"), "{\"data\":\"");
    for _ in 0..33 {
        accumulator.tool_args_delta(&pid("call_1"), &chunk);
    }
    accumulator.tool_args_delta(&pid("call_1"), "\"}");
    assert!(
        accumulator
            .tool_input_end(end("call_1", UnparseableToolInput::Drop))
            .expect("no error")
            .is_none(),
        "the truncated input must not fabricate a call"
    );
}
