use super::*;
use crate::error::ErrorReport;
use crate::message::AdditionalParams;
use crate::streaming::{MintKind, non_empty_id};

/// Test-side key syntax: legacy minted renderings decode to minted
/// keys; anything else is wire-derived.
fn pid(id: &str) -> BlockId {
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
    BlockId::wire(id)
}

/// Provider handle matching the key syntax: wire-shaped ids carry
/// themselves; minted renderings carry none.
fn wid(id: &str) -> Option<String> {
    pid(id).wire_str().and_then(|_| non_empty_id(id))
}

fn full(id: &str, content: ReasoningContent) -> Reasoning {
    Reasoning {
        id: wid(id),
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

fn tool_calls(parts: &[AssistantContent]) -> usize {
    parts
        .iter()
        .filter(|part| matches!(part, AssistantContent::ToolCall(_)))
        .count()
}

// --- event helpers: the accumulator is driven only through `apply` ---

/// A text delta for the block `id` (a wire id per text block).
fn text(accumulator: &mut BlockAccumulator, id: &str, text: &str) {
    accumulator
        .apply(&StreamEvent::text(pid(id), text))
        .expect("text deltas never fail");
}

fn text_start(
    accumulator: &mut BlockAccumulator,
    id: &str,
    additional_params: Option<AdditionalParams>,
) {
    accumulator
        .apply(&StreamEvent::BlockStart {
            id: pid(id),
            kind: BlockKind::Text { additional_params },
        })
        .expect("text starts never fail");
}

fn text_end(accumulator: &mut BlockAccumulator, id: &str) {
    accumulator
        .apply(&StreamEvent::BlockEnd {
            id: pid(id),
            end: BlockClose::Text,
            block: None,
        })
        .expect("text ends never fail");
}

/// A reasoning delta for `id`, preceded by its `BlockStart` (carrying the
/// key's provider handle) when the key has no open part — what an adapter
/// emits for a delta under an unseen id.
fn reasoning_delta(accumulator: &mut BlockAccumulator, id: &str, text: &str) {
    let key = pid(id);
    if !accumulator.open_reasoning.contains_key(&key) {
        accumulator
            .apply(&StreamEvent::BlockStart {
                id: key.clone(),
                kind: BlockKind::Reasoning {
                    provider_id: wid(id),
                },
            })
            .expect("reasoning starts never fail");
    }
    accumulator
        .apply(&StreamEvent::BlockDelta {
            id: key,
            delta: Delta::Reasoning {
                text: text.to_owned(),
            },
        })
        .expect("reasoning deltas never fail");
}

/// A wire-sent reasoning end; returns the completed part.
fn reasoning_end(
    accumulator: &mut BlockAccumulator,
    id: &str,
    reasoning: Option<Reasoning>,
    signature: Option<String>,
) -> Option<Reasoning> {
    accumulator
        .apply(&StreamEvent::BlockEnd {
            id: pid(id),
            end: BlockClose::Reasoning {
                reasoning,
                signature,
                wire_sent: true,
            },
            block: None,
        })
        .expect("reasoning ends never fail")
        .map(|(_, block)| match block {
            AssistantContent::Reasoning(reasoning) => reasoning,
            other => panic!("a reasoning end completed a non-reasoning block: {other:?}"),
        })
}

fn tool_name_delta(accumulator: &mut BlockAccumulator, id: &str, name: &str) {
    accumulator
        .apply(&StreamEvent::BlockDelta {
            id: pid(id),
            delta: Delta::ToolName {
                name: name.to_owned(),
            },
        })
        .expect("tool name deltas never fail");
}

fn tool_args_delta(accumulator: &mut BlockAccumulator, id: &str, arguments: &str) {
    accumulator
        .apply(&StreamEvent::BlockDelta {
            id: pid(id),
            delta: Delta::ToolArguments {
                arguments: arguments.to_owned(),
            },
        })
        .expect("tool argument deltas never fail");
}

fn end(mode: UnparseableToolInput) -> ToolCallEnd {
    ToolCallEnd::new(mode)
}

/// Close the tool call `id` with `end`; returns the completed call.
fn tool_end(
    accumulator: &mut BlockAccumulator,
    id: &str,
    end: ToolCallEnd,
) -> Result<Option<ToolCall>, ErrorReport> {
    Ok(accumulator
        .apply(&StreamEvent::BlockEnd {
            id: pid(id),
            end: BlockClose::ToolCall(end),
            block: None,
        })?
        .and_then(|(_, block)| match block {
            AssistantContent::ToolCall(call) => Some(call),
            AssistantContent::Text(_)
            | AssistantContent::Reasoning(_)
            | AssistantContent::Image(_) => None,
        }))
}

/// A whole call under the key `id` (a wire key doubles as the durable
/// tool id); returns the block id the call was published under.
fn tool_call(
    accumulator: &mut BlockAccumulator,
    id: &str,
    name: &str,
    arguments: serde_json::Value,
) -> BlockId {
    let key = pid(id);
    let mut end = ToolCallEnd::whole(name, arguments);
    if let Some(tool_id) = key.wire_str() {
        end = end.with_tool_id(tool_id);
    }
    let (published, block) = accumulator
        .apply(&StreamEvent::BlockEnd {
            id: key,
            end: BlockClose::ToolCall(end),
            block: None,
        })
        .expect("whole calls never fail")
        .expect("a whole call always finalizes");
    assert!(matches!(block, AssistantContent::ToolCall(_)));
    published
}

// --- reasoning lifecycle ---

/// An end's authoritative restatement supersedes the open part's delta
/// accumulation (pydantic-ai `_replace_part` semantics).
#[test]
fn an_end_restatement_replaces_its_delta_accumulation() {
    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "rs_1", "partial ");
    reasoning_delta(&mut accumulator, "rs_1", "thought");
    reasoning_end(
        &mut accumulator,
        "rs_1",
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
    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "rs_1", "partial");
    let completed = reasoning_end(
        &mut accumulator,
        "rs_1",
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

    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "rs_2", "partial");
    let completed = reasoning_end(
        &mut accumulator,
        "rs_2",
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
    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "rs_1", "thinking");
    tool_call(&mut accumulator, "call_1", "probe", serde_json::json!({}));
    reasoning_end(
        &mut accumulator,
        "rs_1",
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
    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "reasoning-0", "thought A");
    reasoning_end(
        &mut accumulator,
        "reasoning-0",
        None,
        Some("sig_A".to_string()),
    );
    // A signature-only end for the already-finished constant key: the
    // wire signed a second segment with nothing new streamed to sign.
    reasoning_end(
        &mut accumulator,
        "reasoning-0",
        None,
        Some("sig_B".to_string()),
    );

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
    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "reasoning-0", "A");
    reasoning_end(&mut accumulator, "reasoning-0", None, None);
    text(&mut accumulator, "text-0", "visible");
    reasoning_delta(&mut accumulator, "reasoning-0", "B");

    let parts = accumulator.finish();
    assert_eq!(reasoning_texts(&parts), vec!["A", "B"]);
    assert!(matches!(
        parts.get(1),
        Some(AssistantContent::Text(text)) if text.text == "visible"
    ));
}

/// A bare delta (no start) after the end also opens a NEW part: the
/// lenient bare-delta rule never resurrects the finished part.
#[test]
fn a_bare_delta_after_the_end_opens_a_fresh_part() {
    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "reasoning-0", "A");
    reasoning_end(&mut accumulator, "reasoning-0", None, None);
    accumulator
        .apply(&StreamEvent::BlockDelta {
            id: pid("reasoning-0"),
            delta: Delta::Reasoning {
                text: "B".to_owned(),
            },
        })
        .expect("no error");

    assert_eq!(reasoning_texts(&accumulator.finish()), vec!["A", "B"]);
}

/// Same-key whole blocks after the entity finished are siblings: every
/// part survives, in arrival order (the Responses multi-part item).
#[test]
fn same_key_whole_blocks_are_siblings_and_all_survive() {
    let mut accumulator = BlockAccumulator::new();
    for content in [
        summary("s1"),
        summary("s2"),
        reasoning_text("visible"),
        ReasoningContent::Encrypted("enc".to_owned()),
    ] {
        reasoning_end(&mut accumulator, "rs_1", Some(full("rs_1", content)), None);
    }

    let parts = accumulator.finish();
    assert_eq!(reasoning_texts(&parts), vec!["s1", "s2", "visible", "enc"]);
}

/// Deltas then the item's multi-part done block: the first restatement
/// supersedes the delta accumulation, the rest append as siblings.
#[test]
fn deltas_then_sibling_whole_blocks_keep_each_part_once() {
    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "rs_1", "s1");
    for content in [
        summary("s1"),
        summary("s2"),
        ReasoningContent::Encrypted("enc".to_owned()),
    ] {
        reasoning_end(&mut accumulator, "rs_1", Some(full("rs_1", content)), None);
    }

    let parts = accumulator.finish();
    assert_eq!(reasoning_texts(&parts), vec!["s1", "s2", "enc"]);
}

#[test]
fn distinct_keys_never_interact() {
    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "rs_1", "first item deltas");
    reasoning_end(
        &mut accumulator,
        "rs_2",
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
    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "reasoning-0", "the chain");
    reasoning_end(&mut accumulator, "reasoning-0", None, None);
    text(&mut accumulator, "text-0", "answer");
    reasoning_end(
        &mut accumulator,
        "reasoning-0",
        None,
        Some("sig".to_owned()),
    );

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
    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "reasoning-0", "the chain");
    reasoning_end(
        &mut accumulator,
        "reasoning-0",
        None,
        Some("sig".to_owned()),
    );

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
    let mut accumulator = BlockAccumulator::new();
    reasoning_end(
        &mut accumulator,
        "reasoning-0",
        None,
        Some("sig".to_owned()),
    );
    let parts = accumulator.finish();
    assert_eq!(parts.len(), 1);
}

/// A bare repeated end is a no-op: idempotence belongs to the entity.
#[test]
fn repeated_bare_ends_are_no_ops() {
    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "reasoning-0", "A");
    assert!(reasoning_end(&mut accumulator, "reasoning-0", None, None).is_some());
    assert!(reasoning_end(&mut accumulator, "reasoning-0", None, None).is_none());
    assert!(reasoning_end(&mut accumulator, "reasoning-0", None, None).is_none());
    assert_eq!(accumulator.finish().len(), 1);
}

/// Only a bare end the adapter *synthesized* stays silent on `apply`; the
/// part it closed is still in the choice.
#[test]
fn a_synthesized_bare_end_finalizes_silently() {
    let mut accumulator = BlockAccumulator::new();
    reasoning_delta(&mut accumulator, "reasoning-0", "A");
    let completed = accumulator
        .apply(&StreamEvent::BlockEnd {
            id: pid("reasoning-0"),
            end: BlockClose::Reasoning {
                reasoning: None,
                signature: None,
                wire_sent: false,
            },
            block: None,
        })
        .expect("no error");
    assert!(
        completed.is_none(),
        "a synthesized bare end publishes nothing"
    );
    assert_eq!(reasoning_texts(&accumulator.finish()), vec!["A"]);
}

// --- text lifecycle ---

#[test]
fn text_and_reasoning_interleave_in_arrival_order() {
    let mut accumulator = BlockAccumulator::new();
    text(&mut accumulator, "text-0", "intro");
    reasoning_end(
        &mut accumulator,
        "rs_1",
        Some(full("rs_1", summary("thinking"))),
        None,
    );
    text_start(&mut accumulator, "msg_2", None);
    text(&mut accumulator, "msg_2", "out");
    text(&mut accumulator, "msg_2", "ro");

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
    let mut accumulator = BlockAccumulator::new();
    text_start(&mut accumulator, "msg_1", None);
    text(&mut accumulator, "msg_1", "first");
    text_start(&mut accumulator, "msg_2", None);
    text(&mut accumulator, "msg_2", "second");

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

/// A text start whose key was already seen reactivates that block
/// across interleaved output instead of opening a duplicate part.
#[test]
fn a_seen_text_start_id_reactivates_its_block() {
    let mut accumulator = BlockAccumulator::new();
    text_start(&mut accumulator, "msg_1", None);
    text(&mut accumulator, "msg_1", "collapsing ");
    reasoning_end(
        &mut accumulator,
        "rs_1",
        Some(full("rs_1", summary("thinking"))),
        None,
    );
    text_start(&mut accumulator, "msg_1", None);
    text(&mut accumulator, "msg_1", "text");

    let parts = accumulator.finish();
    assert_eq!(parts.len(), 2, "one text part, one reasoning part");
    assert!(matches!(
        parts.first(),
        Some(AssistantContent::Text(text)) if text.text == "collapsing text"
    ));
}

/// A text start that never receives content leaves no empty part.
#[test]
fn a_content_less_text_start_leaves_no_empty_part() {
    let mut accumulator = BlockAccumulator::new();
    text_start(&mut accumulator, "msg_1", None);
    reasoning_end(
        &mut accumulator,
        "rs_1",
        Some(full("rs_1", summary("thinking"))),
        None,
    );

    let parts = accumulator.finish();
    assert_eq!(parts.len(), 1);
    assert!(matches!(
        parts.first(),
        Some(AssistantContent::Reasoning(_))
    ));
}

/// A text start's metadata opens the block with it, and later metadata
/// deltas merge into the same block.
#[test]
fn text_metadata_opens_and_merges_into_the_block() {
    let params = |value: serde_json::Value| {
        AdditionalParams::try_from_value(value)
            .expect("object")
            .expect("non-empty")
    };
    let mut accumulator = BlockAccumulator::new();
    text_start(
        &mut accumulator,
        "msg_1",
        Some(params(serde_json::json!({"block": 1}))),
    );
    text(&mut accumulator, "msg_1", "cited");
    accumulator
        .apply(&StreamEvent::BlockDelta {
            id: pid("msg_1"),
            delta: Delta::TextMeta {
                additional_params: params(serde_json::json!({"citations": ["a"]})),
            },
        })
        .expect("no error");

    let parts = accumulator.finish();
    assert_eq!(parts.len(), 1);
    let Some(AssistantContent::Text(text)) = parts.first() else {
        panic!("expected a text part");
    };
    assert_eq!(text.text, "cited");
    let params = text.additional_params.as_ref().expect("metadata kept");
    assert_eq!(params["block"], 1);
    assert_eq!(params["citations"], serde_json::json!(["a"]));
}

/// An explicit text end closes the block: a later delta under a fresh
/// id (what an adapter mints for bare text after an end) opens a fresh
/// part rather than extending the closed one.
#[test]
fn text_end_closes_the_block_for_later_deltas() {
    let mut accumulator = BlockAccumulator::new();
    text_start(&mut accumulator, "msg_1", None);
    text(&mut accumulator, "msg_1", "first");
    text_end(&mut accumulator, "msg_1");
    text(&mut accumulator, "text-0", "second");

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
    let mut accumulator = BlockAccumulator::new();
    let parts = accumulator.finish();
    // Was one fabricated empty-text part, which the old non-empty content
    // type forced and which was indistinguishable on the wire from a real
    // empty text block.
    assert!(parts.is_empty());
}

// --- tool-call lifecycle (the settled semantics) ---

#[test]
fn fragments_assemble_into_a_completed_tool_call() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_1", "get_weather");
    tool_args_delta(&mut accumulator, "call_1", "{\"location\":");
    tool_args_delta(&mut accumulator, "call_1", "\"Paris\"}");

    let tool_call = tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Drop))
        .expect("no error")
        .expect("call must finalize");
    assert_eq!(tool_call.id, "call_1");
    assert_eq!(tool_call.function.name, "get_weather");
    assert!(accumulator.saw_tool_call());
}

#[test]
fn an_empty_name_fragment_does_not_erase_an_established_name() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_1", "get_weather");
    tool_name_delta(&mut accumulator, "call_1", "");
    tool_args_delta(&mut accumulator, "call_1", "{}");

    let tool_call = tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Drop))
        .expect("no error")
        .expect("call must finalize under the established name");
    assert_eq!(tool_call.function.name, "get_weather");
}

#[test]
fn a_call_with_no_streamed_arguments_is_a_parameterless_invocation() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_1", "ping");
    let tool_call = tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Drop))
        .expect("no error")
        .expect("parameterless calls are preserved");
    assert_eq!(tool_call.function.arguments, serde_json::json!({}));
}

#[test]
fn drop_mode_drops_partial_arguments_and_nameless_calls() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_1", "ping");
    tool_args_delta(&mut accumulator, "call_1", "{\"x\":");
    assert!(
        tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Drop))
            .expect("no error")
            .is_none()
    );
    tool_args_delta(&mut accumulator, "call_2", "{\"y\":1}");
    assert!(
        tool_end(&mut accumulator, "call_2", end(UnparseableToolInput::Drop))
            .expect("no error")
            .is_none(),
        "a call whose name never arrived is dropped"
    );
    assert!(!accumulator.saw_tool_call());
}

#[test]
fn error_mode_surfaces_malformed_input_as_an_error() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_1", "get_weather");
    tool_args_delta(&mut accumulator, "call_1", "{\"location\": not-json");
    let err = tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Error))
        .expect_err("malformed complete input must error");
    assert!(err.to_string().contains("get_weather"));
}

#[test]
fn keep_mode_leaves_the_call_open_for_later_fragments() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_1", "search");
    tool_args_delta(&mut accumulator, "call_1", "{\"q\":\"ru");
    assert!(
        tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Keep))
            .expect("no error")
            .is_none()
    );
    tool_args_delta(&mut accumulator, "call_1", "st\"}");
    let tool_call = tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Drop))
        .expect("no error")
        .expect("extended call finalizes");
    assert_eq!(
        tool_call.function.arguments,
        serde_json::json!({"q": "rust"})
    );
}

#[test]
fn authoritative_end_fields_supersede_assembly() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "fc_1", "provisional");
    tool_args_delta(&mut accumulator, "fc_1", "{\"partial\":");

    let mut done = end(UnparseableToolInput::Drop);
    done.name = Some("final_name".to_owned());
    done.arguments = Some(serde_json::json!({"x": 1}));
    done.call_id = Some("call_abc".to_owned());
    let tool_call = tool_end(&mut accumulator, "fc_1", done)
        .expect("no error")
        .expect("authoritative payload finalizes");
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
    let mut accumulator = BlockAccumulator::new();
    let mut done = end(UnparseableToolInput::Drop);
    done.name = Some("add".to_owned());
    done.arguments = Some(serde_json::json!({"x": 2}));
    let tool_call = tool_end(&mut accumulator, "fc_1", done)
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
    let mut accumulator = BlockAccumulator::new();
    let mut done = end(UnparseableToolInput::Drop);
    done.name = Some("add".to_owned());
    done.arguments = Some(serde_json::json!({"x": 2}));
    tool_end(&mut accumulator, "fc_1", done.clone())
        .expect("no error")
        .expect("finalizes once");
    assert!(
        tool_end(&mut accumulator, "fc_1", done)
            .expect("no error")
            .is_none(),
        "a repeated authoritative end must not duplicate the call"
    );
    assert_eq!(tool_calls(&accumulator.finish()), 1);
}

/// A drop is a finalization: a truncated call dismissed under `Drop`
/// must not be resurrected by a later payload-bearing end for the same
/// key — that end is the same repeated-end defect the finished-set
/// guard exists for, arriving after a drop instead of a success.
#[test]
fn a_dropped_truncated_call_is_not_resurrected_by_a_payload_bearing_end() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_1", "get_weather");
    tool_args_delta(&mut accumulator, "call_1", "{\"loc\":");
    assert!(
        tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Drop))
            .expect("no error")
            .is_none(),
        "truncated arguments drop"
    );
    let mut retry = end(UnparseableToolInput::Drop);
    retry.name = Some("get_weather".to_owned());
    retry.arguments = Some(serde_json::json!({"loc": "Paris"}));
    assert!(
        tool_end(&mut accumulator, "call_1", retry)
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
    let mut accumulator = BlockAccumulator::new();
    tool_args_delta(&mut accumulator, "call_1", "{\"y\":1}");
    assert!(
        tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Drop))
            .expect("no error")
            .is_none(),
        "nameless call drops"
    );
    let mut retry = end(UnparseableToolInput::Drop);
    retry.name = Some("late_name".to_owned());
    retry.arguments = Some(serde_json::json!({"y": 1}));
    assert!(
        tool_end(&mut accumulator, "call_1", retry)
            .expect("no error")
            .is_none(),
        "the dropped entity is finished; a late name must not fabricate a phantom call"
    );
    assert!(!accumulator.saw_tool_call());
}

#[test]
fn a_stale_end_for_a_finalized_key_is_a_no_op() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_1", "ping");
    tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Drop))
        .expect("no error")
        .expect("finalizes");
    assert!(
        tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Drop))
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
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_1", "big");
    let oversized = "x".repeat(MAX_TOOL_INPUT_BYTES + 1);
    tool_args_delta(&mut accumulator, "call_1", &oversized);
    assert!(
        tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Drop))
            .expect("no error")
            .is_none(),
        "overflow forces the unparseable path; the call must not finalize"
    );
    assert!(!accumulator.saw_tool_call());
}

#[test]
fn null_placeholder_is_replaced_by_following_json_fragments() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_123", "web_search");
    tool_args_delta(&mut accumulator, "call_123", "null");
    tool_args_delta(&mut accumulator, "call_123", "{\"query\": \"META");
    tool_args_delta(&mut accumulator, "call_123", " Platforms news\"}");
    let tool_call = tool_end(
        &mut accumulator,
        "call_123",
        end(UnparseableToolInput::Drop),
    )
    .expect("no error")
    .expect("call must finalize");
    assert_eq!(
        tool_call.function.arguments,
        serde_json::json!({"query": "META Platforms news"})
    );
}

#[test]
fn parallel_calls_assemble_independently() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_a", "get_weather");
    tool_name_delta(&mut accumulator, "call_b", "get_time");
    tool_args_delta(&mut accumulator, "call_a", "{\"location\":\"Paris\"}");
    tool_args_delta(&mut accumulator, "call_b", "{\"zone\":\"UTC\"}");
    let call_b = tool_end(&mut accumulator, "call_b", end(UnparseableToolInput::Drop))
        .expect("no error")
        .expect("finalizes");
    let call_a = tool_end(&mut accumulator, "call_a", end(UnparseableToolInput::Drop))
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
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "tool-0", "get_weather");
    tool_name_delta(&mut accumulator, "tool-1", "get_time");
    tool_args_delta(&mut accumulator, "tool-0", "{\"city\":");
    tool_args_delta(&mut accumulator, "tool-1", "{\"zone\":");
    tool_args_delta(&mut accumulator, "tool-0", "\"Tokyo\"}");
    tool_args_delta(&mut accumulator, "tool-1", "\"UTC\"}");
    let first = tool_end(&mut accumulator, "tool-0", end(UnparseableToolInput::Drop))
        .expect("no error")
        .expect("finalizes");
    let second = tool_end(&mut accumulator, "tool-1", end(UnparseableToolInput::Drop))
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
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_1", "ping");
    tool_args_delta(&mut accumulator, "call_1", "{\"x\":");
    let parts = accumulator.finish();
    // Discarding the only (incomplete) call leaves the turn with nothing,
    // which is now representable instead of being padded with an empty
    // text part.
    assert!(parts.is_empty());
    assert!(!accumulator.saw_tool_call());
}

#[test]
fn the_tool_id_override_supersedes_the_assembly_key() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "tool-0", "ping");
    let done = end(UnparseableToolInput::Drop).with_tool_id("call_late");
    let tool_call = tool_end(&mut accumulator, "tool-0", done)
        .expect("no error")
        .expect("finalizes");
    assert_eq!(tool_call.id, "call_late");
}

#[test]
fn a_full_call_adopts_the_block_id_its_deltas_published() {
    let mut accumulator = BlockAccumulator::new();
    let published = pid("tc1");
    tool_name_delta(&mut accumulator, "tc1", "add");
    tool_args_delta(&mut accumulator, "tc1", "{\"x\":1}");

    let adopted = tool_call(&mut accumulator, "tc1", "add", serde_json::json!({}));
    assert_eq!(adopted, published);
    assert_eq!(tool_calls(&accumulator.finish()), 1);
}

#[test]
fn an_end_after_a_full_call_for_the_same_key_is_a_no_op() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "tc1", "add");
    tool_call(&mut accumulator, "tc1", "add", serde_json::json!({}));
    let mut done = end(UnparseableToolInput::Drop);
    done.name = Some("add".to_owned());
    done.arguments = Some(serde_json::json!({"x": 1}));
    assert!(
        tool_end(&mut accumulator, "tc1", done)
            .expect("no error")
            .is_none()
    );
    assert_eq!(tool_calls(&accumulator.finish()), 1);
}

#[test]
fn fragments_reusing_a_finished_key_open_a_fresh_call() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "tc1", "add");
    tool_call(&mut accumulator, "tc1", "add", serde_json::json!({}));
    tool_name_delta(&mut accumulator, "tc1", "subtract");
    tool_args_delta(&mut accumulator, "tc1", "{\"y\":2}");
    let call = tool_end(&mut accumulator, "tc1", end(UnparseableToolInput::Drop))
        .expect("no error")
        .expect("the reused key's call finalizes");
    assert_eq!(call.function.name, "subtract");
    assert_eq!(
        tool_calls(&accumulator.finish()),
        2,
        "the reused key opened a second, distinct call"
    );
}

#[test]
fn a_wire_restatement_adopts_the_single_open_minted_assembly() {
    let mut accumulator = BlockAccumulator::new();
    let published = pid("tool-0");
    tool_name_delta(&mut accumulator, "tool-0", "add");
    tool_args_delta(&mut accumulator, "tool-0", "{\"x\":1}");
    // A genuine restatement: same tool, arguments covering the
    // streamed fragments.
    let adopted = tool_call(
        &mut accumulator,
        "call_late",
        "add",
        serde_json::json!({"x": 1}),
    );
    assert_eq!(adopted, published);
    assert!(
        tool_end(&mut accumulator, "tool-0", end(UnparseableToolInput::Drop))
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
    let mut accumulator = BlockAccumulator::new();
    let published = pid("tool-0");
    tool_args_delta(&mut accumulator, "tool-0", "{\"x\":1}");
    let adopted = tool_call(
        &mut accumulator,
        "call_late",
        "add",
        serde_json::json!({"x": 1}),
    );
    assert_eq!(adopted, published);
    assert!(
        tool_end(&mut accumulator, "tool-0", end(UnparseableToolInput::Drop))
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
    let mut accumulator = BlockAccumulator::new();
    let published = pid("tool-0");
    tool_name_delta(&mut accumulator, "tool-0", "add");
    tool_args_delta(&mut accumulator, "tool-0", "null");
    let adopted = tool_call(
        &mut accumulator,
        "call_late",
        "add",
        serde_json::json!({"x": 1}),
    );
    assert_eq!(adopted, published);
    assert!(
        tool_end(&mut accumulator, "tool-0", end(UnparseableToolInput::Drop))
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
    let mut accumulator = BlockAccumulator::new();
    let published = pid("tool-0");
    tool_name_delta(&mut accumulator, "tool-0", "get_weather");
    tool_args_delta(&mut accumulator, "tool-0", "{\"city\":\"Paris\"}");

    let adopted = tool_call(
        &mut accumulator,
        "call_other",
        "get_time",
        serde_json::json!({"zone": "UTC"}),
    );
    assert_eq!(
        adopted,
        pid("call_other"),
        "an unrelated call has nothing to adopt"
    );
    assert_ne!(adopted, published);

    // The assembly still finalizes from its own end, streamed args intact.
    let weather = tool_end(&mut accumulator, "tool-0", end(UnparseableToolInput::Error))
        .expect("no error")
        .expect("the open assembly must finalize");
    assert_eq!(weather.function.name, "get_weather");
    assert_eq!(
        weather.function.arguments,
        serde_json::json!({"city": "Paris"})
    );

    assert_eq!(
        tool_calls(&accumulator.finish()),
        2,
        "both calls survive as distinct parts"
    );
}

/// Same-name adoption still demands the arguments cover the fragments:
/// a same-tool call whose args contradict what streamed is a second
/// invocation of that tool, not a restatement.
#[test]
fn a_same_name_call_with_foreign_arguments_does_not_adopt() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "tool-0", "add");
    tool_args_delta(&mut accumulator, "tool-0", "{\"x\":1}");
    let adopted = tool_call(
        &mut accumulator,
        "call_other",
        "add",
        serde_json::json!({"x": 99}),
    );
    assert_eq!(adopted, pid("call_other"));
    assert!(
        tool_end(&mut accumulator, "tool-0", end(UnparseableToolInput::Error))
            .expect("no error")
            .is_some(),
        "the assembly still finalizes separately"
    );
}

#[test]
fn a_wire_restatement_never_guesses_between_two_minted_assemblies() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "tool-0", "add");
    tool_name_delta(&mut accumulator, "tool-1", "subtract");
    let published = tool_call(&mut accumulator, "call_late", "add", serde_json::json!({}));
    assert_eq!(published, pid("call_late"));
    assert_ne!(published, pid("tool-0"));
    assert_ne!(published, pid("tool-1"));
}

#[test]
fn oversized_tool_input_truncates_and_finalizes_through_policy() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_1", "bulk");
    let chunk = "x".repeat(1024 * 1024);
    tool_args_delta(&mut accumulator, "call_1", "{\"data\":\"");
    for _ in 0..33 {
        tool_args_delta(&mut accumulator, "call_1", &chunk);
    }
    tool_args_delta(&mut accumulator, "call_1", "\"}");
    assert!(
        tool_end(&mut accumulator, "call_1", end(UnparseableToolInput::Drop))
            .expect("no error")
            .is_none(),
        "the truncated input must not fabricate a call"
    );
}

// --- the accumulator as a transport-free fold ---

/// The events one turn would carry: text, a fragmented tool call, a
/// whole reasoning block, and the terminal record.
fn scripted_turn() -> Vec<StreamEvent> {
    vec![
        StreamEvent::BlockStart {
            id: pid("text-0"),
            kind: BlockKind::Text {
                additional_params: None,
            },
        },
        StreamEvent::text(pid("text-0"), "hello "),
        StreamEvent::text(pid("text-0"), "world"),
        StreamEvent::BlockStart {
            id: pid("call_1"),
            kind: BlockKind::ToolCall,
        },
        StreamEvent::BlockDelta {
            id: pid("call_1"),
            delta: Delta::ToolName {
                name: "lookup".to_owned(),
            },
        },
        StreamEvent::BlockDelta {
            id: pid("call_1"),
            delta: Delta::ToolArguments {
                arguments: "{\"q\":\"rust\"}".to_owned(),
            },
        },
        StreamEvent::BlockEnd {
            id: pid("call_1"),
            end: BlockClose::ToolCall(ToolCallEnd::new(UnparseableToolInput::Error)),
            block: None,
        },
        StreamEvent::BlockStart {
            id: pid("rs_1"),
            kind: BlockKind::Reasoning {
                provider_id: wid("rs_1"),
            },
        },
        StreamEvent::BlockEnd {
            id: pid("rs_1"),
            end: BlockClose::Reasoning {
                reasoning: Some(full("rs_1", summary("because"))),
                signature: None,
                wire_sent: true,
            },
            block: None,
        },
        StreamEvent::Final(crate::streaming::StreamFinal::new(
            "test",
            crate::completion::Usage::new(),
        )),
    ]
}

/// The accumulator owns no transport: driven over `futures::stream::iter`
/// with no provider behind it, it folds the events into the same choice
/// a `StreamingCompletionResponse` would, and reports each finalized
/// block as the end that finalized it is applied.
#[test]
fn folds_a_provider_less_event_stream() {
    use futures::StreamExt;

    let (parts, completed) = futures::executor::block_on(async {
        let mut accumulator = BlockAccumulator::new();
        let mut completed = Vec::new();
        let mut events = futures::stream::iter(scripted_turn());
        while let Some(event) = events.next().await {
            if let Some((id, block)) = accumulator.apply(&event).expect("no error") {
                completed.push((id, block));
            }
        }
        (accumulator.finish(), completed)
    });

    assert_eq!(parts.len(), 3, "got {parts:?}");
    assert!(matches!(
        parts.first(),
        Some(AssistantContent::Text(text)) if text.text == "hello world"
    ));
    assert!(matches!(
        parts.get(1),
        Some(AssistantContent::ToolCall(call))
            if call.function.name == "lookup"
                && call.function.arguments == serde_json::json!({"q": "rust"})
    ));
    assert_eq!(reasoning_texts(&parts), vec!["because"]);

    let completed_ids: Vec<&BlockId> = completed.iter().map(|(id, _)| id).collect();
    assert_eq!(completed_ids, vec![&pid("call_1"), &pid("rs_1")]);
    assert!(matches!(
        completed.first(),
        Some((_, AssistantContent::ToolCall(call))) if call.function.name == "lookup"
    ));
    assert!(matches!(
        completed.get(1),
        Some((_, AssistantContent::Reasoning(reasoning))) if reasoning.id.as_deref() == Some("rs_1")
    ));
}

/// `snapshot` is non-destructive: two snapshots mid-stream are equal, and
/// taking them changes neither the rest of the fold nor what `finish`
/// later returns.
#[test]
fn snapshot_is_non_destructive_and_does_not_change_finish() {
    let events = scripted_turn();
    let (split_point, _) = events
        .iter()
        .enumerate()
        .find(|(_, event)| {
            matches!(
                event,
                StreamEvent::BlockEnd {
                    end: BlockClose::ToolCall(_),
                    ..
                }
            )
        })
        .expect("the script closes a call");

    // The same fold without any snapshot is the reference.
    let mut reference = BlockAccumulator::new();
    for event in &events {
        reference.apply(event).expect("no error");
    }
    let reference = reference.finish();

    let mut observed = BlockAccumulator::new();
    for event in &events[..=split_point] {
        observed.apply(event).expect("no error");
    }
    let first = observed.snapshot();
    let second = observed.snapshot();
    assert_eq!(first, second, "two snapshots mid-stream are equal");
    assert_eq!(
        first.len(),
        2,
        "text and the completed call so far: {first:?}"
    );

    for event in &events[split_point + 1..] {
        observed.apply(event).expect("no error");
    }
    let before_finish = observed.snapshot();
    let finished = observed.finish();
    assert_eq!(
        finished, before_finish,
        "finish returns what snapshot would"
    );
    assert_eq!(
        finished, reference,
        "mid-stream snapshots did not change the fold"
    );
}

/// A snapshot omits calls still open (they never fully arrived) but keeps
/// reasoning still open (its deltas are real content).
#[test]
fn snapshot_omits_open_calls_and_keeps_open_reasoning() {
    let mut accumulator = BlockAccumulator::new();
    tool_name_delta(&mut accumulator, "call_1", "ping");
    tool_args_delta(&mut accumulator, "call_1", "{\"x\":");
    reasoning_delta(&mut accumulator, "rs_1", "thinking");

    let snapshot = accumulator.snapshot();
    assert_eq!(snapshot.len(), 1, "got {snapshot:?}");
    assert_eq!(reasoning_texts(&snapshot), vec!["thinking"]);
}

/// A wire that carries no tool id names the call after the block that
/// assembled it — the same handle on every run of the same wire.
#[test]
fn an_id_less_tool_call_is_named_by_its_block_deterministically() {
    fn finalize() -> String {
        let mut accumulator = BlockAccumulator::new();
        let id = BlockId::minted(MintKind::Tool, 3);
        accumulator
            .apply(&StreamEvent::BlockStart {
                id: id.clone(),
                kind: BlockKind::ToolCall,
            })
            .expect("start");
        accumulator
            .apply(&StreamEvent::BlockDelta {
                id: id.clone(),
                delta: Delta::ToolName { name: "add".into() },
            })
            .expect("name");
        accumulator
            .apply(&StreamEvent::BlockDelta {
                id: id.clone(),
                delta: Delta::ToolArguments {
                    arguments: "{}".into(),
                },
            })
            .expect("arguments");
        let finalized = accumulator
            .apply(&StreamEvent::BlockEnd {
                id,
                end: BlockClose::ToolCall(ToolCallEnd::new(UnparseableToolInput::Error)),
                block: None,
            })
            .expect("end")
            .expect("a completed call");
        match finalized.1 {
            AssistantContent::ToolCall(call) => {
                assert!(call.provider.is_none(), "no provider id existed");
                call.id.to_string()
            }
            other => panic!("expected a tool call, got {other:?}"),
        }
    }
    assert_eq!(finalize(), "tool-3");
    assert_eq!(
        finalize(),
        finalize(),
        "the same wire mints the same handle"
    );
}
