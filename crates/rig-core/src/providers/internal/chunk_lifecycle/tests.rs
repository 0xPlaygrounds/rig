use super::*;
use crate::streaming::MintKind;

fn lifecycle() -> MintedReasoningLifecycle {
    MintedReasoningLifecycle::new(MintKind::Reasoning)
}

fn emitted(batches: Vec<ChunkParts<()>>) -> Vec<&'static str> {
    let mut lifecycle = lifecycle();
    let mut out = AdapterOutput::<()>::new();
    for parts in batches {
        lifecycle.emit_chunk(parts, &mut out);
    }
    out.iter()
        .map(|item| match item {
            Ok(RawStreamingChoice::ReasoningDelta { .. }) => "reasoning-delta",
            Ok(RawStreamingChoice::ReasoningEnd {
                signature: Some(_), ..
            }) => "signed-end",
            Ok(RawStreamingChoice::ReasoningEnd { .. }) => "bare-end",
            Ok(RawStreamingChoice::Message(_)) => "text",
            Ok(RawStreamingChoice::ToolCall(_) | RawStreamingChoice::ToolCallDelta { .. }) => {
                "tool"
            }
            _ => "other",
        })
        .collect()
}

fn tool_event() -> RawStreamingChoice<()> {
    RawStreamingChoice::ToolCall(crate::streaming::RawStreamingToolCall::new(
        BlockId::minted(MintKind::Tool, 0),
        "probe".to_owned(),
        serde_json::json!({}),
    ))
}

/// A chunk carrying every class emits canonical order, with the boundary
/// end derived between reasoning and the interleaving content.
#[test]
fn a_full_chunk_emits_canonical_order_with_the_boundary_end() {
    let order = emitted(vec![ChunkParts {
        reasoning: Some("thinking".to_owned()),
        reasoning_signature: None,
        text: Some("visible".to_owned()),
        tool_events: vec![tool_event()],
    }]);
    assert_eq!(order, vec!["reasoning-delta", "bare-end", "text", "tool"]);
}

/// A class change across chunks closes the open block exactly once.
#[test]
fn interleaving_content_closes_the_open_block_once() {
    let order = emitted(vec![
        ChunkParts {
            reasoning: Some("thinking".to_owned()),
            ..ChunkParts::default()
        },
        ChunkParts {
            text: Some("visible".to_owned()),
            ..ChunkParts::default()
        },
        ChunkParts {
            text: Some("more".to_owned()),
            ..ChunkParts::default()
        },
    ]);
    assert_eq!(order, vec!["reasoning-delta", "bare-end", "text", "text"]);
}

/// A wire-carried signature closes the block authoritatively; interleaved
/// content after it needs no synthesized end.
#[test]
fn a_signature_closes_the_block_before_text() {
    let order = emitted(vec![ChunkParts {
        reasoning: Some("thinking".to_owned()),
        reasoning_signature: Some("sig".to_owned()),
        text: Some("visible".to_owned()),
        tool_events: Vec::new(),
    }]);
    assert_eq!(order, vec!["reasoning-delta", "signed-end", "text"]);
}

/// A signature with nothing streamed still emits its close (the
/// signature-only stream — replay-required provider state).
#[test]
fn a_signature_only_chunk_emits_its_close() {
    let order = emitted(vec![ChunkParts {
        reasoning_signature: Some("sig".to_owned()),
        ..ChunkParts::default()
    }]);
    assert_eq!(order, vec!["signed-end"]);
}

/// An empty chunk (and empty-string content) emits nothing.
#[test]
fn an_empty_chunk_emits_nothing() {
    let order = emitted(vec![ChunkParts {
        reasoning: Some(String::new()),
        reasoning_signature: None,
        text: Some(String::new()),
        tool_events: Vec::new(),
    }]);
    assert!(order.is_empty());
}

/// Reasoning after a boundary opens a NEW block, closed again by the
/// next interleaving content — the reasoning→tool→reasoning shape.
#[test]
fn reasoning_reopens_after_a_boundary() {
    let order = emitted(vec![
        ChunkParts {
            reasoning: Some("before".to_owned()),
            ..ChunkParts::default()
        },
        ChunkParts {
            tool_events: vec![tool_event()],
            ..ChunkParts::default()
        },
        ChunkParts {
            reasoning: Some("after".to_owned()),
            ..ChunkParts::default()
        },
        ChunkParts {
            text: Some("done".to_owned()),
            ..ChunkParts::default()
        },
    ]);
    assert_eq!(
        order,
        vec![
            "reasoning-delta",
            "bare-end",
            "tool",
            "reasoning-delta",
            "bare-end",
            "text"
        ]
    );
}

/// Every block gets its own key: reasoning that resumes after a boundary
/// streams under a fresh minted id, while a trailing signature after a
/// synthesized boundary still addresses the block that streamed.
#[test]
fn each_block_streams_under_its_own_key() {
    let mut lifecycle = lifecycle();
    let mut out = AdapterOutput::<()>::new();
    lifecycle.emit_chunk(
        ChunkParts {
            reasoning: Some("first".into()),
            ..ChunkParts::default()
        },
        &mut out,
    );
    lifecycle.emit_chunk(
        ChunkParts {
            text: Some("visible".into()),
            ..ChunkParts::default()
        },
        &mut out,
    );
    // A late signature signs the first block, not a new one.
    lifecycle.emit_chunk(
        ChunkParts {
            reasoning_signature: Some("sig_1".into()),
            ..ChunkParts::default()
        },
        &mut out,
    );
    lifecycle.emit_chunk(
        ChunkParts {
            reasoning: Some("second".into()),
            ..ChunkParts::default()
        },
        &mut out,
    );
    let ids: Vec<BlockId> = out
        .iter()
        .filter_map(|item| match item {
            Ok(RawStreamingChoice::ReasoningDelta { id, .. })
            | Ok(RawStreamingChoice::ReasoningEnd { id, .. }) => Some(id.clone()),
            _ => None,
        })
        .collect();
    let first = BlockId::minted(MintKind::Reasoning, 0);
    let second = BlockId::minted(MintKind::Reasoning, 1);
    assert_eq!(
        ids,
        vec![first.clone(), first.clone(), first, second],
        "delta, synthesized end, late signature end, then a fresh block"
    );
}
