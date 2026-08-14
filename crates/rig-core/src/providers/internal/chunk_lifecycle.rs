//! Shared lifecycle derivation for boundary-less constant-key wires.
//!
//! Wires with no reasoning boundary of their own (ollama's `thinking`,
//! cohere's `thinking` content, gemini REST's `thought` parts, gemini
//! Interactions' thought summaries) used to hand-roll the same algorithm
//! per adapter: track `reasoning_open`, emit
//! the delta under the per-stream constant minted key, and synthesize a
//! silent `ReasoningEnd` before any other content class. Every review round
//! found one adapter that missed a piece of it ("close the open reasoning
//! block before any other part class" took two rounds across six adapters;
//! "emit a chunk's parts in canonical order" took another).
//!
//! Here the adapter *declares* what one wire chunk carried — a
//! [`ChunkParts`] — and [`MintedReasoningLifecycle::emit_chunk`] derives the
//! canonical event sequence: reasoning first, a wire-signed close when the
//! chunk carried one, the synthesized boundary end when other content
//! interleaves, then text, then tool events. "Forgot the boundary" and
//! "wrong intra-chunk order" are not expressible through this interface
//! (langchain's declarative-chunk + core-side merge factoring;
//! semantic-kernel converged on the same shape independently). The driver's
//! debug-mode sequence laws (`sequence_law`) still watch the emitted stream,
//! so an adapter bypassing this helper fails its own tests.
//!
//! `pub` (not `pub(crate)`) for the same reason as [`adapter`](super::adapter)
//! and [`tool_call_bridge`](super::tool_call_bridge): companion provider
//! crates implementing [`WireAdapter`](super::adapter::WireAdapter) over a
//! boundary-less wire (rig-gemini-grpc) must inherit this derivation rather
//! than hand-roll it; it is not part of rig-core's stable public API.
//!
//! Wires that announce their own boundaries (anthropic `content_block_stop`,
//! OpenAI Responses `output_item.done`) do not use this — their lifecycle is
//! the wire's, not a derivation. The chat-completions compat family keeps its
//! `CompatibleStreamProfile` system (in the crate-private
//! `openai_chat_completions_compatible` module, hence named rather than
//! linked): that IS the shared derivation for its ~15 gateway providers, with
//! wire quirks (slot eviction, encrypted reasoning details, tool-call
//! decorations) this declarative shape does not model.

use crate::streaming::{RawStreamingChoice, StreamPartId};

use super::adapter::AdapterOutput;

/// What one wire chunk (or one wire part, for parts-array wires) carried,
/// declared by the adapter with no lifecycle events of its own.
#[derive(Default)]
pub struct ChunkParts<R> {
    /// Reasoning content accumulating under the wire's constant minted key.
    pub reasoning: Option<String>,
    /// A wire-carried signature closing the reasoning block (gemini's
    /// `thoughtSignature`) — the one authoritative close these wires spell.
    pub reasoning_signature: Option<String>,
    /// Visible text content.
    pub text: Option<String>,
    /// Tool-call events in wire order — whole calls, fragments, or input
    /// ends, prebuilt by the adapter (keys and ids are wire policy, not
    /// lifecycle). Emitted after the boundary close, in the canonical slot.
    pub tool_events: Vec<RawStreamingChoice<R>>,
}

impl<R> ChunkParts<R> {
    /// Whether the chunk carries content that interleaves — and therefore
    /// closes — an open reasoning block.
    fn has_boundary_content(&self) -> bool {
        self.text.as_ref().is_some_and(|text| !text.is_empty()) || !self.tool_events.is_empty()
    }
}

/// The lifecycle state for one stream's constant-key reasoning block.
///
/// Owns the open/close bookkeeping the adapters used to hand-roll; an
/// adapter never touches a `reasoning_open` flag or emits a lifecycle event
/// directly.
pub struct MintedReasoningLifecycle {
    key: StreamPartId,
    open: bool,
}

impl MintedReasoningLifecycle {
    /// A lifecycle for the given per-stream constant minted key.
    pub fn new(key: StreamPartId) -> Self {
        Self { key, open: false }
    }

    /// Emit one declared chunk as the canonical event sequence.
    ///
    /// Order and boundary are derived, not stated per adapter:
    /// 1. reasoning delta (opens the block);
    /// 2. a wire-carried signature closes the block authoritatively;
    /// 3. other content in the chunk closes a still-open block with a
    ///    synthesized silent end (`wire_sent: false` — the wire never spelled
    ///    the boundary, so downstream must not observe a fabricated event);
    /// 4. text, then tool events.
    pub fn emit_chunk<R>(&mut self, parts: ChunkParts<R>, out: &mut AdapterOutput<R>) {
        if let Some(reasoning) = parts
            .reasoning
            .as_ref()
            .filter(|reasoning| !reasoning.is_empty())
        {
            self.open = true;
            out.push(Ok(RawStreamingChoice::ReasoningDelta {
                id: self.key.clone(),
                provider_id: None,
                reasoning: reasoning.clone(),
            }));
        }

        if let Some(signature) = parts.reasoning_signature.clone() {
            // The wire's own authoritative close: signs the accumulated
            // deltas, the already-finished block that holds the
            // chain-of-thought, or a signature-only part when nothing
            // streamed — the shared accumulator owns the per-case behavior.
            self.open = false;
            out.push(Ok(RawStreamingChoice::ReasoningEnd {
                id: self.key.clone(),
                reasoning: None,
                signature: Some(signature),
                wire_sent: false,
            }));
        }

        if parts.has_boundary_content() && self.open {
            // Interleaving output ends an open reasoning block — the
            // boundary these wires never announce, synthesized once here
            // instead of once per adapter.
            self.open = false;
            out.push(Ok(RawStreamingChoice::ReasoningEnd {
                id: self.key.clone(),
                reasoning: None,
                signature: None,
                wire_sent: false,
            }));
        }

        if let Some(text) = parts.text.filter(|text| !text.is_empty()) {
            out.push(Ok(RawStreamingChoice::Message(text)));
        }

        for event in parts.tool_events {
            out.push(Ok(event));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::MintKind;

    fn lifecycle() -> MintedReasoningLifecycle {
        MintedReasoningLifecycle::new(StreamPartId::minted(MintKind::Reasoning, 0))
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
            StreamPartId::minted(MintKind::Tool, 0),
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
}
