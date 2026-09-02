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

use crate::streaming::{BlockId, MintKind, RawStreamingChoice, SyntheticIds};

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

/// The lifecycle state for one stream's minted-key reasoning blocks.
///
/// Owns the open/close bookkeeping the adapters used to hand-roll; an
/// adapter never touches a `reasoning_open` flag or emits a lifecycle event
/// directly. Every block gets its own minted key: the key is the block's
/// public identity for the life of the stream, so a wire that reasons,
/// interleaves other content, then reasons again yields two blocks with two
/// distinct ids. A trailing close (a late `thoughtSignature` after a
/// synthesized boundary) still addresses the block that streamed, because
/// the next key is minted only when reasoning resumes.
pub struct MintedReasoningLifecycle {
    ids: SyntheticIds,
    key: BlockId,
    open: bool,
    /// Whether `key`'s block has been closed, so the next reasoning delta
    /// opens a new block under a fresh key.
    closed: bool,
}

impl MintedReasoningLifecycle {
    /// A lifecycle minting per-block keys of `kind`.
    pub fn new(kind: MintKind) -> Self {
        let mut ids = SyntheticIds::new(kind);
        let key = ids.mint();
        Self {
            ids,
            key,
            open: false,
            closed: false,
        }
    }

    /// The key of the block currently (or most recently) streaming.
    pub fn key(&self) -> &BlockId {
        &self.key
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
            if self.closed {
                self.key = self.ids.mint();
                self.closed = false;
            }
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
            self.closed = true;
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
            self.closed = true;
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
mod tests;
