//! Reasoning lifecycle derivation for wires without explicit block boundaries.
//! Companion decoders can declare chunk contents and receive ordered reasoning,
//! text, and tool events. This module is not a stable public API.
//!
//! ```
//! use rig_core::operation::AdapterOutput;
//! use rig_core::providers::internal::chunk_lifecycle::{ChunkParts, MintedReasoningLifecycle};
//! use rig_core::streaming::MintKind;
//! let mut lifecycle = MintedReasoningLifecycle::new(MintKind::Reasoning);
//! let mut output = AdapterOutput::new();
//! lifecycle.emit_chunk(ChunkParts { reasoning: Some("Thinking".into()), ..Default::default() }, &mut output);
//! ```

use crate::operation::AdapterOutput;
use crate::streaming::{BlockId, MintKind, StreamEvent, SyntheticIds};

/// What one wire chunk (or one wire part, for parts-array wires) carried,
/// declared by the adapter with no lifecycle events of its own.
#[derive(Default)]
pub struct ChunkParts {
    /// Reasoning content accumulating under the wire's constant minted key.
    pub reasoning: Option<String>,
    /// Wire-carried signature closing the reasoning block.
    pub reasoning_signature: Option<String>,
    /// Visible text content.
    pub text: Option<String>,
    /// Provider extras for the text block this chunk's text lands in, such
    /// as a signature the wire put on that text part.
    pub text_meta: Option<crate::message::AdditionalParams>,
    /// Tool-call events in wire order, with adapter-assigned keys and IDs.
    /// Emitted after reasoning closes and text is emitted.
    pub tool_events: Vec<StreamEvent>,
}

impl ChunkParts {
    /// Whether this chunk's text or tool events close an open reasoning block.
    fn has_boundary_content(&self) -> bool {
        self.text.as_ref().is_some_and(|text| !text.is_empty())
            || self.text_meta.is_some()
            || !self.tool_events.is_empty()
    }
}

/// Tracks reasoning blocks under distinct stream-local minted keys.
/// A late signature after a synthesized close addresses the preceding block.
/// Resumed reasoning or a signature after a signed close receives a new key.
pub struct MintedReasoningLifecycle {
    ids: SyntheticIds,
    key: BlockId,
    open: bool,
    /// Whether `key`'s block has been closed, so the next reasoning delta
    /// opens a new block under a fresh key.
    closed: bool,
    /// The last close was the wire's own signature: the block is signed,
    /// and a further signature is a block of its own.
    signed: bool,
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
            signed: false,
        }
    }

    /// The key of the block currently (or most recently) streaming.
    pub fn key(&self) -> &BlockId {
        &self.key
    }

    /// Emit reasoning, its signature or synthesized close, text, then tool events.
    /// Interleaving text or tools closes an open block. Ends use `wire_sent: false`
    /// so downstream observers do not receive fabricated boundary events.
    pub fn emit_chunk(&mut self, parts: ChunkParts, out: &mut AdapterOutput) {
        if let Some(reasoning) = parts
            .reasoning
            .as_ref()
            .filter(|reasoning| !reasoning.is_empty())
        {
            if self.closed {
                self.key = self.ids.mint();
                self.closed = false;
                self.signed = false;
            }
            self.open = true;
            out.reasoning_delta(&self.key, None, reasoning.clone());
        }

        if let Some(signature) = parts.reasoning_signature.clone() {
            // Late signatures reuse an unsigned block; consecutive signed closes
            // need distinct keys to preserve one signed end per block.
            if self.signed && !self.open {
                self.key = self.ids.mint();
            }
            self.open = false;
            self.closed = true;
            self.signed = true;
            out.reasoning_end(self.key.clone(), None, Some(signature), false);
        }

        if parts.has_boundary_content() && self.open {
            // These wires omit the boundary before interleaving output.
            self.open = false;
            self.closed = true;
            out.reasoning_end(self.key.clone(), None, None, false);
        }

        if let Some(text) = parts.text.filter(|text| !text.is_empty()) {
            out.text(text);
        }
        if let Some(meta) = parts.text_meta {
            out.text_meta(meta);
        }

        for event in parts.tool_events {
            out.push(Ok(event));
        }
    }
}

#[cfg(test)]
mod tests;
