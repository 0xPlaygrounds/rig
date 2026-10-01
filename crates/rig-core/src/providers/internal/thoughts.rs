//! Reasoning on a wire that marks no boundaries (OpenAI-compatible
//! `reasoning_content`, Gemini thought parts): fragments extend one part
//! until text or a tool call interleaves, and the wire's signature closes
//! it. This module is not a stable public API.
//!
//! ```
//! use rig_core::providers::internal::thoughts::Thoughts;
//!
//! let thoughts = Thoughts::new();
//! assert!(!thoughts.is_open());
//! ```

use crate::message::{Reasoning, ReasoningContent};
use crate::operation::{Completion, ReasoningPart, Seal};
use crate::wire::Out;

/// The reasoning a boundary-less wire is streaming.
pub struct Thoughts<'id> {
    /// The latest part. Output interleaving it stops its fragments, but a
    /// late signature still closes it.
    part: Option<ReasoningPart<'id>>,
    open: bool,
    /// Whether the last part closed with the wire's own signature, so a
    /// further signature is a part of its own.
    signed: bool,
}

impl Default for Thoughts<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'id> Thoughts<'id> {
    /// No reasoning yet.
    pub fn new() -> Self {
        Self {
            part: None,
            open: false,
            signed: false,
        }
    }

    /// Whether fragments still extend the current part.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// A reasoning fragment: extends the open part, or starts a new one
    /// after output interleaved the last.
    pub fn fragment(&mut self, out: &mut Out<'id, Completion>, text: &str) {
        if text.is_empty() {
            return;
        }
        if !self.open {
            self.close(out, None);
            self.part = Some(out.reasoning());
            self.open = true;
            self.signed = false;
        }
        if let Some(part) = &self.part {
            out.push_reasoning(part, text);
        }
    }

    /// The wire's signature for the reasoning so far: it closes the part it
    /// signs, or stands as a part of its own when nothing unsigned is left.
    pub fn signature(&mut self, out: &mut Out<'id, Completion>, signature: String) {
        if self.part.is_some() {
            self.close(out, Some(signature));
            return;
        }
        self.signed = true;
        out.reasoning_block(Reasoning {
            id: None,
            content: vec![ReasoningContent::Text {
                text: String::new(),
                signature: Some(signature),
            }],
            native: None,
        });
    }

    /// Output interleaved: later fragments are a new part.
    pub fn boundary(&mut self) {
        self.open = false;
    }

    /// Close the current part, signed with `signature` when the wire sent
    /// one.
    pub fn close(&mut self, out: &mut Out<'id, Completion>, signature: Option<String>) {
        if let Some(part) = self.part.take() {
            self.signed = signature.is_some();
            out.close_reasoning(
                part,
                Seal {
                    signature,
                    ..Seal::default()
                },
            );
        }
        self.open = false;
    }

    /// Whether the last part closed with the wire's own signature.
    pub fn signed(&self) -> bool {
        self.signed
    }
}
