//! The one stream vocabulary: what a provider adapter emits and what a
//! consumer receives are the same [`StreamEvent`].
//!
//! A stream is a sequence of **blocks** — text, reasoning, tool calls, and
//! the assistant message itself — each identified by a [`BlockId`] for the
//! life of the stream. Every block runs `BlockStart → BlockDelta* →
//! BlockEnd`; a start is optional (a delta for an unseen id opens its block
//! leniently) and an end may carry the wire's authoritative payload for the
//! block (a restated reasoning block, a completed tool call's fields).
//!
//! The accumulator ([`BlockAccumulator`](super::BlockAccumulator)) folds the
//! same events into the aggregated assistant choice, and fills
//! [`BlockEnd::block`] on the events it yields to consumers with the block it
//! just finalized — so a consumer that wants the completed tool call or
//! reasoning item reads it off the end event, and one that only wants the
//! deltas ignores it. Adapters always emit `block: None`.
//!
//! Everything here is data: serde, `Clone + Send + Sync + 'static`, no
//! lifetimes, no `dyn`. The bus (phase C) sends these over a channel and the
//! effect log records them; a host can persist an in-flight stream event.

use serde::{Deserialize, Serialize};

use crate::message::{AdditionalParams, AssistantContent, Reasoning};

use super::{BlockId, StreamFinal, UnknownPayload, UnparseableToolInput};

/// One event of a completion stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[allow(
    clippy::large_enum_variant,
    reason = "the terminal record is one event per stream and is moved, not copied; boxing it would put an allocation on every consumer's terminal match"
)]
pub enum StreamEvent {
    /// A block opened.
    BlockStart {
        /// The block's identity for the life of the stream.
        id: BlockId,
        /// What kind of block, with the metadata the wire announced at the
        /// boundary.
        kind: BlockKind,
    },
    /// A block grew.
    BlockDelta {
        /// The block this fragment extends.
        id: BlockId,
        /// The fragment.
        delta: Delta,
    },
    /// A block closed.
    BlockEnd {
        /// The block that closed.
        id: BlockId,
        /// What the wire said at the boundary.
        end: BlockClose,
        /// The block as finalized by the accumulator, when the end
        /// finalized one that consumers need whole (a completed tool call, a
        /// completed reasoning item). `None` from an adapter; `None` from
        /// the accumulator when the end finalized nothing (a dropped call, a
        /// silent synthesized boundary) or when the block is text (its
        /// deltas are the content).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block: Option<AssistantContent>,
    },
    /// The provider's normalized terminal record. At most one per stream,
    /// last among the content events.
    Final(StreamFinal),
    /// A provider-native item rig does not model — e.g. an OpenAI Responses
    /// hosted-tool result. Passed through verbatim; never folded into the
    /// aggregated choice.
    Unknown(UnknownPayload),
}

/// What kind of block a [`StreamEvent::BlockStart`] opened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BlockKind {
    /// The assistant message itself: the wire announced its provider
    /// message id (`id` is `BlockId::wire(message_id)`). Captured into
    /// `StreamingCompletionResponse::message_id`; outranks the terminal
    /// record's id.
    Message,
    /// A text block, with the provider metadata attached at its start.
    Text {
        /// Provider-specific metadata for the block.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        additional_params: Option<AdditionalParams>,
    },
    /// A reasoning block, with the provider-issued durable id when the wire
    /// has one — the value that becomes [`Reasoning::id`] and round-trips
    /// upstream. A minted block id never does.
    Reasoning {
        /// The provider-issued reasoning item id.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_id: Option<String>,
    },
    /// A tool call under assembly.
    ToolCall,
}

/// A fragment of a block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "delta", rename_all = "snake_case")]
pub enum Delta {
    /// Text.
    Text {
        /// The text fragment.
        text: String,
    },
    /// Provider metadata merged into the text block.
    TextMeta {
        /// The metadata.
        additional_params: AdditionalParams,
    },
    /// Reasoning text.
    Reasoning {
        /// The reasoning fragment.
        text: String,
    },
    /// The tool name (OpenAI-compatible wires stream it as a fragment; the
    /// last non-empty value is the established name).
    ToolName {
        /// The name fragment.
        name: String,
    },
    /// A raw JSON argument fragment.
    ToolArguments {
        /// The fragment; concatenated in arrival order.
        arguments: String,
    },
}

/// What the wire said when a block closed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "close", rename_all = "snake_case")]
pub enum BlockClose {
    /// A text block closed; later text under a fresh id opens a new block.
    Text,
    /// A reasoning block closed.
    Reasoning {
        /// The wire's authoritative whole-block restatement, when it sent
        /// one; it supersedes the delta accumulation.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<Reasoning>,
        /// A provider signature closing the block; attaches to the block's
        /// text.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        /// Whether the wire itself sent this end (anthropic's
        /// `content_block_stop`), as opposed to the adapter synthesizing it
        /// at a boundary the wire never announces. Wire-sent ends yield the
        /// completed block even when bare; a synthesized bare end stays
        /// silent (`block: None`).
        wire_sent: bool,
    },
    /// A tool call's input ended: the accumulator finalizes the assembled
    /// fragments, or the end's authoritative payload, into a completed
    /// call.
    ToolCall(ToolCallEnd),
}

/// The end of a streamed tool call's input.
///
/// Optional fields are authoritative wire values that supersede the
/// assembled state — a wire whose completed item restates the call (OpenAI
/// Responses `output_item.done`, a whole-call wire) carries them; delta-only
/// wires leave them `None` and the assembled fragments are parsed instead.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallEnd {
    /// Authoritative provider-issued tool id, when one exists (e.g. an id
    /// that arrived after the call opened id-less). The durable handle;
    /// absence is `None`, never an empty string (see
    /// [`non_empty_id`](super::non_empty_id)).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_id: Option<String>,
    /// Authoritative tool name from the wire's completed item.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Authoritative parsed arguments from the wire's completed item.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
    /// Provider call-correlation id (e.g. OpenAI Responses `call_id`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    /// Provider signature attached to the completed call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Provider-specific metadata attached to the completed call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
    /// Wire-family policy for assembled arguments that fail to parse.
    pub on_unparseable: UnparseableToolInput,
}

impl ToolCallEnd {
    /// End a call, finalizing from assembled fragments with the given
    /// unparseable-input policy.
    pub fn new(on_unparseable: UnparseableToolInput) -> Self {
        Self {
            tool_id: None,
            name: None,
            arguments: None,
            call_id: None,
            signature: None,
            additional_params: None,
            on_unparseable,
        }
    }

    /// A whole call delivered at once: name and arguments are authoritative
    /// and malformed input is a response defect.
    pub fn whole(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            name: Some(name.into()),
            arguments: Some(arguments),
            ..Self::new(UnparseableToolInput::Error)
        }
    }

    /// Attach the authoritative provider tool id (empty means absent).
    pub fn with_tool_id(mut self, tool_id: impl Into<String>) -> Self {
        self.tool_id = super::non_empty_id(tool_id);
        self
    }

    /// Attach the provider call-correlation id.
    pub fn with_call_id(mut self, call_id: impl Into<String>) -> Self {
        self.call_id = Some(call_id.into());
        self
    }

    /// Attach or clear a provider signature.
    pub fn with_signature(mut self, signature: Option<String>) -> Self {
        self.signature = signature;
        self
    }

    /// Attach provider-specific metadata.
    pub fn with_additional_params(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }
}

impl StreamEvent {
    /// A text fragment for block `id`.
    pub fn text(id: BlockId, text: impl Into<String>) -> Self {
        Self::BlockDelta {
            id,
            delta: Delta::Text { text: text.into() },
        }
    }

    /// The block this event is about, if it is a block event.
    pub fn block_id(&self) -> Option<&BlockId> {
        match self {
            Self::BlockStart { id, .. }
            | Self::BlockDelta { id, .. }
            | Self::BlockEnd { id, .. } => Some(id),
            Self::Final(_) | Self::Unknown(_) => None,
        }
    }

    /// Stable variant name for logs and law-violation messages — never the
    /// payload (events carry wire content that must not reach logs).
    pub const fn name(&self) -> &'static str {
        match self {
            Self::BlockStart { .. } => "BlockStart",
            Self::BlockDelta { .. } => "BlockDelta",
            Self::BlockEnd { .. } => "BlockEnd",
            Self::Final(_) => "Final",
            Self::Unknown(_) => "Unknown",
        }
    }
}

// The stream vocabulary crosses threads and serializes on every target: the
// bus sends it over a channel and the effect log records it.
const _: fn() = || {
    fn assert_wire<T: Clone + Send + Sync + 'static + Serialize + serde::de::DeserializeOwned>() {}
    assert_wire::<StreamEvent>();
    assert_wire::<BlockKind>();
    assert_wire::<Delta>();
    assert_wire::<BlockClose>();
    assert_wire::<ToolCallEnd>();
    assert_wire::<StreamFinal>();
    assert_wire::<UnknownPayload>();
};

#[cfg(test)]
mod tests;
