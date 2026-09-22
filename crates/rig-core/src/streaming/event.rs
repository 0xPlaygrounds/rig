//! Serializable completion-stream events shared by adapters and consumers.
//! Blocks share a [`BlockId`] across starts, deltas, and ends. A delta can
//! implicitly open a block; an end can supply an authoritative payload.
//!
//! Adapters emit `block: None` on end events. The
//! [`BlockAccumulator`](super::BlockAccumulator) fills it with finalized tool
//! calls or reasoning while assembling the assistant response.
//!
//! ```
//! use rig_core::streaming::{BlockId, MintKind, StreamEvent};
//!
//! let id = BlockId::minted(MintKind::Text, 0);
//! let event = StreamEvent::text(id.clone(), "Hello");
//! assert_eq!(event.block_id(), Some(&id));
//! ```

use serde::{Deserialize, Serialize};

use crate::message::{AdditionalParams, AssistantContent, Reasoning};

use super::{BlockId, StreamFinal, UnknownPayload, UnparseableToolInput};

/// One event of a completion stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
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
    /// Unmodeled provider data, passed through without joining the aggregated choice.
    Unknown(UnknownPayload),
}

/// What kind of block a [`StreamEvent::BlockStart`] opened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BlockKind {
    /// Assistant message boundary with a provider-issued block ID.
    /// Its ID takes precedence over the terminal record's message ID.
    Message,
    /// A text block, with the provider metadata attached at its start.
    Text {
        /// Provider-specific metadata for the block.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        additional_params: Option<AdditionalParams>,
    },
    /// A reasoning block with an optional provider ID for [`Reasoning::id`].
    /// Minted block keys must not become replayed provider IDs.
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
    /// Tool name update. The last nonempty value becomes the established name.
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
        /// Whether the provider explicitly ended the block. Explicit ends yield
        /// a completed block even when bare; synthesized bare ends yield `None`.
        wire_sent: bool,
    },
    /// A tool call's input ended: the accumulator finalizes the assembled
    /// fragments, or the end's authoritative payload, into a completed
    /// call.
    ToolCall(ToolCallEnd),
}

/// The end of a streamed tool call's input.
///
/// Authoritative names and arguments supersede assembled fragments when present.
/// Absent arguments are parsed from the accumulated deltas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallEnd {
    /// An already assigned local correlation handle, when re-emitting a
    /// completed response. This does not supply provider provenance; provider
    /// handles remain in `tool_id` and `call_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub durable_id: Option<crate::message::ToolCallId>,
    /// Provider-issued tool ID, including IDs received after the block opened.
    /// Represent absence as `None`, not an empty string.
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
            durable_id: None,
            tool_id: None,
            name: None,
            arguments: None,
            call_id: None,
            signature: None,
            additional_params: None,
            on_unparseable,
        }
    }

    /// Creates a completed call with authoritative name and parsed arguments.
    pub fn whole(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            name: Some(name.into()),
            arguments: Some(arguments),
            ..Self::new(UnparseableToolInput::Error)
        }
    }

    /// Preserve an existing local correlation handle through stream folding.
    pub fn with_durable_id(mut self, id: crate::message::ToolCallId) -> Self {
        self.durable_id = Some(id);
        self
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

    /// Returns a stable variant name without exposing wire payloads to logs.
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
