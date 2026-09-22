//! Maps wire indices to stable assembly identities for streamed tool calls.
//! Calls without provider IDs receive distinct minted keys, without claiming
//! provider provenance. Argument assembly remains in the shared accumulator.
//!
//! ```
//! use rig_core::providers::internal::tool_call_bridge::ToolCallBridge;
//! let mut bridge = ToolCallBridge::<usize>::new();
//! let slot = bridge.open(0, None, Some("search"));
//! assert!(slot.key().is_minted());
//! ```

use std::collections::HashMap;
use std::hash::Hash;

use crate::streaming::{
    BlockId, StreamEvent, SyntheticIds, ToolCallDecoration, ToolCallEnd, UnparseableToolInput,
};

/// Assembly identity and provider metadata for one streaming tool call.
#[derive(Debug, Clone)]
pub struct ToolCallSlot {
    /// Assembly key fixed at opening: the initial provider ID or a unique mint.
    key: BlockId,
    /// Established provider id: updated when a later chunk carries one.
    /// Empty until the wire supplies one.
    pub id: String,
    /// Established tool name: the last non-empty value seen.
    pub name: String,
    /// Provider-specific decoration carried onto the call's end event.
    pub signature: Option<String>,
    /// Provider-specific decoration carried onto the call's end event.
    pub additional_params: Option<serde_json::Value>,
    /// Whether raw argument fragments have streamed. Re-emit an unparseable
    /// restatement only when false to avoid duplicating buffered bytes.
    pub saw_arguments_delta: bool,
    /// Whether any argument fragment carried a non-whitespace byte. An empty
    /// argument slot under an output-length finish reason is an incomplete
    /// call, not evidence that the model deliberately invoked a zero-argument
    /// tool.
    saw_non_whitespace_arguments_delta: bool,
    /// Opening arguments used only if no argument delta arrives.
    /// Never concatenated with streamed fragments.
    pub announce_arguments: Option<serde_json::Value>,
}

impl ToolCallSlot {
    /// The assembly id this call's fragments are emitted under.
    pub fn key(&self) -> &BlockId {
        &self.key
    }

    /// Record a raw argument fragment before it is forwarded to the shared
    /// accumulator.
    pub fn observe_arguments_delta(&mut self, arguments: &str) {
        self.saw_arguments_delta = true;
        self.saw_non_whitespace_arguments_delta |= !arguments.trim().is_empty();
    }

    /// Whether the wire supplied enough argument bytes to distinguish this
    /// slot from a call cut off before its first argument token.
    pub fn has_substantive_arguments(&self) -> bool {
        self.saw_non_whitespace_arguments_delta || self.announce_arguments.is_some()
    }

    /// The end payload that closes this call's assembly in the shared
    /// accumulator, carrying the established provider id and any decoration.
    pub fn end(&self, on_unparseable: UnparseableToolInput) -> ToolCallEnd {
        let mut end = ToolCallEnd::new(on_unparseable);
        // Only an established provider id overrides the assembly key; a
        // call whose wire never supplied one carries no durable handle at
        // all (`non_empty_id` rejects the empty string by construction).
        end.tool_id = crate::streaming::non_empty_id(self.id.clone());
        end.signature.clone_from(&self.signature);
        end.additional_params.clone_from(&self.additional_params);
        if !self.saw_arguments_delta {
            end.arguments.clone_from(&self.announce_arguments);
        }
        end
    }

    /// The end event that closes this call's assembly, keyed by the slot's
    /// assembly id.
    pub fn end_event(&self, on_unparseable: UnparseableToolInput) -> StreamEvent {
        StreamEvent::BlockEnd {
            id: self.key.clone(),
            end: crate::streaming::BlockClose::ToolCall(self.end(on_unparseable)),
            block: None,
        }
    }
}

/// Per-stream map from wire indices to tool-call assembly identities.
/// `I` must support equality, hashing, copying, and ordering for sorted drains.
#[derive(Debug)]
pub struct ToolCallBridge<I> {
    slots: HashMap<I, ToolCallSlot>,
    /// Minter for slot identities on id-less wires. Defaults to the tool
    /// kind (chat-compat, bedrock); the Responses adapter uses the output
    /// kind so its tool mints share its reasoning mints' id space on the
    /// same wire.
    minted: SyntheticIds,
}

impl<I> Default for ToolCallBridge<I>
where
    I: Eq + Hash + Ord + Copy,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<I> ToolCallBridge<I>
where
    I: Eq + Hash + Ord + Copy,
{
    pub fn new() -> Self {
        Self {
            slots: HashMap::new(),
            minted: SyntheticIds::tool(),
        }
    }

    /// A bridge minting slot identities in the given reserved namespace.
    pub fn with_minted_namespace(minted: SyntheticIds) -> Self {
        Self {
            slots: HashMap::new(),
            minted,
        }
    }

    /// Open or update a slot, fixing its assembly key on first insertion.
    /// An absent or empty initial wire ID receives a minted key. Later nonempty
    /// IDs and names update metadata without changing that key.
    pub fn open(
        &mut self,
        index: I,
        wire_id: Option<&str>,
        name: Option<&str>,
    ) -> &mut ToolCallSlot {
        let minted = &mut self.minted;
        let slot = self.slots.entry(index).or_insert_with(|| ToolCallSlot {
            key: match wire_id {
                Some(id) if !id.is_empty() => BlockId::wire(id),
                // Minted keys cannot collide with or claim provider-issued identity.
                _ => minted.mint(),
            },
            id: String::new(),
            name: String::new(),
            signature: None,
            additional_params: None,
            saw_arguments_delta: false,
            saw_non_whitespace_arguments_delta: false,
            announce_arguments: None,
        });

        if let Some(id) = wire_id
            && !id.is_empty()
        {
            id.clone_into(&mut slot.id);
        }

        if let Some(name) = name
            && !name.is_empty()
        {
            name.clone_into(&mut slot.name);
        }

        slot
    }

    /// The open slot at a wire index, if any.
    pub fn get(&self, index: I) -> Option<&ToolCallSlot> {
        self.slots.get(&index)
    }

    /// Mutably borrow an existing slot without inserting one.
    pub fn get_mut(&mut self, index: I) -> Option<&mut ToolCallSlot> {
        self.slots.get_mut(&index)
    }

    /// The bridge's identity minter, for adapters that also mint
    /// *whole-call* identities: assemblies and whole calls must draw from
    /// ONE counter so their minted keys stay disjoint.
    pub fn minted_ids(&mut self) -> &mut SyntheticIds {
        &mut self.minted
    }

    /// Close and take the slot at a wire index, if any.
    pub fn remove(&mut self, index: I) -> Option<ToolCallSlot> {
        self.slots.remove(&index)
    }

    /// Remove and return an existing slot if `should_evict` accepts it.
    /// The caller must flush the returned slot before reusing its index.
    pub fn evict_if(
        &mut self,
        index: I,
        should_evict: impl FnOnce(&ToolCallSlot) -> bool,
    ) -> Option<ToolCallSlot> {
        if self.slots.get(&index).is_some_and(should_evict) {
            return self.slots.remove(&index);
        }
        None
    }

    /// Fill unset decoration fields on a slot matching a nonempty provider ID.
    /// Existing fields win. Callers must keep provider IDs unique among open slots.
    pub fn decorate(&mut self, decoration: ToolCallDecoration) {
        if decoration.tool_id.is_empty() {
            return;
        }
        if let Some(slot) = self
            .slots
            .values_mut()
            .find(|slot| slot.id == decoration.tool_id)
        {
            if slot.signature.is_none() {
                slot.signature = decoration.signature;
            }
            if slot.additional_params.is_none() {
                slot.additional_params = decoration.additional_params;
            }
        }
    }

    /// Whether no calls remain open.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Drain every open slot in wire-index order, so a multi-call turn keeps
    /// its wire ordering when flushed. The caller chooses the unparseable
    /// policy per flush site when building end events.
    pub fn drain_ordered(&mut self) -> Vec<ToolCallSlot> {
        self.drain_ordered_indexed()
            .into_iter()
            .map(|(_, slot)| slot)
            .collect()
    }

    /// Drain all slots in ascending wire-index order, retaining their indices.
    pub fn drain_ordered_indexed(&mut self) -> Vec<(I, ToolCallSlot)> {
        let mut slots: Vec<(I, ToolCallSlot)> = self.slots.drain().collect();
        slots.sort_by_key(|(index, _)| *index);
        slots
    }
}

#[cfg(test)]
mod tests;
