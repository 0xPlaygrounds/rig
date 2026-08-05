//! Index → grammar-identity bridging for streamed tool calls.
//!
//! Several wires key tool-call fragments by a numeric index (the chat-compat
//! chunk index, Bedrock's `contentBlockIndex`) rather than by the grammar id
//! the shared accumulator assembles under. Every adapter on such a wire needs
//! the same bridge: a per-stream map from the wire's index to the identity the
//! adapter established for that call. [`ToolCallBridge`] is that map, shared
//! so the non-empty-id invariant on
//! [`RawStreamingChoice::ToolCallDelta`](crate::streaming::RawStreamingChoice)
//! is enforced in exactly one place: when the wire supplies no id, the slot's
//! grammar id is minted from the wire's own index in the reserved
//! [`SyntheticIds`] `tool-` namespace, so parallel id-less calls can never
//! share an assembly key downstream and a minted id can never serialize
//! upstream as a wire-genuine one.
//!
//! Only the *bridging state* lives here. Argument assembly, internal-id
//! minting for finalized calls, and finalize policy stay in the shared
//! accumulator (`PartsAccumulator`); frame triage stays in the driver.

use std::collections::HashMap;
use std::hash::Hash;

use crate::providers::internal::adapter::SyntheticIds;
use crate::streaming::{ToolCallDecoration, ToolInputEnd, UnparseableToolInput};

/// Wire identity of a tool call whose input is streaming, as tracked by an
/// adapter. The slot keeps only what the wire keys by — an index — mapped to
/// the identity the adapter established for that call.
#[derive(Debug, Clone)]
pub struct ToolCallSlot {
    /// Assembly id: the id under which this call's fragments are emitted.
    /// Fixed at open: the first-seen provider id, or a `tool-{index}` id
    /// minted from the wire index when the wire omits one — never empty, so
    /// parallel id-less calls can never share an assembly key downstream.
    key: String,
    /// Established provider id: updated when a later chunk carries one.
    /// Empty until the wire supplies one.
    pub id: String,
    /// Established tool name: the last non-empty value seen.
    pub name: String,
    /// Provider-specific decoration carried onto the call's end event.
    pub signature: Option<String>,
    /// Provider-specific decoration carried onto the call's end event.
    pub additional_params: Option<serde_json::Value>,
}

impl ToolCallSlot {
    /// The assembly id this call's fragments are emitted under. Never empty.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Build the end event that closes this call's assembly in the shared
    /// accumulator, carrying the established provider id and any decoration.
    pub fn end_event(&self, on_unparseable: UnparseableToolInput) -> ToolInputEnd {
        let mut end = ToolInputEnd::new(self.key.clone(), on_unparseable);
        // Only an established provider id overrides the assembly key; a call
        // whose wire never supplied one keeps its minted `tool-{index}` id.
        if !self.id.is_empty() {
            end.tool_id = Some(self.id.clone());
        }
        end.signature = self.signature.clone();
        end.additional_params = self.additional_params.clone();
        end
    }
}

/// Per-stream index → grammar-identity map for streamed tool calls.
///
/// `I` is the wire's own index type (`usize` for chat-compat chunk indices,
/// `i32` for Bedrock content-block indices); it must display so a minted id
/// can derive from it, and order so a drain preserves wire ordering.
#[derive(Debug, Default)]
pub struct ToolCallBridge<I> {
    slots: HashMap<I, ToolCallSlot>,
}

impl<I> ToolCallBridge<I>
where
    I: Eq + Hash + Ord + Copy + std::fmt::Display,
{
    pub fn new() -> Self {
        Self {
            slots: HashMap::new(),
        }
    }

    /// Open (or update) the slot for a wire index, establishing its identity.
    ///
    /// On first sight the assembly key is fixed: the wire id when one is
    /// supplied, else an id minted from the index in the reserved `tool-`
    /// namespace — the single enforcement point of the non-empty grammar-id
    /// invariant. Later fragments update the established provider id and name
    /// from any non-empty values they carry.
    pub fn open(&mut self, index: I, wire_id: Option<&str>, name: Option<&str>) -> &ToolCallSlot {
        let slot = self.slots.entry(index).or_insert_with(|| ToolCallSlot {
            key: match wire_id {
                Some(id) if !id.is_empty() => id.to_owned(),
                // Id-less wires (several llama.cpp/vllm-style gateways) key
                // tool calls by index alone; the grammar id is minted from
                // that index in the reserved namespace so it is never empty
                // and never collides with a wire-genuine id.
                _ => SyntheticIds::tool().for_index(index),
            },
            id: String::new(),
            name: String::new(),
            signature: None,
            additional_params: None,
        });

        if let Some(id) = wire_id
            && !id.is_empty()
        {
            slot.id = id.to_owned();
        }

        if let Some(name) = name
            && !name.is_empty()
        {
            slot.name = name.to_owned();
        }

        slot
    }

    /// The open slot at a wire index, if any.
    pub fn get(&self, index: I) -> Option<&ToolCallSlot> {
        self.slots.get(&index)
    }

    /// Close and take the slot at a wire index, if any.
    pub fn remove(&mut self, index: I) -> Option<ToolCallSlot> {
        self.slots.remove(&index)
    }

    /// Evict the slot at a wire index when the predicate says the incoming
    /// fragment belongs to a *different* call reusing the same index (the
    /// per-profile eviction semantics — e.g. a distinct id + name pair on a
    /// wire that restarts indices per call). Returns the evicted slot so the
    /// caller can flush it to the consumer.
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

    /// Apply a provider decoration to the in-flight call it names, matched by
    /// the established provider id. Decorations ride the slot onto its end
    /// event; assembly itself is untouched.
    pub fn decorate(&mut self, decoration: ToolCallDecoration) {
        if let Some(slot) = self
            .slots
            .values_mut()
            .find(|slot| slot.id == decoration.tool_id)
        {
            slot.signature = decoration.signature;
            slot.additional_params = decoration.additional_params;
        }
    }

    /// Whether any call is still open.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Drain every open slot in wire-index order, so a multi-call turn keeps
    /// its wire ordering when flushed. The caller chooses the unparseable
    /// policy per flush site when building end events.
    pub fn drain_ordered(&mut self) -> Vec<ToolCallSlot> {
        let mut slots: Vec<(I, ToolCallSlot)> = self.slots.drain().collect();
        slots.sort_by_key(|(index, _)| *index);
        slots.into_iter().map(|(_, slot)| slot).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_id_becomes_the_assembly_key() {
        let mut bridge = ToolCallBridge::<usize>::new();
        let slot = bridge.open(0, Some("call_abc"), Some("get_weather"));
        assert_eq!(slot.key(), "call_abc");
        assert_eq!(slot.id, "call_abc");
        assert_eq!(slot.name, "get_weather");

        // The established id rides the end event as the override.
        let end = slot.end_event(UnparseableToolInput::Drop);
        assert_eq!(end.id, "call_abc");
        assert_eq!(end.tool_id.as_deref(), Some("call_abc"));
    }

    #[test]
    fn id_less_open_mints_a_distinct_provenance_gated_key_per_index() {
        let mut bridge = ToolCallBridge::<usize>::new();
        let first_key = bridge.open(0, None, Some("get_weather")).key().to_owned();
        let second_key = bridge.open(1, None, Some("get_time")).key().to_owned();

        // Parallel id-less calls must never share an assembly key, and a
        // minted key must be recognized by the boundary-minted provenance
        // gate so it can never serialize upstream as wire-genuine.
        assert_ne!(first_key, second_key);
        assert_eq!(first_key, "tool-0");
        assert_eq!(second_key, "tool-1");
        assert!(crate::streaming::is_boundary_minted_id(&first_key));

        // A call whose wire never supplied an id keeps its minted key with
        // no provider-id override.
        let slot = bridge.remove(0).expect("slot must be open");
        let end = slot.end_event(UnparseableToolInput::Drop);
        assert_eq!(end.id, "tool-0");
        assert!(end.tool_id.is_none());
    }

    #[test]
    fn late_wire_id_updates_the_override_but_not_the_key() {
        let mut bridge = ToolCallBridge::<usize>::new();
        bridge.open(0, None, Some("get_weather"));
        let slot = bridge.open(0, Some("call_late"), None);
        // The assembly key is fixed at open; the late provider id becomes
        // the end-event override the accumulator surfaces to the consumer.
        assert_eq!(slot.key(), "tool-0");
        assert_eq!(slot.id, "call_late");
        assert_eq!(slot.name, "get_weather");
    }

    #[test]
    fn evict_if_takes_the_slot_only_when_the_predicate_says_so() {
        let mut bridge = ToolCallBridge::<usize>::new();
        bridge.open(0, Some("call_a"), Some("get_weather"));

        assert!(bridge.evict_if(0, |slot| slot.id == "call_b").is_none());
        assert!(bridge.get(0).is_some(), "a refused eviction keeps the slot");

        let evicted = bridge
            .evict_if(0, |slot| slot.id == "call_a")
            .expect("predicate matched: slot must be evicted");
        assert_eq!(evicted.key(), "call_a");
        assert!(bridge.get(0).is_none());
    }

    #[test]
    fn decoration_matches_by_established_provider_id_and_rides_the_end_event() {
        let mut bridge = ToolCallBridge::<usize>::new();
        bridge.open(0, Some("call_a"), Some("get_weather"));
        bridge.open(1, Some("call_b"), Some("get_time"));

        bridge.decorate(ToolCallDecoration {
            tool_id: "call_b".to_owned(),
            signature: Some("sig-b".to_owned()),
            additional_params: Some(serde_json::json!({"k": "v"})),
        });

        let undecorated = bridge.remove(0).expect("slot 0 open");
        let end = undecorated.end_event(UnparseableToolInput::Drop);
        assert!(end.signature.is_none());

        let decorated = bridge.remove(1).expect("slot 1 open");
        let end = decorated.end_event(UnparseableToolInput::Drop);
        assert_eq!(end.signature.as_deref(), Some("sig-b"));
        assert_eq!(end.additional_params, Some(serde_json::json!({"k": "v"})));
    }

    #[test]
    fn drain_ordered_preserves_wire_index_order() {
        let mut bridge = ToolCallBridge::<i32>::new();
        bridge.open(2, Some("call_c"), None);
        bridge.open(0, Some("call_a"), None);
        bridge.open(1, Some("call_b"), None);

        let keys: Vec<String> = bridge
            .drain_ordered()
            .into_iter()
            .map(|slot| slot.key().to_owned())
            .collect();
        assert_eq!(keys, vec!["call_a", "call_b", "call_c"]);
        assert!(bridge.is_empty());
    }
}
