//! Index → grammar-identity bridging for streamed tool calls.
//!
//! Several wires key tool-call fragments by a numeric index (the chat-compat
//! chunk index, Bedrock's `contentBlockIndex`) rather than by the grammar id
//! the shared accumulator assembles under. Every adapter on such a wire needs
//! the same bridge: a per-stream map from the wire's index to the identity the
//! adapter established for that call. [`ToolCallBridge`] is that map, shared
//! so the mandatory-identity invariant on
//! [`RawStreamingChoice::ToolCallDelta`](crate::streaming::RawStreamingChoice)
//! is enforced in exactly one place: when the wire supplies no id, the slot's
//! grammar id is a `StreamPartId::Minted` from the bridge's [`SyntheticIds`]
//! counter, so parallel id-less calls can never share an assembly key
//! downstream — and a minted id structurally cannot serialize upstream as a
//! wire-genuine one.
//!
//! Only the *bridging state* lives here. Argument assembly, internal-id
//! minting for finalized calls, and finalize policy stay in the shared
//! accumulator (`PartsAccumulator`); frame triage stays in the driver.

use std::collections::HashMap;
use std::hash::Hash;

use crate::streaming::{
    StreamPartId, SyntheticIds, ToolCallDecoration, ToolInputEnd, UnparseableToolInput,
};

/// Wire identity of a tool call whose input is streaming, as tracked by an
/// adapter. The slot keeps only what the wire keys by — an index — mapped to
/// the identity the adapter established for that call.
#[derive(Debug, Clone)]
pub struct ToolCallSlot {
    /// Assembly id: the id under which this call's fragments are emitted.
    /// Fixed at open: the first-seen provider id, or a minted identity when
    /// the wire omits one — so parallel id-less calls can never share an
    /// assembly key downstream.
    key: StreamPartId,
    /// Established provider id: updated when a later chunk carries one.
    /// Empty until the wire supplies one.
    pub id: String,
    /// Established tool name: the last non-empty value seen.
    pub name: String,
    /// Provider-specific decoration carried onto the call's end event.
    pub signature: Option<String>,
    /// Provider-specific decoration carried onto the call's end event.
    pub additional_params: Option<serde_json::Value>,
    /// Whether any raw argument fragment streamed for this slot. The done
    /// item's unparseable restatement re-emits its raw bytes only when NO
    /// fragment preceded it — the buffer already holds streamed bytes, and
    /// re-emitting the restatement doubled them.
    pub saw_arguments_delta: bool,
    /// Whether any argument fragment carried a non-whitespace byte. An empty
    /// argument slot under an output-length finish reason is an incomplete
    /// call, not evidence that the model deliberately invoked a zero-argument
    /// tool.
    saw_non_whitespace_arguments_delta: bool,
    /// The payload the wire announced when it opened the call (Gemini
    /// Interactions `step.start` usually carries `"arguments": {}`, and
    /// sometimes the whole payload). Replace-if-no-deltas, never
    /// concatenated: fragments, when they arrive, ARE the arguments and
    /// the announce is ignored; only a slot that fragmented nothing falls
    /// back to what it announced.
    pub announce_arguments: Option<serde_json::Value>,
}

impl ToolCallSlot {
    /// The assembly id this call's fragments are emitted under.
    pub fn key(&self) -> &StreamPartId {
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

    /// Build the end event that closes this call's assembly in the shared
    /// accumulator, carrying the established provider id and any decoration.
    pub fn end_event(&self, on_unparseable: UnparseableToolInput) -> ToolInputEnd {
        let mut end = ToolInputEnd::new(self.key.clone(), on_unparseable);
        // Only an established provider id overrides the assembly key; a
        // call whose wire never supplied one carries no durable handle at
        // all (`WireId::new` rejects the empty string by construction).
        end.tool_id = crate::streaming::WireId::new(self.id.clone());
        end.signature = self.signature.clone();
        end.additional_params = self.additional_params.clone();
        if !self.saw_arguments_delta {
            end.arguments = self.announce_arguments.clone();
        }
        end
    }
}

/// Per-stream index → grammar-identity map for streamed tool calls.
///
/// `I` is the wire's own index type (`usize` for chat-compat chunk indices,
/// `i32` for Bedrock content-block indices); it must display so a minted id
/// can derive from it, and order so a drain preserves wire ordering.
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

    /// Open (or update) the slot for a wire index, establishing its identity.
    ///
    /// On first sight the assembly key is fixed: the wire id when one is
    /// supplied, else a freshly minted identity — the single enforcement
    /// point of the mandatory-identity invariant. Later fragments update the
    /// established provider id and name from any non-empty values they
    /// carry.
    pub fn open(
        &mut self,
        index: I,
        wire_id: Option<&str>,
        name: Option<&str>,
    ) -> &mut ToolCallSlot {
        let minted = &mut self.minted;
        let slot = self.slots.entry(index).or_insert_with(|| ToolCallSlot {
            key: match wire_id {
                Some(id) if !id.is_empty() => StreamPartId::wire(id),
                // Id-less wires (several llama.cpp/vllm-style gateways) key
                // tool calls by index alone; the slot identity is minted so
                // it can never collide with a wire-genuine id — and can
                // never serialize upstream.
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

    /// The open slot at a wire index, mutably — for fragment bookkeeping
    /// on an already-open slot without `open`'s insert-if-absent.
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
    ///
    /// Two matching rules keep this deterministic:
    /// - A slot whose wire never established a provider id (empty `id`) never
    ///   matches — a decoration for the empty string would otherwise pick an
    ///   arbitrary id-less slot out of `HashMap` iteration order.
    /// - Each field is **first-wins**: a later decoration for the same call
    ///   fills only the fields still unset, so a gemini-style
    ///   signature-then-params sequence composes instead of the second
    ///   decoration clobbering the first's signature with `None`.
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

    /// Whether any call is still open.
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

    /// [`ToolCallBridge::drain_ordered`], keeping each slot's wire index —
    /// for adapters that track per-slot state of their own beside the
    /// bridge (the Responses adapter's pending `call_id`s).
    pub fn drain_ordered_indexed(&mut self) -> Vec<(I, ToolCallSlot)> {
        let mut slots: Vec<(I, ToolCallSlot)> = self.slots.drain().collect();
        slots.sort_by_key(|(index, _)| *index);
        slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_id_becomes_the_assembly_key() {
        let mut bridge = ToolCallBridge::<usize>::new();
        let slot = bridge.open(0, Some("call_abc"), Some("get_weather"));
        assert_eq!(slot.key(), &StreamPartId::wire("call_abc"));
        assert_eq!(slot.id, "call_abc");
        assert_eq!(slot.name, "get_weather");

        // The established id rides the end event as the override.
        let end = slot.end_event(UnparseableToolInput::Drop);
        assert_eq!(end.id, StreamPartId::wire("call_abc"));
        assert_eq!(end.tool_id.as_ref().map(|id| id.as_str()), Some("call_abc"));
    }

    #[test]
    fn id_less_open_mints_a_distinct_minted_key_per_index() {
        let mut bridge = ToolCallBridge::<usize>::new();
        let first_key = bridge.open(0, None, Some("get_weather")).key().clone();
        let second_key = bridge.open(1, None, Some("get_time")).key().clone();

        // Parallel id-less calls must never share an assembly key, and a
        // minted key is minted — structurally unable to serialize upstream
        // as wire-genuine.
        assert_ne!(first_key, second_key);
        assert!(first_key.is_minted());
        assert!(second_key.is_minted());

        // A call whose wire never supplied an id keeps its minted key with
        // no provider-id override.
        let slot = bridge.remove(0).expect("slot must be open");
        let end = slot.end_event(UnparseableToolInput::Drop);
        assert_eq!(end.id, first_key);
        assert!(end.tool_id.is_none());
    }

    #[test]
    fn late_wire_id_updates_the_override_but_not_the_key() {
        let mut bridge = ToolCallBridge::<usize>::new();
        bridge.open(0, None, Some("get_weather"));
        let slot = bridge.open(0, Some("call_late"), None);
        // The assembly key is fixed at open; the late provider id becomes
        // the end-event override the accumulator surfaces to the consumer.
        assert!(slot.key().is_minted());
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
        assert_eq!(evicted.key(), &StreamPartId::wire("call_a"));
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

    /// An empty-id decoration must never match: id-less slots keep an empty
    /// established id, and matching `""` would decorate an arbitrary one of
    /// them in `HashMap` iteration order.
    #[test]
    fn an_empty_id_decoration_never_matches_an_id_less_slot() {
        let mut bridge = ToolCallBridge::<usize>::new();
        bridge.open(0, None, Some("get_weather"));
        bridge.open(1, None, Some("get_time"));

        bridge.decorate(ToolCallDecoration {
            tool_id: String::new(),
            signature: Some("sig".to_owned()),
            additional_params: None,
        });

        for index in [0, 1] {
            let slot = bridge.remove(index).expect("slot open");
            assert!(
                slot.signature.is_none(),
                "an empty-id decoration must not land on slot {index}"
            );
        }
    }

    /// Decoration fields are first-wins: a gemini-style signature-then-params
    /// sequence composes, and a later decoration cannot clobber an earlier
    /// signature with `None`.
    #[test]
    fn decoration_fields_are_first_wins_per_field() {
        let mut bridge = ToolCallBridge::<usize>::new();
        bridge.open(0, Some("call_a"), Some("get_weather"));

        bridge.decorate(ToolCallDecoration {
            tool_id: "call_a".to_owned(),
            signature: Some("sig-1".to_owned()),
            additional_params: None,
        });
        bridge.decorate(ToolCallDecoration {
            tool_id: "call_a".to_owned(),
            signature: None,
            additional_params: Some(serde_json::json!({"thought": true})),
        });
        // A third decoration cannot overwrite either established field.
        bridge.decorate(ToolCallDecoration {
            tool_id: "call_a".to_owned(),
            signature: Some("sig-2".to_owned()),
            additional_params: Some(serde_json::json!({"other": 1})),
        });

        let slot = bridge.remove(0).expect("slot open");
        assert_eq!(slot.signature.as_deref(), Some("sig-1"));
        assert_eq!(
            slot.additional_params,
            Some(serde_json::json!({"thought": true}))
        );
    }

    #[test]
    fn drain_ordered_preserves_wire_index_order() {
        let mut bridge = ToolCallBridge::<i32>::new();
        bridge.open(2, Some("call_c"), None);
        bridge.open(0, Some("call_a"), None);
        bridge.open(1, Some("call_b"), None);

        let keys: Vec<StreamPartId> = bridge
            .drain_ordered()
            .into_iter()
            .map(|slot| slot.key().clone())
            .collect();
        assert_eq!(
            keys,
            vec![
                StreamPartId::wire("call_a"),
                StreamPartId::wire("call_b"),
                StreamPartId::wire("call_c")
            ]
        );
        assert!(bridge.is_empty());
    }
}
