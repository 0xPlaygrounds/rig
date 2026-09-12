use super::*;
use crate::streaming::{BlockClose, StreamEvent};

#[test]
fn wire_id_becomes_the_assembly_key() {
    let mut bridge = ToolCallBridge::<usize>::new();
    let slot = bridge.open(0, Some("call_abc"), Some("get_weather"));
    assert_eq!(slot.key(), &BlockId::wire("call_abc"));
    assert_eq!(slot.id, "call_abc");
    assert_eq!(slot.name, "get_weather");

    // The established id rides the end event as the override, and the
    // end event is keyed by the assembly key.
    let end = slot.end(UnparseableToolInput::Drop);
    assert_eq!(end.tool_id.as_deref(), Some("call_abc"));
    assert_eq!(
        slot.end_event(UnparseableToolInput::Drop),
        StreamEvent::BlockEnd {
            id: BlockId::wire("call_abc"),
            end: BlockClose::ToolCall(end),
            block: None,
        }
    );
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
    let end = slot.end(UnparseableToolInput::Drop);
    assert!(end.tool_id.is_none());
    assert_eq!(
        slot.end_event(UnparseableToolInput::Drop).block_id(),
        Some(&first_key)
    );
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
    assert_eq!(evicted.key(), &BlockId::wire("call_a"));
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
    let end = undecorated.end(UnparseableToolInput::Drop);
    assert!(end.signature.is_none());

    let decorated = bridge.remove(1).expect("slot 1 open");
    let end = decorated.end(UnparseableToolInput::Drop);
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

    let keys: Vec<BlockId> = bridge
        .drain_ordered()
        .into_iter()
        .map(|slot| slot.key().clone())
        .collect();
    assert_eq!(
        keys,
        vec![
            BlockId::wire("call_a"),
            BlockId::wire("call_b"),
            BlockId::wire("call_c")
        ]
    );
    assert!(bridge.is_empty());
}
