use super::*;

#[test]
fn an_absent_provider_handle_is_none_not_empty() {
    assert!(non_empty_id("").is_none());
    assert_eq!(non_empty_id("rs_123").as_deref(), Some("rs_123"));
}

#[test]
fn mint_counts_up_per_stream() {
    let mut ids = SyntheticIds::new(MintKind::Reasoning);
    assert_eq!(ids.mint(), BlockId::minted(MintKind::Reasoning, 0));
    assert_eq!(ids.mint(), BlockId::minted(MintKind::Reasoning, 1));
}

#[test]
fn minted_keys_are_distinct_across_kinds_and_indices() {
    let kinds = [
        MintKind::Reasoning,
        MintKind::EncryptedReasoning,
        MintKind::Block,
        MintKind::Output,
        MintKind::Tool,
        MintKind::Text,
    ];
    let mut seen = std::collections::HashSet::new();
    for kind in kinds {
        for index in [0u64, 1, 7, u64::MAX] {
            assert!(
                seen.insert(BlockId::minted(kind, index)),
                "collision at {kind:?}:{index}"
            );
        }
    }
}

#[test]
fn provenance_survives_serde_and_renders_for_logs() {
    let wire = BlockId::wire("tool-0");
    let minted = BlockId::minted(MintKind::Tool, 0);
    assert_ne!(wire, minted, "a wire id that looks minted stays a wire id");
    assert_eq!(wire.to_string(), "tool-0");
    assert_eq!(minted.to_string(), "tool-0");
    assert!(minted.is_minted() && !wire.is_minted());
    assert_eq!(wire.wire_str(), Some("tool-0"));
    assert_eq!(minted.wire_str(), None);

    for id in [wire, minted] {
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(serde_json::from_str::<BlockId>(&json).unwrap(), id);
    }
    assert_eq!(
        serde_json::to_value(BlockId::minted(MintKind::EncryptedReasoning, 3)).unwrap(),
        serde_json::json!("minted:encrypted_reasoning:3")
    );
    assert_eq!(
        serde_json::to_value(BlockId::wire("rs_1")).unwrap(),
        serde_json::json!("wire:rs_1")
    );
    // A string form keys JSON maps, and decoding never guesses.
    let keyed: std::collections::HashMap<BlockId, u8> = [
        (BlockId::wire("a:b"), 1),
        (BlockId::minted(MintKind::Tool, 2), 2),
    ]
    .into_iter()
    .collect();
    let back: std::collections::HashMap<BlockId, u8> =
        serde_json::from_str(&serde_json::to_string(&keyed).unwrap()).unwrap();
    assert_eq!(back, keyed);
    assert!(serde_json::from_str::<BlockId>("\"tool-0\"").is_err());
    assert!(serde_json::from_str::<BlockId>("\"minted:nope:1\"").is_err());
}
