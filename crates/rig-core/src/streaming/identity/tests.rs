use super::*;

/// The opaque-key contract, at the API-shape level (the stronger
/// property — no `Serialize`, no rendering, `WireId` rejecting
/// emptiness at its only constructor — is enforced by the
/// `identity_leak` compile-fail tests).
#[test]
fn an_absent_provider_handle_is_none_not_empty() {
    assert!(WireId::new("").is_none());
    assert_eq!(
        WireId::new("rs_123").expect("non-empty").into_string(),
        "rs_123"
    );
}

#[test]
fn mint_counts_up_per_stream() {
    let mut ids = SyntheticIds::new(MintKind::Reasoning);
    assert_eq!(ids.mint(), StreamPartId::minted(MintKind::Reasoning, 0));
    assert_eq!(ids.mint(), StreamPartId::minted(MintKind::Reasoning, 1));
}

/// Keys minted by different subsystems stay distinct even at equal
/// indices — bookkeeping hygiene (nothing can observe a collision, but
/// the accumulator's maps deserve distinct keys anyway).
#[test]
fn minted_keys_are_distinct_across_kinds_and_indices() {
    let kinds = [
        MintKind::Reasoning,
        MintKind::Block,
        MintKind::Output,
        MintKind::Tool,
        MintKind::Text,
    ];
    let mut seen = std::collections::HashSet::new();
    for kind in kinds {
        for index in [0u64, 1, 7, u64::MAX] {
            assert!(
                seen.insert(StreamPartId::minted(kind, index)),
                "collision at {kind:?}:{index}"
            );
        }
    }
}
