//! Asset identity, exact transport spelling and bounded retention.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rig_core::message::DocumentSourceKind;
use rig_ecs::agent::content::binary::*;

#[test]
fn equivalent_sources_share_bytes_but_keep_each_wire_spelling() {
    let mut store = BinaryAssets::default();
    for spelling in ["Zg==", "Zg=", "Zg", "Zh==", "Zh=", "Zh"] {
        let original = DocumentSourceKind::Base64(spelling.into());
        let handle = store.intern(original.clone()).unwrap();
        assert_eq!(store.resolve(&handle).unwrap(), original);
        assert_eq!(store.len(), 1);
        assert_eq!(store.byte_len(), 1);
    }
    let raw = store
        .intern(DocumentSourceKind::Raw(b"f".to_vec()))
        .unwrap();
    assert_eq!(
        store.resolve(&raw).unwrap(),
        DocumentSourceKind::Raw(b"f".to_vec())
    );
    assert_eq!(store.len(), 1);
    assert_eq!(store.get(BinaryId::of(b"f")).unwrap(), b"f");
}

#[test]
fn every_short_payload_preserves_optional_padding_and_trailing_bits() {
    use base64::{Engine, prelude::BASE64_STANDARD};
    let mut store = BinaryAssets::default();
    for size in 0..32 {
        let payload: Vec<u8> = (0..size).map(|n| n * 7).collect();
        let canonical = BASE64_STANDARD.encode(&payload);
        for spelling in [
            canonical.clone(),
            canonical.trim_end_matches('=').to_owned(),
        ] {
            let source = DocumentSourceKind::Base64(spelling);
            let handle = store.intern(source.clone()).unwrap();
            assert_eq!(store.resolve(&handle).unwrap(), source);
        }
    }
}

#[test]
fn nonbinary_sources_stay_data_without_asset_or_io() {
    let mut store = BinaryAssets::default();
    for source in [
        DocumentSourceKind::Url("https://invalid.invalid/image".into()),
        DocumentSourceKind::FileId("file-123".into()),
        DocumentSourceKind::String("not base64".into()),
        DocumentSourceKind::Unknown,
    ] {
        let handle = store.intern(source.clone()).unwrap();
        assert_eq!(store.resolve(&handle).unwrap(), source);
    }
    assert!(store.is_empty());
}

#[test]
fn corrupt_and_missing_payloads_are_rejected_without_retention_changes() {
    let mut store = BinaryAssets::default();
    let id = store.insert(b"kept".to_vec()).unwrap();
    let missing = BinaryId::of(b"absent");
    assert_eq!(store.retain([missing]), Err(BinaryError::Missing));
    assert_eq!(store.get(id).unwrap(), b"kept");
    assert!(matches!(
        BinaryAssets::from_payloads([(id, b"corrupt".to_vec())], BinaryLimits::default()),
        Err(BinaryError::Corrupt)
    ));
    assert!(matches!(
        BinaryAssets::from_payloads(
            [(id, b"kept".to_vec()), (id, b"kept".to_vec())],
            BinaryLimits::default()
        ),
        Err(BinaryError::Corrupt)
    ));
    let reference = PartSource::Binary {
        id: missing,
        encoding: BinaryEncoding::Raw,
    };
    assert_eq!(store.resolve(&reference), Err(BinaryError::Missing));
}

#[test]
fn limits_cover_decode_total_and_distinct_count() {
    let mut store = BinaryAssets::with_limits(BinaryLimits {
        per_asset: 3,
        total: 4,
        count: 2,
    });
    assert_eq!(
        store.intern(DocumentSourceKind::Base64("YWFhYQ==".into())),
        Err(BinaryError::Limit)
    );
    assert_eq!(
        store.intern(DocumentSourceKind::Base64("!".into())),
        Err(BinaryError::Base64)
    );
    assert!(store.is_empty());
    let first = store.insert(vec![1, 2, 3]).unwrap();
    assert_eq!(store.insert(vec![1, 2, 3]).unwrap(), first);
    assert_eq!(store.insert(vec![2, 3]), Err(BinaryError::Limit));
    let second = store.insert(vec![4]).unwrap();
    assert_eq!(store.insert(vec![]), Err(BinaryError::Limit));
    store.retain([second]).unwrap();
    assert_eq!(store.byte_len(), 1);
    assert_eq!(store.get(first), Err(BinaryError::Missing));
    store.retain([]).unwrap();
    assert!(store.is_empty());
}

#[test]
fn scene_spelling_metadata_cannot_change_decoded_bytes() {
    let mut store = BinaryAssets::default();
    let id = store.insert(vec![102]).unwrap();
    for encoding in [
        BinaryEncoding::Base64 {
            padding: 3,
            last_symbol: None,
        },
        BinaryEncoding::Base64 {
            padding: 2,
            last_symbol: Some(b'A'),
        },
        BinaryEncoding::Base64 {
            padding: 2,
            last_symbol: Some(255),
        },
    ] {
        assert_eq!(
            store.resolve(&PartSource::Binary { id, encoding }),
            Err(BinaryError::Spelling)
        );
    }
}
