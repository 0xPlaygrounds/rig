//! Asset identity, exact transport spelling and bounded retention.
use rig_core::message::DocumentSourceKind;
use rig_ecs::agent::content::binary::*;

#[test]
fn nonbinary_sources_stay_data_without_asset_or_io() {
    let mut store = BinaryAssets::default();
    for source in [
        DocumentSourceKind::Url("https://invalid.invalid/image".into()),
        DocumentSourceKind::FileId("file-123".into()),
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

/// A legacy checkpoint's `String` media source loads as base64, the same
/// reading as the legacy `"string"` JSON spelling.
#[test]
fn a_legacy_string_media_source_resolves_as_base64() {
    let store = BinaryAssets::default();
    let legacy = PartSource::String("iVBORw0KGgo=".into());
    assert_eq!(
        store.resolve(&legacy),
        Ok(DocumentSourceKind::Base64("iVBORw0KGgo=".into()))
    );
}
