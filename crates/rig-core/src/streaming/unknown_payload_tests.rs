use super::UnknownPayload;

/// The redaction is a property of the type: no Debug rendering — direct,
/// via a containing derive, or through a `warn!(?value)` capture — can
/// reproduce payload content.
#[test]
fn debug_output_never_contains_payload_content() {
    let payload = UnknownPayload::new(serde_json::json!({
        "secret_field": "SENSITIVE-CONTENT",
    }));
    let rendered = format!("{payload:?}");
    assert!(!rendered.contains("SENSITIVE-CONTENT"));
    assert!(!rendered.contains("secret_field"));
    assert!(rendered.contains("redacted"));
}

/// Serialization stays transparent, so wire round-trips are unchanged.
#[test]
fn serde_round_trip_is_transparent() {
    let value = serde_json::json!({"type": "future_event", "n": 1});
    let payload = UnknownPayload::new(value.clone());
    let encoded = serde_json::to_string(&payload).expect("serializes");
    assert_eq!(encoded, serde_json::to_string(&value).expect("serializes"));
    let decoded: UnknownPayload = serde_json::from_str(&encoded).expect("deserializes");
    assert_eq!(decoded, payload);
}
