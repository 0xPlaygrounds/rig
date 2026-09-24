use super::*;

/// Absent values are skipped on write, so a response the provider gave no
/// request id loads with the field `None`.
#[test]
fn completion_response_without_request_id_deserializes() {
    let response: CompletionResponse = serde_json::from_str(
        r#"{"choice": [{"type": "text", "text": "hi"}],
            "end": {"meta": {"provider": "test",
                             "usage": {"input_tokens": 1, "output_tokens": 1}}}}"#,
    )
    .expect("a CompletionResponse without a request id should load");
    assert_eq!(response.end.meta.provider_request_id, None);
    assert_eq!(response.end.message_id, None);
    assert_eq!(response.end.meta.usage.input_tokens, Some(1));
}

/// An empty identifier is not an identifier: a record carrying one is
/// refused rather than loaded as present.
#[test]
fn completion_response_with_empty_request_id_is_refused() {
    let loaded = serde_json::from_str::<CompletionResponse>(
        r#"{"choice": [], "end": {"meta": {"provider": "test", "provider_request_id": ""}}}"#,
    );
    assert!(loaded.is_err(), "an empty request id must not load");
}
