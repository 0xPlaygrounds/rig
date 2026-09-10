use super::*;

/// Serde compatibility (rig#2265): responses persisted before
/// `provider_request_id` existed still load, with the field `None`.
#[test]
fn completion_response_without_request_id_still_deserializes() {
    let response: CompletionResponse = serde_json::from_str(
        r#"{"choice": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                          "cached_input_tokens": 0, "cache_creation_input_tokens": 0,
                          "reasoning_tokens": 0},
                "provider": "test"}"#,
    )
    .expect("pre-identity CompletionResponse JSON should load");
    assert_eq!(response.provider_request_id, None);
    assert_eq!(response.identity(), ResponseIdentity::default());
}

/// The identity accessor mirrors the flat fields exactly.
#[test]
fn identity_accessor_mirrors_flat_fields() {
    let response = CompletionResponse::new(
        vec![crate::completion::AssistantContent::text("hi")],
        Usage::new(),
        "test",
    )
    .with_message_id("msg_1")
    .with_response_id("resp_1")
    .with_provider_request_id("req_1");
    assert_eq!(
        response.identity(),
        ResponseIdentity {
            message_id: Some("msg_1".into()),
            response_id: Some("resp_1".into()),
            provider_request_id: Some("req_1".into()),
        }
    );
}
