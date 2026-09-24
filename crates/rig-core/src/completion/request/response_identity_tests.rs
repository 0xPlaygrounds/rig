use super::*;

/// `provider_request_id` is skipped when `None`, so a response written
/// without one loads with the field `None` (rig#2265).
#[test]
fn completion_response_without_request_id_deserializes() {
    let response: CompletionResponse = serde_json::from_str(
        r#"{"choice": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                          "cached_input_tokens": 0, "cache_creation_input_tokens": 0,
                          "reasoning_tokens": 0},
                "provider": "test", "raw": null}"#,
    )
    .expect("a CompletionResponse without a request id should load");
    assert_eq!(response.provider_request_id, None);
    assert_eq!(response.identity(), ResponseIdentity::default());
}

/// The identity accessor mirrors the flat fields exactly.
#[test]
fn identity_accessor_mirrors_flat_fields() {
    let response = CompletionResponse {
        message_id: Some("msg_1".try_into().expect("a non-empty id")),
        response_id: Some("resp_1".try_into().expect("a non-empty id")),
        provider_request_id: Some("req_1".try_into().expect("a non-empty id")),
        ..CompletionResponse::new(
            vec![crate::completion::AssistantContent::text("hi")],
            Usage::default(),
            "test",
            serde_json::json!({}),
        )
    };
    assert_eq!(
        response.identity(),
        ResponseIdentity {
            message_id: Some("msg_1".try_into().expect("a non-empty id")),
            response_id: Some("resp_1".try_into().expect("a non-empty id")),
            provider_request_id: Some("req_1".try_into().expect("a non-empty id")),
        }
    );
}
