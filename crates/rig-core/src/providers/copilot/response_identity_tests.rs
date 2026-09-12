use super::*;

/// The chat route's streaming terminal carries the transport request id
/// (stamped by the shared SSE capture) into the normalized `StreamFinal`.
/// Deterministic and credential-free: the transport halves — the shared
/// OpenAI chat wrapper's capture and `stamp_terminal_request_id` on both
/// routes — are covered by the shared-path tests; this locks the
/// conversion layer the chat route reuses under Copilot's name.
#[test]
fn chat_streaming_terminal_carries_request_id_into_stream_final() {
    let mut chat_terminal = openai::completion::streaming::StreamingCompletionResponse::<
        openai::completion::Usage,
    >::new(openai::completion::Usage::default());
    chat_terminal.provider_request_id = Some("req-chat".to_string());
    let chat_final = chat_terminal.into_stream_final(PROVIDER_NAME);
    assert_eq!(chat_final.provider_request_id.as_deref(), Some("req-chat"));
    assert_eq!(chat_final.provider, PROVIDER_NAME);
}

/// The Responses-route unary wire type carries the stamped id through
/// `normalize` into the core response; the chat route has no wire slot,
/// so `completion()` stamps the normalized response from the returned
/// pair — asserted here at the conversion layer for the responses half.
#[test]
fn responses_unary_wire_id_survives_normalize() {
    use crate::completion::NormalizeCompletionResponse;

    let payload = serde_json::json!({
        "id": "resp_123",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-test",
        "output": [{
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}]
        }]
    });
    let mut response: responses_api::CompletionResponse =
        serde_json::from_value(payload).expect("wire response should parse");
    response.provider_request_id = Some("req-unary".to_string());

    let normalized = response
        .normalize(PROVIDER_NAME)
        .expect("response should normalize");
    assert_eq!(normalized.provider_request_id.as_deref(), Some("req-unary"));
    assert_eq!(normalized.response_id.as_deref(), Some("resp_123"));
    assert_eq!(normalized.provider, PROVIDER_NAME);
}
