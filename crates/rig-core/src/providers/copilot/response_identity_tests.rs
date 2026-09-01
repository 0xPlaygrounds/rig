use super::*;

/// Both Copilot routes' streaming terminals carry the transport request id
/// (stamped by the shared SSE capture) into the normalized `StreamFinal`.
/// Deterministic and credential-free: the transport halves — the shared
/// OpenAI chat wrapper's capture and `stamp_terminal_request_id` on the
/// Responses route — are covered by the shared-path tests; this locks the
/// Copilot-specific conversion layer.
#[test]
fn streaming_terminals_carry_request_id_into_stream_final() {
    let mut chat_terminal = openai::completion::streaming::StreamingCompletionResponse::<
        openai::completion::Usage,
    >::new(openai::completion::Usage::default());
    chat_terminal.provider_request_id = Some("req-chat".to_string());
    let chat_final: crate::streaming::StreamFinal =
        (PROVIDER_NAME, CopilotStreamingResponse::Chat(chat_terminal)).into();
    assert_eq!(chat_final.provider_request_id.as_deref(), Some("req-chat"));

    let mut responses_terminal = responses_api::streaming::StreamingCompletionResponse::new(
        serde_json::from_value(
            serde_json::json!({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
        )
        .expect("usage should parse"),
    );
    responses_terminal.provider_request_id = Some("req-responses".to_string());
    let responses_final: crate::streaming::StreamFinal = (
        PROVIDER_NAME,
        CopilotStreamingResponse::Responses(responses_terminal),
    )
        .into();
    assert_eq!(
        responses_final.provider_request_id.as_deref(),
        Some("req-responses")
    );
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
