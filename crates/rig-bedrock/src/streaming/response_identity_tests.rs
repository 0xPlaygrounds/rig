use super::*;

/// Blocking/streaming parity (rig#2265): the streaming end's AWS request id,
/// captured from the SDK operation output (the same source the unary
/// surface reads), is the end's `provider_request_id`.
#[test]
fn streaming_request_id_is_the_ends_request_id() {
    let response = BedrockStreamingResponse {
        usage: None,
        stop_reason: Some(StopReason::EndTurn),
        provider_request_id: Some("aws-req-1".to_string()),
    };
    assert_eq!(
        finish_of(&response).provider_request_id.as_deref(),
        Some("aws-req-1")
    );

    // And a response without one stays None — never an error.
    let without = BedrockStreamingResponse {
        usage: None,
        stop_reason: None,
        provider_request_id: None,
    };
    assert_eq!(finish_of(&without).provider_request_id, None);
}
