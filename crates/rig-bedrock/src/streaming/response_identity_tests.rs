use super::*;

/// Blocking/streaming parity (rig#2265): the streaming terminal's AWS
/// request id — stamped from the SDK operation output, the same source
/// the unary surface reads — normalizes into
/// `StreamFinal.provider_request_id`.
#[test]
fn streaming_terminal_request_id_normalizes_into_stream_final() {
    let response = BedrockStreamingResponse {
        usage: None,
        stop_reason: Some(StopReason::EndTurn),
        provider_request_id: Some("aws-req-1".to_string()),
    };

    let usage = (&response).into();
    let terminal = rig_core::streaming::StreamFinal::new(PROVIDER_NAME, usage)
        .with_optional_provider_request_id(response.provider_request_id.clone())
        .with_optional_finish_reason(response.stop_reason.as_ref().map(map_stop_reason));
    assert_eq!(terminal.provider_request_id.as_deref(), Some("aws-req-1"));

    // And a response without one stays None — never an error.
    let without = BedrockStreamingResponse {
        usage: None,
        stop_reason: None,
        provider_request_id: None,
    };
    let terminal = rig_core::streaming::StreamFinal::new(PROVIDER_NAME, (&without).into())
        .with_optional_provider_request_id(without.provider_request_id.clone());
    assert_eq!(terminal.provider_request_id, None);
}
