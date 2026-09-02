use super::*;

/// Blocking/streaming parity (rig#2265): the streaming terminal's AWS
/// request id — captured from the SDK operation output, the same source
/// the unary surface reads — normalizes through the adapter's terminal
/// mapping into
/// `StreamFinal.provider_request_id`.
#[test]
fn streaming_terminal_request_id_normalizes_into_stream_final() {
    let response = BedrockStreamingResponse {
        usage: None,
        stop_reason: Some(StopReason::EndTurn),
        provider_request_id: Some("aws-req-1".to_string()),
    };

    let terminal = terminal_record(response).expect("terminal record");
    assert_eq!(terminal.provider_request_id.as_deref(), Some("aws-req-1"));

    // And a response without one stays None — never an error.
    let without = BedrockStreamingResponse {
        usage: None,
        stop_reason: None,
        provider_request_id: None,
    };
    let terminal = terminal_record(without).expect("terminal record");
    assert_eq!(terminal.provider_request_id, None);
}
