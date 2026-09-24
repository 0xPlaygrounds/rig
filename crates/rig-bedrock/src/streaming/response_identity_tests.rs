use super::*;
use futures::StreamExt;

/// Blocking/streaming parity: the streaming terminal's AWS request id, read
/// off the SDK operation output as the unary surface reads it, reaches the
/// terminal record through the driver and stays in Bedrock's own terminal
/// document.
#[tokio::test]
async fn the_sdk_request_id_reaches_the_terminal_record() {
    let state = StreamState {
        provider_request_id: Some("aws-req-1".to_string()),
        ..StreamState::default()
    };
    let stop = aws_bedrock::MessageStopEvent::builder()
        .stop_reason(aws_bedrock::StopReason::EndTurn)
        .build()
        .expect("message stop should build");
    let metadata = aws_bedrock::ConverseStreamMetadataEvent::builder().build();
    let events = futures::stream::iter([
        Ok(aws_bedrock::ConverseStreamOutput::MessageStop(stop)),
        Ok(aws_bedrock::ConverseStreamOutput::Metadata(metadata)),
    ]);
    let mut stream = run_wire_stream(
        PROVIDER_NAME,
        RequestId::non_empty("aws-req-1"),
        events,
        state,
    );
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let rig_core::streaming::StreamEvent::Final(end) = item.expect("stream item") {
            terminal = Some(end);
        }
    }
    let terminal = terminal.expect("a terminal record");
    assert_eq!(
        terminal.meta.provider_request_id.as_deref(),
        Some("aws-req-1")
    );
    assert_eq!(terminal.meta.raw["provider_request_id"], "aws-req-1");
}

/// A terminal without a request id reports none: never an error.
#[test]
fn a_terminal_without_a_request_id_reports_none() {
    let without = BedrockStreamingResponse {
        usage: None,
        stop_reason: None,
        provider_request_id: None,
    };
    let terminal = terminal_record(without).expect("terminal record");
    assert!(terminal.raw.get("provider_request_id").is_none());
}
