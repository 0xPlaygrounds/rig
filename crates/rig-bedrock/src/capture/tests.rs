use super::*;
use aws_smithy_types::event_stream::{Header, HeaderValue};
use serde_json::json;

fn message(kind: &str, event: &str, payload: &[u8]) -> Vec<u8> {
    let name = if kind == "exception" {
        ":exception-type"
    } else {
        ":event-type"
    };
    let message = Message::new(payload.to_vec())
        .add_header(Header::new(
            ":message-type",
            HeaderValue::String(kind.to_owned().into()),
        ))
        .add_header(Header::new(
            name,
            HeaderValue::String(event.to_owned().into()),
        ));
    let mut bytes = Vec::new();
    aws_smithy_eventstream::frame::write_message_to(&message, &mut bytes).expect("writes");
    bytes
}

/// Events and exceptions come out whole however the body is chunked.
#[test]
fn events_read_whole_messages_across_chunks() {
    let mut body = message("event", "messageStop", br#"{"stopReason":"end_turn"}"#);
    body.extend(message(
        "exception",
        "throttlingException",
        br#"{"message":"slow"}"#,
    ));
    body.extend(message(
        "event",
        "metadata",
        br#"{"metrics":{"latencyMs":5}}"#,
    ));
    let mut events = Events::default();
    let mut read = Vec::new();
    for chunk in body.chunks(7) {
        read.extend(events.read(chunk));
    }
    assert_eq!(
        read,
        [
            json!({ "messageStop": { "stopReason": "end_turn" } }),
            json!({ "throttlingException": { "message": "slow" } }),
            json!({ "metadata": { "metrics": { "latencyMs": 5 } } }),
        ]
    );
}

/// A body reads as the bytes it carries, then ends.
#[tokio::test]
async fn a_body_reads_as_its_bytes() {
    let mut body = SdkBody::from(r#"{"stopReason":"end_turn"}"#);
    let mut read = Vec::new();
    while let Some(bytes) = chunk(&mut body).await {
        read.extend(bytes.expect("bytes"));
    }
    assert_eq!(read, br#"{"stopReason":"end_turn"}"#);
    assert!(Capture::new(Vec::new()).reply().is_none());
}
