use super::*;
use bytes::Bytes;

/// Every frame the protocol carries must survive a round trip through
/// tungstenite's own representation, or a session sees the wrong event.
#[test]
fn frames_round_trip_through_tungstenite() {
    let cases = [
        Frame::Text("{\"type\":\"response.create\"}".to_string()),
        Frame::Binary(Bytes::from_static(b"\x00\x01")),
        Frame::Ping(Bytes::from_static(b"ping")),
        Frame::Pong(Bytes::new()),
        Frame::Close(Some(CloseFrame {
            code: 1000,
            reason: "done".to_string(),
        })),
        Frame::Close(None),
    ];

    for frame in cases {
        assert_eq!(
            from_message(into_message(frame.clone())),
            Some(frame.clone()),
            "frame should round-trip"
        );
    }
}

/// A raw frame is not protocol payload: it must be skipped, not handed on
/// as bytes a session would try to parse.
#[test]
fn a_raw_frame_carries_no_protocol_payload() {
    let raw = Message::Frame(tungstenite::protocol::frame::Frame::message(
        Bytes::from_static(b"raw"),
        tungstenite::protocol::frame::coding::OpCode::Data(
            tungstenite::protocol::frame::coding::Data::Binary,
        ),
        true,
    ));

    assert_eq!(from_message(raw), None);
}
