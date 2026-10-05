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

/// Releasing the last command handle ends the actor, so a connection whose
/// owner is gone never keeps its socket. The owner's drop also aborts the
/// actor, and that abort can land first, so this drives the actor directly.
#[tokio::test]
async fn the_actor_ends_when_its_last_handle_is_released() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a loopback listener");
    let address = listener.local_addr().expect("listener address");
    let (client, accepted) = tokio::join!(TcpStream::connect(address), listener.accept());
    let client = client.expect("connect to the listener");
    let _server = accepted.expect("accept the client");
    let socket = WebSocketStream::from_raw_socket(
        MaybeTlsStream::Plain(client),
        tungstenite::protocol::Role::Client,
        None,
    )
    .await;
    let (commands, requests) = futures::channel::mpsc::channel::<Command>(1);
    drop(commands);

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        run_actor(socket, requests),
    )
    .await
    .expect("the actor should end once no handle remains");
}
