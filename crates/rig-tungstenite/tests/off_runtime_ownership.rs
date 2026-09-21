//! Who owns the work this backend moves onto the fallback runtime.
//!
//! Off-runtime, a connect and a connection are both tokio tasks the caller
//! cannot see. Ownership means dropping the caller's handle releases the
//! socket: an abandoned handshake must stop connecting, and a dropped
//! connection must hand the transport back even when its actor is parked in a
//! write the peer is not reading. Neither is observable from the client side —
//! the only witness is the server, so these drive real loopback sockets.

#![cfg(not(target_family = "wasm"))]
#![allow(clippy::expect_used, clippy::panic)]

use bytes::Bytes;
use rig_core::http_client::{NoBody, Request};
use rig_core::ws_client::{
    ConnectOptions, Frame, WebSocketClientExt as _, WebSocketConnection as _,
};
use rig_tungstenite::TungsteniteClient;
use std::sync::mpsc;
use std::time::Duration;
use tokio::io::AsyncReadExt as _;
use tokio::net::TcpListener;

/// How long a server thread waits for the client's socket to close before
/// reporting that it never did. Long enough that a slow machine cannot fail
/// the test, short enough that a regression fails rather than hangs.
const CLOSE_DEADLINE: Duration = Duration::from_secs(10);

/// Under tungstenite's default frame cap. Size alone is not proof of a stall:
/// the test also observes bytes arriving, a pending send, and an incomplete
/// payload after cancellation while the peer remains open.
const STALLED_PAYLOAD: usize = 12 * 1024 * 1024;

/// Read everything the peer still sends, returning the byte count once it
/// closes. An error counts as closed: a socket dropped mid-write may reach the
/// peer as a reset rather than a clean end of stream.
async fn drain_to_close(stream: &mut tokio::net::TcpStream) -> usize {
    let mut buffer = vec![0u8; 64 * 1024];
    let mut total = 0;
    loop {
        match stream.read(&mut buffer).await {
            Ok(0) | Err(_) => return total,
            Ok(read) => total += read,
        }
    }
}

/// A server that accepts the TCP connection, reads the upgrade request and
/// then answers nothing, leaving the handshake pending forever.
///
/// Reports whether the client's socket closed within [`CLOSE_DEADLINE`].
fn stall_the_handshake() -> (
    String,
    futures::channel::oneshot::Receiver<()>,
    mpsc::Receiver<bool>,
) {
    let (arrived, arrival) = futures::channel::oneshot::channel();
    let (address_tx, address_rx) = mpsc::channel();
    let (closed_tx, closed_rx) = mpsc::channel();

    // The server needs a reactor on its own thread; the client under test must
    // not have one.
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("server runtime should build");
        runtime.block_on(async move {
            let _ = tokio::time::timeout(Duration::from_secs(30), async move {
                let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                address_tx
                    .send(listener.local_addr().expect("address"))
                    .expect("address should send");

                let (mut stream, _) = listener.accept().await.expect("accept");
                let mut request = [0u8; 4096];
                assert!(stream.read(&mut request).await.expect("upgrade request") > 0);
                arrived.send(()).expect("arrival");

                let closed = tokio::time::timeout(CLOSE_DEADLINE, drain_to_close(&mut stream))
                    .await
                    .is_ok();
                let _ = closed_tx.send(closed);
            })
            .await;
        });
    });

    let address = address_rx
        .recv_timeout(CLOSE_DEADLINE)
        .expect("server should report its address");
    (format!("ws://{address}/"), arrival, closed_rx)
}

/// Hold the peer open until the test explicitly permits draining. Peeking
/// proves the actor started writing without releasing socket backpressure.
fn accept_then_stop_reading(
    observe_write: bool,
) -> (
    String,
    futures::channel::oneshot::Receiver<()>,
    futures::channel::oneshot::Sender<()>,
    mpsc::Receiver<(usize, bool)>,
) {
    let (address_tx, address_rx) = mpsc::channel();
    let (report_tx, report_rx) = mpsc::channel();
    let (arrived, arrival) = futures::channel::oneshot::channel();
    let (drain, draining) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("server runtime");
        runtime.block_on(async move {
            // This safety deadline exceeds every client-side observation.
            let _ = tokio::time::timeout(Duration::from_secs(30), async move {
                let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
                address_tx
                    .send(listener.local_addr().expect("address"))
                    .expect("address send");
                let (stream, _) = listener.accept().await.expect("accept");
                let mut socket = tokio_tungstenite::accept_async(stream)
                    .await
                    .expect("upgrade");
                let stream = socket.get_mut();
                if observe_write {
                    let mut byte = [0; 1];
                    assert!(
                        stream.peek(&mut byte).await.expect("peek") > 0,
                        "peer closed before writing"
                    );
                    arrived.send(()).expect("arrival");
                }
                draining.await.expect("permit draining");
                let drained = tokio::time::timeout(CLOSE_DEADLINE, drain_to_close(stream)).await;
                let closed = drained.is_ok();
                let _ = report_tx.send((drained.unwrap_or_default(), closed));
            })
            .await;
        });
    });
    let address = address_rx
        .recv_timeout(CLOSE_DEADLINE)
        .expect("server address");
    (format!("ws://{address}/"), arrival, drain, report_rx)
}

fn handshake_request(url: &str) -> Request<NoBody> {
    Request::builder()
        .uri(url)
        .body(NoBody)
        .expect("request should build")
}

/// Every test here is about the fallback runtime, which only exists when the
/// caller has none of their own.
fn assert_no_tokio_runtime() {
    assert!(
        tokio::runtime::Handle::try_current().is_err(),
        "this test is meaningless inside a tokio runtime: it exercises the fallback one"
    );
}

/// A caller that gives up on a connect — its own `select!`, its own timeout,
/// a cancelled request — must not leave a handshake running on the fallback
/// runtime, holding a socket nothing can ever reach.
#[test]
fn an_abandoned_connect_releases_the_socket_it_opened() {
    let (url, arrival, closed) = stall_the_handshake();
    assert_no_tokio_runtime();

    futures::executor::block_on(async {
        rig_core::wasm_compat::timeout(CLOSE_DEADLINE, async {
            let client = TungsteniteClient::new();
            let connecting = client.connect(handshake_request(&url), ConnectOptions::new());
            futures::pin_mut!(connecting);
            match futures::future::select(connecting, arrival).await {
                futures::future::Either::Right((arrival, _)) => arrival.expect("upgrade arrived"),
                futures::future::Either::Left(_) => {
                    panic!("held handshake completed before cancellation")
                }
            }
        })
        .await
        .expect("client deadline");
    });

    assert!(
        closed
            .recv_timeout(CLOSE_DEADLINE * 2)
            .expect("the server thread should report"),
        "dropping the connect must close the socket, not leave a detached handshake connecting"
    );
}

/// The connection owns its actor: dropping an idle one hands the transport
/// back instead of keeping a task alive for the life of the process.
#[test]
fn a_dropped_idle_connection_releases_the_socket() {
    let (url, _arrival, drain, report) = accept_then_stop_reading(false);
    assert_no_tokio_runtime();

    futures::executor::block_on(async {
        rig_core::wasm_compat::timeout(CLOSE_DEADLINE, async {
            let connection = TungsteniteClient::new()
                .connect(handshake_request(&url), ConnectOptions::new())
                .await
                .expect("the session should connect off-runtime");
            drop(connection);
            drain.send(()).expect("drain after owner drop");
        })
        .await
        .expect("client deadline");
    });

    let (received, closed) = report
        .recv_timeout(CLOSE_DEADLINE * 2)
        .expect("the server thread should report");
    assert!(
        closed,
        "dropping the connection must close the socket; the server saw {received} bytes and no close"
    );
}

/// The harder half: the actor is parked in a write to a peer that stopped
/// reading, so it is not sitting on the command channel and cannot notice the
/// connection go away. Dropping the connection must still end it.
#[test]
fn a_dropped_stalled_connection_releases_the_socket() {
    let (url, arrival, drain, report) = accept_then_stop_reading(true);
    assert_no_tokio_runtime();

    futures::executor::block_on(async {
        rig_core::wasm_compat::timeout(CLOSE_DEADLINE, async {
        let mut connection = TungsteniteClient::new()
            .connect(handshake_request(&url), ConnectOptions::new())
            .await
            .expect("the session should connect off-runtime");

        let payload = Bytes::from(vec![0u8; STALLED_PAYLOAD]);
        let mut sending = connection.send(Frame::Binary(payload));
        match futures::future::select(sending.as_mut(), arrival).await {
            futures::future::Either::Right((arrival, _)) => arrival.expect("write reached peer"),
            futures::future::Either::Left(_) => panic!("write completed before stall observation"),
        }
        let stalled = rig_core::wasm_compat::timeout(
            Duration::from_millis(200),
            sending.as_mut(),
        )
        .await;
        assert!(
            stalled.is_err(),
            "a peer that is not reading cannot absorb {STALLED_PAYLOAD} bytes: the write should still be in flight"
        );
        drop(sending);

        drop(connection);
        drain.send(()).expect("drain after owner drop");
        }).await.expect("client deadline");
    });

    let (received, closed) = report
        .recv_timeout(CLOSE_DEADLINE * 2)
        .expect("the server thread should report");
    assert!(
        closed,
        "dropping the connection must close the socket even mid-write"
    );
    assert!(
        received < STALLED_PAYLOAD,
        "the abandoned write must be cancelled with the connection, but the server received all \
         {received} bytes of it"
    );
}
