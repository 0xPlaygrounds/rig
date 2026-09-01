//! The two ways this backend hands a socket to a session.
//!
//! [`DirectConnection`] is the ordinary one: the caller is already inside a
//! tokio runtime, so the session's futures can poll the socket themselves.
//!
//! [`ForwardedConnection`] is for a caller with no tokio runtime. The socket
//! moves onto the fallback runtime as a small actor and the connection becomes
//! a pair of `futures` channels, so nothing the caller polls ever touches
//! tokio. It reads and serves commands concurrently rather than one at a time;
//! [`run_actor`] documents why that is required and not merely faster.

use futures::{SinkExt, StreamExt};
use rig_core::http_client::{Error, Result};
use rig_core::wasm_compat::WasmBoxedFuture;
use rig_core::ws_client::{CloseFrame, Frame, WebSocketConnection};
use std::collections::VecDeque;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::protocol::CloseFrame as TungsteniteCloseFrame;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream,
    tungstenite::{self, Message},
};

pub(crate) type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// A socket polled by the caller's own tokio runtime.
pub(crate) struct DirectConnection(Socket);

impl DirectConnection {
    pub(crate) fn new(socket: Socket) -> Self {
        Self(socket)
    }
}

impl WebSocketConnection for DirectConnection {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
        Box::pin(async move {
            self.0
                .send(into_message(frame))
                .await
                .map_err(crate::from_tungstenite)
        })
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
        Box::pin(async move {
            loop {
                match self.0.next().await {
                    // A raw frame carries no protocol payload; skip it rather
                    // than hand a session bytes it will try to parse as JSON.
                    Some(Ok(message)) => match from_message(message) {
                        Some(frame) => return Ok(Some(frame)),
                        None => continue,
                    },
                    Some(Err(error)) => return Err(crate::from_tungstenite(error)),
                    None => return Ok(None),
                }
            }
        })
    }

    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
        Box::pin(async move {
            self.0
                .close(frame.map(into_close_frame))
                .await
                .map_err(crate::from_tungstenite)
        })
    }
}

/// One request to the connection actor, with the channel its answer goes back
/// on.
enum Command {
    Send(Frame, futures::channel::oneshot::Sender<Result<()>>),
    Recv(futures::channel::oneshot::Sender<Result<Option<Frame>>>),
    Close(
        Option<CloseFrame>,
        futures::channel::oneshot::Sender<Result<()>>,
    ),
}

/// A socket living on the fallback runtime, reached over channels.
pub(crate) struct ForwardedConnection {
    commands: futures::channel::mpsc::Sender<Command>,
}

/// The actor's end of the connection went away — it only stops when the
/// fallback runtime is torn down, which means the process is on its way out.
#[derive(Debug, thiserror::Error)]
#[error("the websocket connection task has stopped")]
struct ConnectionTaskGone;

/// The actor that owns the socket on the fallback runtime.
///
/// It reads and serves commands *concurrently*, over a split socket, rather
/// than executing one command to completion at a time. That is not an
/// optimization — a serial actor deadlocks:
///
/// - A read command would park the actor in `stream.next()`. When the session's
///   event timeout fires it drops its `recv()` future, but the actor stays
///   parked, so the `close()` that follows a timeout would wait forever for an
///   actor that is waiting for a frame that is never coming.
/// - A caller who cancels a read (their own `select!`, their own timeout) would
///   lose the frame the actor had already taken off the socket, and the *next*
///   read would hang waiting for a frame that had already arrived.
///
/// So inbound frames are buffered as they arrive and a read is answered from
/// that buffer; a frame whose caller has gone away goes back on the front of it.
/// Reading continuously also keeps tungstenite's automatic pong replies flowing,
/// which only happen when the stream is polled.
/// How many frames the actor will read ahead of the session.
///
/// Reading continuously is what makes a cancelled read safe and keeps
/// tungstenite's pong replies flowing, but an unbounded buffer would let a host
/// that reads slowly (rendering, disk, its own rate limit) accumulate a whole
/// streamed response in memory while TCP happily kept delivering. Past this
/// depth the actor stops draining the socket, which is where the backpressure
/// the pre-split code got for free comes back.
const READ_AHEAD: usize = 256;

async fn run_actor(socket: Socket, mut requests: futures::channel::mpsc::Receiver<Command>) {
    use futures::{FutureExt, select};

    let (mut sink, mut stream) = socket.split();
    let mut inbound: VecDeque<Result<Frame>> = VecDeque::new();
    let mut pending_read: Option<futures::channel::oneshot::Sender<Result<Option<Frame>>>> = None;
    let mut stream_ended = false;

    loop {
        // Answer an outstanding read as soon as the buffer (or the end of the
        // stream) can answer it.
        match pending_read.take() {
            Some(reply) if !inbound.is_empty() || stream_ended => {
                let answer = match inbound.pop_front() {
                    Some(Ok(frame)) => Ok(Some(frame)),
                    Some(Err(error)) => Err(error),
                    None => Ok(None),
                };
                // A caller that cancelled its read must not cost us what we
                // took off the socket — an error as much as a frame: a lost
                // protocol failure resurfaces later as a bare "connection
                // closed before the turn finished", hiding the real cause.
                if let Err(answer) = reply.send(answer) {
                    match answer {
                        Ok(Some(frame)) => inbound.push_front(Ok(frame)),
                        Err(error) => inbound.push_front(Err(error)),
                        // End of stream: `stream_ended` already records it.
                        Ok(None) => {}
                    }
                }
                continue;
            }
            // Nothing to answer it with yet (or nothing outstanding): keep it.
            still_pending => pending_read = still_pending,
        }

        let command = if stream_ended || inbound.len() >= READ_AHEAD {
            // Either no further frames can arrive, or the reader is far enough
            // behind that we stop taking them off the socket; in both cases
            // only a command can make progress, and selecting on the stream
            // would spin (or read ahead without bound).
            requests.next().await
        } else {
            select! {
                command = requests.next().fuse() => command,
                message = stream.next().fuse() => {
                    match message {
                        Some(Ok(message)) => {
                            // A raw frame carries no protocol payload; skip it
                            // rather than hand a session bytes it will try to
                            // parse as JSON.
                            if let Some(frame) = from_message(message) {
                                inbound.push_back(Ok(frame));
                            }
                        }
                        Some(Err(error)) => inbound.push_back(Err(crate::from_tungstenite(error))),
                        None => stream_ended = true,
                    }
                    continue;
                }
            }
        };

        // The connection handle was dropped: nothing can reach us again, so
        // drop the socket rather than leaving the task (and the connection)
        // alive for the life of the process.
        let Some(command) = command else {
            return;
        };

        match command {
            Command::Send(frame, reply) => {
                let result = sink
                    .send(into_message(frame))
                    .await
                    .map_err(crate::from_tungstenite);
                let _ = reply.send(result);
            }
            // The sequential contract means there is at most one outstanding
            // read; a second one supersedes a caller that has gone away.
            Command::Recv(reply) => pending_read = Some(reply),
            Command::Close(frame, reply) => {
                let mut result = sink
                    .send(Message::Close(frame.map(into_close_frame)))
                    .await
                    .map_err(crate::from_tungstenite);
                if let Err(error) = SinkExt::close(&mut sink).await {
                    result = result.and(Err(crate::from_tungstenite(error)));
                }
                let _ = reply.send(result);
                return;
            }
        }
    }
}

impl ForwardedConnection {
    /// Move `socket` onto the fallback runtime and return the channel-backed
    /// connection.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn spawn(socket: Socket) -> Result<rig_core::ws_client::BoxedWebSocketConnection> {
        // A depth of one is enough: the contract is one outstanding command at
        // a time, and a bound keeps a runaway caller from queueing frames the
        // socket has not accepted.
        let (commands, requests) = futures::channel::mpsc::channel::<Command>(1);
        crate::runtime::spawn_off_runtime(run_actor(socket, requests))?;
        Ok(Box::new(Self { commands }))
    }

    /// Send one command and await its answer.
    async fn request<T, F>(&mut self, command: F) -> Result<T>
    where
        F: FnOnce(futures::channel::oneshot::Sender<Result<T>>) -> Command,
    {
        let (reply, answer) = futures::channel::oneshot::channel();
        self.commands
            .send(command(reply))
            .await
            .map_err(|_| Error::instance(ConnectionTaskGone))?;
        answer
            .await
            .map_err(|_| Error::instance(ConnectionTaskGone))?
    }
}

impl WebSocketConnection for ForwardedConnection {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
        Box::pin(async move { self.request(|reply| Command::Send(frame, reply)).await })
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
        Box::pin(async move { self.request(Command::Recv).await })
    }

    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
        Box::pin(async move { self.request(|reply| Command::Close(frame, reply)).await })
    }
}

fn into_message(frame: Frame) -> Message {
    match frame {
        Frame::Text(text) => Message::text(text),
        Frame::Binary(bytes) => Message::binary(bytes),
        Frame::Ping(bytes) => Message::Ping(bytes),
        Frame::Pong(bytes) => Message::Pong(bytes),
        Frame::Close(frame) => Message::Close(frame.map(into_close_frame)),
    }
}

/// Lower one tungstenite message onto the transport-agnostic frame.
///
/// `None` is a message with no protocol payload, which the caller skips.
/// Today that is only `Message::Frame`, which tungstenite produces on the write
/// side and never from a read — but the contract has no variant for a raw frame
/// (no session could act on one), and skipping it is what the pre-split code
/// did. Mapping it onto `Binary` instead would hand a session bytes it would
/// try to parse as JSON, turning an unexpected control frame into a fatal
/// decode error.
fn from_message(message: Message) -> Option<Frame> {
    Some(match message {
        Message::Text(text) => Frame::Text(text.to_string()),
        Message::Binary(bytes) => Frame::Binary(bytes),
        Message::Ping(bytes) => Frame::Ping(bytes),
        Message::Pong(bytes) => Frame::Pong(bytes),
        Message::Close(frame) => Frame::Close(frame.map(|frame| CloseFrame {
            code: frame.code.into(),
            reason: frame.reason.to_string(),
        })),
        Message::Frame(_) => return None,
    })
}

fn into_close_frame(frame: CloseFrame) -> TungsteniteCloseFrame {
    TungsteniteCloseFrame {
        code: tungstenite::protocol::frame::coding::CloseCode::from(frame.code),
        reason: frame.reason.into(),
    }
}

#[cfg(test)]
mod tests {
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
}
