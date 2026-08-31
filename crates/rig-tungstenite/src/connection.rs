//! The two ways this backend hands a socket to a session.
//!
//! [`DirectConnection`] is the ordinary one: the caller is already inside a
//! tokio runtime, so the session's futures can poll the socket themselves.
//!
//! [`ForwardedConnection`] is for a caller with no tokio runtime. The socket
//! moves onto the fallback runtime as a small actor and the connection becomes
//! a pair of `futures` channels, so nothing the caller polls ever touches
//! tokio. The actor is strictly request/response — one command at a time,
//! answered on a oneshot — which is exactly the contract
//! [`WebSocketConnection`] states, so no buffering or select loop is needed.

use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use rig_core::http_client::{Error, Result};
use rig_core::wasm_compat::WasmBoxedFuture;
use rig_core::ws_client::{CloseFrame, Frame, WebSocketConnection};
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
            match self.0.next().await {
                Some(Ok(message)) => Ok(Some(from_message(message))),
                Some(Err(error)) => Err(crate::from_tungstenite(error)),
                None => Ok(None),
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

impl ForwardedConnection {
    /// Move `socket` onto the fallback runtime and return the channel-backed
    /// connection.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn spawn(socket: Socket) -> Result<rig_core::ws_client::BoxedWebSocketConnection> {
        // A depth of one is enough: the contract is one outstanding command at
        // a time, and a bound keeps a runaway caller from queueing frames the
        // socket has not accepted.
        let (commands, mut requests) = futures::channel::mpsc::channel::<Command>(1);

        crate::runtime::spawn_off_runtime(async move {
            let mut socket = socket;
            while let Some(command) = requests.next().await {
                match command {
                    Command::Send(frame, reply) => {
                        let result = socket
                            .send(into_message(frame))
                            .await
                            .map_err(crate::from_tungstenite);
                        // A dropped receiver means the caller lost interest;
                        // the socket stays usable for the next command.
                        let _ = reply.send(result);
                    }
                    Command::Recv(reply) => {
                        let result = match socket.next().await {
                            Some(Ok(message)) => Ok(Some(from_message(message))),
                            Some(Err(error)) => Err(crate::from_tungstenite(error)),
                            None => Ok(None),
                        };
                        let _ = reply.send(result);
                    }
                    Command::Close(frame, reply) => {
                        let result = socket
                            .close(frame.map(into_close_frame))
                            .await
                            .map_err(crate::from_tungstenite);
                        let _ = reply.send(result);
                        return;
                    }
                }
            }
        })?;

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

fn from_message(message: Message) -> Frame {
    match message {
        Message::Text(text) => Frame::Text(text.to_string()),
        Message::Binary(bytes) => Frame::Binary(bytes),
        Message::Ping(bytes) => Frame::Ping(bytes),
        Message::Pong(bytes) => Frame::Pong(bytes),
        Message::Close(frame) => Frame::Close(frame.map(|frame| CloseFrame {
            code: frame.code.into(),
            reason: frame.reason.to_string(),
        })),
        // A raw frame never surfaces from a read: tungstenite only produces it
        // on the write side. Modeling it in `Frame` would put a variant in
        // rig-core's contract that no session can act on.
        Message::Frame(frame) => Frame::Binary(Bytes::from(frame.into_payload().to_vec())),
    }
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
                frame,
                "frame should round-trip"
            );
        }
    }
}
