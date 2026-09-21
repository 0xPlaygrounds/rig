//! Direct and channel-backed websocket connections.
//!
//! Direct connections require the caller's Tokio runtime. Forwarded connections
//! keep socket I/O on the fallback runtime and preserve unread frames across
//! cancelled receives.

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
///
/// The connection owns the actor and aborts it on drop. Cancelling a receive
/// leaves the actor alive and preserves undelivered frames.
pub(crate) struct ForwardedConnection {
    commands: futures::channel::mpsc::Sender<Command>,
    /// Aborts the actor on drop, including during a blocked write that cannot
    /// observe the command channel closing.
    _actor: crate::runtime::OwnedTask<()>,
}

/// The connection actor stopped before accepting or answering a command.
#[derive(Debug, thiserror::Error)]
#[error("the websocket connection task has stopped")]
struct ConnectionTaskGone;

/// Maximum queued inbound frames before socket reads pause for backpressure.
const READ_AHEAD: usize = 256;

async fn run_actor(socket: Socket, mut requests: futures::channel::mpsc::Receiver<Command>) {
    use futures::{FutureExt, select};

    let (mut sink, mut stream) = socket.split();
    let mut inbound: VecDeque<Result<Frame>> = VecDeque::new();
    let mut pending_read: Option<futures::channel::oneshot::Sender<Result<Option<Frame>>>> = None;
    let mut stream_ended = false;

    loop {
        match pending_read.take() {
            Some(reply) if !inbound.is_empty() || stream_ended => {
                let answer = match inbound.pop_front() {
                    Some(Ok(frame)) => Ok(Some(frame)),
                    Some(Err(error)) => Err(error),
                    None => Ok(None),
                };
                // Preserve both frames and errors when the receiver cancels,
                // so the next read observes the original result.
                if let Err(answer) = reply.send(answer) {
                    match answer {
                        Ok(Some(frame)) => inbound.push_front(Ok(frame)),
                        Err(error) => inbound.push_front(Err(error)),
                        Ok(None) => {}
                    }
                }
                continue;
            }
            still_pending => pending_read = still_pending,
        }

        let command = if stream_ended || inbound.len() >= READ_AHEAD {
            // Avoid polling an exhausted stream or reading beyond the buffer bound.
            requests.next().await
        } else {
            select! {
                command = requests.next().fuse() => command,
                message = stream.next().fuse() => {
                    match message {
                        Some(Ok(message)) => {
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

        // Releasing the last handle must also release the socket and actor.
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
    /// connection, or an error if the runtime cannot start.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn spawn(socket: Socket) -> Result<rig_core::ws_client::BoxedWebSocketConnection> {
        // The sequential contract needs one pending command; bounding the queue
        // prevents callers from buffering unaccepted frames.
        let (commands, requests) = futures::channel::mpsc::channel::<Command>(1);
        let actor = crate::runtime::spawn_off_runtime(run_actor(socket, requests))?;
        Ok(Box::new(Self {
            commands,
            _actor: actor,
        }))
    }

    /// Send one command and await its answer, returning an error if the actor stops.
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

/// Convert a tungstenite message to a transport frame, returning `None` for raw
/// frames without a session-level protocol payload.
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
mod tests;
