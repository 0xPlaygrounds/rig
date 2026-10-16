//! The websocket connection: a socket the caller polls itself, each operation
//! in the runtime context of [`rig_http::runtime`].

use futures::{SinkExt, StreamExt};
use rig_http::http_client::Result;
use rig_http::wasm_compat::WasmBoxedFuture;
use rig_http::ws_client::{CloseFrame, Frame, WebSocketConnection};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::protocol::CloseFrame as TungsteniteCloseFrame;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream,
    tungstenite::{self, Message},
};

pub(crate) type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// A socket the caller owns and polls. Dropping it closes the transport;
/// a cancelled receive leaves unread frames in the socket.
pub(crate) struct Connection(pub(crate) Socket);

impl WebSocketConnection for Connection {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, Result<()>> {
        let send = async move {
            self.0
                .send(into_message(frame))
                .await
                .map_err(crate::from_tungstenite)
        };
        Box::pin(async move { rig_http::runtime::bind(send)?.await })
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, Result<Option<Frame>>> {
        let recv = async move {
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
        };
        Box::pin(async move { rig_http::runtime::bind(recv)?.await })
    }

    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, Result<()>> {
        let close = async move {
            self.0
                .close(frame.map(into_close_frame))
                .await
                .map_err(crate::from_tungstenite)
        };
        Box::pin(async move { rig_http::runtime::bind(close)?.await })
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
