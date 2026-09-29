//! A scripted in-memory [`WebSocketConnection`] for the OpenAI Responses
//! websocket transport tests: the "server" is a queue of frames.
//!
//! The script is expressed in *turns*, which is how the protocol works: each
//! `response.create` the transport writes releases the next turn's server
//! frames. [`Script::sent`] reads what the transport wrote.

#![allow(dead_code)]

use rig_core::driver::Model;
use rig_core::http_client;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::test_utils::RecordingHttpClient;
use rig_core::wasm_compat::WasmBoxedFuture;
use rig_core::ws_client::{BoxedWebSocketConnection, CloseFrame, Frame, WebSocketConnection};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// The HTTP model whose wire the websocket wire wraps. Its HTTP transport
/// is never exercised, so a recording stub stands in for it.
pub type TestClient = Model<Responses>;

/// What a scripted connection does once its scripted frames run out.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum WhenDrained {
    /// End the stream, as a peer that hung up would.
    #[default]
    EndStream,
    /// Never resolve, so an event timeout is the only way out. This is the
    /// in-memory equivalent of a server that accepted the turn and went quiet.
    Stall,
}

#[derive(Default)]
struct ScriptState {
    /// Server frames per turn, released one turn per write: the transport opens
    /// every turn with a `response.create`, so the first write releases turn
    /// one, the second write turn two, and so on. Nothing is released before
    /// the first write.
    turns: VecDeque<Vec<Frame>>,
    inbound: VecDeque<Frame>,
    sent: Vec<String>,
    closed: bool,
    drained: WhenDrained,
}

/// A scripted connection, cloneable so a test can inspect what the transport
/// wrote after handing the connection to it.
#[derive(Clone, Default)]
pub struct Script(Arc<Mutex<ScriptState>>);

impl Script {
    /// A script whose turns are lists of JSON payloads, one list per
    /// `response.create` the transport sends.
    pub fn turns<I, J>(turns: I) -> Self
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = String>,
    {
        let state = ScriptState {
            turns: turns
                .into_iter()
                .map(|turn| turn.into_iter().map(Frame::Text).collect())
                .collect(),
            ..ScriptState::default()
        };
        Self(Arc::new(Mutex::new(state)))
    }

    /// A single-turn script.
    pub fn turn<I: IntoIterator<Item = String>>(frames: I) -> Self {
        Self::turns([frames])
    }

    /// A script that goes quiet instead of ending the stream when it runs out.
    #[must_use]
    pub fn stalling(self) -> Self {
        self.0.lock().expect("script lock").drained = WhenDrained::Stall;
        self
    }

    /// Every text payload the transport has written, in order.
    pub fn sent(&self) -> Vec<String> {
        self.0.lock().expect("script lock").sent.clone()
    }

    /// Whether the transport completed a close handshake.
    pub fn closed(&self) -> bool {
        self.0.lock().expect("script lock").closed
    }

    /// The scripted connection handle to hand to a transport.
    pub fn connection(&self) -> BoxedWebSocketConnection {
        Box::new(ScriptedConnection(self.clone()))
    }
}

struct ScriptedConnection(Script);

impl WebSocketConnection for ScriptedConnection {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        let mut state = self.0.0.lock().expect("script lock");
        match frame {
            Frame::Text(text) => state.sent.push(text),
            other => panic!("the transport only writes text frames, got {other:?}"),
        }
        // The write is the cue: a real endpoint answers a `response.create`
        // with that turn's events.
        if let Some(turn) = state.turns.pop_front() {
            state.inbound.extend(turn);
        }
        Box::pin(std::future::ready(Ok(())))
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        let next = {
            let mut state = self.0.0.lock().expect("script lock");
            match state.inbound.pop_front() {
                Some(frame) => Some(Some(frame)),
                None => match state.drained {
                    WhenDrained::EndStream => Some(None),
                    WhenDrained::Stall => None,
                },
            }
        };
        match next {
            Some(frame) => Box::pin(std::future::ready(Ok(frame))),
            None => Box::pin(std::future::pending()),
        }
    }

    fn close(
        &mut self,
        _frame: Option<CloseFrame>,
    ) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        self.0.0.lock().expect("script lock").closed = true;
        Box::pin(std::future::ready(Ok(())))
    }
}

/// A model whose HTTP transport is a stub: these tests never send over it.
pub fn test_client() -> TestClient {
    OpenAIConfig::new("test-key")
        .connect(RecordingHttpClient::new("{}"))
        .responses("gpt-4o")
}
