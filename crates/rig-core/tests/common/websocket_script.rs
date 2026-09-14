//! A scripted in-memory [`WebSocketConnection`] for the OpenAI Responses
//! websocket session tests.
//!
//! These session tests used to need a live local websocket server, a TLS-capable
//! backend and a tokio reactor to assert on protocol behaviour that involves
//! none of those things. Now that the session drives a
//! [`WebSocketConnection`](rig_core::ws_client::WebSocketConnection), the
//! "server" is a queue of frames.
//!
//! The script is expressed in *turns*, which is how the protocol works: each
//! `response.create` the session writes releases the next turn's server frames.
//! Assertions that used to read the payload the server received now read
//! [`Script::sent`].

#![allow(dead_code)]

use rig_core::http_client;
use rig_core::providers::openai::responses_api::websocket::ResponsesWebSocketSession;
use rig_core::test_utils::RecordingHttpClient;
use rig_core::wasm_compat::WasmBoxedFuture;
use rig_core::ws_client::{BoxedWebSocketConnection, CloseFrame, Frame, WebSocketConnection};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The provider client these sessions wrap. The HTTP transport is never
/// exercised — a websocket session only borrows the model for its request
/// mapping — so a recording stub stands in for it.
pub type TestClient = rig_core::providers::openai::Client<RecordingHttpClient>;
pub type TestModel =
    rig_core::providers::openai::responses_api::ResponsesCompletionModel<RecordingHttpClient>;
pub type TestSession = ResponsesWebSocketSession<RecordingHttpClient>;

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
    /// Server frames per turn, released one turn per write: the session opens
    /// every turn with a `response.create`, so the first write releases turn
    /// one, the second write turn two, and so on. Nothing is released before
    /// the first write.
    turns: VecDeque<Vec<Frame>>,
    inbound: VecDeque<Frame>,
    sent: Vec<String>,
    closed: bool,
    drained: WhenDrained,
}

/// A scripted connection, cloneable so a test can inspect what the session
/// wrote after handing the connection to it.
#[derive(Clone, Default)]
pub struct Script(Arc<Mutex<ScriptState>>);

impl Script {
    /// A script whose turns are lists of JSON payloads, one list per
    /// `response.create` the session sends.
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

    /// Every text payload the session has written, in order.
    pub fn sent(&self) -> Vec<String> {
        self.0.lock().expect("script lock").sent.clone()
    }

    /// Whether the session completed a close handshake.
    pub fn closed(&self) -> bool {
        self.0.lock().expect("script lock").closed
    }

    /// The scripted connection handle to hand to a session.
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
            other => panic!("the session only writes text frames, got {other:?}"),
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

/// A client whose HTTP transport is a stub: these tests never send one.
pub fn test_client() -> TestClient {
    rig_core::providers::openai::Client::builder()
        .api_key("test-key")
        .base_url("https://api.openai.com/v1")
        .http_client(RecordingHttpClient::new("{}"))
        .build()
        .expect("client should build")
}

/// The model a session wraps, for building completion requests in tests.
pub fn test_model(client: &TestClient) -> TestModel {
    use rig_core::client::CompletionClient as _;
    client.completion_model("gpt-4o")
}

/// A session over `script`, with no event timeout.
pub fn session(client: &TestClient, script: &Script) -> TestSession {
    session_with_timeout(client, script, None)
}

/// A session over `script` with an explicit event timeout.
pub fn session_with_timeout(
    client: &TestClient,
    script: &Script,
    event_timeout: Option<Duration>,
) -> TestSession {
    ResponsesWebSocketSession::from_connection(
        test_model(client),
        script.connection(),
        event_timeout,
    )
}
