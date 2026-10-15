//! Recording and replay for WebSocket turns. A [`WebSocketCassette`] hands out
//! a [`CassetteSocket`], a [`WebSocketClientExt`] backend: when recording it
//! wraps a live backend and keeps what each turn sent and received; when
//! replaying it serves each message the client sends from the fixture,
//! without a network.
//!
//! A turn is one fixture interaction in the HTTP format: the connection's
//! upgrade (`GET`, its path and recorded headers) with the client's message as
//! the body, answered by `101` with the messages the provider sent until the
//! next client message, as SSE `data:` lines. Replay matches each message as
//! the HTTP replay matches a request, so request snapshots, shape matching and
//! the stored-state rule apply unchanged. It refuses a message sent before the
//! previous turn's messages were read, and a turn left unplayed fails the
//! session when it finishes. Close frames and read errors are not recorded:
//! replay ends where the recorded messages end.
//!
//! ```no_run
//! # use rig_cassette::http::websocket::WebSocketCassette;
//! # async fn test(live: impl rig_core::ws_client::WebSocketClientExt) {
//! let root = std::path::Path::new("fixtures/cassettes");
//! let origin = "https://api.openai.com/v1";
//! let cassette = WebSocketCassette::start(root, "openai", "websocket/turn", origin).await;
//! let backend = cassette.backend(live);
//! // ... connect through `backend` and run the turns ...
//! cassette.finish().await;
//! # }
//! ```

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex};

use axum::body::Bytes;
use axum::http::Method;
use rig_core::http_client::{self, NoBody, Request};
use rig_core::wasm_compat::{WasmBoxedFuture, WasmCompatSend};
use rig_core::ws_client::{
    BoxedWebSocketConnection, CloseFrame, ConnectOptions, Frame, WebSocketClientExt,
    WebSocketConnection,
};
use tokio::sync::Mutex;

use super::{
    CassetteServer, CassetteSpec, DirectHttpRequest, DirectHttpResponse, DirectRecorder,
    IncomingRequest, PanicPayload, ProviderCassette, RecordVia, ReplayMiss, ReplayState,
    decode_body,
};

/// The status every recorded turn answers with: the connection's upgrade.
const SWITCHING_PROTOCOLS: u16 = 101;

/// A WebSocket recording or replay session: a [`ProviderCassette`] whose
/// interactions are WebSocket turns.
pub struct WebSocketCassette {
    cassette: ProviderCassette,
    /// The turns a recording connection saw, written when the session ends.
    turns: Turns,
}

type Turns = Arc<StdMutex<Vec<RecordedTurn>>>;

/// One turn as the recording connection saw it.
struct RecordedTurn {
    uri: String,
    headers: Vec<(String, String)>,
    sent: String,
    received: Vec<String>,
}

impl WebSocketCassette {
    /// Start a session for `spec` beneath `cassette_root`, in the ambient
    /// mode. Panics as [`ProviderCassette::start`] does.
    pub async fn start(
        cassette_root: &Path,
        provider: &'static str,
        spec: impl Into<CassetteSpec>,
        real_base_url: &str,
    ) -> Self {
        Self::over(
            ProviderCassette::start_via(
                RecordVia::Direct,
                cassette_root,
                provider,
                spec,
                real_base_url,
            )
            .await,
        )
    }

    /// The session over `cassette`, which must record directly.
    pub(crate) fn over(cassette: ProviderCassette) -> Self {
        Self {
            cassette,
            turns: Arc::default(),
        }
    }

    /// The underlying session: its clock, its pause and its keys.
    pub fn cassette(&self) -> &ProviderCassette {
        &self.cassette
    }

    /// The backend to connect through: `live` wrapped for the fixture when
    /// recording, the fixture alone when replaying.
    pub fn backend<B: WebSocketClientExt>(&self, live: B) -> CassetteSocket<B> {
        let mode = match &self.cassette.server {
            CassetteServer::Replay(server) => SocketMode::Replay(Arc::clone(&server.state)),
            _ => SocketMode::Record(Arc::clone(&self.turns), self.cassette.direct_recorder()),
        };
        CassetteSocket { live, mode }
    }

    /// Write the scrubbed turns, or check that replay played every turn,
    /// as [`ProviderCassette::finish`] does.
    pub async fn finish(self) {
        self.record_turns().await;
        self.cassette.finish().await;
    }

    /// Finish after a successful test, preserving its panic otherwise, as
    /// [`ProviderCassette::finish_after_test`] does.
    pub async fn finish_after_test(self, test_result: Result<(), PanicPayload>) {
        self.record_turns().await;
        self.cassette.finish_after_test(test_result).await;
    }

    /// Hand every recorded turn to the direct recorder, which scrubs it. The
    /// ledger logged what each turn created as its replies arrived.
    async fn record_turns(&self) {
        let Some(recorder) = self.cassette.direct_recorder() else {
            return;
        };
        let turns = std::mem::take(&mut *lock(&self.turns));
        for turn in turns {
            let body: String = turn.received.iter().map(|message| sse(message)).collect();
            recorder
                .keep_http_interaction(
                    DirectHttpRequest {
                        method: "GET",
                        uri: &turn.uri,
                        headers: turn.headers.iter().map(|(n, v)| (n.as_str(), v.as_str())),
                        body: turn.sent.as_bytes(),
                    },
                    DirectHttpResponse {
                        status: SWITCHING_PROTOCOLS,
                        headers: std::iter::empty::<(&str, &str)>(),
                        body: body.as_bytes(),
                    },
                )
                .await;
        }
    }
}

/// `message` as one SSE event.
fn sse(message: &str) -> String {
    let mut event: String = message
        .split('\n')
        .map(|line| format!("data: {line}\n"))
        .collect();
    event.push('\n');
    event
}

/// The messages an SSE body of [`sse`] events carries.
fn messages(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter(|event| !event.trim().is_empty())
        .map(|event| {
            event
                .lines()
                .filter_map(|line| line.strip_prefix("data: ").or(line.strip_prefix("data:")))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .collect()
}

fn lock<T>(mutex: &StdMutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A [`WebSocketClientExt`] backend over a [`WebSocketCassette`].
#[derive(Clone)]
pub struct CassetteSocket<B> {
    live: B,
    mode: SocketMode,
}

#[derive(Clone)]
enum SocketMode {
    Record(Turns, Option<DirectRecorder>),
    Replay(Arc<Mutex<ReplayState>>),
}

impl<B: WebSocketClientExt> WebSocketClientExt for CassetteSocket<B> {
    fn connect(
        &self,
        request: Request<NoBody>,
        options: ConnectOptions,
    ) -> impl Future<Output = http_client::Result<BoxedWebSocketConnection>> + WasmCompatSend {
        let (live, mode) = (self.live.clone(), self.mode.clone());
        async move {
            match mode {
                SocketMode::Record(turns, recorder) => {
                    let uri = request.uri().to_string();
                    let headers = request
                        .headers()
                        .iter()
                        .filter_map(|(name, value)| {
                            Some((name.as_str().to_owned(), value.to_str().ok()?.to_owned()))
                        })
                        .collect();
                    let inner = live.connect(request, options).await?;
                    Ok(Box::new(Recording {
                        inner,
                        uri,
                        headers,
                        turns,
                        turn: None,
                        recorder,
                    }) as BoxedWebSocketConnection)
                }
                SocketMode::Replay(state) => {
                    let uri = request
                        .uri()
                        .path_and_query()
                        .map_or("/", |path| path.as_str())
                        .parse()
                        .map_err(http_client::Error::instance)?;
                    Ok(Box::new(Replaying {
                        state,
                        uri,
                        headers: request.headers().clone(),
                        inbound: VecDeque::new(),
                    }) as BoxedWebSocketConnection)
                }
            }
        }
    }
}

/// A live connection whose turns are kept for the fixture.
struct Recording {
    inner: BoxedWebSocketConnection,
    uri: String,
    headers: Vec<(String, String)>,
    turns: Turns,
    /// This connection's turn in flight, by its index in `turns`.
    turn: Option<usize>,
    recorder: Option<DirectRecorder>,
}

impl WebSocketConnection for Recording {
    /// A message opens a turn once the provider accepted it: a send that
    /// failed leaves no turn in the fixture.
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        Box::pin(async move {
            let sent = match &frame {
                Frame::Text(text) => Some(text.clone()),
                _ => None,
            };
            self.inner.send(frame).await?;
            if let Some(sent) = sent {
                let mut turns = lock(&self.turns);
                self.turn = Some(turns.len());
                turns.push(RecordedTurn {
                    uri: self.uri.clone(),
                    headers: self.headers.clone(),
                    sent,
                    received: Vec::new(),
                });
            }
            Ok(())
        })
    }

    fn recv(&mut self) -> WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        Box::pin(async move {
            let received = self.inner.recv().await;
            // A binary message the client reads as UTF-8 text is kept as text.
            let text = match &received {
                Ok(Some(Frame::Text(text))) => Some(text.clone()),
                Ok(Some(Frame::Binary(bytes))) => String::from_utf8(bytes.to_vec()).ok(),
                _ => None,
            };
            // A message before the first send belongs to no turn; the
            // Responses protocol sends none.
            if let Some(text) = text
                && let Some(turn) = self.turn
                && let Some(turn) = lock(&self.turns).get_mut(turn)
            {
                // The ledger sees what a turn created before the caller
                // does: the response the turn's first event names.
                if let Some(recorder) = &self.recorder
                    && turn.received.is_empty()
                {
                    let body = sse(&text);
                    recorder.log_created(
                        "GET",
                        &turn.uri,
                        turn.sent.as_bytes(),
                        SWITCHING_PROTOCOLS,
                        body.as_bytes(),
                    );
                }
                turn.received.push(text);
            }
            received
        })
    }

    fn close(&mut self, frame: Option<CloseFrame>) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        self.inner.close(frame)
    }
}

/// A connection that plays the fixture's turns back.
struct Replaying {
    state: Arc<Mutex<ReplayState>>,
    uri: axum::http::Uri,
    headers: axum::http::HeaderMap,
    /// The messages of the last turn not yet read.
    inbound: VecDeque<String>,
}

impl Replaying {
    /// Serve `text` as the next turn's message and queue its replies.
    async fn turn(&mut self, text: String) -> Result<(), String> {
        let mut state = self.state.lock().await;
        if !self.inbound.is_empty() {
            let diagnostic = format!(
                "a websocket turn was sent with {} message(s) of the previous turn unread",
                self.inbound.len()
            );
            state.misses.push(ReplayMiss {
                diagnostic: diagnostic.clone(),
            });
            return Err(diagnostic);
        }
        let request = IncomingRequest {
            method: Method::GET,
            uri: self.uri.clone(),
            headers: self.headers.clone(),
            body: Bytes::from(text),
        };
        let index = state.serve(&request)?;
        let body = state
            .interactions
            .get(index)
            .and_then(|interaction| {
                let response = &interaction.then;
                decode_body(response.body.as_deref()?, response.body_encoding).ok()
            })
            .unwrap_or_default();
        self.inbound
            .extend(messages(&String::from_utf8_lossy(&body)));
        Ok(())
    }
}

impl WebSocketConnection for Replaying {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        Box::pin(async move {
            match frame {
                Frame::Text(text) => self
                    .turn(text)
                    .await
                    .map_err(|miss| http_client::Error::instance(std::io::Error::other(miss))),
                _ => Ok(()),
            }
        })
    }

    /// Each read yields once before it answers, as a network read would, so
    /// turns sent at once contend for the connection. The recording ends
    /// where the provider stopped sending: past it, the peer is gone.
    fn recv(&mut self) -> WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        Box::pin(async move {
            tokio::task::yield_now().await;
            Ok(self.inbound.pop_front().map(Frame::Text))
        })
    }

    fn close(
        &mut self,
        _frame: Option<CloseFrame>,
    ) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        Box::pin(std::future::ready(Ok(())))
    }
}

#[cfg(test)]
mod tests;
