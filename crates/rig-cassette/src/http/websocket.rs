//! Recording and replay for WebSocket turns. A [`WebSocketCassette`] hands out
//! a [`CassetteSocket`], a [`WebSocketClientExt`] backend: when recording it
//! wraps a live backend and keeps what each turn sent and received; when
//! replaying it checks what the client sends against the fixture and answers
//! with the recorded events, without a network.
//!
//! A turn is one fixture interaction, in the same format as HTTP exchanges:
//! the request is the connection's upgrade (path and scrubbed headers) with
//! the client's message as its body, and the response is `101` with the text
//! the provider sent until the next client message, as `data:` lines. Replay
//! plays turns strictly in order: a message that is not the next turn's, or
//! one sent before the previous turn was read, is refused, and a turn left
//! unplayed or unread fails the replay when the session finishes.
//!
//! ```no_run
//! # use rig_cassette::http::websocket::WebSocketCassette;
//! # async fn test(live: impl rig_core::ws_client::WebSocketClientExt) {
//! let root = std::path::Path::new("fixtures/cassettes");
//! let cassette =
//!     WebSocketCassette::start(root, "openai", "websocket/turn", "https://api.openai.com/v1").await;
//! let backend = cassette.backend(live);
//! // ... connect through `backend` and run the turns ...
//! cassette.finish().await;
//! # }
//! ```

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

use rig_core::http_client::{self, NoBody, Request};
use rig_core::wasm_compat::{WasmBoxedFuture, WasmCompatSend};
use rig_core::ws_client::{
    BoxedWebSocketConnection, CloseFrame, ConnectOptions, Frame, WebSocketClientExt,
    WebSocketConnection,
};

use super::{
    CassetteMode, CassettePolicy, CassetteScrubber, CassetteSpec, DirectHttpRequest,
    DirectHttpResponse, PanicPayload, ProviderCassette, RecordVia, canonical_json, cassette_path,
    parse_cassette_interactions,
};

/// The status every recorded turn answers with: the connection's upgrade.
const SWITCHING_PROTOCOLS: u16 = 101;

/// A WebSocket recording or replay session with an explicitly located
/// fixture.
pub struct WebSocketCassette {
    session: Session,
    cassette_path: PathBuf,
}

enum Session {
    Recording {
        cassette: ProviderCassette,
        turns: Arc<StdMutex<Vec<RecordedTurn>>>,
    },
    Replay(Arc<StdMutex<Replay>>),
}

/// One turn as the recording connection saw it.
struct RecordedTurn {
    uri: String,
    headers: Vec<(String, String)>,
    sent: String,
    received: Vec<String>,
}

struct Replay {
    policy: CassettePolicy,
    turns: Vec<ReplayTurn>,
    /// The index of the next turn to match.
    next: usize,
    inbound: VecDeque<String>,
    misses: Vec<String>,
    /// `finish` checked the replay; the drop guard has nothing to add.
    checked: bool,
}

impl Drop for Replay {
    fn drop(&mut self) {
        if !self.checked && !std::thread::panicking() {
            panic!("a websocket cassette replay was dropped without `finish`");
        }
    }
}

struct ReplayTurn {
    path: String,
    sent: Option<String>,
    received: Vec<String>,
}

impl WebSocketCassette {
    /// Start a session for `scenario` beneath `cassette_root`, in the ambient
    /// [`CassetteMode`]. Panics for a missing or malformed replay fixture.
    /// Replay is always in order: [`CassetteSpec::unordered`] does not apply.
    /// A turn answers `101`, so the account-failure classification of error
    /// statuses never sees one.
    pub async fn start(
        cassette_root: &Path,
        provider: &'static str,
        spec: impl Into<CassetteSpec>,
        real_base_url: &str,
    ) -> Self {
        let spec = spec.into();
        let cassette_path = cassette_path(cassette_root, provider, spec.scenario());
        Self::start_at(
            provider,
            spec,
            real_base_url,
            CassetteMode::current(),
            cassette_path,
            super::attempt_root(),
        )
        .await
    }

    /// [`Self::start`] with an explicit mode, fixture path and attempt root.
    pub(crate) async fn start_at(
        provider: &'static str,
        spec: CassetteSpec,
        real_base_url: &str,
        mode: CassetteMode,
        cassette_path: PathBuf,
        attempt_root: PathBuf,
    ) -> Self {
        let policy = CassettePolicy::for_scenario(provider, spec.scenario(), spec.replay_matching);
        let session = match mode {
            CassetteMode::Record => Session::Recording {
                cassette: ProviderCassette::start_with_attempts(
                    RecordVia::Direct,
                    provider,
                    spec,
                    real_base_url,
                    mode,
                    cassette_path.clone(),
                    attempt_root,
                )
                .await,
                turns: Arc::default(),
            },
            CassetteMode::Replay => {
                let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|error| {
                    panic!(
                        "missing websocket cassette {}; run with RIG_PROVIDER_TEST_MODE=record and \
                         the real API key to create it: {error}",
                        cassette_path.display()
                    )
                });
                let turns = parse_cassette_interactions(&cassette_path, &contents)
                    .into_iter()
                    .map(|interaction| ReplayTurn {
                        path: interaction.when.path,
                        sent: interaction.when.body,
                        received: interaction
                            .then
                            .body
                            .unwrap_or_default()
                            .lines()
                            .filter_map(|line| line.strip_prefix("data: ").map(str::to_owned))
                            .collect(),
                    })
                    .collect();
                Session::Replay(Arc::new(StdMutex::new(Replay {
                    policy,
                    turns,
                    next: 0,
                    inbound: VecDeque::new(),
                    misses: Vec::new(),
                    checked: false,
                })))
            }
        };
        Self {
            session,
            cassette_path,
        }
    }

    /// Whether this session records.
    pub fn records(&self) -> bool {
        matches!(self.session, Session::Recording { .. })
    }

    /// The key to connect with: the real one from `env_name` when recording,
    /// a placeholder when replaying.
    pub fn api_key(&self, env_name: &str) -> String {
        match &self.session {
            Session::Recording { cassette, .. } => cassette.api_key(env_name),
            Session::Replay(_) => super::DUMMY_API_KEY.to_string(),
        }
    }

    /// The backend to connect through: `live` when recording, the fixture
    /// when replaying.
    pub fn backend<B: WebSocketClientExt>(&self, live: B) -> CassetteSocket<B> {
        let mode = match &self.session {
            Session::Recording { turns, .. } => SocketMode::Record(Arc::clone(turns)),
            Session::Replay(replay) => SocketMode::Replay(Arc::clone(replay)),
        };
        CassetteSocket { live, mode }
    }

    /// Write the scrubbed turns, or assert that replay used every recorded
    /// turn and matched every sent message. Panics otherwise.
    pub async fn finish(self) {
        let Self {
            session,
            cassette_path,
        } = self;
        match session {
            Session::Replay(replay) => {
                let mut replay = replay
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                replay.checked = true;
                if let Some(message) = replay.failure(&cassette_path) {
                    panic!("{message}");
                }
            }
            Session::Recording { cassette, turns } => {
                record_turns(&cassette, &turns).await;
                cassette.finish().await;
            }
        }
    }

    /// Finish after a successful test, preserving its panic otherwise. A
    /// failed recording is kept under the attempt root, never over the
    /// fixture.
    pub async fn finish_after_test(self, test_result: Result<(), PanicPayload>) {
        let Self {
            session,
            cassette_path,
        } = self;
        match session {
            Session::Replay(replay) => match test_result {
                Ok(()) => {
                    Self {
                        session: Session::Replay(replay),
                        cassette_path,
                    }
                    .finish()
                    .await
                }
                Err(payload) => {
                    // The test's own panic is the report.
                    replay
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .checked = true;
                    std::panic::resume_unwind(payload)
                }
            },
            Session::Recording { cassette, turns } => {
                record_turns(&cassette, &turns).await;
                cassette.finish_after_test(test_result).await;
            }
        }
    }
}

/// Hand every recorded turn to the direct recorder, which scrubs it.
async fn record_turns(cassette: &ProviderCassette, turns: &StdMutex<Vec<RecordedTurn>>) {
    let Some(recorder) = cassette.direct_recorder() else {
        return;
    };
    let turns = std::mem::take(
        &mut *turns
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    for turn in turns {
        let body: String = turn
            .received
            .iter()
            .map(|message| format!("data: {message}\n\n"))
            .collect();
        recorder
            .record_http_interaction(
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

impl Replay {
    fn failure(&self, cassette_path: &Path) -> Option<String> {
        let mut failures = Vec::new();
        if self.next < self.turns.len() {
            failures.push(format!(
                "left {} recorded turn(s) unplayed, from turn {}",
                self.turns.len() - self.next,
                self.next
            ));
        }
        if !self.inbound.is_empty() {
            failures.push(format!(
                "left {} recorded message(s) of turn {} unread",
                self.inbound.len(),
                self.next.saturating_sub(1)
            ));
        }
        if !self.misses.is_empty() {
            failures.push(format!(
                "received unexpected message(s):\n{}",
                self.misses.join("\n")
            ));
        }
        (!failures.is_empty()).then(|| {
            format!(
                "websocket cassette replay failed for {}:\n{}",
                cassette_path.display(),
                failures.join("\n\n")
            )
        })
    }

    /// Match a sent message against the next recorded turn and queue its
    /// replies.
    fn send(&mut self, text: &str) -> Result<(), String> {
        if !self.inbound.is_empty() {
            return Err(format!(
                "turn {} was sent with {} message(s) of the previous turn unread",
                self.next,
                self.inbound.len()
            ));
        }
        let Some(turn) = self.turns.get(self.next) else {
            return Err(format!("no recorded turn left for {text}"));
        };
        let mut scrubber = CassetteScrubber::new(self.policy);
        let sent = canonical_json(&scrubber.scrub_body(text));
        let recorded = turn.sent.as_deref().and_then(canonical_json);
        if sent.is_none() || sent != recorded {
            return Err(format!(
                "turn {} sent {}\nrecorded {}",
                self.next,
                scrubber.scrub_body(text),
                turn.sent.as_deref().unwrap_or("<nothing>")
            ));
        }
        self.inbound.extend(turn.received.iter().cloned());
        self.next += 1;
        Ok(())
    }
}

/// A [`WebSocketClientExt`] backend over a [`WebSocketCassette`].
#[derive(Clone)]
pub struct CassetteSocket<B> {
    live: B,
    mode: SocketMode,
}

#[derive(Clone)]
enum SocketMode {
    Record(Arc<StdMutex<Vec<RecordedTurn>>>),
    Replay(Arc<StdMutex<Replay>>),
}

impl<B: WebSocketClientExt> WebSocketClientExt for CassetteSocket<B> {
    fn connect(
        &self,
        request: Request<NoBody>,
        options: ConnectOptions,
    ) -> impl Future<Output = http_client::Result<BoxedWebSocketConnection>> + WasmCompatSend {
        let uri = request.uri().to_string();
        let path = request.uri().path().to_owned();
        let headers = request
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_owned(), value.to_owned()))
            })
            .collect::<Vec<_>>();
        let (live, mode) = (self.live.clone(), self.mode.clone());
        async move {
            match mode {
                SocketMode::Record(turns) => {
                    let connection = live.connect(request, options).await?;
                    Ok(Box::new(Recording {
                        inner: connection,
                        uri,
                        headers,
                        turns,
                    }) as BoxedWebSocketConnection)
                }
                SocketMode::Replay(replay) => {
                    {
                        let mut state = replay
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if let Some(turn) = state.turns.get(state.next)
                            && turn.path != path
                        {
                            let miss = format!("connected to {path}, recorded {}", turn.path);
                            state.misses.push(miss);
                        }
                        // The handshake authenticates the connection: it must
                        // carry what an HTTP request to the provider must.
                        for required in state.policy.required_request_headers() {
                            let present = headers.iter().any(|(name, value)| {
                                name.eq_ignore_ascii_case(required) && !value.trim().is_empty()
                            });
                            if !present {
                                state
                                    .misses
                                    .push(format!("the handshake carried no {required} header"));
                            }
                        }
                    }
                    Ok(Box::new(Replaying(replay)) as BoxedWebSocketConnection)
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
    turns: Arc<StdMutex<Vec<RecordedTurn>>>,
}

impl Recording {
    fn turns(&self) -> std::sync::MutexGuard<'_, Vec<RecordedTurn>> {
        self.turns
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl WebSocketConnection for Recording {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        if let Frame::Text(text) = &frame {
            let turn = RecordedTurn {
                uri: self.uri.clone(),
                headers: self.headers.clone(),
                sent: text.clone(),
                received: Vec::new(),
            };
            self.turns().push(turn);
        }
        self.inner.send(frame)
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
            // A message before the first send belongs to no turn and is not
            // kept; the Responses protocol sends none.
            if let Some(text) = text
                && let Some(turn) = self.turns().last_mut()
            {
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
struct Replaying(Arc<StdMutex<Replay>>);

impl Replaying {
    fn replay(&self) -> std::sync::MutexGuard<'_, Replay> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl WebSocketConnection for Replaying {
    fn send(&mut self, frame: Frame) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        let sent = match frame {
            Frame::Text(text) => {
                let mut replay = self.replay();
                replay.send(&text).map_err(|miss| {
                    replay.misses.push(miss.clone());
                    http_client::Error::instance(std::io::Error::other(miss))
                })
            }
            _ => Ok(()),
        };
        Box::pin(std::future::ready(sent))
    }

    /// Each read yields once before it answers, as a network read would, so
    /// turns sent at once contend for the connection. The recording ends
    /// where the provider stopped sending: past it, the peer is gone.
    fn recv(&mut self) -> WasmBoxedFuture<'_, http_client::Result<Option<Frame>>> {
        Box::pin(async move {
            YieldOnce(false).await;
            Ok(self.replay().inbound.pop_front().map(Frame::Text))
        })
    }

    fn close(
        &mut self,
        _frame: Option<CloseFrame>,
    ) -> WasmBoxedFuture<'_, http_client::Result<()>> {
        Box::pin(std::future::ready(Ok(())))
    }
}

/// Yields to the executor once, then completes.
struct YieldOnce(bool);

impl Future for YieldOnce {
    type Output = ();

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        if self.0 {
            return std::task::Poll::Ready(());
        }
        self.0 = true;
        cx.waker().wake_by_ref();
        std::task::Poll::Pending
    }
}

#[cfg(test)]
mod tests;
