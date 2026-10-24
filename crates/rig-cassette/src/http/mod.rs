//! Native provider HTTP recording and replay with explicit fixture roots.
//!
//! Replay is the default; `RIG_PROVIDER_TEST_MODE=record` contacts the provider
//! and overwrites scrubbed fixtures, unless the test failed or the recording
//! holds an undeclared account failure or undeleted stored state, in which
//! case it goes under [`attempt_root`] instead. The `http` feature enables this
//! engine; `bedrock` adds binary Smithy event-stream scrubbing. Invalid
//! fixtures and failed replay assertions panic in `start_at` and `finish`;
//! `try_start_at` and `try_finish` return them as a [`CassetteError`].
//!
//! ```
//! use rig_cassette::http::CassetteSpec;
//! let spec = CassetteSpec::new("completion");
//! assert_eq!(spec.scenario(), "completion");
//! ```
// This test-support engine reports invalid fixtures and replay mismatches as assertions.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable
)]
#![warn(missing_docs)]

use crate::scrub::{
    REDACTED, Scrubber, VOLATILE_JSON_KEYS, canonical_json, contains_case_insensitive,
    is_secret_header, is_secret_param,
};
use crate::traffic::{self, Exchange, RequestBody};
#[cfg(feature = "bedrock")]
use aws_smithy_eventstream::frame::{read_message_from, write_message_to};
#[cfg(feature = "bedrock")]
use aws_smithy_types::event_stream::Message as EventStreamMessage;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue, Method, StatusCode};
use axum::response::Response;
use axum::{Router, routing::any};
use base64::{Engine, prelude::BASE64_STANDARD};
use futures::stream;
use httpmock::MockServer;
use rig_core::http_client::{self, DynHttpClient};
use rig_core::redact;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::any::Any;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::fmt;
use std::fs;
use std::net::SocketAddr;
use std::panic::resume_unwind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;

mod account;
mod clock;
pub use clock::{CassetteClock, MAX_PAUSE, clock_sidecar};
pub use snapshot::request_snapshot;
use snapshot::{RequestSnapshots, SnapshotMode};
mod error;
pub use error::CassetteError;
pub mod ledger;
mod relay;
mod shape;
mod snapshot;
pub use account::{AccountFailure, account_failure, reply_account_failure};
use shape::BodyMatching;

const MODE_ENV: &str = "RIG_PROVIDER_TEST_MODE";
const ATTEMPT_DIR_ENV: &str = "RIG_CASSETTE_ATTEMPT_DIR";
const DUMMY_API_KEY: &str = REDACTED;
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

type PanicPayload = Box<dyn Any + Send + 'static>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplayMatching {
    Ordered,
    Unordered,
}

/// The fixture scenario and its replay matching policy: ordered or
/// unordered, and exact or shape-matched request bodies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CassetteSpec {
    scenario: &'static str,
    replay_matching: ReplayMatching,
    /// `None` takes `RIG_CASSETTE_MATCHING`.
    body_matching: Option<BodyMatching>,
    expected_failures: u8,
}

impl CassetteSpec {
    /// Select a scenario with strict wire-order matching.
    pub const fn new(scenario: &'static str) -> Self {
        Self {
            scenario,
            replay_matching: ReplayMatching::Ordered,
            body_matching: None,
            expected_failures: 0,
        }
    }

    /// The scenario identity used for recording and replay.
    pub const fn scenario(self) -> &'static str {
        self.scenario
    }

    /// Allow matching an unused interaction in any order.
    pub const fn unordered(mut self) -> Self {
        self.replay_matching = ReplayMatching::Unordered;
        self
    }

    /// Match request bodies by their coarse shape instead of exactly, as
    /// `RIG_CASSETTE_MATCHING=shape` does for a spec that chooses neither.
    /// The method, path, query and recorded headers still match exactly.
    /// The key keeps the body's field structure, the order of its arrays of
    /// objects, and the values of `model`, `role`, `name` and `stream`; it
    /// erases other values, the types of content, tool schemas and tool
    /// arguments. Unordered replay serves, among the interactions with the
    /// request's key, the one its body is closest to. Pair it with
    /// `RIG_CASSETTE_SNAPSHOTS=check`, which pins the bytes.
    pub const fn shape_matched(mut self) -> Self {
        self.body_matching = Some(BodyMatching::Shape);
        self
    }

    /// Match request bodies exactly whatever `RIG_CASSETTE_MATCHING` says,
    /// for a cell whose subject is the exact matcher. Exact is also what a
    /// spec that chooses neither gets when the variable is unset.
    pub const fn exact_matched(mut self) -> Self {
        self.body_matching = Some(BodyMatching::Exact);
        self
    }

    /// Declare `failure` as this cell's subject, so a recording may keep a
    /// reply that [`account_failure`] classifies as one. Without the
    /// declaration the recorder refuses such a reply.
    pub const fn expects_account_failure(mut self, failure: AccountFailure) -> Self {
        self.expected_failures |= failure.bit();
        self
    }
}

impl From<&'static str> for CassetteSpec {
    fn from(scenario: &'static str) -> Self {
        Self::new(scenario)
    }
}

#[derive(Clone, Copy, Debug)]
struct CassettePolicy {
    required_request_headers: &'static [&'static str],
    replay_matching: ReplayMatching,
    body_matching: BodyMatching,
}

impl CassettePolicy {
    fn for_scenario(provider: &str, spec: CassetteSpec) -> Self {
        let scenario = spec.scenario;
        let required_request_headers = match provider {
            "openai" | "doubleword" | "venice" => OPENAI_REQUIRED_REQUEST_HEADERS,
            "chatgpt" => CHATGPT_REQUIRED_REQUEST_HEADERS,
            "anthropic" => ANTHROPIC_REQUIRED_REQUEST_HEADERS,
            "gemini" if scenario.starts_with("interactions_api/") => {
                GEMINI_INTERACTIONS_REQUIRED_REQUEST_HEADERS
            }
            _ => NO_REQUIRED_REQUEST_HEADERS,
        };

        Self {
            required_request_headers,
            replay_matching: spec.replay_matching,
            body_matching: spec.body_matching.unwrap_or(BodyMatching::Exact),
        }
    }
}

impl Default for CassettePolicy {
    fn default() -> Self {
        Self {
            required_request_headers: NO_REQUIRED_REQUEST_HEADERS,
            replay_matching: ReplayMatching::Ordered,
            body_matching: BodyMatching::Exact,
        }
    }
}

/// Whether to replay frozen traffic or record genuine upstream exchanges.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CassetteMode {
    /// Serve only the recorded interactions from a local replay server.
    Replay,
    /// Contact the configured upstream and save scrubbed exchanges.
    Record,
}

impl CassetteMode {
    /// Read `RIG_PROVIDER_TEST_MODE`, defaulting to replay when unset or non-Unicode.
    /// Panics for values other than `record` or `replay`, ignoring case.
    pub fn current() -> Self {
        match std::env::var(MODE_ENV) {
            Ok(value) if value.eq_ignore_ascii_case("record") => Self::Record,
            Ok(value) if value.eq_ignore_ascii_case("replay") => Self::Replay,
            Ok(value) => panic!("{MODE_ENV} must be replay or record; got {value:?}"),
            Err(_) => Self::Replay,
        }
    }

    fn records(self) -> bool {
        matches!(self, Self::Record)
    }
}

/// Return whether recording is enabled, printing `reason` to stderr when true.
/// Panics if the recording-mode environment variable has an unsupported value.
/// Use this guard to avoid overwriting fixtures that live providers cannot produce.
///
/// ```no_run
/// # use rig_cassette::http::skip_when_recording;
/// # async fn test() {
/// if skip_when_recording("fixture is hand-corrupted tool JSON") {
///     return;
/// }
/// # }
/// ```
#[must_use]
pub fn skip_when_recording(reason: &str) -> bool {
    skips_recording(CassetteMode::current(), reason)
}

fn skips_recording(mode: CassetteMode, reason: &str) -> bool {
    if mode.records() {
        eprintln!("skip (not live-recordable): {reason}");
        return true;
    }
    false
}

/// Recorder for transports that bypass the HTTP proxy, including binary bodies.
#[derive(Clone, Debug)]
pub struct DirectRecorder {
    interactions: Arc<std::sync::Mutex<Vec<CassetteInteraction>>>,
    ledger: Arc<relay::LedgerTarget>,
}

/// Borrowed request bytes and headers supplied to the direct recorder.
pub struct DirectHttpRequest<'a, Headers> {
    /// HTTP method as sent on the wire.
    pub method: &'a str,
    /// Request URI, including its query.
    pub uri: &'a str,
    /// Header name/value pairs as observed on the wire.
    pub headers: Headers,
    /// Original body bytes before any cassette encoding.
    pub body: &'a [u8],
}

/// Borrowed response bytes and headers supplied to the direct recorder.
pub struct DirectHttpResponse<'a, Headers> {
    /// HTTP response status code.
    pub status: u16,
    /// Header name/value pairs as observed on the wire.
    pub headers: Headers,
    /// Original body bytes before any cassette encoding.
    pub body: &'a [u8],
}

impl DirectRecorder {
    /// Append what a successful reply to `method uri` created to the
    /// created-resource ledger. Called before the reply reaches the caller.
    fn log_created(
        &self,
        method: &str,
        uri: &str,
        request_body: &[u8],
        status: u16,
        response_body: &[u8],
    ) {
        let path = url::Url::parse(uri)
            .map(|url| url.path().to_owned())
            .unwrap_or_else(|_| uri.to_owned());
        let creation = relay::TappedCreation {
            status,
            method: method.to_owned(),
            path,
            request_body: request_body.to_vec(),
            response_body: response_body.to_vec(),
        };
        relay::record(&self.ledger, vec![creation]);
    }

    /// Scrub and append one complete direct request/response exchange.
    pub fn record_http_interaction(
        &self,
        request: DirectHttpRequest<
            '_,
            impl IntoIterator<Item = (impl AsRef<str>, impl AsRef<str>)>,
        >,
        response: DirectHttpResponse<
            '_,
            impl IntoIterator<Item = (impl AsRef<str>, impl AsRef<str>)>,
        >,
    ) {
        let mut scrubber = Scrubber::default();
        scrubber.learn_url(request.uri);
        let request_headers: Vec<_> = request.headers.into_iter().collect();
        for (name, value) in &request_headers {
            scrubber.learn_header(name.as_ref(), value.as_ref());
        }
        let mut interaction = CassetteInteraction {
            when: recorded_request(request.method, request.uri, request_headers, request.body),
            then: recorded_response(response.status, response.headers, response.body),
        };
        // The ledger sees the exchange before its reply reaches the caller.
        self.log_created(
            request.method,
            request.uri,
            request.body,
            response.status,
            response.body,
        );
        scrubber.learn_response(&interaction.then);
        scrubber.scrub_request(&mut interaction.when);
        scrubber.scrub_response(&mut interaction.then);
        self.keep(interaction);
    }

    /// Scrub and keep one exchange the [`direct_http`](ProviderCassette::direct_http)
    /// client recorded; one that got no reply has nothing to replay. A
    /// multipart request is only logged in the ledger.
    pub(crate) fn record_exchange(&self, exchange: &Exchange) {
        let Some(status) = exchange.status else {
            return;
        };
        let status = status.as_u16();
        let response_body = exchange.response_bytes().unwrap_or_default();
        let response_headers = exchange
            .response_headers
            .as_ref()
            .map(owned_headers)
            .unwrap_or_default();
        let request_body = match &exchange.request_body {
            RequestBody::Bytes(body) => body,
            RequestBody::Multipart(_) => {
                self.log_created(&exchange.method, &exchange.uri, &[], status, &response_body);
                return;
            }
        };
        self.record_http_interaction(
            DirectHttpRequest {
                method: &exchange.method,
                uri: &exchange.uri,
                headers: owned_headers(&exchange.request_headers),
                body: request_body,
            },
            DirectHttpResponse {
                status,
                headers: response_headers,
                body: &response_body,
            },
        );
    }

    fn keep(&self, interaction: CassetteInteraction) {
        self.interactions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(interaction);
    }
}

/// How a recording session reaches the provider. Replay never reaches it,
/// so the transport is irrelevant then.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordVia {
    /// A local mock server proxies every request to the provider and
    /// records the exchange (the default; the client is pointed at the
    /// proxy's URL).
    Proxy,
    /// The client talks to the provider directly and the session's
    /// [`DirectRecorder`] captures the exchanges it is handed.
    Direct,
}

/// A recording or replay session with an explicitly located fixture.
pub struct ProviderCassette {
    server: CassetteServer,
    cassette_path: PathBuf,
    base_path: String,
    mode: CassetteMode,
    provider: &'static str,
    scenario: &'static str,
    expected_failures: Arc<AtomicU8>,
    attempt_root: PathBuf,
    clock: CassetteClock,
}

enum CassetteServer {
    Recording {
        server: MockServer,
        recording_id: usize,
        credentials: Arc<std::sync::Mutex<Scrubber>>,
        relay: relay::Relay,
    },
    DirectRecording(DirectRecordingServer),
    Replay(ReplayServer),
}

struct DirectRecordingServer {
    base_url: String,
    interactions: Arc<std::sync::Mutex<Vec<CassetteInteraction>>>,
    ledger: Arc<relay::LedgerTarget>,
}

impl CassetteServer {
    /// The exchanges a recording session captured, as YAML; `None` for replay
    /// or when nothing was exchanged.
    async fn recorded_yaml(&self) -> Option<String> {
        match self {
            Self::Recording {
                server,
                recording_id,
                credentials,
                ..
            } => {
                let recording = httpmock::Recording::new(*recording_id, server);
                let bytes = recording.export_async().await.ok()??;
                let yaml = String::from_utf8(bytes.to_vec()).ok()?;
                // httpmock can export an empty document before the first response.
                let mut parsed = serde_yaml::Deserializer::from_str(&yaml)
                    .map(CassetteInteraction::deserialize)
                    .collect::<Result<Vec<_>, _>>()
                    .ok()?;
                if parsed.is_empty() {
                    return None;
                }
                let mut scrubber = credentials
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                scrubber.scrub_interactions(&mut parsed);
                Some(serialize_cassette_interactions(&parsed))
            }
            Self::DirectRecording(server) => {
                let interactions = server
                    .interactions
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                (!interactions.is_empty()).then(|| serialize_cassette_interactions(&interactions))
            }
            Self::Replay(_) => None,
        }
    }

    fn base_url(&self) -> String {
        match self {
            Self::Recording { relay, .. } => relay.base_url.clone(),
            Self::DirectRecording(server) => server.base_url.clone(),
            Self::Replay(server) => server.base_url(),
        }
    }
}

impl fmt::Debug for ProviderCassette {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderCassette")
            .field("cassette_path", &self.cassette_path)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl ProviderCassette {
    /// Start a session beneath `cassette_root`, using the ambient mode and proxy
    /// recording. Panics for an invalid upstream URL or unusable replay fixture.
    pub async fn start(
        cassette_root: &Path,
        provider: &'static str,
        spec: impl Into<CassetteSpec>,
        real_base_url: &str,
    ) -> Self {
        Self::start_via(
            RecordVia::Proxy,
            cassette_root,
            provider,
            spec,
            real_base_url,
        )
        .await
    }

    /// Start a session in the ambient [`CassetteMode`], using `transport` for
    /// recording only. Panics for an invalid mode, URL, or unusable replay fixture.
    pub async fn start_via(
        transport: RecordVia,
        cassette_root: &Path,
        provider: &'static str,
        spec: impl Into<CassetteSpec>,
        real_base_url: &str,
    ) -> Self {
        let spec = spec.into();
        let cassette_path = cassette_path(cassette_root, provider, spec.scenario);
        Self::start_at(
            transport,
            provider,
            spec,
            real_base_url,
            CassetteMode::current(),
            cassette_path,
        )
        .await
    }

    /// Start with an explicit mode and fixture path, ignoring the ambient mode.
    /// `transport` selects how recording reaches the provider. Panics with the
    /// [`CassetteError`] that [`Self::try_start_at`] would return.
    pub async fn start_at(
        transport: RecordVia,
        provider: &'static str,
        spec: CassetteSpec,
        real_base_url: &str,
        mode: CassetteMode,
        cassette_path: PathBuf,
    ) -> Self {
        Self::try_start_at(
            transport,
            provider,
            spec,
            real_base_url,
            mode,
            cassette_path,
        )
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    }

    /// [`Self::start_at`] without panicking. Returns an error for an invalid
    /// upstream URL, a missing, unreadable or malformed replay fixture or
    /// clock sidecar, and a local server that cannot bind.
    ///
    /// ```no_run
    /// # use rig_cassette::http::{CassetteError, CassetteMode, CassetteSpec, ProviderCassette, RecordVia};
    /// # async fn replay() -> Result<(), CassetteError> {
    /// let cassette = ProviderCassette::try_start_at(
    ///     RecordVia::Proxy,
    ///     "openai",
    ///     CassetteSpec::new("completion"),
    ///     "https://api.openai.com/v1",
    ///     CassetteMode::Replay,
    ///     "sessions/openai/completion.yaml".into(),
    /// )
    /// .await?;
    /// // Point a client at `cassette.base_url()` and run the session.
    /// cassette.try_finish().await
    /// # }
    /// ```
    pub async fn try_start_at(
        transport: RecordVia,
        provider: &'static str,
        spec: CassetteSpec,
        real_base_url: &str,
        mode: CassetteMode,
        cassette_path: PathBuf,
    ) -> Result<Self, CassetteError> {
        Self::try_start_with_attempts(
            transport,
            provider,
            spec,
            real_base_url,
            mode,
            cassette_path,
            attempt_root(),
        )
        .await
    }

    /// [`Self::try_start_at`] with failed recordings and the ledger under
    /// `attempt_root` instead of the ambient [`attempt_root`].
    async fn try_start_with_attempts(
        transport: RecordVia,
        provider: &'static str,
        spec: CassetteSpec,
        real_base_url: &str,
        mode: CassetteMode,
        cassette_path: PathBuf,
        attempt_root: PathBuf,
    ) -> Result<Self, CassetteError> {
        let (snapshots, spec) = if mode.records() {
            (SnapshotMode::Off, spec)
        } else {
            let spec = match spec.body_matching {
                Some(_) => spec,
                None => CassetteSpec {
                    body_matching: Some(BodyMatching::current(&cassette_path)?),
                    ..spec
                },
            };
            (SnapshotMode::current(&cassette_path)?, spec)
        };
        Self::try_start_session(
            transport,
            provider,
            spec,
            real_base_url,
            mode,
            cassette_path,
            attempt_root,
            snapshots,
        )
        .await
    }

    /// [`Self::try_start_with_attempts`] with an explicit request snapshot
    /// mode instead of the ambient `RIG_CASSETTE_SNAPSHOTS`, and the body
    /// matching `spec` names instead of the ambient `RIG_CASSETTE_MATCHING`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the session inputs, each from a distinct public entry point"
    )]
    async fn try_start_session(
        transport: RecordVia,
        provider: &'static str,
        spec: CassetteSpec,
        real_base_url: &str,
        mode: CassetteMode,
        cassette_path: PathBuf,
        attempt_root: PathBuf,
        snapshots: SnapshotMode,
    ) -> Result<Self, CassetteError> {
        let scenario = spec.scenario;
        let ledger_path = attempt_root.join(ledger::LEDGER_FILE);
        let policy = CassettePolicy::for_scenario(provider, spec);
        let upstream = UpstreamBase::parse(real_base_url, &cassette_path)?;
        // Everything that can fail runs before the replay server exists, so
        // an error never drops an unplayed server and trips its guard.
        let replay = if mode.records() {
            None
        } else {
            if !cassette_path.exists() {
                return Err(CassetteError::MissingFixture {
                    path: cassette_path,
                });
            }
            let interactions = load_replay_interactions(&cassette_path).await?;
            let recorded: Vec<Value> = interactions
                .iter()
                .map(|interaction| recorded_body_view(&interaction.when))
                .collect();
            let shapes = (policy.body_matching == BodyMatching::Shape).then(|| ReplayShapes {
                keys: recorded.iter().map(shape::shape_key).collect(),
                views: recorded.clone(),
            });
            let snapshots = RequestSnapshots::load(snapshots, &cassette_path, recorded)?;
            Some((interactions, snapshots, shapes))
        };
        let clock = CassetteClock::start(mode, &cassette_path)?;
        let server = if let Some((interactions, snapshots, shapes)) = replay {
            CassetteServer::Replay(
                ReplayServer::start(&cassette_path, interactions, policy, snapshots, shapes)
                    .await?,
            )
        } else {
            match transport {
                RecordVia::Direct => CassetteServer::DirectRecording(DirectRecordingServer {
                    base_url: upstream.origin.clone(),
                    interactions: Arc::default(),
                    ledger: Arc::new(relay::LedgerTarget {
                        path: ledger_path.clone(),
                        provider: provider.to_owned(),
                        scenario: scenario.to_owned(),
                        origin: upstream.origin.clone(),
                    }),
                }),
                RecordVia::Proxy => {
                    let server = MockServer::start_async().await;
                    server
                        .forward_to_async(&upstream.origin, |rule| {
                            rule.filter(|when| {
                                when.any_request();
                            });
                        })
                        .await;

                    let credentials = Arc::new(std::sync::Mutex::new(Scrubber::default()));
                    let learned = credentials.clone();
                    let recording = server
                        .record_async(move |rule| {
                            rule.record_request_headers(RECORDED_REQUEST_HEADERS.to_vec())
                                .filter(move |when| {
                                    when.is_true(move |request| {
                                        // Learn before the recording allowlist drops credential headers.
                                        let mut scrubber = learned
                                            .lock()
                                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                                        for (name, value) in request.headers_vec() {
                                            scrubber.learn_header(name, value);
                                        }
                                        true
                                    });
                                });
                        })
                        .await;
                    let recording_id = recording.id;
                    // Clients reach the proxy through the relay, which writes
                    // the created-resource ledger as replies pass.
                    let relay = relay::Relay::start(
                        server.address().to_string(),
                        relay::LedgerTarget {
                            path: ledger_path.clone(),
                            provider: provider.to_owned(),
                            scenario: scenario.to_owned(),
                            origin: upstream.origin.clone(),
                        },
                    )
                    .await
                    .map_err(|source| CassetteError::Bind {
                        path: cassette_path.clone(),
                        source,
                    })?;
                    CassetteServer::Recording {
                        server,
                        recording_id,
                        credentials,
                        relay,
                    }
                }
            }
        };

        Ok(Self {
            server,
            cassette_path,
            base_path: upstream.path,
            mode,
            provider,
            scenario,
            expected_failures: Arc::new(AtomicU8::new(spec.expected_failures)),
            attempt_root,
            clock,
        })
    }

    /// The session's recorded clock: wall time while recording, the same
    /// readings in the same order on replay. See [`CassetteClock`].
    pub fn clock(&self) -> CassetteClock {
        self.clock.clone()
    }

    /// Wait `duration` while recording, so a live provider sees the pause;
    /// return at once on replay. Panics above [`MAX_PAUSE`] (60 s): no
    /// cassette test waits longer.
    pub async fn pause(&self, duration: std::time::Duration) {
        self.clock.pause(duration).await;
    }

    /// Write completed, scrubbed exchanges to `path` without finalizing recording.
    /// Return `false` for replay, empty recordings, export or parsing failures,
    /// failed safety checks, or write errors. Empty snapshots never replace a file.
    pub async fn checkpoint_recording(&self, path: &Path) -> bool {
        let Some(yaml) = self.server.recorded_yaml().await else {
            return false;
        };
        let redacted = scrub_cassette_contents(&yaml);
        cassette_safety_failures(path, &redacted).is_empty()
            && write_cassette_atomically(path, redacted.as_bytes())
                .await
                .is_ok()
    }

    /// A reqwest client for a [`RecordVia::Direct`] session: while recording,
    /// it keeps every exchange it sends, binary bodies and streamed replies
    /// included, through the traffic recorder; in replay it is a plain
    /// client for the replay server.
    pub fn direct_http(&self) -> DynHttpClient {
        let http = DynHttpClient::new(rig_reqwest::ReqwestClient::default());
        match self.direct_recorder() {
            Some(recorder) => traffic::recorded(http, traffic::Sink::Cassette(recorder)),
            None => http,
        }
    }

    /// Return the direct recorder only for an active direct-recording session.
    pub fn direct_recorder(&self) -> Option<DirectRecorder> {
        match &self.server {
            CassetteServer::DirectRecording(server) => Some(DirectRecorder {
                interactions: server.interactions.clone(),
                ledger: server.ledger.clone(),
            }),
            _ => None,
        }
    }

    /// Return the session endpoint, preserving the upstream base path.
    pub fn base_url(&self) -> String {
        format!("{}{}", self.server.base_url(), self.base_path)
    }

    /// Return an invalid key for recording auth failures, or the scrubbed dummy
    /// key for replay so matched query credentials agree with the fixture.
    pub fn bogus_api_key(&self) -> String {
        // Handing out a rejected credential makes the rejection the subject.
        self.expect_account_failure(AccountFailure::Auth);
        if self.mode.records() {
            "invalid-edge-matrix-key".to_string()
        } else {
            DUMMY_API_KEY.to_string()
        }
    }

    /// Declare `failure` as this session's subject, so the recording may keep
    /// a reply [`account_failure`] classifies as one. Call it from a wrapper
    /// that presents a rejected or missing credential on purpose.
    pub fn expect_account_failure(&self, failure: AccountFailure) {
        self.expected_failures
            .fetch_or(failure.bit(), Ordering::Relaxed);
    }

    /// Return the named environment key when recording, or a dummy key for replay.
    /// Panics when recording and the variable is missing or non-Unicode.
    pub fn api_key(&self, env_name: &str) -> String {
        if self.mode.records() {
            std::env::var(env_name).unwrap_or_else(|_| {
                panic!("{env_name} must be set when {MODE_ENV}={:?}", self.mode)
            })
        } else {
            DUMMY_API_KEY.to_string()
        }
    }

    /// Write a scrubbed recording or assert complete replay consumption and shut
    /// down. Panics with the [`CassetteError`] that [`Self::try_finish`] would
    /// return. A recording holding an undeclared account failure (see
    /// [`account_failure`]) is not written to the fixture: it goes to
    /// [`attempt_root`] and this panics.
    pub async fn finish(self) {
        if let Err(error) = self.try_finish().await {
            panic!("{error}");
        }
    }

    /// [`Self::finish`] without panicking. Returns an error for a replay that
    /// left interactions or clock readings unused or refused a request, and
    /// for a recording that is empty, refused, unsafe after scrubbing, or
    /// cannot be written. A refused recording is kept under [`attempt_root`],
    /// and [`CassetteError::RecordingRefused`] says where. The local servers
    /// are shut down on every path, and dropping what remains never panics.
    pub async fn try_finish(self) -> Result<(), CassetteError> {
        let expected = self.expected_failures.load(Ordering::Relaxed);
        let Self {
            server,
            cassette_path,
            provider,
            scenario,
            attempt_root,
            clock,
            ..
        } = self;

        let yaml = match server {
            CassetteServer::Replay(mut server) => {
                let consumed = server.check_consumed(&cassette_path).await;
                server.shutdown().await;
                consumed?;
                return clock.finish(&cassette_path).await;
            }
            server => {
                server
                    .recorded_yaml()
                    .await
                    .ok_or_else(|| CassetteError::EmptyRecording {
                        path: cassette_path.clone(),
                    })?
            }
        };

        let refusals = recording_refusals(provider, &yaml, expected);
        if !refusals.is_empty() {
            let kept_at = write_attempt(&attempt_root, provider, scenario, &yaml).await;
            return Err(CassetteError::RecordingRefused {
                path: cassette_path,
                refusals,
                kept_at,
            });
        }
        write_scrubbed_cassette(&cassette_path, &yaml).await?;
        clock.finish(&cassette_path).await
    }

    /// Finalize after a successful test, preserving its original panic otherwise.
    /// A failed recording is kept under [`attempt_root`], never over the fixture.
    pub async fn finish_after_test(self, test_result: Result<(), PanicPayload>) {
        match test_result {
            Ok(()) => self.finish().await,
            Err(payload) => {
                self.keep_failed_attempt().await;
                resume_unwind(payload);
            }
        }
    }

    /// Finalize after a successful fallible test, preserving its failure otherwise.
    /// A failed recording is kept under [`attempt_root`], never over the fixture.
    pub async fn finish_after_test_result<E>(
        mut self,
        test_result: Result<Result<(), E>, PanicPayload>,
    ) -> Result<(), E> {
        match test_result {
            Ok(Ok(())) => {
                self.finish().await;
                Ok(())
            }
            Ok(Err(error)) => {
                // The test's own failure is the report; the session it left
                // unplayed must not replace it with the drop guard's.
                self.disarm_drop_guard();
                self.keep_failed_attempt().await;
                Err(error)
            }
            Err(payload) => {
                self.keep_failed_attempt().await;
                resume_unwind(payload)
            }
        }
    }

    /// Save a failed recording pass's exchanges under [`attempt_root`], so
    /// the ids it created stay recoverable. Replay sessions keep nothing.
    async fn keep_failed_attempt(self) {
        if !self.mode.records() {
            return;
        }
        let Self {
            server,
            provider,
            scenario,
            attempt_root,
            ..
        } = self;
        if let Some(yaml) = server.recorded_yaml().await
            && let Some(path) = write_attempt(&attempt_root, provider, scenario, &yaml).await
        {
            eprintln!("failed recording kept at {}", path.display());
        }
    }

    fn disarm_drop_guard(&mut self) {
        if let CassetteServer::Replay(server) = &mut self.server {
            server.checked = true;
        }
    }
}

struct ReplayServer {
    addr: SocketAddr,
    state: Arc<Mutex<ReplayState>>,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
    /// `finish` asserted consumption: the drop guard has nothing to add.
    checked: bool,
}

impl ReplayServer {
    async fn start(
        cassette_path: &Path,
        interactions: Vec<ReplayInteraction>,
        policy: CassettePolicy,
        snapshots: Option<RequestSnapshots>,
        shapes: Option<ReplayShapes>,
    ) -> Result<Self, CassetteError> {
        let bind_error = |source| CassetteError::Bind {
            path: cassette_path.to_path_buf(),
            source,
        };
        let listener = TcpListener::bind("127.0.0.1:0").await.map_err(bind_error)?;
        let addr = listener.local_addr().map_err(bind_error)?;
        let state = Arc::new(Mutex::new(ReplayState {
            cassette_path: cassette_path.to_path_buf(),
            interactions,
            misses: Vec::new(),
            policy,
            snapshots,
            shapes,
        }));
        let app = Router::new()
            .fallback(any(replay_request))
            .with_state(state.clone());

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            let result = axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .map_err(|error| format!("replay server should run: {error}"));

            if let Err(error) = result {
                panic!("{error}");
            }
        });

        Ok(Self {
            addr,
            state,
            shutdown: Some(shutdown_tx),
            task: Some(task),
            checked: false,
        })
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Check that replay played every interaction and refused nothing, then
    /// check or write the request snapshot. This disarms the drop guard
    /// whatever the outcome.
    async fn check_consumed(&mut self, cassette_path: &Path) -> Result<(), CassetteError> {
        self.checked = true;
        let state = self.state.lock().await;
        if let Some(error) = replay_mismatch(cassette_path, &state.interactions, &state.misses) {
            return Err(error);
        }
        state
            .snapshots
            .as_ref()
            .map_or(Ok(()), RequestSnapshots::finish)
    }

    async fn shutdown(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }

        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

impl Drop for ReplayServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if self.checked || std::thread::panicking() {
            return;
        }
        // Early test returns must not hide unused interactions or refused requests.
        // Fully consumed sessions can be dropped without an explicit finish.
        let Ok(state) = self.state.try_lock() else {
            return;
        };
        let failure = replay_completion_failure_message(
            &state.cassette_path,
            &state.interactions,
            &state.misses,
        )
        .or_else(|| {
            state
                .snapshots
                .as_ref()
                .and_then(|snapshots| snapshots.finish().err())
                .map(|error| error.to_string())
        });
        if let Some(message) = failure {
            panic!("{message}\n(the replay session was dropped without `finish`)");
        }
    }
}

struct ReplayState {
    cassette_path: PathBuf,
    interactions: Vec<ReplayInteraction>,
    misses: Vec<ReplayMiss>,
    policy: CassettePolicy,
    snapshots: Option<RequestSnapshots>,
    /// Each interaction's coarse shape key and body, under shape matching.
    shapes: Option<ReplayShapes>,
}

/// What shape matching keeps of each recorded interaction.
struct ReplayShapes {
    /// The coarse shape key of each recorded body.
    keys: Vec<Value>,
    /// Each recorded body as [`snapshot::body_view`] reads it.
    views: Vec<Value>,
}

#[derive(Clone, Debug)]
struct ReplayMiss {
    diagnostic: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct CassetteInteraction {
    when: CassetteRequest,
    then: CassetteResponse,
}

#[derive(Debug, Deserialize, Serialize)]
struct CassetteRequest {
    path: String,
    method: String,
    #[serde(default)]
    query_param: Vec<NameValue>,
    #[serde(default)]
    header: Vec<NameValue>,
    body: Option<String>,
    #[serde(default, skip_serializing_if = "BodyEncoding::is_utf8")]
    body_encoding: BodyEncoding,
}

#[derive(Debug, Deserialize, Serialize)]
struct CassetteResponse {
    status: u16,
    #[serde(default)]
    header: Vec<NameValue>,
    body: Option<String>,
    #[serde(default, skip_serializing_if = "BodyEncoding::is_utf8")]
    body_encoding: BodyEncoding,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum BodyEncoding {
    #[default]
    Utf8,
    Base64,
}

impl BodyEncoding {
    fn is_utf8(&self) -> bool {
        matches!(self, Self::Utf8)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct NameValue {
    name: String,
    value: String,
}

struct ReplayInteraction {
    when: CassetteRequest,
    then: CassetteResponse,
    consumed: bool,
}

#[cfg(test)]
fn assert_replay_finished(
    cassette_path: &Path,
    interactions: &[ReplayInteraction],
    misses: &[ReplayMiss],
) {
    if let Some(message) = replay_completion_failure_message(cassette_path, interactions, misses) {
        panic!("{message}");
    }
}

fn replay_completion_failure_message(
    cassette_path: &Path,
    interactions: &[ReplayInteraction],
    misses: &[ReplayMiss],
) -> Option<String> {
    replay_mismatch(cassette_path, interactions, misses).map(|error| error.to_string())
}

/// The replay failure for interactions left unplayed and requests refused,
/// or `None` when replay played the fixture exactly.
fn replay_mismatch(
    cassette_path: &Path,
    interactions: &[ReplayInteraction],
    misses: &[ReplayMiss],
) -> Option<CassetteError> {
    let unused_interactions = interactions
        .iter()
        .enumerate()
        .filter(|(_, interaction)| !interaction.consumed)
        .map(|(index, interaction)| {
            format!(
                "[{index}] {} {}",
                interaction.when.method, interaction.when.path
            )
        })
        .collect::<Vec<_>>();
    let unexpected_requests = misses
        .iter()
        .map(|miss| miss.diagnostic.clone())
        .collect::<Vec<_>>();
    (!unused_interactions.is_empty() || !unexpected_requests.is_empty()).then(|| {
        CassetteError::ReplayMismatch {
            path: cassette_path.to_path_buf(),
            unused_interactions,
            unexpected_requests,
        }
    })
}

/// Read, parse and check the fixture a replay session serves.
async fn load_replay_interactions(
    cassette_path: &Path,
) -> Result<Vec<ReplayInteraction>, CassetteError> {
    let contents = tokio::fs::read_to_string(cassette_path)
        .await
        .map_err(|source| CassetteError::UnreadableFixture {
            path: cassette_path.to_path_buf(),
            source,
        })?;
    try_parse_cassette_interactions(cassette_path, &contents)?
        .into_iter()
        .enumerate()
        .map(|(index, interaction)| match replay_problem(&interaction) {
            Some(reason) => Err(CassetteError::InvalidInteraction {
                path: cassette_path.to_path_buf(),
                index,
                reason,
            }),
            None => Ok(ReplayInteraction {
                when: interaction.when,
                then: interaction.then,
                consumed: false,
            }),
        })
        .collect()
}

/// What would make the replay server fail on `interaction`: a recorded body
/// that does not decode, or a response that is not valid HTTP.
fn replay_problem(interaction: &CassetteInteraction) -> Option<String> {
    let request = &interaction.when;
    if let Some(body) = &request.body
        && let Err(error) = decode_body(body, request.body_encoding)
    {
        return Some(format!("request body does not decode: {error}"));
    }
    let response = &interaction.then;
    if let Err(error) = StatusCode::from_u16(response.status) {
        return Some(format!("response status {}: {error}", response.status));
    }
    for header in &response.header {
        if is_hop_by_hop_header(&header.name) {
            continue;
        }
        if let Err(error) = HeaderName::from_bytes(header.name.as_bytes()) {
            return Some(format!(
                "invalid response header name {:?}: {error}",
                header.name
            ));
        }
        if let Err(error) = HeaderValue::from_str(&header.value) {
            return Some(format!(
                "invalid value for response header {:?}: {error}",
                header.name
            ));
        }
    }
    if let Some(body) = &response.body
        && let Err(error) = decode_body(body, response.body_encoding)
    {
        return Some(format!("response body does not decode: {error}"));
    }
    None
}

fn parse_cassette_interactions(cassette_path: &Path, contents: &str) -> Vec<CassetteInteraction> {
    try_parse_cassette_interactions(cassette_path, contents)
        .unwrap_or_else(|error| panic!("{error}"))
}

fn try_parse_cassette_interactions(
    cassette_path: &Path,
    contents: &str,
) -> Result<Vec<CassetteInteraction>, CassetteError> {
    serde_yaml::Deserializer::from_str(contents)
        .map(|document| {
            CassetteInteraction::deserialize(document).map_err(|source| {
                CassetteError::MalformedFixture {
                    path: cassette_path.to_path_buf(),
                    source,
                }
            })
        })
        .collect()
}

async fn replay_request(
    State(state): State<Arc<Mutex<ReplayState>>>,
    method: Method,
    uri: axum::http::Uri,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Response {
    let mut state = state.lock().await;
    let request = IncomingRequest {
        method,
        uri,
        headers,
        body,
    };
    let policy = state.policy;
    let sent =
        (state.snapshots.is_some() || state.shapes.is_some()).then(|| incoming_body_view(&request));
    let sent_shape = state
        .shapes
        .as_ref()
        .zip(sent.as_ref())
        .map(|(_, sent)| shape::shape_key(sent));
    let shapes = state
        .shapes
        .as_ref()
        .zip(sent_shape.as_ref())
        .map(|(shapes, sent)| ShapeKeys {
            recorded: &shapes.keys,
            sent,
        });
    // Unordered shape matching serves the interaction the request is
    // closest to: its snapshot's expected body when snapshots check, else
    // the recording.
    let distance = |index: usize| match (&state.snapshots, &state.shapes, &sent) {
        (Some(snapshots), _, Some(sent)) => snapshots.distance(index, sent),
        (None, Some(shapes), Some(sent)) => shapes
            .views
            .get(index)
            .map_or(usize::MAX, |view| snapshot::diff(view, sent).len()),
        _ => usize::MAX,
    };

    let Some(index) =
        matching_interaction_index(policy, &state.interactions, &request, shapes, distance)
    else {
        // The interaction the request was most likely meant for, compared
        // with what its snapshot expects.
        let snapshot_diff =
            state
                .snapshots
                .as_ref()
                .zip(sent.as_ref())
                .and_then(|(snapshots, sent)| {
                    let candidate = miss_candidate_index(policy, &state.interactions, &request)?;
                    snapshots.difference(candidate, sent)
                });
        if let Some(difference) = &snapshot_diff {
            eprintln!(
                "replay of {} refused a request that differs from its snapshot at {difference}",
                state.cassette_path.display()
            );
        }
        let message = replay_miss_message(
            policy,
            &request,
            &state.interactions,
            shapes,
            snapshot_diff.as_deref(),
        );
        let diagnostic = match &snapshot_diff {
            Some(difference) => format!("{message}\nsnapshot difference at {difference}"),
            None => message.clone(),
        };
        state.misses.push(ReplayMiss { diagnostic });
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("content-type", "application/json")
            .body(Body::from(message))
            .expect("replay miss response should build");
    };

    if let (Some(snapshots), Some(sent)) = (state.snapshots.as_mut(), sent) {
        snapshots.observe(index, sent);
    }
    let cassette_path = state.cassette_path.clone();
    let interaction = &mut state.interactions[index];
    interaction.consumed = true;
    cassette_response(&interaction.then, &cassette_path)
}

/// The unplayed interaction a refused request was most likely meant for:
/// the next one in order, or for unordered replay the first unplayed one on
/// the request's method and path.
fn miss_candidate_index(
    policy: CassettePolicy,
    interactions: &[ReplayInteraction],
    request: &IncomingRequest,
) -> Option<usize> {
    let mut unplayed = interactions
        .iter()
        .enumerate()
        .filter(|(_, interaction)| !interaction.consumed);
    match policy.replay_matching {
        ReplayMatching::Ordered => unplayed.next().map(|(index, _)| index),
        ReplayMatching::Unordered => unplayed
            .find(|(_, interaction)| {
                request
                    .method
                    .as_str()
                    .eq_ignore_ascii_case(&interaction.when.method)
                    && request.uri.path() == interaction.when.path
            })
            .map(|(index, _)| index),
    }
}

/// A recorded request's body as request snapshots compare it.
fn recorded_body_view(request: &CassetteRequest) -> Value {
    let content_type = request
        .header
        .iter()
        .find(|header| header.name.eq_ignore_ascii_case("content-type"))
        .map(|header| header.value.as_str());
    // `load_replay_interactions` refuses a body that does not decode.
    let body = request
        .body
        .as_deref()
        .and_then(|body| decode_body(body, request.body_encoding).ok())
        .unwrap_or_default();
    snapshot::body_view(content_type, &body)
}

/// An incoming request's body as request snapshots compare it.
fn incoming_body_view(request: &IncomingRequest) -> Value {
    let content_type = request
        .headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok());
    snapshot::body_view(content_type, &request.body)
}

/// Under shape matching, the coarse shape key of every interaction's
/// recorded body and of the request's body.
#[derive(Clone, Copy)]
struct ShapeKeys<'a> {
    recorded: &'a [Value],
    sent: &'a Value,
}

/// The interaction replay serves `request` from. Ordered replay serves the
/// next unplayed interaction when it matches; unordered replay the first
/// unplayed one that matches, and under shape matching the one of those at
/// the least `distance` from the request (the first on a tie), so requests
/// that share a shape still find their own recordings.
fn matching_interaction_index(
    policy: CassettePolicy,
    interactions: &[ReplayInteraction],
    request: &IncomingRequest,
    shapes: Option<ShapeKeys<'_>>,
    distance: impl Fn(usize) -> usize,
) -> Option<usize> {
    let matches = |index: usize, interaction: &ReplayInteraction| match shapes {
        Some(keys) => shape_request_matches(policy, request, &interaction.when, keys, index),
        None => request_matches(policy, request, &interaction.when),
    };
    let mut unplayed = interactions
        .iter()
        .enumerate()
        .filter(|(_, interaction)| !interaction.consumed);
    match policy.replay_matching {
        ReplayMatching::Ordered => {
            let (index, interaction) = unplayed.next()?;
            matches(index, interaction).then_some(index)
        }
        ReplayMatching::Unordered => {
            let candidates: Vec<usize> = unplayed
                .filter(|(index, interaction)| matches(*index, interaction))
                .map(|(index, _)| index)
                .collect();
            match shapes {
                Some(_) => candidates.into_iter().min_by_key(|index| distance(*index)),
                None => candidates.first().copied(),
            }
        }
    }
}

fn replay_miss_message(
    policy: CassettePolicy,
    request: &IncomingRequest,
    interactions: &[ReplayInteraction],
    shapes: Option<ShapeKeys<'_>>,
    snapshot_diff: Option<&str>,
) -> String {
    let candidates = interactions
        .iter()
        .enumerate()
        .map(|(index, interaction)| {
            let method_matches = request
                .method
                .as_str()
                .eq_ignore_ascii_case(&interaction.when.method);
            let path_matches = request.uri.path() == interaction.when.path;
            let query_matches = query_matches(request.uri.query(), &interaction.when.query_param);
            let headers_match = headers_match(&request.headers, &interaction.when.header);
            let required_headers_match = required_headers_present(policy, &request.headers);
            let body_matches = body_matches(
                &request.headers,
                &interaction.when.header,
                &request.body,
                interaction.when.body.as_deref(),
                interaction.when.body_encoding,
            );
            let shape_matches = shapes
                .map(|keys| body_shape_matches(request, &interaction.when, keys, index));

            json!({
                "index": index,
                "consumed": interaction.consumed,
                "method_matches": method_matches,
                "path_matches": path_matches,
                "query_matches": query_matches,
                "expected_query_params": scrub_name_values_for_diagnostics(interaction.when.query_param.clone()),
                "headers_match": headers_match,
                "required_headers_match": required_headers_match,
                "body_matches": body_matches,
                "shape_matches": shape_matches,
                "expected_method": interaction.when.method,
                "expected_path": interaction.when.path,
                "expected_body_preview": interaction.when.body.as_deref().map(body_preview_for_diagnostics),
            })
        })
        .collect::<Vec<_>>();

    let mut message = json!({
        "message": "Request did not match any route or mock",
        "actual_method": request.method.as_str(),
        "actual_path": request.uri.path(),
        "actual_query": request.uri.query().map(scrub_text_for_diagnostics),
        "actual_query_params": scrub_query_pairs_for_diagnostics(request.uri.query()),
        "required_headers": policy.required_request_headers,
        "missing_required_headers": missing_required_headers(policy, &request.headers),
        "actual_body_preview": body_preview_bytes_for_diagnostics(&request.body),
        "candidates": candidates,
    });
    if let (Some(difference), Value::Object(fields)) = (snapshot_diff, &mut message) {
        fields.insert("snapshot_diff".to_owned(), Value::from(difference));
    }
    message.to_string()
}

struct IncomingRequest {
    method: Method,
    uri: axum::http::Uri,
    headers: axum::http::HeaderMap,
    body: Bytes,
}

/// Whether everything but the body matches: method, path, query, the
/// recorded headers and the policy's required headers.
fn request_line_matches(
    policy: CassettePolicy,
    request: &IncomingRequest,
    expected: &CassetteRequest,
) -> bool {
    request
        .method
        .as_str()
        .eq_ignore_ascii_case(&expected.method)
        && request.uri.path() == expected.path
        && query_matches(request.uri.query(), &expected.query_param)
        && headers_match(&request.headers, &expected.header)
        && required_headers_present(policy, &request.headers)
}

/// [`request_line_matches`] with the body compared by its coarse shape key
/// against interaction `index`.
fn shape_request_matches(
    policy: CassettePolicy,
    request: &IncomingRequest,
    expected: &CassetteRequest,
    keys: ShapeKeys<'_>,
    index: usize,
) -> bool {
    request_line_matches(policy, request, expected)
        && body_shape_matches(request, expected, keys, index)
}

/// Whether the request body has interaction `index`'s coarse shape. A
/// recording without a body accepts an empty body or, as exact matching
/// does, any multipart body.
fn body_shape_matches(
    request: &IncomingRequest,
    expected: &CassetteRequest,
    keys: ShapeKeys<'_>,
    index: usize,
) -> bool {
    if expected.body.is_none() {
        return request.body.is_empty() || is_multipart_request(&request.headers, &expected.header);
    }
    keys.recorded.get(index) == Some(keys.sent)
}

fn request_matches(
    policy: CassettePolicy,
    request: &IncomingRequest,
    expected: &CassetteRequest,
) -> bool {
    request_line_matches(policy, request, expected)
        && body_matches(
            &request.headers,
            &expected.header,
            &request.body,
            expected.body.as_deref(),
            expected.body_encoding,
        )
}

fn recorded_request<N, V>(
    method: &str,
    uri: &str,
    headers: impl IntoIterator<Item = (N, V)>,
    body: &[u8],
) -> CassetteRequest
where
    N: AsRef<str>,
    V: AsRef<str>,
{
    let parsed = parse_recorded_uri(uri);
    let recorded_body = recorded_body(body);
    CassetteRequest {
        path: parsed.path().to_string(),
        method: method.to_ascii_uppercase(),
        query_param: parsed
            .query_pairs()
            .into_owned()
            .map(|(name, value)| NameValue { name, value })
            .collect(),
        header: recorded_request_headers(headers),
        body: recorded_body.body,
        body_encoding: recorded_body.encoding,
    }
}

fn recorded_response<N, V>(
    status: u16,
    headers: impl IntoIterator<Item = (N, V)>,
    body: &[u8],
) -> CassetteResponse
where
    N: AsRef<str>,
    V: AsRef<str>,
{
    let recorded_body = recorded_body(body);
    CassetteResponse {
        status,
        header: headers
            .into_iter()
            .map(|(name, value)| NameValue {
                name: name.as_ref().to_ascii_lowercase(),
                value: value.as_ref().to_string(),
            })
            .collect(),
        body: recorded_body.body,
        body_encoding: recorded_body.encoding,
    }
}

fn parse_recorded_uri(uri: &str) -> url::Url {
    url::Url::parse(uri).unwrap_or_else(|_| {
        url::Url::parse(&format!("http://cassette.invalid{uri}"))
            .unwrap_or_else(|error| panic!("recorded URI {uri:?} should parse: {error}"))
    })
}

fn recorded_request_headers<N, V>(headers: impl IntoIterator<Item = (N, V)>) -> Vec<NameValue>
where
    N: AsRef<str>,
    V: AsRef<str>,
{
    headers
        .into_iter()
        .filter_map(|(name, value)| {
            let name = name.as_ref();
            contains_case_insensitive(RECORDED_REQUEST_HEADERS, name).then(|| NameValue {
                name: name.to_ascii_lowercase(),
                value: value.as_ref().to_string(),
            })
        })
        .collect()
}

struct RecordedBody {
    body: Option<String>,
    encoding: BodyEncoding,
}

fn recorded_body(body: &[u8]) -> RecordedBody {
    if body.is_empty() {
        return RecordedBody {
            body: None,
            encoding: BodyEncoding::Utf8,
        };
    }

    match std::str::from_utf8(body) {
        Ok(body) => RecordedBody {
            body: Some(body.to_string()),
            encoding: BodyEncoding::Utf8,
        },
        Err(_) => RecordedBody {
            body: Some(BASE64_STANDARD.encode(body)),
            encoding: BodyEncoding::Base64,
        },
    }
}

fn query_matches(query: Option<&str>, expected: &[NameValue]) -> bool {
    let mut actual: Vec<_> = parsed_query_pairs(query)
        .into_iter()
        .map(|(name, value)| NameValue { name, value })
        .collect();
    scrub_query_params(&mut actual);
    query_pair_counts(actual.into_iter().map(|pair| (pair.name, pair.value)))
        == query_pair_counts(
            expected
                .iter()
                .map(|pair| (pair.name.clone(), pair.value.clone())),
        )
}

fn parsed_query_pairs(query: Option<&str>) -> Vec<(String, String)> {
    url::form_urlencoded::parse(query.unwrap_or_default().as_bytes())
        .into_owned()
        .collect()
}

fn query_pair_counts(
    pairs: impl IntoIterator<Item = (String, String)>,
) -> BTreeMap<(String, String), usize> {
    let mut counts = BTreeMap::new();
    for pair in pairs {
        *counts.entry(pair).or_insert(0) += 1;
    }
    counts
}

fn headers_match(actual: &axum::http::HeaderMap, expected: &[NameValue]) -> bool {
    expected.iter().all(|expected| {
        let Ok(name) = HeaderName::from_bytes(expected.name.as_bytes()) else {
            return false;
        };

        actual
            .get(name)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                if expected.name.eq_ignore_ascii_case("content-type")
                    && expected.value.starts_with("multipart/form-data;")
                {
                    value.starts_with("multipart/form-data;")
                } else {
                    redact::text(value) == expected.value
                }
            })
    })
}

fn required_headers_present(policy: CassettePolicy, actual: &axum::http::HeaderMap) -> bool {
    missing_required_headers(policy, actual).is_empty()
}

fn missing_required_headers(
    policy: CassettePolicy,
    actual: &axum::http::HeaderMap,
) -> Vec<&'static str> {
    policy
        .required_request_headers
        .iter()
        .copied()
        .filter(|required| !has_nonempty_header(actual, required))
        .collect()
}

fn has_nonempty_header(actual: &axum::http::HeaderMap, name: &str) -> bool {
    let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
        return false;
    };

    actual
        .get(name)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| !value.trim().is_empty())
}

fn body_matches(
    actual_headers: &axum::http::HeaderMap,
    expected_headers: &[NameValue],
    actual: &[u8],
    expected: Option<&str>,
    expected_encoding: BodyEncoding,
) -> bool {
    let Some(expected) = expected else {
        // The proxy omits non-UTF-8 multipart bodies, so absent multipart bytes
        // cannot be matched; other requests must still be empty.
        return actual.is_empty() || is_multipart_request(actual_headers, expected_headers);
    };
    let expected_bytes = decode_body(expected, expected_encoding)
        .unwrap_or_else(|error| panic!("cassette request body should decode: {error}"));
    if is_multipart_request(actual_headers, expected_headers) {
        return multipart_bodies_match(actual_headers, expected_headers, actual, &expected_bytes);
    }

    if expected_encoding == BodyEncoding::Base64 {
        return actual == expected_bytes;
    }

    let Ok(actual) = std::str::from_utf8(actual) else {
        return false;
    };
    let Ok(expected) = std::str::from_utf8(&expected_bytes) else {
        return false;
    };
    let actual = Scrubber::default().scrub_body(actual);
    let expected = Scrubber::default().scrub_body(expected);

    if let (Some(actual_json), Some(expected_json)) =
        (canonical_json(&actual), canonical_json(&expected))
    {
        return actual_json == expected_json;
    }

    actual == expected
}

fn decode_body(body: &str, encoding: BodyEncoding) -> Result<Vec<u8>, base64::DecodeError> {
    match encoding {
        BodyEncoding::Utf8 => Ok(body.as_bytes().to_vec()),
        BodyEncoding::Base64 => BASE64_STANDARD.decode(body),
    }
}

fn is_multipart_request(
    actual_headers: &axum::http::HeaderMap,
    expected_headers: &[NameValue],
) -> bool {
    expected_headers.iter().any(|header| {
        header.name.eq_ignore_ascii_case("content-type")
            && header.value.starts_with("multipart/form-data;")
    }) || actual_headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("multipart/form-data;"))
}

#[derive(Debug, Eq, PartialEq)]
struct MultipartPart {
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

fn multipart_bodies_match(
    actual_headers: &axum::http::HeaderMap,
    expected_headers: &[NameValue],
    actual: &[u8],
    expected: &[u8],
) -> bool {
    let Some(actual_boundary) = actual_headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(multipart_boundary)
    else {
        return false;
    };
    let Some(expected_boundary) = expected_headers
        .iter()
        .find(|header| header.name.eq_ignore_ascii_case("content-type"))
        .and_then(|header| multipart_boundary(&header.value))
    else {
        return false;
    };

    match (
        parse_multipart_parts(actual, &actual_boundary),
        parse_multipart_parts(expected, &expected_boundary),
    ) {
        (Some(actual_parts), Some(expected_parts)) => actual_parts == expected_parts,
        _ => false,
    }
}

fn multipart_boundary(content_type: &str) -> Option<String> {
    content_type.split(';').find_map(|part| {
        let (name, value) = part.trim().split_once('=')?;
        name.eq_ignore_ascii_case("boundary")
            .then(|| value.trim_matches('"').to_string())
            .filter(|value| !value.is_empty())
    })
}

fn parse_multipart_parts(body: &[u8], boundary: &str) -> Option<Vec<MultipartPart>> {
    let marker = format!("--{boundary}").into_bytes();
    let mut parts = Vec::new();

    for raw_part in split_bytes(body, &marker).into_iter().skip(1) {
        let raw_part = raw_part.strip_prefix(b"\r\n").unwrap_or(raw_part);
        if raw_part.starts_with(b"--") {
            break;
        }

        let raw_part = raw_part.strip_suffix(b"\r\n").unwrap_or(raw_part);
        if raw_part.iter().all(u8::is_ascii_whitespace) {
            continue;
        }

        let header_end = find_bytes(raw_part, b"\r\n\r\n")?;
        let raw_headers = &raw_part[..header_end];
        let raw_body = &raw_part[header_end + b"\r\n\r\n".len()..];
        let raw_headers = std::str::from_utf8(raw_headers).ok()?;
        let mut headers = raw_headers
            .lines()
            .filter_map(|line| {
                let (name, value) = line.split_once(':')?;
                Some((
                    name.trim().to_ascii_lowercase(),
                    normalize_multipart_header_value(value.trim()),
                ))
            })
            .collect::<Vec<_>>();
        headers.sort();

        parts.push(MultipartPart {
            headers,
            body: raw_body.to_vec(),
        });
    }

    Some(parts)
}

fn split_bytes<'a>(body: &'a [u8], marker: &[u8]) -> Vec<&'a [u8]> {
    let mut parts = Vec::new();
    let mut remainder = body;

    while let Some(index) = find_bytes(remainder, marker) {
        let (before, after_marker) = remainder.split_at(index);
        parts.push(before);
        remainder = &after_marker[marker.len()..];
    }

    parts.push(remainder);
    parts
}

/// Where `needle`, which is not empty, first occurs in `body`.
fn find_bytes(body: &[u8], needle: &[u8]) -> Option<usize> {
    body.windows(needle.len())
        .position(|window| window == needle)
}

fn normalize_multipart_header_value(value: &str) -> String {
    value
        .split(';')
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("; ")
}

fn body_preview(body: &str) -> String {
    const LIMIT: usize = 512;
    let mut preview = body.chars().take(LIMIT).collect::<String>();
    if body.chars().count() > LIMIT {
        preview.push_str("...");
    }
    preview
}

fn body_preview_for_diagnostics(body: &str) -> String {
    body_preview(&Scrubber::default().scrub_body(body))
}

fn body_preview_bytes_for_diagnostics(body: &[u8]) -> String {
    match std::str::from_utf8(body) {
        Ok(body) => body_preview_for_diagnostics(body),
        Err(_) => format!("<{} bytes of non-UTF-8 body>", body.len()),
    }
}

fn scrub_text_for_diagnostics(text: &str) -> String {
    Scrubber::default().scrub_text(text)
}

fn scrub_query_pairs_for_diagnostics(query: Option<&str>) -> Vec<NameValue> {
    let pairs = parsed_query_pairs(query)
        .into_iter()
        .map(|(name, value)| NameValue { name, value });
    scrub_name_values_for_diagnostics(pairs.collect())
}

fn scrub_name_values_for_diagnostics(mut values: Vec<NameValue>) -> Vec<NameValue> {
    scrub_query_params(&mut values);
    for value in &mut values {
        value.value = scrub_text_for_diagnostics(&value.value);
    }
    values
}

fn cassette_response(response: &CassetteResponse, cassette_path: &Path) -> Response {
    let mut builder = Response::builder().status(response.status);
    for header in &response.header {
        if is_hop_by_hop_header(&header.name) {
            continue;
        }
        let name = HeaderName::from_bytes(header.name.as_bytes()).unwrap_or_else(|error| {
            panic!(
                "provider cassette {} contains invalid response header name {:?}: {error}",
                cassette_path.display(),
                header.name
            )
        });
        let value = HeaderValue::from_str(&header.value).unwrap_or_else(|error| {
            panic!(
                "provider cassette {} contains invalid value for response header {:?}: {error}",
                cassette_path.display(),
                header.name
            )
        });
        builder = builder.header(name, value);
    }

    let body = response_body(response);
    builder.body(body).expect("cassette response should build")
}

fn response_body(response: &CassetteResponse) -> Body {
    let body = response.body.clone().unwrap_or_default();
    if response.body_encoding == BodyEncoding::Base64 {
        let bytes = decode_body(&body, BodyEncoding::Base64)
            .expect("base64 cassette response body should decode");
        return Body::from(bytes);
    }

    if is_sse_response(&response.header) {
        let chunks = sse_body_chunks(&body)
            .into_iter()
            .map(Ok::<Bytes, Infallible>);
        Body::from_stream(stream::iter(chunks))
    } else {
        Body::from(body)
    }
}

fn is_sse_response(headers: &[NameValue]) -> bool {
    headers.iter().any(|header| {
        header.name.eq_ignore_ascii_case("content-type")
            && header
                .value
                .trim_start()
                .to_ascii_lowercase()
                .starts_with("text/event-stream")
    })
}

fn sse_body_chunks(body: &str) -> Vec<Bytes> {
    const CHUNK_SIZES: [usize; 7] = [1, 5, 2, 13, 3, 8, 21];

    let mut chunks = Vec::new();
    let mut start = 0;
    let mut size_index = 0;

    while start < body.len() {
        let target_len = CHUNK_SIZES[size_index % CHUNK_SIZES.len()];
        size_index += 1;

        let mut end = (start + target_len).min(body.len());
        while end < body.len() && !body.is_char_boundary(end) {
            end += 1;
        }

        chunks.push(Bytes::copy_from_slice(&body.as_bytes()[start..end]));
        start = end;
    }

    chunks
}

fn is_hop_by_hop_header(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "content-length"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

struct UpstreamBase {
    origin: String,
    path: String,
}

impl UpstreamBase {
    fn parse(real_base_url: &str, cassette_path: &Path) -> Result<Self, CassetteError> {
        let url =
            url::Url::parse(real_base_url).map_err(|source| CassetteError::InvalidBaseUrl {
                path: cassette_path.to_path_buf(),
                url: real_base_url.to_owned(),
                source,
            })?;
        let origin = url.origin().ascii_serialization();
        let path = url.path().trim_end_matches('/');
        let path = if path.is_empty() || path == "/" {
            String::new()
        } else {
            path.to_string()
        };

        Ok(Self { origin, path })
    }
}

/// Resolve a provider/scenario fixture beneath the caller-supplied root.
pub fn cassette_path(cassette_root: &Path, provider: &str, scenario: &str) -> PathBuf {
    let mut path = cassette_root.to_path_buf();
    path.push(provider);
    for segment in scenario.split('/') {
        path.push(sanitize_path_segment(segment));
    }
    path.set_extension("yaml");
    path
}

/// Parse one scenario's interactions in wire order, panicking with the fixture
/// path if reading or deserialization fails.
fn cassette_interactions(
    cassette_root: &Path,
    provider: &str,
    scenario: &str,
) -> Vec<CassetteInteraction> {
    let path = cassette_path(cassette_root, provider, scenario);
    let contents = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable: {error}",
            path.display()
        )
    });

    parse_cassette_interactions(&path, &contents)
}

/// Lowercase one recorded header list, keeping duplicates and wire order.
fn lowercased_header_pairs(headers: Vec<NameValue>) -> Vec<(String, String)> {
    headers
        .into_iter()
        .map(|header| (header.name.to_ascii_lowercase(), header.value))
        .collect()
}

/// Return recorded request and response body strings in wire order, using empty
/// strings for absent bodies. Base64 bodies remain encoded.
/// Panics if the fixture cannot be read or parsed.
pub fn recorded_interaction_bodies(
    cassette_root: &Path,
    provider: &str,
    scenario: &str,
) -> Vec<(String, String)> {
    cassette_interactions(cassette_root, provider, scenario)
        .into_iter()
        .map(|interaction| {
            (
                interaction.when.body.unwrap_or_default(),
                interaction.then.body.unwrap_or_default(),
            )
        })
        .collect()
}

/// Return recorded request and response bodies parsed as JSON in wire order.
/// Panics if the fixture or a body cannot be parsed.
pub fn recorded_json_turns(
    cassette_root: &Path,
    provider: &str,
    scenario: &str,
) -> Vec<(Value, Value)> {
    recorded_interaction_bodies(cassette_root, provider, scenario)
        .into_iter()
        .map(|(request, response)| {
            (
                json_body(&request, provider, scenario, "request"),
                json_body(&response, provider, scenario, "response"),
            )
        })
        .collect()
}

/// The first recorded request and response bodies, as
/// [`recorded_interaction_bodies`] reads them.
fn first_bodies(cassette_root: &Path, provider: &str, scenario: &str) -> (String, String) {
    recorded_interaction_bodies(cassette_root, provider, scenario)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("cassette {provider}/{scenario} should contain an interaction"))
}

/// `body`, the recorded `what` of `provider/scenario`, as JSON.
fn json_body(body: &str, provider: &str, scenario: &str, what: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|error| {
        panic!("cassette {provider}/{scenario} {what} should be JSON: {error}")
    })
}

/// Return a single recorded request and response as JSON.
/// Panics if the fixture is unreadable, has invalid JSON, or has other than one turn.
pub fn recorded_json_turn(cassette_root: &Path, provider: &str, scenario: &str) -> (Value, Value) {
    let mut turns = recorded_json_turns(cassette_root, provider, scenario);
    assert_eq!(
        turns.len(),
        1,
        "cassette {provider}/{scenario} should hold a single completion turn"
    );
    turns.remove(0)
}

/// Return the first recorded request body as JSON.
/// Panics for an unreadable, malformed, or empty fixture or invalid JSON body.
pub fn recorded_json_request(cassette_root: &Path, provider: &str, scenario: &str) -> Value {
    let (request, _) = first_bodies(cassette_root, provider, scenario);
    json_body(&request, provider, scenario, "request")
}

/// Return the first recorded response body as JSON.
/// Panics for an unreadable, malformed, or empty fixture or invalid JSON body.
pub fn recorded_json_response(cassette_root: &Path, provider: &str, scenario: &str) -> Value {
    let (_, response) = first_bodies(cassette_root, provider, scenario);
    json_body(&response, provider, scenario, "response")
}

/// Return recorded request headers in wire order with lowercased names,
/// preserving duplicates. Panics if the fixture cannot be read or parsed.
pub fn recorded_request_header_pairs(
    cassette_root: &Path,
    provider: &str,
    scenario: &str,
) -> Vec<Vec<(String, String)>> {
    cassette_interactions(cassette_root, provider, scenario)
        .into_iter()
        .map(|interaction| lowercased_header_pairs(interaction.when.header))
        .collect()
}

/// Return recorded response headers in wire order with lowercased names,
/// preserving duplicates. Panics if the fixture cannot be read or parsed.
pub fn recorded_response_header_pairs(
    cassette_root: &Path,
    provider: &str,
    scenario: &str,
) -> Vec<Vec<(String, String)>> {
    cassette_interactions(cassette_root, provider, scenario)
        .into_iter()
        .map(|interaction| lowercased_header_pairs(interaction.then.header))
        .collect()
}

/// Return each recorded request's query parameters in wire order.
/// Panics if the fixture cannot be read or parsed.
pub fn recorded_request_query_pairs(
    cassette_root: &Path,
    provider: &str,
    scenario: &str,
) -> Vec<Vec<(String, String)>> {
    cassette_interactions(cassette_root, provider, scenario)
        .into_iter()
        .map(|interaction| {
            interaction
                .when
                .query_param
                .into_iter()
                .map(|pair| (pair.name, pair.value))
                .collect()
        })
        .collect()
}

/// Return recorded request paths in wire order.
/// Panics if the fixture cannot be read or parsed.
pub fn recorded_request_paths(cassette_root: &Path, provider: &str, scenario: &str) -> Vec<String> {
    cassette_interactions(cassette_root, provider, scenario)
        .into_iter()
        .map(|interaction| interaction.when.path)
        .collect()
}

/// Return recorded `(status, body)` pairs in wire order, using an empty string
/// for absent bodies. Panics if the fixture cannot be read or parsed.
pub fn recorded_statuses_and_bodies(
    cassette_root: &Path,
    provider: &str,
    scenario: &str,
) -> Vec<(u16, String)> {
    cassette_interactions(cassette_root, provider, scenario)
        .into_iter()
        .map(|interaction| {
            (
                interaction.then.status,
                interaction.then.body.unwrap_or_default(),
            )
        })
        .collect()
}

/// Return JSON `data:` frames from the first recorded response, excluding `[DONE]`.
/// Panics for an unreadable, malformed, or empty fixture or invalid JSON frame.
pub fn recorded_sse_json_frames(
    cassette_root: &Path,
    provider: &str,
    scenario: &str,
) -> Vec<Value> {
    let (_, response) = first_bodies(cassette_root, provider, scenario);
    let frames = http_client::framing::Framing::Sse.split(response.as_bytes());
    let payloads = frames.iter().map(|frame| frame.as_str());
    payloads
        .filter(|payload| payload != "[DONE]")
        .map(|payload| json_body(&payload, provider, scenario, "SSE frame"))
        .collect()
}

fn sanitize_path_segment(segment: &str) -> String {
    segment
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

/// Return YAML in recorded form: credentials, AWS account numbers and
/// home-directory paths scrubbed, everything the provider sent kept verbatim.
/// Placeholders an older fixture already holds are left in place, so the
/// function is the identity on every committed cassette.
/// Panics if the cassette cannot be parsed.
pub fn scrub_cassette_contents(yaml: &str) -> String {
    let mut interactions = parse_cassette_interactions(Path::new("<cassette>"), yaml);
    let mut scrubber = Scrubber::default();

    scrubber.scrub_interactions(&mut interactions);

    serialize_cassette_interactions(&interactions)
}

/// Return diagnostics for unsanitized or unsafe material in a YAML cassette.
/// Panics if the cassette cannot be parsed.
pub fn cassette_safety_failures(cassette_path: &Path, contents: &str) -> Vec<String> {
    let mut failures = Vec::new();
    let scrubbed = scrub_cassette_contents(contents);

    if scrubbed != contents {
        failures.push(format!(
            "{} is not in scrubbed cassette form",
            cassette_path.display()
        ));
    }

    let decoded = decoded_base64_body_texts(contents);
    let with_decoded = format!("{}\n{}", contents, decoded.join("\n"));
    failures.extend(artifact_safety_failures(cassette_path, &with_decoded));
    failures
}

/// The cassette engine's secret/provider-token checks for JSON sidecars and
/// failure reports. Unlike the cassette validator this does not parse YAML.
pub fn artifact_safety_failures(cassette_path: &Path, contents: &str) -> Vec<String> {
    let mut failures = Vec::new();

    let lower = contents.to_ascii_lowercase();
    for pattern in FORBIDDEN_CASSETTE_PATTERNS {
        if lower.contains(pattern) {
            failures.push(format!("{} contains {pattern:?}", cassette_path.display()));
        }
    }

    // Distinct key-shaped tokens of each kind, named, never echoed.
    let mut keys: Vec<(redact::Kind, BTreeSet<&str>)> = Vec::new();
    for secret in redact::find(contents) {
        let token = contents.get(secret.range).unwrap_or_default();
        match keys.iter_mut().find(|(kind, _)| *kind == secret.kind) {
            Some((_, tokens)) => {
                tokens.insert(token);
            }
            None => keys.push((secret.kind, BTreeSet::from([token]))),
        }
    }
    for (kind, tokens) in keys {
        failures.push(format!(
            "{} contains {} {kind}-shaped token(s)",
            cassette_path.display(),
            tokens.len()
        ));
    }

    failures
}

/// Scrub live diagnostics with the same credential rules as provider
/// traffic. This is separate evidence, never a canonical effect log.
pub fn scrub_artifact(value: &Value) -> Value {
    let mut value = value.clone();
    Scrubber::default().scrub_json_value(None, &mut value);
    value
}

/// One reply in a cassette that [`reply_account_failure`] classifies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CassetteAccountFailure {
    /// The interaction's position, from 0.
    pub index: usize,
    /// The request method and path.
    pub request: String,
    /// The reply's status.
    pub status: u16,
    /// What kind of account failure it is.
    pub failure: AccountFailure,
}

/// Every reply in the cassette `contents` that is an account failure, by
/// [`reply_account_failure`].
/// Panics if the cassette cannot be parsed.
pub fn cassette_account_failures(contents: &str) -> Vec<CassetteAccountFailure> {
    parse_cassette_interactions(Path::new("<cassette>"), contents)
        .iter()
        .enumerate()
        .filter_map(|(index, interaction)| {
            let body = interaction.then.body.as_deref().unwrap_or_default();
            let body = decode_body(body, interaction.then.body_encoding)
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_default();
            reply_account_failure(interaction.then.status, &body).map(|failure| {
                CassetteAccountFailure {
                    index,
                    request: format!("{} {}", interaction.when.method, interaction.when.path),
                    status: interaction.then.status,
                    failure,
                }
            })
        })
        .collect()
}

/// Providers whose Responses API stores a response unless the request sends
/// `store: false`.
pub(crate) const STORING_RESPONSES_PROVIDERS: &[&str] = &["openai", "xai"];

/// The stored responses a cassette for `provider` creates and never deletes:
/// each successful Responses request without `store: false` whose response
/// id no successful `DELETE` in the same cassette removes. Only `openai` and
/// `xai` store. Panics if the cassette cannot be parsed.
pub fn cassette_stored_state(provider: &str, contents: &str) -> Vec<String> {
    if !STORING_RESPONSES_PROVIDERS.contains(&provider) {
        return Vec::new();
    }
    let interactions = parse_cassette_interactions(Path::new("<cassette>"), contents);
    let body_bytes = |body: Option<&str>, encoding: BodyEncoding| {
        decode_body(body.unwrap_or_default(), encoding).unwrap_or_default()
    };
    // Only a DELETE the provider accepted (2xx) removed the response.
    let deleted: Vec<&str> = interactions
        .iter()
        .filter(|interaction| {
            interaction.when.method.eq_ignore_ascii_case("DELETE")
                && (200..300).contains(&interaction.then.status)
        })
        .map(|interaction| interaction.when.path.as_str())
        .collect();
    let mut stored = Vec::new();
    for interaction in &interactions {
        if !(200..300).contains(&interaction.then.status) {
            continue;
        }
        let created = ledger::created_resources(
            provider,
            "",
            &interaction.when.method,
            &interaction.when.path,
            &body_bytes(
                interaction.when.body.as_deref(),
                interaction.when.body_encoding,
            ),
            &body_bytes(
                interaction.then.body.as_deref(),
                interaction.then.body_encoding,
            ),
        );
        for resource in created {
            let removed = deleted.iter().any(|path| {
                path.trim_end_matches('/')
                    .ends_with(&format!("/{}", resource.id))
            });
            if resource.kind == ledger::ResourceKind::Response
                && !removed
                && !stored.contains(&resource.id)
            {
                stored.push(resource.id);
            }
        }
    }
    stored
}

/// Why a finished recording must not become a fixture: each reply that is an
/// account failure the session did not declare, and each response it stored
/// without deleting.
fn recording_refusals(provider: &str, yaml: &str, expected: u8) -> Vec<String> {
    let mut refusals: Vec<String> = cassette_account_failures(yaml)
        .into_iter()
        .filter(|found| expected & found.failure.bit() == 0)
        .map(|found| {
            format!(
                "interaction {} ({}) is an undeclared {:?} failure (status {})",
                found.index, found.request, found.failure, found.status
            )
        })
        .collect();
    let stored = cassette_stored_state(provider, yaml);
    if !stored.is_empty() {
        refusals.push(format!(
            "it stored responses it never deleted ({}); send `store: false`, or delete them in \
             the same session",
            stored.join(", ")
        ));
    }
    refusals
}

/// Where failed and refused recordings go: `RIG_CASSETTE_ATTEMPT_DIR` (a
/// relative path is taken from the current directory), else
/// `cassette-attempts` in the Cargo target directory the running binary was
/// built in, which is where `cargo xtask cassette` looks (a `--target
/// <triple>` build uses `<target>/<triple>` instead). Failing that, an
/// absolute `CARGO_TARGET_DIR`, else the system temporary directory. Never
/// inside the fixture tree.
pub fn attempt_root() -> PathBuf {
    if let Some(dir) = std::env::var_os(ATTEMPT_DIR_ENV) {
        let dir = PathBuf::from(dir);
        return std::env::current_dir()
            .map(|current| current.join(&dir))
            .unwrap_or(dir);
    }
    let target = std::env::current_exe()
        .ok()
        .and_then(|exe| target_dir_of(&exe))
        .or_else(|| {
            std::env::var_os("CARGO_TARGET_DIR")
                .map(PathBuf::from)
                .filter(|dir| dir.is_absolute())
        })
        .unwrap_or_else(std::env::temp_dir);
    target.join("cassette-attempts")
}

/// The Cargo target directory a test or example binary was built in:
/// `<target>/<profile>/deps/<binary>` or `<target>/<profile>/examples/<binary>`.
pub(crate) fn target_dir_of(exe: &Path) -> Option<PathBuf> {
    let kind = exe.parent()?;
    let name = kind.file_name()?;
    (name == "deps" || name == "examples")
        .then(|| kind.parent()?.parent().map(Path::to_path_buf))
        .flatten()
}

/// Scrub `yaml` and write it to a fresh file under `root`:
/// `<provider>/<scenario>.<unix seconds>-<pid>-<n>.yaml`. Returns the path,
/// or `None` when the write failed.
async fn write_attempt(root: &Path, provider: &str, scenario: &str, yaml: &str) -> Option<PathBuf> {
    let redacted = scrub_cassette_contents(yaml);
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = root.join(provider).join(format!(
        "{}.{seconds}-{}-{counter}.yaml",
        scenario
            .split('/')
            .map(sanitize_path_segment)
            .collect::<Vec<_>>()
            .join("/"),
        std::process::id()
    ));
    write_cassette_atomically(&path, redacted.as_bytes())
        .await
        .ok()
        .map(|()| path)
}

async fn write_scrubbed_cassette(cassette_path: &Path, yaml: &str) -> Result<(), CassetteError> {
    let redacted = scrub_cassette_contents(yaml);
    let failures = cassette_safety_failures(cassette_path, &redacted);
    if !failures.is_empty() {
        return Err(CassetteError::UnsafeRecording {
            path: cassette_path.to_path_buf(),
            failures,
        });
    }

    write_cassette_atomically(cassette_path, redacted.as_bytes())
        .await
        .map_err(|source| CassetteError::WriteFixture {
            path: cassette_path.to_path_buf(),
            source,
        })
}

fn decoded_base64_body_texts(contents: &str) -> Vec<String> {
    parse_cassette_interactions(Path::new("<cassette>"), contents)
        .into_iter()
        .flat_map(|interaction| {
            [
                (interaction.when.body, interaction.when.body_encoding),
                (interaction.then.body, interaction.then.body_encoding),
            ]
        })
        .filter(|(_, encoding)| *encoding == BodyEncoding::Base64)
        .filter_map(|(body, _)| decode_body(&body?, BodyEncoding::Base64).ok())
        .map(|body| String::from_utf8_lossy(&body).into_owned())
        .collect()
}

fn serialize_cassette_interactions(interactions: &[CassetteInteraction]) -> String {
    let mut output = String::new();

    for (index, interaction) in interactions.iter().enumerate() {
        if index > 0 {
            output.push_str("---\n");
        }
        output.push_str(
            &serde_yaml::to_string(interaction)
                .expect("scrubbed cassette interaction should serialize"),
        );
    }

    output
}

async fn write_cassette_atomically(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let temp_path = temporary_cassette_path(path);
    let result = async {
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .await?;
        file.write_all(contents).await?;
        file.flush().await?;
        file.sync_all().await?;
        drop(file);
        tokio::fs::rename(&temp_path, path).await
    }
    .await;

    if result.is_err() {
        let _ = tokio::fs::remove_file(&temp_path).await;
    }

    result
}

fn temporary_cassette_path(path: &Path) -> PathBuf {
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("cassette.yaml");
    let temp_name = format!(".{file_name}.tmp-{}-{counter}", std::process::id());

    path.with_file_name(temp_name)
}

const FORBIDDEN_CASSETTE_PATTERNS: &[&str] = &[
    "authorization:",
    "bearer ",
    "x-api-key:",
    "x-goog-api-key:",
    "openai_api_key",
    "anthropic_api_key",
    "gemini_api_key",
    "venice_api_key",
    // Error bodies can echo inference keys, so scan outside credential fields too.
    "venice_inference_key_",
    "__cf_bm=",
    "proj_",
    "set-cookie",
    "cookie:",
    "chatgpt-account-id:",
    "openai-organization",
    "openai-project",
    "anthropic-organization-id",
    "aws4-hmac-sha256",
    "credential=",
    "signedheaders=",
    "x-amz-credential",
    "x-amz-signature",
    "x-amz-security-token",
];

const NO_REQUIRED_REQUEST_HEADERS: &[&str] = &[];
const OPENAI_REQUIRED_REQUEST_HEADERS: &[&str] = &["authorization"];
const CHATGPT_REQUIRED_REQUEST_HEADERS: &[&str] = &["authorization"];
const ANTHROPIC_REQUIRED_REQUEST_HEADERS: &[&str] = &["x-api-key"];
const GEMINI_INTERACTIONS_REQUIRED_REQUEST_HEADERS: &[&str] = &["x-goog-api-key"];

const RECORDED_REQUEST_HEADERS: &[&str] = &[
    "accept",
    "content-type",
    "anthropic-version",
    "anthropic-beta",
    "openai-beta",
];

// AWS SDKs require the error-type header for exception classification and read
// request IDs from headers.
const RESPONSE_HEADER_ALLOWLIST: &[&str] = &[
    "content-type",
    // Replay must retain the provider's retry hint for adapter diagnostics.
    "retry-after",
    "x-amzn-errortype",
    "x-amzn-requestid",
    "request-id",
    "x-request-id",
    "mistral-correlation-id",
];

/// Whether the recorder normalizes `key`, at any depth and in any ASCII
/// case, in recorded JSON bodies (a timestamp that changes on every
/// recording). A recording pass therefore sees a live value the fixture does
/// not hold; compare such keys by JSON type there and exactly in replay.
pub fn is_volatile_json_key(key: &str) -> bool {
    VOLATILE_JSON_KEYS
        .iter()
        .any(|volatile| volatile.eq_ignore_ascii_case(key))
}

/// Every key [`is_volatile_json_key`] accepts.
pub fn volatile_json_keys() -> &'static [&'static str] {
    VOLATILE_JSON_KEYS
}

/// The cassette's own steps of the shared [`Scrubber`].
impl Scrubber {
    fn scrub_interactions(&mut self, interactions: &mut [CassetteInteraction]) {
        for interaction in interactions {
            self.learn_response(&interaction.then);
            self.scrub_request(&mut interaction.when);
            self.scrub_response(&mut interaction.then);
        }
    }

    fn scrub_request(&mut self, request: &mut CassetteRequest) {
        for header in &request.header {
            self.learn_header(&header.name, &header.value);
        }
        for pair in &request.query_param {
            self.learn_param(&pair.name, &pair.value);
        }
        if let Some(body) = &request.body {
            self.learn_encoded_body(body, request.body_encoding);
        }
        request.path = self.scrub_text(&request.path);
        scrub_headers(&mut request.header, HeaderMode::Request);
        for header in &mut request.header {
            header.value = self.scrub_text(&header.value);
        }
        scrub_query_params(&mut request.query_param);

        for query_param in &mut request.query_param {
            query_param.value = self.scrub_text(&query_param.value);
        }

        if let Some(body) = &mut request.body {
            *body = self.scrub_encoded_body(body, request.body_encoding);
        }
    }

    fn learn_response(&mut self, response: &CassetteResponse) {
        for header in &response.header {
            self.learn_header(&header.name, &header.value);
        }
        if let Some(body) = &response.body {
            self.learn_encoded_body(body, response.body_encoding);
        }
    }

    fn learn_encoded_body(&mut self, body: &str, encoding: BodyEncoding) {
        if let Ok(bytes) = decode_body(body, encoding)
            && let Ok(text) = std::str::from_utf8(&bytes)
        {
            self.learn_body(text);
        }
    }

    fn scrub_response(&mut self, response: &mut CassetteResponse) {
        self.learn_response(response);
        scrub_headers(&mut response.header, HeaderMode::Response);
        for header in &mut response.header {
            header.value = self.scrub_text(&header.value);
        }

        // A retry hint is protocol data, not an arbitrary diagnostic string.
        // Canonicalize valid values and discard malformed/credential-bearing text.
        response.header.retain_mut(|header| {
            if !header.name.eq_ignore_ascii_case("retry-after") {
                return true;
            }
            let value = header.value.trim();
            let canonical = if value.len() <= 20 && value.bytes().all(|b| b.is_ascii_digit()) {
                value.parse::<u64>().ok().map(|seconds| seconds.to_string())
            } else if value.len() <= 128 {
                httpdate::parse_http_date(value)
                    .ok()
                    .map(httpdate::fmt_http_date)
            } else {
                None
            };
            if let Some(value) = canonical {
                header.value = value;
                true
            } else {
                false
            }
        });

        if let Some(body) = &mut response.body {
            *body = self.scrub_encoded_body(body, response.body_encoding);
        }
    }

    fn scrub_encoded_body(&mut self, body: &str, encoding: BodyEncoding) -> String {
        match encoding {
            BodyEncoding::Utf8 => self.scrub_body(body),
            BodyEncoding::Base64 => {
                let Ok(bytes) = decode_body(body, BodyEncoding::Base64) else {
                    return body.to_string();
                };
                if let Ok(text) = std::str::from_utf8(&bytes) {
                    return BASE64_STANDARD.encode(self.scrub_body(text));
                }

                #[cfg(feature = "bedrock")]
                {
                    self.scrub_event_stream_body(bytes)
                        .map_or_else(|| body.to_string(), |bytes| BASE64_STANDARD.encode(bytes))
                }
                #[cfg(not(feature = "bedrock"))]
                {
                    body.to_string()
                }
            }
        }
    }

    #[cfg(feature = "bedrock")]
    fn scrub_event_stream_body(&mut self, bytes: Vec<u8>) -> Option<Vec<u8>> {
        let mut input = Bytes::from(bytes);
        let mut output = Vec::new();

        while !input.is_empty() {
            let message = read_message_from(&mut input).ok()?;
            let payload = std::str::from_utf8(message.payload()).ok()?;
            let scrubbed_payload = self.scrub_body(payload);
            let scrubbed = EventStreamMessage::new_from_parts(
                message.headers().to_vec(),
                scrubbed_payload.into_bytes(),
            );
            write_message_to(&scrubbed, &mut output).ok()?;
        }

        Some(output)
    }
}

#[derive(Clone, Copy)]
enum HeaderMode {
    Request,
    Response,
}

fn scrub_headers(headers: &mut Vec<NameValue>, mode: HeaderMode) {
    match mode {
        HeaderMode::Request => {
            for header in headers.iter_mut() {
                if is_secret_header(&header.name) {
                    header.value = REDACTED.to_string();
                }
            }
        }
        HeaderMode::Response => {
            headers.retain(|header| {
                contains_case_insensitive(RESPONSE_HEADER_ALLOWLIST, &header.name)
            });
        }
    }
    // A key-shaped token in any header that stays is a secret too.
    for header in headers {
        if let Cow::Owned(value) = redact::text(&header.value) {
            header.value = value;
        }
    }
}

fn scrub_query_params(query_params: &mut [NameValue]) {
    for query_param in query_params {
        if is_secret_param(&query_param.name) {
            query_param.value = REDACTED.to_string();
        }
    }
}

#[cfg(test)]
mod tests;

/// Collect UTF-8 HTTP headers as owned name/value pairs.
pub fn owned_headers(headers: &http_client::HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_string(), value.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod explicit_destination_tests;
#[cfg(test)]
mod paths;
#[cfg(test)]
mod recording_guard_tests;
#[cfg(test)]
mod replay_session_tests;
#[cfg(test)]
mod try_session_tests;

#[cfg(test)]
mod local_path_scrub_tests;
