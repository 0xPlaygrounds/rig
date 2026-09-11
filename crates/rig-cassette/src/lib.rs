//! Cassette-backed provider test helpers.
//!
//! Provider cassette tests run in replay mode by default. Set
//! `RIG_PROVIDER_TEST_MODE=record` to hit the real provider and write cassette
//! fixtures. Record mode overwrites existing cassette files.
//!
//! Every constructor and recorded-request reader receives its fixture root
//! explicitly. The root contains provider directories; this crate does not
//! depend on a repository layout or consumer registry. `start_at` instead takes
//! an exact destination and mode, for controlled offline/candidate workflows.
//! Enable `bedrock` to scrub binary Smithy event-stream payloads. Ordinary
//! binary bodies and SSE work without that feature. This is native test support;
//! invalid fixtures and failed replay assertions deliberately panic.
// Preserve the existing assertion-based test-support contract during extraction.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unreachable
)]
#![warn(missing_docs)]

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
use futures::{FutureExt, stream};
use httpmock::MockServer;
use rig_core::http_client::{
    self, HttpClientExt, LazyBody, MultipartForm, Request as HttpRequest, Response as HttpResponse,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::any::Any;
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::fmt;
use std::fs;
use std::net::SocketAddr;
use std::panic::{AssertUnwindSafe, resume_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;

const MODE_ENV: &str = "RIG_PROVIDER_TEST_MODE";
const REDACTED: &str = "[REDACTED]";
/// Stand-in for a generated image payload (`"hello"` in base64).
const IMAGE_PAYLOAD_PLACEHOLDER: &str = "aGVsbG8=";
const DUMMY_API_KEY: &str = REDACTED;
static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

type PanicPayload = Box<dyn Any + Send + 'static>;

#[derive(Clone, Copy, Debug)]
struct TokenPrefix {
    raw: &'static str,
    placeholder_prefix: &'static str,
    min_suffix_len: usize,
}

impl TokenPrefix {
    const fn new(
        raw: &'static str,
        placeholder_prefix: &'static str,
        min_suffix_len: usize,
    ) -> Self {
        Self {
            raw,
            placeholder_prefix,
            min_suffix_len,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplayMatching {
    Ordered,
    Unordered,
}

/// The fixture scenario and its ordered or unordered replay matching policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CassetteSpec {
    scenario: &'static str,
    replay_matching: ReplayMatching,
}

impl CassetteSpec {
    /// Select a scenario with strict wire-order matching.
    pub const fn new(scenario: &'static str) -> Self {
        Self {
            scenario,
            replay_matching: ReplayMatching::Ordered,
        }
    }

    /// Allow matching an unused interaction in any order.
    pub const fn unordered(mut self) -> Self {
        self.replay_matching = ReplayMatching::Unordered;
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
    recorded_request_headers: &'static [&'static str],
    required_request_headers: &'static [&'static str],
    sensitive_headers: &'static [&'static str],
    sensitive_query_params: &'static [&'static str],
    response_header_allowlist: &'static [&'static str],
    forbidden_patterns: &'static [&'static str],
    generated_token_prefixes: &'static [TokenPrefix],
    replay_matching: ReplayMatching,
}

impl CassettePolicy {
    fn for_scenario(provider: &str, scenario: &str, replay_matching: ReplayMatching) -> Self {
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
            replay_matching,
            ..Self::default()
        }
    }

    fn is_sensitive_header(self, name: &str) -> bool {
        contains_case_insensitive(self.sensitive_headers, name)
    }

    fn required_request_headers(self) -> &'static [&'static str] {
        self.required_request_headers
    }

    fn is_sensitive_query_param(self, name: &str) -> bool {
        contains_case_insensitive(self.sensitive_query_params, name)
    }

    fn is_allowed_response_header(self, name: &str) -> bool {
        contains_case_insensitive(self.response_header_allowlist, name)
    }

    fn generated_prefix_for(self, value: &str) -> Option<TokenPrefix> {
        // Callers pass a value taken from a known id field, so the
        // id-position gate is satisfied by construction.
        self.generated_token_prefixes
            .iter()
            .copied()
            .find(|prefix| is_generated_token(value, *prefix, true))
    }

    fn matching_generated_prefix(self, text: &str, index: usize) -> Option<TokenPrefix> {
        if index > 0 {
            let previous = text[..index].chars().next_back()?;
            if is_token_char(previous) {
                return None;
            }
        }

        self.generated_token_prefixes
            .iter()
            .copied()
            .find(|prefix| text[index..].starts_with(prefix.raw))
    }
}

impl Default for CassettePolicy {
    fn default() -> Self {
        Self {
            recorded_request_headers: RECORDED_REQUEST_HEADERS,
            required_request_headers: NO_REQUIRED_REQUEST_HEADERS,
            sensitive_headers: SENSITIVE_HEADER_NAMES,
            sensitive_query_params: SENSITIVE_QUERY_PARAMS,
            response_header_allowlist: RESPONSE_HEADER_ALLOWLIST,
            forbidden_patterns: FORBIDDEN_CASSETTE_PATTERNS,
            generated_token_prefixes: GENERATED_TOKEN_PREFIXES,
            replay_matching: ReplayMatching::Ordered,
        }
    }
}

fn contains_case_insensitive(values: &[&str], needle: &str) -> bool {
    values
        .iter()
        .any(|value| needle.eq_ignore_ascii_case(value))
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
    /// Read RIG_PROVIDER_TEST_MODE, defaulting to replay; reject unknown values.
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

/// Recorder for transports that bypass the HTTP proxy, including binary bodies.
#[derive(Clone, Debug)]
pub struct DirectRecorder {
    interactions: Arc<Mutex<Vec<CassetteInteraction>>>,
    policy: CassettePolicy,
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
    /// Scrub and append one complete direct request/response exchange.
    pub async fn record_http_interaction<RequestHeaders, ResponseHeaders>(
        &self,
        request: DirectHttpRequest<'_, RequestHeaders>,
        response: DirectHttpResponse<'_, ResponseHeaders>,
    ) where
        RequestHeaders: IntoIterator,
        RequestHeaders::Item: DirectHeader,
        ResponseHeaders: IntoIterator,
        ResponseHeaders::Item: DirectHeader,
    {
        let mut scrubber = CassetteScrubber::new(self.policy);
        let mut interaction = CassetteInteraction {
            when: recorded_request(
                self.policy,
                request.method,
                request.uri,
                request.headers.into_iter().map(DirectHeader::into_pair),
                request.body,
            ),
            then: recorded_response(
                response.status,
                response.headers.into_iter().map(DirectHeader::into_pair),
                response.body,
            ),
        };
        scrubber.scrub_request(&mut interaction.when);
        scrubber.scrub_response(&mut interaction.then);
        self.interactions.lock().await.push(interaction);
    }
}

/// Convert a direct-recording header into its name and value.
pub trait DirectHeader {
    /// The header name representation.
    type Name: AsRef<str>;
    /// The header value representation.
    type Value: AsRef<str>;

    /// Return the header name and value.
    fn into_pair(self) -> (Self::Name, Self::Value);
}

impl<Name, Value> DirectHeader for (Name, Value)
where
    Name: AsRef<str>,
    Value: AsRef<str>,
{
    type Name = Name;
    type Value = Value;

    /// Return the header name and value.
    fn into_pair(self) -> (Self::Name, Self::Value) {
        self
    }
}

/// A recording or replay session with an explicitly located fixture.
pub struct ProviderCassette {
    server: CassetteServer,
    cassette_path: PathBuf,
    base_path: String,
    mode: CassetteMode,
    policy: CassettePolicy,
    recording_id: Option<usize>,
}

enum CassetteServer {
    Recording(MockServer),
    DirectRecording(DirectRecordingServer),
    Replay(ReplayServer),
}

struct DirectRecordingServer {
    base_url: String,
    interactions: Arc<Mutex<Vec<CassetteInteraction>>>,
}

impl CassetteServer {
    fn base_url(&self) -> String {
        match self {
            Self::Recording(server) => server.base_url(),
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
    /// Start with an explicit cassette root containing provider directories.
    pub async fn start(
        cassette_root: &Path,
        provider: &'static str,
        spec: impl Into<CassetteSpec>,
        real_base_url: &str,
    ) -> Self {
        let spec = spec.into();
        Self::start_at(
            provider,
            spec,
            real_base_url,
            CassetteMode::current(),
            cassette_path(cassette_root, provider, spec.scenario),
        )
        .await
    }

    /// Explicit mode and destination for consumers that stage candidates.
    /// Verification callers pass Replay even if the ambient environment asks
    /// for recording; a candidate is never implicitly promoted to a fixture.
    pub async fn start_at(
        provider: &'static str,
        spec: CassetteSpec,
        real_base_url: &str,
        mode: CassetteMode,
        cassette_path: PathBuf,
    ) -> Self {
        let scenario = spec.scenario;
        let policy = CassettePolicy::for_scenario(provider, scenario, spec.replay_matching);
        let upstream = UpstreamBase::parse(real_base_url);
        let (server, recording_id) = if mode.records() {
            let server = MockServer::start_async().await;
            server
                .forward_to_async(&upstream.origin, |rule| {
                    rule.filter(|when| {
                        when.any_request();
                    });
                })
                .await;

            let recorded_request_headers = policy.recorded_request_headers.to_vec();
            let recording = server
                .record_async(move |rule| {
                    rule.record_request_headers(recorded_request_headers)
                        .filter(|when| {
                            when.any_request();
                        });
                })
                .await;

            let recording_id = recording.id;

            (CassetteServer::Recording(server), Some(recording_id))
        } else {
            if !cassette_path.exists() {
                panic!(
                    "missing provider cassette {}; run with {MODE_ENV}=record and the real API key to create it",
                    cassette_path.display()
                );
            }
            (
                CassetteServer::Replay(ReplayServer::start(&cassette_path, policy).await),
                None,
            )
        };

        Self {
            server,
            cassette_path,
            base_path: upstream.path,
            mode,
            policy,
            recording_id,
        }
    }

    /// Start direct recording or replay under the supplied cassette root.
    pub async fn start_direct_recording(
        cassette_root: &Path,
        provider: &'static str,
        spec: impl Into<CassetteSpec>,
        real_base_url: &str,
    ) -> Self {
        let spec = spec.into();
        let scenario = spec.scenario;
        let mode = CassetteMode::current();
        let policy = CassettePolicy::for_scenario(provider, scenario, spec.replay_matching);
        let cassette_path = cassette_path(cassette_root, provider, scenario);
        let upstream = UpstreamBase::parse(real_base_url);
        let (server, recording_id) = if mode.records() {
            let interactions = Arc::new(Mutex::new(Vec::new()));
            (
                CassetteServer::DirectRecording(DirectRecordingServer {
                    base_url: upstream.origin.clone(),
                    interactions,
                }),
                None,
            )
        } else {
            if !cassette_path.exists() {
                panic!(
                    "missing provider cassette {}; run with {MODE_ENV}=record and the real API key to create it",
                    cassette_path.display()
                );
            }
            (
                CassetteServer::Replay(ReplayServer::start(&cassette_path, policy).await),
                None,
            )
        };

        Self {
            server,
            cassette_path,
            base_path: upstream.path,
            mode,
            policy,
            recording_id,
        }
    }

    /// Return the direct recorder only for an active direct-recording session.
    pub fn direct_recorder(&self) -> Option<DirectRecorder> {
        match &self.server {
            CassetteServer::DirectRecording(server) => Some(DirectRecorder {
                interactions: server.interactions.clone(),
                policy: self.policy,
            }),
            _ => None,
        }
    }

    /// Return the session endpoint, preserving the upstream base path.
    pub fn base_url(&self) -> String {
        format!("{}{}", self.server.base_url(), self.base_path)
    }

    /// A deliberately invalid API key for recording real auth failures: the
    /// bogus literal in record mode, the dummy key in replay — so providers
    /// that carry the key in a *matched* location (Gemini's query string)
    /// still replay (the recorded value is scrubbed either way).
    pub fn bogus_api_key(&self) -> String {
        if self.mode.records() {
            "invalid-edge-matrix-key".to_string()
        } else {
            DUMMY_API_KEY.to_string()
        }
    }

    /// Use the named environment key when recording, or a dummy key for replay.
    pub fn api_key(&self, env_name: &str) -> String {
        if self.mode.records() {
            std::env::var(env_name).unwrap_or_else(|_| {
                panic!("{env_name} must be set when {MODE_ENV}={:?}", self.mode)
            })
        } else {
            DUMMY_API_KEY.to_string()
        }
    }

    /// Retain completed, scrubbed exchanges while a live consumer is still
    /// running. This is an unaccepted partial recording, not finalization.
    pub async fn checkpoint_recording(&self, path: &Path) -> bool {
        if !self.mode.records() {
            return false;
        }
        let yaml = match &self.server {
            CassetteServer::Recording(server) => {
                let Some(id) = self.recording_id else {
                    return false;
                };
                let recording = httpmock::Recording::new(id, server);
                let Ok(Some(bytes)) = recording.export_async().await else {
                    return false;
                };
                let Ok(yaml) = String::from_utf8(bytes.to_vec()) else {
                    return false;
                };
                yaml
            }
            CassetteServer::DirectRecording(server) => {
                let interactions = server.interactions.lock().await;
                if interactions.is_empty() {
                    return false;
                }
                serialize_cassette_interactions(&interactions)
            }
            CassetteServer::Replay(_) => return false,
        };
        // httpmock can export an empty YAML document before the first response.
        // A best-effort snapshot must not panic or replace a previous snapshot.
        let parsed = serde_yaml::Deserializer::from_str(&yaml)
            .map(CassetteInteraction::deserialize)
            .collect::<Result<Vec<_>, _>>();
        let Ok(interactions) = parsed else {
            return false;
        };
        if interactions.is_empty() {
            return false;
        }
        let redacted = scrub_cassette_contents_with_policy(self.policy, &yaml);
        if !cassette_safety_failures_with_policy(self.policy, path, &redacted).is_empty() {
            return false;
        }
        write_cassette_atomically(path, redacted.as_bytes())
            .await
            .is_ok()
    }

    /// Finalize a recording or assert complete replay consumption and shut down.
    pub async fn finish(self) {
        let Self {
            server,
            cassette_path,
            policy,
            recording_id,
            ..
        } = self;

        let server = match server {
            CassetteServer::Replay(mut server) => {
                let result = AssertUnwindSafe(server.assert_consumed(&cassette_path))
                    .catch_unwind()
                    .await;
                server.shutdown().await;
                if let Err(payload) = result {
                    resume_unwind(payload);
                }
                return;
            }
            CassetteServer::DirectRecording(server) => {
                let yaml = {
                    let interactions = server.interactions.lock().await;
                    assert!(
                        !interactions.is_empty(),
                        "provider cassette {} should contain at least one interaction",
                        cassette_path.display()
                    );
                    serialize_cassette_interactions(&interactions)
                };
                write_scrubbed_cassette(&cassette_path, policy, &yaml).await;
                return;
            }
            CassetteServer::Recording(server) => server,
        };

        let Some(recording_id) = recording_id else {
            return;
        };

        let recording = httpmock::Recording::new(recording_id, &server);
        let bytes = recording
            .export_async()
            .await
            .expect("provider cassette should export")
            .expect("provider cassette should contain at least one interaction");
        let yaml = String::from_utf8(bytes.to_vec()).expect("cassette YAML should be UTF-8");
        write_scrubbed_cassette(&cassette_path, policy, &yaml).await;
    }

    /// Finalize after a successful test, preserving its original panic otherwise.
    pub async fn finish_after_test(self, test_result: Result<(), PanicPayload>) {
        match test_result {
            Ok(()) => {
                if let Err(payload) = self.finish_catching_unwind().await {
                    resume_unwind(payload);
                }
            }
            Err(payload) => {
                resume_unwind(payload);
            }
        }
    }

    /// Finalize after a successful fallible test, preserving its failure otherwise.
    pub async fn finish_after_test_result<E>(
        self,
        test_result: Result<Result<(), E>, PanicPayload>,
    ) -> Result<(), E> {
        match test_result {
            Ok(Ok(())) => {
                if let Err(payload) = self.finish_catching_unwind().await {
                    resume_unwind(payload);
                }
                Ok(())
            }
            Ok(Err(error)) => Err(error),
            Err(payload) => resume_unwind(payload),
        }
    }

    async fn finish_catching_unwind(self) -> Result<(), PanicPayload> {
        AssertUnwindSafe(self.finish()).catch_unwind().await
    }
}

struct ReplayServer {
    addr: SocketAddr,
    state: Arc<Mutex<ReplayState>>,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

impl ReplayServer {
    async fn start(cassette_path: &Path, policy: CassettePolicy) -> Self {
        let contents = tokio::fs::read_to_string(cassette_path)
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "provider cassette {} should be readable: {error}",
                    cassette_path.display()
                )
            });
        let interactions = parse_cassette(cassette_path, &contents);
        let state = Arc::new(Mutex::new(ReplayState {
            cassette_path: cassette_path.to_path_buf(),
            interactions,
            misses: Vec::new(),
            policy,
        }));
        let app = Router::new()
            .fallback(any(replay_request))
            .with_state(state.clone());
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("replay server should bind");
        let addr = listener
            .local_addr()
            .expect("replay server address should be available");

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

        Self {
            addr,
            state,
            shutdown: Some(shutdown_tx),
            task: Some(task),
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    async fn assert_consumed(&self, cassette_path: &Path) {
        let state = self.state.lock().await;
        assert_replay_finished(cassette_path, &state.interactions, &state.misses);
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
    }
}

struct ReplayState {
    cassette_path: PathBuf,
    interactions: Vec<ReplayInteraction>,
    misses: Vec<ReplayMiss>,
    policy: CassettePolicy,
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

#[derive(Debug, Deserialize, Serialize)]
struct NameValue {
    name: String,
    value: String,
}

struct ReplayInteraction {
    when: CassetteRequest,
    then: CassetteResponse,
    consumed: bool,
}

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
    let mut failures = Vec::new();
    let unused = interactions
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

    if !unused.is_empty() {
        failures.push(format!("left unused interactions:\n{}", unused.join("\n")));
    }

    if !misses.is_empty() {
        let formatted_misses = misses
            .iter()
            .enumerate()
            .map(|(index, miss)| format!("[{index}] {}", miss.diagnostic))
            .collect::<Vec<_>>()
            .join("\n");
        failures.push(format!(
            "received unexpected replay request(s):\n{formatted_misses}"
        ));
    }

    (!failures.is_empty()).then(|| {
        format!(
            "provider cassette replay failed for {}:\n{}",
            cassette_path.display(),
            failures.join("\n\n")
        )
    })
}

fn parse_cassette(cassette_path: &Path, contents: &str) -> Vec<ReplayInteraction> {
    parse_cassette_interactions(cassette_path, contents)
        .into_iter()
        .map(|interaction| ReplayInteraction {
            when: interaction.when,
            then: interaction.then,
            consumed: false,
        })
        .collect()
}

fn parse_cassette_interactions(cassette_path: &Path, contents: &str) -> Vec<CassetteInteraction> {
    serde_yaml::Deserializer::from_str(contents)
        .map(|document| {
            CassetteInteraction::deserialize(document).unwrap_or_else(|error| {
                panic!(
                    "provider cassette {} should deserialize: {error}",
                    cassette_path.display()
                )
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

    let Some(index) = matching_interaction_index(policy, &state.interactions, &request) else {
        let message = replay_miss_message(policy, &request, &state.interactions);
        state.misses.push(ReplayMiss {
            diagnostic: message.clone(),
        });
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("content-type", "application/json")
            .body(Body::from(message))
            .expect("replay miss response should build");
    };

    let cassette_path = state.cassette_path.clone();
    let interaction = &mut state.interactions[index];
    interaction.consumed = true;
    cassette_response(&interaction.then, &cassette_path)
}

fn matching_interaction_index(
    policy: CassettePolicy,
    interactions: &[ReplayInteraction],
    request: &IncomingRequest,
) -> Option<usize> {
    match policy.replay_matching {
        ReplayMatching::Ordered => {
            let index = interactions
                .iter()
                .position(|interaction| !interaction.consumed)?;
            request_matches(policy, request, &interactions[index].when).then_some(index)
        }
        ReplayMatching::Unordered => interactions.iter().position(|interaction| {
            !interaction.consumed && request_matches(policy, request, &interaction.when)
        }),
    }
}

fn replay_miss_message(
    policy: CassettePolicy,
    request: &IncomingRequest,
    interactions: &[ReplayInteraction],
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
                policy,
                &request.headers,
                &interaction.when.header,
                &request.body,
                interaction.when.body.as_deref(),
                interaction.when.body_encoding,
            );

            json!({
                "index": index,
                "consumed": interaction.consumed,
                "method_matches": method_matches,
                "path_matches": path_matches,
                "query_matches": query_matches,
                "expected_query_params": scrub_name_values_for_diagnostics(policy, &interaction.when.query_param),
                "headers_match": headers_match,
                "required_headers_match": required_headers_match,
                "body_matches": body_matches,
                "expected_method": interaction.when.method,
                "expected_path": interaction.when.path,
                "expected_body_preview": interaction.when.body.as_deref().map(|body| body_preview_for_diagnostics(policy, body)),
            })
        })
        .collect::<Vec<_>>();

    json!({
        "message": "Request did not match any route or mock",
        "actual_method": request.method.as_str(),
        "actual_path": request.uri.path(),
        "actual_query": request.uri.query().map(|query| scrub_text_for_diagnostics(policy, query)),
        "actual_query_params": scrub_query_pairs_for_diagnostics(policy, request.uri.query()),
        "required_headers": policy.required_request_headers(),
        "missing_required_headers": missing_required_headers(policy, &request.headers),
        "actual_body_preview": body_preview_bytes_for_diagnostics(policy, &request.body),
        "candidates": candidates,
    })
    .to_string()
}

struct IncomingRequest {
    method: Method,
    uri: axum::http::Uri,
    headers: axum::http::HeaderMap,
    body: Bytes,
}

fn request_matches(
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
        && body_matches(
            policy,
            &request.headers,
            &expected.header,
            &request.body,
            expected.body.as_deref(),
            expected.body_encoding,
        )
}

fn recorded_request<N, V>(
    policy: CassettePolicy,
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
        header: recorded_request_headers(policy, headers),
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

fn recorded_request_headers<N, V>(
    policy: CassettePolicy,
    headers: impl IntoIterator<Item = (N, V)>,
) -> Vec<NameValue>
where
    N: AsRef<str>,
    V: AsRef<str>,
{
    headers
        .into_iter()
        .filter_map(|(name, value)| {
            let name = name.as_ref();
            contains_case_insensitive(policy.recorded_request_headers, name).then(|| NameValue {
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
    let actual = parsed_query_pairs(query);
    query_pair_counts(actual)
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
                    value == expected.value
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
        .required_request_headers()
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
    policy: CassettePolicy,
    actual_headers: &axum::http::HeaderMap,
    expected_headers: &[NameValue],
    actual: &[u8],
    expected: Option<&str>,
    expected_encoding: BodyEncoding,
) -> bool {
    let Some(expected) = expected else {
        // A cassette recorded through the httpmock proxy stores bodies as
        // strings, so a non-UTF-8 multipart upload (an audio file posted to a
        // transcription endpoint) is exported with no body at all. Requiring
        // an empty request body there would make every such scenario
        // unreplayable; the multipart *shape* those endpoints receive is
        // pinned by unit tests next to each provider instead. Non-multipart
        // requests still have to be body-less to match.
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
    let actual = CassetteScrubber::new(policy).scrub_body(actual);
    let expected = CassetteScrubber::new(policy).scrub_body(expected);

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
        let raw_part = strip_prefix_bytes(raw_part, b"\r\n");
        if raw_part.starts_with(b"--") {
            break;
        }

        let raw_part = strip_suffix_bytes(raw_part, b"\r\n");
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

fn find_bytes(body: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }

    body.windows(needle.len())
        .position(|window| window == needle)
}

fn strip_prefix_bytes<'a>(body: &'a [u8], prefix: &[u8]) -> &'a [u8] {
    body.strip_prefix(prefix).unwrap_or(body)
}

fn strip_suffix_bytes<'a>(body: &'a [u8], suffix: &[u8]) -> &'a [u8] {
    body.strip_suffix(suffix).unwrap_or(body)
}

fn normalize_multipart_header_value(value: &str) -> String {
    value
        .split(';')
        .map(str::trim)
        .collect::<Vec<_>>()
        .join("; ")
}

fn canonical_json(body: &str) -> Option<Value> {
    serde_json::from_str::<Value>(body)
        .ok()
        .map(sort_json_objects)
}

fn sort_json_objects(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| (key, sort_json_objects(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(sort_json_objects).collect()),
        value => value,
    }
}

fn body_preview(body: &str) -> String {
    const LIMIT: usize = 512;
    let mut preview = body.chars().take(LIMIT).collect::<String>();
    if body.chars().count() > LIMIT {
        preview.push_str("...");
    }
    preview
}

fn body_preview_for_diagnostics(policy: CassettePolicy, body: &str) -> String {
    body_preview(&scrub_body_for_diagnostics(policy, body))
}

fn body_preview_bytes_for_diagnostics(policy: CassettePolicy, body: &[u8]) -> String {
    match std::str::from_utf8(body) {
        Ok(body) => body_preview_for_diagnostics(policy, body),
        Err(_) => format!("<{} bytes of non-UTF-8 body>", body.len()),
    }
}

fn scrub_body_for_diagnostics(policy: CassettePolicy, body: &str) -> String {
    CassetteScrubber::new(policy).scrub_body(body)
}

/// Replace the directory part of an absolute home-directory path, keeping the
/// final component.
///
/// Locally-hosted providers echo the path they were launched with. `llama-server`
/// puts it in `model` on **every** chat response — the whole
/// `/Users/<name>/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<sha>/<file>.gguf`
/// — so recording one turn against it writes the operator's username, their
/// cache layout and a snapshot hash into a fixture that is then committed to a
/// public repository. Nothing else in the scrubber reaches this: it is not a
/// token, not a header, not a query parameter, and it trips no
/// `FORBIDDEN_CASSETTE_PATTERNS` entry, so the safety scan passes it.
///
/// The final component survives because it is the useful, non-identifying part
/// — which model the fixture was recorded against — and because a cell may
/// legitimately assert on it. Everything to its left is replaced.
///
/// The match is anchored to the **start of a JSON string value**, which is what
/// separates a local path from a URL that merely contains one of these
/// segments. Anthropic's recorded web-search results cite
/// `https://math.ucr.edu/home/baez/physics/...`; an unanchored rule rewrites
/// that public URL and corrupts the fixture, which the safety scan catches as
/// "not in scrubbed cassette form". A real local path occupies the entire value
/// (`"model":"/Users/…"`), so requiring a `"` immediately before it keeps the
/// rule precise.
fn scrub_local_filesystem_paths(text: &str) -> String {
    const HOME_PREFIXES: &[&str] = &["/Users/", "/home/", "/root/"];

    let mut output = String::with_capacity(text.len());
    let mut rest = text;
    let mut consumed = 0usize;

    'outer: while let Some((prefix, at)) = HOME_PREFIXES
        .iter()
        .filter_map(|p| rest.find(*p).map(|i| (*p, i)))
        .min_by_key(|(_, i)| *i)
    {
        // Anchored: the value must begin here. Anything else is a path segment
        // inside a larger string (a URL, prose) and is not ours to rewrite.
        let absolute = consumed + at;
        let starts_value = absolute == 0 || text.as_bytes().get(absolute - 1) == Some(&b'"');
        if !starts_value {
            let skip = at + prefix.len();
            output.push_str(&rest[..skip]);
            rest = &rest[skip..];
            consumed = absolute + prefix.len();
            continue;
        }

        output.push_str(&rest[..at]);
        let path = &rest[at..];
        // A path ends at the closing quote of its JSON string, and at nothing
        // else. Whitespace is emphatically *not* a terminator: macOS home
        // directories created from a full account name routinely contain a
        // space, and ending the path at it would keep the first word as the
        // "basename" and copy `Smith/models/m.gguf` through verbatim — writing
        // the operator's name into the fixture while leaving output the rule
        // considers already-scrubbed, so nothing downstream would notice.
        let end = path.find(['"', '\\', '\n', '\r']).unwrap_or(path.len());
        let (path, tail) = path.split_at(end);

        // Keep the basename; replace everything before it.
        match path.rsplit_once('/') {
            Some((_, base)) if !base.is_empty() => {
                output.push_str("/REDACTED_PATH/");
                output.push_str(base);
            }
            _ => output.push_str("/REDACTED_PATH"),
        }
        consumed = absolute + path.len();
        rest = tail;
        if rest.is_empty() {
            break 'outer;
        }
    }

    output.push_str(rest);
    output
}

fn scrub_text_for_diagnostics(policy: CassettePolicy, text: &str) -> String {
    CassetteScrubber::new(policy).scrub_text(text)
}

fn scrub_query_pairs_for_diagnostics(
    policy: CassettePolicy,
    query: Option<&str>,
) -> Vec<NameValue> {
    let mut pairs = parsed_query_pairs(query)
        .into_iter()
        .map(|(name, value)| NameValue { name, value })
        .collect::<Vec<_>>();
    scrub_query_params(policy, &mut pairs);
    for pair in &mut pairs {
        pair.value = scrub_text_for_diagnostics(policy, &pair.value);
    }
    pairs
}

fn scrub_name_values_for_diagnostics(
    policy: CassettePolicy,
    values: &[NameValue],
) -> Vec<NameValue> {
    let mut values = values
        .iter()
        .map(|value| NameValue {
            name: value.name.clone(),
            value: value.value.clone(),
        })
        .collect::<Vec<_>>();
    scrub_query_params(policy, &mut values);
    for value in &mut values {
        value.value = scrub_text_for_diagnostics(policy, &value.value);
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
    fragmented_sse_body_chunks(body)
        .into_iter()
        .map(Bytes::from)
        .collect()
}

fn fragmented_sse_body_chunks(body: &str) -> Vec<String> {
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

        chunks.push(body[start..end].to_string());
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
    fn parse(real_base_url: &str) -> Self {
        let url = url::Url::parse(real_base_url)
            .unwrap_or_else(|error| panic!("invalid provider base URL {real_base_url:?}: {error}"));
        let origin = url.origin().ascii_serialization();
        let path = url.path().trim_end_matches('/');
        let path = if path.is_empty() || path == "/" {
            String::new()
        } else {
            path.to_string()
        };

        Self { origin, path }
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

/// Recorded request/response bodies for one provider scenario, in wire order.
///
/// Provider edge matrices use these bytes to prove that a replay fixture still
/// carries the premise it claims to exercise. Keeping the reader beside the
/// cassette parser avoids every provider growing a subtly different YAML
/// decoder.
pub fn recorded_interaction_bodies(
    cassette_root: &Path,
    provider: &str,
    scenario: &str,
) -> Vec<(String, String)> {
    let path = cassette_path(cassette_root, provider, scenario);
    let contents = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable: {error}",
            path.display()
        )
    });

    parse_cassette_interactions(&path, &contents)
        .into_iter()
        .map(|interaction| {
            (
                interaction.when.body.unwrap_or_default(),
                interaction.then.body.unwrap_or_default(),
            )
        })
        .collect()
}

/// First recorded request body for one provider scenario, parsed as JSON.
pub fn recorded_json_request(cassette_root: &Path, provider: &str, scenario: &str) -> Value {
    let (request, _) = recorded_interaction_bodies(cassette_root, provider, scenario)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("cassette {provider}/{scenario} should contain an interaction"));
    serde_json::from_str(&request).unwrap_or_else(|error| {
        panic!("cassette {provider}/{scenario} request should be JSON: {error}")
    })
}

/// First recorded non-streaming response body, parsed as JSON.
pub fn recorded_json_response(cassette_root: &Path, provider: &str, scenario: &str) -> Value {
    let (_, response) = recorded_interaction_bodies(cassette_root, provider, scenario)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("cassette {provider}/{scenario} should contain an interaction"));
    serde_json::from_str(&response).unwrap_or_else(|error| {
        panic!("cassette {provider}/{scenario} response should be JSON: {error}")
    })
}

/// Request headers recorded for one provider scenario, lowercased, in wire
/// order.
///
/// Only the names on `RECORDED_REQUEST_HEADERS` are ever written, so this is
/// the way to assert that a *sensitive* header never reaches a fixture —
/// which is a claim about the recorder, not about the provider.
pub fn recorded_request_header_pairs(
    cassette_root: &Path,
    provider: &str,
    scenario: &str,
) -> Vec<Vec<(String, String)>> {
    let path = cassette_path(cassette_root, provider, scenario);
    let contents = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable: {error}",
            path.display()
        )
    });

    parse_cassette_interactions(&path, &contents)
        .into_iter()
        .map(|interaction| {
            interaction
                .when
                .header
                .into_iter()
                .map(|header| (header.name.to_ascii_lowercase(), header.value))
                .collect()
        })
        .collect()
}

/// Request paths recorded for one provider scenario, in wire order.
///
/// The path is the half of a request that no assertion on the *body* can
/// reach, and it is exactly what a base-URL composition bug corrupts: a
/// doubled `/v1`, a missing one, or a capability routed at the wrong endpoint
/// all leave the body untouched.
pub fn recorded_request_paths(cassette_root: &Path, provider: &str, scenario: &str) -> Vec<String> {
    let path = cassette_path(cassette_root, provider, scenario);
    let contents = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable: {error}",
            path.display()
        )
    });

    parse_cassette_interactions(&path, &contents)
        .into_iter()
        .map(|interaction| interaction.when.path)
        .collect()
}

/// Recorded `(status, body)` pairs for one provider scenario, in wire order.
///
/// Error matrices assert on the status *class* and on the preserved envelope;
/// both live here rather than in whatever the client turned them into.
pub fn recorded_statuses_and_bodies(
    cassette_root: &Path,
    provider: &str,
    scenario: &str,
) -> Vec<(u16, String)> {
    let path = cassette_path(cassette_root, provider, scenario);
    let contents = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable: {error}",
            path.display()
        )
    });

    parse_cassette_interactions(&path, &contents)
        .into_iter()
        .map(|interaction| {
            (
                interaction.then.status,
                interaction.then.body.unwrap_or_default(),
            )
        })
        .collect()
}

/// JSON `data:` frames from the first recorded SSE response, excluding
/// `[DONE]`.
pub fn recorded_sse_json_frames(
    cassette_root: &Path,
    provider: &str,
    scenario: &str,
) -> Vec<Value> {
    let (_, response) = recorded_interaction_bodies(cassette_root, provider, scenario)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("cassette {provider}/{scenario} should contain an interaction"));

    response
        .lines()
        .filter_map(|line| line.trim().strip_prefix("data:"))
        .map(str::trim)
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| {
            serde_json::from_str(payload).unwrap_or_else(|error| {
                panic!("cassette {provider}/{scenario} SSE frame should be JSON: {error}")
            })
        })
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

/// Scrub credentials and generated identifiers in a YAML cassette.
pub fn scrub_cassette_contents(yaml: &str) -> String {
    scrub_cassette_contents_with_policy(CassettePolicy::default(), yaml)
}

fn scrub_cassette_contents_with_policy(policy: CassettePolicy, yaml: &str) -> String {
    let mut interactions = parse_cassette_interactions(Path::new("<cassette>"), yaml);
    let mut scrubber = CassetteScrubber::new(policy);

    for interaction in &mut interactions {
        scrubber.scrub_request(&mut interaction.when);
        scrubber.scrub_response(&mut interaction.then);
    }

    serialize_cassette_interactions(&interactions)
}

/// Report unsanitized or unsafe material in a YAML cassette.
pub fn cassette_safety_failures(cassette_path: &Path, contents: &str) -> Vec<String> {
    cassette_safety_failures_with_policy(CassettePolicy::default(), cassette_path, contents)
}

fn cassette_safety_failures_with_policy(
    policy: CassettePolicy,
    cassette_path: &Path,
    contents: &str,
) -> Vec<String> {
    let mut failures = Vec::new();
    let scrubbed = scrub_cassette_contents_with_policy(policy, contents);

    if scrubbed != contents {
        failures.push(format!(
            "{} is not in scrubbed cassette form",
            cassette_path.display()
        ));
    }

    let decoded = decoded_base64_body_texts(contents);
    let with_decoded = format!("{}\n{}", contents, decoded.join("\n"));
    failures.extend(artifact_safety_failures_with_policy(
        policy,
        cassette_path,
        &with_decoded,
    ));
    failures
}

/// The cassette engine's secret/provider-token checks for JSON sidecars and
/// failure reports. Unlike the cassette validator this does not parse YAML.
pub fn artifact_safety_failures(path: &Path, contents: &str) -> Vec<String> {
    artifact_safety_failures_with_policy(CassettePolicy::default(), path, contents)
}

/// Scrub live diagnostics with the same ID remapping and credential rules as
/// provider traffic. This is separate evidence, never a canonical effect log.
pub fn scrub_artifact(value: &Value) -> Value {
    let mut value = value.clone();
    CassetteScrubber::new(CassettePolicy::default()).scrub_json_value(None, &mut value);
    value
}

fn artifact_safety_failures_with_policy(
    policy: CassettePolicy,
    cassette_path: &Path,
    contents: &str,
) -> Vec<String> {
    let mut failures = Vec::new();

    let lower = contents.to_ascii_lowercase();
    for pattern in policy.forbidden_patterns {
        if lower.contains(pattern) {
            failures.push(format!("{} contains {pattern:?}", cassette_path.display()));
        }
    }

    let generated_tokens = generated_tokens(policy, contents);
    if !generated_tokens.is_empty() {
        failures.push(format!(
            "{} contains {} unsanitized provider artifact(s)",
            cassette_path.display(),
            generated_tokens.len()
        ));
    }

    let openai_api_key_tokens = openai_api_key_tokens(contents);
    if !openai_api_key_tokens.is_empty() {
        failures.push(format!(
            "{} contains {} OpenAI API key-shaped token(s)",
            cassette_path.display(),
            openai_api_key_tokens.len()
        ));
    }

    let anthropic_api_key_tokens = anthropic_api_key_tokens(contents);
    if !anthropic_api_key_tokens.is_empty() {
        failures.push(format!(
            "{} contains {} Anthropic API key-shaped token(s)",
            cassette_path.display(),
            anthropic_api_key_tokens.len()
        ));
    }

    let google_api_key_tokens = google_api_key_tokens(contents);
    if !google_api_key_tokens.is_empty() {
        failures.push(format!(
            "{} contains {} Google API key-shaped token(s)",
            cassette_path.display(),
            google_api_key_tokens.len()
        ));
    }

    let aws_access_key_tokens = aws_access_key_tokens(contents);
    if !aws_access_key_tokens.is_empty() {
        failures.push(format!(
            "{} contains {} AWS access key-shaped token(s)",
            cassette_path.display(),
            aws_access_key_tokens.len()
        ));
    }

    failures
}

async fn write_scrubbed_cassette(cassette_path: &Path, policy: CassettePolicy, yaml: &str) {
    let redacted = scrub_cassette_contents_with_policy(policy, yaml);
    let failures = cassette_safety_failures_with_policy(policy, cassette_path, &redacted);
    assert!(
        failures.is_empty(),
        "provider cassette {} still contains unsafe artifacts after scrubbing:\n{}",
        cassette_path.display(),
        failures.join("\n")
    );

    write_cassette_atomically(cassette_path, redacted.as_bytes())
        .await
        .expect("provider cassette should be written");
}

fn decoded_base64_body_texts(contents: &str) -> Vec<String> {
    parse_cassette_interactions(Path::new("<cassette>"), contents)
        .into_iter()
        .flat_map(|interaction| {
            [
                interaction.when,
                cassette_request_from_response(interaction.then),
            ]
        })
        .filter(|request| request.body_encoding == BodyEncoding::Base64)
        .filter_map(|request| request.body)
        .filter_map(|body| decode_body(&body, BodyEncoding::Base64).ok())
        .map(|body| String::from_utf8_lossy(&body).into_owned())
        .collect()
}

fn cassette_request_from_response(response: CassetteResponse) -> CassetteRequest {
    CassetteRequest {
        path: String::new(),
        method: String::new(),
        query_param: Vec::new(),
        header: response.header,
        body: response.body,
        body_encoding: response.body_encoding,
    }
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
    // Venice inference keys carry this literal prefix, so a recording that
    // ever echoes one back (Venice quotes request material in some error
    // bodies) fails the scan instead of being committed.
    "venice_inference_key_",
    "__cf_bm=",
    "proj_",
    "set-cookie",
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

const SENSITIVE_HEADER_NAMES: &[&str] = &[
    "authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "ocp-apim-subscription-key",
    "set-cookie",
    "openai-organization",
    "openai-project",
    "anthropic-organization-id",
    "x-amz-security-token",
    "x-amz-content-sha256",
    "x-amz-date",
    "key",
];

const SENSITIVE_QUERY_PARAMS: &[&str] = &[
    "key",
    "api_key",
    "apikey",
    "access_token",
    "x-amz-credential",
    "x-amz-signature",
    "x-amz-security-token",
];

// `x-amzn-errortype` is how the AWS SDKs classify an error response into a
// modeled exception; dropping it made every recorded AWS error replay as an
// unclassified `Unhandled` error, so a cassette could not reproduce the error
// path it recorded. The value is an exception class name, not account state.
// `x-amzn-requestid` is the only place the AWS request id appears — the SDK
// reads it off the header, not the body — so a cassette that drops it cannot
// replay any behavior that reads the id. It is a per-call opaque identifier,
// not account state, and the scrubber placeholders its value.
const RESPONSE_HEADER_ALLOWLIST: &[&str] = &[
    "content-type",
    // Replay must retain the provider's retry hint for adapter diagnostics.
    "retry-after",
    "x-amzn-errortype",
    "x-amzn-requestid",
    // Provider transport request ids (rig#2265): Anthropic / OpenAI-and-xAI.
    "request-id",
    "x-request-id",
    // Mistral's own spelling for the same thing.
    "mistral-correlation-id",
];

const VOLATILE_JSON_KEYS: &[&str] = &[
    "completed_at",
    "created",
    "created_at",
    "updated",
    "updated_at",
];

const SENSITIVE_STRING_KEYS: &[&str] = &[
    "encrypted_content",
    "encryptedcontent",
    // Anthropic's server-tool locators. Named like opaque handles but both
    // base64-decode to a protobuf carrying the *same* plaintext UUID, stable
    // across separate calls, so each is preserved verbatim in a committed
    // fixture unless scrubbed — exactly what their sibling `encrypted_content`
    // is on this list to prevent. Matching lowercases the key without
    // stripping underscores, so each needs its squashed twin like the pairs
    // above.
    "encrypted_index",
    "encryptedindex",
    "encrypted_stdout",
    "encryptedstdout",
    "obfuscation",
    "prompt_cache_key",
    "safety_identifier",
    "signature",
    "thoughtsignature",
];

/// Allowlisted response headers whose value is a generated per-call id.
const GENERATED_ID_HEADERS: &[&str] = &[
    "x-amzn-requestid",
    "request-id",
    "x-request-id",
    "mistral-correlation-id",
];

const GENERATED_ID_KEYS: &[&str] = &[
    "call_id",
    // Pagination cursors: opaque to rig, but they encode provider-side resource
    // ids, so a recorded cursor leaks what the rest of the fixture redacted.
    "nextpagetoken",
    "next_page_token",
    "item_id",
    "previous_interaction_id",
    "previous_response_id",
    "request_id",
    "response_id",
    "responseid",
    "tool_call_id",
    "tool_use_id",
    "tooluseid",
];

const GENERATED_TOKEN_PREFIXES: &[TokenPrefix] = &[
    TokenPrefix::new("chatcmpl-", "chatcmpl-", 8),
    TokenPrefix::new("resp_", "resp_", 8),
    TokenPrefix::new("msg_", "msg_", 8),
    TokenPrefix::new("call_", "call_", 8),
    TokenPrefix::new("toolu_", "toolu_", 8),
    TokenPrefix::new("tooluse_", "tooluse_", 8),
    TokenPrefix::new("file_", "file_", 6),
    TokenPrefix::new("req_", "req_", 8),
    TokenPrefix::new("rs_", "rs_", 8),
    TokenPrefix::new("fc_", "fc_", 8),
    TokenPrefix::new("fp_", "fp_", 6),
    TokenPrefix::new("v1_", "v1_", 8),
    TokenPrefix::new("run_", "run_", 8),
    TokenPrefix::new("step_", "step_", 8),
    TokenPrefix::new("thread_", "thread_", 8),
    TokenPrefix::new("asst_", "asst_", 8),
    TokenPrefix::new("batch_", "batch_", 8),
    TokenPrefix::new("upload_", "upload_", 8),
    TokenPrefix::new("document-", "document-", 8),
];

struct CassetteScrubber {
    policy: CassettePolicy,
    placeholders: BTreeMap<String, String>,
    counters: BTreeMap<&'static str, usize>,
}

impl CassetteScrubber {
    fn new(policy: CassettePolicy) -> Self {
        Self {
            policy,
            placeholders: BTreeMap::new(),
            counters: BTreeMap::new(),
        }
    }

    fn scrub_request(&mut self, request: &mut CassetteRequest) {
        request.path = self.scrub_text(&request.path);
        scrub_headers(self.policy, &mut request.header, HeaderMode::Request);
        scrub_query_params(self.policy, &mut request.query_param);

        for query_param in &mut request.query_param {
            // A pagination cursor is an opaque blob to rig, but not to the
            // provider: Gemini's `cachedContents` cursors are base64 protobuf
            // carrying the *real* resource ids of the entries either side of the
            // page boundary, so a recorded cursor re-exposes ids that were
            // placeholdered everywhere else in the same fixture.
            //
            // Placeholdered rather than blanket-redacted because distinct
            // cursors must stay distinct: a loop that follows them compares each
            // against the last to detect a server that stops advancing, and
            // collapsing every cursor to one value would end that loop early on
            // replay. `placeholder` maps equal originals to equal placeholders
            // and distinct ones to distinct, which is exactly the property the
            // cursor needs — and the same scrubber instance handles the response
            // body, so the request's cursor and the response that issued it
            // agree.
            if query_param.name.eq_ignore_ascii_case("pageToken") {
                // Scrubbing must be idempotent: the safety check re-scrubs its
                // own output and requires a fixed point, and minting a fresh
                // placeholder for an already-placeholdered cursor is not one.
                if !is_redacted_placeholder(&query_param.value) {
                    query_param.value = self.placeholder(&query_param.value, "cursor-");
                }
                continue;
            }
            query_param.value = self.scrub_text(&query_param.value);
        }

        if let Some(body) = &mut request.body {
            *body = self.scrub_encoded_body(body, request.body_encoding);
        }
    }

    fn scrub_response(&mut self, response: &mut CassetteResponse) {
        scrub_headers(self.policy, &mut response.header, HeaderMode::Response);

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

        // An allowlisted header is kept for its *shape*, not its value: the
        // request id is a per-call generated token and gets the same
        // placeholder treatment as one carried in a body.
        for header in response.header.iter_mut() {
            if contains_case_insensitive(GENERATED_ID_HEADERS, &header.name) {
                header.value = self.placeholder(&header.value, "req_");
            }
        }

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

    fn scrub_body(&mut self, body: &str) -> String {
        if let Some(mut json) = canonical_json(body) {
            self.scrub_json_value(None, &mut json);
            return serde_json::to_string(&json).expect("scrubbed JSON body should serialize");
        }

        if body
            .lines()
            .any(|line| line.trim_start().starts_with("data:"))
        {
            return self.scrub_sse_body(body);
        }

        self.scrub_text(body)
    }

    fn scrub_sse_body(&mut self, body: &str) -> String {
        let mut output = String::with_capacity(body.len());

        for line in body.split_inclusive('\n') {
            let (line_without_newline, newline) = line
                .strip_suffix('\n')
                .map_or((line, ""), |line| (line, "\n"));
            let (line_without_cr, cr) = line_without_newline
                .strip_suffix('\r')
                .map_or((line_without_newline, ""), |line| (line, "\r"));
            let trimmed = line_without_cr.trim_start();
            let indentation_len = line_without_cr.len() - trimmed.len();

            if let Some(payload) = trimmed.strip_prefix("data:") {
                let payload = payload.trim_start();
                if payload == "[DONE]" {
                    output.push_str(line_without_cr);
                } else if let Some(mut json) = canonical_json(payload) {
                    self.scrub_json_value(None, &mut json);
                    output.push_str(&line_without_cr[..indentation_len]);
                    output.push_str("data: ");
                    output.push_str(
                        &serde_json::to_string(&json)
                            .expect("scrubbed SSE JSON payload should serialize"),
                    );
                } else {
                    output.push_str(&self.scrub_text(line_without_cr));
                }
            } else {
                output.push_str(&self.scrub_text(line_without_cr));
            }

            output.push_str(cr);
            output.push_str(newline);
        }

        output
    }

    fn scrub_json_value(&mut self, key: Option<&str>, value: &mut Value) {
        let key_lower = key.map(str::to_ascii_lowercase);

        match value {
            Value::Object(map) => {
                let object_type = map
                    .get("type")
                    .and_then(Value::as_str)
                    .map(str::to_ascii_lowercase);
                let object_name = map
                    .get("object")
                    .and_then(Value::as_str)
                    .map(str::to_ascii_lowercase);
                // Venice's image payload carries neither `object` nor `type`:
                // it is `{ id, images: [base64], … }`, and its `id` is a bare
                // account-scoped token with no prefix for the generated-token
                // rules to recognize. Both halves of the shape are required —
                // cohere's image-embedding *request* also has an `images`
                // array of (data-URI) strings but no `id`, and its response's
                // `images` holds metadata objects rather than payloads.
                let venice_image_payload = map.get("id").is_some_and(Value::is_string)
                    && map.get("images").is_some_and(|images| {
                        images
                            .as_array()
                            .is_some_and(|images| images.iter().all(Value::is_string))
                    });

                for (key, value) in map {
                    if key == "data" && object_type.as_deref() == Some("reasoning.encrypted") {
                        if let Value::String(data) = value {
                            *data = self.placeholder(data, "encrypted_reasoning_");
                        }
                        continue;
                    }

                    // Anthropic redacted thinking carries an opaque encrypted
                    // blob; placeholder it like OpenAI encrypted reasoning so
                    // fixtures never commit provider ciphertext.
                    if key == "data" && object_type.as_deref() == Some("redacted_thinking") {
                        if let Value::String(data) = value {
                            *data = self.placeholder(data, "redacted_thinking_");
                        }
                        continue;
                    }

                    if key == "id"
                        && venice_image_payload
                        && let Value::String(id) = value
                    {
                        *id = self.placeholder(id, "id_");
                        continue;
                    }

                    // Venice returns generated images as bare base64 strings
                    // in an `images` array rather than OpenAI's `b64_json`
                    // objects; same payload, same treatment — keeping the
                    // bytes would commit generated media and inflate the
                    // fixture (Gemini's unscrubbed image cassette is 328 KB).
                    // Scoped to the response shape, not the key name: cohere's
                    // image-embedding requests carry their own `images` array
                    // of data URIs that must survive verbatim.
                    if key == "images"
                        && venice_image_payload
                        && let Value::Array(images) = value
                    {
                        for image in images.iter_mut() {
                            if let Value::String(image) = image {
                                *image = IMAGE_PAYLOAD_PLACEHOLDER.to_string();
                            }
                        }
                        continue;
                    }

                    if key == "id"
                        && should_scrub_id_for_object(
                            self.policy,
                            value.as_str(),
                            object_type.as_deref(),
                            object_name.as_deref(),
                        )
                    {
                        if let Value::String(id) = value {
                            *id = self.placeholder(
                                id,
                                placeholder_kind_for_id(
                                    self.policy,
                                    id,
                                    object_type.as_deref(),
                                    object_name.as_deref(),
                                ),
                            );
                        }
                        continue;
                    }

                    self.scrub_json_value(Some(key), value);
                }
            }
            Value::Array(values) => {
                for value in values {
                    self.scrub_json_value(key, value);
                }
            }
            Value::String(text) => {
                if is_redacted_placeholder(text) {
                    return;
                }

                if let Some(key) = key_lower.as_deref() {
                    if VOLATILE_JSON_KEYS.contains(&key) {
                        *text = "1970-01-01T00:00:00Z".to_string();
                        return;
                    }

                    if SENSITIVE_STRING_KEYS.contains(&key) || GENERATED_ID_KEYS.contains(&key) {
                        *text = self
                            .placeholder(text, placeholder_kind_for_value(self.policy, text, key));
                        return;
                    }

                    if key == "url" && text.contains("grounding-api-redirect/") {
                        *text = self.placeholder(text, "url");
                        return;
                    }

                    if key == "b64_json" {
                        *text = IMAGE_PAYLOAD_PLACEHOLDER.to_string();
                        return;
                    }
                }

                *text = self.scrub_text(text);
            }
            Value::Number(number) => {
                if key_lower
                    .as_deref()
                    .is_some_and(|key| VOLATILE_JSON_KEYS.contains(&key))
                {
                    *value = Value::Number(0.into());
                } else {
                    let _ = number;
                }
            }
            Value::Bool(_) | Value::Null => {}
        }
    }

    fn scrub_text(&mut self, text: &str) -> String {
        let mut scrubbed = text.to_string();
        for key in self.policy.sensitive_query_params {
            scrubbed = scrub_query_param(&scrubbed, key, REDACTED);
        }
        let scrubbed = self.scrub_grounding_redirects(&scrubbed);
        let scrubbed = self.scrub_aws_account_ids(&scrubbed);
        let scrubbed = self.scrub_resource_names(&scrubbed);
        let scrubbed = scrub_local_filesystem_paths(&scrubbed);
        self.scrub_generated_tokens(&scrubbed)
    }

    /// Scrub server-assigned resource handles of the form `collection/<id>`.
    ///
    /// The generated-token machinery cannot reach these: it keys on a prefix and
    /// then consumes `is_token_char`, which excludes `/`, so a `TokenPrefix` of
    /// `"cachedContents/"` would match the prefix and then stop the token at the
    /// slash. Gemini's explicit context cache hands back exactly that shape
    /// (`cachedContents/n3v1qk0nqz9k`), the id is account-scoped, and it appears
    /// in request bodies, request *paths* and response bodies alike — so it
    /// needs a rule of its own or it goes into a fixture verbatim.
    fn scrub_resource_names(&mut self, text: &str) -> String {
        const RESOURCE_COLLECTIONS: &[&str] = &["cachedContents/"];

        let mut output = String::with_capacity(text.len());
        let mut index = 0;

        while index < text.len() {
            if !text.is_char_boundary(index) {
                index += 1;
                continue;
            }

            let matched = RESOURCE_COLLECTIONS
                .iter()
                .find(|collection| text[index..].starts_with(**collection));

            if let Some(collection) = matched {
                let id_start = index + collection.len();
                let id_end = token_end(text, id_start);
                let id = &text[id_start..id_end];
                // A bare `cachedContents` path segment with no id (the
                // collection endpoint itself) must survive untouched, or the
                // recorded request path stops matching on replay.
                // `token_end` accepts the character at offset 0 unconditionally,
                // so a non-token char right after the collection prefix comes
                // back as the "id" — `cachedContents/"` would swallow the
                // closing quote and emit invalid JSON into a fixture. Require
                // the id to be entirely token characters.
                if !id.is_empty() && id.chars().all(is_token_char) && !is_redacted_placeholder(id) {
                    output.push_str(collection);
                    output.push_str(&self.placeholder(id, "cached-"));
                    index = id_end;
                    continue;
                }
            }

            let ch = text[index..]
                .chars()
                .next()
                .expect("index should be on a char boundary");
            output.push(ch);
            index += ch.len_utf8();
        }

        output
    }

    /// Replace the account-id segment of every ARN.
    ///
    /// A Bedrock guardrail assessment echoes the guardrail's ARN, which
    /// carries the caller's 12-digit AWS account id — a provider account
    /// identifier, which cassettes must not commit. The rest of the ARN
    /// (partition, service, region, resource) stays readable so the fixture
    /// still shows which resource answered.
    fn scrub_aws_account_ids(&mut self, text: &str) -> String {
        const PREFIX: &str = "arn:";
        const ACCOUNT_FIELD: usize = 4;
        const ACCOUNT_LEN: usize = 12;

        let mut output = String::with_capacity(text.len());
        let mut rest = text;

        while let Some(start) = rest.find(PREFIX) {
            output.push_str(&rest[..start]);
            let arn = &rest[start..];
            // An ARN ends at the first character that cannot appear in one;
            // the surrounding JSON quote or comma is the usual terminator.
            let end = arn
                .find(['"', ',', ' ', '\n', '}', ']'])
                .unwrap_or(arn.len());
            let (arn, tail) = arn.split_at(end);

            let mut fields = arn.split(':').map(str::to_string).collect::<Vec<_>>();
            match fields.get(ACCOUNT_FIELD) {
                Some(account)
                    if account.len() == ACCOUNT_LEN
                        && account.chars().all(|ch| ch.is_ascii_digit()) =>
                {
                    let placeholder = self.placeholder(account, "account_");
                    fields[ACCOUNT_FIELD] = placeholder;
                    output.push_str(&fields.join(":"));
                }
                _ => output.push_str(arn),
            }

            rest = tail;
        }

        output.push_str(rest);
        output
    }

    fn scrub_grounding_redirects(&mut self, text: &str) -> String {
        const PREFIX: &str = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/";
        let mut output = String::with_capacity(text.len());
        let mut remaining = text;

        while let Some(index) = remaining.find(PREFIX) {
            let (before, after_before) = remaining.split_at(index);
            output.push_str(before);

            let end = after_before
                .find(['"', '\'', '<', ' ', '\n', '\r'])
                .unwrap_or(after_before.len());
            let token = &after_before[..end];
            output.push_str(&self.placeholder(token, "url"));
            remaining = &after_before[end..];
        }

        output.push_str(remaining);
        output
    }

    fn scrub_generated_tokens(&mut self, text: &str) -> String {
        let mut output = String::with_capacity(text.len());
        let mut index = 0;

        while index < text.len() {
            if !text.is_char_boundary(index) {
                index += 1;
                continue;
            }

            if let Some(prefix) = self.policy.matching_generated_prefix(text, index) {
                let end = token_end(text, index);
                let token = &text[index..end];

                if is_generated_token(token, prefix, in_id_field_position(text, index)) {
                    output.push_str(&self.placeholder(token, prefix.placeholder_prefix));
                    index = end;
                    continue;
                }
            }

            let ch = text[index..]
                .chars()
                .next()
                .expect("index should be on a char boundary");
            output.push(ch);
            index += ch.len_utf8();
        }

        output
    }

    fn placeholder(&mut self, original: &str, kind: &'static str) -> String {
        // An empty value carries nothing to redact, and minting a placeholder
        // for it *invents data*: a recording would show a non-empty token
        // where the wire sent `""`, changing replay semantics for any code
        // that reads the field (observed on Anthropic's
        // `content_block_start.signature`, which is empty on the wire).
        if original.is_empty() {
            return String::new();
        }

        if let Some(existing) = self.placeholders.get(original) {
            return existing.clone();
        }

        let counter = self.counters.entry(kind).or_insert(0);
        *counter += 1;
        let placeholder = format!("{kind}REDACTED_{counter}");
        self.placeholders
            .insert(original.to_string(), placeholder.clone());
        placeholder
    }
}

#[derive(Clone, Copy)]
enum HeaderMode {
    Request,
    Response,
}

fn scrub_headers(policy: CassettePolicy, headers: &mut Vec<NameValue>, mode: HeaderMode) {
    match mode {
        HeaderMode::Request => {
            for header in headers {
                if policy.is_sensitive_header(&header.name) {
                    header.value = REDACTED.to_string();
                }
            }
        }
        HeaderMode::Response => {
            headers.retain(|header| policy.is_allowed_response_header(&header.name));
        }
    }
}

fn scrub_query_params(policy: CassettePolicy, query_params: &mut [NameValue]) {
    for query_param in query_params {
        if policy.is_sensitive_query_param(&query_param.name) {
            query_param.value = REDACTED.to_string();
        }
    }
}

fn should_scrub_id_for_object(
    policy: CassettePolicy,
    value: Option<&str>,
    object_type: Option<&str>,
    object_name: Option<&str>,
) -> bool {
    let Some(value) = value else {
        return false;
    };

    if is_redacted_placeholder(value) {
        return false;
    }

    if placeholder_kind_from_generated_token(policy, value).is_some() {
        return true;
    }

    matches!(
        object_type,
        Some("function_call")
            | Some("function")
            | Some("message")
            | Some("tool_use")
            | Some("reasoning")
            | Some("file")
    ) || matches!(
        object_name,
        Some("response")
            | Some("chat.completion")
            | Some("chat.completion.chunk")
            | Some("interaction")
    )
}

fn placeholder_kind_for_value(policy: CassettePolicy, value: &str, fallback: &str) -> &'static str {
    placeholder_kind_from_generated_token(policy, value).unwrap_or(match fallback {
        "call_id" | "tool_call_id" => "call_",
        "encrypted_content" | "encryptedcontent" => "encrypted_content_",
        "item_id" => "item_",
        "obfuscation" => "obfuscation_",
        "previous_interaction_id" | "previous_response_id" | "response_id" | "responseid" => "id_",
        "request_id" => "req_",
        "signature" | "thoughtsignature" => "signature_",
        "system_fingerprint" => "fp_",
        "tool_use_id" => "toolu_",
        "tooluseid" => "tooluse_",
        "url" => "url_",
        _ => "id_",
    })
}

fn placeholder_kind_for_id(
    policy: CassettePolicy,
    value: &str,
    object_type: Option<&str>,
    object_name: Option<&str>,
) -> &'static str {
    placeholder_kind_from_generated_token(policy, value).unwrap_or(match object_type {
        Some("file") => "file_",
        Some("function") => "call_",
        Some("function_call") => "fc_",
        Some("message") => "msg_",
        Some("tool_use") => "toolu_",
        _ => match object_name {
            Some("chat.completion") | Some("chat.completion.chunk") => "chatcmpl-",
            Some("interaction") => "v1_",
            Some("response") => "resp_",
            _ => "id_",
        },
    })
}

fn placeholder_kind_from_generated_token(
    policy: CassettePolicy,
    value: &str,
) -> Option<&'static str> {
    policy
        .generated_prefix_for(value)
        .map(|prefix| prefix.placeholder_prefix)
}

fn token_end(text: &str, start: usize) -> usize {
    let mut end = start;

    for (offset, ch) in text[start..].char_indices() {
        if offset == 0 || is_token_char(ch) {
            end = start + offset + ch.len_utf8();
        } else {
            break;
        }
    }

    end
}

fn is_token_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-')
}

fn is_generated_token(token: &str, prefix: TokenPrefix, in_id_field: bool) -> bool {
    if is_redacted_placeholder(token) {
        return false;
    }

    let Some(suffix) = token.strip_prefix(prefix.raw) else {
        return false;
    };

    if suffix.len() < prefix.min_suffix_len
        || !suffix
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
    {
        return false;
    }

    // The digit requirement keeps prose identifiers (`call_id`,
    // `tool_call_id`) out of the generated-token class. Ollama daemons mint
    // digit-less lowercase call ids (`call_kqpofucm`), so for `call_` a
    // long all-lowercase suffix also counts as generated — but only in an
    // id-bearing field position: a real tool named `call_forwarding` in a
    // name field or prose must survive recording untouched. Redaction keys
    // off field identity, not value shape.
    suffix.chars().any(|ch| ch.is_ascii_digit())
        || (prefix.raw == "call_"
            && in_id_field
            && suffix.len() >= 8
            && suffix.chars().all(|ch| ch.is_ascii_lowercase()))
}

/// Whether the token starting at `token_start` is the value of an
/// id-bearing JSON field (`"id":"…"`, `"tool_call_id":"…"`, `"toolCallId":"…"`),
/// tolerating the escaped-quote spelling of bodies that are themselves
/// JSON-encoded (`\"id\":\"…\"`).
fn in_id_field_position(text: &str, token_start: usize) -> bool {
    // The value's opening string delimiter: `"` or `\"`.
    let Some(rest) = text[..token_start].strip_suffix('"') else {
        return false;
    };
    let rest = rest.strip_suffix('\\').unwrap_or(rest);
    let rest = rest.trim_end();
    let Some(rest) = rest.strip_suffix(':') else {
        return false;
    };
    // The field name's closing delimiter.
    let Some(rest) = rest.trim_end().strip_suffix('"') else {
        return false;
    };
    let rest = rest.strip_suffix('\\').unwrap_or(rest);
    let name_start = rest
        .rfind(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .map_or(0, |at| at + 1);
    let name = &rest[name_start..];
    name == "id" || name.ends_with("_id") || name.ends_with("Id")
}

fn is_redacted_placeholder(value: &str) -> bool {
    let Some((kind, counter)) = value.split_once("REDACTED_") else {
        return false;
    };

    !kind.is_empty()
        && !counter.is_empty()
        && kind
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
        && counter.chars().all(|ch| ch.is_ascii_digit())
}

fn generated_tokens(policy: CassettePolicy, contents: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut index = 0;

    while index < contents.len() {
        if !contents.is_char_boundary(index) {
            index += 1;
            continue;
        }

        if let Some(prefix) = policy.matching_generated_prefix(contents, index) {
            let end = token_end(contents, index);
            let token = &contents[index..end];
            if is_generated_token(token, prefix, in_id_field_position(contents, index))
                && !token.contains("REDACTED_")
            {
                tokens.push(token.to_string());
            }
            index = end;
            continue;
        }

        let ch = contents[index..]
            .chars()
            .next()
            .expect("index should be on a char boundary");
        index += ch.len_utf8();
    }

    tokens.sort();
    tokens.dedup();
    tokens
}

fn openai_api_key_tokens(contents: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut index = 0;

    while let Some(relative_index) = contents[index..].find("sk-") {
        let start = index + relative_index;
        if start > 0
            && contents[..start]
                .chars()
                .next_back()
                .is_some_and(is_token_char)
        {
            index = start + "sk-".len();
            continue;
        }

        let end = token_end(contents, start);
        let token = &contents[start..end];
        if is_openai_api_key_token(token) {
            tokens.push(token.to_string());
        }
        index = end;
    }

    tokens.sort();
    tokens.dedup();
    tokens
}

fn is_openai_api_key_token(token: &str) -> bool {
    if is_redacted_placeholder(token) {
        return false;
    }

    if let Some(suffix) = token.strip_prefix("sk-proj-") {
        return token_suffix_is_plausible_secret(suffix, 16);
    }

    if let Some(suffix) = token.strip_prefix("sk-svcacct-") {
        return token_suffix_is_plausible_secret(suffix, 16);
    }

    let Some(suffix) = token.strip_prefix("sk-") else {
        return false;
    };
    if suffix.starts_with("ant-") {
        return false;
    }
    token_suffix_is_plausible_secret(suffix, 32)
}

fn anthropic_api_key_tokens(contents: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut index = 0;

    while let Some(relative_index) = contents[index..].find("sk-ant-") {
        let start = index + relative_index;
        if start > 0
            && contents[..start]
                .chars()
                .next_back()
                .is_some_and(is_token_char)
        {
            index = start + "sk-ant-".len();
            continue;
        }

        let end = token_end(contents, start);
        let token = &contents[start..end];
        if is_anthropic_api_key_token(token) {
            tokens.push(token.to_string());
        }
        index = end;
    }

    tokens.sort();
    tokens.dedup();
    tokens
}

fn is_anthropic_api_key_token(token: &str) -> bool {
    if is_redacted_placeholder(token) {
        return false;
    }

    let Some(suffix) = token.strip_prefix("sk-ant-") else {
        return false;
    };
    token_suffix_is_plausible_secret(suffix, 16)
}

fn token_suffix_is_plausible_secret(suffix: &str, min_len: usize) -> bool {
    suffix.len() >= min_len
        && suffix
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
        && suffix.chars().any(|ch| ch.is_ascii_alphabetic())
}

fn aws_access_key_tokens(contents: &str) -> Vec<String> {
    let mut tokens = Vec::new();

    for prefix in ["AKIA", "ASIA"] {
        let mut remaining = contents;
        while let Some(index) = remaining.find(prefix) {
            let after_prefix = &remaining[index + prefix.len()..];
            let suffix_len = after_prefix
                .chars()
                .take_while(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit())
                .map(char::len_utf8)
                .sum::<usize>();
            let token = &remaining[index..index + prefix.len() + suffix_len];

            if suffix_len == 16 {
                tokens.push(token.to_string());
            }

            remaining = &remaining[index + prefix.len()..];
        }
    }

    tokens.sort();
    tokens.dedup();
    tokens
}

fn google_api_key_tokens(contents: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut remaining = contents;
    const PREFIX: &str = "AIza";
    const MIN_SUFFIX_LEN: usize = 20;

    while let Some(index) = remaining.find(PREFIX) {
        let after_prefix = &remaining[index + PREFIX.len()..];
        let suffix_len = after_prefix
            .chars()
            .take_while(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
            .map(char::len_utf8)
            .sum::<usize>();
        let token = &remaining[index..index + PREFIX.len() + suffix_len];

        if suffix_len >= MIN_SUFFIX_LEN {
            tokens.push(token.to_string());
        }

        remaining = &remaining[index + PREFIX.len()..];
    }

    tokens.sort();
    tokens.dedup();
    tokens
}

fn scrub_query_param(input: &str, key: &str, replacement: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut remainder = input;
    let needle = format!("{key}=");

    while let Some(index) = find_query_param(remainder, &needle) {
        let (prefix, after_prefix) = remainder.split_at(index);
        output.push_str(prefix);

        let value_start = needle.len();
        output.push_str(&after_prefix[..value_start]);
        let after_value_start = &after_prefix[value_start..];
        let value_end = after_value_start
            .find(['&', '"', '\'', ' ', '\n', '\r', '<'])
            .unwrap_or(after_value_start.len());
        output.push_str(replacement);
        remainder = &after_value_start[value_end..];
    }

    output.push_str(remainder);
    output
}

fn find_query_param(input: &str, needle: &str) -> Option<usize> {
    let mut search_start = 0;

    while let Some(relative_index) = find_ascii_case_insensitive(&input[search_start..], needle) {
        let index = search_start + relative_index;
        let starts_param = index == 0
            || input[..index]
                .chars()
                .next_back()
                .is_some_and(|ch| matches!(ch, '?' | '&' | '"' | '\'' | ' '));

        if starts_param {
            return Some(index);
        }

        search_start = index + needle.len();
    }

    None
}

fn find_ascii_case_insensitive(input: &str, needle: &str) -> Option<usize> {
    let input = input.as_bytes();
    let needle = needle.as_bytes();

    input
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle))
}

#[cfg(test)]
mod tests;

/// Client used by the scenarios whose *response* body is binary.
///
/// The shared proxy recorder exports cassettes through httpmock, which stores
/// bodies as strings and therefore drops a non-UTF-8 payload entirely — a
/// recorded speech response came back as `body: null` and replayed as zero
/// bytes. A text-to-speech endpoint answers with raw audio, so those
/// scenarios take the direct-recording path (the same one Bedrock's
/// event-stream cassettes use), which stores non-UTF-8 bodies as base64.
///
/// Shared by every suite with that shape — OpenAI and Venice today — so the
/// recording behavior they depend on has one definition beside [`DirectRecorder`].
///
/// In replay mode this is a plain reqwest client pointed at the replay
/// server; only record mode carries a recorder.
#[derive(Clone, Debug, Default)]
pub struct DirectRecordingHttpClient {
    inner: rig_reqwest::ReqwestClient,
    recorder: Option<DirectRecorder>,
}

impl DirectRecordingHttpClient {
    /// A client that records through `recorder` when one is present, i.e. in
    /// record mode; in replay mode it is a plain client pointed at the replay
    /// server.
    pub fn new(recorder: Option<DirectRecorder>) -> Self {
        Self {
            inner: rig_reqwest::ReqwestClient::default(),
            recorder,
        }
    }
}

impl HttpClientExt for DirectRecordingHttpClient {
    fn send<T, U>(
        &self,
        req: rig_core::http_client::Request<T>,
    ) -> impl std::future::Future<
        Output = rig_core::http_client::Result<
            rig_core::http_client::Response<rig_core::http_client::LazyBody<U>>,
        >,
    > + Send
    + 'static
    where
        T: Into<Bytes> + Send,
        U: From<Bytes> + Send + 'static,
    {
        let inner = self.inner.clone();
        let recorder = self.recorder.clone();
        let (parts, body) = req.into_parts();
        let body: Bytes = body.into();
        let method = parts.method.to_string();
        let uri = parts.uri.to_string();
        let request_headers = owned_headers(&parts.headers);
        let request = HttpRequest::from_parts(parts, body.clone());

        async move {
            let response = HttpClientExt::send::<Bytes, Bytes>(&inner, request).await?;
            let (parts, lazy_body) = response.into_parts();
            // Buffered, not streamed: the recorder needs the whole payload,
            // and the caller gets the same bytes back below.
            let bytes = lazy_body.await?;

            if let Some(recorder) = recorder {
                let response_headers = owned_headers(&parts.headers);
                recorder
                    .record_http_interaction(
                        DirectHttpRequest {
                            method: &method,
                            uri: &uri,
                            headers: request_headers.iter().map(|(name, value)| (name, value)),
                            body: &body,
                        },
                        DirectHttpResponse {
                            status: parts.status.as_u16(),
                            headers: response_headers.iter().map(|(name, value)| (name, value)),
                            body: &bytes,
                        },
                    )
                    .await;
            }

            let body: LazyBody<U> = Box::pin(async move { Ok(U::from(bytes)) });
            Ok(HttpResponse::from_parts(parts, body))
        }
    }

    // Only the unary path is recorded: the scenarios on this client are
    // JSON-in/audio-out. Multipart and streaming pass through so the type
    // still satisfies the trait.
    fn send_multipart<U>(
        &self,
        req: HttpRequest<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<HttpResponse<LazyBody<U>>>> + Send + 'static
    where
        U: From<Bytes> + Send + 'static,
    {
        self.inner.send_multipart(req)
    }

    fn send_streaming<T>(
        &self,
        req: HttpRequest<T>,
    ) -> impl Future<Output = http_client::Result<http_client::StreamingResponse>> + Send
    where
        T: Into<Bytes> + Send,
    {
        self.inner.send_streaming(req)
    }
}

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
mod paths;

#[cfg(test)]
mod cached_content_scrub_tests;

#[cfg(test)]
mod cached_content_scrub_edge_tests;

#[cfg(test)]
mod pagination_cursor_scrub_tests;

#[cfg(test)]
mod local_path_scrub_tests;
