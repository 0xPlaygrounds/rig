//! Cassette-backed provider test helpers.
//!
//! Provider cassette tests run in replay mode by default. Set
//! `RIG_PROVIDER_TEST_MODE=record` to hit the real provider and write cassette
//! fixtures. Record mode overwrites existing cassette files.
#![allow(dead_code)]

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue, Method, StatusCode};
use axum::response::Response;
use axum::{Router, routing::any};
use futures::FutureExt;
use httpmock::MockServer;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::any::Any;
use std::collections::BTreeMap;
use std::fmt;
use std::net::SocketAddr;
use std::panic::{AssertUnwindSafe, resume_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

const MODE_ENV: &str = "RIG_PROVIDER_TEST_MODE";
const CASSETTE_ROOT: &str = "tests/cassettes";
const REDACTED: &str = "[REDACTED]";
const DUMMY_API_KEY: &str = REDACTED;

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

#[derive(Clone, Copy, Debug)]
pub(crate) struct CassettePolicy {
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
    fn for_scenario(provider: &str, scenario: &str) -> Self {
        let required_request_headers = match provider {
            "openai" => OPENAI_REQUIRED_REQUEST_HEADERS,
            "anthropic" => ANTHROPIC_REQUIRED_REQUEST_HEADERS,
            "gemini" if scenario.starts_with("interactions_api/") => {
                GEMINI_INTERACTIONS_REQUIRED_REQUEST_HEADERS
            }
            _ => NO_REQUIRED_REQUEST_HEADERS,
        };

        let replay_matching = if provider == "openai" && scenario.starts_with("multi_extract/") {
            ReplayMatching::Unordered
        } else {
            ReplayMatching::Ordered
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
        self.generated_token_prefixes
            .iter()
            .copied()
            .find(|prefix| is_generated_token(value, *prefix))
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CassetteMode {
    Replay,
    Record,
}

impl CassetteMode {
    fn current() -> Self {
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

pub(crate) struct ProviderCassette {
    server: CassetteServer,
    cassette_path: PathBuf,
    base_path: String,
    mode: CassetteMode,
    policy: CassettePolicy,
    recording_id: Option<usize>,
}

enum CassetteServer {
    Recording(MockServer),
    Replay(ReplayServer),
}

impl CassetteServer {
    fn base_url(&self) -> String {
        match self {
            Self::Recording(server) => server.base_url(),
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
    pub(crate) async fn start(
        provider: &'static str,
        scenario: &'static str,
        real_base_url: &str,
    ) -> Self {
        let mode = CassetteMode::current();
        let policy = CassettePolicy::for_scenario(provider, scenario);
        let cassette_path = cassette_path(provider, scenario);
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

    pub(crate) fn base_url(&self) -> String {
        format!("{}{}", self.server.base_url(), self.base_path)
    }

    pub(crate) fn api_key(&self, env_name: &str) -> String {
        if self.mode.records() {
            std::env::var(env_name).unwrap_or_else(|_| {
                panic!("{env_name} must be set when {MODE_ENV}={:?}", self.mode)
            })
        } else {
            DUMMY_API_KEY.to_string()
        }
    }

    pub(crate) async fn finish(self) {
        if let CassetteServer::Replay(server) = &self.server {
            server.assert_consumed(&self.cassette_path).await;
            return;
        }

        let Some(recording_id) = self.recording_id else {
            return;
        };

        let CassetteServer::Recording(server) = &self.server else {
            return;
        };

        let recording = httpmock::Recording::new(recording_id, server);
        let bytes = recording
            .export_async()
            .await
            .expect("provider cassette should export")
            .expect("provider cassette should contain at least one interaction");
        let yaml = String::from_utf8(bytes.to_vec()).expect("cassette YAML should be UTF-8");
        let redacted = scrub_cassette_contents_with_policy(self.policy, &yaml);
        let failures =
            cassette_safety_failures_with_policy(self.policy, &self.cassette_path, &redacted);
        assert!(
            failures.is_empty(),
            "provider cassette {} still contains unsafe artifacts after scrubbing:\n{}",
            self.cassette_path.display(),
            failures.join("\n")
        );

        if let Some(parent) = self.cassette_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .expect("cassette directory should be created");
        }

        tokio::fs::write(&self.cassette_path, redacted)
            .await
            .expect("provider cassette should be written");
    }

    pub(crate) async fn finish_after_test(self, test_result: Result<(), PanicPayload>) {
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

    pub(crate) async fn finish_after_test_result<E>(
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
            interactions,
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

        tokio::spawn(async move {
            axum::serve(listener, app)
                .await
                .expect("replay server should run");
        });

        Self { addr, state }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    async fn assert_consumed(&self, cassette_path: &Path) {
        let state = self.state.lock().await;
        let unused = state
            .interactions
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

        assert!(
            unused.is_empty(),
            "provider cassette {} left unused interactions:\n{}",
            cassette_path.display(),
            unused.join("\n")
        );
    }
}

struct ReplayState {
    interactions: Vec<ReplayInteraction>,
    policy: CassettePolicy,
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
}

#[derive(Debug, Deserialize, Serialize)]
struct CassetteResponse {
    status: u16,
    #[serde(default)]
    header: Vec<NameValue>,
    body: Option<String>,
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
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("content-type", "application/json")
            .body(message.into())
            .expect("replay miss response should build");
    };

    let interaction = &mut state.interactions[index];
    interaction.consumed = true;
    cassette_response(&interaction.then)
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
        )
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
    _policy: CassettePolicy,
    actual_headers: &axum::http::HeaderMap,
    expected_headers: &[NameValue],
    actual: &[u8],
    expected: Option<&str>,
) -> bool {
    let Some(expected) = expected else {
        return actual.is_empty();
    };
    if is_multipart_request(actual_headers, expected_headers) {
        return multipart_bodies_match(
            actual_headers,
            expected_headers,
            actual,
            expected.as_bytes(),
        );
    }
    let Ok(actual) = std::str::from_utf8(actual) else {
        return false;
    };

    if let (Some(actual_json), Some(expected_json)) =
        (canonical_json(actual), canonical_json(expected))
    {
        return actual_json == expected_json;
    }

    actual == expected
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

fn cassette_response(response: &CassetteResponse) -> Response {
    let mut builder = Response::builder().status(response.status);
    for header in &response.header {
        if is_hop_by_hop_header(&header.name) {
            continue;
        }
        let Ok(name) = HeaderName::from_bytes(header.name.as_bytes()) else {
            continue;
        };
        let Ok(value) = HeaderValue::from_str(&header.value) else {
            continue;
        };
        builder = builder.header(name, value);
    }

    builder
        .body(response.body.clone().unwrap_or_default().into())
        .expect("cassette response should build")
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

pub(crate) fn cassette_path(provider: &str, scenario: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push(CASSETTE_ROOT);
    path.push(provider);
    for segment in scenario.split('/') {
        path.push(sanitize_path_segment(segment));
    }
    path.set_extension("yaml");
    path
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

pub(crate) fn scrub_cassette_contents(yaml: &str) -> String {
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

pub(crate) fn cassette_safety_failures(cassette_path: &Path, contents: &str) -> Vec<String> {
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

    failures
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

const FORBIDDEN_CASSETTE_PATTERNS: &[&str] = &[
    "authorization:",
    "bearer ",
    "x-api-key:",
    "x-goog-api-key:",
    "openai_api_key",
    "anthropic_api_key",
    "gemini_api_key",
    "__cf_bm=",
    "proj_",
    "set-cookie",
    "openai-organization",
    "openai-project",
    "anthropic-organization-id",
];

const NO_REQUIRED_REQUEST_HEADERS: &[&str] = &[];
const OPENAI_REQUIRED_REQUEST_HEADERS: &[&str] = &["authorization"];
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
    "key",
];

const SENSITIVE_QUERY_PARAMS: &[&str] = &["key", "api_key", "apikey", "access_token"];

const RESPONSE_HEADER_ALLOWLIST: &[&str] = &["content-type"];

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
    "obfuscation",
    "signature",
    "thoughtsignature",
];

const GENERATED_ID_KEYS: &[&str] = &[
    "call_id",
    "item_id",
    "previous_interaction_id",
    "previous_response_id",
    "request_id",
    "response_id",
    "responseid",
    "tool_call_id",
    "tool_use_id",
];

const GENERATED_TOKEN_PREFIXES: &[TokenPrefix] = &[
    TokenPrefix::new("chatcmpl-", "chatcmpl-", 8),
    TokenPrefix::new("resp_", "resp_", 8),
    TokenPrefix::new("msg_", "msg_", 8),
    TokenPrefix::new("call_", "call_", 8),
    TokenPrefix::new("toolu_", "toolu_", 8),
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
            query_param.value = self.scrub_text(&query_param.value);
        }

        if let Some(body) = &mut request.body {
            *body = self.scrub_body(body);
        }
    }

    fn scrub_response(&mut self, response: &mut CassetteResponse) {
        scrub_headers(self.policy, &mut response.header, HeaderMode::Response);

        if let Some(body) = &mut response.body {
            *body = self.scrub_body(body);
        }
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
                .map(|line| (line, "\n"))
                .unwrap_or((line, ""));
            let (line_without_cr, cr) = line_without_newline
                .strip_suffix('\r')
                .map(|line| (line, "\r"))
                .unwrap_or((line_without_newline, ""));
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
        let key_lower = key.map(|key| key.to_ascii_lowercase());

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

                for (key, value) in map {
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
        self.scrub_generated_tokens(&scrubbed)
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

                if is_generated_token(token, prefix) {
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

fn is_generated_token(token: &str, prefix: TokenPrefix) -> bool {
    if is_redacted_placeholder(token) {
        return false;
    }

    let Some(suffix) = token.strip_prefix(prefix.raw) else {
        return false;
    };

    suffix.len() >= prefix.min_suffix_len
        && suffix
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-'))
        && suffix.chars().any(|ch| ch.is_ascii_digit())
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
            if is_generated_token(token, prefix) && !token.contains("REDACTED_") {
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

#[allow(unused)]
fn assert_path_is_repo_relative(path: &Path) {
    assert!(path.starts_with(env!("CARGO_MANIFEST_DIR")));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query_pair(name: &str, value: &str) -> NameValue {
        NameValue {
            name: name.to_string(),
            value: value.to_string(),
        }
    }

    fn cassette_request(path: &str) -> CassetteRequest {
        CassetteRequest {
            path: path.to_string(),
            method: "POST".to_string(),
            query_param: Vec::new(),
            header: Vec::new(),
            body: None,
        }
    }

    fn cassette_response() -> CassetteResponse {
        CassetteResponse {
            status: 200,
            header: Vec::new(),
            body: None,
        }
    }

    fn incoming_request(path: &str, body: impl Into<Bytes>) -> IncomingRequest {
        IncomingRequest {
            method: Method::POST,
            uri: path.parse().expect("test URI should parse"),
            headers: axum::http::HeaderMap::new(),
            body: body.into(),
        }
    }

    #[test]
    fn query_matches_exact_pairs_in_any_order() {
        let expected = [query_pair("a", "1"), query_pair("b", "2")];

        assert!(query_matches(Some("b=2&a=1"), &expected));
    }

    #[test]
    fn query_matches_counts_duplicate_pairs() {
        let expected = [query_pair("a", "1"), query_pair("a", "1")];

        assert!(query_matches(Some("a=1&a=1"), &expected));
        assert!(!query_matches(Some("a=1"), &expected));
        assert!(!query_matches(Some("a=1&a=2"), &expected));
    }

    #[test]
    fn query_matches_rejects_extra_actual_params() {
        let expected = [query_pair("a", "1")];

        assert!(!query_matches(Some("a=1&b=2"), &expected));
    }

    #[test]
    fn query_matches_rejects_missing_actual_params() {
        let expected = [query_pair("a", "1"), query_pair("b", "2")];

        assert!(!query_matches(Some("a=1"), &expected));
    }

    #[test]
    fn query_matches_empty_expected_only_matches_empty_actual_query() {
        assert!(query_matches(None, &[]));
        assert!(query_matches(Some(""), &[]));
        assert!(!query_matches(Some("a=1"), &[]));
    }

    #[test]
    fn bodyless_cassette_request_only_matches_empty_actual_body() {
        let policy = CassettePolicy::default();
        let headers = axum::http::HeaderMap::new();

        assert!(body_matches(policy, &headers, &[], &[], None));
        assert!(!body_matches(
            policy,
            &headers,
            &[],
            br#"{"unexpected":true}"#,
            None
        ));
    }

    #[test]
    fn replay_matching_is_ordered_by_default() {
        let policy = CassettePolicy::default();
        let request = incoming_request("/v1/second", Bytes::new());
        let interactions = vec![
            ReplayInteraction {
                when: cassette_request("/v1/first"),
                then: cassette_response(),
                consumed: false,
            },
            ReplayInteraction {
                when: cassette_request("/v1/second"),
                then: cassette_response(),
                consumed: false,
            },
        ];

        assert_eq!(
            matching_interaction_index(policy, &interactions, &request),
            None
        );
    }

    #[test]
    fn replay_matching_can_be_explicitly_unordered() {
        let policy = CassettePolicy {
            replay_matching: ReplayMatching::Unordered,
            ..CassettePolicy::default()
        };
        let request = incoming_request("/v1/second", Bytes::new());
        let interactions = vec![
            ReplayInteraction {
                when: cassette_request("/v1/first"),
                then: cassette_response(),
                consumed: false,
            },
            ReplayInteraction {
                when: cassette_request("/v1/second"),
                then: cassette_response(),
                consumed: false,
            },
        ];

        assert_eq!(
            matching_interaction_index(policy, &interactions, &request),
            Some(1)
        );
    }

    #[test]
    fn replay_matching_requires_provider_auth_header_presence_without_recorded_value() {
        let policy = CassettePolicy::for_scenario("openai", "agent/completion_smoke");
        let interaction = ReplayInteraction {
            when: cassette_request("/v1/responses"),
            then: cassette_response(),
            consumed: false,
        };
        let interactions = vec![interaction];
        let request_without_auth = incoming_request("/v1/responses", Bytes::new());
        let mut request_with_auth = incoming_request("/v1/responses", Bytes::new());
        request_with_auth.headers.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer [REDACTED]"),
        );

        assert_eq!(
            matching_interaction_index(policy, &interactions, &request_without_auth),
            None
        );
        assert_eq!(
            missing_required_headers(policy, &request_without_auth.headers),
            vec!["authorization"]
        );
        assert_eq!(
            matching_interaction_index(policy, &interactions, &request_with_auth),
            Some(0)
        );
    }

    #[test]
    fn gemini_interactions_policy_requires_api_key_header() {
        let policy = CassettePolicy::for_scenario("gemini", "interactions_api/tool_result");
        let mut request = incoming_request("/v1beta/models", Bytes::new());

        assert_eq!(
            missing_required_headers(policy, &request.headers),
            vec!["x-goog-api-key"]
        );

        request
            .headers
            .insert("x-goog-api-key", HeaderValue::from_static("[REDACTED]"));

        assert!(required_headers_present(policy, &request.headers));
    }

    #[test]
    fn replay_miss_diagnostics_scrub_actual_request_details() {
        let policy = CassettePolicy::default();
        let request = incoming_request(
            "/v1/miss?key=AIzaSyExampleSecretToken123456&api_key=raw-api-key",
            r#"{"url":"https://example.test/file?api_key=body-secret&access_token=body-token","id":"resp_12345678"}"#,
        );
        let interactions = vec![ReplayInteraction {
            when: cassette_request("/v1/other"),
            then: cassette_response(),
            consumed: false,
        }];

        let message = replay_miss_message(policy, &request, &interactions);

        assert!(!message.contains("AIzaSyExampleSecretToken123456"));
        assert!(!message.contains("raw-api-key"));
        assert!(!message.contains("body-secret"));
        assert!(!message.contains("body-token"));
        assert!(!message.contains("resp_12345678"));
        assert!(message.contains(REDACTED));
        assert!(message.contains("resp_REDACTED_1"));
    }

    #[test]
    fn replay_miss_diagnostics_report_missing_required_headers_without_values() {
        let policy = CassettePolicy::for_scenario("anthropic", "agent/completion_smoke");
        let request = incoming_request("/v1/messages", Bytes::new());
        let interactions = vec![ReplayInteraction {
            when: cassette_request("/v1/messages"),
            then: cassette_response(),
            consumed: false,
        }];

        let message = replay_miss_message(policy, &request, &interactions);

        assert!(message.contains("x-api-key"));
        assert!(message.contains("missing_required_headers"));
        assert!(!message.contains("Bearer"));
        assert!(!message.contains("secret"));
    }

    #[test]
    fn safety_detects_token_shaped_openai_keys_without_echoing_value() {
        let cassette = scrub_cassette_contents(
            r#"when:
  path: /v1/responses
  method: POST
then:
  status: 200
  body: '{"leaked":"sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"}'
"#,
        );

        let failures = cassette_safety_failures(Path::new("fixture.yaml"), &cassette);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("OpenAI API key-shaped token(s)"))
        );
        assert!(
            failures
                .iter()
                .all(|failure| !failure.contains("sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"))
        );
    }

    #[test]
    fn safety_detects_token_shaped_anthropic_keys_without_echoing_value() {
        let cassette = scrub_cassette_contents(
            r#"when:
  path: /v1/messages
  method: POST
then:
  status: 200
  body: '{"leaked":"sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"}'
"#,
        );

        let failures = cassette_safety_failures(Path::new("fixture.yaml"), &cassette);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("Anthropic API key-shaped token(s)"))
        );
        assert!(
            failures
                .iter()
                .all(|failure| !failure
                    .contains("sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"))
        );
        assert!(
            failures
                .iter()
                .all(|failure| !failure.contains("OpenAI API key-shaped token(s)"))
        );
    }

    #[test]
    fn safety_detects_google_api_keys_without_echoing_value() {
        let cassette = scrub_cassette_contents(
            r#"when:
  path: /v1beta/models/gemini
  method: POST
then:
  status: 200
  body: '{"leaked":"AIzaSyExampleSecretToken1234567890"}'
"#,
        );

        let failures = cassette_safety_failures(Path::new("fixture.yaml"), &cassette);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("Google API key-shaped token(s)"))
        );
        assert!(
            failures
                .iter()
                .all(|failure| !failure.contains("AIzaSyExampleSecretToken1234567890"))
        );
    }

    #[test]
    fn safety_does_not_flag_short_sk_substrings() {
        let cassette = scrub_cassette_contents(
            r#"when:
  path: /v1/responses
  method: POST
then:
  status: 200
  body: '{"text":"the sk-etch marker is harmless"}'
"#,
        );

        assert!(cassette_safety_failures(Path::new("fixture.yaml"), &cassette).is_empty());
    }

    #[test]
    fn scrubber_preserves_repeated_ids_across_json_bodies() {
        let cassette = r#"when:
  path: /v1/files
  method: POST
then:
  status: 200
  body: '{"id":"file_011Cb1W1wnAxQP1a6AuVcPx5","type":"file","created_at":"2026-05-14T00:18:05Z"}'
---
when:
  path: /v1/messages
  method: POST
  body: '{"source":{"type":"file","file_id":"file_011Cb1W1wnAxQP1a6AuVcPx5"}}'
then:
  status: 200
  body: '{"id":"msg_01D9wgWnWe16jLatSL7ce5Gm","content":[{"type":"text","text":"rig-file-id-page-two-verifier-8c27"}]}'
---
when:
  path: /v1/files/file_011Cb1W1wnAxQP1a6AuVcPx5
  method: DELETE
  query_param:
  - name: resource
    value: file_011Cb1W1wnAxQP1a6AuVcPx5
then:
  status: 200
  body: '{"id":"file_011Cb1W1wnAxQP1a6AuVcPx5","type":"file_deleted"}'
"#;

        let scrubbed = scrub_cassette_contents(cassette);

        assert!(!scrubbed.contains("file_011Cb1W1wnAxQP1a6AuVcPx5"));
        assert_eq!(scrubbed.matches("file_REDACTED_1").count(), 5);
        assert!(scrubbed.contains("msg_REDACTED_1"));
        assert!(scrubbed.contains("rig-file-id-page-two-verifier-8c27"));
        assert_eq!(scrub_cassette_contents(&scrubbed), scrubbed);
    }

    #[test]
    fn scrubber_scrubs_sse_json_payloads() {
        let cassette = r#"when:
  path: /v1/chat/completions
  method: POST
then:
  status: 200
  header:
  - name: date
    value: Thu, 14 May 2026 00:00:00 GMT
  - name: content-type
    value: text/event-stream
  body: "data: {\"id\":\"chatcmpl-DfEFWCScgKdeItzBxcAl2DTWsWPwj\",\"created\":1778718594,\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call_vJUubymOrhXJwTYjJvSnqzAe\",\"type\":\"function\"}]}}],\"system_fingerprint\":\"fp_c27f75025a\"}\n\ndata: [DONE]\n"
"#;

        let scrubbed = scrub_cassette_contents(cassette);

        assert!(!scrubbed.contains("chatcmpl-DfEFWCScgKdeItzBxcAl2DTWsWPwj"));
        assert!(!scrubbed.contains("call_vJUubymOrhXJwTYjJvSnqzAe"));
        assert!(!scrubbed.contains("fp_c27f75025a"));
        assert!(!scrubbed.contains("date"));
        assert!(scrubbed.contains("chatcmpl-REDACTED_1"));
        assert!(scrubbed.contains("call_REDACTED_1"));
        assert!(scrubbed.contains("data: [DONE]"));
        assert!(scrubbed.contains("content-type"));
    }

    #[test]
    fn scrubber_keeps_public_model_ids() {
        let cassette = r#"when:
  path: /v1/models
  method: GET
then:
  status: 200
  body: '{"data":[{"type":"model","id":"gpt-5.2"},{"type":"model","id":"claude-sonnet-4-6"}]}'
"#;

        let scrubbed = scrub_cassette_contents(cassette);

        assert!(scrubbed.contains("gpt-5.2"));
        assert!(scrubbed.contains("claude-sonnet-4-6"));
        assert!(!scrubbed.contains("id_REDACTED"));
    }

    #[test]
    fn scrubber_removes_volatile_headers_and_sensitive_query_params() {
        let cassette = r#"when:
  path: /v1beta/models
  method: GET
  query_param:
  - name: key
    value: AIzaSySecret
then:
  status: 200
  header:
  - name: content-type
    value: application/json
  - name: x-request-id
    value: req_abc123456789
  - name: set-cookie
    value: __cf_bm=secret
  body: '{}'
"#;

        let scrubbed = scrub_cassette_contents(cassette);

        assert!(scrubbed.contains("value: '[REDACTED]'"));
        assert!(!scrubbed.contains("AIzaSySecret"));
        assert!(!scrubbed.contains("x-request-id"));
        assert!(!scrubbed.contains("set-cookie"));
        assert!(scrubbed.contains("content-type"));
    }

    #[test]
    fn scrubber_removes_sensitive_query_params_embedded_in_body_text() {
        let cassette = r#"when:
  path: /v1beta/models
  method: POST
  body: '{"url":"https://example.test/download?KEY=AIzaSyExampleSecretToken123456&API_KEY=body-secret&apikey=body-secret-2&access_token=body-token"}'
then:
  status: 200
  body: '{}'
"#;

        let scrubbed = scrub_cassette_contents(cassette);

        assert!(!scrubbed.contains("AIzaSyExampleSecretToken123456"));
        assert!(!scrubbed.contains("body-secret"));
        assert!(!scrubbed.contains("body-secret-2"));
        assert!(!scrubbed.contains("body-token"));
        assert_eq!(scrubbed.matches(REDACTED).count(), 4);
    }
}
