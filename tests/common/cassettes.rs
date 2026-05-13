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
use httpmock::MockServer;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

const MODE_ENV: &str = "RIG_PROVIDER_TEST_MODE";
const CASSETTE_ROOT: &str = "tests/cassettes";
const REDACTED: &str = "[REDACTED]";
const DUMMY_API_KEY: &str = REDACTED;

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

            let recording = server
                .record_async(|rule| {
                    rule.record_request_headers(vec![
                        "accept",
                        "content-type",
                        "anthropic-version",
                        "anthropic-beta",
                        "openai-beta",
                    ])
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
                CassetteServer::Replay(ReplayServer::start(&cassette_path).await),
                None,
            )
        };

        Self {
            server,
            cassette_path,
            base_path: upstream.path,
            mode,
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
        let redacted = redact_secrets(&yaml);

        if let Some(parent) = self.cassette_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .expect("cassette directory should be created");
        }

        tokio::fs::write(&self.cassette_path, redacted)
            .await
            .expect("provider cassette should be written");
    }
}

struct ReplayServer {
    addr: SocketAddr,
}

impl ReplayServer {
    async fn start(cassette_path: &Path) -> Self {
        let contents = tokio::fs::read_to_string(cassette_path)
            .await
            .unwrap_or_else(|error| {
                panic!(
                    "provider cassette {} should be readable: {error}",
                    cassette_path.display()
                )
            });
        let interactions = parse_cassette(cassette_path, &contents);
        let state = Arc::new(Mutex::new(ReplayState { interactions }));
        let app = Router::new()
            .fallback(any(replay_request))
            .with_state(state);
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

        Self { addr }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

struct ReplayState {
    interactions: Vec<ReplayInteraction>,
}

#[derive(Debug, Deserialize)]
struct CassetteInteraction {
    when: CassetteRequest,
    then: CassetteResponse,
}

#[derive(Debug, Deserialize)]
struct CassetteRequest {
    path: String,
    method: String,
    #[serde(default)]
    query_param: Vec<NameValue>,
    #[serde(default)]
    header: Vec<NameValue>,
    body: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CassetteResponse {
    status: u16,
    #[serde(default)]
    header: Vec<NameValue>,
    body: Option<String>,
}

#[derive(Debug, Deserialize)]
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
    serde_yaml::Deserializer::from_str(contents)
        .map(|document| {
            CassetteInteraction::deserialize(document).unwrap_or_else(|error| {
                panic!(
                    "provider cassette {} should deserialize: {error}",
                    cassette_path.display()
                )
            })
        })
        .map(|interaction| ReplayInteraction {
            when: interaction.when,
            then: interaction.then,
            consumed: false,
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

    let Some(index) = state.interactions.iter().position(|interaction| {
        !interaction.consumed && request_matches(&request, &interaction.when)
    }) else {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("content-type", "application/json")
            .body(r#"{"message":"Request did not match any route or mock"}"#.into())
            .expect("replay miss response should build");
    };

    let interaction = &mut state.interactions[index];
    interaction.consumed = true;
    cassette_response(&interaction.then)
}

struct IncomingRequest {
    method: Method,
    uri: axum::http::Uri,
    headers: axum::http::HeaderMap,
    body: Bytes,
}

fn request_matches(request: &IncomingRequest, expected: &CassetteRequest) -> bool {
    request
        .method
        .as_str()
        .eq_ignore_ascii_case(&expected.method)
        && request.uri.path() == expected.path
        && query_matches(request.uri.query(), &expected.query_param)
        && headers_match(&request.headers, &expected.header)
        && body_matches(
            &request.headers,
            &expected.header,
            &request.body,
            expected.body.as_deref(),
        )
}

fn query_matches(query: Option<&str>, expected: &[NameValue]) -> bool {
    let actual = url::form_urlencoded::parse(query.unwrap_or_default().as_bytes())
        .into_owned()
        .collect::<Vec<_>>();

    expected.iter().all(|expected| {
        actual
            .iter()
            .any(|(name, value)| name == &expected.name && value == &expected.value)
    })
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

fn body_matches(
    actual_headers: &axum::http::HeaderMap,
    expected_headers: &[NameValue],
    actual: &[u8],
    expected: Option<&str>,
) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    if is_multipart_request(actual_headers, expected_headers) {
        return true;
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

fn cassette_path(provider: &str, scenario: &str) -> PathBuf {
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

fn redact_secrets(yaml: &str) -> String {
    let mut redacted = Vec::new();
    let mut redact_next_value = false;

    for line in yaml.lines() {
        let trimmed = line.trim_start();
        let lower = trimmed.to_ascii_lowercase();
        let is_sensitive_header_name = lower.starts_with("- name: set-cookie")
            || lower.starts_with("- name: openai-organization")
            || lower.starts_with("- name: openai-project")
            || lower.starts_with("- name: x-api-key")
            || lower.starts_with("- name: x-goog-api-key")
            || lower.starts_with("- name: key");

        let line = if redact_next_value && lower.starts_with("value:") {
            redact_next_value = false;
            let indentation_len = line.len() - trimmed.len();
            format!("{}value: '{}'", &line[..indentation_len], REDACTED)
        } else if lower.starts_with("authorization:")
            || lower.starts_with("x-api-key:")
            || lower.starts_with("api-key:")
            || lower.starts_with("x-goog-api-key:")
            || lower.starts_with("ocp-apim-subscription-key:")
            || lower.starts_with("set-cookie:")
            || lower.starts_with("openai-organization:")
            || lower.starts_with("openai-project:")
        {
            let indentation_len = line.len() - trimmed.len();
            format!(
                "{}{}: {}",
                &line[..indentation_len],
                key_before_colon(trimmed),
                REDACTED
            )
        } else {
            if is_sensitive_header_name {
                redact_next_value = true;
            }
            redact_query_api_key(line)
        };

        redacted.push(line);
    }

    let mut output = redacted.join("\n");
    if yaml.ends_with('\n') {
        output.push('\n');
    }
    output
}

fn key_before_colon(line: &str) -> &str {
    line.split_once(':')
        .map(|(key, _)| key)
        .expect("redacted header line should contain colon")
}

fn redact_query_api_key(line: &str) -> String {
    redact_query_param(line, "key")
}

fn redact_query_param(line: &str, key: &str) -> String {
    let mut output = String::with_capacity(line.len());
    let mut remainder = line;

    while let Some(index) = remainder.find(&format!("{key}=")) {
        let (prefix, after_prefix) = remainder.split_at(index);
        output.push_str(prefix);
        output.push_str(key);
        output.push('=');

        let value_start = key.len() + 1;
        let after_value_start = &after_prefix[value_start..];
        let value_end = after_value_start
            .find(['&', '"', '\'', ' ', '\n'])
            .unwrap_or(after_value_start.len());
        output.push_str(REDACTED);
        remainder = &after_value_start[value_end..];
    }

    output.push_str(remainder);
    output
}

#[allow(unused)]
fn assert_path_is_repo_relative(path: &Path) {
    assert!(path.starts_with(env!("CARGO_MANIFEST_DIR")));
}
