//! Cassette-backed provider test helpers.
//!
//! Provider cassette tests run in replay mode by default. Set
//! `RIG_PROVIDER_TEST_MODE=record` or `RIG_PROVIDER_TEST_MODE=refresh` to hit
//! the real provider and write cassette fixtures.
#![allow(dead_code)]

use httpmock::MockServer;
use std::fmt;
use std::path::{Path, PathBuf};

const MODE_ENV: &str = "RIG_PROVIDER_TEST_MODE";
const CASSETTE_ROOT: &str = "tests/cassettes";
const REDACTED: &str = "[REDACTED]";
const DUMMY_API_KEY: &str = "rig-test-redacted";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CassetteMode {
    Replay,
    Record,
    Refresh,
}

impl CassetteMode {
    fn current() -> Self {
        match std::env::var(MODE_ENV) {
            Ok(value) if value.eq_ignore_ascii_case("record") => Self::Record,
            Ok(value) if value.eq_ignore_ascii_case("refresh") => Self::Refresh,
            Ok(value) if value.eq_ignore_ascii_case("replay") => Self::Replay,
            Ok(value) => {
                panic!("{MODE_ENV} must be one of replay, record, or refresh; got {value:?}")
            }
            Err(_) => Self::Replay,
        }
    }

    fn records(self) -> bool {
        matches!(self, Self::Record | Self::Refresh)
    }
}

pub(crate) struct ProviderCassette {
    server: MockServer,
    cassette_path: PathBuf,
    base_path: String,
    mode: CassetteMode,
    recording_id: Option<usize>,
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
        let server = MockServer::start_async().await;

        let recording_id = if mode.records() {
            if mode == CassetteMode::Record && cassette_path.exists() {
                panic!(
                    "cassette already exists at {}; use {MODE_ENV}=refresh to replace it",
                    cassette_path.display()
                );
            }

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

            Some(recording.id)
        } else {
            if !cassette_path.exists() {
                panic!(
                    "missing provider cassette {}; run with {MODE_ENV}=record and the real API key to create it",
                    cassette_path.display()
                );
            }
            server.playback_async(&cassette_path).await;
            None
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

        let recording = httpmock::Recording::new(recording_id, &self.server);
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

    for line in yaml.lines() {
        let trimmed = line.trim_start();
        let lower = trimmed.to_ascii_lowercase();
        let line = if lower.starts_with("authorization:")
            || lower.starts_with("x-api-key:")
            || lower.starts_with("api-key:")
            || lower.starts_with("x-goog-api-key:")
            || lower.starts_with("ocp-apim-subscription-key:")
        {
            let indentation_len = line.len() - trimmed.len();
            format!(
                "{}{}: {}",
                &line[..indentation_len],
                key_before_colon(trimmed),
                REDACTED
            )
        } else {
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
