use futures::FutureExt;
use rig::providers::anthropic;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

pub(super) struct AnthropicFilesCassette {
    pub(super) client: anthropic::Client,
    pub(super) base_url: String,
    pub(super) api_key: String,
}

async fn anthropic_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, anthropic::Client) {
    let cassette = ProviderCassette::start("anthropic", spec, "https://api.anthropic.com").await;
    let client = anthropic::Client::builder()
        .api_key(cassette.api_key("ANTHROPIC_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

pub(super) async fn with_anthropic_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(anthropic::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = anthropic_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Drive rig's Anthropic client against an Anthropic-*compatible* gateway
/// rather than `api.anthropic.com`.
///
/// The code under test is still the Anthropic provider — only the endpoint
/// differs. Gateways that reimplement the Messages API do not always reproduce
/// Anthropic's own wire choices, and where they diverge the Anthropic adapter
/// is what has to cope. Recording that divergence needs traffic from the
/// gateway itself, so this wrapper points upstream at OpenRouter's Messages
/// endpoint and records with `OPENROUTER_API_KEY`. Replay needs no key, like
/// every other cassette.
///
/// Cassettes recorded through here live under `tests/cassettes/anthropic/` with
/// the rest of the provider's scenarios; the gateway is an implementation
/// detail of how the fixture was obtained, not a separate provider suite.
pub(super) async fn with_anthropic_gateway_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(anthropic::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start("anthropic", spec, OPENROUTER_MESSAGES_BASE_URL).await;
    let client = anthropic::Client::builder()
        .api_key(cassette.api_key("OPENROUTER_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// OpenRouter's Anthropic Messages endpoint, minus the `/v1/messages` suffix
/// the Anthropic client appends itself.
const OPENROUTER_MESSAGES_BASE_URL: &str = "https://openrouter.ai/api";

pub(super) async fn with_anthropic_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(anthropic::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = anthropic_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

pub(super) async fn with_anthropic_files_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    beta_header: &'static str,
    test_body: F,
) where
    F: FnOnce(AnthropicFilesCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start("anthropic", spec, "https://api.anthropic.com").await;
    let base_url = normalize_anthropic_base_url(&cassette.base_url());
    let api_key = cassette.api_key("ANTHROPIC_API_KEY");
    let client = anthropic::Client::builder()
        .api_key(&api_key)
        .base_url(&base_url)
        .anthropic_beta(beta_header)
        .build()
        .expect("client should build");

    let parts = AnthropicFilesCassette {
        client,
        base_url,
        api_key,
    };
    let result = AssertUnwindSafe(test_body(parts)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

fn normalize_anthropic_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');

    if let Some(stripped) = trimmed.strip_suffix("/v1/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/v1") {
        stripped.to_string()
    } else {
        trimmed.to_string()
    }
}

/// The JSON response body a *blocking* Anthropic cassette recorded.
///
/// Reading the fixture back is how a cell asserts its own premise: a recorded
/// turn that quietly stopped producing the shape the cell is about would
/// otherwise keep the cell green while it covers nothing.
pub(super) fn recorded_response_body(scenario: &str) -> serde_json::Value {
    let path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));

    let (_, response) = contents.split_once("\nthen:\n").unwrap_or_else(|| {
        panic!(
            "cassette {} should have a `then:` response section",
            path.display()
        )
    });
    // The harness can record a non-UTF-8 body as base64 (`body_encoding`).
    // No caller of this helper records one, and reading such a body as JSON
    // would be silently wrong — so refuse rather than guess.
    assert!(
        !response.contains("body_encoding: base64"),
        "cassette {} records a base64 body; this helper reads JSON response \
         bodies only",
        path.display()
    );
    let line = response
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("body: "))
        .unwrap_or_else(|| panic!("cassette {} should record a response body", path.display()));

    // The harness writes the body as a single-line YAML single-quoted scalar,
    // in which a literal quote is doubled.
    let json = line
        .strip_prefix('\'')
        .and_then(|rest| rest.strip_suffix('\''))
        .map(|body| body.replace("''", "'"))
        .unwrap_or_else(|| line.to_string());

    serde_json::from_str(&json).unwrap_or_else(|err| {
        panic!(
            "response body in cassette {} should be JSON: {err}",
            path.display()
        )
    })
}

/// Cassette wrapper for the `stop_sequence`-on-the-streamed-terminal matrix.
///
/// Delegates to [`with_anthropic_cassette`] — the behavior is identical, and
/// deliberately shared so the three cannot drift apart when the base wrapper
/// gains policy. What the separate name buys is a per-bug entry in the
/// cassette-safety registry, so `tests/cassettes/anthropic/stop_sequence_terminal_matrix/`
/// is auditable as one bug's evidence. (The fixture *path* comes from the
/// scenario string, not from the wrapper.)
pub(super) async fn with_anthropic_stop_sequence_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(anthropic::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// Cassette wrapper for the empty-content-`stop_sequence` normalization matrix.
///
/// Delegates to [`with_anthropic_cassette`], separate for the same per-bug
/// registry reason as [`with_anthropic_stop_sequence_cassette`] (see
/// `tests/cassettes/anthropic/empty_stop_sequence_matrix/`).
pub(super) async fn with_anthropic_empty_stop_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(anthropic::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// Like [`with_anthropic_cassette`], but the client authenticates with a
/// deliberately invalid API key — for recording real 401 responses without a
/// secret anywhere near the fixture (auth headers are neither recorded nor
/// matched by the harness).
pub(super) async fn with_anthropic_cassette_bogus_key<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(anthropic::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start("anthropic", spec, "https://api.anthropic.com").await;
    let client = anthropic::Client::builder()
        .api_key("sk-invalid-edge-matrix-key")
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
