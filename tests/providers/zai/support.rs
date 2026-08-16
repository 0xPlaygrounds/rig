//! Cassette wrappers for Z.AI's three client surfaces.
//!
//! Z.AI is a dual-dialect provider: one OpenAI-compatible client served from
//! two base URLs (`/api/paas/v4` general, `/api/coding/paas/v4` coding) and one
//! Anthropic-compatible client (`/api/anthropic`). Each gets its own wrapper so
//! a scenario's fixture proves which endpoint the call actually reached — the
//! cassette's recorded path is the only evidence that base-URL composition did
//! not drop or double a prefix. `ProviderCassette::start` splits the base into
//! origin + path and serves the path back, so the wrappers pass the *full*
//! documented base, suffix included.
//!
//! Scenario names are prefixed by dialect (`general/`, `coding/`,
//! `anthropic/`), which is also what selects the required-header policy in
//! `tests/common/cassettes.rs` — see the `zai` arm of
//! `CassettePolicy::for_scenario`.

use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::providers::zai;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// A deliberately invalid key, sent in *both* record and replay modes.
///
/// `ProviderCassette::bogus_api_key` would also satisfy the `zai` policy's
/// required-header check (it returns `[REDACTED]` in replay, never an empty
/// value), so this is not a correctness requirement — it is the
/// `with_anthropic_cassette_bogus_key` shape: one literal, identical on both
/// sides, so the header a rejection cell sends is legible in the source rather
/// than mode-dependent.
const BOGUS_API_KEY: &str = "invalid-edge-matrix-key";

/// The scenario prefix is not cosmetic: `CassettePolicy::for_scenario` reads it
/// to decide which auth header the replayed request must carry (`x-api-key` for
/// `anthropic/…`, `authorization` otherwise). A cell filed under the wrong
/// prefix would demand a header its client never sends, and every replay would
/// fail as an opaque mock miss — so each wrapper refuses the mismatch up front.
fn assert_anthropic_dialect_scenario(spec: &CassetteSpec) {
    assert!(
        spec.scenario().starts_with("anthropic/"),
        "Anthropic-dialect scenarios must be prefixed `anthropic/`, got {:?}",
        spec.scenario()
    );
}

fn assert_openai_dialect_scenario(spec: &CassetteSpec) {
    assert!(
        !spec.scenario().starts_with("anthropic/"),
        "the `anthropic/` prefix selects the x-api-key policy; an OpenAI-dialect \
         cell must use `general/` or `coding/`, got {:?}",
        spec.scenario()
    );
}

async fn zai_openai_cassette(
    spec: impl Into<CassetteSpec>,
    real_base_url: &str,
) -> (ProviderCassette, zai::Client) {
    let spec = spec.into();
    assert_openai_dialect_scenario(&spec);
    let cassette = ProviderCassette::start("zai", spec, real_base_url).await;
    let client = zai::Client::builder()
        .api_key(cassette.api_key("ZAI_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("Z.AI client should build");

    (cassette, client)
}

/// The general OpenAI-compatible endpoint (`https://api.z.ai/api/paas/v4`).
pub(super) async fn with_zai_general_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(zai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = zai_openai_cassette(spec, zai::GENERAL_API_BASE_URL).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// The coding OpenAI-compatible endpoint
/// (`https://api.z.ai/api/coding/paas/v4`), which serves a smaller model set
/// than the general one.
pub(super) async fn with_zai_coding_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(zai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = zai_openai_cassette(spec, zai::CODING_API_BASE_URL).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// The Anthropic-compatible endpoint (`https://api.z.ai/api/anthropic`), which
/// rig reaches through a whole different client and shared layer.
pub(super) async fn with_zai_anthropic_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(zai::AnthropicClient) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    assert_anthropic_dialect_scenario(&spec);
    let cassette = ProviderCassette::start("zai", spec, zai::ANTHROPIC_API_BASE_URL).await;
    let client = zai::AnthropicClient::builder()
        .api_key(cassette.api_key("ZAI_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("Z.AI Anthropic-compatible client should build");

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Bogus-key variant of [`with_zai_general_cassette`], for recording a real
/// rejection rather than a synthesized one.
pub(super) async fn with_zai_general_cassette_bogus_key<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(zai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    assert_openai_dialect_scenario(&spec);
    let cassette = ProviderCassette::start("zai", spec, zai::GENERAL_API_BASE_URL).await;
    let client = zai::Client::builder()
        .api_key(BOGUS_API_KEY)
        .base_url(cassette.base_url())
        .build()
        .expect("Z.AI client should build");

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Bogus-key variant of [`with_zai_anthropic_cassette`].
pub(super) async fn with_zai_anthropic_cassette_bogus_key<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(zai::AnthropicClient) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    assert_anthropic_dialect_scenario(&spec);
    let cassette = ProviderCassette::start("zai", spec, zai::ANTHROPIC_API_BASE_URL).await;
    let client = zai::AnthropicClient::builder()
        .api_key(BOGUS_API_KEY)
        .base_url(cassette.base_url())
        .build()
        .expect("Z.AI Anthropic-compatible client should build");

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// The path of a scenario's **first** recorded request.
///
/// This is the only record of how the base URL composed with the endpoint
/// suffix — a doubled `/v4` or a dropped `/api` is invisible from the response
/// — so the smoke cell of each dialect asserts it.
pub(super) fn recorded_request_path(scenario: &str) -> String {
    let path = crate::cassettes::cassette_path("zai", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));

    contents
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("path: "))
        .unwrap_or_else(|| panic!("cassette {} should record a request path", path.display()))
        .to_string()
}

/// The request body of a scenario's **first** recorded interaction, as JSON.
///
/// A cassette replays *responses*; the recorded request is the only record of
/// what rig put on the wire, and several Z.AI cells are about exactly that (a
/// `response_format` that must not be sent, a `stream_options` block Z.AI does
/// not document). Reading it back is how those cells assert their own premise
/// instead of assuming it.
///
/// A multi-turn cassette holds one `---`-separated document per interaction;
/// this reads the first, which is the turn every caller here is about. Call it
/// *after* the cassette wrapper returns: in record mode the fixture is written
/// on the way out, so an in-body read would see the previous recording.
pub(super) fn recorded_request_body(scenario: &str) -> serde_json::Value {
    let path = crate::cassettes::cassette_path("zai", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));

    let (request, _) = contents.split_once("\nthen:\n").unwrap_or_else(|| {
        panic!(
            "cassette {} should have a `then:` response section",
            path.display()
        )
    });
    let line = request
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("body: "))
        .unwrap_or_else(|| panic!("cassette {} should record a request body", path.display()));

    parse_yaml_scalar_json(line, &path.display().to_string())
}

/// The response body of a scenario's **first** recorded interaction, as JSON.
pub(super) fn recorded_response_body(scenario: &str) -> serde_json::Value {
    let path = crate::cassettes::cassette_path("zai", scenario);
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
        "cassette {} records a base64 body; this helper reads JSON response bodies only",
        path.display()
    );
    let line = response
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("body: "))
        .unwrap_or_else(|| panic!("cassette {} should record a response body", path.display()));

    parse_yaml_scalar_json(line, &path.display().to_string())
}

/// Everything a cassette recorded from its first `then:` onwards, verbatim.
///
/// Streaming fixtures record SSE frames rather than one JSON document, so the
/// JSON readers above cannot be reused for them; callers match on frame text.
pub(super) fn recorded_response_text(scenario: &str) -> String {
    let path = crate::cassettes::cassette_path("zai", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));

    let (_, response) = contents.split_once("\nthen:\n").unwrap_or_else(|| {
        panic!(
            "cassette {} should have a `then:` response section",
            path.display()
        )
    });

    // Stop at the next document separator: a multi-turn cassette continues with
    // turn 2's *request*, whose body would otherwise be searched as if it were
    // response text.
    match response.split_once("\n---\n") {
        Some((first, _)) => first.to_string(),
        None => response.to_string(),
    }
}

/// The harness writes a body as a single-line YAML single-quoted scalar, in
/// which a literal quote is doubled.
fn parse_yaml_scalar_json(line: &str, cassette: &str) -> serde_json::Value {
    assert_ne!(
        line, "null",
        "cassette {cassette} recorded no body here; this reader is for cells whose \
         turn carries a JSON document"
    );
    let json = line
        .strip_prefix('\'')
        .and_then(|rest| rest.strip_suffix('\''))
        .map(|body| body.replace("''", "'"))
        .unwrap_or_else(|| line.to_string());

    serde_json::from_str(&json)
        .unwrap_or_else(|err| panic!("cassette {cassette} body should be JSON: {err}"))
}
