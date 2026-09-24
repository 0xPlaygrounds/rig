use futures::FutureExt;
use rig::driver::{Bind, Bound};
use rig::http_client::{BoxedHttpClient, ReqwestClient};
use rig::prelude::*;
use rig::providers::anthropic::wire::Anthropic;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

pub(super) struct AnthropicFilesCassette {
    pub(super) bound: Bound<Anthropic>,
    pub(super) base_url: String,
    pub(super) api_key: String,
}

async fn anthropic_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, Bound<Anthropic>) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "anthropic",
        spec,
        "https://api.anthropic.com",
    )
    .await;
    let bound = Anthropic::new(cassette.api_key("ANTHROPIC_API_KEY"))
        .with_base_url(cassette.base_url())
        .bound()
        .expect("transport should build");

    (cassette, bound)
}

/// Like [`with_anthropic_cassette`], but the client sends through the erased
/// [`BoxedHttpClient`] wrapping the same bundled transport — the client type
/// names no concrete `H`. Replaying a recorded scenario through it proves the
/// erasure is byte-transparent: the replay server matches on body bytes.
pub(super) async fn with_anthropic_boxed_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic, BoxedHttpClient>) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "anthropic",
        spec,
        "https://api.anthropic.com",
    )
    .await;
    let bound = Anthropic::new(cassette.api_key("ANTHROPIC_API_KEY"))
        .with_base_url(cassette.base_url())
        .bind(ReqwestClient::default().boxed());
    let result = AssertUnwindSafe(test_body(bound)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Cassette wrapper for the run-lifecycle matrix (PR #2407): the client sends
/// through a [`BoxedHttpClient`] carrying the supplied [`HttpMiddleware`], so
/// the same recorded exchange exercises the transport middleware seam and the
/// run lifecycle hooks together (see
/// `crates/rig-cassette/fixtures/cassettes/anthropic/lifecycle_matrix/`).
pub(super) async fn with_anthropic_lifecycle_cassette<M, F, Fut>(
    spec: impl Into<CassetteSpec>,
    middleware: M,
    test_body: F,
) where
    M: rig::http_client::HttpMiddleware + 'static,
    F: FnOnce(Bound<Anthropic, BoxedHttpClient>) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "anthropic",
        spec,
        "https://api.anthropic.com",
    )
    .await;
    let bound = Anthropic::new(cassette.api_key("ANTHROPIC_API_KEY"))
        .with_base_url(cassette.base_url())
        .bind(ReqwestClient::default().boxed().with_middleware(middleware));
    let result = AssertUnwindSafe(test_body(bound)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_anthropic_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, bound) = anthropic_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(bound)).catch_unwind().await;
    crate::cassettes::checkpoint_attempt(&cassette, "anthropic", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}

/// Per-bug wrapper for the model-turn termination-metadata matrix
/// (`crates/rig-cassette/fixtures/cassettes/anthropic/turn_termination_matrix/`), rig#2184.
pub(super) async fn with_anthropic_turn_metadata_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
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
/// Cassettes recorded through here live under `crates/rig-cassette/fixtures/cassettes/anthropic/`
/// with
/// the rest of the provider's scenarios; the gateway is an implementation
/// detail of how the fixture was obtained, not a separate provider suite.
pub(super) async fn with_anthropic_gateway_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "anthropic",
        spec,
        OPENROUTER_MESSAGES_BASE_URL,
    )
    .await;
    let bound = Anthropic::new(cassette.api_key("OPENROUTER_API_KEY"))
        .with_base_url(cassette.base_url())
        .bound()
        .expect("transport should build");

    let result = AssertUnwindSafe(test_body(bound)).catch_unwind().await;
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
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, bound) = anthropic_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(bound)).catch_unwind().await;
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
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "anthropic",
        spec,
        "https://api.anthropic.com",
    )
    .await;
    let base_url = normalize_anthropic_base_url(&cassette.base_url());
    let api_key = cassette.api_key("ANTHROPIC_API_KEY");
    let bound = Anthropic::new(api_key.as_str())
        .with_base_url(&base_url)
        .with_beta(beta_header)
        .bound()
        .expect("transport should build");

    let parts = AnthropicFilesCassette {
        bound,
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
        .map_or_else(|| line.to_string(), |body| body.replace("''", "'"));

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
/// cassette-safety registry, so `crates/rig-
/// cassette/fixtures/cassettes/anthropic/stop_sequence_terminal_matrix/`
/// is auditable as one bug's evidence. (The fixture *path* comes from the
/// scenario string, not from the wrapper.)
pub(super) async fn with_anthropic_stop_sequence_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// Cassette wrapper for the empty-content-`stop_sequence` normalization matrix.
///
/// Delegates to [`with_anthropic_cassette`], separate for the same per-bug
/// registry reason as [`with_anthropic_stop_sequence_cassette`] (see
/// `crates/rig-cassette/fixtures/cassettes/anthropic/empty_stop_sequence_matrix/`).
pub(super) async fn with_anthropic_empty_stop_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
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
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "anthropic",
        spec,
        "https://api.anthropic.com",
    )
    .await;
    // The rejected credential is this wrapper's subject.
    cassette.expect_account_failure(crate::cassettes::AccountFailure::Auth);
    let bound = Anthropic::new("sk-invalid-edge-matrix-key")
        .with_base_url(cassette.base_url())
        .bound()
        .expect("transport should build");
    let result = AssertUnwindSafe(test_body(bound)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Cassette wrapper for the extended-thinking usage matrix.
///
/// Delegates to [`with_anthropic_cassette`], separate for the same per-bug
/// registry reason as [`with_anthropic_stop_sequence_cassette`] (see
/// `crates/rig-cassette/fixtures/cassettes/anthropic/reasoning_usage_matrix/`).
pub(super) async fn with_anthropic_reasoning_usage_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The effect corpus's request-shape matrix (Matrix E), one wrapper per
/// matrix so its cassettes are one suite directory
/// (`crates/rig-cassette/fixtures/cassettes/anthropic/corpus_request_shape/`).
pub(super) async fn with_anthropic_corpus_request_shape_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The effect corpus's hook matrix (Matrix B):
/// `crates/rig-cassette/fixtures/cassettes/anthropic/corpus_hooks/`.
pub(super) async fn with_anthropic_corpus_hooks_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The effect corpus's serving matrix (Matrix C):
/// `crates/rig-cassette/fixtures/cassettes/anthropic/corpus_serving/`.
pub(super) async fn with_anthropic_corpus_serving_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The effect corpus's outcome matrix (Matrix D):
/// `crates/rig-cassette/fixtures/cassettes/anthropic/corpus_outcome/`.
pub(super) async fn with_anthropic_corpus_outcome_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The effect corpus's endings matrix (Matrix F):
/// `crates/rig-cassette/fixtures/cassettes/anthropic/corpus_endings/`.
pub(super) async fn with_anthropic_corpus_endings_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The effect corpus's oracle matrix (Matrix O):
/// `crates/rig-cassette/fixtures/cassettes/anthropic/corpus_oracle/`.
pub(super) async fn with_anthropic_corpus_oracle_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The effect corpus's per-turn shaping matrix (Matrix M):
/// `crates/rig-cassette/fixtures/cassettes/anthropic/corpus_shaping/`.
pub(super) async fn with_anthropic_corpus_shaping_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The effect corpus's memory matrix (Matrix J):
/// `crates/rig-cassette/fixtures/cassettes/anthropic/corpus_memory/`.
pub(super) async fn with_anthropic_corpus_memory_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The effect corpus's layers matrix (Matrix P):
/// `crates/rig-cassette/fixtures/cassettes/anthropic/corpus_layers/`.
pub(super) async fn with_anthropic_corpus_layers_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The effect corpus's causal-dispatch matrix (Matrix Q):
/// `crates/rig-cassette/fixtures/cassettes/anthropic/corpus_causal/`.
pub(super) async fn with_anthropic_corpus_causal_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The effect corpus's host-families matrix (Matrix I):
/// `crates/rig-cassette/fixtures/cassettes/anthropic/corpus_host/`.
pub(super) async fn with_anthropic_corpus_host_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The effect corpus's output-mode matrix (Matrix H):
/// `crates/rig-cassette/fixtures/cassettes/anthropic/corpus_output/`.
pub(super) async fn with_anthropic_corpus_output_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<Anthropic>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_anthropic_cassette(spec, test_body).await;
}

/// The `request-id` response header each interaction of an Anthropic cassette
/// recorded, in wire order — one entry per interaction, `None` for an
/// interaction whose response carried no such header.
///
/// The header is the transport id Anthropic support asks for, and the harness
/// keeps it (placeholder-scrubbed) precisely so a cell can prove the wire
/// reported one. Reading it back from the fixture is how the raw-capture and
/// parity matrices assert that premise instead of assuming it.
pub(super) fn recorded_request_id_headers(scenario: &str) -> Vec<Option<String>> {
    let path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));

    let mut ids = Vec::new();
    let mut in_response = false;
    let mut pending_request_id_header = false;
    for line in contents.lines() {
        if line == "when:" {
            in_response = false;
        } else if line == "then:" {
            in_response = true;
            ids.push(None);
        } else if in_response {
            let trimmed = line.trim_start();
            if pending_request_id_header {
                pending_request_id_header = false;
                if let Some(value) = trimmed.strip_prefix("value: ")
                    && let Some(slot) = ids.last_mut()
                {
                    *slot = Some(value.trim().to_string());
                }
            } else if trimmed == "- name: request-id" {
                pending_request_id_header = true;
            }
        }
    }
    ids
}

/// JSON `data:` frames of one recorded SSE body, excluding `[DONE]`.
///
/// `crate::cassettes::recorded_sse_json_frames` reads only a scenario's first
/// interaction; multi-interaction streamed cells (agent tool runs) need the
/// frames of *each* interaction, which they get by pairing this with
/// `crate::cassettes::recorded_interaction_bodies`.
pub(super) fn sse_json_frames(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.trim().strip_prefix("data:"))
        .map(str::trim)
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| {
            serde_json::from_str(payload)
                .unwrap_or_else(|err| panic!("recorded SSE frame should be JSON: {err}"))
        })
        .collect()
}

/// Assert that a sequence of ids the code under test observed is the sequence
/// the fixture recorded — in both cassette modes.
///
/// On replay the observed values are the recorded ids (or, in a fixture
/// recorded before ids were kept verbatim, their numbered placeholders), so
/// the two sequences must be identical. In record mode the observed values
/// are fresh live ids, so only their equality structure is comparable: which positions repeat, and
/// which are new. Checking that
/// structure in both modes (and exact equality on replay) means the same cell
/// proves "each attempt reports its own id, in recorded order" whether it is
/// being recorded or replayed.
pub(super) fn assert_ids_match_recording(
    observed: &[Option<String>],
    recorded: &[Option<String>],
    context: &str,
) {
    fn ranks(ids: &[Option<String>]) -> Vec<Option<usize>> {
        let mut seen: Vec<&str> = Vec::new();
        ids.iter()
            .map(|id| {
                id.as_deref().map(|id| {
                    seen.iter()
                        .position(|known| *known == id)
                        .unwrap_or_else(|| {
                            seen.push(id);
                            seen.len() - 1
                        })
                })
            })
            .collect()
    }

    assert_eq!(
        observed.len(),
        recorded.len(),
        "{context}: observed {observed:?} and recorded {recorded:?} must have one id per interaction"
    );
    assert_eq!(
        ranks(observed),
        ranks(recorded),
        "{context}: observed {observed:?} must repeat/differ exactly as the recording {recorded:?}"
    );
    if crate::cassettes::CassetteMode::current() == crate::cassettes::CassetteMode::Replay {
        assert_eq!(
            observed, recorded,
            "{context}: on replay the observed ids are the fixture's own"
        );
    }
}
