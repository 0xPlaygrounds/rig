use futures::FutureExt;
use rig::client::DefaultTransportBuilder as _;
use rig::providers::perplexity;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn perplexity_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, perplexity::Client) {
    let cassette = ProviderCassette::start("perplexity", spec, "https://api.perplexity.ai").await;
    let client = perplexity::Client::builder()
        .api_key(cassette.api_key("PERPLEXITY_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("Perplexity cassette client should build");

    (cassette, client)
}

pub(super) async fn with_perplexity_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(perplexity::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = perplexity_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Compare a generated token (response id, system fingerprint, request id)
/// observed by a test with the value its fixture holds.
///
/// Fixtures are placeholder-scrubbed (`chatcmpl-REDACTED_1`, `fp_REDACTED_1`,
/// `req_REDACTED_1`), so on the recording pass the live token cannot equal
/// the fixture's; both are then required to be present and non-empty. On
/// replay the harness serves the scrubbed bytes back, so equality is exact —
/// which is what CI runs. Presence must agree in both modes.
pub(super) fn assert_matches_recorded_token(
    actual: Option<&str>,
    recorded: Option<&str>,
    context: &str,
) {
    match crate::cassettes::CassetteMode::current() {
        crate::cassettes::CassetteMode::Replay => {
            assert_eq!(
                actual, recorded,
                "{context}: replay serves the fixture's token back"
            );
        }
        crate::cassettes::CassetteMode::Record => {
            assert_eq!(
                actual.is_some(),
                recorded.is_some(),
                "{context}: live and recorded token presence must agree"
            );
            if let (Some(actual), Some(recorded)) = (actual, recorded) {
                assert!(
                    !actual.trim().is_empty() && !recorded.trim().is_empty(),
                    "{context}: live and recorded token must both be non-empty"
                );
            }
        }
    }
}

/// Cassette wrapper for the perplexity prompt-caching matrix
/// (`tests/cassettes/perplexity/prompt_caching/`).
///
/// Delegates to [`with_perplexity_cassette`] — the behavior is identical, and deliberately shared
/// so the two cannot drift apart when the base wrapper gains policy. What the
/// separate name buys is a per-suite entry in the cassette-safety registry, so
/// the cache fixtures are auditable as one concern's evidence.
pub(super) async fn with_perplexity_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(perplexity::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_perplexity_cassette(spec, test_body).await;
}
