use futures::FutureExt;
use rig::driver::Bound;
use rig::http_client::BoxedHttpClient;
use rig::prelude::*;
use rig::providers::openai::wire::{OpenAI, PERPLEXITY};
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// The Perplexity dialect of the OpenAI config bound to the bundled
/// transport — what a cassette test builds its models from, now that a model
/// is a bound wire.
pub(super) type BoundPerplexity = Bound<OpenAI, BoxedHttpClient>;

async fn perplexity_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, BoundPerplexity) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "perplexity",
        spec,
        "https://api.perplexity.ai",
    )
    .await;
    let perplexity = OpenAI::with_key(&PERPLEXITY, cassette.api_key("PERPLEXITY_API_KEY"))
        .with_base_url(cassette.base_url())
        .bound()
        .expect("Perplexity cassette transport should build");

    (cassette, perplexity)
}

pub(super) async fn with_perplexity_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(BoundPerplexity) -> Fut,
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
    F: FnOnce(BoundPerplexity) -> Fut,
    Fut: Future<Output = ()>,
{
    with_perplexity_cassette(spec, test_body).await;
}
