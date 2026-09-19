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

/// Cassette wrapper for the perplexity prompt-caching matrix
/// (`crates/rig-cassette/fixtures/cassettes/perplexity/prompt_caching/`).
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
