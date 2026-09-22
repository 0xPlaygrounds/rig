use futures::FutureExt;
use rig::driver::Bound;
use rig::http_client::BoxedHttpClient;
use rig::prelude::*;
use rig::providers::openai::wire::{OpenAI, VENICE};
use rig::providers::venice;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

// The direct-recording client is shared with the other suites whose response
// bodies are binary; it is gated on the same feature that compiles this
// scenario, since the PR gate builds this target with the default feature set
// where an ungated helper would be dead code under `-D warnings`.
use crate::cassettes::DirectRecordingHttpClient;

const VENICE_BASE_URL: &str = venice::VENICE_API_BASE_URL;

/// The Venice dialect of the OpenAI config bound to the bundled transport —
/// what a cassette test builds its models from, now that a model is a bound
/// wire.
pub(super) type BoundVenice = Bound<OpenAI, BoxedHttpClient>;

/// The same config on the direct-recording transport, for the suites whose
/// response bodies are binary.
pub(super) type DirectVenice = Bound<OpenAI, DirectRecordingHttpClient>;

/// The Venice config pointed at `cassette`.
fn venice_config(cassette: &ProviderCassette) -> OpenAI {
    OpenAI::with_key(&VENICE, cassette.api_key("VENICE_API_KEY")).with_base_url(cassette.base_url())
}

async fn venice_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, BoundVenice) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "venice",
        spec,
        VENICE_BASE_URL,
    )
    .await;
    let venice = venice_config(&cassette)
        .bound()
        .expect("transport should build");

    (cassette, venice)
}

pub(super) async fn with_venice_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(BoundVenice) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, client) = venice_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    crate::cassettes::checkpoint_attempt(&cassette, "venice", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}

/// Cassette wrapper for scenarios whose response body is binary; see
/// [`DirectRecordingHttpClient`].
pub(super) async fn with_venice_direct_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(DirectVenice) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start_via(
        rig_cassette::http::Transport::Direct,
        &crate::cassettes::cassette_root(),
        "venice",
        spec,
        VENICE_BASE_URL,
    )
    .await;
    let client =
        venice_config(&cassette).bind(DirectRecordingHttpClient::new(cassette.direct_recorder()));

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_venice_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(BoundVenice) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = venice_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Cassette wrapper for the venice prompt-caching matrix
/// (`crates/rig-cassette/fixtures/cassettes/venice/prompt_caching/`).
///
/// Delegates to [`with_venice_cassette`] — the behavior is identical, and deliberately shared
/// so the two cannot drift apart when the base wrapper gains policy. What the
/// separate name buys is a per-suite entry in the cassette-safety registry, so
/// the cache fixtures are auditable as one concern's evidence.
pub(super) async fn with_venice_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(BoundVenice) -> Fut,
    Fut: Future<Output = ()>,
{
    with_venice_cassette(spec, test_body).await;
}
