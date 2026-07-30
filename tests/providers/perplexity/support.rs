use futures::FutureExt;
use rig::providers::perplexity;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Connection data for a running Perplexity cassette server.
///
/// Providers are data now: the cassette hands the test the base URL and API
/// key, and each test mints a `functions::Config` for the model it exercises.
pub(super) struct PerplexityCassetteEnv {
    base_url: String,
    api_key: String,
}

impl PerplexityCassetteEnv {
    pub(super) fn config(&self, model: &str) -> perplexity::functions::Config {
        perplexity::functions::Config::new(model)
            .with_api_key(&self.api_key)
            .with_base_url(&self.base_url)
    }
}

async fn perplexity_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, PerplexityCassetteEnv) {
    let cassette = ProviderCassette::start("perplexity", spec, "https://api.perplexity.ai").await;
    let env = PerplexityCassetteEnv {
        base_url: cassette.base_url(),
        api_key: cassette.api_key("PERPLEXITY_API_KEY"),
    };

    (cassette, env)
}

pub(super) async fn with_perplexity_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(PerplexityCassetteEnv) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, env) = perplexity_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(env)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
