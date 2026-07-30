use futures::FutureExt;
use rig::provider::ProviderConfig;
use rig::providers::doubleword;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

const DOUBLEWORD_BASE_URL: &str = "https://api.doubleword.ai/v1";

/// Connection data for a running Doubleword cassette server.
///
/// Providers are data now: the cassette hands the test the base URL and API
/// key, and each test mints a `functions::Config` for the model it exercises.
pub(super) struct DoublewordCassetteEnv {
    base_url: String,
    api_key: String,
}

impl DoublewordCassetteEnv {
    pub(super) fn config(&self, model: &str) -> doubleword::functions::Config {
        doubleword::functions::Config::new(model)
            .with_api_key(&self.api_key)
            .with_base_url(&self.base_url)
    }

    pub(super) fn provider(&self, model: &str) -> ProviderConfig {
        ProviderConfig::Doubleword(self.config(model))
    }

    pub(super) fn embedding_config(&self, model: &str) -> doubleword::functions::EmbeddingConfig {
        doubleword::functions::EmbeddingConfig::new(model)
            .with_api_key(&self.api_key)
            .with_base_url(&self.base_url)
    }
}

async fn doubleword_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, DoublewordCassetteEnv) {
    let cassette = ProviderCassette::start("doubleword", spec, DOUBLEWORD_BASE_URL).await;
    let env = DoublewordCassetteEnv {
        base_url: cassette.base_url(),
        api_key: cassette.api_key("DOUBLEWORD_API_KEY"),
    };

    (cassette, env)
}

pub(super) async fn with_doubleword_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(DoublewordCassetteEnv) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, env) = doubleword_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(env)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_doubleword_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(DoublewordCassetteEnv) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, env) = doubleword_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(env)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
