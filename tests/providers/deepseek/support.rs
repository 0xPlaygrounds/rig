use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::provider::ProviderConfig;
use rig::providers::deepseek;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Connection data for a running DeepSeek cassette server.
///
/// Providers are data now: the cassette hands the test the base URL and API
/// key, and each test mints a `functions::Config` for the model it exercises.
pub(super) struct DeepSeekCassetteEnv {
    base_url: String,
    api_key: String,
}

impl DeepSeekCassetteEnv {
    pub(super) fn config(&self, model: &str) -> deepseek::functions::Config {
        deepseek::functions::Config::new(model)
            .with_api_key(&self.api_key)
            .with_base_url(&self.base_url)
    }

    pub(super) fn provider(&self, model: &str) -> ProviderConfig {
        ProviderConfig::DeepSeek(self.config(model))
    }
}

async fn deepseek_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, DeepSeekCassetteEnv) {
    let cassette = ProviderCassette::start("deepseek", spec, "https://api.deepseek.com").await;
    let env = DeepSeekCassetteEnv {
        base_url: cassette.base_url(),
        api_key: cassette.api_key("DEEPSEEK_API_KEY"),
    };

    (cassette, env)
}

pub(super) async fn with_deepseek_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(DeepSeekCassetteEnv) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, env) = deepseek_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(env)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_deepseek_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(DeepSeekCassetteEnv) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, env) = deepseek_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(env)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
