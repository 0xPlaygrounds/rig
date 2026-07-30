use futures::FutureExt;
use rig::provider::ProviderConfig;
use rig::providers::xai;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Connection data for a running xAI cassette server.
///
/// Providers are data now: the cassette hands the test the base URL and API
/// key, and each test mints a `functions::Config` for the model it exercises.
pub(super) struct XaiCassetteEnv {
    base_url: String,
    api_key: String,
}

impl XaiCassetteEnv {
    pub(super) fn config(&self, model: &str) -> xai::functions::Config {
        xai::functions::Config::new(model)
            .with_api_key(&self.api_key)
            .with_base_url(&self.base_url)
    }

    pub(super) fn provider_config(&self, model: &str) -> ProviderConfig {
        ProviderConfig::Xai(self.config(model))
    }
}

async fn xai_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, XaiCassetteEnv) {
    let cassette = ProviderCassette::start("xai", spec, "https://api.x.ai").await;
    let env = XaiCassetteEnv {
        base_url: cassette.base_url(),
        api_key: cassette.api_key("XAI_API_KEY"),
    };

    (cassette, env)
}

pub(super) async fn with_xai_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(XaiCassetteEnv) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, env) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(env)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_xai_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(XaiCassetteEnv) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, env) = xai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(env)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
