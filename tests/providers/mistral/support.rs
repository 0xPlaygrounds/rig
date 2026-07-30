use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::provider::ProviderConfig;
use rig::providers::mistral;

use crate::cassettes::{CassetteSpec, ProviderCassette};

const MISTRAL_BASE_URL: &str = "https://api.mistral.ai";

/// Connection data for a running Mistral cassette server.
pub(super) struct MistralCassetteEnv {
    base_url: String,
    api_key: String,
}

impl MistralCassetteEnv {
    pub(super) fn config(&self, model: &str) -> mistral::functions::Config {
        mistral::functions::Config::new(model)
            .with_api_key(&self.api_key)
            .with_base_url(&self.base_url)
    }

    pub(super) fn provider(&self, model: &str) -> ProviderConfig {
        ProviderConfig::Mistral(self.config(model))
    }
}

async fn mistral_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, MistralCassetteEnv) {
    let cassette = ProviderCassette::start("mistral", spec, MISTRAL_BASE_URL).await;
    let env = MistralCassetteEnv {
        base_url: cassette.base_url(),
        api_key: cassette.api_key("MISTRAL_API_KEY"),
    };

    (cassette, env)
}

pub(super) async fn with_mistral_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(MistralCassetteEnv) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, env) = mistral_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(env)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
