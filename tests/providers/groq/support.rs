use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::provider::ProviderConfig;
use rig::providers::groq;

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Connection data for a running Groq cassette server.
pub(super) struct GroqCassetteEnv {
    base_url: String,
    api_key: String,
}

impl GroqCassetteEnv {
    pub(super) fn config(&self, model: &str) -> groq::functions::Config {
        groq::functions::Config::new(model)
            .with_api_key(&self.api_key)
            .with_base_url(&self.base_url)
    }

    pub(super) fn provider(&self, model: &str) -> ProviderConfig {
        ProviderConfig::Groq(self.config(model))
    }
}

async fn groq_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, GroqCassetteEnv) {
    let cassette = ProviderCassette::start("groq", spec, "https://api.groq.com/openai/v1").await;
    let env = GroqCassetteEnv {
        base_url: cassette.base_url(),
        api_key: cassette.api_key("GROQ_API_KEY"),
    };

    (cassette, env)
}

pub(super) async fn with_groq_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(GroqCassetteEnv) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, env) = groq_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(env)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}
