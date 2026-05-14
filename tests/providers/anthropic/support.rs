use futures::FutureExt;
use rig::providers::anthropic;
use std::future::Future;
use std::panic::{AssertUnwindSafe, resume_unwind};

use crate::cassettes::ProviderCassette;

pub(super) struct AnthropicFilesCassette {
    pub(super) client: anthropic::Client,
    pub(super) base_url: String,
    pub(super) api_key: String,
}

async fn anthropic_cassette(scenario: &'static str) -> (ProviderCassette, anthropic::Client) {
    let cassette =
        ProviderCassette::start("anthropic", scenario, "https://api.anthropic.com").await;
    let client = anthropic::Client::builder()
        .api_key(cassette.api_key("ANTHROPIC_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

pub(super) async fn with_anthropic_cassette<F, Fut>(scenario: &'static str, test_body: F)
where
    F: FnOnce(anthropic::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = anthropic_cassette(scenario).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish().await;
    if let Err(payload) = result {
        resume_unwind(payload);
    }
}

pub(super) async fn with_anthropic_cassette_result<F, Fut, E>(
    scenario: &'static str,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(anthropic::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = anthropic_cassette(scenario).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish().await;
    match result {
        Ok(result) => result,
        Err(payload) => resume_unwind(payload),
    }
}

pub(super) async fn with_anthropic_files_cassette<F, Fut>(
    scenario: &'static str,
    beta_header: &'static str,
    test_body: F,
) where
    F: FnOnce(AnthropicFilesCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette =
        ProviderCassette::start("anthropic", scenario, "https://api.anthropic.com").await;
    let base_url = normalize_anthropic_base_url(&cassette.base_url());
    let api_key = cassette.api_key("ANTHROPIC_API_KEY");
    let client = anthropic::Client::builder()
        .api_key(&api_key)
        .base_url(&base_url)
        .anthropic_beta(beta_header)
        .build()
        .expect("client should build");

    let parts = AnthropicFilesCassette {
        client,
        base_url,
        api_key,
    };
    let result = AssertUnwindSafe(test_body(parts)).catch_unwind().await;
    cassette.finish().await;
    if let Err(payload) = result {
        resume_unwind(payload);
    }
}

fn normalize_anthropic_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');

    if let Some(stripped) = trimmed.strip_suffix("/v1/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/v1") {
        stripped.to_string()
    } else {
        trimmed.to_string()
    }
}
