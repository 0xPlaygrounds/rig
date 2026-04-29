//! Demonstrates supplying a custom reqwest client with retry middleware.
//! Requires `GEMINI_API_KEY` and the `reqwest-middleware` feature.
//! Run it to verify the provider client can use your preconfigured HTTP stack.

use anyhow::{Context, Result};
use reqwest_middleware::ClientBuilder;
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use rig::{client::CompletionClient, completion::Prompt, providers::gemini};

fn build_http_client() -> reqwest_middleware::ClientWithMiddleware {
    let retry_policy = ExponentialBackoff::builder().build_with_max_retries(5);

    ClientBuilder::new(Default::default())
        .with(RetryTransientMiddleware::new_with_policy(retry_policy))
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("GEMINI_API_KEY").context("GEMINI_API_KEY is not set")?;
    let http_client = build_http_client();
    let client = gemini::Client::builder()
        .http_client(http_client)
        .api_key(api_key)
        .build()?;

    let agent = client
        .agent(gemini::completion::GEMINI_3_1_FLASH_LITE_PREVIEW)
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent.prompt("What is 2 + 2?").await?;
    println!("Response: {}", response);

    Ok(())
}
