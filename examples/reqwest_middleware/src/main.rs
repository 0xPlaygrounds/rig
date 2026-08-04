//! Demonstrates driving an agent over a caller-supplied middleware HTTP stack.
//! Requires `ANTHROPIC_API_KEY`.
//!
//! Transport is not a provider or runtime type parameter. Instead,
//! [`HttpRuntime::from_transport`] erases the configured middleware client at
//! construction and the concrete provider client retains that runtime. The
//! same runtime is available through `client.http()` for low-level provider
//! free functions.

use std::time::Duration;

use anyhow::Result;
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use rig::prelude::*;
use rig::providers::anthropic;

/// A preconfigured middleware client with bounded timeouts and retries.
fn build_http_client() -> Result<ClientWithMiddleware> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .connect_timeout(Duration::from_secs(10))
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(8)
        .build()?;
    let retry_policy = ExponentialBackoff::builder().build_with_max_retries(3);

    Ok(ClientBuilder::new(client)
        .with(RetryTransientMiddleware::new_with_policy(retry_policy))
        .build())
}

#[tokio::main]
async fn main() -> Result<()> {
    let http = HttpRuntime::from_transport(build_http_client()?);
    let client = anthropic::Client::builder()
        .api_key(std::env::var("ANTHROPIC_API_KEY")?)
        .http_runtime(http)
        .build()?;

    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent.prompt("What is 2 + 2?").await?;
    println!("Response: {response}");

    Ok(())
}
