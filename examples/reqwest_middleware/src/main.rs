//! Demonstrates driving an agent over a caller-supplied HTTP stack.
//! Requires `ANTHROPIC_API_KEY`.
//!
//! Transport is no longer a type parameter. The concrete provider client
//! builder accepts a live [`HttpRuntime`]: build your own
//! `reqwest::Client` — timeouts, connection pool, default headers, proxy,
//! TLS — hand it to [`HttpRuntime::from_reqwest`], then build agents from the
//! resulting client. The same runtime is available through `client.http()` for
//! low-level provider free functions.
//!
//! Note: `reqwest_middleware::ClientWithMiddleware` (this example's former
//! subject) is not currently one of `HttpRuntime`'s transport arms, so
//! retry/tracing middleware has to be expressed on the `reqwest::Client`
//! itself or around the call site.

use std::time::Duration;

use anyhow::Result;
use rig::prelude::*;
use rig::providers::anthropic;

/// A preconfigured HTTP client: bounded timeouts and a warm connection pool.
fn build_http_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .connect_timeout(Duration::from_secs(10))
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(8)
        .build()?)
}

#[tokio::main]
async fn main() -> Result<()> {
    let http = HttpRuntime::from_reqwest(build_http_client()?);
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
