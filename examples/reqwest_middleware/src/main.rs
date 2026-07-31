//! Demonstrates driving an agent over a caller-supplied HTTP stack.
//! Requires `ANTHROPIC_API_KEY`.
//!
//! Transport is no longer a type parameter or a provider-client builder
//! option. The provider is plain data (`anthropic::functions::Config`), and
//! the live HTTP handle lives in [`HttpRuntime`]: build your own
//! `reqwest::Client` — timeouts, connection pool, default headers, proxy,
//! TLS — hand it to [`HttpRuntime::from_reqwest`], wrap it in a
//! [`Runtime`] with [`Runtime::with_http`], and pass that to
//! `AgentBuilder::runtime`. The same `HttpRuntime` also goes straight into
//! any provider free function (`anthropic::functions::complete(&cfg, &rt, …)`).
//!
//! Note: `reqwest_middleware::ClientWithMiddleware` (this example's former
//! subject) is not currently one of `HttpRuntime`'s transport arms, so
//! retry/tracing middleware has to be expressed on the `reqwest::Client`
//! itself or around the call site.

use std::sync::Arc;
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
    let cfg = anthropic::functions::Config::from_env(anthropic::completion::CLAUDE_SONNET_4_6)?;

    let agent = AgentBuilder::new(cfg)
        .runtime(Arc::new(Runtime::with_http(http)))
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent.prompt("What is 2 + 2?").await?;
    println!("Response: {response}");

    Ok(())
}
