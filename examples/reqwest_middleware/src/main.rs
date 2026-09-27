//! Demonstrates supplying a custom reqwest client with retry middleware.
//! Requires `ANTHROPIC_API_KEY` and the `reqwest-middleware` feature.
//! Run it to verify a wire can be bound to your preconfigured HTTP stack.

use reqwest_middleware::ClientBuilder;
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use rig::RigError;
use rig::{prelude::*, providers::anthropic, providers::anthropic::Anthropic};

fn build_http_client() -> rig::rig_reqwest::ReqwestMiddlewareClient {
    let retry_policy = ExponentialBackoff::builder().build_with_max_retries(5);
    ClientBuilder::new(Default::default())
        .with(RetryTransientMiddleware::new_with_policy(retry_policy))
        .build()
        .into()
}

#[tokio::main]
async fn main() -> Result<(), RigError> {
    let api_key = rig::client::env::required("ANTHROPIC_API_KEY")?;
    let http_client = build_http_client();
    let agent = AgentBuilder::new(
        Anthropic::new(api_key)
            .with_http(http_client)
            .completion(anthropic::completion::CLAUDE_SONNET_4_6),
    )
    .preamble("You are a helpful assistant.")
    .build();

    let response = agent.prompt("What is 2 + 2?").await?.output;
    println!("Response: {response}");

    Ok(())
}
