use reqwest_middleware::ClientBuilder;
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use rig::{client::CompletionClient, completion::Prompt, providers::anthropic};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a reqwest client with middleware (retry logic)
    let retry_policy = ExponentialBackoff::builder().build_with_max_retries(5);
    let http_client = ClientBuilder::new(Default::default())
        .with(RetryTransientMiddleware::new_with_policy(retry_policy))
        .build();

    // API key must be set menually as we cannot use a default client
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("'ANTHROPIC_API_KEY' not set");

    // Create an Anthropic client using the middleware-enabled HTTP client
    let client = anthropic::Client::builder()
        .http_client(http_client)
        .api_key(api_key)
        .build()
        .unwrap();

    // Create an agent and send a prompt
    let agent = client
        .agent("claude-sonnet-4-20250514")
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent.prompt("What is 2 + 2?").await?;
    println!("Response: {}", response);

    Ok(())
}
