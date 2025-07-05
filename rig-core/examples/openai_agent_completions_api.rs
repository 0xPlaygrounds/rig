//! This example shows how you can use OpenAI's Completions API.
//! By default, the OpenAI integration uses the Responses API. However, for the sake of backwards compatibility you may wish to use the Completions API.

use rig::completion::Prompt;
use rig::prelude::*;
use std::env;

use rig::providers;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let agent = providers::openai::Client::new(
        &env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
    )
    .completion_model("gpt-4o")
    .completions_api()
    .into_agent_builder()
    .preamble("You are a helpful assistant")
    .build();

    let res = agent.prompt("Hello world!").await.unwrap();

    println!("GPT-4o: {res}");

    Ok(())
}
