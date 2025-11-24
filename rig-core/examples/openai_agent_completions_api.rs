//! This example shows how you can use OpenAI's Completions API.
//! By default, the OpenAI integration uses the Responses API. However, for the sake of backwards compatibility you may wish to use the Completions API.

use rig::completion::Prompt;
use rig::prelude::*;

use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let agent = openai::Client::from_env()
        .completion_model(openai::GPT_4O)
        .completions_api()
        .into_agent_builder()
        .preamble("You are a helpful assistant")
        .build();

    let res = agent.prompt("Hello world!").await.unwrap();

    println!("GPT-4o: {res}");

    Ok(())
}
