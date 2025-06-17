use rig::pipeline::{self, Op};
use rig::prelude::*;
use rig::providers::openai::client::Client;
use std::env;
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);

    let rng_agent = openai_client.agent("gpt-4")
        .preamble("
            You are a random number generator designed to only either output a single whole integer that is 0 or 1. Only return the number.
        ")
        .build();

    let adder_agent = openai_client.agent("gpt-4")
        .preamble("
            You are a mathematician who adds 1000 to every number passed into the context, except if the number is 0 - in which case don't add anything. Only return the number.
        ")
        .build();

    let chain = pipeline::new()
        // Generate a whole number that is either 0 and 1
        .prompt(rng_agent)
        .map(|x| x.unwrap())
        .prompt(adder_agent);

    // Prompt the agent and print the response
    let response = chain
        .call("Please generate a single whole integer that is 0 or 1".to_string())
        .await;

    println!("Pipeline result: {response:?}");

    Ok(())
}
