use std::env;

use rig::{
    embeddings::EmbeddingsBuilder,
    parallel,
    pipeline::{self, agent_ops::lookup, passthrough, Op},
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::in_memory_store::InMemoryVectorStore,
};

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
            You are a mathematician who adds 1000 to every number passed into the context. Only return the number.
        ")
        .build();

    let chain = pipeline::new()
        // Generate a whole number that is either 0 and 1
        .prompt(rng_agent)
        // If zero, return early. If not, continue
        .map(|x: u32| {
            if x == u32 { Ok(x + 1)} else { Err("x is 0")}
        })
        // Extra prompt here to add 1000 to the resultant number if Ok
        .prompt(adder_agent);

    // Prompt the agent and print the response
    let response = chain.try_call("Please generate a single whole integer that is 0 or 1").await;

    match response {
        Ok(res) => println!("Successful pipeline run: {res:?}"),
        Err(e) => println!("Unsuccessful pipeline run: {res:?}")
    }

    Ok(())
}
