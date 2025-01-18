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

    // Note that you can also create your own semantic router for this
    // that uses a vector store under the hood
    let animal_agent = openai_client.agent("gpt-4")
        .preamble("
            Your role is to categorise the user's statement using the following values: [sheep, cow, dog]

            Return only the value.
        ")
        .build();

    let default_agent = openai_client.agent("gpt-4")
        .build();

    let chain = pipeline::new()
        // Use our classifier agent to classify the agent under a number of fixed topics
        .prompt(animal_agent)
        // Change the prompt depending on the output from the prompt
        .map(|x: String| {
            match x.trim() {
                "cow" => Ok("Tell me a fact about the United States of America.".into()),
                "sheep" => Ok("Calculate 5+5 for me. Return only the number.".into()),
                "dog" => Ok("Write me a poem about cashews".into()),
                message => Err(format!("Could not process - received category: {message}"))
            }
        })
        // Send the prompt back into another agent with no pre-amble
        .prompt(default_agent);

    // Prompt the agent and print the response
    let response = chain.try_call("Sheep can self-medicate").await?;

    match response {
        Ok(res) => println!("Successful pipeline run: {res:?}"),
        Err(e) => println!("Unsuccessful pipeline run: {res:?}")
    }

    Ok(())
}
