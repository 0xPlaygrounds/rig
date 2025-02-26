use std::env;

use rig::{
    pipeline::{self, Op, TryOp},
    providers::openai::Client,
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

    let default_agent = openai_client.agent("gpt-4").build();

    let chain = pipeline::new()
        // Use our classifier agent to classify the agent under a number of fixed topics
        .prompt(animal_agent)
        // Change the prompt depending on the output from the prompt
        .map_ok(|x: String| match x.trim() {
            "cow" => Ok("Tell me a fact about the United States of America.".to_string()),
            "sheep" => Ok("Calculate 5+5 for me. Return only the number.".to_string()),
            "dog" => Ok("Write me a poem about cashews".to_string()),
            message => Err(format!("Could not process - received category: {message}")),
        })
        .map(|x| x.unwrap().unwrap())
        // Send the prompt back into another agent with no pre-amble
        .prompt(default_agent);

    // Prompt the agent and print the response
    let response = chain.try_call("Sheep can self-medicate").await?;

    println!("Pipeline result: {response:?}");

    Ok(())
}
