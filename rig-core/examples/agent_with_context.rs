use std::env;

use rig::{agent::AgentBuilder, completion::Prompt, providers::cohere};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI and Cohere clients
    // let openai_client = openai::Client::new(&env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"));
    let cohere_client =
        cohere::Client::new(&env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set"));

    // let model = openai_client.completion_model("gpt-4");
    let model = cohere_client.completion_model("command-r");

    // Create an agent with multiple context documents
    let agent = AgentBuilder::new(model)
        .context("Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .context("Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .context("Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build();

    // Prompt the agent and print the response
    let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;

    println!("{}", response);

    Ok(())
}
