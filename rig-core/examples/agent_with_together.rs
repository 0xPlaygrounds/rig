use rig::{agent::AgentBuilder, completion::Prompt, providers::together_ai};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Together AI client
    let together_ai_client = together_ai::Client::new(&std::env::var("TOGETHER_AI_API_KEY").expect("TOGETHER_AI_API_KEY not set"));

    // Choose a model, replace "together-model-v1" with an actual Together AI model name
    let model = together_ai_client.completion_model("together-model-v1");

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
