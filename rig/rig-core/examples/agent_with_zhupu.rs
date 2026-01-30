use rig::agent::AgentBuilder;
use rig::prelude::*;
use rig::providers::zhipu::{CompletionModel, ZHIPU_CHAT};
use rig::{completion::Prompt, providers};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("Running basic agent with zhipu");
    basic_zhipu().await?;
    println!("\nRunning zhipu agent with context");
    context_zhipu().await?;
    println!("\n\nAll agents ran successfully");
    Ok(())
}

fn client() -> providers::zhipu::Client {
    providers::zhipu::Client::from_env()
}

fn partial_agent_zhipu() -> AgentBuilder<CompletionModel> {
    let client = client();
    client.agent(ZHIPU_CHAT)
}

async fn basic_zhipu() -> Result<(), anyhow::Error> {
    let comedian_agent = partial_agent_zhipu()
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .temperature(0.5)
        .max_tokens(1024)
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{response}");
    Ok(())
}

async fn context_zhipu() -> Result<(), anyhow::Error> {
    let model = client().completion_model(ZHIPU_CHAT);

    // Create an agent with multiple context documents
    let agent = AgentBuilder::new(model)
        .preamble("Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .build();

    // Prompt the agent and print the response
    let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;
    println!("{response}");
    Ok(())
}
