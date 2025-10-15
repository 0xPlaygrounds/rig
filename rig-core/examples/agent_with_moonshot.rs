use rig::agent::AgentBuilder;
use rig::prelude::*;
use rig::providers::moonshot::{CompletionModel, MOONSHOT_CHAT};
use rig::{completion::Prompt, providers};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("Running basic agent with moonshot");
    basic_moonshot().await?;
    println!("\nRunning moonshot agent with context");
    context_moonshot().await?;
    println!("\n\nAll agents ran successfully");
    Ok(())
}

fn client() -> providers::moonshot::Client {
    providers::moonshot::Client::from_env()
}

fn partial_agent_moonshot() -> AgentBuilder<CompletionModel> {
    let client = client();
    client.agent(MOONSHOT_CHAT)
}

async fn basic_moonshot() -> Result<(), anyhow::Error> {
    let comedian_agent = partial_agent_moonshot()
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .temperature(0.5)
        .max_tokens(1024)
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{response}");
    Ok(())
}

async fn context_moonshot() -> Result<(), anyhow::Error> {
    let model = client().completion_model(MOONSHOT_CHAT);

    // Create an agent with multiple context documents
    let agent = AgentBuilder::new(model)
        .preamble("Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .build();

    // Prompt the agent and print the response
    let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;
    println!("{response}");
    Ok(())
}
