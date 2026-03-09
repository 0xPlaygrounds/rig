use rig::agent::AgentBuilder;
use rig::completion::Prompt;
use rig_eternalai::providers::eternalai::{
    Client, CompletionModel, NOUS_RESEARCH_HERMES_3_LLAMA_3_1_70B_FP8,
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    println!("Running basic agent with eternalai");
    basic_eternalai().await?;

    println!("\nRunning eternalai agent with context");
    context_eternalai().await?;

    println!("\n\nAll agents ran successfully");
    Ok(())
}

fn client() -> Client {
    Client::from_env()
}

fn partial_agent_eternalai() -> AgentBuilder<CompletionModel> {
    let client = client();
    client.agent(
        NOUS_RESEARCH_HERMES_3_LLAMA_3_1_70B_FP8,
        // Option::from("45762"),
        None,
    )
}

async fn basic_eternalai() -> Result<(), anyhow::Error> {
    let comedian_agent = partial_agent_eternalai()
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{response}");

    Ok(())
}

async fn context_eternalai() -> Result<(), anyhow::Error> {
    let model = client().completion_model(
        NOUS_RESEARCH_HERMES_3_LLAMA_3_1_70B_FP8,
        Option::from("45762"),
        // None,
    );

    // Create an agent with multiple context documents
    let agent = AgentBuilder::new(model)
        .context("Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .context("Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .context("Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build();

    // Prompt the agent and print the response
    let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;

    println!("{response}");

    Ok(())
}
