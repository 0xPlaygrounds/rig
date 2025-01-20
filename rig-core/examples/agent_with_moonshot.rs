use rig::agent::AgentBuilder;
use rig::providers::mootshot::{CompletionModel, MOOTSHOT_CHAT};
use rig::{completion::Prompt, providers};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("Running basic agent with mootshot");
    basic_mootshot().await?;

    println!("\nRunning mootshot agent with context");
    context_mootshot().await?;

    println!("\n\nAll agents ran successfully");
    Ok(())
}

fn client() -> providers::mootshot::Client {
    providers::mootshot::Client::from_env()
}

fn partial_agent_mootshot() -> AgentBuilder<CompletionModel> {
    let client = client();
    client.agent(MOOTSHOT_CHAT)
}

async fn basic_mootshot() -> Result<(), anyhow::Error> {
    let comedian_agent = partial_agent_mootshot()
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;
    println!("{}", response);

    Ok(())
}

async fn context_mootshot() -> Result<(), anyhow::Error> {
    let model = client().completion_model(MOOTSHOT_CHAT);

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
