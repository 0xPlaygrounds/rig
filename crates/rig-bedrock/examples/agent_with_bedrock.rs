use rig_agent::{agent::AgentBuilder, provider::ProviderConfig};
use rig_bedrock::completion::AMAZON_NOVA_LITE;
use rig_core::loaders::FileLoader;
use tracing::info;

mod common;

/// Runs 4 agents based on AWS Bedrock (derived from the agent_with_grok example)
#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    info!("Running basic agent");
    basic().await?;

    info!("\nRunning agent with tools");
    tools().await?;

    info!("\nRunning agent with loaders");
    loaders().await?;

    info!("\nRunning agent with context");
    context().await?;

    info!("\n\nAll agents ran successfully");
    Ok(())
}

/// Bedrock provider configuration using the AWS SDK's default credential
/// chain and region resolution (the config-level equivalent of
/// `Client::from_env`).
fn provider() -> ProviderConfig {
    ProviderConfig::Bedrock(rig_bedrock::functions::Config::new(AMAZON_NOVA_LITE))
}

fn partial_agent() -> AgentBuilder {
    AgentBuilder::new(provider())
}

/// Create an AWS Bedrock agent with a system prompt
async fn basic() -> Result<(), anyhow::Error> {
    let agent = partial_agent()
        .preamble("Answer with json format only")
        .build();

    let response = agent.prompt("Describe solar system").await?;
    info!("{}", response);

    Ok(())
}

/// Create an AWS Bedrock with tools
async fn tools() -> Result<(), anyhow::Error> {
    let calculator_agent = partial_agent()
        .preamble("You must only do math by using a tool.")
        .max_tokens(1024)
        .tool(common::Adder)
        .build();

    info!(
        "Calculator Agent: add 400 and 20\nResult: {}",
        calculator_agent.prompt("add 400 and 20").await?
    );

    Ok(())
}

async fn context() -> Result<(), anyhow::Error> {
    // Create an agent with multiple context documents
    let agent = AgentBuilder::new(provider())
        .preamble("Answer the question")
        .context("Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .context("Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .context("Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build();

    // Prompt the agent and print the response
    let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;

    info!("What does \"glarb-glarb\" mean?\n{}", response);

    Ok(())
}

/// Based upon the `loaders` example
///
/// This example loads in all the rust examples from the rig-core crate and uses them as\\
///  context for the agent
async fn loaders() -> Result<(), anyhow::Error> {
    // Load in all the rust examples
    let examples = FileLoader::with_glob("examples/*.rs")?
        .read_with_path()
        .ignore_errors()
        .into_iter();

    // Create an agent with multiple context documents
    let agent = examples
        .fold(AgentBuilder::new(provider()), |builder, (path, content)| {
            builder.context(format!("Rust Example {path:?}:\n{content}").as_str())
        })
        .preamble("Answer the question")
        .build();

    // Prompt the agent and print the response
    let response = agent
        .prompt("Which rust example is best suited for the operation 1 + 2")
        .await?;

    info!("{}", response);

    Ok(())
}
