use std::{
    error::Error,
    fmt::{Display, Formatter},
};

use rig::{
    agent::AgentBuilder,
    completion::{Prompt, ToolDefinition},
    loaders::FileLoader,
    tool::Tool,
};
use rig_bedrock::{
    client::{Client, ClientBuilder},
    completion::AMAZON_NOVA_LITE_V1,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::info;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathError {
    Reason(&'static str),
}

impl Display for MathError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use MathError::*;
        match self {
            Reason(err) => write!(f, "Math error: {}", err),
        }
    }
}

impl Error for MathError {}

#[derive(Deserialize, Serialize)]
struct Adder;
impl Tool for Adder {
    const NAME: &'static str = "add";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x + args.y;
        Ok(result)
    }
}

/// Runs 4 agents based on AWS Bedrock (dervived from the agent_with_grok example)
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

async fn client() -> Client {
    ClientBuilder::new().build().await
}

async fn partial_agent() -> AgentBuilder<rig_bedrock::completion::CompletionModel> {
    let client = client().await;
    client.agent(AMAZON_NOVA_LITE_V1)
}

/// Create an AWS Bedrock agent with a system prompt
async fn basic() -> Result<(), anyhow::Error> {
    let agent = partial_agent()
        .await
        .preamble("Answer with json format only")
        .build();

    let response = agent.prompt("Describe solar system").await?;
    info!("{}", response);

    Ok(())
}

/// Create an AWS Bedrock with tools
async fn tools() -> Result<(), anyhow::Error> {
    let calculator_agent = partial_agent()
        .await
        .preamble("You must only do math by using a tool.")
        .max_tokens(1024)
        .tool(Adder)
        .build();

    info!(
        "Calculator Agent: add 400 and 20\nResult: {}",
        calculator_agent.prompt("add 400 and 20").await?
    );

    Ok(())
}

async fn context() -> Result<(), anyhow::Error> {
    let model = client().await.completion_model(AMAZON_NOVA_LITE_V1);

    // Create an agent with multiple context documents
    let agent = AgentBuilder::new(model)
        .preamble("Answer the question")
        .context("Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .context("Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
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
    let model = client().await.completion_model(AMAZON_NOVA_LITE_V1);

    // Load in all the rust examples
    let examples = FileLoader::with_glob("rig-core/examples/*.rs")?
        .read_with_path()
        .ignore_errors()
        .into_iter();

    // Create an agent with multiple context documents
    let agent = examples
        .fold(AgentBuilder::new(model), |builder, (path, content)| {
            builder.context(format!("Rust Example {:?}:\n{}", path, content).as_str())
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
