use rig::prelude::*;
use rig::{
    agent::AgentBuilder,
    completion::{Prompt, ToolDefinition},
    providers::together,
    tool::Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("Running basic agent with together");
    basic().await?;
    println!("\nRunning tools agent with tools");
    tools().await?;
    println!("\nRunning together agent with context");
    context().await?;
    println!("\n\nAll agents ran successfully");
    Ok(())
}

async fn basic() -> Result<(), anyhow::Error> {
    let together_ai_client = together::Client::from_env();
    // Choose a model, replace "together-model-v1" with an actual Together AI model name
    let model =
        together_ai_client.completion_model(rig::providers::together::MIXTRAL_8X7B_INSTRUCT_V0_1);
    let agent = AgentBuilder::new(model)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();
    // Prompt the agent and print the response
    let response = agent.prompt("Entertain me!").await?;
    println!("{response}");
    Ok(())
}

async fn tools() -> Result<(), anyhow::Error> {
    // Create Together AI client
    let together_ai_client = together::Client::from_env();
    // Choose a model, replace "together-model-v1" with an actual Together AI model name
    let model =
        together_ai_client.completion_model(rig::providers::together::MIXTRAL_8X7B_INSTRUCT_V0_1);
    // Create an agent with multiple context documents
    let calculator_agent = AgentBuilder::new(model)
        .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
        .tool(Adder)
        .build();
    // Prompt the agent and print the response
    println!("Calculate 5 + 3");
    println!(
        "Calculator Agent: {}",
        calculator_agent.prompt("Calculate 5 + 3").await?
    );
    Ok(())
}

async fn context() -> Result<(), anyhow::Error> {
    // Create Together AI client
    let together_ai_client = together::Client::from_env();

    // Choose a model, replace "together-model-v1" with an actual Together AI model name
    let model =
        together_ai_client.completion_model(rig::providers::together::MIXTRAL_8X7B_INSTRUCT_V0_1);

    // Create an agent with multiple context documents
    let agent = AgentBuilder::new(model)
        .context("Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
        .context("Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
        .context("Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
        .build();

    // Prompt the agent and print the response
    let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;
    println!("{response}");
    Ok(())
}

#[derive(Debug, Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

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
        println!("The args: {args:?}");
        let result = args.x + args.y;
        Ok(result)
    }
}
