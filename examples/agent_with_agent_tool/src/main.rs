use anyhow::Result;
use rig::prelude::*;
use rig::providers;
use rig::providers::openai::OpenAI;
use rig::tool::{Schema, Tool};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, schemars::JsonSchema)]
struct OperationArgs {
    /// The first operand
    x: i32,
    /// The second operand
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

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> Schema {
        Schema::arguments::<Self::Args>()
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        println!("[tool-call] Adding {} and {}", args.x, args.y);
        let result = args.x + args.y;
        Ok(result)
    }
}

#[derive(Deserialize, Serialize)]
struct Subtract;

impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Subtract y from x (i.e.: x - y)".to_string()
    }

    fn parameters(&self) -> Schema {
        Schema::arguments::<Self::Args>()
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        println!("[tool-call] Subtracting {} from {}", args.y, args.x);
        let result = args.x - args.y;
        Ok(result)
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // The OpenAI client, from `OPENAI_API_KEY`.
    let openai_client = OpenAI::from_env()?;

    // One model serves both agents: erase it once, clone the handle.
    let model = openai_client.completion(providers::openai::GPT_4O).erase();

    // Create agent with a single context prompt and two tools
    let calculator_agent = AgentBuilder::new(model.clone())
        .preamble("You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.")
        .max_tokens(1024)
        .default_max_turns(2)
        .tool(Adder)
        .tool(Subtract)
        .build();

    // Create agent which has the calculator_agent as a tool
    let agent_using_agent = AgentBuilder::new(model)
        .preamble("You are a helpful assistant that can solve problems. Use the tool provided to answer the user's question.")
        .max_tokens(1024)
        .default_max_turns(2)
        .dynamic_tool(calculator_agent.into_tool()?)
        .build();

    // Prompt the agent and print the response
    println!("Calculate 2 - 5");

    println!(
        "OpenAI Agent-Using Agent: {}",
        agent_using_agent.prompt("Calculate 2 - 5").await?.output()
    );

    Ok(())
}
