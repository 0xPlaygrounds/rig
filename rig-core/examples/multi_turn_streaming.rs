use rig::streaming::StreamingPrompt;
use rig::{
    agent::stream_to_stdout,
    client::{CompletionClient, ProviderClient},
    completion::ToolDefinition,
    providers::anthropic,
    tool::Tool,
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // tracing_subscriber::registry()
    //     .with(
    //         tracing_subscriber::EnvFilter::try_from_default_env()
    //             .unwrap_or_else(|_| "stdout=info".into()),
    //     )
    //     .with(tracing_subscriber::fmt::layer())
    //     .init();

    // Create Anthropic client
    let anthropic_client = anthropic::Client::from_env();

    // Create agent with a single context prompt and a calculator tools
    let calculator_agent = anthropic_client
        .agent(anthropic::CLAUDE_3_5_SONNET)
        .preamble(
            "You are an assistant here to help the user select which tool is most appropriate to perform arithmetic operations.
            Follow these instructions closely.
            1. Consider the user's request carefully and identify the core elements of the request.
            2. Select which tool among those made available to you is appropriate given the context.
            3. This is very important: never perform the operation yourself.
            "
        )
        .tool(Add)
        .tool(Subtract)
        .tool(Multiply)
        .tool(Divide)
        .build();

    // Prompt the agent and get the stream
    let prompt = "Calculate 2 * (3 + 5) / 9  = ?. Describe the result to me.";
    let mut stream = calculator_agent.stream_prompt(prompt).multi_turn(10).await;

    stream_to_stdout(&mut stream).await?;

    Ok(())
}

#[derive(Deserialize, JsonSchema)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Deserialize, Serialize)]
struct Add;
impl Tool for Add {
    const NAME: &'static str = "add";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y together".to_string(),
            parameters: serde_json::to_value(schema_for!(OperationArgs))
                .expect("converting JSON schema to JSON value should never fail"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "subtract".to_string(),
            description: "Subtract y from x (i.e.: x - y)".to_string(),
            parameters: serde_json::to_value(schema_for!(OperationArgs))
                .expect("converting JSON schema to JSON value should never fail"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x - args.y;
        Ok(result)
    }
}

struct Multiply;
impl Tool for Multiply {
    const NAME: &'static str = "multiply";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "multiply".to_string(),
            description: "Compute the product of x and y (i.e.: x * y)".to_string(),
            parameters: serde_json::to_value(schema_for!(OperationArgs))
                .expect("converting JSON schema to JSON value should never fail"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x * args.y;
        Ok(result)
    }
}

struct Divide;
impl Tool for Divide {
    const NAME: &'static str = "divide";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "divide".to_string(),
            description: "Compute the Quotient of x and y (i.e.: x / y). Useful for ratios."
                .to_string(),
            parameters: serde_json::to_value(schema_for!(OperationArgs))
                .expect("converting JSON schema to JSON value should never fail"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x / args.y;
        Ok(result)
    }
}
