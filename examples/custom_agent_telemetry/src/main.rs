use rig::client::{CompletionClient, ProviderClient};
use rig::core::tool::{PortableTool, ToolExecutionError};
use rig::custom_agent::CustomAgentBuilder;
use rig::providers::openai;
use serde::Deserialize;
use serde_json::json;
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Deserialize)]
struct AddArgs {
    x: i32,
    y: i32,
}

struct AddTool;

impl PortableTool for AddTool {
    const NAME: &'static str = "add";
    type Args = AddArgs;
    type Output = i32;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Add two numbers".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": { "type": "integer" },
                "y": { "type": "integer" }
            },
            "required": ["x", "y"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing to stdout with debug level
    fmt()
        .with_env_filter(EnvFilter::new(
            "rig_custom_agent=debug,custom_agent_telemetry=debug",
        ))
        .init();

    // Initialize the OpenAI client
    let client = openai::Client::from_env()?;

    // Create the custom agent, injecting the custom tool
    let agent = CustomAgentBuilder::new(client.completion_model(openai::GPT_4O))
        .preamble(
            "You are a helpful assistant that adds numbers together. You MUST use the add tool.",
        )
        .tool(AddTool)
        .build();

    println!("Sending prompt to custom agent... Check the tracing output above!");

    // Run the agent chat loop
    let response = agent.chat("What is 1500 plus 243?", 5).await?;

    println!("\nFinal Response:\n{}", response);

    Ok(())
}
