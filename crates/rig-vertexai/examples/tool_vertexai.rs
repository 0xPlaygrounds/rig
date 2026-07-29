//! A calculator tool driven through the sans-IO agent protocol.
//!
//! Vertex AI authenticates through Google's OAuth credential chain, so its
//! model cannot be expressed as a plain `ProviderConfig`. Out-of-tree
//! providers like this one drive the public `AgentRun` + `prepare_request`
//! protocol directly: prepare a request, call the model, feed the outcome
//! back into the run state machine, and execute the tool calls it surfaces.

use anyhow::Result;
use rig_agent::agent::AgentConfig;
use rig_agent::agent::prepare::{ToolCatalog, prepare_request};
use rig_agent::agent::run::{AgentRun, AgentRunStep, ModelTurn};
use rig_core::OneOrMany;
use rig_core::client::completion::CompletionClient;
use rig_core::completion::{CompletionModel as CompletionModelTrait, ToolDefinition};
use rig_core::message::{ToolResultContent, UserContent};
use rig_vertexai::{Client, completion::GEMINI_2_5_FLASH_LITE};
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize, JsonSchema)]
struct OperationArgs {
    x: i32,
    y: i32,
}

fn add(args: &OperationArgs) -> i32 {
    println!("[tool-call] Adding {} and {}", args.x, args.y);
    args.x + args.y
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().with_target(false).init();

    // Create Vertex AI client using implicit credentials
    let client = Client::from_env()?;
    let model = client.completion_model(GEMINI_2_5_FLASH_LITE);

    let mut config = AgentConfig::new();
    config.max_tokens = Some(1024);
    let catalog = ToolCatalog::new(vec![ToolDefinition {
        name: "add".to_string(),
        description: "Add x and y together".to_string(),
        parameters: json!(schema_for!(OperationArgs)),
    }]);

    println!("Calculate 15 + 27");
    let mut run = AgentRun::new("Calculate 15 + 27").max_turns(3);

    loop {
        match run.next_step()? {
            AgentRunStep::CallModel {
                prompt, history, ..
            } => {
                let prepared = prepare_request(
                    &config,
                    &catalog,
                    false,
                    prompt,
                    &history,
                    run.output_tool_name(),
                    None,
                )?;
                let response = model.completion(prepared.request).await?;
                let turn = ModelTurn::new(
                    response.message_id.clone(),
                    response.choice.clone(),
                    response.usage,
                    prepared.executable_tool_names,
                    prepared.allowed_tool_names,
                );
                run.model_response(turn)?;
            }
            AgentRunStep::CallTools { calls } => {
                let mut results = Vec::new();
                for call in &calls {
                    let args: OperationArgs =
                        serde_json::from_value(call.tool_call.function.arguments.clone())?;
                    let sum = add(&args);
                    results.push(UserContent::tool_result(
                        call.tool_call.id.clone(),
                        OneOrMany::one(ToolResultContent::text(sum.to_string())),
                    ));
                }
                run.tool_results(results)?;
            }
            AgentRunStep::Done(response) => {
                println!("Vertex AI Calculator Agent: {}", response.output);
                break;
            }
        }
    }

    Ok(())
}
