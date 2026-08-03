//! A calculator tool driven through the sans-IO agent protocol.
//!
//! Vertex AI authenticates through Google's OAuth credential chain, so its
//! live handle cannot be expressed as a plain `ProviderConfig` and rig-agent
//! has no Vertex AI arm. Out-of-tree providers like this one drive the public
//! `AgentRun` + `prepare_request` protocol directly: prepare a request, call
//! `functions::complete` yourself, feed the outcome back into the run state
//! machine, and execute the tool calls it surfaces.

use anyhow::Result;
use rig_agent::agent::AgentConfig;
use rig_agent::agent::prepare::{ToolCatalog, prepare_request};
use rig_agent::agent::run::{AgentRun, AgentRunStep};
use rig_core::OneOrMany;
use rig_core::completion::ToolDefinition;
use rig_core::message::{ToolResultContent, UserContent};
use rig_vertexai::{completion::GEMINI_2_5_FLASH_LITE, functions};
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

    // Plain-data config; the live handle resolves ADC credentials.
    let provider = functions::Config::new(GEMINI_2_5_FLASH_LITE);
    let client = functions::client_from_config(&provider)?;

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
                let model_attempt = prepared.model_attempt.clone();
                let response =
                    functions::complete(&client, &provider.model, prepared.request).await?;
                let turn = model_attempt.into_model_turn(
                    response.message_id.clone(),
                    response.choice.clone(),
                    response.usage,
                );
                run.model_response(turn)?;
                run.continue_model_turn()?;
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
