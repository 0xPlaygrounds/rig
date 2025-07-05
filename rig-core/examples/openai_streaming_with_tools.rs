use anyhow::Result;
use futures::{StreamExt, stream};
use rig::OneOrMany;
use rig::agent::Agent;
use rig::completion::{CompletionError, CompletionModel};
use rig::message::{AssistantContent, UserContent};
use rig::prelude::*;
use rig::streaming::{StreamingChat, stream_to_stdout};
use rig::tool::ToolSetError;
use rig::{
    completion::{Message, ToolDefinition},
    providers,
    streaming::StreamingPrompt,
    tool::Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
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
                },
                "required": ["x", "y"],
            }),
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
        serde_json::from_value(json!({
            "name": "subtract",
            "description": "Subtract y from x (i.e.: x - y)",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The number to subtract from"
                    },
                    "y": {
                        "type": "number",
                        "description": "The number to subtract"
                    }
                },
                "required": ["x", "y"],
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x - args.y;
        Ok(result)
    }
}

/// This is a (temporary) helper function to call tools when given a choice from a provider. This is
/// designed to be used with streaming, where tool calling needs to be invoked by the client atm.
async fn tool_call_helper<M: CompletionModel>(
    choice: OneOrMany<AssistantContent>,
    agent: &Agent<M>,
) -> Result<OneOrMany<UserContent>, CompletionError> {
    let (tool_calls, _): (Vec<_>, Vec<_>) = choice
        .iter()
        .partition(|choice| matches!(choice, AssistantContent::ToolCall(_)));
    let tool_content = stream::iter(tool_calls)
        .then(async |choice| {
            if let AssistantContent::ToolCall(tool_call) = choice {
                let output = agent
                    .tools
                    .call(
                        &tool_call.function.name,
                        tool_call.function.arguments.to_string(),
                    )
                    .await?;
                if let Some(call_id) = tool_call.call_id.clone() {
                    Ok(UserContent::tool_result_with_call_id(
                        tool_call.id.clone(),
                        call_id,
                        OneOrMany::one(output.into()),
                    ))
                } else {
                    Ok(UserContent::tool_result(
                        tool_call.id.clone(),
                        OneOrMany::one(output.into()),
                    ))
                }
            } else {
                unreachable!("This should never happen as we already filtered for `ToolCall`")
            }
        })
        .collect::<Vec<Result<UserContent, ToolSetError>>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| CompletionError::RequestError(Box::new(e)))?;
    Ok(OneOrMany::many(tool_content).expect("Should always have at least one tool call"))
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();

    // Create agent with a single context prompt and two tools
    let calculator_agent = providers::openai::Client::from_env()
        .agent(providers::openai::GPT_4O)
        .preamble(
            "You are a calculator here to help the user perform arithmetic
            operations. Use the tools provided to answer the user's question.
            make your answer long, so we can test the streaming functionality,
            like 20 words",
        )
        .max_tokens(1024)
        .tool(Adder)
        .tool(Subtract)
        .build();

    println!("Calculate 2 - 5");

    let prompt = "Calculate 2 - 5";
    let mut chat_history = vec![Message::user(prompt)];
    let mut stream = calculator_agent.stream_prompt(prompt).await?;
    stream_to_stdout(&calculator_agent, &mut stream).await?;

    if let Some(response) = stream.response {
        println!("Usage: {:?}", response.usage)
    };

    println!("Message: {:?}", stream.choice);
    chat_history.push(stream.choice.clone().into());
    let tool_results = tool_call_helper(stream.choice, &calculator_agent).await?;

    let mut stream = calculator_agent
        .stream_chat(tool_results, chat_history)
        .await?;
    stream_to_stdout(&calculator_agent, &mut stream).await?;

    if let Some(response) = stream.response {
        println!("Usage: {:?}", response.usage)
    };

    println!("Message: {:?}", stream.choice);

    Ok(())
}
