//! Migrated from `examples/multi_turn_streaming_gemini.rs`.

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::{Stream, StreamExt};
use rig::OneOrMany;
use rig::agent::Agent;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{self, CompletionError, CompletionModel, PromptError, ToolDefinition};
use rig::message::{AssistantContent, Message, Text, ToolResultContent, UserContent};
use rig::providers::gemini;
use rig::streaming::{StreamedAssistantContent, StreamingCompletion};
use rig::tool::{Tool, ToolError, ToolSetError};
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use thiserror::Error;

use crate::support::{
    MULTI_TURN_STREAMING_EXPECTED_RESULT, MULTI_TURN_STREAMING_PROMPT,
    assert_mentions_expected_number, assert_nonempty_response,
};

#[derive(Debug, Error)]
enum StreamingError {
    #[error("CompletionError: {0}")]
    Completion(#[from] CompletionError),
    #[error("PromptError: {0}")]
    Prompt(#[from] Box<PromptError>),
    #[error("ToolSetError: {0}")]
    Tool(#[from] ToolSetError),
}

type StreamingResult = Pin<Box<dyn Stream<Item = Result<Text, StreamingError>> + Send>>;

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn manual_multi_turn_streaming_loop() {
    let add_calls = Arc::new(AtomicUsize::new(0));
    let subtract_calls = Arc::new(AtomicUsize::new(0));
    let multiply_calls = Arc::new(AtomicUsize::new(0));
    let divide_calls = Arc::new(AtomicUsize::new(0));

    let client = gemini::Client::from_env();
    let agent = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble("You must use tools to answer arithmetic prompts.")
        .tool(Add::new(add_calls.clone()))
        .tool(Subtract::new(subtract_calls.clone()))
        .tool(Multiply::new(multiply_calls.clone()))
        .tool(Divide::new(divide_calls.clone()))
        .build();

    let mut stream = multi_turn_prompt(agent, MULTI_TURN_STREAMING_PROMPT, Vec::new()).await;
    let response = collect_text(&mut stream)
        .await
        .expect("manual multi-turn streaming should succeed");

    assert_nonempty_response(&response);
    assert!(
        response.trim().len() >= 30,
        "expected a substantial streamed response, got {:?}",
        response
    );
    assert_mentions_expected_number(&response, MULTI_TURN_STREAMING_EXPECTED_RESULT);
    assert!(
        add_calls.load(Ordering::SeqCst) >= 1,
        "add should be called"
    );
    assert!(
        subtract_calls.load(Ordering::SeqCst) >= 1,
        "subtract should be called"
    );
    assert!(
        multiply_calls.load(Ordering::SeqCst) >= 1,
        "multiply should be called"
    );
    assert!(
        divide_calls.load(Ordering::SeqCst) >= 1,
        "divide should be called"
    );
}

async fn multi_turn_prompt<M>(
    agent: Agent<M>,
    prompt: impl Into<Message> + Send,
    mut chat_history: Vec<completion::Message>,
) -> StreamingResult
where
    M: CompletionModel + 'static,
    M::StreamingResponse: Send,
{
    let prompt: Message = prompt.into();

    Box::pin(async_stream::stream! {
        let mut current_prompt = prompt;
        let mut did_call_tool = false;

        'outer: loop {
            let mut stream = agent
                .stream_completion(current_prompt.clone(), &chat_history)
                .await?
                .stream()
                .await?;

            chat_history.push(current_prompt.clone());
            let mut tool_calls = vec![];
            let mut tool_results = vec![];

            while let Some(content) = stream.next().await {
                match content {
                    Ok(StreamedAssistantContent::Text(text)) => {
                        yield Ok(Text { text: text.text });
                        did_call_tool = false;
                    }
                    Ok(StreamedAssistantContent::ToolCall { tool_call, .. }) => {
                        let tool_result = agent
                            .tool_server_handle
                            .call_tool(
                                &tool_call.function.name,
                                &tool_call.function.arguments.to_string(),
                            )
                            .await
                            .map_err(|error| {
                                StreamingError::Tool(ToolSetError::ToolCallError(
                                    ToolError::ToolCallError(error.into()),
                                ))
                            })?;

                        tool_calls.push(AssistantContent::ToolCall(tool_call.clone()));
                        tool_results.push((tool_call.id, tool_call.call_id, tool_result));
                        did_call_tool = true;
                    }
                    Ok(StreamedAssistantContent::Reasoning(reasoning)) => {
                        let rendered = reasoning.display_text();
                        if !rendered.is_empty() {
                            yield Ok(Text { text: rendered });
                        }
                        did_call_tool = false;
                    }
                    Ok(_) => {}
                    Err(error) => {
                        yield Err(error.into());
                        break 'outer;
                    }
                }
            }

            if !tool_calls.is_empty() {
                chat_history.push(Message::Assistant {
                    id: None,
                    content: OneOrMany::many(tool_calls).expect("tool calls should be non-empty"),
                });
            }

            for (id, call_id, tool_result) in tool_results {
                let content = if let Some(call_id) = call_id {
                    UserContent::tool_result_with_call_id(
                        id,
                        call_id,
                        OneOrMany::one(ToolResultContent::text(tool_result)),
                    )
                } else {
                    UserContent::tool_result(
                        id,
                        OneOrMany::one(ToolResultContent::text(tool_result)),
                    )
                };
                chat_history.push(Message::User {
                    content: OneOrMany::one(content),
                });
            }

            current_prompt = match chat_history.pop() {
                Some(prompt) => prompt,
                None => unreachable!("chat history should not be empty"),
            };

            if !did_call_tool {
                break;
            }
        }
    })
}

async fn collect_text(stream: &mut StreamingResult) -> Result<String, StreamingError> {
    let mut text = String::new();
    while let Some(content) = stream.next().await {
        text.push_str(&content?.text);
    }
    Ok(text)
}

#[derive(Deserialize, JsonSchema)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

struct Add {
    call_count: Arc<AtomicUsize>,
}

impl Add {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

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
                .expect("schema should serialize"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(args.x + args.y)
    }
}

struct Subtract {
    call_count: Arc<AtomicUsize>,
}

impl Subtract {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

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
                .expect("schema should serialize"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(args.x - args.y)
    }
}

struct Multiply {
    call_count: Arc<AtomicUsize>,
}

impl Multiply {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

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
                .expect("schema should serialize"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(args.x * args.y)
    }
}

struct Divide {
    call_count: Arc<AtomicUsize>,
}

impl Divide {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for Divide {
    const NAME: &'static str = "divide";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "divide".to_string(),
            description: "Compute the quotient of x and y.".to_string(),
            parameters: serde_json::to_value(schema_for!(OperationArgs))
                .expect("schema should serialize"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(args.x / args.y)
    }
}
