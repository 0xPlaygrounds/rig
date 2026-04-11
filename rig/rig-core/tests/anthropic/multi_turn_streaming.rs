//! Migrated from `examples/multi_turn_streaming.rs`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::ToolDefinition;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;

use crate::support::{
    MULTI_TURN_STREAMING_EXPECTED_RESULT, MULTI_TURN_STREAMING_PROMPT,
    assert_mentions_expected_number, collect_stream_observation,
};

#[derive(Deserialize, JsonSchema)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("math error")]
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
            description: "Compute the product of x and y (i.e.: x * y).".to_string(),
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

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn multi_turn_streaming_tools() {
    let add_calls = Arc::new(AtomicUsize::new(0));
    let subtract_calls = Arc::new(AtomicUsize::new(0));
    let multiply_calls = Arc::new(AtomicUsize::new(0));
    let divide_calls = Arc::new(AtomicUsize::new(0));

    let client = anthropic::Client::from_env();
    let agent = client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble("You must use tools for arithmetic.")
        .tool(Add::new(add_calls.clone()))
        .tool(Subtract::new(subtract_calls.clone()))
        .tool(Multiply::new(multiply_calls.clone()))
        .tool(Divide::new(divide_calls.clone()))
        .build();

    let mut stream = agent
        .stream_prompt(MULTI_TURN_STREAMING_PROMPT)
        .multi_turn(10)
        .await;
    let observation = collect_stream_observation(&mut stream).await;

    assert!(
        observation.errors.is_empty(),
        "stream should not emit errors: {:?}",
        observation.errors
    );
    assert!(
        observation.got_final_response,
        "stream should emit a final response"
    );
    assert_eq!(
        observation.final_response_text.as_deref(),
        Some(observation.final_turn_text.as_str()),
        "FinalResponse.response() should match the final turn's streamed text"
    );
    assert!(
        observation.tool_results >= 4,
        "expected at least 4 tool-result events, got {}",
        observation.tool_results
    );
    for tool_name in ["add", "subtract", "multiply", "divide"] {
        assert!(
            observation.tool_calls.iter().any(|name| name == tool_name),
            "expected tool call for {tool_name}, saw {:?}",
            observation.tool_calls
        );
    }

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

    let response = observation
        .final_response_text
        .expect("stream should produce a final response string");
    assert_mentions_expected_number(&response, MULTI_TURN_STREAMING_EXPECTED_RESULT);
}
