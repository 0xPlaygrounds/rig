//! Migrated from `examples/multi_turn_streaming_gemini.rs`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::client::CompletionClient;
use rig::providers::gemini;
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};
use rig::tool::Tool;
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;

use crate::support::{
    MULTI_TURN_STREAMING_EXPECTED_RESULT, MULTI_TURN_STREAMING_PROMPT,
    assert_mentions_expected_number, assert_nonempty_response,
};

#[tokio::test]
async fn manual_multi_turn_streaming_loop() {
    let add_calls = Arc::new(AtomicUsize::new(0));
    let subtract_calls = Arc::new(AtomicUsize::new(0));
    let multiply_calls = Arc::new(AtomicUsize::new(0));
    let divide_calls = Arc::new(AtomicUsize::new(0));

    super::super::support::with_gemini_cassette(
        "multi_turn_streaming/manual_multi_turn_streaming_loop",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble("You must use tools to answer arithmetic prompts.")
                .tool(Add::new(add_calls.clone()))
                .tool(Subtract::new(subtract_calls.clone()))
                .tool(Multiply::new(multiply_calls.clone()))
                .tool(Divide::new(divide_calls.clone()))
                .build();

            let mut stream = agent
                .stream_prompt(MULTI_TURN_STREAMING_PROMPT)
                .max_turns(8)
                .await;
            let mut response = String::new();
            while let Some(item) = stream.next().await {
                match item.expect("runner-backed streaming should succeed") {
                    MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                        text,
                    )) => response.push_str(&text.text),
                    MultiTurnStreamItem::FinalResponse(final_response) => {
                        if response.is_empty() {
                            response.push_str(&final_response.output);
                        }
                    }
                    _ => {}
                }
            }

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
        },
    )
    .await;
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

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(OperationArgs)).expect("schema should serialize")
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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

    fn description(&self) -> String {
        "Subtract y from x (i.e.: x - y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(OperationArgs)).expect("schema should serialize")
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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

    fn description(&self) -> String {
        "Compute the product of x and y (i.e.: x * y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(OperationArgs)).expect("schema should serialize")
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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

    fn description(&self) -> String {
        "Compute the quotient of x and y.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(OperationArgs)).expect("schema should serialize")
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(args.x / args.y)
    }
}
