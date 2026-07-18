//! Migrated from `examples/multi_turn_streaming_gemini.rs`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::prelude::AgentClientExt;
use rig::providers::gemini;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;

use crate::support::{
    MULTI_TURN_STREAMING_EXPECTED_RESULT, MULTI_TURN_STREAMING_PROMPT,
    assert_mentions_expected_number, assert_nonempty_response,
};

#[tokio::test]
async fn runner_driven_multi_turn_streaming_loop() {
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
                .max_turns(10)
                .await;
            let mut response = None;
            while let Some(item) = stream.next().await {
                if let MultiTurnStreamItem::FinalResponse(final_response) =
                    item.expect("runner-driven multi-turn streaming should succeed")
                {
                    response = Some(final_response.output);
                }
            }
            let response = response.expect("stream should emit a final response");

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

    fn description(&self) -> String {
        "Subtract y from x (i.e.: x - y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(OperationArgs)).expect("schema should serialize")
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

    fn description(&self) -> String {
        "Compute the product of x and y (i.e.: x * y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(OperationArgs)).expect("schema should serialize")
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

    fn description(&self) -> String {
        "Compute the quotient of x and y.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::to_value(schema_for!(OperationArgs)).expect("schema should serialize")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(args.x / args.y)
    }
}
