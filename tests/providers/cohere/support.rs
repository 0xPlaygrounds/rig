use futures::FutureExt;
use rig::providers::cohere;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use crate::support::{MathError, OperationArgs};

const COHERE_BASE_URL: &str = "https://api.cohere.ai";

async fn cohere_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, cohere::Client) {
    let cassette = ProviderCassette::start("cohere", spec, COHERE_BASE_URL).await;
    let client = cohere::Client::builder()
        .api_key(cassette.api_key("COHERE_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("Cohere cassette client should build");

    (cassette, client)
}

pub(super) async fn with_cohere_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(cohere::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = cohere_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

// The shared `Adder`/`Subtract` fixtures advertise `"type": "number"` while
// deserializing into `i32`; Cohere follows that literally and emits
// `{"x":2.0,"y":5.0}`, which then fails to parse. These copies declare
// `"type": "integer"` instead.

#[derive(Deserialize, Serialize)]
pub(super) struct IntegerAdder;

impl Tool for IntegerAdder {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "The first number to add"},
                "y": {"type": "integer", "description": "The second number to add"}
            },
            "required": ["x", "y"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[derive(Deserialize, Serialize)]
pub(super) struct IntegerSubtract;

impl Tool for IntegerSubtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Subtract y from x (i.e.: x - y)".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "The number to subtract from"},
                "y": {"type": "integer", "description": "The number to subtract"}
            },
            "required": ["x", "y"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}
