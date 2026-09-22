use futures::FutureExt;
use rig::driver::Bound;
use rig::http_client::BoxedHttpClient;
use rig::prelude::*;
use rig::providers::cohere::wire::Cohere;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use crate::support::{MathError, OperationArgs};

const COHERE_BASE_URL: &str = "https://api.cohere.ai";

/// The Cohere config bound to the bundled transport — what a cassette test
/// builds its models from, now that a model is a bound wire.
pub(super) type BoundCohere = Bound<Cohere, BoxedHttpClient>;

async fn cohere_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, BoundCohere) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "cohere",
        spec,
        COHERE_BASE_URL,
    )
    .await;
    let cohere = Cohere::new(cassette.api_key("COHERE_API_KEY"))
        .with_base_url(cassette.base_url())
        .bound()
        .expect("Cohere cassette transport should build");

    (cassette, cohere)
}

pub(super) async fn with_cohere_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(BoundCohere) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, client) = cohere_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    crate::cassettes::checkpoint_attempt(&cassette, "cohere", spec.scenario()).await;
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

/// Cassette wrapper for the cohere prompt-caching matrix
/// (`crates/rig-cassette/fixtures/cassettes/cohere/prompt_caching/`).
///
/// Delegates to [`with_cohere_cassette`] — the behavior is identical, and deliberately shared
/// so the two cannot drift apart when the base wrapper gains policy. What the
/// separate name buys is a per-suite entry in the cassette-safety registry, so
/// the cache fixtures are auditable as one concern's evidence.
pub(super) async fn with_cohere_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(BoundCohere) -> Fut,
    Fut: Future<Output = ()>,
{
    with_cohere_cassette(spec, test_body).await;
}
