// `.expect(...)` in the runtime builder test below is the idiomatic assertion
// style for integration tests here (see `tests/core.rs`); allow it crate-wide.
#![allow(clippy::expect_used)]

//! Regression tests for the root `rig::tool` facade surface (PR #2188):
//! `rig::tool::Tool` is the one contextual tool trait.

use rig::tool::{Tool, ToolContext, ToolExecutionError};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct Amount {
    x: i32,
}

/// (1) `rig::tool::Tool` accepts a contextual `call(&mut ToolContext, Args)`.
#[derive(Default)]
struct ContextualAdder;

impl Tool for ContextualAdder {
    const NAME: &'static str = "contextual_adder";
    type Args = Amount;
    type Output = i32;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "adds one".to_string()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object" })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x + 1)
    }
}

#[test]
fn classic_contextual_tool_impls_facade_tool() {
    fn assert_tool<T: Tool>() {}
    assert_tool::<ContextualAdder>();
}

/// A single `use rig::prelude::*` provides `Model` and `AgentBuilder`: a
/// provider client's model, and an agent over it.
#[test]
fn completion_client_single_import_surface() {
    use rig::prelude::*;

    // Building a client and its models performs no network call, so every
    // spelling below runs to completion offline. A regression in any of them
    // fails here, not merely a signature change.
    let openai = rig::providers::openai::OpenAI::new("test-key");
    let _model: Model<_> = openai.completion("gpt-4o");
    let _agent = AgentBuilder::new(openai.completion("gpt-4o")).build();
}

/// The same surface is reachable through explicit imports, without the
/// prelude glob, including the typed extractor.
#[test]
fn completion_provider_explicit_facade_import_surface() {
    use rig::agent::AgentBuilder;
    use rig::extractor::ExtractorBuilder;

    #[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
    struct Extracted {
        value: String,
    }

    let openai = rig::providers::openai::OpenAI::new("test-key");
    let _model = openai.completion("gpt-4o");
    let _agent = AgentBuilder::new(openai.completion("gpt-4o")).build();
    let _extractor = ExtractorBuilder::<Extracted>::new(openai.completion("gpt-4o")).build();
}

/// `use rig::prelude::*` still brings the classic contextual `Tool` and
/// `ToolContext` into scope (pre-split prelude behaviour).
mod prelude_regression {
    use rig::prelude::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Args {
        n: i32,
    }

    struct PreludeTool;

    impl Tool for PreludeTool {
        const NAME: &'static str = "prelude_tool";
        type Args = Args;
        type Output = i32;
        type Error = rig::tool::ToolExecutionError;

        fn description(&self) -> String {
            "prelude".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({ "type": "object" })
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            Ok(args.n)
        }
    }

    #[test]
    fn prelude_exposes_classic_tool() {
        fn assert_tool<T: Tool>() {}
        assert_tool::<PreludeTool>();
    }
}
