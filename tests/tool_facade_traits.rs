// `.expect(...)` in the runtime builder test below is the idiomatic assertion
// style for integration tests here (see `tests/core.rs`); allow it crate-wide.
#![allow(clippy::expect_used)]

//! Regression tests for the root `rig::tool` facade surface (PR #2188).
//!
//! With default features, `rig::tool::Tool` must remain the classic *contextual*
//! trait (so pre-split `use rig::tool::{Tool, ToolContext};` keeps compiling),
//! while the runtime-independent contract stays reachable as
//! `rig::tool::PortableTool`. Portable tools must still register with the classic
//! runtime through the blanket impl. `rig_core::tool::Tool` no longer exists —
//! these tests use `PortableTool`, and nothing in the workspace references the
//! removed alias.

use rig::tool::{PortableTool, Tool, ToolContext, ToolExecutionError, ToolSet};
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

/// (2) `rig::tool::PortableTool` accepts a context-free `call(Args)`.
#[derive(Default)]
struct PortableAdder;

impl PortableTool for PortableAdder {
    const NAME: &'static str = "portable_adder";
    type Args = Amount;
    type Output = i32;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "adds two".to_string()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object" })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + 2)
    }
}

#[test]
fn classic_contextual_tool_impls_facade_tool() {
    fn assert_tool<T: Tool>() {}
    assert_tool::<ContextualAdder>();
}

#[test]
fn portable_tool_impls_facade_portable_tool() {
    fn assert_portable<T: PortableTool>() {}
    assert_portable::<PortableAdder>();
}

/// (3) A portable tool registers with the classic runtime via the blanket
/// `impl<T: PortableTool> Tool for T`, so `static_tool` (which requires the
/// classic `Tool`) accepts it directly.
#[test]
fn portable_tool_registers_with_classic_toolset() {
    let mut set = ToolSet::default();
    set.add_tool(PortableAdder);
    let names: Vec<String> = set
        .tool_definitions()
        .into_iter()
        .map(|definition| definition.name)
        .collect();
    assert!(names.iter().any(|name| name == "portable_adder"));
}

/// The portable contract is also reachable through the always-available
/// explicit paths, regardless of the classic re-exports.
#[test]
fn portable_contract_paths_resolve() {
    fn assert_portable<T: rig_core::tool::PortableTool>() {}
    assert_portable::<PortableAdder>();

    fn assert_portable_facade<T: rig::tool::portable::PortableTool>() {}
    assert_portable_facade::<PortableAdder>();
}

/// A single `use rig::prelude::*` provides `bound`, `completion`, `agent` and
/// `extractor` — the whole construction surface from one import.
#[test]
fn completion_client_single_import_surface() {
    use rig::prelude::*;

    #[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
    struct Extracted {
        value: String,
    }

    // Binding a provider config to the bundled transport performs no network
    // call, so all four spellings reachable through the single
    // `rig::prelude::*` import run to completion offline. A regression in any
    // of them fails here, not merely a signature change.
    let bound = rig::providers::openai::wire::OpenAI::with_key(
        &rig::providers::openai::wire::OPENAI,
        "test-key",
    )
    .bound()
    .expect("the bundled transport builds");
    let _model = bound.completion("gpt-4o");
    let _agent = bound.agent("gpt-4o").build();
    let _extractor = bound.extractor::<Extracted>("gpt-4o").build();
}

/// The same surface is reachable through explicit imports, without the
/// prelude glob: `Bind`/`DefaultTransport` for construction and
/// `AgentProviderExt` for the agent sugar.
#[test]
fn completion_provider_explicit_facade_import_surface() {
    use rig::client::AgentProviderExt;
    use rig_reqwest::client::DefaultTransport;

    #[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
    struct Extracted {
        value: String,
    }

    let bound = rig::providers::openai::wire::OpenAI::with_key(
        &rig::providers::openai::wire::OPENAI,
        "test-key",
    )
    .bound() // DefaultTransport
    .expect("the bundled transport builds");
    let _model = bound.completion("gpt-4o"); // Bound::completion
    let _agent = bound.agent("gpt-4o").build(); // AgentProviderExt
    let _extractor = bound.extractor::<Extracted>("gpt-4o").build(); // AgentProviderExt
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
