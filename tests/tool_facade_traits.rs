// `.expect(...)` in the runtime builder test below is the idiomatic assertion
// style for integration tests here (see `tests/core.rs`); allow it crate-wide.
#![allow(clippy::expect_used)]

//! Regression tests for the root `rig::tool` facade surface.
//!
//! With the data-oriented migration, `rig::tool::Tool` is the classic *name*
//! for the one portable, context-free contract: an alias for
//! `rig::tool::PortableTool` (the contextual trait, `ToolContext`, and
//! `ToolSet` were removed). Pre-split `impl Tool for X` sites keep compiling
//! once `call` drops the context parameter, and the same trait stays reachable
//! through the explicit portable paths and the prelude.

use rig::tool::{PortableTool, Tool, ToolExecutionError};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct Amount {
    x: i32,
}

/// (1) `rig::tool::Tool` (the classic name) accepts a context-free
/// `call(Args)` — it is the portable contract under its pre-split name.
#[derive(Default)]
struct ClassicNamedAdder;

impl Tool for ClassicNamedAdder {
    const NAME: &'static str = "classic_named_adder";
    type Args = Amount;
    type Output = i32;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "adds one".to_string()
    }

    fn parameters(&self) -> Value {
        json!({ "type": "object" })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + 1)
    }
}

/// (2) `rig::tool::PortableTool` is the same trait under its explicit name.
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
fn classic_tool_name_is_the_portable_contract() {
    // The alias is trait-identical: one impl satisfies both names.
    fn assert_tool<T: Tool>() {}
    fn assert_portable<T: PortableTool>() {}
    assert_tool::<ClassicNamedAdder>();
    assert_portable::<ClassicNamedAdder>();
    assert_tool::<PortableAdder>();
    assert_portable::<PortableAdder>();
}

/// (3) A trait-authored tool erases to a `PortableDynamicTool` record and
/// registers with the data-oriented runtime executor.
#[tokio::test]
async fn portable_tool_registers_with_the_executor() {
    let executor = rig::executor::ToolExecutor::new()
        .register(rig::tool::PortableDynamicTool::from_portable(PortableAdder));
    let names: Vec<String> = executor
        .catalog()
        .definitions
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

/// A single `use rig::prelude::*` provides the whole construction surface:
/// a provider `functions::Config`, the `ProviderConfig` that wraps it, and
/// `AgentBuilder`. No provider type is threaded through the agent. (The classic
/// `extractor` builder is gone too; structured extraction now goes through
/// `rig::extract::*` over a `ProviderConfig`.)
#[test]
fn provider_config_single_import_surface() {
    use rig::prelude::*;

    // Building a config performs no network call, so every surface reachable
    // through the single `rig::prelude::*` import runs to completion offline.
    // A regression in a builder itself (not merely its signature) fails here.
    let cfg = rig::providers::openai::functions::Config::new("gpt-4o").with_api_key("test-key");
    let provider = ProviderConfig::OpenAi(cfg);
    assert_eq!(provider.model(), "gpt-4o");
    let _agent = AgentBuilder::new(provider).build();
}

/// The same surface is reachable through explicit facade paths, without a
/// prelude glob and without depending on `rig-core`: `rig::provider::ProviderConfig`
/// plus `rig::AgentBuilder` (documented in `README.md` / `MIGRATING.md`).
#[test]
fn provider_config_explicit_facade_import_surface() {
    use rig::AgentBuilder;
    use rig::provider::{ProviderConfig, Runtime};

    let cfg = rig::providers::openai::functions::Config::new("gpt-4o").with_api_key("test-key");
    let provider = ProviderConfig::OpenAi(cfg);
    assert_eq!(
        provider
            .descriptor(&Runtime::new())
            .expect("bundled descriptor is always available")
            .name,
        "openai"
    );
    let _agent = AgentBuilder::new(provider).build();
}

/// `use rig::prelude::*` still brings `Tool` into scope under its classic
/// name (pre-split prelude behaviour), now context-free.
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

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok(args.n)
        }
    }

    #[test]
    fn prelude_exposes_classic_tool() {
        fn assert_tool<T: Tool>() {}
        assert_tool::<PreludeTool>();
    }
}
