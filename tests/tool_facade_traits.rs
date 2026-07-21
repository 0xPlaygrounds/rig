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
    let set: ToolSet = ToolSet::builder().static_tool(PortableAdder).build();
    let names: Vec<String> = set
        .get_tool_definitions()
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
