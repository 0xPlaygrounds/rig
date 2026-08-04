use std::convert::Infallible;

use rig::tool::PortableTool;
use serde::Deserialize;

#[derive(Deserialize)]
struct Arguments {
    value: String,
}

struct StablePortableTool;

impl PortableTool for StablePortableTool {
    const NAME: &'static str = "stable_portable_tool";
    type Args = Arguments;
    type Output = String;
    type Error = Infallible;

    fn description(&self) -> String {
        "stable portable tool".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"]
        })
    }

    async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(arguments.value)
    }
}

// The portable contract is reachable through every explicit path, in every
// feature combination (including `--no-default-features`).
fn assert_root_portable<T: rig::tool::PortableTool>() {}
fn assert_core_portable<T: rig::core::tool::PortableTool>() {}
fn assert_namespaced_portable<T: rig::tool::portable::PortableTool>() {}
fn assert_prelude_portable<T: rig::prelude::PortableTool>() {}

fn main() {
    assert_root_portable::<StablePortableTool>();
    assert_core_portable::<StablePortableTool>();
    assert_namespaced_portable::<StablePortableTool>();
    assert_prelude_portable::<StablePortableTool>();

    let portable_dynamic = rig::tool::PortableDynamicTool::new(
        "portable_dynamic",
        "portable dynamic tool",
        serde_json::json!({"type": "object"}),
        |arguments| async move { Ok(rig::tool::ToolOutput::json(arguments)) },
    );
    let _ = &portable_dynamic;

    // Tool discovery is owned data, and its record is reachable in every
    // feature combination too.
    let discovery = rig::embeddings::ToolSchema::new(
        <StablePortableTool as rig::tool::PortableTool>::NAME,
        vec!["stable portable document".to_string()],
    );
    assert_eq!(discovery.name, "stable_portable_tool");
    assert!(discovery.context.is_null());

    // Enabling the agent runtime is purely additive: `rig::tool::Tool` stays
    // the portable, context-free trait it is under `--no-default-features`,
    // and the runtime-defined tool record stays `PortableDynamicTool`.
    #[cfg(feature = "agent")]
    {
        use rig::tool::Tool;

        struct RuntimeTool;

        impl Tool for RuntimeTool {
            const NAME: &'static str = "runtime_tool";
            type Args = Arguments;
            type Output = String;
            type Error = Infallible;

            fn description(&self) -> String {
                "runtime tool".to_string()
            }

            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({"type": "object"})
            }

            async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
                Ok(arguments.value)
            }
        }

        fn assert_runtime_tool<T: rig::tool::Tool>() {}

        // `rig::tool::Tool` is the same trait with or without the runtime, so
        // an impl written against either path satisfies it.
        assert_runtime_tool::<RuntimeTool>();
        assert_runtime_tool::<StablePortableTool>();

        // Runtime-defined tools stay portable records — there is no separate
        // runtime-only dynamic tool type to convert into.
        let _runtime_dynamic: rig::tool::PortableDynamicTool = portable_dynamic;

        // Regression: `#[rig_tool]` expands against the facade and produces a
        // type implementing the same portable `Tool` trait. `ToolContext` was
        // removed, so a context parameter is now a hard compile error (see
        // rig-derive `is_tool_context_parameter`); the supported way to reach
        // host state is to close over it or use `PortableDynamicTool::new`.
        #[cfg(feature = "derive")]
        {
            #[rig::tool_macro(description = "echoes through the facade")]
            fn facade_echo_tool(value: String) -> Result<String, rig::tool::ToolExecutionError> {
                Ok(value)
            }

            assert_runtime_tool::<FacadeEchoTool>();
        }
    }
}
