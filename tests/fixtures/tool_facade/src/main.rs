use std::convert::Infallible;

use rig::tool::{PortableTool, PortableToolEmbedding};
use serde::{Deserialize, Serialize};

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

#[derive(Clone, Deserialize, Serialize)]
struct StableContext {
    label: String,
}

impl PortableToolEmbedding for StablePortableTool {
    type InitError = Infallible;
    type Context = StableContext;
    type State = ();

    fn embedding_docs(&self) -> Vec<String> {
        vec!["stable portable document".to_string()]
    }

    fn context(&self) -> Self::Context {
        StableContext {
            label: "stable".to_string(),
        }
    }

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Self)
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
        |arguments| Box::pin(async move { Ok(rig::tool::ToolOutput::json(arguments)) }),
    );
    let _ = &portable_dynamic;

    // With the classic runtime (default), `rig::tool::Tool` is the *contextual*
    // trait, and a portable tool still registers through the blanket impl.
    #[cfg(feature = "agent")]
    {
        use rig::tool::{Tool, ToolContext};

        struct ContextualTool;

        impl Tool for ContextualTool {
            const NAME: &'static str = "contextual_tool";
            type Args = Arguments;
            type Output = String;
            type Error = Infallible;

            fn description(&self) -> String {
                "contextual tool".to_string()
            }

            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({"type": "object"})
            }

            async fn call(
                &self,
                _context: &mut ToolContext,
                arguments: Self::Args,
            ) -> Result<Self::Output, Self::Error> {
                Ok(arguments.value)
            }
        }

        fn assert_classic_tool<T: rig::tool::Tool>() {}

        // The classic contextual trait accepts a `call(&mut ToolContext, Args)`
        // implementation, and a portable tool is usable as a classic tool via
        // the blanket impl.
        assert_classic_tool::<ContextualTool>();
        assert_classic_tool::<StablePortableTool>();

        let _classic_dynamic = rig::tool::DynamicTool::from_portable(portable_dynamic);
    }
}
