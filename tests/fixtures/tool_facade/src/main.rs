use std::convert::Infallible;

use rig::tool::{Tool, ToolEmbedding};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Arguments {
    value: String,
}

struct StablePortableTool;

impl Tool for StablePortableTool {
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

impl ToolEmbedding for StablePortableTool {
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

struct PreludePortableTool;

impl rig::prelude::Tool for PreludePortableTool {
    const NAME: &'static str = "prelude_portable_tool";
    type Args = Arguments;
    type Output = String;
    type Error = Infallible;

    fn description(&self) -> String {
        "prelude portable tool".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(arguments.value)
    }
}

fn assert_root_tool<T: rig::tool::Tool>() {}
fn assert_prelude_tool<T: rig::prelude::Tool>() {}

fn main() {
    assert_root_tool::<StablePortableTool>();
    assert_prelude_tool::<StablePortableTool>();
    assert_root_tool::<PreludePortableTool>();

    let portable_dynamic = rig::tool::DynamicTool::new(
        "portable_dynamic",
        "portable dynamic tool",
        serde_json::json!({"type": "object"}),
        |arguments| Box::pin(async move { Ok(rig::tool::ToolOutput::json(arguments)) }),
    );

    #[cfg(feature = "agent")]
    {
        fn assert_classic_tool<T: rig::agent::tool::Tool>() {}
        fn assert_classic_embedding<T: rig::agent::tool::ToolEmbedding>() {}

        assert_classic_tool::<StablePortableTool>();
        assert_classic_embedding::<StablePortableTool>();
        let _classic_dynamic =
            rig::agent::tool::DynamicTool::from_portable(portable_dynamic.clone());
    }

    let _ = portable_dynamic;
}
