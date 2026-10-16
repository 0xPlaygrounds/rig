use std::convert::Infallible;

use rig::tool::{Tool, ToolContext, ToolEmbedding};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Arguments {
    value: String,
}

struct StableTool;

impl Tool for StableTool {
    const NAME: &'static str = "stable_tool";
    type Args = Arguments;
    type Output = String;
    type Error = Infallible;

    fn description(&self) -> String {
        "stable tool".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"]
        })
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        arguments: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(arguments.value)
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct StableContext {
    label: String,
}

impl rig::tool::ContextValue for StableContext {
    const KEY: &'static str = "fixture.stable-context.v1";
}

impl ToolEmbedding for StableTool {
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

// The tool contract is reachable through every explicit path, in every
// feature combination (including `--no-default-features`).
fn assert_root_tool<T: rig::tool::Tool>() {}
fn assert_prelude_tool<T: rig::prelude::Tool>() {}

#[cfg(feature = "derive")]
use rig::tool::ContextValue;

#[cfg(feature = "derive")]
#[derive(Serialize, Deserialize, ContextValue)]
#[context(key = "fixture.derived-context.v1")]
struct DerivedContext(String);

fn main() {
    #[cfg(feature = "derive")]
    {
        let mut derived = rig::tool::ToolContext::new();
        derived.insert(DerivedContext("derived".into())).unwrap();
        assert_eq!(derived.require::<DerivedContext>().unwrap().0, "derived");
    }

    let mut context = rig::tool::ToolContext::new();
    context
        .insert(StableContext {
            label: "facade".into(),
        })
        .unwrap();
    assert_eq!(context.require::<StableContext>().unwrap().label, "facade");
    assert_root_tool::<StableTool>();
    assert_prelude_tool::<StableTool>();

    let dynamic = rig::tool::DynamicTool::new(
        rig::message::ToolName::new("dynamic").unwrap(),
        "context-free dynamic tool",
        serde_json::json!({"type": "object"}),
        |arguments| Box::pin(async move { Ok(rig::tool::ToolOutput::json(arguments)) }),
    );
    let _ = &dynamic;

    #[cfg(feature = "agent")]
    {
        fn assert_classic_tool<T: rig::tool::Tool>() {}
        assert_classic_tool::<StableTool>();

        // Regression: `#[rig_tool]` must auto-detect the *fully-qualified* facade
        // path `rig::tool::ToolContext` as runtime context (not a model
        // argument), matching the documented `use rig::tool::{Tool, ToolContext};`
        // compatibility promise. If detection regresses, the context param is
        // treated as a model argument and this fixture fails to compile.
        // See rig-derive `is_tool_context_type`.
        #[cfg(feature = "derive")]
        {
            #[rig::rig_tool(description = "echoes using fully-qualified facade context")]
            fn facade_qualified_context_tool(
                context: &mut rig::tool::ToolContext,
                value: String,
            ) -> Result<String, rig::tool::ToolExecutionError> {
                let _ = context;
                Ok(value)
            }

            // The context param is stripped from the model-visible schema, so the
            // generated tool implements the classic contextual `Tool` trait.
            assert_classic_tool::<FacadeQualifiedContextTool>();
        }
    }
}
