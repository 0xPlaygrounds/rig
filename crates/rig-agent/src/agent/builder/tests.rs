use super::*;
use crate::test_utils::{MockAddTool, MockCompletionModel, MockSubtractTool, MockToolIndex};
use crate::tool::{ToolContext, ToolExecutionError};

#[derive(Clone)]
struct BuilderHook;

impl AgentHook for BuilderHook {}

/// A model without any `Clone` impl must pass through the builder's
/// erasure seam (`AgentBuilder::new` → `ModelHandle::new`). The bound is
/// the test: a regression is a compile error. (The handle-level twin of
/// this probe lives in `rig_core::completion::handle`.)
#[test]
fn builder_accepts_non_clone_model() {
    struct NonCloneModel;

    impl rig_core::completion::CompletionModel for NonCloneModel {
        fn completion(
            &self,
            _request: rig_core::completion::CompletionRequest,
        ) -> impl Future<
            Output = Result<
                rig_core::completion::CompletionResponse,
                rig_core::completion::CompletionError,
            >,
        > + rig_core::wasm_compat::WasmCompatSend {
            std::future::ready(Err(rig_core::completion::CompletionError::ProviderError(
                "compile-time probe".to_string(),
            )))
        }

        fn stream(
            &self,
            _request: rig_core::completion::CompletionRequest,
        ) -> impl Future<
            Output = Result<
                rig_core::streaming::StreamingCompletionResponse,
                rig_core::completion::CompletionError,
            >,
        > + rig_core::wasm_compat::WasmCompatSend {
            std::future::ready(Err(rig_core::completion::CompletionError::ProviderError(
                "compile-time probe".to_string(),
            )))
        }
    }

    let _ = || AgentBuilder::new(NonCloneModel);
}

#[test]
fn hook_can_be_set_after_tool_configuration() {
    let _agent = AgentBuilder::new(MockCompletionModel::text("ok"))
        .tool(MockAddTool)
        .add_hook(BuilderHook)
        .build();
}

struct NamedTool;

impl NamedTool {
    fn new() -> Self {
        Self
    }
}

impl Tool for NamedTool {
    const NAME: &'static str = "registered_named";
    type Error = rig::tool::ToolExecutionError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "uses its canonical name".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        Ok("ok".to_string())
    }
}

#[tokio::test]
async fn typed_tool_builder_paths_advertise_canonical_name() {
    for agent in [
        AgentBuilder::new(MockCompletionModel::text("ok"))
            .tool(NamedTool::new())
            .build(),
        AgentBuilder::new(MockCompletionModel::text("ok"))
            .tool(MockAddTool)
            .tool(NamedTool::new())
            .build(),
    ] {
        let definitions = agent.tool_server_handle.tool_defs(None).await.unwrap();
        assert!(
            definitions
                .iter()
                .any(|definition| definition.name == NamedTool::NAME),
            "the provider definitions dropped the canonical tool name"
        );

        let mut context = ToolContext::new();
        let result = agent
            .tool_server_handle
            .execute(NamedTool::NAME, "{}", &mut context)
            .await;
        assert!(result.is_success());
        assert_eq!(result.output().as_text(), Some("ok"));
    }
}

#[tokio::test]
async fn retrieved_tools_are_exposed_only_for_prompted_retrieval() {
    let retrieval_only = AgentBuilder::new(MockCompletionModel::text("ok"))
        .retrieved_tools(
            1,
            MockToolIndex::new(["add"]),
            ToolSet::from_tools(vec![MockAddTool]),
        )
        .build();
    assert!(
        retrieval_only
            .tool_server_handle
            .tool_defs(None)
            .await
            .unwrap()
            .is_empty()
    );

    let agent = AgentBuilder::new(MockCompletionModel::text("ok"))
        .tool(MockSubtractTool)
        .retrieved_tools(
            1,
            MockToolIndex::new(["add"]),
            ToolSet::from_tools(vec![MockAddTool]),
        )
        .build();

    let always = agent.tool_server_handle.tool_defs(None).await.unwrap();
    assert_eq!(
        always
            .iter()
            .map(|definition| definition.name.as_str())
            .collect::<Vec<_>>(),
        vec!["subtract"]
    );

    let with_retrieval = agent
        .tool_server_handle
        .tool_defs(Some("add two numbers".to_string()))
        .await
        .unwrap();
    assert_eq!(
        with_retrieval
            .iter()
            .map(|definition| definition.name.as_str())
            .collect::<Vec<_>>(),
        vec!["add", "subtract"]
    );
}
