use std::sync::{Arc, Mutex, MutexGuard};

use rig_core::{
    OneOrMany,
    message::{ImageMediaType, ToolResultContent},
    tool::{
        PortableDynamicTool, PortableTool, PortableToolEmbedding, ToolExecutionError, ToolOutput,
    },
};
use serde::{Deserialize, Serialize};

/// Scripted portable completion model shared by runtime conformance tests.
pub type ScriptedCompletionModel = rig_core::test_utils::MockCompletionModel;
/// One scripted blocking completion turn.
pub type ScriptedCompletionTurn = rig_core::test_utils::MockTurn;
/// Provider-typed raw response used by scripted blocking and streaming calls.
pub type ScriptedRawResponse = rig_core::test_utils::MockResponse;
/// One scripted provider streaming event.
pub type ScriptedStreamEvent = rig_core::test_utils::MockStreamEvent;

/// Stable image payload used by portable rich-output parity tests.
pub const PORTABLE_FIXTURE_IMAGE: &str = "cG9ydGFibGUtZml4dHVyZQ==";

/// Arguments accepted by [`PortableEmbeddingFixture`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PortableEmbeddingArgs {
    /// Value included in the rich output.
    pub value: String,
    /// Whether to return a classified rich error instead.
    #[serde(default)]
    pub fail: bool,
}

/// Serializable reconstruction context for [`PortableEmbeddingFixture`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PortableEmbeddingContext {
    /// Prefix included in outputs and embedding documents.
    pub prefix: String,
}

/// Typed source error emitted by [`PortableEmbeddingFixture`].
#[derive(Debug, thiserror::Error)]
#[error("portable fixture failure")]
pub struct PortableFixtureError;

/// Build the exact rich output used to compare classic and Bevy adaptation.
pub fn portable_fixture_output(label: impl Into<String>) -> ToolOutput {
    let mut content = OneOrMany::one(ToolResultContent::json(
        serde_json::json!({"label": label.into()}),
    ));
    content.push(ToolResultContent::image_base64(
        PORTABLE_FIXTURE_IMAGE,
        Some(ImageMediaType::PNG),
        None,
    ));
    ToolOutput::content(content)
}

/// Build the dynamic portable tool used by both runtime adapter suites.
pub fn portable_dynamic_fixture() -> PortableDynamicTool {
    PortableDynamicTool::new(
        "portable_runtime_name",
        "portable dynamic definition",
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "fail": {"type": "boolean"}
            },
            "required": ["value"]
        }),
        |arguments| {
            Box::pin(async move {
                if arguments
                    .get("fail")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or_default()
                {
                    Err(ToolExecutionError::provider("portable dynamic failure")
                        .with_code("portable_dynamic_fixture")
                        .with_model_output(portable_fixture_output("portable dynamic failure")))
                } else {
                    Ok(portable_fixture_output(format!(
                        "dynamic:{}",
                        arguments
                            .get("value")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or_default()
                    )))
                }
            })
        },
    )
}

/// Portable embedding tool used by both runtime adapter suites.
#[derive(Clone)]
pub struct PortableEmbeddingFixture {
    context: PortableEmbeddingContext,
}

impl PortableEmbeddingFixture {
    /// Construct the fixture with stable serialized context.
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            context: PortableEmbeddingContext {
                prefix: prefix.into(),
            },
        }
    }
}

impl PortableTool for PortableEmbeddingFixture {
    const NAME: &'static str = "portable_embedding_fixture";
    type Args = PortableEmbeddingArgs;
    type Output = ToolOutput;
    type Error = PortableFixtureError;

    fn description(&self) -> String {
        format!("{} portable embedding fixture", self.context.prefix)
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "fail": {"type": "boolean"}
            },
            "required": ["value"]
        })
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::provider(error.to_string())
            .with_code("portable_fixture")
            .with_model_output(portable_fixture_output("portable failure"))
            .with_source(error)
    }

    async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
        if arguments.fail {
            Err(PortableFixtureError)
        } else {
            Ok(portable_fixture_output(format!(
                "{}:{}",
                self.context.prefix, arguments.value
            )))
        }
    }
}

impl PortableToolEmbedding for PortableEmbeddingFixture {
    type InitError = std::convert::Infallible;
    type Context = PortableEmbeddingContext;
    type State = ();

    fn embedding_docs(&self) -> Vec<String> {
        vec![format!(
            "{} portable embedding document",
            self.context.prefix
        )]
    }

    fn context(&self) -> Self::Context {
        self.context.clone()
    }

    fn init(_state: Self::State, context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Self { context })
    }
}

/// Create a one-turn scripted text model.
pub fn scripted_text_model(text: impl Into<String>) -> ScriptedCompletionModel {
    ScriptedCompletionModel::text(text)
}

/// Arguments accepted by [`CountingPortableTool`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CountingPortableToolArgs {
    /// Value recorded by the tool.
    pub value: String,
}

/// Context-free tool fixture that records owned invocations.
#[derive(Clone, Default)]
pub struct CountingPortableTool {
    calls: Arc<Mutex<Vec<String>>>,
}

impl CountingPortableTool {
    /// Return the calls observed so far in execution order.
    pub fn calls(&self) -> Vec<String> {
        self.calls_guard().clone()
    }

    fn calls_guard(&self) -> MutexGuard<'_, Vec<String>> {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl PortableTool for CountingPortableTool {
    const NAME: &'static str = "counting_portable_tool";
    type Args = CountingPortableToolArgs;
    type Output = serde_json::Value;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Record one owned tool invocation".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"]
        })
    }

    async fn call(&self, arguments: Self::Args) -> Result<Self::Output, Self::Error> {
        self.calls_guard().push(arguments.value.clone());
        Ok(serde_json::json!({"recorded": arguments.value}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn portable_fixture_records_owned_invocations() {
        let tool = CountingPortableTool::default();
        let output = tool
            .call(CountingPortableToolArgs {
                value: "first".to_string(),
            })
            .await;

        assert_eq!(output.ok(), Some(serde_json::json!({"recorded": "first"})));
        assert_eq!(tool.calls(), ["first"]);
    }
}
