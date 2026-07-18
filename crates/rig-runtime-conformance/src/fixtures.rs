use std::sync::{Arc, Mutex, MutexGuard};

use rig_core::tool::{PortableTool, ToolExecutionError};
use serde::{Deserialize, Serialize};

/// Scripted portable completion model shared by runtime conformance tests.
pub type ScriptedCompletionModel = rig_core::test_utils::MockCompletionModel;
/// One scripted blocking completion turn.
pub type ScriptedCompletionTurn = rig_core::test_utils::MockTurn;
/// Provider-typed raw response used by scripted blocking and streaming calls.
pub type ScriptedRawResponse = rig_core::test_utils::MockResponse;
/// One scripted provider streaming event.
pub type ScriptedStreamEvent = rig_core::test_utils::MockStreamEvent;

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
