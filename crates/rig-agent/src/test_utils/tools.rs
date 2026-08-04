//! Tool helpers for deterministic tests.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::json;

use rig_core::{
    OneOrMany,
    message::{ImageMediaType, ToolResultContent},
};

use crate::executor::ToolExecutor;
use crate::tool::{
    PortableDynamicTool, PortableTool, ToolErrorKind, ToolExecutionError, ToolOutput,
};

/// Shared error type for mock tools.
#[derive(Debug, thiserror::Error)]
#[error("Mock tool error")]
pub struct MockToolError;

/// Arguments for arithmetic mock tools.
#[derive(Deserialize)]
pub struct MockOperationArgs {
    x: i32,
    y: i32,
}

/// A mock tool that adds `x` and `y`.
#[derive(Deserialize, Serialize)]
pub struct MockAddTool;

impl PortableTool for MockAddTool {
    const NAME: &'static str = "add";
    type Error = MockToolError;
    type Args = MockOperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The first number to add"
                },
                "y": {
                    "type": "number",
                    "description": "The second number to add"
                }
            },
            "required": ["x", "y"],
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

/// A mock tool that subtracts `y` from `x`.
#[derive(Deserialize, Serialize)]
pub struct MockSubtractTool;

impl PortableTool for MockSubtractTool {
    const NAME: &'static str = "subtract";
    type Error = MockToolError;
    type Args = MockOperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Subtract y from x".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": "The number to subtract from"
                },
                "y": {
                    "type": "number",
                    "description": "The number to subtract"
                }
            },
            "required": ["x", "y"],
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

/// Create a [`ToolExecutor`] containing [`MockAddTool`] and [`MockSubtractTool`].
pub fn mock_math_executor() -> ToolExecutor {
    ToolExecutor::new()
        .register(PortableDynamicTool::from_portable(MockAddTool))
        .register(PortableDynamicTool::from_portable(MockSubtractTool))
}

/// A mock tool that returns a multiline string.
#[derive(Deserialize, Serialize)]
pub struct MockStringOutputTool;

impl PortableTool for MockStringOutputTool {
    const NAME: &'static str = "string_output";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "Returns a multiline string".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok("Hello\nWorld".to_string())
    }
}

/// A mock tool that returns explicit image content.
#[derive(Deserialize, Serialize)]
pub struct MockImageOutputTool;

impl PortableTool for MockImageOutputTool {
    const NAME: &'static str = "image_output";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = ToolOutput;

    fn description(&self) -> String {
        "Returns an image".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(ToolOutput::content(OneOrMany::one(
            ToolResultContent::image_base64("base64data==", Some(ImageMediaType::PNG), None),
        )))
    }
}

/// A mock tool named `generate_test_image` that returns a 1x1 red PNG image payload.
#[derive(Debug, Deserialize, Serialize)]
pub struct MockImageGeneratorTool;

impl PortableTool for MockImageGeneratorTool {
    const NAME: &'static str = "generate_test_image";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = ToolOutput;

    fn description(&self) -> String {
        "Generates a small test image (a 1x1 red pixel). Call this tool when asked to generate or show an image.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(ToolOutput::content(OneOrMany::one(
            ToolResultContent::image_base64(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==",
                Some(ImageMediaType::PNG),
                None,
            ),
        )))
    }
}

/// A mock tool that returns a JSON object.
#[derive(Deserialize, Serialize)]
pub struct MockObjectOutputTool;

impl PortableTool for MockObjectOutputTool {
    const NAME: &'static str = "object_output";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = serde_json::Value;

    fn description(&self) -> String {
        "Returns an object".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(json!({
            "status": "ok",
            "count": 42
        }))
    }
}

/// A mock tool named `example_tool` that returns `"Example answer"`.
pub struct MockExampleTool;

impl PortableTool for MockExampleTool {
    const NAME: &'static str = "example_tool";
    type Error = MockToolError;
    type Args = ();
    type Output = String;

    fn description(&self) -> String {
        "A tool that returns some example text.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn call(&self, _input: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok("Example answer".to_string())
    }
}

/// A mock tool that waits at a barrier before returning `"done"`.
#[derive(Clone)]
pub struct MockBarrierTool {
    /// Barrier waited on during each tool call.
    pub barrier: Arc<tokio::sync::Barrier>,
}

impl MockBarrierTool {
    /// Create a barrier-backed tool.
    pub fn new(barrier: Arc<tokio::sync::Barrier>) -> Self {
        Self { barrier }
    }
}

impl PortableTool for MockBarrierTool {
    const NAME: &'static str = "barrier_tool";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "Waits at a barrier to test concurrency".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {}})
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.barrier.wait().await;
        Ok("done".to_string())
    }
}

/// A mock tool that notifies when started and waits for an explicit finish signal.
#[derive(Clone)]
pub struct MockControlledTool {
    /// Notified when a tool call starts.
    pub started: Arc<tokio::sync::Notify>,
    /// Waited on before a tool call finishes.
    pub allow_finish: Arc<tokio::sync::Notify>,
}

impl MockControlledTool {
    /// Create a controlled tool from notification primitives.
    pub fn new(started: Arc<tokio::sync::Notify>, allow_finish: Arc<tokio::sync::Notify>) -> Self {
        Self {
            started,
            allow_finish,
        }
    }
}

impl PortableTool for MockControlledTool {
    const NAME: &'static str = "controlled";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = i32;

    fn description(&self) -> String {
        "Test tool".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {}})
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.started.notify_one();
        self.allow_finish.notified().await;
        Ok(42)
    }
}

/// Error type for [`MockFailingTool`], carrying a fixed message.
#[derive(Debug, thiserror::Error)]
#[error("mock tool call failed")]
pub struct MockFailure;

/// A tool that always fails with a configured [`ToolErrorKind`]. Used to exercise structured
/// tool-failure surfacing (timeout, not-found, rate-limited, …) without a live
/// provider. Registered under the name `flaky_tool`.
#[derive(Clone)]
pub struct MockFailingTool {
    kind: ToolErrorKind,
}

impl MockFailingTool {
    /// A tool that fails with the given classification every call.
    pub fn new(kind: ToolErrorKind) -> Self {
        Self { kind }
    }
}

impl PortableTool for MockFailingTool {
    const NAME: &'static str = "flaky_tool";
    type Error = MockFailure;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "A tool that always fails".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Err(MockFailure)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        let error = ToolExecutionError::new(self.kind, error.to_string()).with_source(error);
        match self.kind {
            ToolErrorKind::NotFound => error.with_http_status(404),
            ToolErrorKind::RateLimited => error.with_http_status(429),
            _ => error,
        }
    }
}

/// A tool failure with separate operator and model-visible feedback.
#[derive(Clone)]
pub struct MockHandledFailureTool;

impl PortableTool for MockHandledFailureTool {
    const NAME: &'static str = "lookup";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "Looks up a record".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Err(MockToolError)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::not_found("record id 42 is missing")
            .with_http_status(404)
            .with_model_feedback("no record found for id 42; try a different id")
            .with_source(error)
    }
}

/// A tool that refuses execution, distinct from a framework policy skip.
#[derive(Clone)]
pub struct MockDeniedTool;

impl PortableTool for MockDeniedTool {
    const NAME: &'static str = "guarded";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "A tool with an internal authorization check".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Err(MockToolError)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::refused("operator authorization policy rejected the request")
            .with_model_feedback("access to this resource is not permitted")
            .with_source(error)
    }
}
