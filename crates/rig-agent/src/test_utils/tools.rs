//! Tool helpers for deterministic tests.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    OneOrMany,
    message::{ImageMediaType, ToolResultContent},
    tool::{ContextualTool, ToolContext, ToolErrorKind, ToolExecutionError, ToolOutput, ToolSet},
    vector_store::{VectorSearchRequest, VectorStoreError, VectorStoreIndex, request::Filter},
    wasm_compat::WasmCompatSend,
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

impl ContextualTool for MockAddTool {
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

    async fn call(
        &self,
        _context: &mut crate::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

/// A caller-injected context value, like a session id or auth token carried in
/// a [`ToolContext`](crate::tool::ToolContext).
#[derive(Clone)]
pub struct SessionId(pub String);

/// A mock tool that records whatever it observed in its per-call
/// [`ToolContext`], so tests can assert the context reached tool execution.
///
/// The single `call` method records `session:<id>` (or `no-session`).
#[derive(Clone, Default)]
pub struct MockContextProbeTool {
    /// One entry per call, in call order — lets tests assert across multiple
    /// tool-call rounds, not just the most recent.
    seen: Arc<Mutex<Vec<String>>>,
}

impl MockContextProbeTool {
    /// What the tool observed on its most recent call, if it has been called.
    pub fn observed(&self) -> Option<String> {
        self.seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .last()
            .cloned()
    }

    /// Everything the tool observed, one entry per call in call order.
    pub fn observations(&self) -> Vec<String> {
        self.seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl ContextualTool for MockContextProbeTool {
    const NAME: &'static str = "context_probe";
    type Error = rig::tool::ToolExecutionError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "Records the SessionId observed in its call context".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {}})
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        let observed = match context.get::<SessionId>() {
            Some(session) => format!("session:{}", session.0),
            None => "no-session".to_string(),
        };
        self.seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(observed.clone());
        Ok(observed)
    }
}

/// A mock tool that subtracts `y` from `x`.
#[derive(Deserialize, Serialize)]
pub struct MockSubtractTool;

impl ContextualTool for MockSubtractTool {
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

    async fn call(
        &self,
        _context: &mut crate::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

/// Create a [`ToolSet`] containing [`MockAddTool`] and [`MockSubtractTool`].
pub fn mock_math_toolset() -> ToolSet {
    let mut toolset = ToolSet::default();
    toolset.add_tool(MockAddTool);
    toolset.add_tool(MockSubtractTool);
    toolset
}

/// A mock tool that returns a multiline string.
#[derive(Deserialize, Serialize)]
pub struct MockStringOutputTool;

impl ContextualTool for MockStringOutputTool {
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

    async fn call(
        &self,
        _context: &mut crate::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok("Hello\nWorld".to_string())
    }
}

/// A mock tool that returns explicit image content.
#[derive(Deserialize, Serialize)]
pub struct MockImageOutputTool;

impl ContextualTool for MockImageOutputTool {
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

    async fn call(
        &self,
        _context: &mut crate::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(ToolOutput::content(OneOrMany::one(
            ToolResultContent::image_base64("base64data==", Some(ImageMediaType::PNG), None),
        )))
    }
}

/// A mock tool named `generate_test_image` that returns a 1x1 red PNG image payload.
#[derive(Debug, Deserialize, Serialize)]
pub struct MockImageGeneratorTool;

impl ContextualTool for MockImageGeneratorTool {
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

    async fn call(
        &self,
        _context: &mut crate::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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

impl ContextualTool for MockObjectOutputTool {
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

    async fn call(
        &self,
        _context: &mut crate::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(json!({
            "status": "ok",
            "count": 42
        }))
    }
}

/// A mock tool named `example_tool` that returns `"Example answer"`.
pub struct MockExampleTool;

impl ContextualTool for MockExampleTool {
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

    async fn call(
        &self,
        _context: &mut crate::tool::ToolContext,
        _input: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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

impl ContextualTool for MockBarrierTool {
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

    async fn call(
        &self,
        _context: &mut crate::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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

impl ContextualTool for MockControlledTool {
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

    async fn call(
        &self,
        _context: &mut crate::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.started.notify_one();
        self.allow_finish.notified().await;
        Ok(42)
    }
}

/// A vector index that returns a predefined list of tool IDs from `top_n_ids`.
pub struct MockToolIndex {
    tool_ids: Vec<String>,
}

impl MockToolIndex {
    /// Create a tool index that returns the given IDs in order.
    pub fn new(tool_ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            tool_ids: tool_ids.into_iter().map(Into::into).collect(),
        }
    }
}

impl VectorStoreIndex for MockToolIndex {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        Ok(vec![])
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        Ok(self
            .tool_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (1.0 - (i as f64 * 0.1), id.clone()))
            .collect())
    }
}

/// A vector index that waits at a barrier before returning one tool ID.
pub struct BarrierMockToolIndex {
    barrier: Arc<tokio::sync::Barrier>,
    tool_id: String,
}

impl BarrierMockToolIndex {
    /// Create a barrier-backed tool index.
    pub fn new(barrier: Arc<tokio::sync::Barrier>, tool_id: impl Into<String>) -> Self {
        Self {
            barrier,
            tool_id: tool_id.into(),
        }
    }
}

impl VectorStoreIndex for BarrierMockToolIndex {
    type Filter = Filter<serde_json::Value>;

    async fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
        Ok(vec![])
    }

    async fn top_n_ids(
        &self,
        _req: VectorSearchRequest,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        self.barrier.wait().await;
        Ok(vec![(1.0, self.tool_id.clone())])
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

impl ContextualTool for MockFailingTool {
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

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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

impl ContextualTool for MockHandledFailureTool {
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

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
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

impl ContextualTool for MockDeniedTool {
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

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Err(MockToolError)
    }

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        ToolExecutionError::refused("operator authorization policy rejected the request")
            .with_model_feedback("access to this resource is not permitted")
            .with_source(error)
    }
}

/// Cloneable metadata a [`MockMetadataTool`] attaches to its result, used to
/// verify that result metadata reaches hooks without being sent to the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MockRequestId(pub String);

/// A tool whose success carries a [`MockRequestId`] in its result metadata.
/// Registered under the name `with_meta`.
#[derive(Clone)]
pub struct MockMetadataTool;

impl ContextualTool for MockMetadataTool {
    const NAME: &'static str = "with_meta";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    fn description(&self) -> String {
        "Succeeds and attaches request metadata".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {} })
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        context.insert_result(MockRequestId("req-7".to_string()));
        Ok("done".to_string())
    }
}
