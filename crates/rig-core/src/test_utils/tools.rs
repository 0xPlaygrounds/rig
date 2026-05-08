//! Tool helpers for deterministic tests.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    completion::ToolDefinition,
    tool::{Tool, ToolSet},
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

impl Tool for MockAddTool {
    const NAME: &'static str = "add";
    type Error = MockToolError;
    type Args = MockOperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Add x and y together".to_string(),
            parameters: json!({
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
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

/// A mock tool that subtracts `y` from `x`.
#[derive(Deserialize, Serialize)]
pub struct MockSubtractTool;

impl Tool for MockSubtractTool {
    const NAME: &'static str = "subtract";
    type Error = MockToolError;
    type Args = MockOperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Subtract y from x".to_string(),
            parameters: json!({
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
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
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

impl Tool for MockStringOutputTool {
    const NAME: &'static str = "string_output";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Returns a multiline string".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok("Hello\nWorld".to_string())
    }
}

/// A mock tool that returns image JSON as a string.
#[derive(Deserialize, Serialize)]
pub struct MockImageOutputTool;

impl Tool for MockImageOutputTool {
    const NAME: &'static str = "image_output";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Returns image JSON".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(json!({
            "type": "image",
            "data": "base64data==",
            "mimeType": "image/png"
        })
        .to_string())
    }
}

/// A mock tool named `generate_test_image` that returns a 1x1 red PNG image payload.
#[derive(Debug, Deserialize, Serialize)]
pub struct MockImageGeneratorTool;

impl Tool for MockImageGeneratorTool {
    const NAME: &'static str = "generate_test_image";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Generates a small test image (a 1x1 red pixel). Call this tool when asked to generate or show an image.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(json!({
            "type": "image",
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==",
            "mimeType": "image/png"
        })
        .to_string())
    }
}

/// A mock tool that returns a JSON object.
#[derive(Deserialize, Serialize)]
pub struct MockObjectOutputTool;

impl Tool for MockObjectOutputTool {
    const NAME: &'static str = "object_output";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = serde_json::Value;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Returns an object".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
        }
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

impl Tool for MockExampleTool {
    const NAME: &'static str = "example_tool";
    type Error = MockToolError;
    type Args = ();
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "A tool that returns some example text.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
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

impl Tool for MockBarrierTool {
    const NAME: &'static str = "barrier_tool";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Waits at a barrier to test concurrency".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }
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

impl Tool for MockControlledTool {
    const NAME: &'static str = "controlled";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Test tool".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
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
