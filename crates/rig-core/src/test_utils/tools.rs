//! Tool helpers for deterministic tests.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    completion::ToolDefinition,
    tool::{Tool, ToolCallExtensions, ToolFailure, ToolFailureKind, ToolReturn, ToolSet},
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

/// A caller-injected context value, like a session id or auth token carried in
/// a [`ToolCallExtensions`](crate::tool::ToolCallExtensions).
#[derive(Clone)]
pub struct SessionId(pub String);

/// A mock tool that records whatever it observed in its per-call
/// [`ToolCallExtensions`], so tests can assert the context reached tool execution.
///
/// `call_with_extensions` records `session:<id>` (or `no-session` when no
/// [`SessionId`] is present). The plain `call` body records `call-no-context` as
/// a sentinel: because an overridden `call_with_extensions` is the single dispatch
/// entry point, that sentinel must never surface from a dispatched run —
/// observing it would mean dispatch wrongly bypassed the context-aware path.
#[derive(Clone, Default)]
pub struct MockExtensionsProbeTool {
    /// One entry per call, in call order — lets tests assert across multiple
    /// tool-call rounds, not just the most recent.
    seen: Arc<Mutex<Vec<String>>>,
}

impl MockExtensionsProbeTool {
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

impl Tool for MockExtensionsProbeTool {
    const NAME: &'static str = "context_probe";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Records the SessionId observed in its call context".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push("call-no-context".to_string());
        Ok("call-no-context".to_string())
    }

    async fn call_with_extensions(
        &self,
        _args: Self::Args,
        extensions: &ToolCallExtensions,
    ) -> Result<Self::Output, Self::Error> {
        let observed = match extensions.get::<SessionId>() {
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

/// Error type for [`MockFailingTool`], carrying a fixed message.
#[derive(Debug, thiserror::Error)]
#[error("mock tool call failed")]
pub struct MockFailure;

/// A tool that always fails, classifying its error as a configured
/// [`ToolFailureKind`] via [`Tool::classify_error`]. Used to exercise structured
/// tool-failure surfacing (timeout, not-found, rate-limited, …) without a live
/// provider. Registered under the name `flaky_tool`.
#[derive(Clone)]
pub struct MockFailingTool {
    kind: ToolFailureKind,
}

impl MockFailingTool {
    /// A tool that fails with the given classification every call.
    pub fn new(kind: ToolFailureKind) -> Self {
        Self { kind }
    }
}

impl Tool for MockFailingTool {
    const NAME: &'static str = "flaky_tool";
    type Error = MockFailure;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "A tool that always fails".to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Err(MockFailure)
    }

    fn classify_error(&self, error: &Self::Error) -> ToolFailure {
        let message = error.to_string();
        match self.kind {
            ToolFailureKind::Timeout => ToolFailure::timeout(message),
            ToolFailureKind::NotFound => ToolFailure::not_found(message).with_http_status(404),
            ToolFailureKind::RateLimited => {
                ToolFailure::rate_limited(message).with_http_status(429)
            }
            other => ToolFailure::new(other, message),
        }
    }
}

/// A tool that reports a *handled* failure via [`ToolReturn`]: the Rust call
/// succeeds, but the returned outcome is a classified [`ToolFailure`] while the
/// model still receives useful output. Registered under the name `lookup`.
#[derive(Clone)]
pub struct MockHandledFailureTool;

impl Tool for MockHandledFailureTool {
    const NAME: &'static str = "lookup";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Looks up a record".to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Overridden by `call_structured` under dynamic dispatch; present so the
        // trait is satisfied for direct callers.
        Ok("no record found for id 42".to_string())
    }

    async fn call_structured(
        &self,
        _args: Self::Args,
        _extensions: &ToolCallExtensions,
    ) -> Result<ToolReturn<Self::Output>, Self::Error> {
        Ok(ToolReturn::failed(
            "no record found for id 42; try a different id".to_string(),
            ToolFailure::not_found("record id 42 is missing").with_http_status(404),
        ))
    }
}

/// A tool that declares the call denied from inside the tool (via
/// [`ToolReturn::denied`]), producing a [`ToolOutcome::Denied`](crate::tool::ToolOutcome::Denied)
/// outcome — as opposed to a hook `Flow::Skip`, which is `Skipped`. Registered
/// under the name `guarded`.
#[derive(Clone)]
pub struct MockDeniedTool;

impl Tool for MockDeniedTool {
    const NAME: &'static str = "guarded";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "A tool with an internal authorization check".to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok("ok".to_string())
    }

    async fn call_structured(
        &self,
        _args: Self::Args,
        _extensions: &ToolCallExtensions,
    ) -> Result<ToolReturn<Self::Output>, Self::Error> {
        Ok(ToolReturn::denied(
            "access to this resource is not permitted".to_string(),
        ))
    }
}

/// A cloneable extension value a [`MockMetadataTool`] attaches to its result, to
/// verify result extensions reach hooks without being sent to the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MockRequestId(pub String);

/// A tool whose success carries a [`MockRequestId`] in its result extensions.
/// Registered under the name `with_meta`.
#[derive(Clone)]
pub struct MockMetadataTool;

impl Tool for MockMetadataTool {
    const NAME: &'static str = "with_meta";
    type Error = MockToolError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Succeeds and attaches request metadata".to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok("done".to_string())
    }

    async fn call_structured(
        &self,
        _args: Self::Args,
        _extensions: &ToolCallExtensions,
    ) -> Result<ToolReturn<Self::Output>, Self::Error> {
        Ok(ToolReturn::success("done".to_string())
            .with_extension(MockRequestId("req-7".to_string())))
    }
}
