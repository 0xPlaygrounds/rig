use rig::OneOrMany;
use rig::agent::{AgentBuilder, HookAction, PromptHook, ToolCallHookAction};
use rig::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, Prompt,
    ToolDefinition, Usage,
};
use rig::message::{AssistantContent, ToolCall, ToolFunction};
use rig::streaming::{StreamingCompletionResponse, StreamingResult};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// Mock model: returns add(x=1, y=2) tool call on turn 0, text on turn 1
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct AddToolCallModel {
    turn: Arc<AtomicUsize>,
}

impl AddToolCallModel {
    fn new() -> Self {
        Self {
            turn: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[allow(refining_impl_trait)]
impl CompletionModel for AddToolCallModel {
    type Response = ();
    type StreamingResponse = ();
    type Client = ();

    fn make(_: &Self::Client, _: impl Into<String>) -> Self {
        Self::new()
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let turn = self.turn.fetch_add(1, Ordering::SeqCst);

        if turn == 0 {
            Ok(CompletionResponse {
                choice: OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                    "tc_add".to_string(),
                    ToolFunction::new("add".to_string(), json!({"x": 1, "y": 2})),
                ))),
                usage: Usage::new(),
                raw_response: (),
                message_id: None,
            })
        } else {
            Ok(CompletionResponse {
                choice: OneOrMany::one(AssistantContent::text("done")),
                usage: Usage::new(),
                raw_response: (),
                message_id: None,
            })
        }
    }

    async fn stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let stream: StreamingResult<()> = Box::pin(futures::stream::empty());
        Ok(StreamingCompletionResponse::stream(stream))
    }
}

// ---------------------------------------------------------------------------
// Adder tool
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct AddArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("Math error")]
struct MathError;

#[derive(Deserialize, Serialize)]
struct Adder;

impl Tool for Adder {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = AddArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number" },
                    "y": { "type": "number" }
                },
                "required": ["x", "y"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ArgCapture {
    tool_call_args: Arc<Mutex<Vec<String>>>,
    tool_result_args: Arc<Mutex<Vec<String>>>,
    tool_results: Arc<Mutex<Vec<String>>>,
}

impl ArgCapture {
    fn new() -> Self {
        Self {
            tool_call_args: Arc::new(Mutex::new(Vec::new())),
            tool_result_args: Arc::new(Mutex::new(Vec::new())),
            tool_results: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[derive(Clone)]
struct PassthroughHook {
    capture: ArgCapture,
}

impl<M: CompletionModel> PromptHook<M> for PassthroughHook {
    async fn on_tool_call(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
    ) -> ToolCallHookAction {
        self.capture
            .tool_call_args
            .lock()
            .unwrap()
            .push(args.to_string());
        ToolCallHookAction::cont()
    }

    async fn on_tool_result(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
        result: &str,
    ) -> HookAction {
        self.capture
            .tool_result_args
            .lock()
            .unwrap()
            .push(args.to_string());
        self.capture
            .tool_results
            .lock()
            .unwrap()
            .push(result.to_string());
        HookAction::cont()
    }
}

#[derive(Clone)]
struct RewriteHook {
    capture: ArgCapture,
}

impl<M: CompletionModel> PromptHook<M> for RewriteHook {
    async fn on_tool_call(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
    ) -> ToolCallHookAction {
        self.capture
            .tool_call_args
            .lock()
            .unwrap()
            .push(args.to_string());
        ToolCallHookAction::continue_with(r#"{"x":100,"y":200}"#)
    }

    async fn on_tool_result(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
        result: &str,
    ) -> HookAction {
        self.capture
            .tool_result_args
            .lock()
            .unwrap()
            .push(args.to_string());
        self.capture
            .tool_results
            .lock()
            .unwrap()
            .push(result.to_string());
        HookAction::cont()
    }
}

#[derive(Clone)]
struct TerminateHook;

impl<M: CompletionModel> PromptHook<M> for TerminateHook {
    async fn on_tool_call(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
    ) -> ToolCallHookAction {
        ToolCallHookAction::terminate("stopped by hook")
    }
}

#[derive(Clone)]
struct SkipHook {
    capture: ArgCapture,
}

impl<M: CompletionModel> PromptHook<M> for SkipHook {
    async fn on_tool_call(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
    ) -> ToolCallHookAction {
        ToolCallHookAction::skip("tool not allowed")
    }

    async fn on_tool_result(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
        result: &str,
    ) -> HookAction {
        self.capture
            .tool_results
            .lock()
            .unwrap()
            .push(result.to_string());
        HookAction::cont()
    }
}

// ---------------------------------------------------------------------------
// P0: Regression — Continue passes original args
// ---------------------------------------------------------------------------

#[tokio::test]
async fn continue_passes_original_args() {
    let agent = AgentBuilder::new(AddToolCallModel::new())
        .tool(Adder)
        .build();

    let capture = ArgCapture::new();
    let hook = PassthroughHook {
        capture: capture.clone(),
    };

    let response: String = agent
        .prompt("add 1 and 2")
        .max_turns(3)
        .with_hook(hook)
        .await
        .expect("prompt should succeed");

    assert_eq!(response, "done");

    let results = capture.tool_results.lock().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], "3");

    let result_args = capture.tool_result_args.lock().unwrap();
    assert_eq!(result_args.len(), 1);
    assert!(
        result_args[0].contains("1") && result_args[0].contains("2"),
        "on_tool_result should see original args, got: {}",
        result_args[0]
    );
}

// ---------------------------------------------------------------------------
// P0: Contract — ContinueWith replacement args reach call_tool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn continue_with_replaces_args_in_call_tool() {
    let agent = AgentBuilder::new(AddToolCallModel::new())
        .tool(Adder)
        .build();

    let capture = ArgCapture::new();
    let hook = RewriteHook {
        capture: capture.clone(),
    };

    let response: String = agent
        .prompt("add 1 and 2")
        .max_turns(3)
        .with_hook(hook)
        .await
        .expect("prompt should succeed");

    assert_eq!(response, "done");

    let results = capture.tool_results.lock().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0], "300",
        "tool should compute 100+200=300, not 1+2=3"
    );
}

// ---------------------------------------------------------------------------
// P0: Contract — on_tool_result receives replacement args
// ---------------------------------------------------------------------------

#[tokio::test]
async fn continue_with_replacement_args_reach_on_tool_result() {
    let agent = AgentBuilder::new(AddToolCallModel::new())
        .tool(Adder)
        .build();

    let capture = ArgCapture::new();
    let hook = RewriteHook {
        capture: capture.clone(),
    };

    let _response: String = agent
        .prompt("add")
        .max_turns(3)
        .with_hook(hook)
        .await
        .expect("prompt should succeed");

    let call_args = capture.tool_call_args.lock().unwrap();
    assert_eq!(call_args.len(), 1);
    assert!(
        call_args[0].contains("1") && call_args[0].contains("2"),
        "on_tool_call should see original args from LLM, got: {}",
        call_args[0]
    );

    let result_args = capture.tool_result_args.lock().unwrap();
    assert_eq!(result_args.len(), 1);
    assert!(
        result_args[0].contains("100") && result_args[0].contains("200"),
        "on_tool_result should see replacement args, got: {}",
        result_args[0]
    );
}

// ---------------------------------------------------------------------------
// P1: Regression — Terminate still cancels the loop
// ---------------------------------------------------------------------------

#[tokio::test]
async fn terminate_still_cancels_loop() {
    use rig::completion::PromptError;

    let agent = AgentBuilder::new(AddToolCallModel::new())
        .tool(Adder)
        .build();

    let result = agent
        .prompt("add")
        .max_turns(3)
        .with_hook(TerminateHook)
        .await;

    match result {
        Err(PromptError::PromptCancelled { reason, .. }) => {
            assert_eq!(reason, "stopped by hook");
        }
        Ok(_) => panic!("expected PromptCancelled, got Ok"),
        Err(other) => panic!("expected PromptCancelled, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// P1: Regression — Skip returns reason as tool result, on_tool_result does NOT fire
// ---------------------------------------------------------------------------

#[tokio::test]
async fn skip_returns_reason_and_bypasses_on_tool_result() {
    let agent = AgentBuilder::new(AddToolCallModel::new())
        .tool(Adder)
        .build();

    let capture = ArgCapture::new();
    let hook = SkipHook {
        capture: capture.clone(),
    };

    let response: String = agent
        .prompt("add")
        .max_turns(3)
        .with_hook(hook)
        .await
        .expect("prompt should succeed");

    assert_eq!(response, "done");

    let results = capture.tool_results.lock().unwrap();
    assert!(
        results.is_empty(),
        "on_tool_result should NOT fire for skipped tool calls, but got: {results:?}"
    );
}

// ---------------------------------------------------------------------------
// P1: Boundary — ContinueWith with invalid JSON
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct InvalidJsonHook;

impl<M: CompletionModel> PromptHook<M> for InvalidJsonHook {
    async fn on_tool_call(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
    ) -> ToolCallHookAction {
        ToolCallHookAction::continue_with("not valid json")
    }
}

#[tokio::test]
async fn continue_with_invalid_json_does_not_panic() {
    let agent = AgentBuilder::new(AddToolCallModel::new())
        .tool(Adder)
        .build();

    let result: String = agent
        .prompt("add")
        .max_turns(3)
        .with_hook(InvalidJsonHook)
        .await
        .expect("should not panic, tool error becomes string result");

    assert_eq!(result, "done");
}

// ---------------------------------------------------------------------------
// P1: Error path — ContinueWith args that fail tool deserialization
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct WrongFieldsHook {
    capture: ArgCapture,
}

impl<M: CompletionModel> PromptHook<M> for WrongFieldsHook {
    async fn on_tool_call(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
    ) -> ToolCallHookAction {
        ToolCallHookAction::continue_with(r#"{"z": 999}"#)
    }

    async fn on_tool_result(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
        result: &str,
    ) -> HookAction {
        self.capture
            .tool_results
            .lock()
            .unwrap()
            .push(result.to_string());
        HookAction::cont()
    }
}

#[tokio::test]
async fn continue_with_wrong_fields_triggers_tool_error() {
    let agent = AgentBuilder::new(AddToolCallModel::new())
        .tool(Adder)
        .build();

    let capture = ArgCapture::new();
    let hook = WrongFieldsHook {
        capture: capture.clone(),
    };

    let response: String = agent
        .prompt("add")
        .max_turns(3)
        .with_hook(hook)
        .await
        .expect("should succeed — tool error becomes string result");

    assert_eq!(response, "done");

    let results = capture.tool_results.lock().unwrap();
    assert_eq!(
        results.len(),
        1,
        "on_tool_result should still fire for tool errors"
    );
    assert!(
        results[0].contains("missing field"),
        "tool error should mention missing field, got: {}",
        results[0]
    );
}
