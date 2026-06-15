//! Shared fixtures for the tool-pipeline cassette suites: counting tools,
//! deliberately failing tools, prompt hooks, and embeddable tools. These
//! suites lock in the externally observable behavior of the handrolled tool
//! plumbing (`ToolSet`, `ToolServer`, hook dispatch, result shaping) ahead of
//! the planned migration onto `rmcp`.
//!
//! ## On loose assertions
//!
//! Many assertions in these suites are intentionally loose (`contains`,
//! `any`, `>=`, "mentions the number") rather than exact equality. Cassette
//! replay is deterministic, so the *recorded* values are fixed — but these
//! cassettes are periodically re-recorded against the live provider, and any
//! value shaped by model-generated text, model-chosen call count/ordering,
//! provider token counts, or live embedding rankings can legitimately change
//! at re-record time. Only values synthesized purely by rig code with no
//! model input (a verbatim hook reason, a fixed adapter string, a
//! code-enforced sample cap) are pinned to exact equality. Please keep that
//! split when adding assertions: tightening a model-dependent assertion to
//! `assert_eq!` will spuriously fail the next re-recording.
#![allow(dead_code)]

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::agent::{HookAction, PromptHook, ToolCallHookAction};
use rig::completion::{CompletionModel, ToolDefinition};
use rig::tool::server::ToolServerHandle;
use rig::tool::{Tool, ToolEmbedding};
use serde::{Deserialize, Serialize};
use serde_json::json;

pub(crate) const FORCE_TOOLS_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for every arithmetic operation instead of computing results yourself. Once you have all the tool results you need, reply with the final numeric answer in plain text.";

/// 1x1 red pixel PNG used for image tool-result fixtures.
pub(crate) const RED_PIXEL_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

/// Cloneable execution counter shared between a tool fixture and its test.
#[derive(Clone, Default)]
pub(crate) struct CallCounter(Arc<AtomicUsize>);

impl CallCounter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }

    fn bump(&self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Deserialize, Serialize)]
pub(crate) struct OperationArgs {
    pub(crate) x: i64,
    pub(crate) y: i64,
}

#[derive(Debug, thiserror::Error)]
#[error("math error")]
pub(crate) struct MathError;

fn operation_definition(name: &str, description: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: description.to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "x": { "type": "number", "description": "The first operand" },
                "y": { "type": "number", "description": "The second operand" }
            },
            "required": ["x", "y"]
        }),
    }
}

/// `add` tool that counts its real executions.
#[derive(Clone, Default)]
pub(crate) struct CountingAdd {
    pub(crate) counter: CallCounter,
}

impl Tool for CountingAdd {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        operation_definition(Self::NAME, "Add x and y together")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.counter.bump();
        Ok(args.x + args.y)
    }
}

/// `subtract` tool that counts its real executions.
#[derive(Clone, Default)]
pub(crate) struct CountingSubtract {
    pub(crate) counter: CallCounter,
}

impl Tool for CountingSubtract {
    const NAME: &'static str = "subtract";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        operation_definition(Self::NAME, "Subtract y from x (i.e. x - y)")
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.counter.bump();
        Ok(args.x - args.y)
    }
}

#[derive(Deserialize, Serialize)]
pub(crate) struct EmptyArgs {}

/// Zero-argument tool: pins the null-arguments -> `{}` normalization on the
/// execution path and a verbatim string output.
#[derive(Clone, Default)]
pub(crate) struct CountingPing {
    pub(crate) counter: CallCounter,
}

pub(crate) const PING_OUTPUT: &str = "pong-crimson-7423";

impl Tool for CountingPing {
    const NAME: &'static str = "ping";
    type Error = MathError;
    type Args = EmptyArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Return the current ping marker. Takes no arguments.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.counter.bump();
        Ok(PING_OUTPUT.to_string())
    }
}

#[derive(Deserialize, Serialize)]
pub(crate) struct CodewordArgs {
    pub(crate) team: String,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct CodewordError(pub(crate) String);

pub(crate) const CODEWORD_GUIDANCE: &str =
    "the red team was disbanded; look up the codeword for the \"blue\" team instead";
pub(crate) const BLUE_CODEWORD: &str = "azure-falcon";

/// Tool whose first expected call fails with corrective guidance, pinning how
/// tool `Err` values are stringified into tool results the model can act on.
#[derive(Clone, Default)]
pub(crate) struct CodewordLookup {
    pub(crate) counter: CallCounter,
}

impl Tool for CodewordLookup {
    const NAME: &'static str = "lookup_codeword";
    type Error = CodewordError;
    type Args = CodewordArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Look up the secret codeword for a team.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "team": { "type": "string", "description": "The team name, lowercase" }
                },
                "required": ["team"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.counter.bump();
        match args.team.as_str() {
            "blue" => Ok(BLUE_CODEWORD.to_string()),
            _ => Err(CodewordError(CODEWORD_GUIDANCE.to_string())),
        }
    }
}

#[derive(Deserialize, Serialize)]
pub(crate) struct StrictRegisterArgs {
    /// Deliberately stricter than the advertised schema: the wire schema
    /// declares `seats` as a string, so the model's arguments fail to
    /// deserialize and surface a `ToolError::JsonError` to the model.
    pub(crate) seats: u64,
}

/// Tool whose advertised schema disagrees with its `Args` type.
#[derive(Clone, Default)]
pub(crate) struct StrictRegister {
    pub(crate) counter: CallCounter,
}

impl Tool for StrictRegister {
    const NAME: &'static str = "register_guests";
    type Error = MathError;
    type Args = StrictRegisterArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Register guests for the event.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "seats": {
                        "type": "string",
                        "description": "Number of seats, spelled out as a lowercase English word (e.g. \"four\")"
                    }
                },
                "required": ["seats"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.counter.bump();
        Ok(format!("registered {} guests", args.seats))
    }
}

pub(crate) const MOTTO_OUTPUT: &str = "steady hands\ncalm waters";

/// Tool with a `String` output: pins that string outputs are sent verbatim
/// (newlines preserved, no JSON quoting).
#[derive(Clone, Default)]
pub(crate) struct MottoTool;

impl Tool for MottoTool {
    const NAME: &'static str = "fetch_motto";
    type Error = MathError;
    type Args = EmptyArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Fetch the two-line workshop motto.".to_string(),
            parameters: json!({ "type": "object", "properties": {}, "required": [] }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(MOTTO_OUTPUT.to_string())
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct ConfigOutput {
    pub(crate) service: String,
    pub(crate) max_retries: u64,
}

/// Tool with a structured output: pins that non-string outputs are
/// JSON-serialized into the tool result.
#[derive(Clone, Default)]
pub(crate) struct ConfigTool;

impl ConfigTool {
    pub(crate) fn expected_output_json() -> String {
        serde_json::to_string(&ConfigOutput {
            service: "cassette-lab".to_string(),
            max_retries: 3,
        })
        .expect("config output should serialize")
    }
}

impl Tool for ConfigTool {
    const NAME: &'static str = "fetch_config";
    type Error = MathError;
    type Args = EmptyArgs;
    type Output = ConfigOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Fetch the service configuration object.".to_string(),
            parameters: json!({ "type": "object", "properties": {}, "required": [] }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(ConfigOutput {
            service: "cassette-lab".to_string(),
            max_retries: 3,
        })
    }
}

/// Tool returning the top-level image JSON shape that
/// `ToolResultContent::from_tool_output` converts into an image part.
#[derive(Clone, Default)]
pub(crate) struct BadgeImageTool;

impl Tool for BadgeImageTool {
    const NAME: &'static str = "fetch_badge_image";
    type Error = MathError;
    type Args = EmptyArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Fetch the attendee badge as an image the assistant must inspect."
                .to_string(),
            parameters: json!({ "type": "object", "properties": {}, "required": [] }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(json!({
            "type": "image",
            "data": RED_PIXEL_PNG_BASE64,
            "mimeType": "image/png"
        })
        .to_string())
    }
}

/// Hook that records every tool call and tool result it observes.
#[derive(Clone, Default)]
pub(crate) struct ToolEventRecorder {
    pub(crate) calls: Arc<Mutex<Vec<(String, String)>>>,
    pub(crate) results: Arc<Mutex<Vec<(String, String, String)>>>,
}

impl ToolEventRecorder {
    pub(crate) fn recorded_calls(&self) -> Vec<(String, String)> {
        self.calls
            .lock()
            .expect("calls lock should not be poisoned")
            .clone()
    }

    pub(crate) fn recorded_results(&self) -> Vec<(String, String, String)> {
        self.results
            .lock()
            .expect("results lock should not be poisoned")
            .clone()
    }
}

impl<M> PromptHook<M> for ToolEventRecorder
where
    M: CompletionModel,
{
    async fn on_tool_call(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
    ) -> ToolCallHookAction {
        self.calls
            .lock()
            .expect("calls lock should not be poisoned")
            .push((tool_name.to_string(), args.to_string()));
        ToolCallHookAction::cont()
    }

    async fn on_tool_result(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        args: &str,
        result: &str,
    ) -> HookAction {
        self.results
            .lock()
            .expect("results lock should not be poisoned")
            .push((tool_name.to_string(), args.to_string(), result.to_string()));
        HookAction::cont()
    }
}

/// Hook that skips a named tool with a fixed reason instead of executing it.
#[derive(Clone)]
pub(crate) struct SkipToolHook {
    pub(crate) tool_name: &'static str,
    pub(crate) reason: &'static str,
}

impl<M> PromptHook<M> for SkipToolHook
where
    M: CompletionModel,
{
    async fn on_tool_call(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
    ) -> ToolCallHookAction {
        if tool_name == self.tool_name {
            ToolCallHookAction::skip(self.reason)
        } else {
            ToolCallHookAction::cont()
        }
    }
}

/// Hook that terminates the run when a named tool is about to execute.
#[derive(Clone)]
pub(crate) struct TerminateOnToolHook {
    pub(crate) tool_name: &'static str,
    pub(crate) reason: &'static str,
}

impl<M> PromptHook<M> for TerminateOnToolHook
where
    M: CompletionModel,
{
    async fn on_tool_call(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
    ) -> ToolCallHookAction {
        if tool_name == self.tool_name {
            ToolCallHookAction::terminate(self.reason)
        } else {
            ToolCallHookAction::cont()
        }
    }
}

/// Hook that removes a tool from the shared tool server right before it would
/// execute, forcing the execution-time `ToolNotFoundError` path.
#[derive(Clone)]
pub(crate) struct RemoveToolBeforeExecutionHook {
    pub(crate) handle: ToolServerHandle,
    pub(crate) tool_name: &'static str,
}

impl<M> PromptHook<M> for RemoveToolBeforeExecutionHook
where
    M: CompletionModel,
{
    async fn on_tool_call(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
    ) -> ToolCallHookAction {
        if tool_name == self.tool_name {
            self.handle
                .remove_tool(self.tool_name)
                .await
                .expect("tool removal should succeed");
        }
        ToolCallHookAction::cont()
    }
}

#[derive(Debug, thiserror::Error)]
#[error("init error")]
pub(crate) struct InitError;

macro_rules! embeddable_operation {
    ($name:ident, $tool_name:literal, $description:literal, $embedding_doc:literal, $op:expr) => {
        #[derive(Clone, Default, Deserialize, Serialize)]
        pub(crate) struct $name {
            #[serde(skip)]
            pub(crate) counter: CallCounter,
        }

        impl Tool for $name {
            const NAME: &'static str = $tool_name;
            type Error = MathError;
            type Args = OperationArgs;
            type Output = i64;

            async fn definition(&self, _prompt: String) -> ToolDefinition {
                operation_definition(Self::NAME, $description)
            }

            async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
                self.counter.bump();
                let op: fn(i64, i64) -> i64 = $op;
                Ok(op(args.x, args.y))
            }
        }

        impl ToolEmbedding for $name {
            type InitError = InitError;
            type Context = ();
            type State = ();

            fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
                Ok(Self::default())
            }

            fn embedding_docs(&self) -> Vec<String> {
                vec![$embedding_doc.into()]
            }

            fn context(&self) -> Self::Context {}
        }
    };
}

embeddable_operation!(
    EmbedAdd,
    "add",
    "Add x and y together",
    "Add two numbers together to get their sum",
    |x, y| x + y
);
embeddable_operation!(
    EmbedSubtract,
    "subtract",
    "Subtract y from x (i.e. x - y)",
    "Subtract one number from another to get their difference",
    |x, y| x - y
);
embeddable_operation!(
    EmbedMultiply,
    "multiply",
    "Multiply x and y together",
    "Multiply two numbers together to get their product",
    |x, y| x * y
);
