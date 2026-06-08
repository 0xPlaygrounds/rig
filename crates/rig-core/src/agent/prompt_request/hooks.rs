//! Optional hooks for agent prompting.
//! Hooks can be used to create custom behaviour like logging, calling external services or conditionally skipping tool calls.
//! Alternatively, you can also use them to terminate agent loops early.

use crate::{
    completion::CompletionModel,
    message::{Message, ToolChoice},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

/// Context passed to [`PromptHook::on_invalid_tool_call`] when the model emits a tool call
/// that Rig would reject before normal tool-call hooks or execution.
#[derive(Debug, Clone)]
pub struct InvalidToolCallContext {
    /// Tool name emitted by the model.
    pub tool_name: String,
    /// Provider-supplied tool call ID, when available.
    pub tool_call_id: Option<String>,
    /// Internal Rig call ID, when available.
    pub internal_call_id: Option<String>,
    /// JSON arguments emitted for the tool call, when available.
    pub args: Option<String>,
    /// Executable Rig tools advertised to the provider for this turn.
    pub available_tools: Vec<String>,
    /// Tools allowed by the active [`ToolChoice`] for this turn.
    pub allowed_tools: Vec<String>,
    /// Active tool choice for this turn.
    pub tool_choice: Option<ToolChoice>,
    /// Diagnostic chat history including the rejected model output when available.
    pub chat_history: Vec<Message>,
    /// Whether the rejected call came from the streaming path.
    pub is_streaming: bool,
}

/// Recovery action for invalid tool-call hooks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidToolCallHookAction {
    /// Preserve Rig's default fail-fast behavior.
    Fail,
    /// Retry the model turn with corrective feedback.
    Retry { feedback: String },
    /// Rewrite only the emitted tool name. The repaired name is revalidated
    /// against registered tools and the current `ToolChoice` before use.
    Repair { tool_name: String },
    /// Treat an invalid structured tool call as skipped by returning synthetic
    /// feedback as its tool result. This does not execute the invalid tool.
    Skip { reason: String },
}

impl InvalidToolCallHookAction {
    /// Preserve Rig's default fail-fast behavior.
    pub fn fail() -> Self {
        Self::Fail
    }

    /// Retry the model turn with corrective feedback.
    pub fn retry(feedback: impl Into<String>) -> Self {
        Self::Retry {
            feedback: feedback.into(),
        }
    }

    /// Repair the emitted tool name.
    pub fn repair(tool_name: impl Into<String>) -> Self {
        Self::Repair {
            tool_name: tool_name.into(),
        }
    }

    /// Skip the invalid call with a synthetic tool result.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }
}

/// Trait for per-request hooks to observe tool call events.
pub trait PromptHook<M>: Clone + WasmCompatSend + WasmCompatSync
where
    M: CompletionModel,
{
    /// Called before the prompt is sent to the model
    fn on_completion_call(
        &self,
        _prompt: &Message,
        _history: &[Message],
    ) -> impl Future<Output = HookAction> + WasmCompatSend {
        async { HookAction::cont() }
    }

    /// Called after the prompt is sent to the model and a response is received.
    fn on_completion_response(
        &self,
        _prompt: &Message,
        _response: &crate::completion::CompletionResponse<M::Response>,
    ) -> impl Future<Output = HookAction> + WasmCompatSend {
        async { HookAction::cont() }
    }

    /// Called when a model-emitted tool call is unknown or disallowed by the
    /// current request's tool choice.
    ///
    /// The default behavior remains fail-fast. Override this method to opt into
    /// retry, repair, or skip recovery for invalid tool calls.
    fn on_invalid_tool_call(
        &self,
        _context: &InvalidToolCallContext,
    ) -> impl Future<Output = InvalidToolCallHookAction> + WasmCompatSend {
        async { InvalidToolCallHookAction::fail() }
    }

    /// Called before a tool is invoked.
    ///
    /// # Returns
    /// - `ToolCallHookAction::Continue` - Allow tool execution to proceed
    /// - `ToolCallHookAction::Skip { reason }` - Reject tool execution; `reason` will be returned to the LLM as the tool result
    fn on_tool_call(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
    ) -> impl Future<Output = ToolCallHookAction> + WasmCompatSend {
        async { ToolCallHookAction::cont() }
    }

    /// Called after a tool is invoked (and a result has been returned).
    fn on_tool_result(
        &self,
        _tool_name: &str,
        _tool_call_id: Option<String>,
        _internal_call_id: &str,
        _args: &str,
        _result: &str,
    ) -> impl Future<Output = HookAction> + WasmCompatSend {
        async { HookAction::cont() }
    }

    /// Called when receiving a text delta (streaming responses only)
    fn on_text_delta(
        &self,
        _text_delta: &str,
        _aggregated_text: &str,
    ) -> impl Future<Output = HookAction> + Send {
        async { HookAction::cont() }
    }

    /// Called when receiving a tool call delta (streaming_responses_only).
    /// `tool_name` is Some on the first delta for a tool call, None on subsequent deltas.
    fn on_tool_call_delta(
        &self,
        _tool_call_id: &str,
        _internal_call_id: &str,
        _tool_name: Option<&str>,
        _tool_call_delta: &str,
    ) -> impl Future<Output = HookAction> + Send {
        async { HookAction::cont() }
    }

    /// Called after the model provider has finished streaming a text response from their completion API to the client.
    fn on_stream_completion_response_finish(
        &self,
        _prompt: &Message,
        _response: &<M as CompletionModel>::StreamingResponse,
    ) -> impl Future<Output = HookAction> + Send {
        async { HookAction::cont() }
    }
}

impl<M> PromptHook<M> for () where M: CompletionModel {}

/// Control flow action for tool call hooks. This is different from the regular [`HookAction`] in that tool call executions may be skipped for one or more reasons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallHookAction {
    /// Continue tool execution as normal.
    Continue,
    /// Skip tool execution and return the provided reason as the tool result.
    Skip { reason: String },
    /// Terminate agent loop early
    Terminate { reason: String },
}

impl ToolCallHookAction {
    /// Continue the agentic loop as normal
    pub fn cont() -> Self {
        Self::Continue
    }

    /// Skip a given tool call (with a provided reason).
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }

    /// Terminates the agentic loop entirely.
    pub fn terminate(reason: impl Into<String>) -> Self {
        Self::Terminate {
            reason: reason.into(),
        }
    }
}

/// Control flow action for hooks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookAction {
    /// Continue agentic loop execution as normal.
    Continue,
    /// Terminate agent loop early
    Terminate { reason: String },
}

impl HookAction {
    /// Continue the agentic loop as normal
    pub fn cont() -> Self {
        Self::Continue
    }

    /// Terminates the agentic loop entirely.
    pub fn terminate(reason: impl Into<String>) -> Self {
        Self::Terminate {
            reason: reason.into(),
        }
    }
}
