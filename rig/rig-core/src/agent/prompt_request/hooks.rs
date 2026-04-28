//! Optional hooks for agent prompting.
//! Hooks can be used to create custom behaviour like logging, calling external services or conditionally skipping tool calls.
//! Alternatively, you can also use them to terminate agent loops early.

use crate::{
    completion::CompletionModel,
    message::Message,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

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

    /// Called before a tool is invoked.
    ///
    /// # Returns
    /// - `ToolCallHookAction::Continue` - Allow tool execution to proceed
    /// - `ToolCallHookAction::ContinueWith { args }` - Allow tool execution with replacement arguments
    /// - `ToolCallHookAction::Skip { reason }` - Reject tool execution; `reason` will be returned to the LLM as the tool result
    /// - `ToolCallHookAction::Terminate { reason }` - Terminate the agent loop early
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
#[non_exhaustive]
pub enum ToolCallHookAction {
    /// Continue tool execution as normal.
    Continue,
    /// Continue tool execution with replacement arguments.
    ///
    /// The tool will be invoked with the provided `args` instead of the
    /// original arguments from the LLM. Downstream hooks (`on_tool_result`)
    /// and tracing spans will also see the replacement arguments.
    ContinueWith { args: String },
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

    /// Continue tool execution with replacement arguments.
    pub fn continue_with(args: impl Into<String>) -> Self {
        Self::ContinueWith { args: args.into() }
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
#[non_exhaustive]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_hook_action_continue_with() {
        let action = ToolCallHookAction::continue_with(r#"{"x": 10}"#);
        assert_eq!(
            action,
            ToolCallHookAction::ContinueWith {
                args: r#"{"x": 10}"#.to_string()
            }
        );
    }

    #[test]
    fn tool_call_hook_action_continue_with_from_string() {
        let args = String::from(r#"{"path": "/sandbox/file.txt"}"#);
        let action = ToolCallHookAction::continue_with(args.clone());
        assert_eq!(action, ToolCallHookAction::ContinueWith { args });
    }

    #[test]
    fn tool_call_hook_action_variants_are_distinct() {
        assert_ne!(
            ToolCallHookAction::cont(),
            ToolCallHookAction::continue_with("{}")
        );
        assert_ne!(
            ToolCallHookAction::continue_with("{}"),
            ToolCallHookAction::skip("{}")
        );
    }
}
