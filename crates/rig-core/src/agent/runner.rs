//! [`AgentRunner`]: the hook-aware driver that turns a sans-IO
//! [`AgentRun`] into a complete agent loop.
//!
//! [`AgentRun`] decides *what* to do next; it
//! performs no IO and carries no hooks. `AgentRunner` pairs that machine with
//! the side-effecting concerns — building and sending completion requests,
//! executing tools, loading/saving conversation memory — and fires an
//! [`AgentHook`] at every observable point. Both the blocking
//! [`PromptRequest`](crate::agent::prompt_request::PromptRequest) and the
//! [`StreamingPromptRequest`](crate::agent::prompt_request::streaming::StreamingPromptRequest)
//! APIs are thin wrappers over an `AgentRunner`, and you can build one directly
//! to drive an agent with custom, composable hooks:
//!
//! ```rust,no_run
//! # use rig_core::agent::Agent;
//! # use rig_core::completion::CompletionModel;
//! # async fn example<M: CompletionModel + 'static>(agent: Agent<M>) -> Result<(), Box<dyn std::error::Error>> {
//! let response = agent
//!     .runner("What is 2 + 2?")
//!     .max_turns(3)
//!     .run()
//!     .await?;
//! println!("{}", response.output);
//! # Ok(())
//! # }
//! ```

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use futures::StreamExt;
use tracing::{Instrument, info_span, span::Id};

use super::{
    completion::{Agent, DynamicContextStore, PreparedCompletionRequest},
    hook::{
        AgentHook, Flow, HookContext, HookStack, InvalidToolCallHookAction, RequestPatch, StepEvent,
    },
    prompt_request::{
        PromptResponse,
        streaming::{
            DriveItem, DriveStream, MultiTurnStreamItem, StreamingError, TurnSource, drive_agent,
            drive_tool_calls, record_usage_on_span, streaming_error_into_prompt,
        },
        tool_result_message, tool_result_output,
    },
    run::{
        AgentRun, DEFAULT_OUTPUT_RETRIES, ModelTurn, ModelTurnOutcome, OutputMode, PendingToolCall,
    },
};
use crate::{
    completion::{CompletionError, CompletionModel, Document, Message, PromptError},
    json_utils,
    memory::ConversationMemory,
    message::{ToolCall, ToolChoice, UserContent},
    tool::{ToolCallExtensions, server::ToolServerHandle},
};

use super::UNKNOWN_AGENT_NAME;

/// Build the per-turn `chat` span shared by both turn sources.
///
/// The span *name* must be a string literal — `tracing` bakes it into static
/// metadata — so this is a macro parameterized by the name rather than a
/// function (the two surfaces keep distinct names, `chat` vs `chat_streaming`,
/// which dashboards split on). Every other field is identical across the
/// blocking and streaming surfaces, so it lives here once instead of being
/// copy-pasted into each `TurnSource::open_chat_span`.
macro_rules! build_chat_span {
    ($runner:expr, $effective_preamble:expr, $name:literal) => {
        ::tracing::info_span!(
            target: "rig::agent_chat",
            parent: ::tracing::Span::current(),
            $name,
            gen_ai.operation.name = "chat",
            gen_ai.agent.name = $runner.agent_name_or_default(),
            gen_ai.system_instructions = $effective_preamble,
            gen_ai.provider.name = ::tracing::field::Empty,
            gen_ai.request.model = ::tracing::field::Empty,
            gen_ai.response.id = ::tracing::field::Empty,
            gen_ai.response.model = ::tracing::field::Empty,
            gen_ai.usage.output_tokens = ::tracing::field::Empty,
            gen_ai.usage.input_tokens = ::tracing::field::Empty,
            gen_ai.usage.cache_read.input_tokens = ::tracing::field::Empty,
            gen_ai.usage.cache_creation.input_tokens = ::tracing::field::Empty,
            gen_ai.usage.tool_use_prompt_tokens = ::tracing::field::Empty,
            gen_ai.usage.reasoning_tokens = ::tracing::field::Empty,
            gen_ai.input.messages = ::tracing::field::Empty,
            gen_ai.output.messages = ::tracing::field::Empty,
        )
    };
}
pub(crate) use build_chat_span;

/// Human-readable name of a [`Flow`] variant, for fail-closed diagnostics.
fn flow_name(flow: &Flow) -> &'static str {
    match flow {
        Flow::Continue => "Continue",
        Flow::Terminate { .. } => "Terminate",
        Flow::Skip { .. } => "Skip",
        Flow::RewriteArgs { .. } => "RewriteArgs",
        Flow::RewriteResult { .. } => "RewriteResult",
        Flow::PatchRequest { .. } => "PatchRequest",
        Flow::Fail => "Fail",
        Flow::Retry { .. } => "Retry",
        Flow::Repair { .. } => "Repair",
    }
}

/// Resolve a hook's [`Flow`] for an *observe-only* event — one that honors only
/// [`Flow::Continue`] and [`Flow::Terminate`].
///
/// Returns `Some(reason)` when the run must terminate, `None` to proceed. This is
/// **fail-closed and total**: any action other than `Continue`/`Terminate` is a
/// hook misuse and terminates the run with a diagnostic rather than being
/// silently dropped.
pub(crate) fn observe_flow(flow: Flow) -> Option<String> {
    match flow {
        Flow::Continue => None,
        Flow::Terminate { reason } => Some(reason),
        other => Some(format!(
            "hook returned `{}` for an observe-only event, which only honors \
             Continue/Terminate — terminating the run (fail-closed)",
            flow_name(&other)
        )),
    }
}

/// Decision for a [`StepEvent::ToolCall`] event.
pub(crate) enum ToolCallDecision {
    /// Execute the tool as normal.
    Proceed,
    /// Execute the tool with these rewritten arguments instead of the ones the
    /// model emitted.
    ProceedWith(serde_json::Value),
    /// Skip execution and return `reason` to the model as the tool result.
    Skip(String),
    /// Terminate the run.
    Terminate(String),
}

/// Resolve a hook's [`Flow`] for a [`StepEvent::ToolCall`] event (honors
/// `Continue`/`RewriteArgs`/`Skip`/`Terminate`). **Fail-closed**: any other
/// action (e.g. `Fail`/`Retry`/`Repair`) never executes the tool — it
/// terminates the run.
pub(crate) fn flow_into_tool_call(flow: Flow) -> ToolCallDecision {
    match flow {
        Flow::Continue => ToolCallDecision::Proceed,
        Flow::RewriteArgs { args } => ToolCallDecision::ProceedWith(args),
        Flow::Skip { reason } => ToolCallDecision::Skip(reason),
        Flow::Terminate { reason } => ToolCallDecision::Terminate(reason),
        other => ToolCallDecision::Terminate(format!(
            "hook returned `{}` for a tool-call event, which only honors \
             Continue/RewriteArgs/Skip/Terminate — terminating the run (fail-closed) \
             rather than executing the tool",
            flow_name(&other)
        )),
    }
}

/// Decision for a [`StepEvent::ToolResult`] event.
pub(crate) enum ToolResultDecision {
    /// Deliver the tool's actual output to the model unchanged.
    Keep,
    /// Deliver this string to the model in place of the tool's actual output.
    Replace(String),
    /// Terminate the run.
    Terminate(String),
}

/// Resolve a hook's [`Flow`] for a [`StepEvent::ToolResult`] event (honors
/// `Continue`/`RewriteResult`/`Terminate`). **Fail-closed**: any other action
/// terminates the run rather than silently delivering the tool's output.
pub(crate) fn flow_into_tool_result(flow: Flow) -> ToolResultDecision {
    match flow {
        Flow::Continue => ToolResultDecision::Keep,
        Flow::RewriteResult { result } => ToolResultDecision::Replace(result),
        Flow::Terminate { reason } => ToolResultDecision::Terminate(reason),
        other => ToolResultDecision::Terminate(format!(
            "hook returned `{}` for a tool-result event, which only honors \
             Continue/RewriteResult/Terminate — terminating the run (fail-closed)",
            flow_name(&other)
        )),
    }
}

/// Decision for a [`StepEvent::CompletionCall`] event.
pub(crate) enum CompletionCallDecision {
    /// Build and send the request as configured.
    Proceed,
    /// Build and send the request with this per-turn patch applied (the merged
    /// patch from every hook that contributed one).
    Patch(RequestPatch),
    /// Terminate the run.
    Terminate(String),
}

/// Resolve a hook's [`Flow`] for a [`StepEvent::CompletionCall`] event (honors
/// `Continue`/`PatchRequest`/`Terminate`). **Fail-closed**: any other action
/// terminates the run rather than silently sending the request. Across a
/// [`HookStack`] the `flow` is already the merged patch of every hook.
pub(crate) fn flow_into_completion_call(flow: Flow) -> CompletionCallDecision {
    match flow {
        Flow::Continue => CompletionCallDecision::Proceed,
        Flow::PatchRequest { patch } => CompletionCallDecision::Patch(patch),
        Flow::Terminate { reason } => CompletionCallDecision::Terminate(reason),
        other => CompletionCallDecision::Terminate(format!(
            "hook returned `{}` for a completion-call event, which only honors \
             Continue/PatchRequest/Terminate — terminating the run (fail-closed)",
            flow_name(&other)
        )),
    }
}

/// Decision for a [`StepEvent::InvalidToolCall`] event.
pub(crate) enum InvalidDecision {
    /// Terminate the run.
    Terminate(String),
    /// Recover via the given [`AgentRun`] action.
    Action(InvalidToolCallHookAction),
}

/// Resolve a hook's [`Flow`] for a [`StepEvent::InvalidToolCall`] event. All
/// variants are meaningful here; `Continue` preserves the documented fail-fast
/// default.
pub(crate) fn flow_into_invalid(flow: Flow) -> InvalidDecision {
    match flow {
        Flow::Terminate { reason } => InvalidDecision::Terminate(reason),
        Flow::Retry { feedback } => {
            InvalidDecision::Action(InvalidToolCallHookAction::retry(feedback))
        }
        Flow::Repair { tool_name } => {
            InvalidDecision::Action(InvalidToolCallHookAction::repair(tool_name))
        }
        Flow::Skip { reason } => InvalidDecision::Action(InvalidToolCallHookAction::skip(reason)),
        // Continue and Fail both preserve fail-fast for invalid calls.
        Flow::Continue | Flow::Fail => InvalidDecision::Action(InvalidToolCallHookAction::fail()),
        // `RewriteArgs`/`RewriteResult`/`PatchRequest` steer a *valid* call;
        // they cannot repair an unknown or disallowed one (use `Repair` to
        // rewrite the name), so they are fail-closed here.
        other @ (Flow::RewriteArgs { .. }
        | Flow::RewriteResult { .. }
        | Flow::PatchRequest { .. }) => InvalidDecision::Terminate(format!(
            "hook returned `{}` for an invalid tool-call event, which only \
                 honors Fail/Retry/Repair/Skip/Terminate — terminating the run \
                 (fail-closed)",
            flow_name(&other)
        )),
    }
}

/// A hook-aware driver over [`AgentRun`].
///
/// Construct one from an [`Agent`] with [`Agent::runner`], attach hooks with
/// [`add_hook`](Self::add_hook), then call
/// [`run`](Self::run) (blocking) or
/// [`stream`](crate::agent::prompt_request::streaming::StreamingPromptRequest)
/// (incremental). Hooks are held in a [`HookStack`], an ordered,
/// runtime-composable list; `run()` and `stream()` share the same loop and fire
/// the same events, so they behave identically apart from the streamed delta
/// events the medium adds.
#[non_exhaustive]
pub struct AgentRunner<M>
where
    M: CompletionModel,
{
    pub(crate) prompt: Message,
    pub(crate) chat_history: Option<Vec<Message>>,
    pub(crate) max_turns: usize,
    pub(crate) max_invalid_tool_call_retries: usize,
    pub(crate) model: Arc<M>,
    pub(crate) agent_name: Option<String>,
    pub(crate) preamble: Option<String>,
    pub(crate) static_context: Vec<Document>,
    pub(crate) temperature: Option<f64>,
    pub(crate) max_tokens: Option<u64>,
    pub(crate) additional_params: Option<serde_json::Value>,
    pub(crate) tool_server_handle: ToolServerHandle,
    /// Per-call runtime extensions made available to every tool executed during
    /// this run via [`Tool::call_with_extensions`](crate::tool::Tool::call_with_extensions).
    /// Empty by default; set with the [`tool_extensions`](Self::tool_extensions())
    /// builder.
    pub(crate) tool_extensions: ToolCallExtensions,
    pub(crate) dynamic_context: DynamicContextStore,
    pub(crate) tool_choice: Option<ToolChoice>,
    pub(crate) output_schema: Option<schemars::Schema>,
    pub(crate) output_mode: OutputMode,
    pub(crate) concurrency: usize,
    pub(crate) memory: Option<Arc<dyn ConversationMemory>>,
    pub(crate) conversation_id: Option<String>,
    pub(crate) hooks: HookStack<M>,
}

impl<M> AgentRunner<M>
where
    M: CompletionModel,
{
    /// Build a runner from an agent, seeding it with the agent's default hook
    /// stack. Prefer [`Agent::runner`].
    pub fn from_agent(agent: &Agent<M>, prompt: impl Into<Message>) -> Self {
        Self {
            prompt: prompt.into(),
            chat_history: None,
            max_turns: agent.default_max_turns.unwrap_or_default(),
            max_invalid_tool_call_retries: 0,
            model: agent.model.clone(),
            agent_name: agent.name.clone(),
            preamble: agent.preamble.clone(),
            static_context: agent.static_context.clone(),
            temperature: agent.temperature,
            max_tokens: agent.max_tokens,
            additional_params: agent.additional_params.clone(),
            tool_server_handle: agent.tool_server_handle.clone(),
            tool_extensions: ToolCallExtensions::new(),
            dynamic_context: agent.dynamic_context.clone(),
            tool_choice: agent.tool_choice.clone(),
            output_schema: agent.output_schema.clone(),
            output_mode: agent.output_mode.clone(),
            concurrency: 1,
            memory: agent.memory.clone(),
            conversation_id: agent.default_conversation_id.clone(),
            hooks: agent.hooks.clone(),
        }
    }

    /// Append a hook to the stack (on top of any the agent already carries).
    /// Hooks run in registration order; how their results compose is
    /// event-dependent (`CompletionCall` request patches accumulate and merge,
    /// `ToolCall`/`ToolResult` rewrites chain, and only observe-only/recovery
    /// events use first-non-[`Flow::Continue`]-wins). See the
    /// [`hook`](crate::agent::hook) module docs.
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook<M> + 'static,
    {
        self.hooks.push(hook);
        self
    }
}

impl<M> AgentRunner<M>
where
    M: CompletionModel,
{
    /// Set the maximum multi-turn depth (tool-calling rounds before a final
    /// answer). Exceeding it returns [`PromptError::MaxTurnsError`].
    pub fn max_turns(mut self, depth: usize) -> Self {
        self.max_turns = depth;
        self
    }

    /// Set the per-call runtime [`ToolCallExtensions`] for this run.
    ///
    /// The extensions are threaded to every tool the agent executes, so tools
    /// can read caller-provided values (auth tokens, session IDs, conversation
    /// state, …) via [`Tool::call_with_extensions`](crate::tool::Tool::call_with_extensions)
    /// without the model ever seeing them. Replaces any extensions already set.
    pub fn tool_extensions(mut self, extensions: ToolCallExtensions) -> Self {
        self.tool_extensions = extensions;
        self
    }

    /// Set the chat history preceding the prompt. Passing explicit history
    /// bypasses conversation memory for this run.
    pub fn history<I, T>(mut self, history: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<Message>,
    {
        self.chat_history = Some(history.into_iter().map(Into::into).collect());
        self
    }

    /// Execute up to `concurrency` tools at once (1 by default). Applies to
    /// **both** the blocking [`run`](Self::run) and the streaming
    /// [`stream`](Self::stream) paths.
    ///
    /// The resulting message history is the same in both paths regardless of
    /// `concurrency`: final tool results are persisted in tool-call order. At
    /// the default `concurrency` of 1 the two paths are fully in lock-step; with
    /// `concurrency > 1` the tools run in parallel, so a `ToolCall`/`ToolResult`
    /// **hook may fire in completion order** rather than call order — the
    /// per-tool side effects interleave even though the final history does not.
    ///
    /// For the streaming path there is one additional caveat: at `concurrency >
    /// 1` the driver emits *all* of a turn's `ToolCall` stream items eagerly (in
    /// call order) and then emits each `ToolResult` stream item as its tool
    /// finishes, which may be completion order rather than call order. The
    /// persisted message history is unchanged.
    ///
    /// A `concurrency` of 0 is clamped to 1; `0` and `1` both run a turn's tools
    /// sequentially (the `buffer_unordered` path is used only at `concurrency > 1`).
    pub fn tool_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Set the conversation id used to load and persist memory for this run.
    pub fn conversation(mut self, id: impl Into<String>) -> Self {
        self.conversation_id = Some(id.into());
        self
    }

    /// Disable conversation memory for this run (no load, no save).
    pub fn without_memory(mut self) -> Self {
        self.memory = None;
        self.conversation_id = None;
        self
    }

    /// Set the retry budget for invalid tool-call recovery. Invalid tool-call
    /// retries also consume normal multi-turn depth.
    pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
        self.max_invalid_tool_call_retries = retries;
        self
    }

    pub(crate) fn agent_name_or_default(&self) -> &str {
        self.agent_name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }

    /// Build the sans-IO [`AgentRun`] for this runner's configuration.
    /// `history_override` replaces the configured chat history (e.g. with
    /// memory-loaded history). Delegates to [`build_agent_run`] — the single
    /// construction site shared with the streaming driver.
    pub(crate) fn build_run(&self, history_override: Option<Vec<Message>>) -> AgentRun {
        build_agent_run(
            self.prompt.clone(),
            self.max_turns,
            self.max_invalid_tool_call_retries,
            self.output_schema.as_ref(),
            history_override.or_else(|| self.chat_history.clone()),
            self.tool_choice.clone(),
        )
    }
}

/// Construct an [`AgentRun`] from explicit run configuration. The single place a
/// run is built, so the blocking and streaming drivers configure runs
/// identically.
pub(crate) fn build_agent_run(
    prompt: Message,
    max_turns: usize,
    max_invalid_tool_call_retries: usize,
    output_schema: Option<&schemars::Schema>,
    history: Option<Vec<Message>>,
    tool_choice: Option<ToolChoice>,
) -> AgentRun {
    let mut run = AgentRun::new(prompt)
        .max_turns(max_turns)
        .max_invalid_tool_call_retries(max_invalid_tool_call_retries)
        .with_output_validation(
            output_schema.map(|schema| schema.as_value().clone()),
            DEFAULT_OUTPUT_RETRIES,
        );
    if let Some(history) = history {
        run = run.with_history(history);
    }
    if let Some(tool_choice) = tool_choice {
        run = run.with_tool_choice(tool_choice);
    }
    run
}

/// Build (or adopt) the top-level `invoke_agent` span for a run, shared by the
/// blocking and streaming drivers so the run-level span shape is defined once.
///
/// Returns the span plus whether it was newly created. When the caller is
/// already inside a span we adopt it and report `false`, so the driver can avoid
/// recording run-level usage onto a span it does not own (see the
/// `created_agent_span` guard in both drivers' `Done` handling).
pub(crate) fn acquire_agent_span(
    agent_name: &str,
    preamble: Option<&str>,
) -> (tracing::Span, bool) {
    if tracing::Span::current().is_disabled() {
        let span = info_span!(
            "invoke_agent",
            gen_ai.operation.name = "invoke_agent",
            gen_ai.agent.name = agent_name,
            gen_ai.system_instructions = preamble,
            gen_ai.prompt = tracing::field::Empty,
            gen_ai.completion = tracing::field::Empty,
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
            gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
            gen_ai.usage.tool_use_prompt_tokens = tracing::field::Empty,
            gen_ai.usage.reasoning_tokens = tracing::field::Empty,
        );
        (span, true)
    } else {
        (tracing::Span::current(), false)
    }
}

/// Outcome of firing the `CompletionCall` hook for a turn.
pub(crate) enum CompletionCallOutcome {
    /// Proceed, optionally applying a per-turn request patch (the merged patch
    /// from every hook that contributed one).
    Proceed(Option<RequestPatch>),
    /// Terminate the run with this reason.
    Terminate(String),
}

/// Fire the `CompletionCall` hook for a turn and resolve its [`Flow`]
/// (fail-closed). Shared by the blocking and streaming drivers so this steering
/// event fires identically on both; each driver surfaces `Terminate` in its own
/// medium (a returned `Err` vs. a yielded error item). Across a [`HookStack`]
/// the resolved flow is the merged patch of every contributing hook.
pub(crate) async fn resolve_completion_call<M>(
    hooks: &HookStack<M>,
    ctx: &HookContext,
    prompt: &Message,
    history: &[Message],
    turn: usize,
) -> CompletionCallOutcome
where
    M: CompletionModel,
{
    match flow_into_completion_call(
        hooks
            .on_event(
                ctx,
                StepEvent::CompletionCall {
                    prompt,
                    history,
                    turn,
                },
            )
            .await,
    ) {
        CompletionCallDecision::Terminate(reason) => CompletionCallOutcome::Terminate(reason),
        CompletionCallDecision::Patch(patch) => CompletionCallOutcome::Proceed(Some(patch)),
        CompletionCallDecision::Proceed => CompletionCallOutcome::Proceed(None),
    }
}

/// Append a finished run's messages to conversation memory, logging and
/// proceeding on failure. Shared `Done`-arm behavior for both drivers.
pub(crate) async fn append_run_messages(
    memory_handle: Option<&(Arc<dyn ConversationMemory>, String)>,
    messages: &[Message],
) {
    // Clone into an owned vec only when there is a backend to append to — the
    // common no-memory path pays nothing.
    if let Some((memory, id)) = memory_handle
        && let Err(err) = memory.append(id, messages.to_vec()).await
    {
        tracing::warn!(
            error = %err,
            conversation_id = %id,
            "conversation memory append failed; surfacing final response anyway"
        );
    }
}

/// Outcome of [`run_single_tool`]: the tool-result content plus whether the
/// tool's body actually ran.
pub(crate) struct ToolCallOutcome {
    /// The tool result delivered to the model (a real output, a redacted
    /// replacement, or a hook skip reason).
    pub content: UserContent,
    /// Whether the tool's `call` actually executed. `false` when a `ToolCall`
    /// hook returned [`Flow::Skip`](crate::agent::Flow::Skip), so the driver
    /// surfaces no [`ToolExecutionStart`](crate::agent::prompt_request::streaming::MultiTurnStreamItem::ToolExecutionStart)
    /// event for it (nothing actually ran).
    pub executed: bool,
}

/// Execute a single tool call, firing the `ToolCall` and `ToolResult` hooks and
/// shaping the result. **Shared by the blocking and streaming drivers** so a
/// tool call behaves identically in both: same hook events, same fail-closed
/// skip/terminate handling, and the same result shaping — a hook skip reason is
/// emitted verbatim ([`tool_result_message`]) while a real tool output is parsed
/// ([`tool_result_output`]). Records `gen_ai.tool.*` on the current span;
/// `error_history` builds a cancellation error if a hook terminates the run.
/// Returns whether the tool body executed via [`ToolCallOutcome::executed`].
pub(crate) async fn run_single_tool<M>(
    hooks: &HookStack<M>,
    ctx: &HookContext,
    tool_server: &ToolServerHandle,
    tool_extensions: &ToolCallExtensions,
    tool_call: &ToolCall,
    internal_call_id: &str,
    error_history: &[Message],
) -> Result<ToolCallOutcome, PromptError>
where
    M: CompletionModel,
{
    let tool_name = &tool_call.function.name;
    // `mut` so a `Flow::RewriteArgs` hook can rewrite the arguments the tool
    // runs with (the model's emitted arguments are otherwise used verbatim).
    let mut args = json_utils::value_to_json_string(&tool_call.function.arguments);

    let tool_span = tracing::Span::current();
    tool_span.record("gen_ai.tool.name", tool_name);
    tool_span.record("gen_ai.tool.call.id", &tool_call.id);
    tool_span.record("gen_ai.tool.call.arguments", &args);

    match flow_into_tool_call(
        hooks
            .on_event(
                ctx,
                StepEvent::ToolCall {
                    tool_name,
                    tool_call_id: tool_call.call_id.as_deref(),
                    internal_call_id,
                    args: &args,
                },
            )
            .await,
    ) {
        ToolCallDecision::Terminate(reason) => {
            return Err(PromptError::prompt_cancelled(
                error_history.to_vec(),
                reason,
            ));
        }
        ToolCallDecision::Skip(reason) => {
            tracing::info!(tool_name = tool_name, reason = reason, "Tool call rejected");
            // Synthetic rejection message: emit verbatim, never re-parsed. The
            // tool did not run, so `executed` is false (no execution-start event).
            return Ok(ToolCallOutcome {
                content: tool_result_message(
                    tool_call.id.clone(),
                    tool_call.call_id.clone(),
                    reason,
                ),
                executed: false,
            });
        }
        ToolCallDecision::ProceedWith(replacement) => {
            // Run the tool with the hook's rewritten arguments. Re-record the
            // span so the trace, and the downstream `ToolResult` event, reflect
            // what the tool actually received rather than what the model emitted.
            args = json_utils::value_to_json_string(&replacement);
            tool_span.record("gen_ai.tool.call.arguments", &args);
            tracing::debug!(
                tool_name = tool_name,
                "tool-call arguments rewritten by a hook"
            );
        }
        ToolCallDecision::Proceed => {}
    }

    let output = match tool_server
        .call_tool_with_extensions(tool_name, &args, tool_extensions)
        .await
    {
        Ok(res) => res,
        Err(e) => {
            tracing::warn!("Error while executing tool: {e}");
            e.to_string()
        }
    };

    // The tool's raw output is deliberately NOT recorded on the span yet: a
    // `ToolResult` hook may redact it. Recording is deferred until after the hook
    // runs — the redacted replacement on `Replace`, the raw output on `Keep`, and
    // nothing on `Terminate` — so a redaction guardrail never leaks the original
    // via the trace. (OpenAI Agents applies tool-output guardrails before
    // tracing / tool-end / model-visible output for the same reason.) The hook
    // still observes the tool's actual output via the event's `result` field.
    match flow_into_tool_result(
        hooks
            .on_event(
                ctx,
                StepEvent::ToolResult {
                    tool_name,
                    tool_call_id: tool_call.call_id.as_deref(),
                    internal_call_id,
                    // The first result hook observes the tool's actual output,
                    // before any `RewriteResult` replacement is applied below.
                    args: &args,
                    result: &output,
                },
            )
            .await,
    ) {
        ToolResultDecision::Terminate(reason) => {
            // Do not record or log the raw output: the model never sees it (the
            // run is terminating) and a result hook may have terminated to prevent
            // exactly that leak.
            tracing::info!(
                "executed tool {tool_name} with args {args}; run terminated by a result hook"
            );
            Err(PromptError::prompt_cancelled(
                error_history.to_vec(),
                reason,
            ))
        }
        ToolResultDecision::Replace(replacement) => {
            // The hook replaced the model-visible result. Record the replacement
            // (the raw output was never recorded on the span before the hook ran)
            // and log only that a rewrite happened — never the tool's raw output —
            // so a redaction hook does not leak the original via the trace or the
            // log. The replacement is hook-supplied content, so it is delivered
            // verbatim (like a `Skip` reason via [`tool_result_message`]) rather
            // than re-parsed as tool output, which would let a JSON-shaped
            // replacement be reinterpreted as a structured/multimodal result.
            tool_span.record("gen_ai.tool.call.result", &replacement);
            tracing::info!(
                "executed tool {tool_name} with args {args}; result rewritten by a hook"
            );
            Ok(ToolCallOutcome {
                content: tool_result_message(
                    tool_call.id.clone(),
                    tool_call.call_id.clone(),
                    replacement,
                ),
                executed: true,
            })
        }
        ToolResultDecision::Keep => {
            // No redaction requested: now that the hook has run without replacing
            // the output, record the tool's real result on the span.
            tool_span.record("gen_ai.tool.call.result", &output);
            tracing::info!("executed tool {tool_name} with args {args}. result: {output}");
            // Real tool output: parsed (may be multimodal).
            Ok(ToolCallOutcome {
                content: tool_result_output(
                    tool_call.id.clone(),
                    tool_call.call_id.clone(),
                    output,
                ),
                executed: true,
            })
        }
    }
}

/// Build the per-tool `execute_tool` span carrying the `gen_ai.tool.*` fields
/// that [`run_single_tool`] records on the current span. Parented to the
/// contextual current span; the blocking driver additionally chains it via
/// `follows_from`, while the streaming driver uses it as-is. Shared by both
/// drivers so the span shape stays defined in one place.
pub(crate) fn new_execute_tool_span() -> tracing::Span {
    info_span!(
        "execute_tool",
        gen_ai.operation.name = "execute_tool",
        gen_ai.tool.type = "function",
        gen_ai.tool.name = tracing::field::Empty,
        gen_ai.tool.call.id = tracing::field::Empty,
        gen_ai.tool.call.arguments = tracing::field::Empty,
        gen_ai.tool.call.result = tracing::field::Empty
    )
}

/// [`TurnSource`] for the blocking surface: each turn issues a unary
/// `model.completion()` request and feeds the whole response into the machine.
/// Emits no intermediate items (the blocking surface folds the engine to its
/// final response), but keeps the blocking driver's linear `follows_from` span
/// chain across chat and tool spans.
pub(crate) struct UnaryTurnSource {
    /// Sequences chat and tool spans into a linear `follows_from` chain (the
    /// streaming surface parents into a tree instead and does not chain).
    ///
    /// Atomic rather than `Cell` despite being driven by a single sequential
    /// task: `run_tool_calls` passes `chain_span` as a closure into
    /// `drive_tool_calls`, whose returned `DriveStream` is `Send`. That makes the
    /// closure capture `&self`, so `&UnaryTurnSource` must be `Send`, i.e.
    /// `UnaryTurnSource: Sync` — which `AtomicU64` provides and `Cell` does not.
    current_span_id: AtomicU64,
}

impl UnaryTurnSource {
    pub(crate) fn new() -> Self {
        Self {
            current_span_id: AtomicU64::new(0),
        }
    }

    /// Chain `span` onto the previous step's span and record it as the new chain
    /// head, preserving the blocking driver's linear causal trace.
    fn chain_span(&self, span: tracing::Span) -> tracing::Span {
        let span = match self.current_span_id.load(Ordering::Relaxed) {
            0 => span,
            id => span.follows_from(Id::from_u64(id)).to_owned(),
        };
        if let Some(id) = span.id() {
            self.current_span_id.store(id.into_u64(), Ordering::Relaxed);
        }
        span
    }
}

impl<M> TurnSource<M> for UnaryTurnSource
where
    M: CompletionModel,
{
    type Raw = M::Response;

    fn open_chat_span(
        &self,
        runner: &AgentRunner<M>,
        effective_preamble: Option<&str>,
    ) -> tracing::Span {
        let chat_span = build_chat_span!(runner, effective_preamble, "chat");
        self.chain_span(chat_span)
    }

    fn run_model_turn<'a>(
        &'a mut self,
        runner: &'a AgentRunner<M>,
        hook_ctx: &'a HookContext,
        run: &'a mut AgentRun,
        prepared: PreparedCompletionRequest<M>,
        chat_span: tracing::Span,
        _agent_span: &'a tracing::Span,
        current_prompt: Message,
    ) -> DriveStream<'a, M::Response> {
        Box::pin(async_stream::stream! {
            let resp = match prepared.builder.send().instrument(chat_span.clone()).await {
                Ok(resp) => resp,
                Err(err) => {
                    yield Err(StreamingError::from(err));
                    return;
                }
            };

            let mut outcome = match run.model_response(ModelTurn::new(
                resp.message_id.clone(),
                resp.choice.clone(),
                resp.usage,
                prepared.executable_tool_names,
                prepared.allowed_tool_names,
            )) {
                Ok(outcome) => outcome,
                Err(err) => {
                    yield Err(Box::new(err).into());
                    return;
                }
            };

            loop {
                match outcome {
                    ModelTurnOutcome::NeedsResolution(context) => {
                        let flow = runner
                            .hooks
                            .on_event(hook_ctx, StepEvent::InvalidToolCall(&context))
                            .await;
                        match flow_into_invalid(flow) {
                            InvalidDecision::Terminate(reason) => {
                                yield Err(StreamingError::Prompt(Box::new(
                                    run.cancel_error(reason),
                                )));
                                return;
                            }
                            InvalidDecision::Action(action) => {
                                outcome = match run.resolve_invalid_tool_call(action) {
                                    Ok(outcome) => outcome,
                                    Err(err) => {
                                        yield Err(Box::new(err).into());
                                        return;
                                    }
                                };
                            }
                        }
                    }
                    ModelTurnOutcome::TurnRetried => break,
                    ModelTurnOutcome::Continue {
                        response_hook_suppressed,
                    } => {
                        if !response_hook_suppressed {
                            // The medium-specific raw response event fires first,
                            // then the normalized per-turn event. Both are
                            // observe-only and suppressed for recovered turns.
                            if let Some(reason) = observe_flow(
                                runner
                                    .hooks
                                    .on_event(hook_ctx, StepEvent::CompletionResponse {
                                        prompt: &current_prompt,
                                        response: &resp,
                                    })
                                    .await,
                            ) {
                                yield Err(StreamingError::Prompt(Box::new(run.cancel_error(reason))));
                                return;
                            }
                            if let Some(reason) = observe_flow(
                                runner
                                    .hooks
                                    .on_event(hook_ctx, StepEvent::ModelTurnFinished {
                                        turn: hook_ctx.turn(),
                                        content: &resp.choice,
                                        usage: resp.usage,
                                    })
                                    .await,
                            ) {
                                yield Err(StreamingError::Prompt(Box::new(run.cancel_error(reason))));
                                return;
                            }
                        }
                        break;
                    }
                }
            }
        })
    }

    fn run_tool_calls<'a>(
        &'a self,
        runner: &'a AgentRunner<M>,
        hook_ctx: &'a HookContext,
        run: &'a mut AgentRun,
        calls: Vec<PendingToolCall>,
    ) -> DriveStream<'a, M::Response> {
        // The blocking surface chains tool spans into its linear `follows_from`
        // sequence (chat -> tool -> chat), and discards the yielded items, so it
        // skips building them.
        drive_tool_calls(
            runner,
            hook_ctx,
            run,
            calls,
            |span| self.chain_span(span),
            false,
        )
    }

    fn record_run_level_telemetry(
        &self,
        agent_span: &tracing::Span,
        response: &PromptResponse,
        created_agent_span: bool,
    ) {
        // Record run-level completion + usage onto the agent span, but only when
        // we created it — never pollute a caller-supplied outer span. The usage
        // fields go through the same recorder the streaming surface uses; the
        // blocking surface additionally records the final completion text.
        if created_agent_span {
            agent_span.record("gen_ai.completion", &response.output);
            record_usage_on_span(agent_span, response.usage);
        }
    }

    fn final_item(&self, _response: &PromptResponse) -> Option<MultiTurnStreamItem<M::Response>> {
        // The blocking surface folds the engine and discards the final item, so
        // building it (an extra full-response clone) is skipped entirely.
        None
    }
}

impl<M> AgentRunner<M>
where
    M: CompletionModel,
{
    /// Drive the agent loop to completion, returning the aggregated
    /// [`PromptResponse`]. Hooks fire at every observable point; the first hook
    /// to terminate cancels the run.
    pub async fn run(self) -> Result<PromptResponse, PromptError> {
        let (agent_span, created_agent_span) =
            acquire_agent_span(self.agent_name_or_default(), self.preamble.as_deref());

        if let Some(text) = self.prompt.rag_text() {
            agent_span.record("gen_ai.prompt", text);
        }

        // When the caller passes explicit history, memory is fully bypassed for
        // this run (no load AND no save). Otherwise, if a memory backend and
        // conversation id are both configured, load prior history.
        let (history_override, memory_handle) = match &self.chat_history {
            Some(_) => (None, None),
            None => match (&self.memory, &self.conversation_id) {
                (Some(memory), Some(id)) => {
                    let loaded = memory.load(id).await?;
                    (Some(loaded), Some((memory.clone(), id.clone())))
                }
                _ => (None, None),
            },
        };

        let run = self.build_run(history_override);

        // Fold the shared engine to its final response. The blocking surface
        // uses a unary model transport and ignores the intermediate items the
        // engine yields; the engine is driven under the caller's ambient span
        // (no `instrument`), keeping the agent span detached and the chat/tool
        // spans on the blocking `follows_from` chain.
        let driver = drive_agent(
            self,
            UnaryTurnSource::new(),
            run,
            agent_span,
            created_agent_span,
            memory_handle,
            false,
        );
        futures::pin_mut!(driver);

        let mut response = None;
        while let Some(item) = driver.next().await {
            match item {
                Ok(DriveItem::Done(done)) => response = Some(*done),
                Ok(DriveItem::Item(_)) => {}
                Err(err) => return Err(streaming_error_into_prompt(err)),
            }
        }

        // The engine yields `Done` unless it errored (handled above).
        response.ok_or_else(|| {
            PromptError::CompletionError(CompletionError::ResponseError(
                "agent run ended without producing a final response".to_string(),
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering::SeqCst},
    };

    use futures::StreamExt;
    use serde_json::json;

    use crate::agent::AgentBuilder;
    use crate::agent::hook::{
        AgentHook, Flow, HookContext, RequestPatch, StepEvent, StepEventKind,
    };
    use crate::agent::prompt_request::streaming::{MultiTurnStreamItem, StreamingError};
    use crate::agent::run::OutputMode;
    use crate::completion::{CompletionModel, Message, PromptError, ToolDefinition};
    use crate::message::{AssistantContent, ToolCall, ToolChoice, ToolFunction, UserContent};
    use crate::streaming::{StreamedAssistantContent, StreamedUserContent};
    use crate::test_utils::{
        MockAddTool, MockBarrierTool, MockCompletionModel, MockOperationArgs, MockStreamEvent,
        MockSubtractTool, MockToolError, MockTurn,
    };
    use crate::tool::Tool;

    /// Records the kind of every hook event (and every tool-result payload) so a
    /// run() and a stream() of the same scenario can be compared.
    #[derive(Clone, Default)]
    struct RecordingHook {
        events: Arc<Mutex<Vec<StepEventKind>>>,
        tool_results: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingHook {
        /// Event kinds that should be identical across streaming and
        /// non-streaming (excludes the medium-specific delta / response-finish
        /// events).
        fn shared_events(&self) -> Vec<StepEventKind> {
            self.events
                .lock()
                .expect("events lock")
                .iter()
                .copied()
                .filter(|kind| {
                    matches!(
                        kind,
                        StepEventKind::CompletionCall
                            | StepEventKind::ToolCall
                            | StepEventKind::ToolResult
                            | StepEventKind::InvalidToolCall
                    )
                })
                .collect()
        }

        fn tool_results(&self) -> Vec<String> {
            self.tool_results.lock().expect("results lock").clone()
        }

        /// Count of a single event kind across the whole run, including the
        /// medium-specific response-finish events that `shared_events` excludes.
        fn count(&self, kind: StepEventKind) -> usize {
            self.events
                .lock()
                .expect("events lock")
                .iter()
                .filter(|recorded| **recorded == kind)
                .count()
        }
    }

    impl<M: CompletionModel> AgentHook<M> for RecordingHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            self.events.lock().expect("events lock").push(event.kind());
            if let StepEvent::ToolResult { result, .. } = event {
                self.tool_results
                    .lock()
                    .expect("results lock")
                    .push(result.to_string());
            }
            Flow::cont()
        }
    }

    fn blocking_model() -> MockCompletionModel {
        MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
            MockTurn::text("the answer is 5"),
        ])
    }

    fn streaming_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_name_delta("tc1", "ic1", "add"),
                MockStreamEvent::tool_call_arguments_delta("tc1", "ic1", "{\"x\":2,\"y\":3}"),
                MockStreamEvent::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("the answer is 5"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ])
    }

    /// run() and stream() of the same tool-calling scenario produce the same
    /// final output, the same final message history, the same tool-result
    /// content, and the same medium-independent hook event sequence.
    #[tokio::test]
    async fn run_and_stream_behave_identically_for_a_tool_call() {
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model())
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(blocking_hook.clone())
            .run()
            .await
            .expect("blocking run should succeed");

        // No `.with_history` on either runner — `stream()` must return the final
        // history just like `run()` returns `messages`.
        let streaming_hook = RecordingHook::default();
        let mut stream = AgentBuilder::new(streaming_model())
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(streaming_hook.clone())
            .stream()
            .await;

        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
                item.map_err(|err| panic!("stream item errored: {err}"))
            {
                final_response = Some(resp);
            }
        }
        let final_response = final_response.expect("stream should yield a final response");

        // Same final output.
        assert_eq!(blocking.output, "the answer is 5");
        assert_eq!(final_response.response(), blocking.output);

        // Same medium-independent hook event sequence (model call, tool call,
        // tool result, second model call).
        assert_eq!(
            blocking_hook.shared_events(),
            streaming_hook.shared_events()
        );
        assert_eq!(
            blocking_hook.shared_events(),
            vec![
                StepEventKind::CompletionCall,
                StepEventKind::ToolCall,
                StepEventKind::ToolResult,
                StepEventKind::CompletionCall,
            ]
        );

        // Same tool-result content seen by the hook.
        assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());
        assert_eq!(blocking_hook.tool_results(), vec!["5".to_string()]);

        // Same final message history (compared via serialized form to normalize).
        let blocking_messages = blocking.messages.expect("blocking messages");
        let streaming_messages = final_response
            .history()
            .expect("streaming history")
            .to_vec();
        assert_eq!(
            serde_json::to_value(&blocking_messages).expect("serialize blocking"),
            serde_json::to_value(&streaming_messages).expect("serialize streaming"),
        );
    }

    /// Safety net for the streaming/non-streaming unification: pins the blocking
    /// driver's span topology (span name, `invoke_agent` creation, the
    /// `follows_from` chain, and `created_agent_span`-gated run-level usage) so a
    /// later refactor onto a shared engine cannot silently drift it. The
    /// streaming side is already pinned by `assert_stream_usage_recorded_on_chat_spans`.
    mod span_safety_net {
        use std::collections::{HashMap, HashSet};
        use std::sync::{Arc, Mutex};

        use tracing::Instrument;
        use tracing::field::{Field, Visit};
        use tracing::span::{Attributes, Record};
        use tracing::{Id, Subscriber};
        use tracing_subscriber::layer::{Context, SubscriberExt};
        use tracing_subscriber::{Layer, Registry, registry::LookupSpan};

        use crate::agent::AgentBuilder;
        use crate::completion::Usage;
        use crate::test_utils::{MockAddTool, MockCompletionModel, MockTurn};

        #[derive(Clone)]
        struct CapturedSpan {
            id: u64,
            name: String,
            field_names: HashSet<String>,
            u64_fields: HashMap<String, u64>,
        }

        #[derive(Clone, Default)]
        struct Captured {
            spans: Arc<Mutex<Vec<CapturedSpan>>>,
            /// `(span, follows_from)` pairs recorded via `Span::follows_from`.
            follows: Arc<Mutex<Vec<(u64, u64)>>>,
        }

        impl Captured {
            fn insert(&self, id: &Id, name: &str) {
                self.spans.lock().expect("spans").push(CapturedSpan {
                    id: id.into_u64(),
                    name: name.to_string(),
                    field_names: HashSet::new(),
                    u64_fields: HashMap::new(),
                });
            }

            fn record(&self, id: &Id, names: HashSet<String>, u64s: HashMap<String, u64>) {
                let id = id.into_u64();
                if let Ok(mut spans) = self.spans.lock()
                    && let Some(span) = spans.iter_mut().find(|s| s.id == id)
                {
                    span.field_names.extend(names);
                    span.u64_fields.extend(u64s);
                }
            }

            fn follows_from(&self, span: &Id, follows: &Id) {
                self.follows
                    .lock()
                    .expect("follows")
                    .push((span.into_u64(), follows.into_u64()));
            }

            fn clear(&self) {
                self.spans.lock().expect("spans").clear();
                self.follows.lock().expect("follows").clear();
            }

            fn snapshot(&self) -> Vec<CapturedSpan> {
                self.spans.lock().expect("spans").clone()
            }

            fn follows_edges(&self) -> Vec<(u64, u64)> {
                self.follows.lock().expect("follows").clone()
            }
        }

        struct CaptureLayer {
            captured: Captured,
        }

        impl<S> Layer<S> for CaptureLayer
        where
            S: Subscriber + for<'l> LookupSpan<'l>,
        {
            fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, _ctx: Context<'_, S>) {
                self.captured.insert(id, attrs.metadata().name());
            }

            fn on_record(&self, span: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
                let mut visitor = FieldVisitor::default();
                values.record(&mut visitor);
                self.captured.record(span, visitor.names, visitor.u64s);
            }

            fn on_follows_from(&self, span: &Id, follows: &Id, _ctx: Context<'_, S>) {
                self.captured.follows_from(span, follows);
            }
        }

        #[derive(Default)]
        struct FieldVisitor {
            names: HashSet<String>,
            u64s: HashMap<String, u64>,
        }

        impl Visit for FieldVisitor {
            fn record_u64(&mut self, field: &Field, value: u64) {
                self.names.insert(field.name().to_string());
                self.u64s.insert(field.name().to_string(), value);
            }

            fn record_str(&mut self, field: &Field, _value: &str) {
                self.names.insert(field.name().to_string());
            }

            fn record_debug(&mut self, field: &Field, _value: &dyn std::fmt::Debug) {
                self.names.insert(field.name().to_string());
            }
        }

        fn usage(input: u64, output: u64) -> Usage {
            Usage {
                input_tokens: input,
                output_tokens: output,
                ..Usage::new()
            }
        }

        /// Two-turn tool scenario: the blocking driver emits chat -> execute_tool
        /// -> chat, exercising the `follows_from` chain.
        fn tool_then_text_model() -> MockCompletionModel {
            MockCompletionModel::from_turns([
                MockTurn::tool_call("tc1", "add", serde_json::json!({"x": 2, "y": 3}))
                    .with_usage(usage(7, 11)),
                MockTurn::text("the answer is 5").with_usage(usage(13, 17)),
            ])
        }

        /// Register the blocking driver's span callsites against the scoped
        /// subscriber before asserting, mirroring the streaming usage test's
        /// interest-cache warm-up (a foreign thread without our subscriber can
        /// otherwise cache `Interest::never` for these callsites).
        async fn warm_blocking_callsites() {
            let agent = AgentBuilder::new(tool_then_text_model())
                .tool(MockAddTool)
                .build();
            let _ = agent.runner("add 2 and 3").max_turns(3).run().await;
        }

        #[tokio::test]
        async fn run_records_usage_and_chains_chat_spans_on_a_created_agent_span() {
            let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
            let captured = Captured::default();
            let subscriber = Registry::default().with(CaptureLayer {
                captured: captured.clone(),
            });
            let _default = tracing::subscriber::set_default(subscriber);

            warm_blocking_callsites().await;
            tracing::callsite::rebuild_interest_cache();
            captured.clear();

            let agent = AgentBuilder::new(tool_then_text_model())
                .tool(MockAddTool)
                .build();
            let response = agent
                .runner("add 2 and 3")
                .max_turns(3)
                .run()
                .await
                .expect("blocking run should succeed");
            assert_eq!(response.output, "the answer is 5");

            let spans = captured.snapshot();

            // The blocking chat span is named "chat" (NOT "chat_streaming").
            let chat_spans: Vec<&CapturedSpan> =
                spans.iter().filter(|s| s.name == "chat").collect();
            assert_eq!(chat_spans.len(), 2, "two model turns -> two chat spans");
            assert!(
                spans.iter().all(|s| s.name != "chat_streaming"),
                "blocking driver must not emit chat_streaming spans"
            );

            // A run with no ambient span creates its own invoke_agent span...
            let agent_span = spans
                .iter()
                .find(|s| s.name == "invoke_agent")
                .expect("blocking run should create an invoke_agent span");

            // ...and records aggregate usage + completion onto it (created_agent_span).
            assert_eq!(
                agent_span.u64_fields.get("gen_ai.usage.input_tokens"),
                Some(&(7 + 13)),
            );
            assert_eq!(
                agent_span.u64_fields.get("gen_ai.usage.output_tokens"),
                Some(&(11 + 17)),
            );
            assert!(
                agent_span.field_names.contains("gen_ai.completion"),
                "the created agent span records the final completion text"
            );

            // The blocking driver links chat/tool spans into a linear
            // follows_from chain (chat#1 -> execute_tool -> chat#2); the
            // streaming driver does not, so this is a blocking-only invariant the
            // unification must keep.
            let tool_span = spans
                .iter()
                .find(|s| s.name == "execute_tool")
                .expect("tool turn should emit an execute_tool span");
            let edges = captured.follows_edges();
            assert!(
                edges.contains(&(tool_span.id, chat_spans[0].id)),
                "execute_tool should follow_from the first chat span; edges={edges:?}"
            );
            assert!(
                edges.contains(&(chat_spans[1].id, tool_span.id)),
                "the second chat span should follow_from execute_tool; edges={edges:?}"
            );
        }

        #[tokio::test]
        async fn run_does_not_record_usage_onto_a_caller_supplied_outer_span() {
            let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
            let captured = Captured::default();
            let subscriber = Registry::default().with(CaptureLayer {
                captured: captured.clone(),
            });
            let _default = tracing::subscriber::set_default(subscriber);

            warm_blocking_callsites().await;
            tracing::callsite::rebuild_interest_cache();
            captured.clear();

            // Declare the fields the guard protects so a regression (recording
            // onto a caller span) is actually observable rather than a silent
            // no-op on an undeclared field.
            let outer = tracing::info_span!(
                "outer",
                gen_ai.completion = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
            );
            async {
                let agent = AgentBuilder::new(tool_then_text_model())
                    .tool(MockAddTool)
                    .build();
                agent
                    .runner("add 2 and 3")
                    .max_turns(3)
                    .run()
                    .await
                    .expect("blocking run should succeed");
            }
            .instrument(outer)
            .await;

            let spans = captured.snapshot();
            // Under an ambient span the driver adopts it; no invoke_agent is created.
            assert!(
                spans.iter().all(|s| s.name != "invoke_agent"),
                "an ambient outer span should be adopted, not wrapped in invoke_agent"
            );
            let outer_span = spans
                .iter()
                .find(|s| s.name == "outer")
                .expect("outer span should be captured");
            assert!(
                outer_span
                    .field_names
                    .iter()
                    .all(|name| !name.starts_with("gen_ai.usage.")),
                "run-level usage must not be recorded onto a caller-supplied outer span"
            );
            assert!(
                !outer_span.field_names.contains("gen_ai.completion"),
                "run-level completion must not be recorded onto a caller-supplied outer span"
            );
        }

        // --- Tool-result redaction: raw output must not leak to the span ---

        /// A tool that returns a secret; a redaction hook replaces it before the
        /// model — and the trace — sees it.
        struct SecretTool;
        impl crate::tool::Tool for SecretTool {
            const NAME: &'static str = "leak";
            type Error = crate::test_utils::MockToolError;
            type Args = serde_json::Value;
            type Output = String;
            async fn definition(&self, _prompt: String) -> crate::completion::ToolDefinition {
                crate::completion::ToolDefinition {
                    name: Self::NAME.to_string(),
                    description: "returns a secret".to_string(),
                    parameters: serde_json::json!({ "type": "object", "properties": {} }),
                }
            }
            async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
                Ok("SUPER_SECRET_TOKEN_42".to_string())
            }
        }

        /// Redacts every tool result before the model sees it.
        struct RedactResultHook;
        impl<M: crate::completion::CompletionModel> crate::agent::AgentHook<M> for RedactResultHook {
            async fn on_event(
                &self,
                _ctx: &crate::agent::HookContext,
                event: crate::agent::StepEvent<'_, M>,
            ) -> crate::agent::Flow {
                if let crate::agent::StepEvent::ToolResult { .. } = event {
                    crate::agent::Flow::rewrite_result("[REDACTED]")
                } else {
                    crate::agent::Flow::cont()
                }
            }
        }

        /// Captures every value recorded into the `gen_ai.tool.call.result` span
        /// field, so a test can assert the raw secret never reaches the trace.
        #[derive(Default)]
        struct ResultValueVisitor {
            values: Vec<String>,
        }
        impl Visit for ResultValueVisitor {
            fn record_str(&mut self, field: &Field, value: &str) {
                if field.name() == "gen_ai.tool.call.result" {
                    self.values.push(value.to_string());
                }
            }
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                if field.name() == "gen_ai.tool.call.result" {
                    self.values.push(format!("{value:?}"));
                }
            }
        }

        struct ResultValueLayer {
            values: Arc<Mutex<Vec<String>>>,
        }
        impl<S> Layer<S> for ResultValueLayer
        where
            S: Subscriber + for<'l> LookupSpan<'l>,
        {
            fn on_record(&self, _span: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
                let mut visitor = ResultValueVisitor::default();
                values.record(&mut visitor);
                if !visitor.values.is_empty() {
                    self.values.lock().expect("values").extend(visitor.values);
                }
            }
        }

        /// A `ToolResult` hook that redacts the tool's output must prevent the raw
        /// secret from ever reaching the `gen_ai.tool.call.result` span field: the
        /// result is recorded only AFTER the hook runs, so only the redacted
        /// replacement is traced.
        #[tokio::test]
        async fn tool_result_redaction_does_not_leak_raw_output_to_the_span() {
            let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
            let values: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
            let subscriber = Registry::default().with(ResultValueLayer {
                values: values.clone(),
            });
            let _default = tracing::subscriber::set_default(subscriber);

            // Warm the `execute_tool` result callsite under this subscriber, then
            // reset — mirroring the usage tests' interest-cache warm-up.
            warm_blocking_callsites().await;
            tracing::callsite::rebuild_interest_cache();
            values.lock().expect("values").clear();

            let model = MockCompletionModel::from_turns([
                MockTurn::tool_call("tc1", "leak", serde_json::json!({})),
                MockTurn::text("ok"),
            ]);
            let response = AgentBuilder::new(model)
                .tool(SecretTool)
                .add_hook(RedactResultHook)
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("run should succeed");
            assert_eq!(response.output, "ok");

            let captured = values.lock().expect("values").clone();
            assert!(
                !captured.iter().any(|v| v.contains("SUPER_SECRET_TOKEN_42")),
                "the raw tool output must never be recorded on the span; captured: {captured:?}"
            );
            assert!(
                captured.iter().any(|v| v.contains("[REDACTED]")),
                "only the redacted replacement is recorded on the span; captured: {captured:?}"
            );
        }
    }

    fn tool_call_content(id: &str, args: serde_json::Value) -> AssistantContent {
        AssistantContent::ToolCall(ToolCall::new(
            id.to_string(),
            ToolFunction::new("add".to_string(), args),
        ))
    }

    /// Whether any tool result in `messages` carries `expected` as verbatim text.
    /// Used to pin a skip reason's actual value (a reason dropped or altered on
    /// both drivers would still satisfy a blocking == streaming equality check).
    fn tool_result_text_in_history(messages: &[Message], expected: &str) -> bool {
        messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.content.iter().any(|c| matches!(
                                c,
                                crate::message::ToolResultContent::Text(text)
                                    if text.text == expected
                            ))
                    ))
            )
        })
    }

    /// Even with `run()` executing tools concurrently, the tool-result order —
    /// and so the whole message history — matches the sequential streaming
    /// driver. (`run()` runs tools with `buffer_unordered` but writes each result
    /// into its original call-index slot, so results still land in call order.)
    #[tokio::test]
    async fn run_and_stream_same_message_history_for_parallel_tool_calls() {
        let blocking_model = MockCompletionModel::from_turns([
            MockTurn::from_contents([
                tool_call_content("tc1", json!({"x": 2, "y": 3})),
                tool_call_content("tc2", json!({"x": 10, "y": 20})),
            ])
            .expect("two tool calls is a valid turn"),
            MockTurn::text("done"),
        ]);
        let blocking = AgentBuilder::new(blocking_model)
            .tool(MockAddTool)
            .build()
            .runner("add two pairs")
            .max_turns(3)
            .tool_concurrency(4)
            .run()
            .await
            .expect("blocking run should succeed");

        let streaming_model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
                MockStreamEvent::tool_call("tc2", "add", json!({"x": 10, "y": 20})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let mut stream = AgentBuilder::new(streaming_model)
            .tool(MockAddTool)
            .build()
            .runner("add two pairs")
            .max_turns(3)
            .stream()
            .await;
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
                item.map_err(|err| panic!("stream item errored: {err}"))
            {
                final_response = Some(resp);
            }
        }
        let final_response = final_response.expect("stream should yield a final response");

        let blocking_messages = blocking.messages.expect("blocking messages");
        let streaming_messages = final_response
            .history()
            .expect("streaming history")
            .to_vec();
        assert_eq!(
            serde_json::to_value(&blocking_messages).expect("serialize blocking"),
            serde_json::to_value(&streaming_messages).expect("serialize streaming"),
        );
    }

    /// A tool whose first-*called* invocation completes *after* the second, so
    /// `buffer_unordered` yields the results in completion order — yet the
    /// persisted history stays in call order because each result is written into
    /// its original call-index slot. The first call (in poll/call order) waits on
    /// a gate the second call releases.
    #[derive(Clone)]
    struct OutOfOrderTool {
        gate: Arc<tokio::sync::Notify>,
        order: Arc<AtomicU32>,
    }

    impl Tool for OutOfOrderTool {
        const NAME: &'static str = "add";
        type Error = MockToolError;
        type Args = MockOperationArgs;
        type Output = i32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            MockAddTool.definition(String::new()).await
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            let nth = self.order.fetch_add(1, SeqCst);
            if nth == 0 {
                // First call: cannot finish until a later call releases us.
                self.gate.notified().await;
            } else {
                // Later call: finishes immediately and releases the first.
                self.gate.notify_one();
            }
            Ok(nth as i32)
        }
    }

    /// `run()` must persist tool results in tool-call (emission) order even when
    /// tools complete out of order under concurrency — it runs them with
    /// `buffer_unordered` but reindexes each result into its original call-index
    /// slot. (This is what keeps its message history identical to the sequential
    /// streaming driver.)
    #[tokio::test]
    async fn run_preserves_tool_call_order_under_out_of_order_completion() {
        let model = MockCompletionModel::from_turns([
            MockTurn::from_contents([
                tool_call_content("tc1", json!({"x": 1, "y": 0})),
                tool_call_content("tc2", json!({"x": 2, "y": 0})),
            ])
            .expect("two tool calls is a valid turn"),
            MockTurn::text("done"),
        ]);
        let response = AgentBuilder::new(model)
            .tool(OutOfOrderTool {
                gate: Arc::new(tokio::sync::Notify::new()),
                order: Arc::new(AtomicU32::new(0)),
            })
            .build()
            .runner("go")
            .max_turns(3)
            .tool_concurrency(4)
            .run()
            .await
            .expect("run should succeed");

        let messages = response.messages.expect("messages");
        let result_ids: Vec<String> = messages
            .iter()
            .flat_map(|message| match message {
                Message::User { content } => content
                    .iter()
                    .filter_map(|item| match item {
                        UserContent::ToolResult(result) => Some(result.id.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect();
        // Call order (tc1 then tc2), even though tc2 finished first.
        assert_eq!(result_ids, vec!["tc1".to_string(), "tc2".to_string()]);
    }

    /// Drive a stream to completion, panicking on any stream error, and return
    /// its final response.
    async fn drive_to_final_response<R: Send + 'static>(
        mut stream: crate::agent::prompt_request::streaming::StreamingResult<R>,
    ) -> crate::agent::prompt_request::streaming::FinalResponse {
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let MultiTurnStreamItem::FinalResponse(resp) =
                item.unwrap_or_else(|err| panic!("stream item errored: {err}"))
            {
                final_response = Some(resp);
            }
        }
        final_response.expect("stream should yield a final response")
    }

    /// Tool-result ids, in history order, across a run's message history.
    fn tool_result_ids(messages: &[Message]) -> Vec<String> {
        messages
            .iter()
            .flat_map(|message| match message {
                Message::User { content } => content
                    .iter()
                    .filter_map(|item| match item {
                        UserContent::ToolResult(result) => Some(result.id.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect()
    }

    /// The streaming driver under `tool_concurrency > 1` produces the **same
    /// message history** as the blocking driver: streamed results may be
    /// observed in completion order, but persisted results are reassembled in
    /// tool-call order, so concurrency never reorders the final history.
    #[tokio::test]
    async fn stream_and_run_same_message_history_for_parallel_tool_calls_under_concurrency() {
        let blocking_model = MockCompletionModel::from_turns([
            MockTurn::from_contents([
                tool_call_content("tc1", json!({"x": 2, "y": 3})),
                tool_call_content("tc2", json!({"x": 10, "y": 20})),
            ])
            .expect("two tool calls is a valid turn"),
            MockTurn::text("done"),
        ]);
        let blocking = AgentBuilder::new(blocking_model)
            .tool(MockAddTool)
            .build()
            .runner("add two pairs")
            .max_turns(3)
            .tool_concurrency(4)
            .run()
            .await
            .expect("blocking run should succeed");

        let streaming_model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
                MockStreamEvent::tool_call("tc2", "add", json!({"x": 10, "y": 20})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let stream = AgentBuilder::new(streaming_model)
            .tool(MockAddTool)
            .build()
            .runner("add two pairs")
            .max_turns(3)
            .tool_concurrency(4)
            .stream()
            .await;
        let final_response = drive_to_final_response(stream).await;

        let blocking_messages = blocking.messages.expect("blocking messages");
        let streaming_messages = final_response
            .history()
            .expect("streaming history")
            .to_vec();
        assert_eq!(
            serde_json::to_value(&blocking_messages).expect("serialize blocking"),
            serde_json::to_value(&streaming_messages).expect("serialize streaming"),
        );
    }

    /// The streaming driver under concurrency persists tool results in **call
    /// order** even when tools complete out of order. `OutOfOrderTool`'s
    /// first-called invocation only finishes once the second runs, so this also
    /// proves the tools run concurrently: sequential execution would deadlock on
    /// the first call.
    #[tokio::test]
    async fn stream_preserves_history_order_under_out_of_order_completion() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 0})),
                MockStreamEvent::tool_call("tc2", "add", json!({"x": 2, "y": 0})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let stream = AgentBuilder::new(model)
            .tool(OutOfOrderTool {
                gate: Arc::new(tokio::sync::Notify::new()),
                order: Arc::new(AtomicU32::new(0)),
            })
            .build()
            .runner("go")
            .max_turns(3)
            .tool_concurrency(4)
            .stream()
            .await;
        // Timeout so a regression to sequential execution fails cleanly instead
        // of hanging (the first call only completes once the second runs).
        let final_response = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            drive_to_final_response(stream),
        )
        .await
        .expect("streamed tools must run concurrently, not deadlock on the first call");

        let messages = final_response.history().expect("history").to_vec();
        // History stays in call order (tc1 then tc2), even though tc2 finished first.
        assert_eq!(
            tool_result_ids(&messages),
            vec!["tc1".to_string(), "tc2".to_string()]
        );
    }

    /// Under concurrency the streaming driver surfaces tool results **atomically
    /// after the whole batch settles**, in **call order** — not as each tool
    /// completes. The second call completes first (via the gate), yet its result
    /// is still surfaced second, matching persisted history order.
    #[tokio::test]
    async fn stream_emits_tool_results_in_call_order_after_batch_settles_under_concurrency() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 0})),
                MockStreamEvent::tool_call("tc2", "add", json!({"x": 2, "y": 0})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let mut stream = AgentBuilder::new(model)
            .tool(OutOfOrderTool {
                gate: Arc::new(tokio::sync::Notify::new()),
                order: Arc::new(AtomicU32::new(0)),
            })
            .build()
            .runner("go")
            .max_turns(3)
            .tool_concurrency(4)
            .stream()
            .await;

        let mut streamed_result_ids = Vec::new();
        let mut final_response = None;
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(item) = stream.next().await {
                match item.unwrap_or_else(|err| panic!("stream item errored: {err}")) {
                    MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                        tool_result,
                        ..
                    }) => streamed_result_ids.push(tool_result.id),
                    MultiTurnStreamItem::FinalResponse(resp) => final_response = Some(resp),
                    _ => {}
                }
            }
        })
        .await
        .expect("streamed tools must run concurrently, not deadlock on the first call");

        // Call order, even though tc2 completed first — results are surfaced only
        // after the whole batch settles.
        assert_eq!(
            streamed_result_ids,
            vec!["tc1".to_string(), "tc2".to_string()]
        );
        let final_response = final_response.expect("stream should yield a final response");
        assert_eq!(
            tool_result_ids(final_response.history().expect("history")),
            vec!["tc1".to_string(), "tc2".to_string()]
        );
    }

    /// Two barrier-synchronized tools in one streamed turn finish only if they
    /// run concurrently — each waits at the barrier for the other. At
    /// `tool_concurrency(2)` the streamed turn completes; sequential execution
    /// would block on the first call forever, so the timeout asserts genuine
    /// concurrency on the streaming path.
    #[tokio::test]
    async fn stream_executes_tools_concurrently_under_concurrency() {
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("b1", "barrier_tool", json!({})),
                MockStreamEvent::tool_call("b2", "barrier_tool", json!({})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let stream = AgentBuilder::new(model)
            .tool(MockBarrierTool::new(barrier))
            .build()
            .runner("hit the barrier twice")
            .max_turns(3)
            .tool_concurrency(2)
            .stream()
            .await;

        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            drive_to_final_response(stream),
        )
        .await
        .expect("streamed tools must run concurrently, not deadlock at the barrier");
    }

    /// The stream-item taxonomy and ordering: the driver emits *all* of a turn's
    /// **model** tool-call items ([`StreamedAssistantContent::ToolCall`], one per
    /// call the model made) first, then — after the whole tool batch settles —
    /// the per-tool **execution** items (`ToolExecutionStart` then the
    /// `ToolResult`) in call order. This holds identically at every concurrency
    /// (the batch is atomic on both the sequential and concurrent paths).
    #[tokio::test]
    async fn stream_emits_model_tool_calls_then_atomic_execution_items() {
        async fn markers(concurrency: usize) -> Vec<&'static str> {
            let model = MockCompletionModel::from_stream_turns([
                vec![
                    MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 1})),
                    MockStreamEvent::tool_call("tc2", "add", json!({"x": 2, "y": 2})),
                    MockStreamEvent::final_response_with_total_tokens(0),
                ],
                vec![
                    MockStreamEvent::text("done"),
                    MockStreamEvent::final_response_with_total_tokens(0),
                ],
            ]);
            let mut stream = AgentBuilder::new(model)
                .tool(MockAddTool)
                .build()
                .runner("add two pairs")
                .max_turns(3)
                .tool_concurrency(concurrency)
                .stream()
                .await;
            let mut markers = Vec::new();
            while let Some(item) = stream.next().await {
                match item.unwrap_or_else(|err| panic!("stream item errored: {err}")) {
                    MultiTurnStreamItem::StreamAssistantItem(
                        StreamedAssistantContent::ToolCall { .. },
                    ) => markers.push("model-call"),
                    MultiTurnStreamItem::ToolExecutionStart { .. } => markers.push("exec-start"),
                    MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                        ..
                    }) => markers.push("result"),
                    _ => {}
                }
            }
            markers
        }

        // Both surfaces: all model tool calls first, then per-tool (start, result)
        // in call order, surfaced atomically after the batch settles.
        let expected = vec![
            "model-call",
            "model-call",
            "exec-start",
            "result",
            "exec-start",
            "result",
        ];
        assert_eq!(markers(1).await, expected);
        assert_eq!(markers(4).await, expected);
    }

    /// Terminates from the `x == 1` tool's result, but only *after* the slow
    /// `x == 2` sibling has signalled it started executing — so that sibling is
    /// genuinely in flight when the terminate fires (not merely not-yet-started).
    struct TerminateAfterSiblingStartedHook {
        sibling_started: Arc<tokio::sync::Notify>,
    }
    impl<M: CompletionModel> AgentHook<M> for TerminateAfterSiblingStartedHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::ToolResult { args, .. } = event
                && serde_json::from_str::<serde_json::Value>(args)
                    .ok()
                    .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64))
                    == Some(1)
            {
                self.sibling_started.notified().await;
                return Flow::terminate("stop after a tool result");
            }
            Flow::cont()
        }
    }

    /// A probe tool for the concurrent drain path: records how many calls
    /// `started` and `completed`. The `x == 2` call signals it has started, then
    /// stays pending across several polls, so it is genuinely in flight — not
    /// merely not-yet-started — when the `x == 1` call's result terminates the
    /// run. A driver that **drains** the concurrent tool stream polls it to
    /// completion (`completed == 2`); one that **cancels** in-flight siblings
    /// would drop it mid-poll (`completed == 1`).
    #[derive(Clone)]
    struct DrainProbeTool {
        started: Arc<AtomicU32>,
        completed: Arc<AtomicU32>,
        slow_started: Arc<tokio::sync::Notify>,
    }

    impl Tool for DrainProbeTool {
        const NAME: &'static str = "add";
        type Error = MockToolError;
        type Args = serde_json::Value;
        type Output = i32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            MockAddTool.definition(String::new()).await
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.started.fetch_add(1, SeqCst);
            if args.get("x").and_then(serde_json::Value::as_i64) == Some(2) {
                // Signal that the slow sibling has started, then stay pending so
                // it is still executing when the fast call's result terminates.
                self.slow_started.notify_one();
                for _ in 0..8 {
                    tokio::task::yield_now().await;
                }
            }
            self.completed.fetch_add(1, SeqCst);
            Ok(0)
        }
    }

    /// On the concurrent path, a terminate surfaces a `StreamingError`, ends the
    /// run with no final response, and — for a sibling that is **already in
    /// flight** — drains it to completion rather than cancelling it mid-poll (so
    /// no detached task is left running and the deterministic terminate reason
    /// still surfaces). The `x == 2` sibling signals it started before the
    /// `x == 1` result terminates, so `completed == 2` holds only under drain.
    #[tokio::test]
    async fn stream_concurrent_tool_result_terminate_drains_in_flight_siblings() {
        let started = Arc::new(AtomicU32::new(0));
        let completed = Arc::new(AtomicU32::new(0));
        let slow_started = Arc::new(tokio::sync::Notify::new());
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 1})),
                MockStreamEvent::tool_call("tc2", "add", json!({"x": 2, "y": 2})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let mut stream = AgentBuilder::new(model)
            .tool(DrainProbeTool {
                started: started.clone(),
                completed: completed.clone(),
                slow_started: slow_started.clone(),
            })
            .build()
            .runner("add two pairs")
            .max_turns(3)
            .tool_concurrency(2)
            .add_hook(TerminateAfterSiblingStartedHook {
                sibling_started: slow_started,
            })
            .stream()
            .await;

        let (saw_error, saw_final_response) =
            tokio::time::timeout(std::time::Duration::from_secs(5), async move {
                let mut saw_error = false;
                let mut saw_final_response = false;
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final_response = true,
                        Ok(_) => {}
                        Err(StreamingError::Prompt(_)) => saw_error = true,
                        Err(other) => panic!("unexpected streaming error: {other}"),
                    }
                }
                (saw_error, saw_final_response)
            })
            .await
            .expect("draining the concurrent tools must not hang");

        assert!(
            saw_error,
            "a terminate hook on the concurrent path must surface a StreamingError::Prompt"
        );
        assert!(
            !saw_final_response,
            "a terminated run must not yield a final response"
        );
        // The already-in-flight slow sibling is drained to completion, not
        // cancelled mid-poll (which would leave `completed == 1`).
        assert_eq!(
            started.load(SeqCst),
            2,
            "both tools started (both in flight)"
        );
        assert_eq!(
            completed.load(SeqCst),
            2,
            "the in-flight sibling must be drained to completion, not cancelled"
        );
    }

    /// A `Flow::Terminate` from the `ToolCall` event with a reason keyed by the
    /// call's `x` arg, forcing the `x == 2` call (tc2) to terminate *before* the
    /// `x == 1` call (tc1): tc2 opens the gate after terminating, tc1 awaits it
    /// first. So completion order (tc2) differs from call order (tc1).
    struct OrderedTerminateHook {
        gate: Arc<tokio::sync::Notify>,
    }

    impl<M: CompletionModel> AgentHook<M> for OrderedTerminateHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::ToolCall { args, .. } = event {
                let x = serde_json::from_str::<serde_json::Value>(args)
                    .ok()
                    .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64));
                match x {
                    Some(2) => {
                        self.gate.notify_one();
                        return Flow::Terminate {
                            reason: "terminated-by-tc2".to_string(),
                        };
                    }
                    Some(1) => {
                        self.gate.notified().await;
                        return Flow::Terminate {
                            reason: "terminated-by-tc1".to_string(),
                        };
                    }
                    _ => {}
                }
            }
            Flow::cont()
        }
    }

    fn two_terminating_tools_blocking_model() -> MockCompletionModel {
        MockCompletionModel::from_turns([
            MockTurn::from_contents([
                tool_call_content("tc1", json!({"x": 1, "y": 1})),
                tool_call_content("tc2", json!({"x": 2, "y": 2})),
            ])
            .expect("two tool calls is non-empty"),
            MockTurn::text("unreachable"),
        ])
    }

    fn two_terminating_tools_streaming_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 1})),
                MockStreamEvent::tool_call("tc2", "add", json!({"x": 2, "y": 2})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("unreachable"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ])
    }

    /// When two tool calls in one turn both terminate the run under
    /// `tool_concurrency > 1`, run() and stream() surface the **same** reason —
    /// the first-called tool's (call order), not whichever finished first. tc2
    /// terminates before tc1, so a completion-order pick would surface tc2's
    /// reason and the two drivers would disagree.
    #[tokio::test]
    async fn concurrent_simultaneous_tool_terminations_pick_call_order_on_both_drivers() {
        let run_err = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            AgentBuilder::new(two_terminating_tools_blocking_model())
                .tool(MockAddTool)
                .build()
                .runner("go")
                .max_turns(3)
                .tool_concurrency(2)
                .add_hook(OrderedTerminateHook {
                    gate: Arc::new(tokio::sync::Notify::new()),
                })
                .run(),
        )
        .await
        .expect("blocking run must not hang")
        .expect_err("the run must terminate");

        let mut stream = AgentBuilder::new(two_terminating_tools_streaming_model())
            .tool(MockAddTool)
            .build()
            .runner("go")
            .max_turns(3)
            .tool_concurrency(2)
            .add_hook(OrderedTerminateHook {
                gate: Arc::new(tokio::sync::Notify::new()),
            })
            .stream()
            .await;

        let stream_err = tokio::time::timeout(std::time::Duration::from_secs(5), async move {
            while let Some(item) = stream.next().await {
                if let Err(err) = item {
                    return Some(err);
                }
            }
            None
        })
        .await
        .expect("streamed run must not hang")
        .expect("the stream must surface a terminate error");

        let run_msg = run_err.to_string();
        let stream_msg = stream_err.to_string();
        assert!(
            run_msg.contains("terminated-by-tc1"),
            "blocking run should surface the first-called tool's reason, got: {run_msg}"
        );
        assert!(
            stream_msg.contains("terminated-by-tc1"),
            "stream should surface the first-called tool's reason, got: {stream_msg}"
        );
        assert!(
            !run_msg.contains("terminated-by-tc2") && !stream_msg.contains("terminated-by-tc2"),
            "neither driver should surface the later-completing tool's reason"
        );
    }

    /// Terminates the run from the `ToolCall` event of the first tool only
    /// (`x == 1`), letting any later tool through.
    struct TerminateOnFirstToolHook;
    impl<M: CompletionModel> AgentHook<M> for TerminateOnFirstToolHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::ToolCall { args, .. } = event
                && serde_json::from_str::<serde_json::Value>(args)
                    .ok()
                    .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64))
                    == Some(1)
            {
                return Flow::Terminate {
                    reason: "stop".to_string(),
                };
            }
            Flow::cont()
        }
    }

    /// Fail-fast, lock-step across surfaces: on a multi-tool turn whose first
    /// tool's hook terminates the run, the SEQUENTIAL default (`tool_concurrency`
    /// == 1) surfaces the terminate immediately and does **not** start the
    /// remaining sibling tools — so tool B's side effect never runs. The
    /// terminating tool's own body never runs either (its `ToolCall` hook fired
    /// first), so `calls == 0` on both drivers, which share the tool driver.
    #[tokio::test]
    async fn default_concurrency_terminate_skips_remaining_tools_on_both_drivers() {
        let blocking_calls = Arc::new(AtomicU32::new(0));
        AgentBuilder::new(two_terminating_tools_blocking_model())
            .tool(CountingAddTool {
                calls: blocking_calls.clone(),
            })
            .build()
            .runner("go")
            .max_turns(3)
            .add_hook(TerminateOnFirstToolHook)
            .run()
            .await
            .expect_err("the run terminates");
        assert_eq!(
            blocking_calls.load(SeqCst),
            0,
            "fail-fast: blocking run() must not start the second tool after the first terminates"
        );

        let streaming_calls = Arc::new(AtomicU32::new(0));
        let mut stream = AgentBuilder::new(two_terminating_tools_streaming_model())
            .tool(CountingAddTool {
                calls: streaming_calls.clone(),
            })
            .build()
            .runner("go")
            .max_turns(3)
            .add_hook(TerminateOnFirstToolHook)
            .stream()
            .await;
        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                saw_error = true;
                assert!(
                    err.to_string().contains("stop"),
                    "stream() should surface the terminate reason, got: {err}"
                );
                break;
            }
        }
        assert!(saw_error, "stream() must surface the terminate error");
        assert_eq!(
            streaming_calls.load(SeqCst),
            0,
            "fail-fast: stream() must not start the second tool after the first terminates"
        );
    }

    /// Records the `x` arg of every tool call that reaches its body. The `x == 1`
    /// sibling signals it has started (via `sibling_started`) and then stays
    /// pending across several polls, so it is genuinely in flight when the
    /// terminator (`x == 0`) fires — while a sibling beyond the concurrency
    /// window is not yet started and must be dropped.
    #[derive(Clone)]
    struct RecordingArgsTool {
        called: Arc<Mutex<Vec<i64>>>,
        sibling_started: Arc<tokio::sync::Notify>,
    }

    impl Tool for RecordingArgsTool {
        const NAME: &'static str = "add";
        type Error = MockToolError;
        type Args = serde_json::Value;
        type Output = i32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            MockAddTool.definition(String::new()).await
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            let x = args.get("x").and_then(serde_json::Value::as_i64);
            if let Some(x) = x {
                self.called.lock().expect("called").push(x);
            }
            if x == Some(1) {
                // Signal that the in-flight sibling has started, then stay pending
                // so it is still executing when the terminator fires.
                self.sibling_started.notify_one();
                for _ in 0..8 {
                    tokio::task::yield_now().await;
                }
            }
            Ok(0)
        }
    }

    fn three_tools_first_terminates_streaming_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([
            vec![
                // tc0 (x==0) terminates on its ToolCall hook after the in-flight
                // sibling starts; tc1 (x==1) is the in-flight sibling (drains);
                // tc2 (x==2) is beyond the concurrency-2 window (not yet started)
                // and must be dropped once tc0 terminates.
                MockStreamEvent::tool_call("tc0", "add", json!({"x": 0, "y": 0})),
                MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 1})),
                MockStreamEvent::tool_call("tc2", "add", json!({"x": 2, "y": 2})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("unreachable"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ])
    }

    /// Terminates from the `x == 0` tool's `ToolCall` hook, but only after the
    /// `x == 1` sibling has signalled it started executing — so tc1 is genuinely
    /// in flight (not merely not-yet-started) when the terminate fires.
    struct TerminateOnArgZeroAfterSiblingHook {
        sibling_started: Arc<tokio::sync::Notify>,
    }
    impl<M: CompletionModel> AgentHook<M> for TerminateOnArgZeroAfterSiblingHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::ToolCall { args, .. } = event
                && serde_json::from_str::<serde_json::Value>(args)
                    .ok()
                    .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64))
                    == Some(0)
            {
                self.sibling_started.notified().await;
                return Flow::terminate("stop");
            }
            Flow::cont()
        }
    }

    /// Concurrent fail-fast: when a tool terminates the turn under
    /// `tool_concurrency > 1`, an **already-in-flight** sibling is drained while a
    /// sibling **beyond the concurrency window** — not yet started — is dropped.
    /// With concurrency 2 and three tools: tc0 (`x == 0`) terminates only after
    /// tc1 (`x == 1`) has started, so tc1 is genuinely in flight and drains
    /// (`called` contains 1); tc2 (`x == 2`) is pulled only after tc0 frees a slot
    /// — by which time the run is terminating — so it is dropped (`called` never
    /// contains 2), and tc0's own body never runs (its `ToolCall` hook terminated).
    /// The pre-fix run-all-then-decide would have executed tc2 too.
    #[tokio::test]
    async fn concurrent_terminate_drops_beyond_window_sibling_but_drains_in_flight() {
        let called = Arc::new(Mutex::new(Vec::new()));
        let sibling_started = Arc::new(tokio::sync::Notify::new());
        let mut stream = AgentBuilder::new(three_tools_first_terminates_streaming_model())
            .tool(RecordingArgsTool {
                called: called.clone(),
                sibling_started: sibling_started.clone(),
            })
            .build()
            .runner("go")
            .max_turns(3)
            .tool_concurrency(2)
            .add_hook(TerminateOnArgZeroAfterSiblingHook { sibling_started })
            .stream()
            .await;

        let (saw_error, saw_final) =
            tokio::time::timeout(std::time::Duration::from_secs(5), async move {
                let mut saw_error = false;
                let mut saw_final = false;
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final = true,
                        Ok(_) => {}
                        Err(_) => saw_error = true,
                    }
                }
                (saw_error, saw_final)
            })
            .await
            .expect("the concurrent tool drive must not hang");

        assert!(saw_error, "the terminated run must surface an error");
        assert!(
            !saw_final,
            "a terminated run must not yield a final response"
        );
        let called = called.lock().expect("called").clone();
        assert!(
            called.contains(&1),
            "the in-flight sibling (x==1) must be drained to completion; called args: {called:?}"
        );
        assert!(
            !called.contains(&2),
            "the not-yet-started sibling beyond the concurrency window (x==2) must be \
             dropped, not executed; called args: {called:?}"
        );
        assert!(
            !called.contains(&0),
            "the terminator's own body never runs (its ToolCall hook terminated); \
             called args: {called:?}"
        );
    }

    /// A tool that, for the `x == 1` call, records it ran and signals a gate; the
    /// terminating sibling waits on that gate so the `x == 1` call completes
    /// *before* the batch terminates.
    #[derive(Clone)]
    struct SignalOnRunTool {
        a_ran: Arc<AtomicU32>,
        a_done: Arc<tokio::sync::Notify>,
    }
    impl Tool for SignalOnRunTool {
        const NAME: &'static str = "add";
        type Error = MockToolError;
        type Args = serde_json::Value;
        type Output = i32;
        async fn definition(&self, _prompt: String) -> ToolDefinition {
            MockAddTool.definition(String::new()).await
        }
        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            if args.get("x").and_then(serde_json::Value::as_i64) == Some(1) {
                self.a_ran.fetch_add(1, SeqCst);
                self.a_done.notify_one();
            }
            Ok(0)
        }
    }

    /// The `x == 2` tool's `ToolCall` hook terminates, but only after the `x == 1`
    /// sibling has finished (via the gate), so a *completed* sibling's result is
    /// still suppressed by the atomic batch.
    struct TerminateAfterSiblingDoneHook {
        a_done: Arc<tokio::sync::Notify>,
    }
    impl<M: CompletionModel> AgentHook<M> for TerminateAfterSiblingDoneHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::ToolCall { args, .. } = event
                && serde_json::from_str::<serde_json::Value>(args)
                    .ok()
                    .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64))
                    == Some(2)
            {
                self.a_done.notified().await;
                return Flow::terminate("stop");
            }
            Flow::cont()
        }
    }

    /// Atomic concurrent batch: when the batch terminates, even a sibling that
    /// completed **successfully** before the terminating sibling produces no
    /// `ToolExecutionStart` and no `ToolResult` stream item (no orphan
    /// execution-start), and its result is not committed. The `x == 1` tool runs
    /// to completion (its side effect happens) and signals; the `x == 2` tool's
    /// hook then terminates.
    #[tokio::test]
    async fn concurrent_termination_surfaces_no_execution_items() {
        let a_ran = Arc::new(AtomicU32::new(0));
        let a_done = Arc::new(tokio::sync::Notify::new());
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 1})),
                MockStreamEvent::tool_call("tc2", "add", json!({"x": 2, "y": 2})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("unreachable"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let mut stream = AgentBuilder::new(model)
            .tool(SignalOnRunTool {
                a_ran: a_ran.clone(),
                a_done: a_done.clone(),
            })
            .build()
            .runner("go")
            .max_turns(3)
            .tool_concurrency(2)
            .add_hook(TerminateAfterSiblingDoneHook {
                a_done: a_done.clone(),
            })
            .stream()
            .await;

        let (exec_starts, results, saw_error, saw_final) =
            tokio::time::timeout(std::time::Duration::from_secs(5), async move {
                let (mut exec_starts, mut results, mut saw_error, mut saw_final) =
                    (0, 0, false, false);
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(MultiTurnStreamItem::ToolExecutionStart { .. }) => exec_starts += 1,
                        Ok(MultiTurnStreamItem::StreamUserItem(
                            StreamedUserContent::ToolResult { .. },
                        )) => results += 1,
                        Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final = true,
                        Ok(_) => {}
                        Err(_) => saw_error = true,
                    }
                }
                (exec_starts, results, saw_error, saw_final)
            })
            .await
            .expect("the concurrent tool drive must not hang");

        assert!(saw_error, "the terminated run must surface an error");
        assert!(
            !saw_final,
            "a terminated run must not yield a final response"
        );
        assert_eq!(
            exec_starts, 0,
            "a terminated batch surfaces no ToolExecutionStart (no orphan start events)"
        );
        assert_eq!(
            results, 0,
            "a terminated batch surfaces no successful ToolResult"
        );
        assert_eq!(
            a_ran.load(SeqCst),
            1,
            "the fast sibling did run (its side effect happened), but its result was suppressed"
        );
    }

    /// `ToolChoice::Required` + a hook whose `active_tools([])` advertises no tools
    /// is a **local** error: the run fails before any provider round-trip.
    #[tokio::test]
    async fn required_with_empty_active_tools_errors_locally_without_provider_call() {
        struct EmptyActiveToolsHook;
        impl<M: CompletionModel> AgentHook<M> for EmptyActiveToolsHook {
            async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
                if let StepEvent::CompletionCall { .. } = event {
                    Flow::patch_request(RequestPatch::new().active_tools(Vec::<String>::new()))
                } else {
                    Flow::cont()
                }
            }
        }

        let model = MockCompletionModel::from_turns([MockTurn::text("unreachable")]);
        let probe = model.clone();
        let err = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Required)
            .add_hook(EmptyActiveToolsHook)
            .build()
            .runner("go")
            .run()
            .await
            .expect_err("Required with an empty active_tools filter must fail locally");

        assert!(
            probe.requests().is_empty(),
            "the request must fail locally, with no provider round-trip"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("Required"),
            "error should mention Required: {msg}"
        );
        assert!(
            msg.contains("active_tools"),
            "error should name active_tools: {msg}"
        );
    }

    /// `ToolChoice::Specific` naming a tool that a hook's `active_tools` filtered
    /// out is a **local** error naming the filter, before any provider round-trip.
    #[tokio::test]
    async fn specific_naming_filtered_out_tool_errors_locally_without_provider_call() {
        struct FilterToAddHook;
        impl<M: CompletionModel> AgentHook<M> for FilterToAddHook {
            async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
                if let StepEvent::CompletionCall { .. } = event {
                    Flow::patch_request(RequestPatch::new().active_tools(["add"]))
                } else {
                    Flow::cont()
                }
            }
        }

        let model = MockCompletionModel::from_turns([MockTurn::text("unreachable")]);
        let probe = model.clone();
        let err = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["subtract".to_string()],
            })
            .add_hook(FilterToAddHook)
            .build()
            .runner("go")
            .run()
            .await
            .expect_err("Specific naming a filtered-out tool must fail locally");

        assert!(
            probe.requests().is_empty(),
            "the request must fail locally, with no provider round-trip"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("subtract"),
            "error should name the missing tool: {msg}"
        );
        assert!(
            msg.contains("active_tools"),
            "error should name active_tools: {msg}"
        );
    }

    /// `tool_concurrency(0)` is clamped to 1 and runs to completion. The timeout
    /// guards against a regression that lets `concurrency == 0` reach a
    /// `buffer_unordered(0)` (which never makes progress) instead of the
    /// sequential `concurrency <= 1` path.
    #[tokio::test]
    async fn tool_concurrency_zero_is_clamped_and_does_not_hang() {
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "add", json!({"x": 1, "y": 2})),
            MockTurn::text("done"),
        ]);
        let run = AgentBuilder::new(model)
            .tool(MockAddTool)
            .build()
            .runner("add")
            .max_turns(3)
            .tool_concurrency(0)
            .run();

        let response = tokio::time::timeout(std::time::Duration::from_secs(5), run)
            .await
            .expect("tool_concurrency(0) must clamp to 1, not hang on buffer_unordered(0)")
            .expect("run should succeed");
        assert_eq!(response.output, "done");
    }

    /// A tool that counts how many times it actually executes.
    #[derive(Clone)]
    struct CountingAddTool {
        calls: Arc<AtomicU32>,
    }

    impl Tool for CountingAddTool {
        const NAME: &'static str = "add";
        type Error = MockToolError;
        type Args = MockOperationArgs;
        type Output = i32;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            MockAddTool.definition(String::new()).await
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, SeqCst);
            Ok(0)
        }
    }

    struct FailOnToolCallHook;
    impl<M: CompletionModel> AgentHook<M> for FailOnToolCallHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::ToolCall { .. } = event {
                // Fail is illegal for a ToolCall event: the runner must be
                // fail-CLOSED and never execute the tool.
                Flow::fail()
            } else {
                Flow::cont()
            }
        }
    }

    /// A hook returning `Flow::Fail` for a tool call (an action that is not
    /// honored for that event) terminates the run fail-closed — the tool never
    /// executes.
    #[tokio::test]
    async fn tool_call_fail_is_fail_closed() {
        let calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_turns([MockTurn::tool_call(
            "tc1",
            "add",
            json!({"x": 1, "y": 2}),
        )]);
        let agent = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: calls.clone(),
            })
            .build();

        let err = agent
            .runner("add")
            .max_turns(2)
            .add_hook(FailOnToolCallHook)
            .run()
            .await
            .expect_err("fail-closed: the run must error rather than execute the tool");

        assert!(matches!(err, PromptError::PromptCancelled { .. }));
        assert_eq!(
            calls.load(SeqCst),
            0,
            "tool must not execute when a hook returns Fail for a tool call"
        );
    }

    #[derive(Clone, Default)]
    struct ToolOnlyHook {
        text_delta_calls: Arc<AtomicU32>,
        other_calls: Arc<AtomicU32>,
    }

    impl<M: CompletionModel> AgentHook<M> for ToolOnlyHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            match event.kind() {
                StepEventKind::TextDelta => {
                    self.text_delta_calls.fetch_add(1, SeqCst);
                }
                _ => {
                    self.other_calls.fetch_add(1, SeqCst);
                }
            }
            Flow::cont()
        }

        fn observes(&self, kind: StepEventKind) -> bool {
            kind != StepEventKind::TextDelta
        }
    }

    /// A hook that declares it does not observe text deltas is never dispatched
    /// for them (the runner skips building/dispatching that event), but still
    /// receives the events it does observe.
    #[tokio::test]
    async fn observes_gates_text_delta_dispatch() {
        let model = MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::text("hel"),
            MockStreamEvent::text("lo"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ]]);
        let hook = ToolOnlyHook::default();
        let mut stream = AgentBuilder::new(model)
            .build()
            .runner("hi")
            .add_hook(hook.clone())
            .stream()
            .await;
        while stream.next().await.is_some() {}

        assert_eq!(
            hook.text_delta_calls.load(SeqCst),
            0,
            "a hook that does not observe TextDelta must not be dispatched for it"
        );
        assert!(
            hook.other_calls.load(SeqCst) > 0,
            "the hook should still receive the events it observes"
        );
    }

    /// Terminates the run when it sees a chosen event kind, observing every other
    /// event as `Continue`.
    struct TerminateOn(StepEventKind);

    impl<M: CompletionModel> AgentHook<M> for TerminateOn {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if event.kind() == self.0 {
                Flow::terminate("stop here")
            } else {
                Flow::cont()
            }
        }
    }

    /// `Flow::Terminate` cancels the blocking run from *every* shared driver
    /// event (model call, model response, tool call, tool result) — none is a
    /// silent no-op.
    #[tokio::test]
    async fn run_terminates_from_each_shared_event() {
        for kind in [
            StepEventKind::CompletionCall,
            StepEventKind::CompletionResponse,
            StepEventKind::ToolCall,
            StepEventKind::ToolResult,
        ] {
            let err = AgentBuilder::new(blocking_model())
                .tool(MockAddTool)
                .build()
                .runner("add 2 and 3")
                .max_turns(3)
                .add_hook(TerminateOn(kind))
                .run()
                .await
                .expect_err(&format!("terminate at {kind:?} must cancel the run"));
            assert!(
                matches!(err, PromptError::PromptCancelled { .. }),
                "terminate at {kind:?} should cancel the run, got {err:?}"
            );
        }
    }

    /// The same fail-closed termination holds for the streaming driver across the
    /// shared events it fires (it surfaces `StreamResponseFinish` instead of
    /// `CompletionResponse`): each yields a stream error and no final response.
    #[tokio::test]
    async fn stream_terminates_from_each_shared_event() {
        for kind in [
            StepEventKind::CompletionCall,
            StepEventKind::ToolCall,
            StepEventKind::ToolResult,
        ] {
            let mut stream = AgentBuilder::new(streaming_model())
                .tool(MockAddTool)
                .build()
                .runner("add 2 and 3")
                .max_turns(3)
                .add_hook(TerminateOn(kind))
                .stream()
                .await;

            let mut saw_error = false;
            let mut saw_final = false;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final = true,
                    Err(_) => saw_error = true,
                    _ => {}
                }
            }
            assert!(saw_error, "terminate at {kind:?} must yield a stream error");
            assert!(
                !saw_final,
                "terminate at {kind:?} must not also produce a final response"
            );
        }
    }

    /// Two hooks pushed onto one stack both observe every event (no short-circuit
    /// on `Continue`), and the stack's shared event sequence is identical across
    /// the blocking and streaming drivers.
    #[tokio::test]
    async fn multi_hook_stack_parity_across_run_and_stream() {
        let a_block = RecordingHook::default();
        let b_block = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model())
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(a_block.clone())
            .add_hook(b_block.clone())
            .run()
            .await
            .expect("blocking run should succeed");

        let a_stream = RecordingHook::default();
        let b_stream = RecordingHook::default();
        let mut stream = AgentBuilder::new(streaming_model())
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(a_stream.clone())
            .add_hook(b_stream.clone())
            .stream()
            .await;
        while stream.next().await.is_some() {}

        // Both hooks in the stack saw the same events (both ran on every Continue).
        assert_eq!(a_block.shared_events(), b_block.shared_events());
        assert_eq!(a_stream.shared_events(), b_stream.shared_events());
        // The stack's shared event sequence is identical across drivers.
        assert_eq!(a_block.shared_events(), a_stream.shared_events());
        assert_eq!(
            a_block.shared_events(),
            vec![
                StepEventKind::CompletionCall,
                StepEventKind::ToolCall,
                StepEventKind::ToolResult,
                StepEventKind::CompletionCall,
            ]
        );
        assert_eq!(blocking.output, "the answer is 5");
    }

    /// Renames an invalid tool call to a known tool; observes everything else.
    struct RepairInvalidToHook(&'static str);

    impl<M: CompletionModel> AgentHook<M> for RepairInvalidToHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::InvalidToolCall(_) = event {
                Flow::repair(self.0)
            } else {
                Flow::cont()
            }
        }
    }

    /// An invalid tool call repaired by a hook recovers identically under run()
    /// and stream(): the renamed tool executes and both drivers reach the same
    /// output, tool-result content, and final message history.
    #[tokio::test]
    async fn invalid_tool_call_repair_parity_across_run_and_stream() {
        let blocking_model = MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("the answer is 5"),
        ]);
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model)
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(blocking_hook.clone())
            .add_hook(RepairInvalidToHook("add"))
            .run()
            .await
            .expect("blocking run should recover via repair");

        // Emit the invalid call as a single complete tool call (mirroring the
        // blocking model). A provider stream carries one tool call via one
        // mechanism — deltas *or* a complete call — so this is the apples-to-
        // apples comparison; mixing both would trip the assembler's two
        // independent invalid-detection sites and fire the event twice.
        let streaming_model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("the answer is 5"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let streaming_hook = RecordingHook::default();
        let mut stream = AgentBuilder::new(streaming_model)
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(streaming_hook.clone())
            .add_hook(RepairInvalidToHook("add"))
            .stream()
            .await;
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
                item.map_err(|err| panic!("stream item errored: {err}"))
            {
                final_response = Some(resp);
            }
        }
        let final_response =
            final_response.expect("stream should recover and yield a final response");

        // Same recovered output.
        assert_eq!(blocking.output, "the answer is 5");
        assert_eq!(final_response.response(), blocking.output);

        // Both drivers reported the invalid tool call to the hook, then executed
        // the repaired tool, so the shared event sequences match.
        assert_eq!(
            blocking_hook.shared_events(),
            streaming_hook.shared_events()
        );
        assert!(
            blocking_hook
                .shared_events()
                .contains(&StepEventKind::InvalidToolCall),
            "the hook must observe the invalid tool call"
        );
        assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());
        assert_eq!(blocking_hook.tool_results(), vec!["5".to_string()]);

        // Same final message history.
        let blocking_messages = blocking.messages.expect("blocking messages");
        let streaming_messages = final_response
            .history()
            .expect("streaming history")
            .to_vec();
        assert_eq!(
            serde_json::to_value(&blocking_messages).expect("serialize blocking"),
            serde_json::to_value(&streaming_messages).expect("serialize streaming"),
        );
    }

    // ----------------------------------------------------------------------
    // Single-source-of-truth parity harness
    // ----------------------------------------------------------------------
    //
    // `run()` and `stream()` are two implementations of one agent loop; testing
    // they agree on the same input is *differential testing*, with each driver
    // acting as the other's oracle. The hazard such tests have (and that bit the
    // invalid-tool-repair test above) is *fixture drift*: when the blocking
    // `MockTurn` list and the streaming `MockStreamEvent` list are hand-written
    // separately, they can silently encode different model behavior, so a
    // passing test proves nothing.
    //
    // The fix — the single-source-of-truth / data-driven principle, embodied by
    // pydantic-ai's `TestModel` (one scripted response replayed as a stream) and
    // litellm's `stream_chunk_builder` (reassemble the stream, compare to the
    // whole) — is to derive *both* encodings from one canonical `ScriptedTurn`
    // list. The two drivers are then provably fed identical model behavior and
    // can be asserted equal on the medium-independent projection (final output,
    // message history, tool-result content, shared hook-event sequence).

    /// One tool call inside a scripted turn.
    #[derive(Clone)]
    struct ScriptedToolCall {
        id: &'static str,
        name: &'static str,
        args: serde_json::Value,
    }

    /// One scripted model turn, described once and rendered into both a blocking
    /// `MockTurn` and a streaming `Vec<MockStreamEvent>`.
    #[derive(Clone)]
    enum ScriptedTurn {
        /// A final text answer.
        Text(&'static str),
        /// One or more tool calls emitted in a single turn.
        ToolCalls(Vec<ScriptedToolCall>),
    }

    /// How a tool call is rendered onto the wire for the streaming driver. Both
    /// shapes must yield the *same* canonical turn ("chunked-input invariance",
    /// the `tokio-util` `LengthDelimitedCodec` lesson): the assembled message
    /// history and tool results may not depend on whether a provider sends a
    /// complete tool call or streams it as deltas.
    #[derive(Clone, Copy)]
    enum StreamShape {
        /// One complete tool-call event per call (mirrors the blocking turn).
        Complete,
        /// Name + argument deltas followed by the complete call, additionally
        /// exercising the delta-hook path and the assembler's delta buffering.
        Chunked,
    }

    impl ScriptedTurn {
        fn as_blocking_turn(&self) -> MockTurn {
            match self {
                ScriptedTurn::Text(text) => MockTurn::text(*text),
                ScriptedTurn::ToolCalls(calls) => {
                    MockTurn::from_contents(calls.iter().map(|call| {
                        AssistantContent::ToolCall(ToolCall::new(
                            call.id.to_string(),
                            ToolFunction::new(call.name.to_string(), call.args.clone()),
                        ))
                    }))
                    .expect("a scripted tool-call turn has at least one call")
                }
            }
        }

        fn as_stream_events(&self, shape: StreamShape) -> Vec<MockStreamEvent> {
            let mut events = Vec::new();
            match self {
                ScriptedTurn::Text(text) => events.push(MockStreamEvent::text(*text)),
                ScriptedTurn::ToolCalls(calls) => {
                    for call in calls {
                        if let StreamShape::Chunked = shape {
                            // Distinct internal id per call; the canonical args
                            // still come from the complete event below, so this
                            // exercises the delta path without changing the turn.
                            let internal = format!("ic-{}", call.id);
                            let args = serde_json::to_string(&call.args)
                                .expect("scripted args serialize to json");
                            events.push(MockStreamEvent::tool_call_name_delta(
                                call.id, &internal, call.name,
                            ));
                            events.push(MockStreamEvent::tool_call_arguments_delta(
                                call.id, &internal, &args,
                            ));
                        }
                        events.push(MockStreamEvent::tool_call(
                            call.id,
                            call.name,
                            call.args.clone(),
                        ));
                    }
                }
            }
            events.push(MockStreamEvent::final_response_with_total_tokens(0));
            events
        }
    }

    /// The medium-independent projection of a run that both drivers must agree
    /// on.
    struct ParityOutcome {
        output: String,
        messages: Vec<Message>,
        shared_events: Vec<StepEventKind>,
        tool_results: Vec<String>,
    }

    async fn run_blocking_scenario(prompt: &'static str, turns: &[ScriptedTurn]) -> ParityOutcome {
        let model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let hook = RecordingHook::default();
        let response = AgentBuilder::new(model)
            .tool(MockAddTool)
            .build()
            .runner(prompt)
            .max_turns(8)
            .add_hook(hook.clone())
            .run()
            .await
            .expect("blocking scenario should succeed");
        ParityOutcome {
            output: response.output,
            messages: response.messages.expect("blocking messages"),
            shared_events: hook.shared_events(),
            tool_results: hook.tool_results(),
        }
    }

    async fn run_streaming_scenario(
        prompt: &'static str,
        turns: &[ScriptedTurn],
        shape: StreamShape,
    ) -> ParityOutcome {
        let model = MockCompletionModel::from_stream_turns(
            turns.iter().map(|turn| turn.as_stream_events(shape)),
        );
        let hook = RecordingHook::default();
        let mut stream = AgentBuilder::new(model)
            .tool(MockAddTool)
            .build()
            .runner(prompt)
            .max_turns(8)
            .add_hook(hook.clone())
            .stream()
            .await;
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
                item.map_err(|err| panic!("stream item errored: {err}"))
            {
                final_response = Some(resp);
            }
        }
        let final_response =
            final_response.expect("streaming scenario should yield a final response");
        ParityOutcome {
            output: final_response.response().to_string(),
            messages: final_response
                .history()
                .expect("streaming history")
                .to_vec(),
            shared_events: hook.shared_events(),
            tool_results: hook.tool_results(),
        }
    }

    fn assert_outcomes_match(blocking: &ParityOutcome, streaming: &ParityOutcome, label: &str) {
        assert_eq!(
            blocking.output, streaming.output,
            "{label}: final output diverged"
        );
        assert_eq!(
            blocking.shared_events, streaming.shared_events,
            "{label}: hook event sequence diverged"
        );
        assert_eq!(
            blocking.tool_results, streaming.tool_results,
            "{label}: tool-result content diverged"
        );
        assert_eq!(
            serde_json::to_value(&blocking.messages).expect("serialize blocking"),
            serde_json::to_value(&streaming.messages).expect("serialize streaming"),
            "{label}: message history diverged"
        );
    }

    /// Drive one canonical scenario through `run()` and through `stream()` in
    /// both wire shapes, asserting the medium-independent projection is
    /// identical every way. Because both stream shapes are compared against the
    /// same blocking outcome, they are also transitively equal to each other.
    async fn assert_run_stream_parity(prompt: &'static str, turns: &[ScriptedTurn]) {
        let blocking = run_blocking_scenario(prompt, turns).await;
        for (shape, label) in [
            (StreamShape::Complete, "complete-stream"),
            (StreamShape::Chunked, "chunked-stream"),
        ] {
            let streaming = run_streaming_scenario(prompt, turns, shape).await;
            assert_outcomes_match(&blocking, &streaming, label);
        }
    }

    fn add_call(id: &'static str, x: i64, y: i64) -> ScriptedToolCall {
        ScriptedToolCall {
            id,
            name: "add",
            args: json!({ "x": x, "y": y }),
        }
    }

    #[tokio::test]
    async fn parity_text_only_run() {
        assert_run_stream_parity("just say hi", &[ScriptedTurn::Text("hi there")]).await;
    }

    #[tokio::test]
    async fn parity_single_tool_then_text() {
        assert_run_stream_parity(
            "add 2 and 3",
            &[
                ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
                ScriptedTurn::Text("the answer is 5"),
            ],
        )
        .await;
    }

    #[tokio::test]
    async fn parity_multiple_tools_in_one_turn() {
        assert_run_stream_parity(
            "add two pairs",
            &[
                ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3), add_call("tc2", 10, 20)]),
                ScriptedTurn::Text("done"),
            ],
        )
        .await;
    }

    #[tokio::test]
    async fn parity_multi_turn_sequential_tools() {
        assert_run_stream_parity(
            "chain two additions",
            &[
                ScriptedTurn::ToolCalls(vec![add_call("tc1", 1, 1)]),
                ScriptedTurn::ToolCalls(vec![add_call("tc2", 2, 2)]),
                ScriptedTurn::Text("chained"),
            ],
        )
        .await;
    }

    /// Skips an invalid tool call (synthetic result, no execution); observes
    /// everything else.
    struct SkipInvalidHook(&'static str);

    impl<M: CompletionModel> AgentHook<M> for SkipInvalidHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::InvalidToolCall(_) = event {
                Flow::skip(self.0)
            } else {
                Flow::cont()
            }
        }
    }

    /// An invalid tool call *skipped* by a hook recovers identically under
    /// `run()` and `stream()`: the synthetic skip result enters the history
    /// verbatim (it is never re-parsed as tool output) and both drivers reach
    /// the same output and message history. Complements the repair-parity test.
    #[tokio::test]
    async fn invalid_tool_call_skip_parity_across_run_and_stream() {
        let blocking_model = MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("acknowledged"),
        ]);
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model)
            .tool(MockAddTool)
            .build()
            .runner("do the thing")
            .max_turns(3)
            .add_hook(blocking_hook.clone())
            .add_hook(SkipInvalidHook("tool not permitted"))
            .run()
            .await
            .expect("blocking run should recover via skip");

        // Single complete tool call (mirrors the blocking model; see the
        // repair-parity test for why deltas are not mixed in here).
        let streaming_model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("acknowledged"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let streaming_hook = RecordingHook::default();
        let mut stream = AgentBuilder::new(streaming_model)
            .tool(MockAddTool)
            .build()
            .runner("do the thing")
            .max_turns(3)
            .add_hook(streaming_hook.clone())
            .add_hook(SkipInvalidHook("tool not permitted"))
            .stream()
            .await;
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
                item.map_err(|err| panic!("stream item errored: {err}"))
            {
                final_response = Some(resp);
            }
        }
        let final_response =
            final_response.expect("stream should recover and yield a final response");

        assert_eq!(blocking.output, "acknowledged");
        assert_eq!(final_response.response(), blocking.output);
        assert_eq!(
            blocking_hook.shared_events(),
            streaming_hook.shared_events()
        );
        assert!(
            blocking_hook
                .shared_events()
                .contains(&StepEventKind::InvalidToolCall),
            "the hook must observe the invalid tool call"
        );

        let blocking_messages = blocking.messages.expect("blocking messages");
        let streaming_messages = final_response
            .history()
            .expect("streaming history")
            .to_vec();
        assert_eq!(
            serde_json::to_value(&blocking_messages).expect("serialize blocking"),
            serde_json::to_value(&streaming_messages).expect("serialize streaming"),
        );
        // Pin the actual reason, not just blocking == streaming (see the valid-tool
        // skip test): a reason dropped or altered on BOTH paths would still pass.
        assert!(
            tool_result_text_in_history(&blocking_messages, "tool not permitted"),
            "the verbatim invalid-tool skip reason must be the tool result content"
        );
    }

    /// A turn that streams *text and* an invalid tool call, then is repaired, is
    /// a recovered turn: its response-finish hook must be suppressed on BOTH
    /// drivers — `CompletionResponse` under `run()`, `StreamResponseFinish` under
    /// `stream()`. The shared-events parity harness deliberately excludes these
    /// medium-specific events, so this asymmetry needs a dedicated assertion (it
    /// is the exact event the harness cannot see).
    #[tokio::test]
    async fn recovered_turn_suppresses_response_finish_hook_on_both_drivers() {
        // Turn 1 emits text then an invalid tool call (repaired to "add"); turn 2
        // is a plain final-text turn whose response event DOES fire on both
        // drivers — so a correct run sees exactly one response-finish event.
        let blocking_model = MockCompletionModel::from_turns([
            MockTurn::from_contents([
                AssistantContent::text("let me compute that"),
                AssistantContent::ToolCall(ToolCall::new(
                    "tc1".to_string(),
                    ToolFunction::new("default_api".to_string(), json!({"x": 2, "y": 3})),
                )),
            ])
            .expect("a text + tool-call turn is valid"),
            MockTurn::text("the answer is 5"),
        ]);
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model)
            .tool(MockAddTool)
            .build()
            .runner("compute")
            .max_turns(3)
            .add_hook(blocking_hook.clone())
            .add_hook(RepairInvalidToHook("add"))
            .run()
            .await
            .expect("blocking run should recover via repair");

        let streaming_model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("let me compute that"),
                MockStreamEvent::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("the answer is 5"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let streaming_hook = RecordingHook::default();
        let mut stream = AgentBuilder::new(streaming_model)
            .tool(MockAddTool)
            .build()
            .runner("compute")
            .max_turns(3)
            .add_hook(streaming_hook.clone())
            .add_hook(RepairInvalidToHook("add"))
            .stream()
            .await;
        while stream.next().await.is_some() {}

        // Recovery still reaches the same final answer.
        assert_eq!(blocking.output, "the answer is 5");

        // Blocking: the recovered turn 1 suppresses `CompletionResponse`; only the
        // plain turn 2 fires it.
        assert_eq!(
            blocking_hook.count(StepEventKind::CompletionResponse),
            1,
            "the recovered turn must not fire CompletionResponse"
        );
        // Streaming: the recovered turn 1 must likewise suppress
        // `StreamResponseFinish` (without the fix this is 2).
        assert_eq!(
            streaming_hook.count(StepEventKind::StreamResponseFinish),
            1,
            "the recovered turn must not fire StreamResponseFinish"
        );
        // Stated as parity: the count of un-suppressed response-finish events is
        // the same across drivers.
        assert_eq!(
            blocking_hook.count(StepEventKind::CompletionResponse),
            streaming_hook.count(StepEventKind::StreamResponseFinish),
        );
    }

    /// Skips a *valid* tool call before execution; observes everything else.
    struct SkipToolCallHook(&'static str);

    impl<M: CompletionModel> AgentHook<M> for SkipToolCallHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::ToolCall { .. } = event {
                Flow::skip(self.0)
            } else {
                Flow::cont()
            }
        }
    }

    /// A hook that skips a *valid* tool call (`Flow::Skip` on `ToolCall`, the
    /// honored-action path — distinct from skipping an *invalid* call) recovers
    /// identically under `run()` and `stream()`: the synthetic skip result enters
    /// the history verbatim without executing the tool, and both drivers reach the
    /// same output, tool-result content and message history.
    #[tokio::test]
    async fn valid_tool_call_skip_parity_across_run_and_stream() {
        let turns = [
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
            ScriptedTurn::Text("acknowledged"),
        ];

        let blocking_model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model)
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(blocking_hook.clone())
            .add_hook(SkipToolCallHook("skipped by policy"))
            .run()
            .await
            .expect("blocking run should succeed with a skipped tool call");

        let streaming_model = MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        );
        let streaming_hook = RecordingHook::default();
        let mut stream = AgentBuilder::new(streaming_model)
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(streaming_hook.clone())
            .add_hook(SkipToolCallHook("skipped by policy"))
            .stream()
            .await;
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
                item.map_err(|err| panic!("stream item errored: {err}"))
            {
                final_response = Some(resp);
            }
        }
        let final_response = final_response.expect("stream should yield a final response");

        assert_eq!(blocking.output, "acknowledged");
        assert_eq!(final_response.response(), blocking.output);
        assert_eq!(
            blocking_hook.shared_events(),
            streaming_hook.shared_events()
        );
        // A skipped valid tool call does not fire the `ToolResult` hook
        // (`run_single_tool` returns the synthetic result before it), so the hook
        // records no tool result on either driver — the verbatim skip reason
        // lands in the message history instead (asserted below).
        assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());
        assert!(
            blocking_hook.tool_results().is_empty(),
            "a skipped tool executes nothing, so no ToolResult hook fires"
        );

        let blocking_messages = blocking.messages.expect("blocking messages");
        let streaming_messages = final_response
            .history()
            .expect("streaming history")
            .to_vec();
        assert_eq!(
            serde_json::to_value(&blocking_messages).expect("serialize blocking"),
            serde_json::to_value(&streaming_messages).expect("serialize streaming"),
        );
        // Pin the actual reason, not just blocking == streaming: a reason dropped
        // or altered on BOTH paths would still satisfy the equality above.
        assert!(
            tool_result_text_in_history(&blocking_messages, "skipped by policy"),
            "the verbatim skip reason must be the tool result content in the history"
        );
    }

    /// A hook that rewrites a valid tool call's arguments (`Flow::RewriteArgs` on
    /// `ToolCall`) so the tool executes with the replacement instead of what the
    /// model emitted.
    struct RewriteToolArgsHook(serde_json::Value);

    impl<M: CompletionModel> AgentHook<M> for RewriteToolArgsHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::ToolCall { .. } = event {
                Flow::rewrite_args(self.0.clone())
            } else {
                Flow::cont()
            }
        }
    }

    /// `Flow::RewriteArgs` resolves to a `ProceedWith` tool-call decision that
    /// carries the replacement arguments, and is named for fail-closed
    /// diagnostics.
    #[test]
    fn rewrite_args_resolves_to_proceed_with_for_tool_call() {
        let args = json!({"x": 1, "y": 2});
        match super::flow_into_tool_call(Flow::rewrite_args(args.clone())) {
            super::ToolCallDecision::ProceedWith(replacement) => assert_eq!(replacement, args),
            _ => panic!("RewriteArgs should resolve to ProceedWith for a tool call"),
        }
        assert_eq!(
            super::flow_name(&Flow::rewrite_args(json!({}))),
            "RewriteArgs"
        );
        // The typed convenience builds the same variant as the value constructor.
        assert_eq!(
            Flow::try_rewrite_args(&json!({"x": 1, "y": 2})).expect("serializes"),
            Flow::rewrite_args(json!({"x": 1, "y": 2})),
        );
    }

    /// `RewriteArgs` is only honored by `ToolCall`; every other event is
    /// fail-closed and terminates the run rather than silently proceeding.
    #[test]
    fn rewrite_args_is_fail_closed_off_the_tool_call_event() {
        // Invalid tool calls only honor Fail/Retry/Repair/Skip/Terminate.
        assert!(matches!(
            super::flow_into_invalid(Flow::rewrite_args(json!({}))),
            super::InvalidDecision::Terminate(_)
        ));
        // Observe-only events only honor Continue/Terminate.
        assert!(super::observe_flow(Flow::rewrite_args(json!({}))).is_some());
    }

    /// A hook that rewrites a *valid* tool call's arguments (`Flow::RewriteArgs`
    /// on `ToolCall`) is honored identically under `run()` and `stream()`: the
    /// tool executes with the replacement, so both drivers observe the same
    /// rewritten tool result and reach the same output, tool-result content and
    /// message history. Both drivers share `run_single_tool`, so they stay in
    /// lock-step.
    #[tokio::test]
    async fn valid_tool_call_rewrite_args_parity_across_run_and_stream() {
        // The model asks to add 2 + 3; the hook rewrites the arguments to 2 + 40,
        // so the tool returns 42 rather than 5.
        let turns = [
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
            ScriptedTurn::Text("acknowledged"),
        ];
        let replacement = json!({"x": 2, "y": 40});

        let blocking_model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model)
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(blocking_hook.clone())
            .add_hook(RewriteToolArgsHook(replacement.clone()))
            .run()
            .await
            .expect("blocking run should succeed with rewritten tool arguments");

        let streaming_model = MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        );
        let streaming_hook = RecordingHook::default();
        let mut stream = AgentBuilder::new(streaming_model)
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(streaming_hook.clone())
            .add_hook(RewriteToolArgsHook(replacement))
            .stream()
            .await;
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
                item.map_err(|err| panic!("stream item errored: {err}"))
            {
                final_response = Some(resp);
            }
        }
        let final_response = final_response.expect("stream should yield a final response");

        // The tool ran with the rewritten arguments (2 + 40 = 42), not the
        // model's emitted 2 + 3 = 5 — on both drivers.
        assert_eq!(blocking_hook.tool_results(), vec!["42".to_string()]);
        assert_eq!(blocking.output, "acknowledged");
        assert_eq!(final_response.response(), blocking.output);
        assert_eq!(
            blocking_hook.shared_events(),
            streaming_hook.shared_events()
        );
        assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());
    }

    /// A hook that rewrites a tool's result (`Flow::RewriteResult` on
    /// `ToolResult`) so the model sees the replacement instead of the tool's
    /// actual output.
    struct RewriteToolResultHook(&'static str);

    impl<M: CompletionModel> AgentHook<M> for RewriteToolResultHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::ToolResult { .. } = event {
                Flow::rewrite_result(self.0)
            } else {
                Flow::cont()
            }
        }
    }

    /// `Flow::RewriteResult` resolves to a `Replace` tool-result decision carrying
    /// the replacement, and is named for fail-closed diagnostics.
    #[test]
    fn rewrite_result_resolves_to_replace_for_tool_result() {
        match super::flow_into_tool_result(Flow::rewrite_result("redacted")) {
            super::ToolResultDecision::Replace(result) => assert_eq!(result, "redacted"),
            _ => panic!("RewriteResult should resolve to Replace for a tool result"),
        }
        assert_eq!(
            super::flow_name(&Flow::rewrite_result("x")),
            "RewriteResult"
        );
    }

    /// `RewriteResult` is only honored by `ToolResult`, and the tool-result event
    /// only honors `RewriteResult` (not `RewriteArgs`) — both directions are
    /// fail-closed.
    #[test]
    fn rewrite_result_is_fail_closed_off_the_tool_result_event() {
        // Invalid tool calls don't honor RewriteResult.
        assert!(matches!(
            super::flow_into_invalid(Flow::rewrite_result("x")),
            super::InvalidDecision::Terminate(_)
        ));
        // Observe-only events (CompletionResponse, deltas, ...) don't honor it.
        assert!(super::observe_flow(Flow::rewrite_result("x")).is_some());
        // The tool-RESULT event rejects RewriteArgs (the pre-tool action),
        // mirroring how the tool-CALL event rejects RewriteResult.
        assert!(matches!(
            super::flow_into_tool_result(Flow::rewrite_args(json!({}))),
            super::ToolResultDecision::Terminate(_)
        ));
    }

    /// A hook that rewrites a tool's result (`Flow::RewriteResult` on
    /// `ToolResult`) is honored identically under `run()` and `stream()`: the
    /// model-visible history carries the replacement while the `ToolResult` event
    /// still observed the tool's actual output, and both drivers reach the same
    /// output and history. Both share `run_single_tool`, so they stay in
    /// lock-step.
    #[tokio::test]
    async fn valid_tool_result_rewrite_parity_across_run_and_stream() {
        // The tool computes 2 + 3 = 5; the hook replaces what the model sees with
        // "redacted-result".
        let turns = [
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
            ScriptedTurn::Text("acknowledged"),
        ];

        let blocking_model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model)
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(blocking_hook.clone())
            .add_hook(RewriteToolResultHook("redacted-result"))
            .run()
            .await
            .expect("blocking run should succeed with a rewritten tool result");

        let streaming_model = MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        );
        let streaming_hook = RecordingHook::default();
        let mut stream = AgentBuilder::new(streaming_model)
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(streaming_hook.clone())
            .add_hook(RewriteToolResultHook("redacted-result"))
            .stream()
            .await;
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
                item.map_err(|err| panic!("stream item errored: {err}"))
            {
                final_response = Some(resp);
            }
        }
        let final_response = final_response.expect("stream should yield a final response");

        assert_eq!(blocking.output, "acknowledged");
        assert_eq!(final_response.response(), blocking.output);

        // The ToolResult event observes the tool's ACTUAL output (5) on both
        // drivers — the replacement is applied after the event fires.
        assert_eq!(blocking_hook.tool_results(), vec!["5".to_string()]);
        assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());

        // The model-visible history carries the REWRITTEN result, not "5", and is
        // byte-identical across drivers.
        let blocking_messages = blocking.messages.expect("blocking messages");
        let streaming_messages = final_response
            .history()
            .expect("streaming history")
            .to_vec();
        assert_eq!(
            serde_json::to_value(&blocking_messages).expect("serialize blocking"),
            serde_json::to_value(&streaming_messages).expect("serialize streaming"),
        );
        assert!(
            tool_result_text_in_history(&blocking_messages, "redacted-result"),
            "the model-visible tool result must be the hook's replacement"
        );
        assert!(
            !tool_result_text_in_history(&blocking_messages, "5"),
            "the tool's original output must not reach the model after a rewrite"
        );
    }

    /// A `RewriteResult` replacement is delivered to the model verbatim, not
    /// re-parsed as structured/multimodal tool output. A JSON-shaped replacement
    /// (here, an image payload that `tool_result_output` would turn into an image
    /// content block for *real* tool output) reaches history as literal text —
    /// so a redaction hook returning JSON cannot be silently restructured.
    #[tokio::test]
    async fn rewrite_result_is_delivered_verbatim_not_reparsed() {
        const IMAGE_JSON: &str = r#"{"type":"image","data":"abc","mimeType":"image/png"}"#;

        let turns = [
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
            ScriptedTurn::Text("done"),
        ];
        let model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let result = AgentBuilder::new(model)
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(RewriteToolResultHook(IMAGE_JSON))
            .run()
            .await
            .expect("run should succeed with a JSON-shaped rewritten result");

        let messages = result.messages.expect("messages");
        assert!(
            tool_result_text_in_history(&messages, IMAGE_JSON),
            "the JSON-shaped replacement must reach history verbatim as text, not be \
             re-parsed into a structured/image content block"
        );
    }

    /// A hook that patches the model request for the turn (`Flow::PatchRequest`
    /// on `CompletionCall`): forces tool_choice + temperature, narrows the
    /// advertised tools to an allow-list, and injects a passthrough param.
    struct PatchRequestHook;

    impl<M: CompletionModel> AgentHook<M> for PatchRequestHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::CompletionCall { .. } = event {
                Flow::patch_request(
                    RequestPatch::new()
                        .preamble(OVERRIDE_PREAMBLE)
                        .temperature(0.25)
                        .max_tokens(OVERRIDE_MAX_TOKENS)
                        .tool_choice(ToolChoice::Required)
                        .active_tools(["add"])
                        .additional_params(json!({"injected": true})),
                )
            } else {
                Flow::cont()
            }
        }
    }

    const OVERRIDE_PREAMBLE: &str = "overridden: critical-step instructions";
    const OVERRIDE_MAX_TOKENS: u64 = 512;

    /// `Flow::PatchRequest` resolves to a `Patch` completion-call decision
    /// carrying the patch, and is named for fail-closed diagnostics.
    #[test]
    fn patch_request_resolves_to_patch_for_completion_call() {
        let patch = RequestPatch::new()
            .temperature(0.25)
            .tool_choice(ToolChoice::Required);
        match super::flow_into_completion_call(Flow::patch_request(patch.clone())) {
            super::CompletionCallDecision::Patch(got) => assert_eq!(got, patch),
            _ => panic!("PatchRequest should resolve to Patch for a completion call"),
        }
        assert_eq!(
            super::flow_name(&Flow::patch_request(RequestPatch::new())),
            "PatchRequest"
        );
    }

    /// `PatchRequest` is only honored by `CompletionCall`; every other event is
    /// fail-closed, and `CompletionCall` only honors Continue/PatchRequest/Terminate.
    #[test]
    fn patch_request_is_fail_closed_off_the_completion_call_event() {
        let patch = || Flow::patch_request(RequestPatch::new());
        assert!(matches!(
            super::flow_into_invalid(patch()),
            super::InvalidDecision::Terminate(_)
        ));
        assert!(matches!(
            super::flow_into_tool_call(patch()),
            super::ToolCallDecision::Terminate(_)
        ));
        assert!(matches!(
            super::flow_into_tool_result(patch()),
            super::ToolResultDecision::Terminate(_)
        ));
        assert!(super::observe_flow(patch()).is_some());
        // The completion-call event rejects an action it can't honor (e.g. Skip).
        assert!(matches!(
            super::flow_into_completion_call(Flow::skip("x")),
            super::CompletionCallDecision::Terminate(_)
        ));
    }

    /// A `Flow::PatchRequest` hook patches the request for the turn identically
    /// under `run()` and `stream()`: the captured completion request shows the
    /// overridden temperature/tool_choice, the merged additional_params, and the
    /// tool set narrowed to the allow-list — on both drivers.
    #[tokio::test]
    async fn patch_request_parity_across_run_and_stream() {
        fn assert_request(req: &crate::completion::CompletionRequest) {
            assert_eq!(
                req.temperature,
                Some(0.25),
                "override temperature wins over the agent's 0.9"
            );
            assert_eq!(
                req.max_tokens,
                Some(OVERRIDE_MAX_TOKENS),
                "override max_tokens wins over the agent's 64"
            );
            // The override preamble wins and is sent as the leading system message.
            let system = req.chat_history.iter().find_map(|m| match m {
                Message::System { content } => Some(content.as_str()),
                _ => None,
            });
            assert_eq!(
                system,
                Some(OVERRIDE_PREAMBLE),
                "override preamble wins over the agent's baseline and is the leading system message"
            );
            assert!(matches!(req.tool_choice, Some(ToolChoice::Required)));
            let tool_names: Vec<&str> = req.tools.iter().map(|t| t.name.as_str()).collect();
            assert_eq!(
                tool_names,
                ["add"],
                "active_tools narrows the advertised set to `add` (drops `subtract`)"
            );
            // additional_params is shallow-merged: the agent baseline survives and
            // the override's key is added.
            let params = req.additional_params.as_ref().expect("additional_params");
            assert_eq!(
                params.get("baseline").and_then(|v| v.as_str()),
                Some("keep")
            );
            assert_eq!(params.get("injected").and_then(|v| v.as_bool()), Some(true));
        }

        let blocking_model = MockCompletionModel::from_turns([MockTurn::text("done")]);
        let blocking_probe = blocking_model.clone();
        let blocking = AgentBuilder::new(blocking_model)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .preamble("baseline preamble")
            .temperature(0.9)
            .max_tokens(64)
            .additional_params(json!({"baseline": "keep"}))
            .add_hook(PatchRequestHook)
            .build()
            .runner("go")
            .max_turns(2)
            .run()
            .await
            .expect("blocking run should succeed");
        assert_eq!(blocking.output, "done");
        let blocking_requests = blocking_probe.requests();
        assert_eq!(blocking_requests.len(), 1);
        assert_request(&blocking_requests[0]);

        let streaming_model = MockCompletionModel::from_stream_turns([
            ScriptedTurn::Text("done").as_stream_events(StreamShape::Complete)
        ]);
        let streaming_probe = streaming_model.clone();
        let mut stream = AgentBuilder::new(streaming_model)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .preamble("baseline preamble")
            .temperature(0.9)
            .max_tokens(64)
            .additional_params(json!({"baseline": "keep"}))
            .add_hook(PatchRequestHook)
            .build()
            .runner("go")
            .max_turns(2)
            .stream()
            .await;
        while let Some(item) = stream.next().await {
            let _ = item.map_err(|err| panic!("stream item errored: {err}"));
        }
        let streaming_requests = streaming_probe.requests();
        assert_eq!(streaming_requests.len(), 1);
        assert_request(&streaming_requests[0]);
    }

    // --- Hook system v2: extra_context, history view, ModelTurnFinished, chained rewrites ---

    fn hook_doc(id: &str, text: &str) -> crate::completion::Document {
        crate::completion::Document {
            id: id.to_string(),
            text: text.to_string(),
            additional_props: Default::default(),
        }
    }

    /// Injects one extra context document on every completion call.
    struct ExtraContextHook {
        id: &'static str,
        text: &'static str,
    }

    impl<M: CompletionModel> AgentHook<M> for ExtraContextHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::CompletionCall { .. } = event {
                Flow::patch_request(RequestPatch::new().context(hook_doc(self.id, self.text)))
            } else {
                Flow::cont()
            }
        }
    }

    /// Injects an extra context document only on the first turn (to prove
    /// per-turn, non-sticky behavior).
    struct ExtraContextTurnOneHook;

    impl<M: CompletionModel> AgentHook<M> for ExtraContextTurnOneHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::CompletionCall { turn, .. } = event
                && turn == 1
            {
                return Flow::patch_request(
                    RequestPatch::new().context(hook_doc("turn-one", "only turn 1")),
                );
            }
            Flow::cont()
        }
    }

    fn one_text_stream_turn(text: &'static str) -> Vec<MockStreamEvent> {
        vec![
            MockStreamEvent::text(text),
            MockStreamEvent::final_response_with_total_tokens(0),
        ]
    }

    /// A single hook's `extra_context` document appears in the completion request,
    /// after the agent's static context, on both `run()` and `stream()`.
    #[tokio::test]
    async fn extra_context_appears_after_static_context_on_both_surfaces() {
        fn assert_docs(req: &crate::completion::CompletionRequest) {
            let ids: Vec<&str> = req.documents.iter().map(|d| d.id.as_str()).collect();
            let static_pos = ids
                .iter()
                .position(|id| id.starts_with("static_doc"))
                .expect("static context document present");
            let extra_pos = ids
                .iter()
                .position(|id| *id == "hook-doc")
                .expect("hook extra_context document present");
            assert!(
                static_pos < extra_pos,
                "static context precedes hook extras: {ids:?}"
            );
            assert!(
                req.documents.iter().any(|d| d.text == "injected"),
                "the hook document's text is present"
            );
        }

        let blocking_model = MockCompletionModel::from_turns([MockTurn::text("done")]);
        let blocking_probe = blocking_model.clone();
        AgentBuilder::new(blocking_model)
            .context("static context text")
            .add_hook(ExtraContextHook {
                id: "hook-doc",
                text: "injected",
            })
            .build()
            .runner("go")
            .run()
            .await
            .expect("blocking run should succeed");
        assert_docs(blocking_probe.requests().first().expect("one request"));

        let streaming_model =
            MockCompletionModel::from_stream_turns([one_text_stream_turn("done")]);
        let streaming_probe = streaming_model.clone();
        let mut stream = AgentBuilder::new(streaming_model)
            .context("static context text")
            .add_hook(ExtraContextHook {
                id: "hook-doc",
                text: "injected",
            })
            .build()
            .runner("go")
            .stream()
            .await;
        while let Some(item) = stream.next().await {
            let _ = item.map_err(|err| panic!("stream item errored: {err}"));
        }
        assert_docs(streaming_probe.requests().first().expect("one request"));
    }

    /// Two hooks' `extra_context` documents append in registration order.
    #[tokio::test]
    async fn multiple_hooks_extra_context_append_in_registration_order() {
        let model = MockCompletionModel::from_turns([MockTurn::text("done")]);
        let probe = model.clone();
        AgentBuilder::new(model)
            .add_hook(ExtraContextHook {
                id: "first",
                text: "1",
            })
            .add_hook(ExtraContextHook {
                id: "second",
                text: "2",
            })
            .build()
            .runner("go")
            .run()
            .await
            .expect("run should succeed");
        let requests = probe.requests();
        let req = requests.first().expect("one request");
        let ids: Vec<&str> = req.documents.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(
            ids,
            vec!["first", "second"],
            "hook extras append in registration order"
        );
    }

    /// A hook's `extra_context` is per-turn and non-sticky: a document injected on
    /// turn 1 does not reappear on turn 2. Checked on both surfaces.
    #[tokio::test]
    async fn extra_context_is_per_turn_non_sticky() {
        fn assert_turns(requests: &[crate::completion::CompletionRequest]) {
            assert_eq!(requests.len(), 2, "two model turns");
            let turn1 = requests.first().expect("turn 1");
            let turn2 = requests.get(1).expect("turn 2");
            assert!(
                turn1.documents.iter().any(|d| d.id == "turn-one"),
                "turn 1 carries the injected document"
            );
            assert!(
                turn2.documents.iter().all(|d| d.id != "turn-one"),
                "turn 2 does not inherit turn 1's per-turn document"
            );
        }

        let blocking_probe = blocking_model();
        let probe = blocking_probe.clone();
        AgentBuilder::new(blocking_probe)
            .tool(MockAddTool)
            .add_hook(ExtraContextTurnOneHook)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .run()
            .await
            .expect("blocking run should succeed");
        assert_turns(&probe.requests());

        let streaming = streaming_model();
        let stream_probe = streaming.clone();
        let mut stream = AgentBuilder::new(streaming)
            .tool(MockAddTool)
            .add_hook(ExtraContextTurnOneHook)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .stream()
            .await;
        while let Some(item) = stream.next().await {
            let _ = item.map_err(|err| panic!("stream item errored: {err}"));
        }
        assert_turns(&stream_probe.requests());
    }

    /// A hook that overrides `history` changes the messages sent to the provider
    /// for the turn without touching the persisted transcript, on both surfaces.
    #[tokio::test]
    async fn history_patch_changes_sent_messages_not_transcript_on_both_surfaces() {
        const SENTINEL: &str = "COMPACTED-HISTORY-SENTINEL";

        struct HistoryOverrideHook;
        impl<M: CompletionModel> AgentHook<M> for HistoryOverrideHook {
            async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
                if let StepEvent::CompletionCall { .. } = event {
                    Flow::patch_request(RequestPatch::new().history([Message::user(SENTINEL)]))
                } else {
                    Flow::cont()
                }
            }
        }

        fn request_has_sentinel(req: &crate::completion::CompletionRequest) -> bool {
            req.chat_history.iter().any(|m| match m {
                Message::User { content } => content
                    .iter()
                    .any(|c| matches!(c, UserContent::Text(text) if text.text.contains(SENTINEL))),
                _ => false,
            })
        }

        fn messages_have_sentinel(messages: &[Message]) -> bool {
            messages.iter().any(|m| match m {
                Message::User { content } => content
                    .iter()
                    .any(|c| matches!(c, UserContent::Text(text) if text.text.contains(SENTINEL))),
                _ => false,
            })
        }

        let blocking_model = MockCompletionModel::from_turns([MockTurn::text("done")]);
        let blocking_probe = blocking_model.clone();
        let blocking = AgentBuilder::new(blocking_model)
            .add_hook(HistoryOverrideHook)
            .build()
            .runner("real prompt")
            .run()
            .await
            .expect("blocking run should succeed");
        assert!(
            request_has_sentinel(blocking_probe.requests().first().expect("one request")),
            "the overridden history reaches the provider"
        );
        assert!(
            !messages_have_sentinel(blocking.messages.as_deref().unwrap_or_default()),
            "the persisted transcript is untouched by the per-turn history override"
        );

        let streaming_model =
            MockCompletionModel::from_stream_turns([one_text_stream_turn("done")]);
        let streaming_probe = streaming_model.clone();
        let stream = AgentBuilder::new(streaming_model)
            .add_hook(HistoryOverrideHook)
            .build()
            .runner("real prompt")
            .stream()
            .await;
        let final_response = drive_to_final_response(stream).await;
        assert!(
            request_has_sentinel(streaming_probe.requests().first().expect("one request")),
            "the overridden history reaches the provider on the streaming surface too"
        );
        assert!(
            !messages_have_sentinel(final_response.history().expect("history")),
            "the persisted transcript is untouched by the per-turn history override on \
             the streaming surface too"
        );
    }

    /// `ModelTurnFinished` fires exactly once per accepted turn on both surfaces,
    /// including a streamed tool-only turn that fires no `StreamResponseFinish`.
    #[tokio::test]
    async fn model_turn_finished_fires_once_per_accepted_turn_including_tool_only() {
        let blocking_hook = RecordingHook::default();
        AgentBuilder::new(blocking_model())
            .tool(MockAddTool)
            .add_hook(blocking_hook.clone())
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .run()
            .await
            .expect("blocking run should succeed");
        assert_eq!(
            blocking_hook.count(StepEventKind::ModelTurnFinished),
            2,
            "one ModelTurnFinished per accepted turn (tool turn + text turn)"
        );

        let streaming_hook = RecordingHook::default();
        let mut stream = AgentBuilder::new(streaming_model())
            .tool(MockAddTool)
            .add_hook(streaming_hook.clone())
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .stream()
            .await;
        while let Some(item) = stream.next().await {
            let _ = item.map_err(|err| panic!("stream item errored: {err}"));
        }
        assert_eq!(
            streaming_hook.count(StepEventKind::ModelTurnFinished),
            2,
            "ModelTurnFinished fires once per turn on the streaming surface too"
        );
        // The tool-only first turn streams no assistant text, so only the second
        // (text) turn fires StreamResponseFinish — proving ModelTurnFinished
        // covers the gap.
        assert_eq!(
            streaming_hook.count(StepEventKind::StreamResponseFinish),
            1,
            "the tool-only turn fires no StreamResponseFinish"
        );
    }

    /// Records the content kinds of the first turn's `ModelTurnFinished`.
    #[derive(Clone, Default)]
    struct CaptureFirstTurnContent {
        kinds: Arc<Mutex<Option<Vec<&'static str>>>>,
    }

    impl<M: CompletionModel> AgentHook<M> for CaptureFirstTurnContent {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::ModelTurnFinished { turn, content, .. } = event
                && turn == 1
            {
                let kinds = content
                    .iter()
                    .map(|c| match c {
                        AssistantContent::Reasoning(_) => "reasoning",
                        AssistantContent::Text(_) => "text",
                        AssistantContent::ToolCall(_) => "tool_call",
                        _ => "other",
                    })
                    .collect();
                *self.kinds.lock().expect("kinds") = Some(kinds);
            }
            Flow::cont()
        }
    }

    /// On the streaming surface, `ModelTurnFinished.content` carries the
    /// **canonical** committed content from `StreamedTurn::finish` (reasoning →
    /// text → tool calls), not the raw `stream.choice` aggregate. The turn streams
    /// reasoning, then a tool call, then text (a non-canonical emission order), so
    /// a raw-choice implementation would surface `reasoning, tool_call, text` —
    /// the canonical event instead reports `reasoning, text, tool_call`.
    #[tokio::test]
    async fn streaming_model_turn_finished_carries_canonical_committed_content() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::reasoning("think"),
                MockStreamEvent::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
                MockStreamEvent::text("answer"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let hook = CaptureFirstTurnContent::default();
        let stream = AgentBuilder::new(model)
            .tool(MockAddTool)
            .add_hook(hook.clone())
            .build()
            .runner("go")
            .max_turns(3)
            .stream()
            .await;
        let _ = drive_to_final_response(stream).await;

        assert_eq!(
            hook.kinds.lock().expect("kinds").clone(),
            Some(vec!["reasoning", "text", "tool_call"]),
            "ModelTurnFinished carries the canonical reasoning->text->tool ordering \
             from StreamedTurn::finish, not the raw stream.choice emission order"
        );
    }

    /// `RewriteArgs` and `RewriteResult` chain across hooks: a later hook observes
    /// (and further rewrites) the value produced by earlier hooks.
    #[tokio::test]
    async fn chained_rewrites_compose_across_hooks() {
        /// Sets one key of the tool arguments, preserving the rest.
        struct SetArg {
            key: &'static str,
            value: i64,
        }
        impl<M: CompletionModel> AgentHook<M> for SetArg {
            async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
                if let StepEvent::ToolCall { args, .. } = event {
                    let mut parsed: serde_json::Value =
                        serde_json::from_str(args).unwrap_or_else(|_| json!({}));
                    parsed[self.key] = json!(self.value);
                    Flow::rewrite_args(parsed)
                } else {
                    Flow::cont()
                }
            }
        }

        /// Wraps the tool result in `label(...)`.
        struct WrapResult(&'static str);
        impl<M: CompletionModel> AgentHook<M> for WrapResult {
            async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
                if let StepEvent::ToolResult { result, .. } = event {
                    Flow::rewrite_result(format!("{}({})", self.0, result))
                } else {
                    Flow::cont()
                }
            }
        }

        // The model asks add(2, 3). SetArg{y:40} then SetArg{x:100} chain, so the
        // tool runs with (100, 40) = 140 — proving arg rewrites compose. Then
        // WrapResult "A" and "B" chain, and a trailing recorder observes the fully
        // chained result "B(A(140))".
        let recorder = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model())
            .tool(MockAddTool)
            .add_hook(SetArg {
                key: "y",
                value: 40,
            })
            .add_hook(SetArg {
                key: "x",
                value: 100,
            })
            .add_hook(WrapResult("A"))
            .add_hook(WrapResult("B"))
            .add_hook(recorder.clone())
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .run()
            .await
            .expect("blocking run should succeed");
        assert_eq!(blocking.output, "the answer is 5");
        assert_eq!(
            recorder.tool_results(),
            vec!["B(A(140))".to_string()],
            "arg rewrites compose (100+40=140) and result rewrites nest B(A(...))"
        );

        // Same on the streaming surface.
        let stream_recorder = RecordingHook::default();
        let mut stream = AgentBuilder::new(streaming_model())
            .tool(MockAddTool)
            .add_hook(SetArg {
                key: "y",
                value: 40,
            })
            .add_hook(SetArg {
                key: "x",
                value: 100,
            })
            .add_hook(WrapResult("A"))
            .add_hook(WrapResult("B"))
            .add_hook(stream_recorder.clone())
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .stream()
            .await;
        while let Some(item) = stream.next().await {
            let _ = item.map_err(|err| panic!("stream item errored: {err}"));
        }
        assert_eq!(
            stream_recorder.tool_results(),
            vec!["B(A(140))".to_string()],
            "chained rewrites compose identically on the streaming surface"
        );
    }

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Answer {
        answer: String,
    }

    /// A real tool whose name equals the default synthetic output-tool name
    /// (`final_result`). Used to prove a per-turn `active_tools` filter cannot
    /// make the picked output-tool name collide with it.
    struct FinalResultTool;

    impl Tool for FinalResultTool {
        const NAME: &'static str = "final_result";
        type Error = MockToolError;
        type Args = serde_json::Value;
        type Output = String;

        async fn definition(&self, _prompt: String) -> ToolDefinition {
            ToolDefinition {
                name: Self::NAME.to_string(),
                description: "A real tool sharing the default output-tool name".to_string(),
                parameters: json!({ "type": "object", "properties": {} }),
            }
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("real final_result output".to_string())
        }
    }

    /// Narrows the advertised tools to `add` for the turn, filtering out the real
    /// `final_result` tool.
    struct ActiveToolsAddOnly;

    impl<M: CompletionModel> AgentHook<M> for ActiveToolsAddOnly {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            if let StepEvent::CompletionCall { .. } = event {
                Flow::patch_request(RequestPatch::new().active_tools(["add"]))
            } else {
                Flow::cont()
            }
        }
    }

    /// Regression guard: a per-turn `active_tools` allow-list that filters out a
    /// real tool whose name equals the default synthetic output-tool name must not
    /// let the picked output-tool name collide with that (filtered) real tool. The
    /// name is pinned for the whole run, so picking it against the FULL advertised
    /// set — not just this turn's narrowed executable set — keeps it collision-safe
    /// once the filter lifts on a later turn. With the bug, the output tool would
    /// be named `final_result` (picked against the narrowed `{add}`), colliding
    /// with the real `final_result` whenever the filter is gone.
    #[tokio::test]
    async fn active_tools_filter_does_not_let_output_tool_collide_with_a_filtered_real_tool() {
        // The model finalizes by calling the (correctly-picked) output tool, so a
        // run on the fixed code completes cleanly in a single turn. Asserting the
        // run succeeds also exercises finalization: the model's call to
        // `final_result_1` must be intercepted as the output tool, so this fails if
        // the picked name and the intercept name ever drift apart.
        let model = MockCompletionModel::from_turns([MockTurn::tool_call(
            "out1",
            "final_result_1",
            json!({ "answer": "done" }),
        )]);
        let probe = model.clone();
        let response = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool(FinalResultTool)
            .output_schema::<Answer>()
            .output_mode(OutputMode::Tool)
            .add_hook(ActiveToolsAddOnly)
            .build()
            .runner("go")
            .max_turns(2)
            .run()
            .await
            .expect("run should finalize via the picked output tool `final_result_1`");
        assert!(
            response.output.contains("done"),
            "the intercepted output-tool call should produce the structured result, \
             got {:?}",
            response.output
        );

        let requests = probe.requests();
        assert!(
            !requests.is_empty(),
            "the first model request should be captured"
        );
        let tool_names: Vec<&str> = requests[0].tools.iter().map(|t| t.name.as_str()).collect();
        assert!(
            tool_names.contains(&"add"),
            "active_tools keeps `add` advertised, saw {tool_names:?}"
        );
        assert!(
            tool_names.contains(&"final_result_1"),
            "the synthetic output tool must avoid the filtered real `final_result` name, \
             saw {tool_names:?}"
        );
        assert!(
            !tool_names.contains(&"final_result"),
            "the real `final_result` is filtered out and the output tool must not reuse \
             its name, saw {tool_names:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Human-in-the-loop (HITL): one hook gates each tool call behind a human
    // decision, mapping approve/deny/edit/abort onto the existing Flow actions
    // (cont / skip / rewrite_args / terminate). The runnable interactive
    // version lives in `examples/agent_with_human_in_the_loop`.
    // -----------------------------------------------------------------------

    /// A human reviewer's decision for a pending tool call.
    enum Decision {
        /// Run the tool as the model requested.
        Approve,
        /// Don't run the tool; feed `reason` back to the model as the result.
        Deny(&'static str),
        /// Run the tool with these arguments instead of the model's.
        Edit(serde_json::Value),
        /// Abort the whole run with this reason.
        Abort(&'static str),
    }

    /// Simulates a human reviewer by popping a scripted decision per `ToolCall`
    /// and mapping it to the matching `Flow`. A real reviewer would `.await`
    /// interactive input here (the hook is async) rather than read a queue.
    #[derive(Clone)]
    struct HumanApprovalHook {
        decisions: Arc<Mutex<std::collections::VecDeque<Decision>>>,
        reviewed: Arc<Mutex<Vec<String>>>,
    }

    impl HumanApprovalHook {
        fn new(decisions: impl IntoIterator<Item = Decision>) -> Self {
            Self {
                decisions: Arc::new(Mutex::new(decisions.into_iter().collect())),
                reviewed: Arc::new(Mutex::new(Vec::new())),
            }
        }

        /// `"name(args)"` for each call presented for review, in order.
        fn reviewed(&self) -> Vec<String> {
            self.reviewed.lock().unwrap().clone()
        }
    }

    impl<M: CompletionModel> AgentHook<M> for HumanApprovalHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            let StepEvent::ToolCall {
                tool_name, args, ..
            } = event
            else {
                return Flow::cont();
            };
            self.reviewed
                .lock()
                .unwrap()
                .push(format!("{tool_name}({args})"));
            let decision = self.decisions.lock().unwrap().pop_front();
            match decision {
                Some(Decision::Approve) => Flow::cont(),
                Some(Decision::Deny(reason)) => Flow::skip(reason),
                Some(Decision::Edit(args)) => Flow::rewrite_args(args),
                Some(Decision::Abort(reason)) => Flow::terminate(reason),
                // Fail closed if the script is exhausted (it shouldn't be) — deny
                // rather than silently approve, matching the example's contract.
                None => Flow::skip("denied: no scripted decision (fail-closed)"),
            }
        }
    }

    /// A HITL hook that approves the first tool call, denies the second, and
    /// edits the third's arguments behaves identically under `run()` and
    /// `stream()`: approved/edited tools execute (and the edit takes effect),
    /// the denied tool runs nothing while its reason reaches the model, and the
    /// model-visible history is identical across drivers (compared structurally).
    #[tokio::test]
    async fn human_in_the_loop_approve_deny_edit_parity_across_run_and_stream() {
        // One turn issues three tool calls; the reviewer decides each differently.
        let turns = [
            ScriptedTurn::ToolCalls(vec![
                add_call("tc1", 2, 3),   // approve -> runs, 2 + 3 = 5
                add_call("tc2", 10, 20), // deny    -> skipped; model sees the reason
                add_call("tc3", 1, 1),   // edit    -> runs 1 + 100 = 101, not 1 + 1 = 2
            ]),
            ScriptedTurn::Text("done"),
        ];
        let denial = "denied by reviewer: amount too large";
        let decisions = || {
            vec![
                Decision::Approve,
                Decision::Deny(denial),
                Decision::Edit(json!({"x": 1, "y": 100})),
            ]
        };

        let blocking_model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let blocking_recorder = RecordingHook::default();
        let blocking_approver = HumanApprovalHook::new(decisions());
        let blocking = AgentBuilder::new(blocking_model)
            .tool(MockAddTool)
            .build()
            .runner("carry out the plan")
            .max_turns(3)
            .add_hook(blocking_recorder.clone())
            .add_hook(blocking_approver.clone())
            .run()
            .await
            .expect("blocking HITL run should succeed");

        let streaming_model = MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        );
        let streaming_recorder = RecordingHook::default();
        let streaming_approver = HumanApprovalHook::new(decisions());
        let mut stream = AgentBuilder::new(streaming_model)
            .tool(MockAddTool)
            .build()
            .runner("carry out the plan")
            .max_turns(3)
            .add_hook(streaming_recorder.clone())
            .add_hook(streaming_approver.clone())
            .stream()
            .await;
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let Ok(MultiTurnStreamItem::FinalResponse(resp)) =
                item.map_err(|err| panic!("stream item errored: {err}"))
            {
                final_response = Some(resp);
            }
        }
        let final_response = final_response.expect("stream should yield a final response");

        // Approved (5) and edited (101) tools executed, in call order; the denied
        // call executed nothing, so it fired no ToolResult — on both drivers.
        assert_eq!(
            blocking_recorder.tool_results(),
            vec!["5".to_string(), "101".to_string()]
        );
        assert_eq!(
            blocking_recorder.tool_results(),
            streaming_recorder.tool_results()
        );

        // The denied call (10 + 20) never executed, so its result 30 is absent —
        // this, with tool_results == [5, 101], rules out the test passing if deny
        // were silently treated as approve.
        assert!(
            !blocking_recorder.tool_results().contains(&"30".to_string()),
            "the denied call must not have executed"
        );

        // The reviewer was consulted for all three calls, in order, identically per
        // driver — pinning each decision to its call (approve=2+3, deny=10+20,
        // edit=the third).
        let reviewed = blocking_approver.reviewed();
        assert_eq!(reviewed.len(), 3);
        assert_eq!(reviewed, streaming_approver.reviewed());
        assert!(
            reviewed[0].contains('2') && reviewed[0].contains('3'),
            "first reviewed call should be add(2, 3): {reviewed:?}"
        );
        assert!(
            reviewed[1].contains("10") && reviewed[1].contains("20"),
            "the denied (second) call should be add(10, 20): {reviewed:?}"
        );

        assert_eq!(blocking.output, "done");
        assert_eq!(final_response.response(), blocking.output);
        assert_eq!(
            blocking_recorder.shared_events(),
            streaming_recorder.shared_events()
        );

        // Model-visible history is identical across drivers (compared structurally
        // as serde_json::Value) and carries the denial reason and the edited result
        // 101 (not the model's 1 + 1 = 2).
        let blocking_messages = blocking.messages.expect("blocking messages");
        let streaming_messages = final_response
            .history()
            .expect("streaming history")
            .to_vec();
        assert_eq!(
            serde_json::to_value(&blocking_messages).expect("serialize blocking"),
            serde_json::to_value(&streaming_messages).expect("serialize streaming"),
        );
        assert!(
            tool_result_text_in_history(&blocking_messages, denial),
            "the denial reason must be the denied call's tool result in the history"
        );
        assert!(
            tool_result_text_in_history(&blocking_messages, "101"),
            "the edited call must have executed with the rewritten arguments"
        );
    }

    /// A HITL hook that aborts a tool call (`Decision::Abort` -> `Flow::terminate`)
    /// stops the run and surfaces the reason as a `PromptCancelled` error — on both
    /// the blocking and streaming drivers.
    #[tokio::test]
    async fn human_in_the_loop_abort_terminates_the_run() {
        let turns = [
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
            ScriptedTurn::Text("unreachable"),
        ];
        const ABORT_REASON: &str = "aborted by the human reviewer";

        // Blocking driver: the run resolves to a PromptCancelled error.
        let blocking_model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let err = AgentBuilder::new(blocking_model)
            .tool(MockAddTool)
            .build()
            .runner("do the sensitive thing")
            .max_turns(3)
            .add_hook(HumanApprovalHook::new([Decision::Abort(ABORT_REASON)]))
            .run()
            .await
            .expect_err("an aborted tool call should terminate the blocking run");
        assert!(
            format!("{err}").contains(ABORT_REASON),
            "the abort reason should surface in the blocking error, got: {err}"
        );

        // Streaming driver: the stream yields an error carrying the same reason and
        // never reaches the "unreachable" final text.
        let streaming_model = MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        );
        let mut stream = AgentBuilder::new(streaming_model)
            .tool(MockAddTool)
            .build()
            .runner("do the sensitive thing")
            .max_turns(3)
            .add_hook(HumanApprovalHook::new([Decision::Abort(ABORT_REASON)]))
            .stream()
            .await;
        let mut stream_error = None;
        while let Some(item) = stream.next().await {
            match item {
                Err(err) => stream_error = Some(format!("{err}")),
                Ok(MultiTurnStreamItem::FinalResponse(resp)) => {
                    panic!("aborted stream must not finalize, got: {}", resp.response())
                }
                Ok(_) => {}
            }
        }
        let stream_error = stream_error.expect("an aborted tool call should error the stream");
        assert!(
            stream_error.contains(ABORT_REASON),
            "the abort reason should surface in the streaming error, got: {stream_error}"
        );
    }

    /// A non-interactive *policy* HITL hook: auto-approve an allow-list, deny
    /// everything else (fail-closed), and cache each decision so a repeated tool
    /// is not re-evaluated ("sticky", like the OpenAI Agents SDK's
    /// `always_approve`). Backs `examples/agent_with_approval_policy`.
    #[derive(Clone)]
    struct PolicyHook {
        auto_approve: std::collections::HashSet<&'static str>,
        /// Tool names the policy actually evaluated (cache misses), in order.
        evaluated: Arc<Mutex<Vec<String>>>,
        /// Sticky cache of prior decisions, keyed by tool name.
        cache: Arc<Mutex<std::collections::HashMap<String, bool>>>,
    }

    impl PolicyHook {
        fn new(auto_approve: impl IntoIterator<Item = &'static str>) -> Self {
            Self {
                auto_approve: auto_approve.into_iter().collect(),
                evaluated: Arc::new(Mutex::new(Vec::new())),
                cache: Arc::new(Mutex::new(std::collections::HashMap::new())),
            }
        }

        fn evaluated(&self) -> Vec<String> {
            self.evaluated.lock().unwrap().clone()
        }
    }

    impl<M: CompletionModel> AgentHook<M> for PolicyHook {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            let StepEvent::ToolCall { tool_name, .. } = event else {
                return Flow::cont();
            };
            let cached = self.cache.lock().unwrap().get(tool_name).copied();
            let approved = match cached {
                Some(decision) => decision, // sticky: reuse without re-evaluating
                None => {
                    self.evaluated.lock().unwrap().push(tool_name.to_string());
                    let decision = self.auto_approve.contains(tool_name);
                    self.cache
                        .lock()
                        .unwrap()
                        .insert(tool_name.to_string(), decision);
                    decision
                }
            };
            if approved {
                Flow::cont()
            } else {
                Flow::skip(format!("denied by policy: `{tool_name}` not allowed"))
            }
        }
    }

    /// The policy hook auto-approves `add` and denies `subtract`, and its decision
    /// is sticky: a second `add` call reuses the cached approval instead of being
    /// re-evaluated. The denied call never runs and its reason reaches the model.
    #[tokio::test]
    async fn approval_policy_allow_list_with_sticky_decisions() {
        // One turn issues three calls: add, subtract (denied), add again (sticky).
        let turns = [
            ScriptedTurn::ToolCalls(vec![
                add_call("c1", 2, 3),
                ScriptedToolCall {
                    id: "c2",
                    name: "subtract",
                    args: json!({ "x": 10, "y": 4 }),
                },
                add_call("c3", 2, 3),
            ]),
            ScriptedTurn::Text("done"),
        ];

        let model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let recorder = RecordingHook::default();
        let policy = PolicyHook::new(["add"]);
        let out = AgentBuilder::new(model)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .build()
            .runner("go")
            .max_turns(3)
            .add_hook(recorder.clone())
            .add_hook(policy.clone())
            .run()
            .await
            .expect("policy run should succeed");

        assert_eq!(out.output, "done");
        // `add` ran twice (auto-approved, then sticky-reused); `subtract` was denied
        // and executed nothing.
        assert_eq!(
            recorder.tool_results(),
            vec!["5".to_string(), "5".to_string()]
        );
        // The policy evaluated each distinct tool once; the second `add` reused the
        // cached decision rather than being re-evaluated.
        assert_eq!(
            policy.evaluated(),
            vec!["add".to_string(), "subtract".to_string()]
        );
        let messages = out.messages.expect("messages");
        assert!(
            tool_result_text_in_history(&messages, "denied by policy: `subtract` not allowed"),
            "the policy denial reason must reach the model as the subtract tool result"
        );
    }
}
