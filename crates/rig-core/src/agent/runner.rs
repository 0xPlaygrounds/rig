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
        AgentHook, CompletionCall, CompletionCallAction,
        CompletionResponse as CompletionResponseEvent, HookContext, HookStack,
        InvalidToolCallAction, ModelTurnFinished, ObservationAction, RequestPatch,
        ToolCall as ToolCallEvent, ToolCallAction, ToolResultAction, ToolResultEvent,
    },
    prompt_request::{
        PromptResponse,
        streaming::{
            DriveItem, DriveStream, MultiTurnStreamItem, StreamingError, TurnSource, drive_agent,
            drive_tool_calls, record_usage_on_span, streaming_error_into_prompt,
        },
        tool_result_output,
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
    tool::{
        ToolContext, ToolDispatch, ToolOutput, ToolResult,
        server::{ToolRegistrySnapshot, ToolServerHandle},
    },
};

use super::UNKNOWN_AGENT_NAME;

/// Build the per-turn `chat` span shared by both turn sources.
///
/// The span *name* must be a string literal — `tracing` bakes it into static
/// metadata — so this is a macro parameterized by the name rather than a
/// function (the two surfaces keep distinct names, `chat` vs `chat_streaming`,
/// which dashboards split on). The matching operation value is passed with the
/// name; every other field is identical across the two surfaces, so it lives
/// here once instead of being copy-pasted into each `TurnSource::open_chat_span`.
macro_rules! build_chat_span {
    ($runner:expr, $effective_preamble:expr, $name:literal, $operation:literal) => {
        ::tracing::info_span!(
            target: "rig::agent_chat",
            parent: ::tracing::Span::current(),
            $name,
            gen_ai.operation.name = $operation,
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

/// Convert an observe-only action into an optional stop reason.
pub(crate) fn observe_action(action: ObservationAction) -> Option<String> {
    match action {
        ObservationAction::Continue => None,
        ObservationAction::Stop(reason) => Some(reason),
    }
}

pub(crate) enum ToolCallDecision {
    Proceed,
    ProceedWith(serde_json::Value),
    Skip(String),
    Terminate(String),
}

pub(crate) fn tool_call_decision(action: ToolCallAction) -> ToolCallDecision {
    match action {
        ToolCallAction::Run => ToolCallDecision::Proceed,
        ToolCallAction::Rewrite(args) => ToolCallDecision::ProceedWith(args),
        ToolCallAction::Skip(reason) => ToolCallDecision::Skip(reason),
        ToolCallAction::Stop(reason) => ToolCallDecision::Terminate(reason),
    }
}

pub(crate) enum ToolResultDecision {
    Keep,
    Replace(ToolOutput),
    Terminate(String),
}

pub(crate) fn tool_result_decision(action: ToolResultAction) -> ToolResultDecision {
    match action {
        ToolResultAction::Keep => ToolResultDecision::Keep,
        ToolResultAction::Rewrite(result) => ToolResultDecision::Replace(result),
        ToolResultAction::Stop(reason) => ToolResultDecision::Terminate(reason),
    }
}

pub(crate) enum CompletionCallDecision {
    Proceed,
    Patch(RequestPatch),
    Terminate(String),
}

pub(crate) fn completion_call_decision(action: CompletionCallAction) -> CompletionCallDecision {
    match action {
        CompletionCallAction::Continue => CompletionCallDecision::Proceed,
        CompletionCallAction::Patch(patch) => CompletionCallDecision::Patch(patch),
        CompletionCallAction::Stop(reason) => CompletionCallDecision::Terminate(reason),
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
    /// Typed context cloned freshly for every tool dispatch.
    pub(crate) tool_context: ToolContext,
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
            max_turns: agent.default_max_turns.unwrap_or(1),
            max_invalid_tool_call_retries: 0,
            model: agent.model.clone(),
            agent_name: agent.name.clone(),
            preamble: agent.preamble.clone(),
            static_context: agent.static_context.clone(),
            temperature: agent.temperature,
            max_tokens: agent.max_tokens,
            additional_params: agent.additional_params.clone(),
            tool_server_handle: agent.tool_server_handle.clone(),
            tool_context: ToolContext::new(),
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
    /// events use their event-specific stop action). See the
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
    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. Zero emits no model calls; one permits only the
    /// initial call. Exceeding the budget returns [`PromptError::MaxTurnsError`].
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Set the typed context cloned for every tool dispatch in this run.
    pub fn tool_context(mut self, context: ToolContext) -> Self {
        self.tool_context = context;
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
    /// For the streaming path: the driver emits *all* of a turn's `ToolCall`
    /// stream items eagerly (in call order) when the model turn commits, then —
    /// only after the whole tool batch settles successfully — surfaces the
    /// per-tool `ToolExecutionCommitted` and `ToolResult` stream items in **call
    /// order** (never completion order), for the tools whose body actually ran.
    /// The persisted message history is unchanged.
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
    /// retries also consume the total model-call budget.
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

/// Fire the event-specific completion-call hook for a turn.
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
    match completion_call_decision(
        hooks
            .on_completion_call(
                ctx,
                CompletionCall {
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

/// Whether (and how) a tool call executed, for [`run_single_tool`].
pub(crate) enum ToolExecution {
    /// The tool's body ran. Carries the **effective** tool call — the model's
    /// call with any [`ToolCallAction::Rewrite`] hook
    /// rewrite applied — so the driver can surface it in the
    /// [`ToolExecutionCommitted`](crate::agent::prompt_request::streaming::MultiTurnStreamItem::ToolExecutionCommitted)
    /// event (what actually ran, not the model's original arguments). Boxed to
    /// keep this enum small (a `ToolCall` is large next to the empty `Skipped`).
    Executed(Box<ToolCall>),
    /// A tool-call hook returned [`ToolCallAction::Skip`]: the
    /// body did not run, so no execution-commit is surfaced — but the skip result
    /// is still delivered to the model (and surfaced as a `ToolResult`).
    Skipped,
}

/// Outcome of [`run_single_tool`]: the tool-result content plus whether the
/// tool's body ran (and the effective call) or a hook skipped it.
pub(crate) struct ToolCallOutcome {
    /// The tool result delivered to the model (a real output, a redacted
    /// replacement, or a hook skip reason).
    pub content: UserContent,
    /// How the call resolved: executed (with the effective tool call) or skipped.
    pub execution: ToolExecution,
}

/// Execute a single tool call, firing the `ToolCall` and `ToolResult` hooks and
/// shaping the result. **Shared by the blocking and streaming drivers** so a
/// tool call behaves identically in both: same hook events, same fail-closed
/// skip/terminate handling, and the same result shaping. Hook skips become
/// [`ToolResult::skipped`], and every result is converted directly into typed
/// message content through [`tool_result_output`] without reparsing text.
/// Records `gen_ai.tool.*` on the current span;
/// `error_history` builds a cancellation error if a hook terminates the run.
/// Returns whether the tool body executed via [`ToolCallOutcome::execution`].
pub(crate) async fn run_single_tool<M>(
    hooks: &HookStack<M>,
    ctx: &HookContext,
    tool_snapshot: &ToolRegistrySnapshot,
    tool_context: &ToolContext,
    tool_call: &ToolCall,
    internal_call_id: &str,
    error_history: &[Message],
) -> Result<ToolCallOutcome, PromptError>
where
    M: CompletionModel,
{
    let tool_name = &tool_call.function.name;
    // `mut` so a tool-call hook can rewrite the arguments the tool
    // runs with (the model's emitted arguments are otherwise used verbatim).
    let mut args = json_utils::serialize_json_value(&tool_call.function.arguments);

    let tool_span = tracing::Span::current();
    tool_span.record("gen_ai.tool.name", tool_name);
    tool_span.record("gen_ai.tool.call.id", &tool_call.id);
    tool_span.record("gen_ai.tool.call.arguments", &args);

    // Resolve the `ToolCall` hook chain. A proceeding chain carries any
    // `ToolCallAction::Rewrite` in the action itself (→ `ProceedWith`); a chain that a
    // later hook short-circuits with `Skip`/`Terminate` salvages the accumulated
    // rewrite into `salvaged_rewrite` so it is *not* lost — the rewritten args
    // must still be reported on the skipped `ToolResult` and in tracing rather
    // than leaking the model's original args (see [`HookStack::resolve_tool_call`]).
    let (action, salvaged_rewrite) = hooks
        .resolve_tool_call(
            ctx,
            ToolCallEvent {
                tool_name,
                tool_call_id: tool_call.call_id.as_deref(),
                internal_call_id,
                args: &args,
            },
        )
        .await;

    // Apply a salvaged rewrite (short-circuit path only) so `args` — what the
    // `ToolResult` reports — and the span reflect the effective arguments.
    if let Some(rewritten) = salvaged_rewrite.as_ref() {
        args = json_utils::serialize_json_value(rewritten);
        tool_span.record("gen_ai.tool.call.arguments", &args);
        tracing::debug!(
            tool_name = tool_name,
            "tool-call arguments rewritten by a hook"
        );
    }

    // On `Skip` the body does not run and the structured outcome is `Skipped`;
    // otherwise the tool executes into a structured `ToolResult`.
    // `effective_args` is what the tool actually ran with (the model's, a hook's
    // `ToolCallAction::Rewrite` replacement, or a salvaged rewrite) — surfaced in the
    // execution-commit event so a redaction rewrite does not leak. Unused for a skip.
    let mut skipped: Option<ToolResult> = None;
    let effective_args: serde_json::Value = match tool_call_decision(action) {
        ToolCallDecision::Terminate(reason) => {
            return Err(PromptError::prompt_cancelled(
                error_history.to_vec(),
                reason,
            ));
        }
        ToolCallDecision::Skip(reason) => {
            tracing::info!(tool_name = tool_name, reason = reason, "Tool call rejected");
            // Synthetic rejection: `Skipped` outcome, message delivered verbatim.
            // Still fires the `ToolResult` hook so a policy observes the skip.
            skipped = Some(ToolResult::skipped(reason));
            // A skip runs nothing; its effective args are the salvaged rewrite
            // (if any) so tracing/history stay consistent, though they go unused.
            salvaged_rewrite.unwrap_or_else(|| tool_call.function.arguments.clone())
        }
        ToolCallDecision::ProceedWith(replacement) => {
            // Proceeding rewrite: re-record the span so the trace, and the
            // downstream `ToolResult` event, reflect what the tool actually
            // received rather than what the model emitted.
            args = json_utils::serialize_json_value(&replacement);
            tool_span.record("gen_ai.tool.call.arguments", &args);
            tracing::debug!(
                tool_name = tool_name,
                "tool-call arguments rewritten by a hook"
            );
            replacement
        }
        ToolCallDecision::Proceed => tool_call.function.arguments.clone(),
    };

    // Resolve the structured execution result and how the call surfaced. A skip
    // produces no execution-commit event; a real execution carries the effective
    // tool call (the model's call with any `ToolCallAction::Rewrite` applied).
    let (exec, execution, dispatch_context) = match skipped {
        Some(exec) => (exec, ToolExecution::Skipped, tool_context.for_dispatch()),
        None => {
            let mut effective_tool_call = tool_call.clone();
            effective_tool_call.function.arguments = effective_args;
            let ToolDispatch {
                result: exec,
                context: dispatch_context,
            } = tool_snapshot.dispatch(tool_name, &args, tool_context).await;
            (
                exec,
                ToolExecution::Executed(Box::new(effective_tool_call)),
                dispatch_context,
            )
        }
    };
    // Presentation rewrites happen after execution. The raw structured result
    // and per-dispatch context remain unchanged for every hook.
    let result_decision = tool_result_decision(
        hooks
            .on_tool_result(
                ctx,
                ToolResultEvent {
                    tool_name,
                    tool_call_id: tool_call.call_id.as_deref(),
                    internal_call_id,
                    args: &args,
                    presentation: exec.output(),
                    raw_result: &exec,
                    tool_context: &dispatch_context,
                },
            )
            .await,
    );
    // Outcome metadata describes the execution itself, while result content
    // follows the same presentation policy as the model. This keeps redaction
    // and stop hooks from leaking raw tool output through telemetry.
    record_tool_result(&tool_span, &exec);

    match result_decision {
        ToolResultDecision::Terminate(reason) => Err(PromptError::prompt_cancelled(
            error_history.to_vec(),
            reason,
        )),
        ToolResultDecision::Replace(replacement) => {
            tool_span.record("gen_ai.tool.call.result", replacement.render());
            Ok(ToolCallOutcome {
                content: tool_result_output(
                    tool_call.id.clone(),
                    tool_call.call_id.clone(),
                    replacement,
                ),
                execution,
            })
        }
        ToolResultDecision::Keep => {
            tool_span.record("gen_ai.tool.call.result", exec.output().render());
            let content = tool_result_output(
                tool_call.id.clone(),
                tool_call.call_id.clone(),
                exec.output().clone(),
            );
            Ok(ToolCallOutcome { content, execution })
        }
    }
}

fn record_tool_result(span: &tracing::Span, result: &ToolResult) {
    span.record("gen_ai.tool.call.outcome", result.status_name());
    if let Some(error) = result.error() {
        span.record("gen_ai.tool.error.type", error.kind().as_str());
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
        gen_ai.tool.call.result = tracing::field::Empty,
        gen_ai.tool.call.outcome = tracing::field::Empty,
        gen_ai.tool.error.type = tracing::field::Empty
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
        let chat_span = build_chat_span!(runner, effective_preamble, "chat", "chat");
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
                        let action = runner
                            .hooks
                            .on_invalid_tool_call(hook_ctx, &context)
                            .await
                            .unwrap_or_else(InvalidToolCallAction::fail);
                        outcome = match run.resolve_invalid_tool_call(action) {
                            Ok(outcome) => outcome,
                            Err(err) => {
                                yield Err(Box::new(err).into());
                                return;
                            }
                        };
                    }
                    ModelTurnOutcome::TurnRetried => break,
                    ModelTurnOutcome::Continue {
                        response_hook_suppressed,
                    } => {
                        if !response_hook_suppressed {
                            // The medium-specific raw response event fires first,
                            // then the normalized per-turn event. Both are
                            // observe-only and suppressed for recovered turns.
                            if let Some(reason) = observe_action(
                                runner
                                    .hooks
                                    .on_completion_response(
                                        hook_ctx,
                                        CompletionResponseEvent {
                                            prompt: &current_prompt,
                                            response: &resp,
                                        },
                                    )
                                    .await,
                            ) {
                                yield Err(StreamingError::Prompt(Box::new(run.cancel_error(reason))));
                                return;
                            }
                            if let Some(reason) = observe_action(
                                runner
                                    .hooks
                                    .on_model_turn_finished(
                                        hook_ctx,
                                        ModelTurnFinished {
                                            turn: hook_ctx.turn(),
                                            content: &resp.choice,
                                            usage: resp.usage,
                                        },
                                    )
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
        tool_snapshot: Arc<ToolRegistrySnapshot>,
    ) -> DriveStream<'a, M::Response> {
        // The blocking surface chains tool spans into its linear `follows_from`
        // sequence (chat -> tool -> chat), and discards the yielded items, so it
        // skips building them.
        drive_tool_calls(
            runner,
            hook_ctx,
            run,
            calls,
            tool_snapshot,
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
        atomic::{AtomicUsize, Ordering},
    };

    use futures::StreamExt;
    use serde_json::json;

    use crate::{
        agent::{AgentBuilder, AgentHook, HookContext, ToolResultAction, ToolResultEvent},
        completion::CompletionModel,
        test_utils::{MockCompletionModel, MockStreamEvent, MockTurn},
        tool::{Tool, ToolContext, ToolErrorKind, ToolExecutionError},
    };

    struct MetadataFailingTool;

    struct SnapshotValue {
        value: usize,
        clones: Arc<AtomicUsize>,
    }

    impl Clone for SnapshotValue {
        fn clone(&self) -> Self {
            self.clones.fetch_add(1, Ordering::SeqCst);
            Self {
                value: self.value,
                clones: self.clones.clone(),
            }
        }
    }

    #[derive(Clone, Default)]
    struct SnapshotMutatingTool(Arc<Mutex<Vec<usize>>>);

    impl Tool for SnapshotMutatingTool {
        const NAME: &'static str = "snapshot_mutator";
        type Error = rig::tool::ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "Mutates its per-dispatch context snapshot".into()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({"type": "object", "properties": {}})
        }

        async fn call(
            &self,
            context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            let initial = context.require::<SnapshotValue>()?.value;
            self.0.lock().expect("observed values").push(initial);
            let updated = {
                let value = context
                    .get_mut::<SnapshotValue>()
                    .expect("required snapshot value");
                value.value += 1;
                value.value
            };
            context.insert_result(updated);
            Ok(updated.to_string())
        }
    }

    #[derive(Clone, Default)]
    struct SnapshotResults(Arc<Mutex<Vec<usize>>>);

    impl<M: CompletionModel> AgentHook<M> for SnapshotResults {
        async fn on_tool_result(
            &self,
            _ctx: &HookContext,
            event: ToolResultEvent<'_>,
        ) -> ToolResultAction {
            self.0.lock().expect("result values").push(
                *event
                    .tool_context
                    .require_result::<usize>()
                    .expect("per-dispatch result metadata"),
            );
            ToolResultAction::keep()
        }
    }

    impl Tool for MetadataFailingTool {
        const NAME: &'static str = "flaky_tool";
        type Error = rig::tool::ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "Fails after attaching result metadata".into()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({"type": "object", "properties": {}})
        }

        async fn call(
            &self,
            context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            context.insert_result("shared-result-metadata".to_string());
            Err(ToolExecutionError::timeout("raw timeout failure"))
        }
    }

    #[derive(Clone, Default)]
    struct Results(Arc<Mutex<Vec<(ToolErrorKind, String, String)>>>);

    impl<M: CompletionModel> AgentHook<M> for Results {
        async fn on_tool_result(
            &self,
            _ctx: &HookContext,
            event: ToolResultEvent<'_>,
        ) -> ToolResultAction {
            if let Some(error) = event.raw_result.error() {
                self.0.lock().expect("results").push((
                    error.kind(),
                    event.raw_result.output().render(),
                    event
                        .tool_context
                        .result::<String>()
                        .expect("tool result metadata")
                        .clone(),
                ));
            }
            ToolResultAction::rewrite("rewritten for model")
        }
    }

    #[tokio::test]
    async fn blocking_and_streaming_preserve_raw_failure_while_rewriting_presentation() {
        let blocking = Results::default();
        let blocking_model = MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "flaky_tool", json!({})),
            MockTurn::text("done"),
        ]);
        AgentBuilder::new(blocking_model.clone())
            .tool(MetadataFailingTool)
            .add_hook(blocking.clone())
            .build()
            .runner("go")
            .max_turns(3)
            .run()
            .await
            .expect("blocking run");

        let streaming = Results::default();
        let streaming_model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_name_delta("tc1", "ic1", "flaky_tool"),
                MockStreamEvent::tool_call_arguments_delta("tc1", "ic1", "{}"),
                MockStreamEvent::tool_call("tc1", "flaky_tool", json!({})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let mut stream = AgentBuilder::new(streaming_model.clone())
            .tool(MetadataFailingTool)
            .add_hook(streaming.clone())
            .build()
            .runner("go")
            .max_turns(3)
            .stream()
            .await;
        while let Some(item) = stream.next().await {
            item.expect("stream item");
        }

        assert_eq!(*blocking.0.lock().unwrap(), *streaming.0.lock().unwrap());
        assert_eq!(
            *blocking.0.lock().unwrap(),
            vec![(
                ToolErrorKind::Timeout,
                "raw timeout failure".into(),
                "shared-result-metadata".into()
            )]
        );

        let blocking_history = serde_json::to_value(
            &blocking_model
                .requests()
                .get(1)
                .expect("second blocking request")
                .chat_history,
        )
        .unwrap();
        let streaming_history = serde_json::to_value(
            &streaming_model
                .requests()
                .get(1)
                .expect("second streaming request")
                .chat_history,
        )
        .unwrap();
        assert_eq!(blocking_history, streaming_history);
        let history = blocking_history.to_string();
        assert!(history.contains("rewritten for model"));
        assert!(!history.contains("raw timeout failure"));
    }

    #[tokio::test]
    async fn agent_dispatch_snapshot_clones_once_and_isolates_tool_mutations() {
        let clones = Arc::new(AtomicUsize::new(0));
        let mut context = ToolContext::new();
        context.insert(SnapshotValue {
            value: 0,
            clones: clones.clone(),
        });
        let tool = SnapshotMutatingTool::default();
        let results = SnapshotResults::default();

        AgentBuilder::new(MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", SnapshotMutatingTool::NAME, json!({})),
            MockTurn::tool_call("tc2", SnapshotMutatingTool::NAME, json!({})),
            MockTurn::text("done"),
        ]))
        .tool(tool.clone())
        .add_hook(results.clone())
        .build()
        .runner("go")
        .tool_context(context)
        .max_turns(4)
        .run()
        .await
        .expect("agent run");

        assert_eq!(*tool.0.lock().expect("observed values"), vec![0, 0]);
        assert_eq!(*results.0.lock().expect("result values"), vec![1, 1]);
        assert_eq!(
            clones.load(Ordering::SeqCst),
            2,
            "each of the two agent dispatches should clone inbound context once"
        );
    }
}

#[cfg(test)]
#[allow(irrefutable_let_patterns, unreachable_patterns)]
mod migrated_tests {
    use crate::agent::{
        CompletionCallAction, CompletionCallEvent, InvalidToolCallAction, InvalidToolCallContext,
        ModelTurnFinished, ObservationAction, StreamResponseFinish, TextDelta, ToolCall,
        ToolCallAction, ToolCallDelta, ToolResultAction, ToolResultEvent,
    };

    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering::SeqCst},
    };

    use futures::StreamExt;
    use serde::Deserialize;
    use serde_json::json;
    use tokio::sync::Notify;

    use crate::agent::AgentBuilder;
    use crate::agent::hook::{AgentHook, HookContext, RequestPatch, StepEventKind};
    use crate::agent::prompt_request::streaming::{MultiTurnStreamItem, StreamingError};
    use crate::agent::run::OutputMode;
    use crate::completion::{CompletionError, CompletionModel, Message, Prompt, PromptError};
    use crate::message::{
        AssistantContent, ToolCall as MessageToolCall, ToolChoice, ToolFunction, UserContent,
    };
    use crate::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingPrompt};
    use crate::test_utils::{
        MockAddTool, MockBarrierTool, MockCompletionModel, MockOperationArgs, MockStreamEvent,
        MockSubtractTool, MockToolError, MockTurn,
    };
    use crate::tool::{
        Tool, ToolContext, ToolExecutionError, ToolSet,
        server::{ToolServer, ToolServerHandle},
    };
    use crate::vector_store::{
        VectorSearchRequest, VectorStoreError, VectorStoreIndex, request::Filter,
    };
    use crate::wasm_compat::WasmCompatSend;

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

    impl RecordingHook {
        fn record(&self, kind: StepEventKind) {
            self.events.lock().expect("events lock").push(kind);
        }
    }

    impl<M: CompletionModel> AgentHook<M> for RecordingHook {
        async fn on_completion_call(
            &self,
            _: &HookContext,
            _: CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            self.record(StepEventKind::CompletionCall);
            CompletionCallAction::continue_run()
        }
        async fn on_completion_response(
            &self,
            _: &HookContext,
            _: crate::agent::hook::CompletionResponse<'_, M>,
        ) -> ObservationAction {
            self.record(StepEventKind::CompletionResponse);
            ObservationAction::continue_run()
        }
        async fn on_model_turn_finished(
            &self,
            _: &HookContext,
            _: ModelTurnFinished<'_>,
        ) -> ObservationAction {
            self.record(StepEventKind::ModelTurnFinished);
            ObservationAction::continue_run()
        }
        async fn on_invalid_tool_call(
            &self,
            _: &HookContext,
            _: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            self.record(StepEventKind::InvalidToolCall);
            None
        }
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            self.record(StepEventKind::ToolCall);
            ToolCallAction::run()
        }
        async fn on_tool_result(
            &self,
            _: &HookContext,
            event: ToolResultEvent<'_>,
        ) -> ToolResultAction {
            self.record(StepEventKind::ToolResult);
            self.tool_results
                .lock()
                .expect("results lock")
                .push(event.presentation.render());
            ToolResultAction::keep()
        }
        async fn on_text_delta(&self, _: &HookContext, _: TextDelta<'_>) -> ObservationAction {
            self.record(StepEventKind::TextDelta);
            ObservationAction::continue_run()
        }
        async fn on_tool_call_delta(
            &self,
            _: &HookContext,
            _: ToolCallDelta<'_>,
        ) -> ObservationAction {
            self.record(StepEventKind::ToolCallDelta);
            ObservationAction::continue_run()
        }
        async fn on_stream_response_finish(
            &self,
            _: &HookContext,
            _: StreamResponseFinish<'_, M>,
        ) -> ObservationAction {
            self.record(StepEventKind::StreamResponseFinish);
            ObservationAction::continue_run()
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

    /// `AgentRunner::from_agent` preserves the distinction between an absent
    /// agent default (the implicit one-call budget) and an explicit zero budget.
    #[tokio::test]
    async fn from_agent_preserves_implicit_one_and_explicit_zero_budgets() {
        let implicit_model = blocking_model();
        let implicit_recorded = implicit_model.clone();
        let implicit_agent = AgentBuilder::new(implicit_model).tool(MockAddTool).build();
        let implicit_runner = super::AgentRunner::from_agent(&implicit_agent, "add 2 and 3");
        assert_eq!(implicit_runner.max_turns, 1);

        let implicit_err = implicit_runner
            .run()
            .await
            .expect_err("implicit budget should reject the second model call");
        assert!(matches!(
            implicit_err,
            PromptError::MaxTurnsError { max_turns: 1, .. }
        ));
        assert_eq!(implicit_recorded.request_count(), 1);

        let zero_model = MockCompletionModel::text("should not be requested");
        let zero_recorded = zero_model.clone();
        let zero_agent = AgentBuilder::new(zero_model).default_max_turns(0).build();
        let zero_runner = super::AgentRunner::from_agent(&zero_agent, "do not call");
        assert_eq!(zero_runner.max_turns, 0);

        let zero_err = zero_runner
            .run()
            .await
            .expect_err("explicit zero budget should reject the initial model call");
        assert!(matches!(
            zero_err,
            PromptError::MaxTurnsError { max_turns: 0, .. }
        ));
        assert_eq!(zero_recorded.request_count(), 0);
    }

    /// The public blocking and streaming prompt surfaces enforce the one-call
    /// boundary identically after executing a tool-producing first turn.
    #[tokio::test]
    async fn prompt_surfaces_reject_second_tool_roundtrip_request_at_budget_one() {
        let blocking_model = blocking_model();
        let blocking_recorded = blocking_model.clone();
        let blocking_agent = AgentBuilder::new(blocking_model).tool(MockAddTool).build();
        let blocking_err = blocking_agent
            .prompt("add 2 and 3")
            .max_turns(1)
            .await
            .expect_err("blocking prompt should reject request two");
        assert!(matches!(
            blocking_err,
            PromptError::MaxTurnsError { max_turns: 1, .. }
        ));
        assert_eq!(blocking_recorded.request_count(), 1);

        let streaming_model = streaming_model();
        let streaming_recorded = streaming_model.clone();
        let streaming_agent = AgentBuilder::new(streaming_model).tool(MockAddTool).build();
        let mut stream = streaming_agent
            .stream_prompt("add 2 and 3")
            .max_turns(1)
            .await;
        let mut streaming_err = None;
        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                streaming_err = Some(err);
                break;
            }
        }
        match streaming_err {
            Some(StreamingError::Prompt(err)) => assert!(matches!(
                *err,
                PromptError::MaxTurnsError { max_turns: 1, .. }
            )),
            other => panic!("expected streaming max-turns error, got {other:?}"),
        }
        assert_eq!(streaming_recorded.request_count(), 1);
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
            .max_turns(2)
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
            .max_turns(2)
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
        assert_eq!(final_response.output(), blocking.output);

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
            .messages()
            .expect("streaming history")
            .to_vec();
        assert_eq!(
            serde_json::to_value(&blocking_messages).expect("serialize blocking"),
            serde_json::to_value(&streaming_messages).expect("serialize streaming"),
        );
    }

    /// Structured tool-execution results reach `ToolResultEvent` as machine
    /// metadata (error/refusal state plus result context), on both the blocking and streaming paths,
    /// so hooks can steer on a classified failure without parsing the result
    /// string.
    mod structured_tool_results {
        use std::sync::{Arc, Mutex};

        use futures::StreamExt;
        use serde_json::json;

        use crate::agent::{
            AgentBuilder, AgentHook, HookContext, HookStack, ToolCall, ToolCallAction,
            ToolResultAction, ToolResultEvent,
        };
        use crate::completion::CompletionModel;
        use crate::test_utils::{
            MockAddTool, MockCompletionModel, MockDeniedTool, MockFailingTool,
            MockHandledFailureTool, MockMetadataTool, MockRequestId, MockStreamEvent, MockTurn,
        };
        use crate::tool::{ToolErrorKind, ToolResult};

        /// Records, for every `ToolResult` event, a compact outcome label and the
        /// model-visible result string — the machine metadata a policy reads.
        #[derive(Clone, Default)]
        struct OutcomeHook {
            outcomes: Arc<Mutex<Vec<String>>>,
            results: Arc<Mutex<Vec<String>>>,
        }

        impl OutcomeHook {
            fn outcomes(&self) -> Vec<String> {
                self.outcomes.lock().expect("outcomes").clone()
            }

            fn results(&self) -> Vec<String> {
                self.results.lock().expect("results").clone()
            }
        }

        /// A compact string label for an outcome, e.g. `error:timeout`.
        fn outcome_label(result: &ToolResult) -> String {
            if result.is_skipped() {
                "skipped".to_string()
            } else if result.is_refused() {
                "denied".to_string()
            } else if let Some(error) = result.error() {
                format!("error:{}", error.kind().as_str())
            } else {
                "success".to_string()
            }
        }

        impl<M: CompletionModel> AgentHook<M> for OutcomeHook {
            async fn on_tool_result(
                &self,
                _ctx: &HookContext,
                event: ToolResultEvent<'_>,
            ) -> ToolResultAction {
                if let ToolResultEvent {
                    presentation,
                    raw_result,
                    ..
                } = event
                {
                    self.outcomes
                        .lock()
                        .expect("outcomes")
                        .push(outcome_label(raw_result));
                    self.results
                        .lock()
                        .expect("results")
                        .push(presentation.render());
                }
                ToolResultAction::keep()
            }
        }

        /// A blocking model that calls `tool` once, then answers.
        fn model_one_tool_then_text(tool: &str) -> MockCompletionModel {
            MockCompletionModel::from_turns([
                MockTurn::tool_call("tc1", tool, json!({})),
                MockTurn::text("done"),
            ])
        }

        /// A streaming model that calls `tool` once, then answers.
        fn stream_model_one_tool_then_text(tool: &str) -> MockCompletionModel {
            MockCompletionModel::from_stream_turns([
                vec![
                    MockStreamEvent::tool_call_name_delta("tc1", "ic1", tool),
                    MockStreamEvent::tool_call_arguments_delta("tc1", "ic1", "{}"),
                    MockStreamEvent::tool_call("tc1", tool, json!({})),
                    MockStreamEvent::final_response_with_total_tokens(0),
                ],
                vec![
                    MockStreamEvent::text("done"),
                    MockStreamEvent::final_response_with_total_tokens(0),
                ],
            ])
        }

        // (1) A `Timeout` failure reaches `ToolResultEvent` as structured
        // metadata (not just a string), with the model-visible feedback intact.
        #[tokio::test]
        async fn timeout_failure_surfaces_structured_outcome() {
            let hook = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("flaky_tool"))
                .tool(MockFailingTool::new(ToolErrorKind::Timeout))
                .add_hook(hook.clone())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("run should succeed; a tool timeout is model-visible feedback, not fatal");

            assert_eq!(hook.outcomes(), vec!["error:timeout".to_string()]);
            // (4) The model still receives useful text for the handled failure.
            assert_eq!(hook.results(), vec!["mock tool call failed".to_string()]);
        }

        // (2) A hook counts timeout failures in the run scratchpad and terminates
        // the run after a threshold — the motivating use case.
        #[tokio::test]
        async fn hook_terminates_after_repeated_timeouts() {
            #[derive(Clone, Default)]
            struct TimeoutCount(usize);

            struct TimeoutTerminator;
            impl<M: CompletionModel> AgentHook<M> for TimeoutTerminator {
                async fn on_tool_result(
                    &self,
                    ctx: &HookContext,
                    event: ToolResultEvent<'_>,
                ) -> ToolResultAction {
                    if let ToolResultEvent { raw_result, .. } = event
                        && raw_result.is_error_kind(ToolErrorKind::Timeout)
                    {
                        let count = ctx.scratchpad().update(|c: &mut TimeoutCount| {
                            c.0 += 1;
                            c.0
                        });
                        if count >= 2 {
                            return ToolResultAction::stop("aborting after repeated tool timeouts");
                        }
                    }
                    ToolResultAction::keep()
                }
            }

            let observer = OutcomeHook::default();
            let err = AgentBuilder::new(MockCompletionModel::from_turns([
                MockTurn::tool_call("tc1", "flaky_tool", json!({})),
                MockTurn::tool_call("tc2", "flaky_tool", json!({})),
                MockTurn::text("unreachable"),
            ]))
            .tool(MockFailingTool::new(ToolErrorKind::Timeout))
            // Observer first so it records both timeouts before the terminator fires.
            .add_hook(observer.clone())
            .add_hook(TimeoutTerminator)
            .build()
            .runner("go")
            .max_turns(5)
            .run()
            .await
            .expect_err("the run must terminate after two timeouts");

            assert!(
                err.to_string()
                    .contains("aborting after repeated tool timeouts"),
                "unexpected error: {err}"
            );
            assert_eq!(
                observer.outcomes(),
                vec!["error:timeout".to_string(), "error:timeout".to_string()],
                "both timeout outcomes must be observed before termination"
            );
        }

        // (3) A not-found (404) failure surfaces as structured `NotFound` metadata
        // but does not terminate the run by default — the model may try another path.
        #[tokio::test]
        async fn not_found_outcome_is_structured_and_non_fatal() {
            let hook = OutcomeHook::default();
            let status: Arc<Mutex<Option<u16>>> = Arc::new(Mutex::new(None));

            struct StatusProbe(Arc<Mutex<Option<u16>>>);
            impl<M: CompletionModel> AgentHook<M> for StatusProbe {
                async fn on_tool_result(
                    &self,
                    _ctx: &HookContext,
                    event: ToolResultEvent<'_>,
                ) -> ToolResultAction {
                    if let Some(error) = event.raw_result.error() {
                        *self.0.lock().expect("status") = error.http_status();
                    }
                    ToolResultAction::keep()
                }
            }

            AgentBuilder::new(model_one_tool_then_text("flaky_tool"))
                .tool(MockFailingTool::new(ToolErrorKind::NotFound))
                .add_hook(hook.clone())
                .add_hook(StatusProbe(status.clone()))
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("a 404 must not terminate the run by default");

            assert_eq!(hook.outcomes(), vec!["error:not_found".to_string()]);
            assert_eq!(
                *status.lock().expect("status"),
                Some(404),
                "the structured failure must carry the HTTP status"
            );
        }

        // (4) A tool that returns a handled failure via ordinary `Result` shows the
        // model useful output while the outcome is a classified error.
        #[tokio::test]
        async fn handled_failure_delivers_model_output_and_error_outcome() {
            let hook = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("lookup"))
                .tool(MockHandledFailureTool)
                .add_hook(hook.clone())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("a handled failure is not fatal");

            assert_eq!(hook.outcomes(), vec!["error:not_found".to_string()]);
            assert_eq!(
                hook.results(),
                vec!["no record found for id 42; try a different id".to_string()],
                "the tool's model-visible output must survive alongside the error outcome"
            );
        }

        // (7) `ToolCallAction::Skip` on the tool-call produces a structured `Skipped`
        // outcome that the result hook observes.
        #[tokio::test]
        async fn flow_skip_produces_skipped_outcome() {
            struct SkipHook;
            impl<M: CompletionModel> AgentHook<M> for SkipHook {
                async fn on_tool_call(
                    &self,
                    _ctx: &HookContext,
                    event: ToolCall<'_>,
                ) -> ToolCallAction {
                    if let ToolCall { .. } = event {
                        ToolCallAction::skip("not executed (denied by policy); do not retry")
                    } else {
                        ToolCallAction::run()
                    }
                }
            }

            let observer = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("flaky_tool"))
                .tool(MockFailingTool::new(ToolErrorKind::Timeout))
                .add_hook(SkipHook)
                .add_hook(observer.clone())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("run should succeed after skipping the tool");

            assert_eq!(observer.outcomes(), vec!["skipped".to_string()]);
            assert_eq!(
                observer.results(),
                vec!["not executed (denied by policy); do not retry".to_string()]
            );
        }

        // A *tool-authored* refusal surfaces as a `Denied`
        // outcome — distinct from a hook `ToolCallAction::Skip`, which is `Skipped`. This
        // pins the documented `Skipped` vs `Denied` split: `Denied` comes only
        // from the tool, never from a hook skip.
        #[tokio::test]
        async fn tool_authored_denial_produces_denied_outcome() {
            let hook = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("guarded"))
                .tool(MockDeniedTool)
                .add_hook(hook.clone())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("a tool-authored denial is not fatal");

            assert_eq!(hook.outcomes(), vec!["denied".to_string()]);
            assert_eq!(
                hook.results(),
                vec!["access to this resource is not permitted".to_string()],
                "the model still receives the tool's denial message"
            );
        }

        #[tokio::test]
        async fn permission_denied_failure_is_not_a_tool_refusal() {
            let hook = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("flaky_tool"))
                .tool(MockFailingTool::new(ToolErrorKind::PermissionDenied))
                .add_hook(hook.clone())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("a permission failure is model-visible feedback, not fatal");

            assert_eq!(hook.outcomes(), vec!["error:permission_denied".to_string()]);
            assert_eq!(hook.results(), vec!["mock tool call failed".to_string()]);
        }

        // A `ToolCallAction::Rewrite` hook followed by a `Skip` hook: the tool must not run,
        // the `ToolResult` reports the *rewritten* args (not the model's
        // original), and the outcome is `Skipped` — the rewrite (e.g. a
        // redaction) is not lost when a later hook short-circuits. Verified on
        // both the blocking and streaming surfaces.
        #[tokio::test]
        async fn rewrite_args_then_skip_reports_rewritten_args() {
            // Rewrites the tool args, replacing whatever the model emitted.
            struct RewriteHook;
            impl<M: CompletionModel> AgentHook<M> for RewriteHook {
                async fn on_tool_call(
                    &self,
                    _ctx: &HookContext,
                    event: ToolCall<'_>,
                ) -> ToolCallAction {
                    if let ToolCall { .. } = event {
                        ToolCallAction::rewrite(json!({ "x": 41, "y": 1 }))
                    } else {
                        ToolCallAction::run()
                    }
                }
            }
            // Skips *after* the rewrite (registered second).
            struct SkipHook;
            impl<M: CompletionModel> AgentHook<M> for SkipHook {
                async fn on_tool_call(
                    &self,
                    _ctx: &HookContext,
                    event: ToolCall<'_>,
                ) -> ToolCallAction {
                    if let ToolCall { .. } = event {
                        ToolCallAction::skip("denied after rewrite")
                    } else {
                        ToolCallAction::run()
                    }
                }
            }
            // Records the args + outcome seen on the `ToolResult` event.
            #[derive(Clone, Default)]
            struct ArgsProbe {
                args: Arc<Mutex<Option<String>>>,
                outcome: Arc<Mutex<Option<String>>>,
            }
            impl<M: CompletionModel> AgentHook<M> for ArgsProbe {
                async fn on_tool_result(
                    &self,
                    _ctx: &HookContext,
                    event: ToolResultEvent<'_>,
                ) -> ToolResultAction {
                    if let ToolResultEvent {
                        args, raw_result, ..
                    } = event
                    {
                        *self.args.lock().expect("args") = Some(args.to_string());
                        *self.outcome.lock().expect("outcome") = Some(outcome_label(raw_result));
                    }
                    ToolResultAction::keep()
                }
            }

            async fn run_surface(streaming: bool) -> (String, String) {
                let probe = ArgsProbe::default();
                // The tool must never execute; `MockAddTool` would produce a
                // `Success` outcome with result "42" if it (wrongly) ran.
                if streaming {
                    let mut stream = AgentBuilder::new(stream_model_one_tool_then_text("add"))
                        .tool(MockAddTool)
                        .add_hook(RewriteHook)
                        .add_hook(SkipHook)
                        .add_hook(probe.clone())
                        .build()
                        .runner("go")
                        .max_turns(3)
                        .stream()
                        .await;
                    while let Some(item) = stream.next().await {
                        if let Err(err) = item {
                            panic!("stream item errored: {err}");
                        }
                    }
                } else {
                    AgentBuilder::new(model_one_tool_then_text("add"))
                        .tool(MockAddTool)
                        .add_hook(RewriteHook)
                        .add_hook(SkipHook)
                        .add_hook(probe.clone())
                        .build()
                        .runner("go")
                        .max_turns(3)
                        .run()
                        .await
                        .expect("run should succeed after skipping the tool");
                }
                let args = probe.args.lock().expect("args").clone().expect("args seen");
                let outcome = probe
                    .outcome
                    .lock()
                    .expect("outcome")
                    .clone()
                    .expect("outcome seen");
                (args, outcome)
            }

            for streaming in [false, true] {
                let (args, outcome) = run_surface(streaming).await;
                assert_eq!(
                    outcome, "skipped",
                    "the skipped tool must produce a Skipped outcome (streaming={streaming})"
                );
                let parsed: serde_json::Value =
                    serde_json::from_str(&args).expect("ToolResult args are valid JSON");
                assert_eq!(
                    parsed,
                    json!({ "x": 41, "y": 1 }),
                    "the skipped ToolResult must report the rewritten args, not the model's \
                     original {{}} (streaming={streaming}); got {args}"
                );
            }
        }

        // End-to-end nesting: a *nested* `HookStack` that rewrites args then skips
        // must still report the rewritten args on the skipped `ToolResult` — the
        // inner rewrite is not lost behind the inner skip when the stack is added
        // as a single composed hook. Guards the nested-composition fix.
        #[tokio::test]
        async fn nested_hook_stack_rewrite_then_skip_reports_rewritten_args() {
            struct RewriteHook;
            impl<M: CompletionModel> AgentHook<M> for RewriteHook {
                async fn on_tool_call(
                    &self,
                    _ctx: &HookContext,
                    event: ToolCall<'_>,
                ) -> ToolCallAction {
                    if let ToolCall { .. } = event {
                        ToolCallAction::rewrite(json!({ "x": 41, "y": 1 }))
                    } else {
                        ToolCallAction::run()
                    }
                }
            }
            struct SkipHook;
            impl<M: CompletionModel> AgentHook<M> for SkipHook {
                async fn on_tool_call(
                    &self,
                    _ctx: &HookContext,
                    event: ToolCall<'_>,
                ) -> ToolCallAction {
                    if let ToolCall { .. } = event {
                        ToolCallAction::skip("denied after nested rewrite")
                    } else {
                        ToolCallAction::run()
                    }
                }
            }
            #[derive(Clone, Default)]
            struct ArgsProbe {
                args: Arc<Mutex<Option<String>>>,
                outcome: Arc<Mutex<Option<String>>>,
            }
            impl<M: CompletionModel> AgentHook<M> for ArgsProbe {
                async fn on_tool_result(
                    &self,
                    _ctx: &HookContext,
                    event: ToolResultEvent<'_>,
                ) -> ToolResultAction {
                    if let ToolResultEvent {
                        args, raw_result, ..
                    } = event
                    {
                        *self.args.lock().expect("args") = Some(args.to_string());
                        *self.outcome.lock().expect("outcome") = Some(outcome_label(raw_result));
                    }
                    ToolResultAction::keep()
                }
            }

            // The rewrite + skip live inside a *nested* stack added as one hook.
            fn nested_stack() -> HookStack<MockCompletionModel> {
                let mut nested = HookStack::<MockCompletionModel>::new();
                nested.push(RewriteHook);
                nested.push(SkipHook);
                nested
            }

            // Verified on both surfaces: run_single_tool (shared) drives the same
            // nested resolution, so blocking and streaming must agree.
            for streaming in [false, true] {
                let probe = ArgsProbe::default();
                if streaming {
                    let mut stream = AgentBuilder::new(stream_model_one_tool_then_text("add"))
                        .tool(MockAddTool)
                        .add_hook(nested_stack())
                        .add_hook(probe.clone())
                        .build()
                        .runner("go")
                        .max_turns(3)
                        .stream()
                        .await;
                    while let Some(item) = stream.next().await {
                        if let Err(err) = item {
                            panic!("stream item errored: {err}");
                        }
                    }
                } else {
                    AgentBuilder::new(model_one_tool_then_text("add"))
                        .tool(MockAddTool)
                        .add_hook(nested_stack())
                        .add_hook(probe.clone())
                        .build()
                        .runner("go")
                        .max_turns(3)
                        .run()
                        .await
                        .expect("run should succeed after the nested stack skips the tool");
                }

                assert_eq!(
                    probe.outcome.lock().expect("outcome").clone(),
                    Some("skipped".to_string()),
                    "streaming={streaming}"
                );
                let args = probe.args.lock().expect("args").clone().expect("args seen");
                let parsed: serde_json::Value =
                    serde_json::from_str(&args).expect("valid JSON args");
                assert_eq!(
                    parsed,
                    json!({ "x": 41, "y": 1 }),
                    "the nested stack's rewrite must survive its skip and reach the ToolResult \
                     (streaming={streaming}); got {args}"
                );
            }
        }

        // (8) Invalid JSON arguments are classified as a structured `InvalidArgs`
        // failure rather than surfacing as an opaque string.
        #[tokio::test]
        async fn invalid_args_are_classified_as_invalid_args() {
            let hook = OutcomeHook::default();
            AgentBuilder::new(MockCompletionModel::from_turns([
                // `add` needs integers; a string is a hard parse failure.
                MockTurn::tool_call("tc1", "add", json!({ "x": "not-a-number", "y": 1 })),
                MockTurn::text("done"),
            ]))
            .tool(MockAddTool)
            .add_hook(hook.clone())
            .build()
            .runner("go")
            .max_turns(3)
            .run()
            .await
            .expect("an invalid-args failure is model-visible feedback, not fatal");

            assert_eq!(hook.outcomes(), vec!["error:invalid_args".to_string()]);
        }

        // Result metadata a tool attaches reaches the hook but never appears in the
        // model-visible output on either execution surface.
        #[tokio::test]
        async fn success_result_metadata_reaches_hook_but_not_model() {
            struct MetadataProbe {
                seen: Arc<Mutex<Option<String>>>,
                model_output: Arc<Mutex<Option<String>>>,
            }
            impl<M: CompletionModel> AgentHook<M> for MetadataProbe {
                async fn on_tool_result(
                    &self,
                    _ctx: &HookContext,
                    event: ToolResultEvent<'_>,
                ) -> ToolResultAction {
                    if let ToolResultEvent {
                        presentation,
                        tool_context,
                        ..
                    } = event
                    {
                        *self.seen.lock().expect("seen") = tool_context
                            .result::<MockRequestId>()
                            .map(|id| id.0.clone());
                        *self.model_output.lock().expect("model_output") =
                            Some(presentation.render());
                    }
                    ToolResultAction::keep()
                }
            }

            async fn run_surface(streaming: bool) -> (Option<String>, String) {
                let seen: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
                let model_output: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
                let probe = MetadataProbe {
                    seen: seen.clone(),
                    model_output: model_output.clone(),
                };

                if streaming {
                    let mut stream =
                        AgentBuilder::new(stream_model_one_tool_then_text("with_meta"))
                            .tool(MockMetadataTool)
                            .add_hook(probe)
                            .build()
                            .runner("go")
                            .max_turns(3)
                            .stream()
                            .await;
                    while let Some(item) = stream.next().await {
                        if let Err(error) = item {
                            panic!("stream item errored: {error}");
                        }
                    }
                } else {
                    AgentBuilder::new(model_one_tool_then_text("with_meta"))
                        .tool(MockMetadataTool)
                        .add_hook(probe)
                        .build()
                        .runner("go")
                        .max_turns(3)
                        .run()
                        .await
                        .expect("run should succeed");
                }

                let seen_value = seen.lock().expect("seen").clone();
                let output = model_output
                    .lock()
                    .expect("model_output")
                    .clone()
                    .expect("output");
                (seen_value, output)
            }

            for streaming in [false, true] {
                let (seen, output) = run_surface(streaming).await;
                assert_eq!(
                    seen,
                    Some("req-7".to_string()),
                    "the tool's result metadata must reach the hook (streaming={streaming})"
                );
                assert_eq!(output, "done");
                assert!(
                    !output.contains("req-7"),
                    "result metadata must never leak into model output (streaming={streaming})"
                );
            }
        }

        // (6) A `ToolResultAction::Rewrite` hook redacts the model-visible text, but a later
        // policy hook still sees the tool's *raw* structured outcome — a rewrite
        // changes only what the model sees, not the classification.
        #[tokio::test]
        async fn rewrite_result_does_not_mask_the_structured_outcome() {
            struct Redact;
            impl<M: CompletionModel> AgentHook<M> for Redact {
                async fn on_tool_result(
                    &self,
                    _ctx: &HookContext,
                    event: ToolResultEvent<'_>,
                ) -> ToolResultAction {
                    if let ToolResultEvent { .. } = event {
                        ToolResultAction::rewrite("[REDACTED]")
                    } else {
                        ToolResultAction::keep()
                    }
                }
            }

            let observer = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("flaky_tool"))
                .tool(MockFailingTool::new(ToolErrorKind::NotFound))
                // Observer AFTER the redactor: it still sees the true outcome, and
                // the chained (redacted) model-visible result.
                .add_hook(Redact)
                .add_hook(observer.clone())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("run should succeed");

            assert_eq!(observer.outcomes(), vec!["error:not_found".to_string()]);
            assert_eq!(observer.results(), vec!["[REDACTED]".to_string()]);
        }

        // (9) The blocking and streaming surfaces observe identical structured
        // outcomes for the same scenario.
        #[tokio::test]
        async fn streaming_and_blocking_outcomes_match() {
            let blocking = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("flaky_tool"))
                .tool(MockFailingTool::new(ToolErrorKind::Timeout))
                .add_hook(blocking.clone())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("blocking run should succeed");

            let streaming = OutcomeHook::default();
            let mut stream = AgentBuilder::new(stream_model_one_tool_then_text("flaky_tool"))
                .tool(MockFailingTool::new(ToolErrorKind::Timeout))
                .add_hook(streaming.clone())
                .build()
                .runner("go")
                .max_turns(3)
                .stream()
                .await;
            while let Some(item) = stream.next().await {
                if let Err(err) = item {
                    panic!("stream item errored: {err}");
                }
            }

            assert_eq!(blocking.outcomes(), vec!["error:timeout".to_string()]);
            assert_eq!(blocking.outcomes(), streaming.outcomes());
            assert_eq!(blocking.results(), streaming.results());
        }

        // (10) With two tools in one turn at `concurrency > 1`, both structured
        // outcomes are observed and the persisted tool results keep call order.
        #[tokio::test]
        async fn concurrent_tools_preserve_order_and_both_outcomes() {
            use crate::message::{
                AssistantContent, ToolCall as MessageToolCall, ToolFunction, UserContent,
            };

            let turn = MockTurn::from_contents([
                AssistantContent::ToolCall(MessageToolCall::new(
                    "tc_add".to_string(),
                    ToolFunction::new("add".to_string(), json!({ "x": 2, "y": 3 })),
                )),
                AssistantContent::ToolCall(MessageToolCall::new(
                    "tc_flaky".to_string(),
                    ToolFunction::new("flaky_tool".to_string(), json!({})),
                )),
            ])
            .expect("two tool calls");

            let observer = OutcomeHook::default();
            let response = AgentBuilder::new(MockCompletionModel::from_turns([
                turn,
                MockTurn::text("done"),
            ]))
            .tool(MockAddTool)
            .tool(MockFailingTool::new(ToolErrorKind::Timeout))
            .add_hook(observer.clone())
            .build()
            .runner("go")
            .max_turns(3)
            .tool_concurrency(2)
            .run()
            .await
            .expect("run should succeed");

            // Hook order may interleave under concurrency, so compare as a set.
            let mut outcomes = observer.outcomes();
            outcomes.sort();
            assert_eq!(
                outcomes,
                vec!["error:timeout".to_string(), "success".to_string()]
            );

            // The persisted tool results must keep tool-call order regardless of
            // completion timing: `add` (tc_add) before `flaky_tool` (tc_flaky).
            let messages = response.messages.expect("messages");
            let tool_result_ids: Vec<String> = messages
                .iter()
                .flat_map(|message| match message {
                    crate::completion::Message::User { content } => content
                        .iter()
                        .filter_map(|c| match c {
                            UserContent::ToolResult(result) => Some(result.id.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                })
                .collect();
            assert_eq!(
                tool_result_ids,
                vec!["tc_add".to_string(), "tc_flaky".to_string()],
                "tool results must be persisted in call order"
            );
        }
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

        use crate::agent::{AgentBuilder, HookContext, ToolResultAction, ToolResultEvent};
        use crate::completion::Usage;
        use crate::test_utils::{MockAddTool, MockCompletionModel, MockTurn};
        use crate::tool::{ToolContext, ToolExecutionError};

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

        // --- Tool-result rewrites preserve raw policy data and redact telemetry ---

        /// A tool that returns a raw marker; a rewrite hook replaces the
        /// effective model and telemetry presentation.
        struct RawOutputTool;
        impl crate::tool::Tool for RawOutputTool {
            const NAME: &'static str = "raw_output";
            type Error = rig::tool::ToolExecutionError;
            type Args = serde_json::Value;
            type Output = String;
            fn description(&self) -> String {
                "returns a raw output marker".to_string()
            }

            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({ "type": "object", "properties": {} })
            }
            async fn call(
                &self,
                _context: &mut ToolContext,
                _args: Self::Args,
            ) -> Result<Self::Output, ToolExecutionError> {
                Ok("RAW_EXECUTION_OUTPUT_42".to_string())
            }
        }

        /// Redacts every tool result before the model sees it.
        struct RedactResultHook;
        impl<M: crate::completion::CompletionModel> crate::agent::AgentHook<M> for RedactResultHook {
            async fn on_tool_result(
                &self,
                _ctx: &HookContext,
                event: ToolResultEvent<'_>,
            ) -> ToolResultAction {
                if let crate::agent::ToolResultEvent { .. } = event {
                    crate::agent::ToolResultAction::rewrite("[REDACTED]")
                } else {
                    crate::agent::ToolResultAction::keep()
                }
            }
        }

        /// Stops the run after observing a completed tool result.
        struct StopOnResultHook;
        impl<M: crate::completion::CompletionModel> crate::agent::AgentHook<M> for StopOnResultHook {
            async fn on_tool_result(
                &self,
                _ctx: &HookContext,
                _event: ToolResultEvent<'_>,
            ) -> ToolResultAction {
                ToolResultAction::stop("stop after raw result")
            }
        }

        /// Captures every value recorded into the `gen_ai.tool.call.result` span
        /// field, so tests can assert telemetry follows result-hook policy.
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

        /// A `ToolResult` rewrite applies to both model presentation and
        /// telemetry so redaction hooks cannot leak the raw output through spans.
        #[tokio::test]
        async fn tool_result_rewrite_redacts_span_output() {
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
                MockTurn::tool_call("tc1", "raw_output", serde_json::json!({})),
                MockTurn::text("ok"),
            ]);
            let response = AgentBuilder::new(model)
                .tool(RawOutputTool)
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
                captured.iter().any(|v| v.contains("[REDACTED]")),
                "the rewritten presentation must reach telemetry; captured: {captured:?}"
            );
            assert!(
                !captured
                    .iter()
                    .any(|v| v.contains("RAW_EXECUTION_OUTPUT_42")),
                "the raw tool output must not leak through telemetry; captured: {captured:?}"
            );
        }

        /// Stopping from the result hook retains outcome metadata but omits
        /// potentially sensitive result content from telemetry.
        #[tokio::test]
        async fn tool_result_stop_omits_span_output() {
            let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
            let values: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
            let subscriber = Registry::default().with(ResultValueLayer {
                values: values.clone(),
            });
            let _default = tracing::subscriber::set_default(subscriber);

            warm_blocking_callsites().await;
            tracing::callsite::rebuild_interest_cache();
            values.lock().expect("values").clear();

            let result = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::tool_call(
                "tc1",
                "raw_output",
                serde_json::json!({}),
            )]))
            .tool(RawOutputTool)
            .add_hook(StopOnResultHook)
            .build()
            .runner("go")
            .max_turns(2)
            .run()
            .await;
            assert!(result.is_err(), "the result hook should stop the run");

            let captured = values.lock().expect("values").clone();
            assert!(
                !captured
                    .iter()
                    .any(|value| value.contains("RAW_EXECUTION_OUTPUT_42")),
                "a Stop must not leak raw execution telemetry; captured: {captured:?}"
            );
        }
    }

    fn tool_call_content(id: &str, args: serde_json::Value) -> AssistantContent {
        AssistantContent::ToolCall(MessageToolCall::new(
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

    /// Whether any tool result in `messages` carries the exact structured JSON value.
    fn tool_result_json_in_history(messages: &[Message], expected: &serde_json::Value) -> bool {
        messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.content.iter().any(|content| matches!(
                                content,
                                crate::message::ToolResultContent::Json { value }
                                    if value == expected
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
            .messages()
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

        fn description(&self) -> String {
            MockAddTool.description()
        }

        fn parameters(&self) -> serde_json::Value {
            MockAddTool.parameters()
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
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
    ) -> crate::agent::prompt_request::PromptResponse {
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
    /// message history** as the blocking driver: streamed results are surfaced in
    /// call order after the batch settles, and persisted results stay in tool-call
    /// order, so concurrency never reorders the final history.
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
            .messages()
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

        let messages = final_response.messages().expect("history").to_vec();
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
            tool_result_ids(final_response.messages().expect("history")),
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
    /// the per-tool **execution** items (`ToolExecutionCommitted` then the
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
                    MultiTurnStreamItem::ToolExecutionCommitted { .. } => {
                        markers.push("exec-commit")
                    }
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
            "exec-commit",
            "result",
            "exec-commit",
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
        async fn on_tool_result(
            &self,
            _ctx: &HookContext,
            event: ToolResultEvent<'_>,
        ) -> ToolResultAction {
            if let ToolResultEvent { args, .. } = event
                && serde_json::from_str::<serde_json::Value>(args)
                    .ok()
                    .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64))
                    == Some(1)
            {
                self.sibling_started.notified().await;
                return ToolResultAction::stop("stop after a tool result");
            }
            ToolResultAction::keep()
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

        fn description(&self) -> String {
            MockAddTool.description()
        }

        fn parameters(&self) -> serde_json::Value {
            MockAddTool.parameters()
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
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

    /// A the event-specific stop action from the `ToolCall` event with a reason keyed by the
    /// call's `x` arg, forcing the `x == 2` call (tc2) to terminate *before* the
    /// `x == 1` call (tc1): tc2 opens the gate after terminating, tc1 awaits it
    /// first. So completion order (tc2) differs from call order (tc1).
    struct OrderedTerminateHook {
        gate: Arc<tokio::sync::Notify>,
    }

    impl<M: CompletionModel> AgentHook<M> for OrderedTerminateHook {
        async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            if let ToolCall { args, .. } = event {
                let x = serde_json::from_str::<serde_json::Value>(args)
                    .ok()
                    .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64));
                match x {
                    Some(2) => {
                        self.gate.notify_one();
                        return ToolCallAction::stop("terminated-by-tc2".to_string());
                    }
                    Some(1) => {
                        self.gate.notified().await;
                        return ToolCallAction::stop("terminated-by-tc1".to_string());
                    }
                    _ => {}
                }
            }
            ToolCallAction::run()
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
        async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            if let ToolCall { args, .. } = event
                && serde_json::from_str::<serde_json::Value>(args)
                    .ok()
                    .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64))
                    == Some(1)
            {
                return ToolCallAction::stop("stop".to_string());
            }
            ToolCallAction::run()
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

        fn description(&self) -> String {
            MockAddTool.description()
        }

        fn parameters(&self) -> serde_json::Value {
            MockAddTool.parameters()
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
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
        async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            if let ToolCall { args, .. } = event
                && serde_json::from_str::<serde_json::Value>(args)
                    .ok()
                    .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64))
                    == Some(0)
            {
                self.sibling_started.notified().await;
                return ToolCallAction::stop("stop");
            }
            ToolCallAction::run()
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
        fn description(&self) -> String {
            MockAddTool.description()
        }

        fn parameters(&self) -> serde_json::Value {
            MockAddTool.parameters()
        }
        async fn call(
            &self,
            _context: &mut ToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
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
        async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            if let ToolCall { args, .. } = event
                && serde_json::from_str::<serde_json::Value>(args)
                    .ok()
                    .and_then(|v| v.get("x").and_then(serde_json::Value::as_i64))
                    == Some(2)
            {
                self.a_done.notified().await;
                return ToolCallAction::stop("stop");
            }
            ToolCallAction::run()
        }
    }

    /// Atomic concurrent batch: when the batch terminates, even a sibling that
    /// completed **successfully** before the terminating sibling produces no
    /// `ToolExecutionCommitted` and no `ToolResult` stream item (no orphan
    /// execution-commit), and its result is not committed. The `x == 1` tool runs
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

        let (exec_commits, results, saw_error, saw_final) =
            tokio::time::timeout(std::time::Duration::from_secs(5), async move {
                let (mut exec_commits, mut results, mut saw_error, mut saw_final) =
                    (0, 0, false, false);
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(MultiTurnStreamItem::ToolExecutionCommitted { .. }) => exec_commits += 1,
                        Ok(MultiTurnStreamItem::StreamUserItem(
                            StreamedUserContent::ToolResult { .. },
                        )) => results += 1,
                        Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final = true,
                        Ok(_) => {}
                        Err(_) => saw_error = true,
                    }
                }
                (exec_commits, results, saw_error, saw_final)
            })
            .await
            .expect("the concurrent tool drive must not hang");

        assert!(saw_error, "the terminated run must surface an error");
        assert!(
            !saw_final,
            "a terminated run must not yield a final response"
        );
        assert_eq!(
            exec_commits, 0,
            "a terminated batch surfaces no ToolExecutionCommitted events"
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

    /// The model tool-call event carries the model's **original** arguments; the
    /// execution-commit event carries the **effective** (hook-rewritten) arguments
    /// — so a `ToolCallAction::Rewrite` (e.g. a redaction) is reflected in what
    /// actually ran, not leaked as the original.
    #[tokio::test]
    async fn stream_tool_execution_committed_carries_effective_rewritten_args() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "add", json!({"x": 2, "y": 3})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let mut stream = AgentBuilder::new(model)
            .tool(MockAddTool)
            .add_hook(RewriteToolArgsHook(json!({"x": 2, "y": 40})))
            .build()
            .runner("go")
            .max_turns(3)
            .stream()
            .await;

        let mut model_args = None;
        let mut exec_args = None;
        while let Some(item) = stream.next().await {
            match item.unwrap_or_else(|err| panic!("stream item errored: {err}")) {
                MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                    tool_call,
                    ..
                }) => model_args = Some(tool_call.function.arguments),
                MultiTurnStreamItem::ToolExecutionCommitted { tool_call, .. } => {
                    exec_args = Some(tool_call.function.arguments)
                }
                _ => {}
            }
        }
        assert_eq!(
            model_args,
            Some(json!({"x": 2, "y": 3})),
            "the model tool-call event carries the model's original arguments"
        );
        assert_eq!(
            exec_args,
            Some(json!({"x": 2, "y": 40})),
            "the execution-commit event carries the hook-rewritten (effective) arguments"
        );
    }

    /// A `ToolCall` hook `ToolCallAction::Skip` surfaces the skip result as a `ToolResult`
    /// (the model sees it, and it is committed to history) but produces **no**
    /// `ToolExecutionCommitted` — nothing actually ran.
    #[tokio::test]
    async fn stream_hook_skip_surfaces_result_without_execution_commit() {
        struct SkipHook;
        impl<M: CompletionModel> AgentHook<M> for SkipHook {
            async fn on_tool_call(
                &self,
                _ctx: &HookContext,
                event: ToolCall<'_>,
            ) -> ToolCallAction {
                if let ToolCall { .. } = event {
                    ToolCallAction::skip("blocked by policy")
                } else {
                    ToolCallAction::run()
                }
            }
        }

        let calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 2})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let stream = AgentBuilder::new(model)
            .tool(CountingAddTool {
                calls: calls.clone(),
            })
            .add_hook(SkipHook)
            .build()
            .runner("go")
            .max_turns(3)
            .stream()
            .await;

        let mut exec_commits = 0;
        let mut results = 0;
        let mut final_response = None;
        let mut stream = stream;
        while let Some(item) = stream.next().await {
            match item.unwrap_or_else(|err| panic!("stream item errored: {err}")) {
                MultiTurnStreamItem::ToolExecutionCommitted { .. } => exec_commits += 1,
                MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult { .. }) => {
                    results += 1
                }
                MultiTurnStreamItem::FinalResponse(resp) => final_response = Some(resp),
                _ => {}
            }
        }

        assert_eq!(calls.load(SeqCst), 0, "a skipped tool's body never runs");
        assert_eq!(
            exec_commits, 0,
            "a hook-skipped tool produces no execution-commit"
        );
        assert_eq!(
            results, 1,
            "the skip result is still surfaced to the consumer"
        );
        let final_response = final_response.expect("stream should yield a final response");
        // The skip result is committed to history (the model sees the reason).
        let history = final_response.messages().expect("history");
        assert!(
            history.iter().any(|m| serde_json::to_string(m)
                .map(|s| s.contains("blocked by policy"))
                .unwrap_or(false)),
            "the skip result is committed to history"
        );
    }

    /// `ToolChoice::Required` + a hook whose `active_tools([])` advertises no tools
    /// is a **local** error: the run fails before any provider round-trip.
    #[tokio::test]
    async fn required_with_empty_active_tools_errors_locally_without_provider_call() {
        struct EmptyActiveToolsHook;
        impl<M: CompletionModel> AgentHook<M> for EmptyActiveToolsHook {
            async fn on_completion_call(
                &self,
                _ctx: &HookContext,
                event: CompletionCallEvent<'_>,
            ) -> CompletionCallAction {
                if let CompletionCallEvent { .. } = event {
                    CompletionCallAction::patch(
                        RequestPatch::new().active_tools(Vec::<String>::new()),
                    )
                } else {
                    CompletionCallAction::continue_run()
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
            async fn on_completion_call(
                &self,
                _ctx: &HookContext,
                event: CompletionCallEvent<'_>,
            ) -> CompletionCallAction {
                if let CompletionCallEvent { .. } = event {
                    CompletionCallAction::patch(RequestPatch::new().active_tools(["add"]))
                } else {
                    CompletionCallAction::continue_run()
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

    /// A tool that counts how many times it executes.
    #[derive(Clone)]
    struct CountingAddTool {
        calls: Arc<AtomicU32>,
    }
    impl Tool for CountingAddTool {
        const NAME: &'static str = "add";
        type Error = MockToolError;
        type Args = MockOperationArgs;
        type Output = i32;
        fn description(&self) -> String {
            MockAddTool.description()
        }
        fn parameters(&self) -> serde_json::Value {
            MockAddTool.parameters()
        }
        async fn call(
            &self,
            _context: &mut ToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, SeqCst);
            MockAddTool.call(_context, args).await
        }
    }

    #[derive(Clone, Default)]
    struct ToolOnlyHook {
        text_delta_calls: Arc<AtomicU32>,
        other_calls: Arc<AtomicU32>,
    }

    impl<M: CompletionModel> AgentHook<M> for ToolOnlyHook {
        async fn on_text_delta(&self, _: &HookContext, _: TextDelta<'_>) -> ObservationAction {
            self.text_delta_calls.fetch_add(1, SeqCst);
            ObservationAction::continue_run()
        }
        async fn on_completion_call(
            &self,
            _: &HookContext,
            _: CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            self.other_calls.fetch_add(1, SeqCst);
            CompletionCallAction::continue_run()
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
        async fn on_completion_call(
            &self,
            _: &HookContext,
            _: CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            if self.0 == StepEventKind::CompletionCall {
                CompletionCallAction::stop("stop here")
            } else {
                CompletionCallAction::continue_run()
            }
        }
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            if self.0 == StepEventKind::ToolCall {
                ToolCallAction::stop("stop here")
            } else {
                ToolCallAction::run()
            }
        }
        async fn on_tool_result(
            &self,
            _: &HookContext,
            _: ToolResultEvent<'_>,
        ) -> ToolResultAction {
            if self.0 == StepEventKind::ToolResult {
                ToolResultAction::stop("stop here")
            } else {
                ToolResultAction::keep()
            }
        }
    }

    /// the event-specific stop action cancels the blocking run from *every* shared driver
    /// event (model call, model response, tool call, tool result) — none is a
    /// silent no-op.
    #[tokio::test]
    async fn run_terminates_from_each_shared_event() {
        for kind in [
            StepEventKind::CompletionCall,
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
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            event: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            Some(if let _ = event {
                InvalidToolCallAction::repair(self.0)
            } else {
                InvalidToolCallAction::fail()
            })
        }
    }

    #[derive(Clone)]
    struct CaptureAndRepairInvalidHook {
        replacement: &'static str,
        args: Arc<Mutex<Vec<Option<String>>>>,
    }

    impl<M: CompletionModel> AgentHook<M> for CaptureAndRepairInvalidHook {
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            event: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            self.args
                .lock()
                .expect("invalid args")
                .push(event.args.clone());
            Some(InvalidToolCallAction::repair(self.replacement))
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
        assert_eq!(final_response.output(), blocking.output);

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
            .messages()
            .expect("streaming history")
            .to_vec();
        assert_eq!(
            serde_json::to_value(&blocking_messages).expect("serialize blocking"),
            serde_json::to_value(&streaming_messages).expect("serialize streaming"),
        );
    }

    #[tokio::test]
    async fn invalid_tool_call_scalar_args_are_canonical_across_run_and_complete_stream() {
        let blocking_args = Arc::new(Mutex::new(Vec::new()));
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "unknown_echo", json!("payload")),
            MockTurn::text("done"),
        ]))
        .tool(EchoStringArgs)
        .build()
        .runner("echo a string")
        .max_turns(3)
        .add_hook(blocking_hook.clone())
        .add_hook(CaptureAndRepairInvalidHook {
            replacement: EchoStringArgs::NAME,
            args: blocking_args.clone(),
        })
        .run()
        .await
        .expect("blocking scalar repair should succeed");

        let streaming_args = Arc::new(Mutex::new(Vec::new()));
        let streaming_hook = RecordingHook::default();
        let mut stream = AgentBuilder::new(MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "unknown_echo", json!("payload")),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]))
        .tool(EchoStringArgs)
        .build()
        .runner("echo a string")
        .max_turns(3)
        .add_hook(streaming_hook.clone())
        .add_hook(CaptureAndRepairInvalidHook {
            replacement: EchoStringArgs::NAME,
            args: streaming_args.clone(),
        })
        .stream()
        .await;
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let MultiTurnStreamItem::FinalResponse(response) =
                item.expect("streaming scalar repair should succeed")
            {
                final_response = Some(response);
            }
        }
        let final_response = final_response.expect("stream should yield a final response");

        let canonical_args = vec![Some(serde_json::to_string("payload").unwrap())];
        assert_eq!(*blocking_args.lock().unwrap(), canonical_args);
        assert_eq!(*streaming_args.lock().unwrap(), canonical_args);
        assert_eq!(blocking_hook.tool_results(), vec!["payload"]);
        assert_eq!(streaming_hook.tool_results(), vec!["payload"]);
        assert_eq!(blocking.output, "done");
        assert_eq!(final_response.output(), "done");
        assert_eq!(
            serde_json::to_value(blocking.messages.expect("blocking history")).unwrap(),
            serde_json::to_value(final_response.messages().expect("streaming history")).unwrap()
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
                        AssistantContent::ToolCall(MessageToolCall::new(
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
            output: final_response.output().to_string(),
            messages: final_response
                .messages()
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
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            event: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            Some(if let _ = event {
                InvalidToolCallAction::skip(self.0)
            } else {
                InvalidToolCallAction::fail()
            })
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
        assert_eq!(final_response.output(), blocking.output);
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
            .messages()
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
                AssistantContent::ToolCall(MessageToolCall::new(
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

        // The normalized per-turn `ModelTurnFinished` is suppressed on the
        // recovered turn 1 on BOTH surfaces too (its own guard, separate from the
        // medium-specific response-finish guards above), so only the accepted turn
        // 2 fires it — count is 1, not 2, on each driver. Without the suppression
        // this would be 2, and a per-turn accounting hook would double-count the
        // recovered turn.
        assert_eq!(
            blocking_hook.count(StepEventKind::ModelTurnFinished),
            1,
            "the recovered turn must not fire ModelTurnFinished"
        );
        assert_eq!(
            streaming_hook.count(StepEventKind::ModelTurnFinished),
            1,
            "the recovered turn must not fire ModelTurnFinished on the streaming surface either"
        );
        // Parity: the normalized per-turn event fires the same number of times on
        // both drivers even when a turn is recovered.
        assert_eq!(
            blocking_hook.count(StepEventKind::ModelTurnFinished),
            streaming_hook.count(StepEventKind::ModelTurnFinished),
        );
    }

    /// A prompt/runner-level `add_hook` APPENDS to the agent's default hooks
    /// rather than replacing them (the `with_hook` → `add_hook` semantic change):
    /// a hook registered on the builder and a hook registered on the runner both
    /// observe the same run.
    #[tokio::test]
    async fn runner_add_hook_appends_to_agent_default_hooks() {
        let agent_hook = RecordingHook::default();
        let runner_hook = RecordingHook::default();

        // `agent_hook` is registered on the builder; `runner_hook` is registered
        // on the runner obtained from that agent. `AgentRunner::from_agent` clones
        // the agent's hook stack and `add_hook` pushes on top, so both must fire.
        AgentBuilder::new(blocking_model())
            .tool(MockAddTool)
            .add_hook(agent_hook.clone())
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(runner_hook.clone())
            .run()
            .await
            .expect("run should succeed");

        assert!(
            agent_hook.count(StepEventKind::CompletionCall) >= 1,
            "the agent-default hook must still observe the run after a runner-level add_hook"
        );
        assert!(
            runner_hook.count(StepEventKind::CompletionCall) >= 1,
            "the runner-level hook must also observe the run"
        );
        // Both saw the same number of completion calls — the runner-level hook
        // appended to the agent stack; it did not replace it.
        assert_eq!(
            agent_hook.count(StepEventKind::CompletionCall),
            runner_hook.count(StepEventKind::CompletionCall),
            "add_hook appends (both hooks observe every turn); it does not replace"
        );
    }

    /// Skips a *valid* tool call before execution; observes everything else.
    struct SkipToolCallHook(&'static str);

    impl<M: CompletionModel> AgentHook<M> for SkipToolCallHook {
        async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            if let ToolCall { .. } = event {
                ToolCallAction::skip(self.0)
            } else {
                ToolCallAction::run()
            }
        }
    }

    /// A hook that skips a *valid* tool call (`ToolCallAction::Skip` on `ToolCall`, the
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
        assert_eq!(final_response.output(), blocking.output);
        assert_eq!(
            blocking_hook.shared_events(),
            streaming_hook.shared_events()
        );
        // A skipped valid tool call fires the `ToolResult` hook carrying a
        // structured `Skipped` outcome (the redesign surfaces the skip to result
        // hooks), so both drivers record the verbatim skip reason as the result.
        assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());
        assert_eq!(
            blocking_hook.tool_results(),
            vec!["skipped by policy".to_string()],
            "a skipped tool fires a ToolResult hook with the verbatim skip reason"
        );

        let blocking_messages = blocking.messages.expect("blocking messages");
        let streaming_messages = final_response
            .messages()
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

    /// A hook that rewrites a valid tool call's arguments (`ToolCallAction::Rewrite` on
    /// `ToolCall`) so the tool executes with the replacement instead of what the
    /// model emitted.
    struct RewriteToolArgsHook(serde_json::Value);

    impl<M: CompletionModel> AgentHook<M> for RewriteToolArgsHook {
        async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            if let ToolCall { .. } = event {
                ToolCallAction::rewrite(self.0.clone())
            } else {
                ToolCallAction::run()
            }
        }
    }

    struct EchoStringArgs;

    impl Tool for EchoStringArgs {
        const NAME: &'static str = "echo_string_args";
        type Error = rig::tool::ToolExecutionError;
        type Args = String;
        type Output = String;

        fn description(&self) -> String {
            "Echo a JSON string argument".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({"type": "string"})
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            Ok(args)
        }
    }

    #[derive(serde::Deserialize)]
    struct FirstGenerationArgs {
        old: String,
    }

    struct FirstGenerationTool(Arc<AtomicU32>);

    impl Tool for FirstGenerationTool {
        const NAME: &'static str = "generation_pinned";
        type Error = rig::tool::ToolExecutionError;
        type Args = FirstGenerationArgs;
        type Output = String;

        fn description(&self) -> String {
            "first generation schema".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {"old": {"type": "string"}},
                "required": ["old"]
            })
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            self.0.fetch_add(1, SeqCst);
            Ok(format!("first:{}", args.old))
        }
    }

    #[derive(serde::Deserialize)]
    struct SecondGenerationArgs {
        new: String,
    }

    struct SecondGenerationTool(Arc<AtomicU32>);

    impl Tool for SecondGenerationTool {
        const NAME: &'static str = FirstGenerationTool::NAME;
        type Error = rig::tool::ToolExecutionError;
        type Args = SecondGenerationArgs;
        type Output = String;

        fn description(&self) -> String {
            "second generation schema".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {"new": {"type": "string"}},
                "required": ["new"]
            })
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, ToolExecutionError> {
            self.0.fetch_add(1, SeqCst);
            Ok(format!("second:{}", args.new))
        }
    }

    /// Pauses the first provider call after its request has been built. Tests
    /// replace the live registry while that request is in flight, then let the
    /// model return a call that is valid only for the advertised generation.
    #[derive(Clone)]
    struct PausingCompletionModel {
        inner: MockCompletionModel,
        request_started: Arc<Notify>,
        release_response: Arc<Notify>,
        requests: Arc<AtomicU32>,
    }

    impl PausingCompletionModel {
        fn new(inner: MockCompletionModel) -> (Self, Arc<Notify>, Arc<Notify>) {
            let request_started = Arc::new(Notify::new());
            let release_response = Arc::new(Notify::new());
            (
                Self {
                    inner,
                    request_started: request_started.clone(),
                    release_response: release_response.clone(),
                    requests: Arc::new(AtomicU32::new(0)),
                },
                request_started,
                release_response,
            )
        }

        async fn inspect_and_pause(&self, request: &crate::completion::CompletionRequest) {
            let request_index = self.requests.fetch_add(1, SeqCst);
            let definition = request
                .tools
                .iter()
                .find(|definition| definition.name == FirstGenerationTool::NAME)
                .expect("generation tool must be advertised");
            if request_index == 0 {
                assert_eq!(definition.description, "first generation schema");
                self.request_started.notify_one();
                self.release_response.notified().await;
            } else {
                assert_eq!(definition.description, "second generation schema");
            }
        }
    }

    impl CompletionModel for PausingCompletionModel {
        type Response = crate::test_utils::MockResponse;
        type StreamingResponse = crate::test_utils::MockResponse;
        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>) -> Self {
            Self::new(MockCompletionModel::default()).0
        }

        async fn completion(
            &self,
            request: crate::completion::CompletionRequest,
        ) -> Result<
            crate::completion::CompletionResponse<Self::Response>,
            crate::completion::CompletionError,
        > {
            self.inspect_and_pause(&request).await;
            self.inner.completion(request).await
        }

        async fn stream(
            &self,
            request: crate::completion::CompletionRequest,
        ) -> Result<
            crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
            crate::completion::CompletionError,
        > {
            self.inspect_and_pause(&request).await;
            self.inner.stream(request).await
        }
    }

    /// `ToolCallAction::Rewrite` resolves to a `ProceedWith` tool-call decision that
    /// carries the replacement arguments, and is named for fail-closed
    /// diagnostics.
    #[test]
    fn rewrite_args_resolves_to_proceed_with_for_tool_call() {
        let args = json!({"x": 1, "y": 2});
        match super::tool_call_decision(ToolCallAction::rewrite(args.clone())) {
            super::ToolCallDecision::ProceedWith(replacement) => assert_eq!(replacement, args),
            _ => panic!("ToolCallAction::Rewrite should resolve to ProceedWith"),
        }
        // The typed convenience builds the same variant as the value constructor.
        assert_eq!(
            ToolCallAction::try_rewrite(&json!({"x": 1, "y": 2})).expect("serializes"),
            ToolCallAction::rewrite(json!({"x": 1, "y": 2})),
        );
    }

    /// A hook that rewrites a *valid* tool call's arguments (`ToolCallAction::Rewrite`
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
        assert_eq!(final_response.output(), blocking.output);
        assert_eq!(
            blocking_hook.shared_events(),
            streaming_hook.shared_events()
        );
        assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());
    }

    #[tokio::test]
    async fn string_tool_call_without_rewrite_is_canonical_across_run_and_stream() {
        let turns = [
            ScriptedTurn::ToolCalls(vec![ScriptedToolCall {
                id: "tc-string",
                name: EchoStringArgs::NAME,
                args: json!("original"),
            }]),
            ScriptedTurn::Text("done"),
        ];

        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(MockCompletionModel::from_turns(
            turns.iter().map(ScriptedTurn::as_blocking_turn),
        ))
        .tool(EchoStringArgs)
        .build()
        .runner("echo a string")
        .max_turns(3)
        .add_hook(blocking_hook.clone())
        .run()
        .await
        .expect("blocking string call should execute");

        let streaming_hook = RecordingHook::default();
        let mut stream = AgentBuilder::new(MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        ))
        .tool(EchoStringArgs)
        .build()
        .runner("echo a string")
        .max_turns(3)
        .add_hook(streaming_hook.clone())
        .stream()
        .await;
        let mut final_output = None;
        while let Some(item) = stream.next().await {
            if let MultiTurnStreamItem::FinalResponse(response) =
                item.expect("streaming string call should execute")
            {
                final_output = Some(response.output().to_string());
            }
        }

        assert_eq!(blocking.output, "done");
        assert_eq!(final_output.as_deref(), Some("done"));
        assert_eq!(blocking_hook.tool_results(), vec!["original"]);
        assert_eq!(streaming_hook.tool_results(), vec!["original"]);
    }

    #[tokio::test]
    async fn string_tool_call_rewrite_is_canonical_json_across_run_and_stream() {
        let turns = [
            ScriptedTurn::ToolCalls(vec![ScriptedToolCall {
                id: "tc-string",
                name: EchoStringArgs::NAME,
                args: json!("original"),
            }]),
            ScriptedTurn::Text("done"),
        ];
        let replacement = json!("sanitized");

        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(MockCompletionModel::from_turns(
            turns.iter().map(ScriptedTurn::as_blocking_turn),
        ))
        .tool(EchoStringArgs)
        .build()
        .runner("echo a string")
        .max_turns(3)
        .add_hook(blocking_hook.clone())
        .add_hook(RewriteToolArgsHook(replacement.clone()))
        .run()
        .await
        .expect("blocking string rewrite should execute");

        let streaming_hook = RecordingHook::default();
        let mut stream = AgentBuilder::new(MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        ))
        .tool(EchoStringArgs)
        .build()
        .runner("echo a string")
        .max_turns(3)
        .add_hook(streaming_hook.clone())
        .add_hook(RewriteToolArgsHook(replacement))
        .stream()
        .await;
        let mut final_output = None;
        while let Some(item) = stream.next().await {
            if let MultiTurnStreamItem::FinalResponse(response) =
                item.expect("streaming string rewrite should execute")
            {
                final_output = Some(response.output().to_string());
            }
        }

        assert_eq!(blocking.output, "done");
        assert_eq!(final_output.as_deref(), Some("done"));
        assert_eq!(blocking_hook.tool_results(), vec!["sanitized"]);
        assert_eq!(streaming_hook.tool_results(), vec!["sanitized"]);
    }

    #[tokio::test]
    async fn blocking_turn_dispatches_the_registry_generation_it_advertised() {
        let first_calls = Arc::new(AtomicU32::new(0));
        let second_calls = Arc::new(AtomicU32::new(0));
        let handle: ToolServerHandle = ToolServer::new()
            .tool(FirstGenerationTool(first_calls.clone()))
            .run();
        let inner = MockCompletionModel::from_turns([
            MockTurn::tool_call(
                "tc-generation",
                FirstGenerationTool::NAME,
                json!({"old": "payload"}),
            ),
            MockTurn::text("done"),
        ]);
        let (model, request_started, release_response) = PausingCompletionModel::new(inner);
        let runner = AgentBuilder::new(model)
            .tool_server_handle(handle.clone())
            .build()
            .runner("use the generation tool")
            .max_turns(3);

        let run = runner.run();
        let replace = async {
            request_started.notified().await;
            handle
                .add_tool(SecondGenerationTool(second_calls.clone()))
                .await;
            release_response.notify_one();
        };
        let (response, ()) = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            tokio::join!(run, replace)
        })
        .await
        .expect("in-flight blocking replacement must not hang");
        let response = response.expect("blocking run should use its pinned tool generation");

        assert_eq!(response.output, "done");
        assert_eq!(first_calls.load(SeqCst), 1);
        assert_eq!(second_calls.load(SeqCst), 0);
    }

    #[tokio::test]
    async fn streaming_turn_dispatches_the_registry_generation_it_advertised() {
        let first_calls = Arc::new(AtomicU32::new(0));
        let second_calls = Arc::new(AtomicU32::new(0));
        let handle: ToolServerHandle = ToolServer::new()
            .tool(FirstGenerationTool(first_calls.clone()))
            .run();
        let turns = [
            ScriptedTurn::ToolCalls(vec![ScriptedToolCall {
                id: "tc-generation",
                name: FirstGenerationTool::NAME,
                args: json!({"old": "payload"}),
            }]),
            ScriptedTurn::Text("done"),
        ];
        let inner = MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        );
        let (model, request_started, release_response) = PausingCompletionModel::new(inner);
        let runner = AgentBuilder::new(model)
            .tool_server_handle(handle.clone())
            .build()
            .runner("use the generation tool")
            .max_turns(3);

        let drive = async {
            let mut stream = runner.stream().await;
            let mut final_output = None;
            while let Some(item) = stream.next().await {
                if let MultiTurnStreamItem::FinalResponse(response) =
                    item.expect("streaming run should use its pinned tool generation")
                {
                    final_output = Some(response.output().to_string());
                }
            }
            final_output
        };
        let replace = async {
            request_started.notified().await;
            handle
                .add_tool(SecondGenerationTool(second_calls.clone()))
                .await;
            release_response.notify_one();
        };
        let (final_output, ()) = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            tokio::join!(drive, replace)
        })
        .await
        .expect("in-flight streaming replacement must not hang");

        assert_eq!(final_output.as_deref(), Some("done"));
        assert_eq!(first_calls.load(SeqCst), 1);
        assert_eq!(second_calls.load(SeqCst), 0);
    }

    /// A hook that rewrites a tool's result (`ToolResultAction::Rewrite` on
    /// `ToolResult`) so the model sees the replacement instead of the tool's
    /// actual output.
    struct RewriteToolResultHook(&'static str);

    impl<M: CompletionModel> AgentHook<M> for RewriteToolResultHook {
        async fn on_tool_result(
            &self,
            _ctx: &HookContext,
            event: ToolResultEvent<'_>,
        ) -> ToolResultAction {
            if let ToolResultEvent { .. } = event {
                ToolResultAction::rewrite(self.0)
            } else {
                ToolResultAction::keep()
            }
        }
    }

    /// `ToolResultAction::Rewrite` resolves to a `Replace` tool-result decision carrying
    /// the replacement, and is named for fail-closed diagnostics.
    #[test]
    fn rewrite_result_resolves_to_replace_for_tool_result() {
        match super::tool_result_decision(ToolResultAction::rewrite("redacted")) {
            super::ToolResultDecision::Replace(result) => {
                assert_eq!(result.as_text(), Some("redacted"))
            }
            _ => panic!("ToolResultAction::Rewrite should resolve to Replace"),
        }
    }

    /// A hook that rewrites a tool's result (`ToolResultAction::Rewrite` on
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
        assert_eq!(final_response.output(), blocking.output);

        // The ToolResult event observes the tool's ACTUAL output (5) on both
        // drivers — the replacement is applied after the event fires.
        assert_eq!(blocking_hook.tool_results(), vec!["5".to_string()]);
        assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());

        // The model-visible history carries the REWRITTEN result, not "5", and is
        // byte-identical across drivers.
        let blocking_messages = blocking.messages.expect("blocking messages");
        let streaming_messages = final_response
            .messages()
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

    /// A `ToolResultAction::Rewrite` replacement is delivered to the model verbatim, not
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

    /// A hook that patches the model request for the turn (`CompletionCallAction::Patch`
    /// on `CompletionCall`): forces tool_choice + temperature, narrows the
    /// advertised tools to an allow-list, and injects a passthrough param.
    struct PatchRequestHook;

    impl<M: CompletionModel> AgentHook<M> for PatchRequestHook {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            event: CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            if let CompletionCallEvent { .. } = event {
                CompletionCallAction::patch(
                    RequestPatch::new()
                        .preamble(OVERRIDE_PREAMBLE)
                        .temperature(0.25)
                        .max_tokens(OVERRIDE_MAX_TOKENS)
                        .tool_choice(ToolChoice::Required)
                        .active_tools(["add"])
                        .additional_params(json!({"injected": true})),
                )
            } else {
                CompletionCallAction::continue_run()
            }
        }
    }

    const OVERRIDE_PREAMBLE: &str = "overridden: critical-step instructions";
    const OVERRIDE_MAX_TOKENS: u64 = 512;

    /// `CompletionCallAction::Patch` resolves to a `Patch` completion-call decision
    /// carrying the patch, and is named for fail-closed diagnostics.
    #[test]
    fn patch_request_resolves_to_patch_for_completion_call() {
        let patch = RequestPatch::new()
            .temperature(0.25)
            .tool_choice(ToolChoice::Required);
        match super::completion_call_decision(CompletionCallAction::patch(patch.clone())) {
            super::CompletionCallDecision::Patch(got) => assert_eq!(got, patch),
            _ => panic!("PatchRequest should resolve to Patch for a completion call"),
        }
    }

    /// A `CompletionCallAction::Patch` hook patches the request for the turn identically
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
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            event: CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            if let CompletionCallEvent { .. } = event {
                CompletionCallAction::patch(
                    RequestPatch::new().context(hook_doc(self.id, self.text)),
                )
            } else {
                CompletionCallAction::continue_run()
            }
        }
    }

    /// Injects an extra context document only on the first turn (to prove
    /// per-turn, non-sticky behavior).
    struct ExtraContextTurnOneHook;

    impl<M: CompletionModel> AgentHook<M> for ExtraContextTurnOneHook {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            event: CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            if let CompletionCallEvent { turn, .. } = event
                && turn == 1
            {
                return CompletionCallAction::patch(
                    RequestPatch::new().context(hook_doc("turn-one", "only turn 1")),
                );
            }
            CompletionCallAction::continue_run()
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
            async fn on_completion_call(
                &self,
                _ctx: &HookContext,
                event: CompletionCallEvent<'_>,
            ) -> CompletionCallAction {
                if let CompletionCallEvent { .. } = event {
                    CompletionCallAction::patch(
                        RequestPatch::new().history([Message::user(SENTINEL)]),
                    )
                } else {
                    CompletionCallAction::continue_run()
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
            !messages_have_sentinel(final_response.messages().expect("history")),
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
        async fn on_model_turn_finished(
            &self,
            _ctx: &HookContext,
            event: ModelTurnFinished<'_>,
        ) -> ObservationAction {
            if let ModelTurnFinished { turn, content, .. } = event
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
            ObservationAction::continue_run()
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

    /// `ToolCallAction::Rewrite` and `ToolResultAction::Rewrite` chain across hooks: a later hook observes
    /// (and further rewrites) the value produced by earlier hooks.
    #[tokio::test]
    async fn chained_rewrites_compose_across_hooks() {
        /// Sets one key of the tool arguments, preserving the rest.
        struct SetArg {
            key: &'static str,
            value: i64,
        }
        impl<M: CompletionModel> AgentHook<M> for SetArg {
            async fn on_tool_call(
                &self,
                _ctx: &HookContext,
                event: ToolCall<'_>,
            ) -> ToolCallAction {
                if let ToolCall { args, .. } = event {
                    let mut parsed: serde_json::Value =
                        serde_json::from_str(args).unwrap_or_else(|_| json!({}));
                    parsed[self.key] = json!(self.value);
                    ToolCallAction::rewrite(parsed)
                } else {
                    ToolCallAction::run()
                }
            }
        }

        /// Wraps the tool result in `label(...)`.
        struct WrapResult(&'static str);
        impl<M: CompletionModel> AgentHook<M> for WrapResult {
            async fn on_tool_result(
                &self,
                _ctx: &HookContext,
                event: ToolResultEvent<'_>,
            ) -> ToolResultAction {
                if let ToolResultEvent { presentation, .. } = event {
                    ToolResultAction::rewrite(format!("{}({})", self.0, presentation.render()))
                } else {
                    ToolResultAction::keep()
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

        fn description(&self) -> String {
            "A real tool sharing the default output-tool name".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({ "type": "object", "properties": {} })
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            _args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            Ok("real final_result output".to_string())
        }
    }

    /// Returns no retrieved tool on the first search, then the colliding real
    /// `final_result` tool on later searches.
    #[derive(Default)]
    struct LateFinalResultIndex {
        searches: AtomicU32,
    }

    impl VectorStoreIndex for LateFinalResultIndex {
        type Filter = Filter<serde_json::Value>;

        async fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
            &self,
            _req: VectorSearchRequest,
        ) -> Result<Vec<(f64, String, T)>, VectorStoreError> {
            Ok(Vec::new())
        }

        async fn top_n_ids(
            &self,
            _req: VectorSearchRequest,
        ) -> Result<Vec<(f64, String)>, VectorStoreError> {
            if self.searches.fetch_add(1, SeqCst) == 0 {
                Ok(Vec::new())
            } else {
                Ok(vec![(1.0, "final_result".to_string())])
            }
        }
    }

    /// Registers a real `final_result` tool after the first model turn, once the
    /// run has already reserved that name for structured output. An optional
    /// second-turn patch lets tests exercise filtering and tool-choice changes
    /// without changing the collision source.
    #[derive(Clone)]
    struct RegisterLateFinalResultTool {
        handle: ToolServerHandle,
        second_turn_patch: Option<RequestPatch>,
    }

    impl<M: CompletionModel> AgentHook<M> for RegisterLateFinalResultTool {
        async fn on_model_turn_finished(
            &self,
            ctx: &HookContext,
            _event: ModelTurnFinished<'_>,
        ) -> ObservationAction {
            if ctx.turn() == 1 {
                self.handle.add_tool(FinalResultTool).await;
            }

            ObservationAction::continue_run()
        }

        async fn on_completion_call(
            &self,
            ctx: &HookContext,
            _event: CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            if ctx.turn() == 2
                && let Some(patch) = &self.second_turn_patch
            {
                return CompletionCallAction::patch(patch.clone());
            }

            CompletionCallAction::continue_run()
        }
    }

    fn assert_structured_output_collision_error(message: &str) {
        assert!(
            message.contains("final_result"),
            "error should name the conflicting tool: {message}"
        );
        assert!(
            message.contains("structured-output") && message.contains("reserved"),
            "error should explain the structured-output reservation: {message}"
        );
        assert!(
            message.contains("rename or remove"),
            "error should provide an actionable resolution: {message}"
        );
    }

    /// An initially effective real `final_result` keeps normal dispatch while
    /// the synthetic structured-output tool is advertised under a unique name.
    #[tokio::test]
    async fn initial_output_tool_collision_uses_a_unique_synthetic_name() {
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("real", "final_result", json!({})),
            MockTurn::tool_call("output", "final_result_1", json!({ "answer": "done" })),
        ]);
        let probe = model.clone();
        let response = AgentBuilder::new(model)
            .tool(FinalResultTool)
            .output_schema::<Answer>()
            .output_mode(OutputMode::Tool)
            .build()
            .runner("go")
            .max_turns(2)
            .run()
            .await
            .expect("the real tool should dispatch before the unique output tool finalizes");

        assert!(response.output.contains("done"));
        let requests = probe.requests();
        assert_eq!(
            requests.len(),
            2,
            "real-tool dispatch must continue to a second model turn"
        );
        let tool_names = requests[0]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(tool_names.len(), 2);
        for expected in ["final_result", "final_result_1"] {
            assert_eq!(
                tool_names.iter().filter(|name| **name == expected).count(),
                1,
                "the first request should advertise `{expected}` exactly once: {tool_names:?}"
            );
        }

        assert!(
            requests[1].chat_history.iter().any(|message| matches!(
                message,
                Message::User { content }
                    if content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "real"
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    crate::message::ToolResultContent::Text(text)
                                        if text.text == "real final_result output"
                                ))
                    ))
            )),
            "the real `final_result` call must execute normally and its result must reach the follow-up request"
        );
    }

    /// Once Tool output mode has committed a name, a real tool registered under
    /// that name must fail the next request locally for every tool-choice shape.
    /// Otherwise the provider receives duplicate definitions and the real call
    /// is intercepted as final output.
    #[tokio::test]
    async fn late_output_tool_collision_fails_before_blocking_provider_for_all_choices() {
        let cases = [
            ("inherited", None),
            (
                "required",
                Some(RequestPatch::new().tool_choice(ToolChoice::Required)),
            ),
            (
                "none",
                Some(RequestPatch::new().tool_choice(ToolChoice::None)),
            ),
            (
                "specific",
                Some(RequestPatch::new().tool_choice(ToolChoice::Specific {
                    function_names: vec!["final_result".to_string()],
                })),
            ),
        ];

        for (case, second_turn_patch) in cases {
            let handle = ToolServer::new().tool(MockAddTool).run();
            let model = MockCompletionModel::from_turns([
                MockTurn::tool_call("add-1", "add", json!({ "x": 1, "y": 2 })),
                MockTurn::tool_call(
                    "shadowed",
                    "final_result",
                    json!({ "answer": "wrongly finalized" }),
                ),
            ]);
            let probe = model.clone();
            let err = AgentBuilder::new(model)
                .tool_server_handle(handle.clone())
                .output_schema::<Answer>()
                .output_mode(OutputMode::Tool)
                .add_hook(RegisterLateFinalResultTool {
                    handle,
                    second_turn_patch,
                })
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .unwrap_err();

            assert!(
                matches!(
                    &err,
                    PromptError::CompletionError(CompletionError::RequestError(_))
                ),
                "{case}: expected a local completion request error, got {err:?}"
            );
            assert_eq!(
                probe.request_count(),
                1,
                "{case}: the colliding second request must not reach the provider"
            );
            assert_structured_output_collision_error(&err.to_string());
        }
    }

    /// The streaming surface uses the same pre-provider collision check as the
    /// blocking surface and terminates without starting a second model stream.
    #[tokio::test]
    async fn late_output_tool_collision_fails_before_streaming_provider() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("add-1", "add", json!({ "x": 1, "y": 2 })),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::tool_call(
                    "shadowed",
                    "final_result",
                    json!({ "answer": "wrongly finalized" }),
                ),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let probe = model.clone();
        let mut stream = AgentBuilder::new(model)
            .tool_server_handle(handle.clone())
            .output_schema::<Answer>()
            .output_mode(OutputMode::Tool)
            .add_hook(RegisterLateFinalResultTool {
                handle,
                second_turn_patch: None,
            })
            .build()
            .runner("go")
            .max_turns(3)
            .stream()
            .await;

        let mut collisions = Vec::new();
        let mut saw_final_response = false;
        while let Some(item) = stream.next().await {
            match item {
                Err(err) => collisions.push(err),
                Ok(MultiTurnStreamItem::FinalResponse(_)) => saw_final_response = true,
                Ok(_) => {}
            }
        }
        assert_eq!(
            collisions.len(),
            1,
            "the stream should terminate with exactly one collision error"
        );
        assert!(
            !saw_final_response,
            "a collision error must terminate the stream without a final response"
        );
        let err = collisions.pop().expect("one collision error was asserted");

        assert!(
            matches!(
                &err,
                StreamingError::Completion(CompletionError::RequestError(_))
            ),
            "expected a local streaming completion request error, got {err:?}"
        );
        assert_eq!(
            probe.request_count(),
            1,
            "the colliding second stream must not reach the provider"
        );
        assert_structured_output_collision_error(&err.to_string());
    }

    /// A late colliding tool is harmless while `active_tools` filters it out,
    /// but the run must fail as soon as the non-sticky filter lifts and the real
    /// tool becomes effective again.
    #[tokio::test]
    async fn late_output_tool_collision_is_checked_after_active_tools_filtering() {
        let handle = ToolServer::new().tool(MockAddTool).run();
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("add-1", "add", json!({ "x": 1, "y": 2 })),
            MockTurn::tool_call("add-2", "add", json!({ "x": 3, "y": 4 })),
            MockTurn::tool_call(
                "shadowed",
                "final_result",
                json!({ "answer": "wrongly finalized" }),
            ),
        ]);
        let probe = model.clone();
        let err = AgentBuilder::new(model)
            .tool_server_handle(handle.clone())
            .output_schema::<Answer>()
            .output_mode(OutputMode::Tool)
            .add_hook(RegisterLateFinalResultTool {
                handle,
                second_turn_patch: Some(RequestPatch::new().active_tools(["add"])),
            })
            .build()
            .runner("go")
            .max_turns(4)
            .run()
            .await
            .expect_err("the exposed third-turn collision should fail locally");

        assert_eq!(
            probe.request_count(),
            2,
            "the filtered second turn may run, but the exposed third turn may not"
        );
        let requests = probe.requests();
        let second_turn_names = requests[1]
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(second_turn_names.len(), 2);
        for expected in ["add", "final_result"] {
            assert_eq!(
                second_turn_names
                    .iter()
                    .filter(|name| **name == expected)
                    .count(),
                1,
                "the second request should advertise `{expected}` exactly once: \
                 {second_turn_names:?}"
            );
        }
        assert_structured_output_collision_error(&err.to_string());
    }

    /// Dynamic retrieval shares the same effective per-turn collision check as
    /// mutable registration: a name absent on turn one may not shadow the
    /// already-reserved output tool when retrieval selects it on turn two.
    #[tokio::test]
    async fn retrieved_output_tool_collision_fails_before_provider_request() {
        let mut retrieved_tools = ToolSet::default();
        retrieved_tools.add_tool(FinalResultTool);
        let handle = ToolServer::new()
            .tool(MockAddTool)
            .retrieved_tools(1, LateFinalResultIndex::default(), retrieved_tools)
            .run();
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("add-1", "add", json!({ "x": 1, "y": 2 })),
            MockTurn::tool_call(
                "shadowed",
                "final_result",
                json!({ "answer": "wrongly finalized" }),
            ),
        ]);
        let probe = model.clone();
        let err = AgentBuilder::new(model)
            .tool_server_handle(handle)
            .output_schema::<Answer>()
            .output_mode(OutputMode::Tool)
            .build()
            .runner("go")
            .max_turns(3)
            .run()
            .await
            .expect_err("the retrieved second-turn collision should fail locally");

        assert!(matches!(
            &err,
            PromptError::CompletionError(CompletionError::RequestError(_))
        ));
        assert_eq!(
            probe.request_count(),
            1,
            "the colliding retrieved tool must prevent the second provider request"
        );
        assert_structured_output_collision_error(&err.to_string());
    }

    /// Narrows the advertised tools to `add` for the turn, filtering out the real
    /// `final_result` tool.
    struct ActiveToolsAddOnly;

    impl<M: CompletionModel> AgentHook<M> for ActiveToolsAddOnly {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            event: CompletionCallEvent<'_>,
        ) -> CompletionCallAction {
            if let CompletionCallEvent { .. } = event {
                CompletionCallAction::patch(RequestPatch::new().active_tools(["add"]))
            } else {
                CompletionCallAction::continue_run()
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

    /// Captures whether any `ModelTurnFinished.content` carried a tool call named
    /// `final_result` — the model-emitted structured-output output-tool call.
    #[derive(Clone, Default)]
    struct CaptureOutputToolInModelTurn {
        saw_output_tool_call: Arc<Mutex<bool>>,
    }

    impl<M: CompletionModel> AgentHook<M> for CaptureOutputToolInModelTurn {
        async fn on_model_turn_finished(
            &self,
            _ctx: &HookContext,
            event: ModelTurnFinished<'_>,
        ) -> ObservationAction {
            if let ModelTurnFinished { content, .. } = event
                && content.iter().any(|c| {
                    matches!(c, AssistantContent::ToolCall(tc) if tc.function.name == "final_result")
                })
            {
                *self.saw_output_tool_call.lock().expect("lock") = true;
            }
            ObservationAction::continue_run()
        }
    }

    /// `ModelTurnFinished.content` carries the **model-emitted** content — including
    /// a structured-output Tool-mode output-tool call — on both surfaces, even though
    /// the run persists that turn as assistant text (the structured output) with the
    /// tool call dropped. Guards the documented `content` contract: it is the model's
    /// committed turn content, not the finalized/persisted content, in Tool mode.
    #[tokio::test]
    async fn model_turn_finished_content_carries_output_tool_call_in_tool_mode() {
        // Blocking surface.
        let hook = CaptureOutputToolInModelTurn::default();
        let response = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::tool_call(
            "out1",
            "final_result",
            json!({ "answer": "done" }),
        )]))
        .output_schema::<Answer>()
        .output_mode(OutputMode::Tool)
        .add_hook(hook.clone())
        .build()
        .runner("go")
        .max_turns(2)
        .run()
        .await
        .expect("run should finalize via the output tool");
        assert!(
            *hook.saw_output_tool_call.lock().expect("lock"),
            "ModelTurnFinished.content must carry the model-emitted output-tool call (blocking)"
        );
        assert!(
            response.output.contains("done"),
            "the run finalizes with the structured output, not the raw tool call: {:?}",
            response.output
        );

        // Streaming surface — same content contract.
        let s_hook = CaptureOutputToolInModelTurn::default();
        let mut stream = AgentBuilder::new(MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::tool_call("out1", "final_result", json!({ "answer": "done" })),
            MockStreamEvent::final_response_with_total_tokens(0),
        ]]))
        .output_schema::<Answer>()
        .output_mode(OutputMode::Tool)
        .add_hook(s_hook.clone())
        .build()
        .runner("go")
        .max_turns(2)
        .stream()
        .await;
        while stream.next().await.is_some() {}
        assert!(
            *s_hook.saw_output_tool_call.lock().expect("lock"),
            "ModelTurnFinished.content must carry the model-emitted output-tool call (streaming)"
        );
    }

    /// A structured-output Tool-mode output-tool call finalizes the run directly, so
    /// on the streaming surface it is **not** re-emitted as a complete
    /// `StreamAssistantItem(StreamedAssistantContent::ToolCall)` item (it bypasses
    /// `drive_tool_calls`); its structured result is surfaced in the final `PromptResponse`.
    /// Guards the narrowed `StreamAssistantItem` contract.
    #[tokio::test]
    async fn output_tool_finalization_emits_no_complete_tool_call_stream_item() {
        let mut stream = AgentBuilder::new(MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::tool_call("out1", "final_result", json!({ "answer": "done" })),
            MockStreamEvent::final_response_with_total_tokens(0),
        ]]))
        .output_schema::<Answer>()
        .output_mode(OutputMode::Tool)
        .build()
        .runner("go")
        .max_turns(2)
        .stream()
        .await;

        let mut saw_complete_output_tool_call = false;
        let mut final_has_output = false;
        while let Some(item) = stream.next().await {
            match item.expect("stream item") {
                MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                    tool_call,
                    ..
                }) if tool_call.function.name == "final_result" => {
                    saw_complete_output_tool_call = true;
                }
                MultiTurnStreamItem::FinalResponse(res) => {
                    final_has_output = res.output().contains("done");
                }
                _ => {}
            }
        }
        assert!(
            !saw_complete_output_tool_call,
            "the output-tool call finalizes the run, so no complete \
             StreamAssistantItem::ToolCall item must be emitted for it"
        );
        assert!(
            final_has_output,
            "the structured output must be surfaced via the FinalResponse"
        );
    }

    // -----------------------------------------------------------------------
    // Human-in-the-loop (HITL): one hook gates each tool call behind a human
    // decision, mapping approve/deny/edit/abort onto the event-specific actions
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
    /// and mapping it to the matching event-specific action. A real reviewer would `.await`
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
        async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            let ToolCall {
                tool_name, args, ..
            } = event
            else {
                return ToolCallAction::run();
            };
            self.reviewed
                .lock()
                .unwrap()
                .push(format!("{tool_name}({args})"));
            let decision = self.decisions.lock().unwrap().pop_front();
            match decision {
                Some(Decision::Approve) => ToolCallAction::run(),
                Some(Decision::Deny(reason)) => ToolCallAction::skip(reason),
                Some(Decision::Edit(args)) => ToolCallAction::rewrite(args),
                Some(Decision::Abort(reason)) => ToolCallAction::stop(reason),
                // Fail closed if the script is exhausted (it shouldn't be) — deny
                // rather than silently approve, matching the example's contract.
                None => ToolCallAction::skip("denied: no scripted decision (fail-closed)"),
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
        // call executed nothing but now fires a ToolResult carrying its verbatim
        // denial reason (structured `Skipped` outcome) — identically on both
        // drivers.
        assert_eq!(
            blocking_recorder.tool_results(),
            vec![
                "5".to_string(),
                "denied by reviewer: amount too large".to_string(),
                "101".to_string()
            ]
        );
        assert_eq!(
            blocking_recorder.tool_results(),
            streaming_recorder.tool_results()
        );

        // The denied call (10 + 20) never executed, so its result 30 is absent —
        // the denial reason stands in its place, ruling out deny being silently
        // treated as approve.
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
        assert_eq!(final_response.output(), blocking.output);
        assert_eq!(
            blocking_recorder.shared_events(),
            streaming_recorder.shared_events()
        );

        // Model-visible history is identical across drivers (compared structurally
        // as serde_json::Value) and carries the denial reason and the edited result
        // 101 (not the model's 1 + 1 = 2).
        let blocking_messages = blocking.messages.expect("blocking messages");
        let streaming_messages = final_response
            .messages()
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
            tool_result_json_in_history(&blocking_messages, &json!(101)),
            "the edited call must have executed with the rewritten arguments"
        );
    }

    /// A HITL hook that aborts a tool call (`Decision::Abort` -> `ToolCallAction::stop`)
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
                    panic!("aborted stream must not finalize, got: {}", resp.output())
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
        async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            let ToolCall { tool_name, .. } = event else {
                return ToolCallAction::run();
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
                ToolCallAction::run()
            } else {
                ToolCallAction::skip(format!("denied by policy: `{tool_name}` not allowed"))
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
        // and executed nothing, but its denial reason now surfaces as a ToolResult
        // (structured `Skipped` outcome) between the two `add` results.
        assert_eq!(
            recorder.tool_results(),
            vec![
                "5".to_string(),
                "denied by policy: `subtract` not allowed".to_string(),
                "5".to_string()
            ]
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
