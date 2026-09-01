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
//! # use rig_agent::Agent;
//! # async fn example(agent: Agent) -> Result<(), Box<dyn std::error::Error>> {
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
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use futures::StreamExt;
use tracing::{Instrument, info_span, span::Id};

use super::{
    ModelHandle,
    completion::{Agent, AgentConfig, PreparedCompletionRequest},
    hook::{
        AgentHook, CompletionCall, CompletionCallAction,
        CompletionResponse as CompletionResponseEvent, HookContext, HookStack,
        InvalidToolCallAction, ModelTurnAction, ModelTurnFinished, ObservationAction, RequestPatch,
        ToolCall as ToolCallEvent, ToolCallAction, ToolResultAction, ToolResultEvent,
    },
    prompt_request::{
        PromptResponse,
        streaming::{
            DriveItem, DriveStream, MultiTurnStreamItem, StreamingError, TurnSource, drive_agent,
            drive_tool_calls, streaming_error_into_prompt,
        },
        tool_result_output,
    },
    run::{AgentRun, ModelTurn, ModelTurnOutcome, PendingToolCall},
};
use rig_core::{
    memory::ConversationMemory,
    message::{ToolCall, ToolChoice, UserContent},
    telemetry::SpanCombinator,
};

use crate::{
    completion::{CompletionError, CompletionModel, Document, Message, PromptError, Usage},
    json_utils,
    tool::{
        ToolContext, ToolDispatch, ToolResult,
        server::{ToolRegistrySnapshot, ToolServerHandle},
    },
};

use super::UNKNOWN_AGENT_NAME;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum UnhandledInvalidToolCallPolicy {
    #[default]
    Fail,
    IgnoreForExtractor,
}

/// Build the per-turn `chat` span shared by both turn sources.
///
/// The span *name* must be a string literal — `tracing` bakes it into static
/// metadata — so this is a macro parameterized by the name rather than a
/// function (the two surfaces keep distinct names, `chat` vs `chat_streaming`,
/// which dashboards split on). The matching operation value is passed with the
/// name; every other field is identical across the two surfaces, so it lives
/// here once instead of being copy-pasted into each `TurnSource::open_chat_span`.
macro_rules! build_chat_span {
    ($runner:expr, $effective_preamble:expr, $name:literal, $operation:literal) => {{
        let system_instructions = $crate::core::telemetry::system_instructions_json(
            $effective_preamble,
            $runner.config.record_telemetry_content,
        );
        // The core macro is the single source of the completion-parent
        // contract (marker + required fields); only the agent-specific field
        // is declared here.
        $crate::core::telemetry::completion_parent_span!(
            target: "rig::agent_chat",
            name: $name,
            operation: $operation,
            system_instructions: system_instructions.as_deref(),
            gen_ai.agent.name = $runner.agent_name_or_default(),
        )
    }};
}
pub(crate) use build_chat_span;

/// Convert an observe-only action into an optional stop reason.
pub(crate) fn observe_action(action: ObservationAction) -> Option<String> {
    match action {
        ObservationAction::Continue => None,
        ObservationAction::Stop(reason) => Some(reason),
    }
}

/// Resolved outcome of the shared, medium-neutral model-turn hook.
pub(crate) enum ModelTurnDecision {
    /// Accept the turn and advance normally.
    Advance,
    /// The turn was rejected and the run is ready to issue another model call.
    Retried,
    /// Stop the run with the supplied reason.
    Terminate(String),
}

/// Apply a model-turn hook action to the sans-IO run.
///
/// Both blocking and streaming sources use this resolver so retry history,
/// tool-turn rejection, and state transitions cannot diverge by medium.
pub(crate) fn resolve_model_turn_action(
    run: &mut AgentRun,
    action: ModelTurnAction,
) -> Result<ModelTurnDecision, PromptError> {
    match action {
        ModelTurnAction::Continue => Ok(ModelTurnDecision::Advance),
        ModelTurnAction::Retry(request) => {
            run.retry_model_turn(request)?;
            Ok(ModelTurnDecision::Retried)
        }
        ModelTurnAction::Stop(reason) => Ok(ModelTurnDecision::Terminate(reason)),
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
pub struct AgentRunner {
    /// The run's own copy of the agent's configuration, cloned as one unit by
    /// [`from_agent`](Self::from_agent). Per-run overrides mutate this copy and
    /// never the source [`Agent`]. `description` rides along unused during
    /// execution — an accepted tradeoff for a single shared config type.
    pub(crate) config: AgentConfig,
    pub(crate) prompt: Message,
    pub(crate) chat_history: Option<Vec<Message>>,
    pub(crate) max_invalid_tool_call_retries: usize,
    pub(crate) tool_server_handle: ToolServerHandle,
    /// Typed context cloned freshly for every tool dispatch.
    pub(crate) tool_context: ToolContext,
    pub(crate) output_tool_name: Option<String>,
    pub(crate) output_tool_description: Option<String>,
    pub(crate) augment_output_preamble: bool,
    pub(crate) unhandled_invalid_tool_call_policy: UnhandledInvalidToolCallPolicy,
    pub(crate) concurrency: usize,
    pub(crate) error_usage: Option<Arc<Mutex<Usage>>>,
}

/// The `(history_override, memory_handle)` pair resolved for one run by
/// [`AgentRunner::resolve_history_and_memory`].
pub(crate) type HistoryAndMemory = (
    Option<Vec<Message>>,
    Option<(Arc<dyn ConversationMemory>, rig_core::id::ConversationId)>,
);

impl AgentRunner {
    /// Build a runner from an agent, seeding it with the agent's default hook
    /// stack. Prefer [`Agent::runner`].
    pub fn from_agent(agent: &Agent, prompt: impl Into<Message>) -> Self {
        Self {
            config: agent.config.clone(),
            prompt: prompt.into(),
            chat_history: None,
            max_invalid_tool_call_retries: 0,
            tool_server_handle: agent.tool_server_handle.clone(),
            tool_context: ToolContext::new(),
            output_tool_name: None,
            output_tool_description: None,
            augment_output_preamble: true,
            unhandled_invalid_tool_call_policy: UnhandledInvalidToolCallPolicy::Fail,
            concurrency: 1,
            error_usage: None,
        }
    }

    /// Append a hook to the stack (on top of any the agent already carries).
    /// Hooks run in registration order; how their results compose is
    /// event-dependent (model selections and `ToolCall`/`ToolResult` rewrites
    /// chain, `CompletionCall` request patches accumulate and merge, while
    /// model-turn steering and observe-only/recovery events use their
    /// event-specific terminal action). See the [`hook`](crate::agent::hook)
    /// module docs.
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook + 'static,
    {
        self.config.hooks.push(hook);
        self
    }
}

impl AgentRunner {
    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. Zero emits no model calls; one permits only the
    /// initial call. Exceeding the budget returns [`PromptError::MaxTurnsError`].
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.config.max_turns = max_turns;
        self
    }

    /// Set the default model candidate for this run.
    ///
    /// This does not suppress registered model-selection hooks, which may
    /// replace the candidate before each model call (including retries).
    /// Append an unconditional selecting hook last when the run must always
    /// use one model.
    pub fn using_model(mut self, model: ModelHandle) -> Self {
        self.config.model = model;
        self
    }

    /// Erase and set a typed default model for this run.
    pub fn using_model_value<M>(self, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        self.using_model(ModelHandle::new(model))
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

    /// Override the agent preamble for this run.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.config.preamble = Some(preamble.into());
        self
    }

    /// Remove the agent's configured preamble for this run.
    pub fn without_preamble(mut self) -> Self {
        self.config.preamble = None;
        self
    }

    /// Append one static context document for this run.
    pub fn document(mut self, document: Document) -> Self {
        self.config.static_context.push(document);
        self
    }

    /// Append static context documents for this run.
    pub fn documents(mut self, documents: impl IntoIterator<Item = Document>) -> Self {
        self.config.static_context.extend(documents);
        self
    }

    /// Override the model temperature for this run.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Remove the agent's configured temperature for this run.
    pub fn without_temperature(mut self) -> Self {
        self.config.temperature = None;
        self
    }

    /// Override the maximum completion token count for this run.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Remove the agent's configured maximum token count for this run.
    pub fn without_max_tokens(mut self) -> Self {
        self.config.max_tokens = None;
        self
    }

    /// Shallow-merge object fields into the provider-specific parameters for
    /// this run. Later fields win. A non-object baseline is replaced by the
    /// supplied object. A later completion-call hook patch has final
    /// precedence: object values shallow-merge, while a non-object on either
    /// side causes wholesale replacement by the hook value.
    pub fn merge_additional_params(
        mut self,
        params: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        let params = serde_json::Value::Object(params);
        self.config.additional_params = Some(match self.config.additional_params.take() {
            Some(baseline) if baseline.is_object() => crate::json_utils::merge(baseline, params),
            _ => params,
        });
        self
    }

    /// Replace all provider-specific parameters for this run. A later
    /// completion-call hook patch has final precedence: object values
    /// shallow-merge, while a non-object on either side causes wholesale
    /// replacement by the hook value.
    pub fn replace_additional_params(mut self, params: serde_json::Value) -> Self {
        self.config.additional_params = Some(params);
        self
    }

    /// Remove the agent's configured provider-specific parameters for this run.
    /// A later completion-call hook may still supply its own parameters.
    pub fn without_additional_params(mut self) -> Self {
        self.config.additional_params = None;
        self
    }

    /// Override the tool-choice policy for this run.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.config.tool_choice = Some(tool_choice);
        self
    }

    /// Remove the agent's configured tool-choice policy for this run.
    pub fn without_tool_choice(mut self) -> Self {
        self.config.tool_choice = None;
        self
    }

    /// Configure the synthetic tool used by an internal Tool-output flow.
    pub(crate) fn output_tool(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        augment_preamble: bool,
    ) -> Self {
        self.output_tool_name = Some(name.into());
        self.output_tool_description = Some(description.into());
        self.augment_output_preamble = augment_preamble;
        self
    }

    /// Ignore invalid tool calls when every registered hook declines to act.
    ///
    /// This is an internal compatibility policy for extractors, whose legacy
    /// transport treated every non-`submit` call as irrelevant response
    /// content. Hooks still receive the invalid-call event first and retain
    /// full control over recovery or termination.
    pub(crate) fn ignore_unhandled_invalid_tool_calls(mut self) -> Self {
        self.unhandled_invalid_tool_call_policy =
            UnhandledInvalidToolCallPolicy::IgnoreForExtractor;
        self
    }

    /// Opt in or out of recording sensitive request, response, and tool content
    /// on GenAI telemetry spans for this run.
    ///
    /// Defaults to the agent's setting, which defaults to `false`. Enabling this
    /// can expose prompts, retrieved context, tool results, model responses, and
    /// other sensitive or high-cardinality data through OpenTelemetry span
    /// attributes, which can increase observability backend storage and query
    /// costs. Only enable it when content telemetry is acceptable for this run.
    /// Structural metadata and token usage remain available when disabled.
    pub fn record_content_telemetry(mut self, enabled: bool) -> Self {
        self.config.record_telemetry_content = enabled;
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
    /// A `concurrency` of 0 is clamped to 1; at `1` the tools of a turn run
    /// strictly sequentially in call order, failing fast on the first
    /// terminating error.
    pub fn tool_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Set the conversation id used to load and persist memory for this run.
    pub fn conversation(mut self, id: impl Into<rig_core::id::ConversationId>) -> Self {
        self.config.conversation_id = Some(id.into());
        self
    }

    /// Disable conversation memory for this run (no load, no save).
    pub fn without_memory(mut self) -> Self {
        self.config.memory = None;
        self.config.conversation_id = None;
        self
    }

    /// Set the retry budget for invalid tool-call recovery. Invalid tool-call
    /// retries also consume the total model-call budget.
    pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
        self.max_invalid_tool_call_retries = retries;
        self
    }

    pub(crate) fn agent_name_or_default(&self) -> &str {
        self.config.name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME)
    }

    /// Build the sans-IO [`AgentRun`] for this runner's configuration.
    /// `history_override` replaces the configured chat history (e.g. with
    /// memory-loaded history). Delegates to [`build_agent_run`] — the single
    /// construction site shared with the streaming driver.
    pub(crate) fn build_run(&self, history_override: Option<Vec<Message>>) -> AgentRun {
        let run = build_agent_run(
            self.prompt.clone(),
            self.config.max_turns,
            self.max_invalid_tool_call_retries,
            self.config.output_schema.as_ref(),
            history_override.or_else(|| self.chat_history.clone()),
            self.config.tool_choice.clone(),
        );
        match &self.output_tool_name {
            Some(name) => run.with_output_tool_name(name.clone()),
            None => run,
        }
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
    let spec = crate::run::spec::RunSpec {
        max_turns: Some(max_turns),
        max_invalid_tool_call_retries,
        output_schema: output_schema.map(|schema| schema.as_value().clone()),
        tool_choice,
        ..crate::run::spec::RunSpec::new()
    };
    AgentRun::from_spec(&spec, prompt, history)
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
    record_content: bool,
) -> (tracing::Span, bool) {
    if tracing::Span::current().is_disabled() {
        let system_instructions =
            rig_core::telemetry::system_instructions_json(preamble, record_content);
        let span = info_span!(
            "invoke_agent",
            gen_ai.operation.name = "invoke_agent",
            gen_ai.agent.name = agent_name,
            gen_ai.system_instructions = system_instructions.as_deref(),
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
pub(crate) async fn resolve_completion_call(
    hooks: &HookStack,
    ctx: &HookContext,
    prompt: &Message,
    history: &[Message],
    turn: usize,
) -> CompletionCallOutcome {
    match hooks
        .on_completion_call(
            ctx,
            CompletionCall {
                prompt,
                history,
                turn,
            },
        )
        .await
    {
        CompletionCallAction::Stop(reason) => CompletionCallOutcome::Terminate(reason),
        CompletionCallAction::Patch(patch) => CompletionCallOutcome::Proceed(Some(patch)),
        CompletionCallAction::Continue => CompletionCallOutcome::Proceed(None),
    }
}

/// Append a finished run's messages to conversation memory, logging and
/// proceeding on failure. Shared `Done`-arm behavior for both drivers.
pub(crate) async fn append_run_messages(
    memory_handle: Option<&(Arc<dyn ConversationMemory>, rig_core::id::ConversationId)>,
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
pub(crate) async fn run_single_tool(
    runner: &AgentRunner,
    ctx: &HookContext,
    tool_snapshot: &ToolRegistrySnapshot,
    tool_call: &ToolCall,
    internal_call_id: rig_core::id::InternalCallId,
    error_history: &[Message],
) -> Result<ToolCallOutcome, PromptError> {
    let hooks = &runner.config.hooks;
    let tool_context = &runner.tool_context;
    let record_content = runner.config.record_telemetry_content;
    let tool_name = &tool_call.function.name;
    // `mut` so a tool-call hook can rewrite the arguments the tool
    // runs with (the model's emitted arguments are otherwise used verbatim).
    let mut args = json_utils::serialize_json_value(&tool_call.function.arguments);

    let tool_span = tracing::Span::current();
    tool_span.record("gen_ai.tool.name", tool_name);
    tool_span.record("gen_ai.tool.call.id", tool_call.id.as_str());
    if record_content {
        tool_span.record("gen_ai.tool.call.arguments", &args);
    }

    // Resolve the `ToolCall` hook chain. A proceeding chain carries any
    // `ToolCallAction::Rewrite` in the action itself; a chain that a later hook
    // short-circuits with `Skip`/`Stop` salvages the accumulated
    // rewrite into `salvaged_rewrite` so it is *not* lost — the rewritten args
    // must still be reported on the skipped `ToolResult` and in tracing rather
    // than leaking the model's original args (see [`HookStack::resolve_tool_call`]).
    let (action, salvaged_rewrite) = hooks
        .resolve_tool_call(
            ctx,
            ToolCallEvent {
                tool_name,
                tool_call_id: Some(tool_call.id.as_str()),
                internal_call_id,
                args: &args,
            },
        )
        .await;

    // Apply a salvaged rewrite (short-circuit path only) so `args` — what the
    // `ToolResult` reports — and the span reflect the effective arguments.
    if let Some(rewritten) = salvaged_rewrite.as_ref() {
        args = json_utils::serialize_json_value(rewritten);
        if record_content {
            tool_span.record("gen_ai.tool.call.arguments", &args);
        }
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
    let effective_args: serde_json::Value = match action {
        ToolCallAction::Stop(reason) => {
            return Err(PromptError::prompt_cancelled(
                error_history.to_vec(),
                reason,
            ));
        }
        ToolCallAction::Skip(reason) => {
            tracing::info!(tool_name = tool_name, reason = reason, "Tool call rejected");
            // Synthetic rejection: `Skipped` outcome, message delivered verbatim.
            // Still fires the `ToolResult` hook so a policy observes the skip.
            skipped = Some(ToolResult::skipped(reason));
            // A skip runs nothing; its effective args are the salvaged rewrite
            // (if any) so tracing/history stay consistent, though they go unused.
            salvaged_rewrite.unwrap_or_else(|| tool_call.function.arguments.clone())
        }
        ToolCallAction::Rewrite(replacement) => {
            // Proceeding rewrite: re-record the span so the trace, and the
            // downstream `ToolResult` event, reflect what the tool actually
            // received rather than what the model emitted.
            args = json_utils::serialize_json_value(&replacement);
            if record_content {
                tool_span.record("gen_ai.tool.call.arguments", &args);
            }
            tracing::debug!(
                tool_name = tool_name,
                "tool-call arguments rewritten by a hook"
            );
            replacement
        }
        ToolCallAction::Run => tool_call.function.arguments.clone(),
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
    let result_action = hooks
        .on_tool_result(
            ctx,
            ToolResultEvent {
                tool_name,
                tool_call_id: Some(tool_call.id.as_str()),
                internal_call_id,
                args: &args,
                presentation: exec.output(),
                raw_result: &exec,
                tool_context: &dispatch_context,
            },
        )
        .await;
    // Outcome metadata describes the execution itself, while result content
    // follows the same presentation policy as the model. This keeps redaction
    // and stop hooks from leaking raw tool output through telemetry.
    record_tool_result(&tool_span, &exec);

    let result_content = match result_action {
        ToolResultAction::Stop(reason) => {
            return Err(PromptError::prompt_cancelled(
                error_history.to_vec(),
                reason,
            ));
        }
        ToolResultAction::Rewrite(replacement) => {
            if record_content {
                tool_span.record("gen_ai.tool.call.result", replacement.render());
            }
            replacement
        }
        ToolResultAction::Keep => {
            if record_content {
                tool_span.record("gen_ai.tool.call.result", exec.output().render());
            }
            exec.output().clone()
        }
    };
    let content = tool_result_output(
        tool_call.id.clone(),
        tool_call.provider.clone(),
        tool_call.function.name.clone(),
        result_content,
    );
    Ok(ToolCallOutcome { content, execution })
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
    record_telemetry_content: bool,
}

impl UnaryTurnSource {
    pub(crate) fn new(record_telemetry_content: bool) -> Self {
        Self {
            current_span_id: AtomicU64::new(0),
            record_telemetry_content,
        }
    }

    /// Chain `span` onto the previous step's span and record it as the new chain
    /// head, preserving the blocking driver's linear causal trace.
    fn chain_span(&self, span: tracing::Span) -> tracing::Span {
        let span = match self.current_span_id.load(Ordering::Relaxed) {
            0 => span,
            id => {
                span.follows_from(Id::from_u64(id));
                span
            }
        };
        if let Some(id) = span.id() {
            self.current_span_id.store(id.into_u64(), Ordering::Relaxed);
        }
        span
    }
}

impl TurnSource for UnaryTurnSource {
    fn open_chat_span(
        &self,
        runner: &AgentRunner,
        effective_preamble: Option<&str>,
    ) -> tracing::Span {
        let chat_span = build_chat_span!(runner, effective_preamble, "chat", "chat");
        self.chain_span(chat_span)
    }

    fn run_model_turn<'a>(
        &'a mut self,
        runner: &'a AgentRunner,
        hook_ctx: &'a HookContext,
        run: &'a mut AgentRun,
        prepared: PreparedCompletionRequest,
        chat_span: tracing::Span,
        _agent_span: &'a tracing::Span,
        current_prompt: Message,
    ) -> DriveStream<'a> {
        Box::pin(async_stream::stream! {
            // Content telemetry for the accepted provider turn. Called at each
            // terminal site (stop, terminate, accept) rather than hoisted: a
            // retried turn must not record output for the discarded attempt.
            let record_accepted_turn = |run: &AgentRun| {
                if runner.config.record_telemetry_content
                    && let Some(choice) = run.accepted_turn_choice()
                {
                    rig_core::telemetry::record_model_output(&chat_span, &choice, true);
                }
            };

            // Bound before the builder is consumed: this is the cap this exact
            // attempt was prepared with, patches included, and it is what the
            // per-turn hook reports. Reading it later off the agent config would
            // silently drop a completion-call hook's patch.
            let attempt_max_tokens = prepared.max_tokens;

            let resp = match prepared.builder.send().instrument(chat_span.clone()).await {
                Ok(resp) => resp,
                Err(err) => {
                    yield Err(StreamingError::from(err));
                    return;
                }
            };

            // Normalized once, then shared by run state and the per-turn hook, so
            // the two cannot report different reasons for one attempt.
            let attempt_finish_reason = resp.finish_reason();

            let mut outcome = match run.model_response(ModelTurn::from_response_parts(
                &resp,
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
                            .config.hooks
                            .on_invalid_tool_call(hook_ctx, &context)
                            .await;
                        let resolution = match action {
                            Some(action) => run.resolve_invalid_tool_call(action),
                            None
                                if runner.unhandled_invalid_tool_call_policy
                                    == UnhandledInvalidToolCallPolicy::IgnoreForExtractor =>
                            {
                                run.ignore_invalid_tool_call()
                            }
                            None => run.resolve_invalid_tool_call(InvalidToolCallAction::fail()),
                        };
                        outcome = match resolution {
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
                            // The response-finish event fires first, then the
                            // normalized per-turn event. The first observes;
                            // the second can accept, retry, or stop the canonical
                            // turn. Both are suppressed for recovered turns.
                            //
                            // Identity comes from this attempt's own `resp` —
                            // a retried turn re-enters `run_model_turn` with a
                            // fresh response, so a stale attempt's ids can
                            // never be attributed here. The raw payload is read
                            // from the same `resp` for the same reason.
                            let identity = resp.identity();
                            let attempt_raw = &resp.raw;
                            if let Some(reason) = observe_action(
                                runner
                                    .config.hooks
                                    .on_completion_response(
                                        hook_ctx,
                                        CompletionResponseEvent {
                                            prompt: &current_prompt,
                                            content: &resp.choice,
                                            usage: resp.usage,
                                            message_id: resp.message_id.as_deref(),
                                            identity: &identity,
                                            raw: attempt_raw,
                                        },
                                    )
                                    .await,
                            ) {
                                record_accepted_turn(run);
                                yield Err(StreamingError::Prompt(Box::new(run.cancel_error(reason))));
                                return;
                            }
                            let action = runner
                                .config.hooks
                                .on_model_turn_finished(
                                    hook_ctx,
                                    ModelTurnFinished {
                                        turn: hook_ctx.turn(),
                                        content: &resp.choice,
                                        usage: resp.usage,
                                        identity: &identity,
                                        finish_reason: attempt_finish_reason.as_ref(),
                                        max_tokens: attempt_max_tokens,
                                        raw: attempt_raw,
                                    },
                                )
                                .await;
                            match resolve_model_turn_action(run, action) {
                                Ok(ModelTurnDecision::Advance) => {}
                                Ok(ModelTurnDecision::Retried) => break,
                                Ok(ModelTurnDecision::Terminate(reason)) => {
                                    record_accepted_turn(run);
                                    yield Err(StreamingError::Prompt(Box::new(
                                        run.cancel_error(reason),
                                    )));
                                    return;
                                }
                                Err(err) => {
                                    yield Err(StreamingError::Prompt(Box::new(err)));
                                    return;
                                }
                            }
                        }

                        record_accepted_turn(run);
                        break;
                    }
                }
            }
        })
    }

    fn run_tool_calls<'a>(
        &'a self,
        runner: &'a AgentRunner,
        hook_ctx: &'a HookContext,
        run: &'a mut AgentRun,
        calls: Vec<PendingToolCall>,
        tool_snapshot: Arc<ToolRegistrySnapshot>,
    ) -> DriveStream<'a> {
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
            if self.record_telemetry_content {
                agent_span.record("gen_ai.completion", &response.output);
            }
            agent_span.record_token_usage(&response.usage);
        }
    }

    fn final_item(&self, _response: &PromptResponse) -> Option<MultiTurnStreamItem> {
        // The blocking surface folds the engine and discards the final item, so
        // building it (an extra full-response clone) is skipped entirely.
        None
    }
}

impl AgentRunner {
    pub(crate) async fn run_with_error_usage(
        mut self,
    ) -> (Result<PromptResponse, PromptError>, Usage) {
        let usage = Arc::new(Mutex::new(Usage::new()));
        self.error_usage = Some(usage.clone());
        let result = self.run().await;
        let observed = result.as_ref().map_or_else(
            |_| {
                *usage
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
            },
            |response| response.usage,
        );
        (result, observed)
    }

    /// Open the per-run agent span, recording the prompt when content
    /// telemetry is enabled. Shared by the blocking and streaming surfaces.
    pub(crate) fn open_agent_span(&self) -> (tracing::Span, bool) {
        let (agent_span, created_agent_span) = acquire_agent_span(
            self.agent_name_or_default(),
            self.config.preamble.as_deref(),
            self.config.record_telemetry_content,
        );

        if self.config.record_telemetry_content
            && let Some(text) = self.prompt.rag_text()
        {
            agent_span.record("gen_ai.prompt", text);
        }

        (agent_span, created_agent_span)
    }

    /// Resolve the history override and memory handle for this run.
    ///
    /// When the caller passes explicit history, memory is fully bypassed
    /// (no load AND no save). Otherwise, if a memory backend and conversation
    /// id are both configured, prior history is loaded. Each surface adapts a
    /// load failure to its own error channel.
    pub(crate) async fn resolve_history_and_memory(
        &self,
    ) -> Result<HistoryAndMemory, rig_core::memory::MemoryError> {
        match &self.chat_history {
            Some(_) => Ok((None, None)),
            None => match (&self.config.memory, &self.config.conversation_id) {
                (Some(memory), Some(id)) => {
                    let loaded = memory.load(id).await?;
                    Ok((Some(loaded), Some((memory.clone(), id.clone()))))
                }
                _ => Ok((None, None)),
            },
        }
    }

    /// Drive the agent loop to completion, returning the aggregated
    /// [`PromptResponse`]. Hooks fire at every observable point; the first hook
    /// to terminate cancels the run.
    pub async fn run(self) -> Result<PromptResponse, PromptError> {
        let (agent_span, created_agent_span) = self.open_agent_span();
        let (history_override, memory_handle) = self.resolve_history_and_memory().await?;
        let run = self.build_run(history_override);

        // Fold the shared engine to its final response. The blocking surface
        // uses a unary model transport and ignores the intermediate items the
        // engine yields; the engine is driven under the caller's ambient span
        // (no `instrument`), keeping the agent span detached and the chat/tool
        // spans on the blocking `follows_from` chain.
        let record_telemetry_content = self.config.record_telemetry_content;
        let driver = drive_agent(
            self,
            UnaryTurnSource::new(record_telemetry_content),
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
mod tests;

#[cfg(test)]
#[allow(irrefutable_let_patterns, unreachable_patterns)]
mod migrated_tests;
