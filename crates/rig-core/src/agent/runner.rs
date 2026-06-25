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

use futures::{StreamExt, stream};
use tracing::{Instrument, info_span, span::Id};

use super::{
    completion::{Agent, DynamicContextStore, build_prepared_completion_request},
    hook::{AgentHook, Flow, HookStack, InvalidToolCallHookAction, StepEvent},
    prompt_request::{PromptResponse, tool_result_message, tool_result_output},
    run::{
        AgentRun, AgentRunStep, DEFAULT_OUTPUT_RETRIES, ModelTurn, ModelTurnOutcome, OutputMode,
        PendingToolCall,
    },
};
use crate::{
    completion::{CompletionModel, Document, Message, PromptError},
    json_utils,
    memory::ConversationMemory,
    message::{ToolCall, ToolChoice, UserContent},
    tool::{ToolCallExtensions, server::ToolServerHandle},
};

const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

/// Human-readable name of a [`Flow`] variant, for fail-closed diagnostics.
fn flow_name(flow: &Flow) -> &'static str {
    match flow {
        Flow::Continue => "Continue",
        Flow::Terminate { .. } => "Terminate",
        Flow::Skip { .. } => "Skip",
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
    /// Skip execution and return `reason` to the model as the tool result.
    Skip(String),
    /// Terminate the run.
    Terminate(String),
}

/// Resolve a hook's [`Flow`] for a [`StepEvent::ToolCall`] event (honors
/// `Continue`/`Skip`/`Terminate`). **Fail-closed**: any other action (e.g.
/// `Fail`/`Retry`/`Repair`) never executes the tool — it terminates the run.
pub(crate) fn flow_into_tool_call(flow: Flow) -> ToolCallDecision {
    match flow {
        Flow::Continue => ToolCallDecision::Proceed,
        Flow::Skip { reason } => ToolCallDecision::Skip(reason),
        Flow::Terminate { reason } => ToolCallDecision::Terminate(reason),
        other => ToolCallDecision::Terminate(format!(
            "hook returned `{}` for a tool-call event, which only honors \
             Continue/Skip/Terminate — terminating the run (fail-closed) rather \
             than executing the tool",
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
    /// Per-call runtime context made available to every tool executed during
    /// this run via [`Tool::call_with_extensions`](crate::tool::Tool::call_with_extensions).
    /// Empty by default; populated with [`tool_extensions`](Self::tool_extensions).
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
    /// Hooks run in registration order; the first to return a
    /// non-[`Flow::Continue`] result short-circuits the rest.
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
    /// The context is threaded to every tool the agent executes, so tools can
    /// read caller-provided values (auth tokens, session IDs, conversation
    /// state, …) via [`Tool::call_with_extensions`](crate::tool::Tool::call_with_extensions)
    /// without the model ever seeing them. Replaces any context already set.
    pub fn with_tool_extensions(mut self, extensions: ToolCallExtensions) -> Self {
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

    /// Execute up to `concurrency` tools at once (1 by default). Only affects
    /// the blocking [`run`](Self::run) path (streaming executes tools
    /// sequentially).
    ///
    /// The resulting tool-result **order — and so the message history — is the
    /// same** in both paths regardless of `concurrency` (`run()` collects with
    /// `buffered`, which preserves tool-call order). At the default
    /// `concurrency` of 1 the two paths are fully in lock-step; with
    /// `concurrency > 1` the tools run in parallel, so a `ToolCall`/`ToolResult`
    /// **hook may fire in completion order** rather than call order — the
    /// per-tool side effects interleave even though the final history does not.
    ///
    /// A `concurrency` of 0 is clamped to 1: `buffered(0)` never makes progress,
    /// so it would otherwise hang the run the first time the model calls a tool.
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

/// Execute a single tool call, firing the `ToolCall` and `ToolResult` hooks and
/// shaping the result. **Shared by the blocking and streaming drivers** so a
/// tool call behaves identically in both: same hook events, same fail-closed
/// skip/terminate handling, and the same result shaping — a hook skip reason is
/// emitted verbatim ([`tool_result_message`]) while a real tool output is parsed
/// ([`tool_result_output`]). Records `gen_ai.tool.*` on the current span;
/// `error_history` builds a cancellation error if a hook terminates the run.
pub(crate) async fn run_single_tool<M>(
    hooks: &HookStack<M>,
    tool_server: &ToolServerHandle,
    tool_extensions: &ToolCallExtensions,
    tool_call: &ToolCall,
    internal_call_id: &str,
    error_history: &[Message],
) -> Result<UserContent, PromptError>
where
    M: CompletionModel,
{
    let tool_name = &tool_call.function.name;
    let args = json_utils::value_to_json_string(&tool_call.function.arguments);

    let tool_span = tracing::Span::current();
    tool_span.record("gen_ai.tool.name", tool_name);
    tool_span.record("gen_ai.tool.call.id", &tool_call.id);
    tool_span.record("gen_ai.tool.call.arguments", &args);

    match flow_into_tool_call(
        hooks
            .on_event(StepEvent::ToolCall {
                tool_name,
                tool_call_id: tool_call.call_id.as_deref(),
                internal_call_id,
                args: &args,
            })
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
            // Synthetic rejection message: emit verbatim, never re-parsed.
            return Ok(tool_result_message(
                tool_call.id.clone(),
                tool_call.call_id.clone(),
                reason,
            ));
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

    // Record the result on the span before consulting the result hook, so the
    // trace keeps the output even if a hook then terminates the run.
    tool_span.record("gen_ai.tool.call.result", &output);
    tracing::info!("executed tool {tool_name} with args {args}. result: {output}");

    if let Some(reason) = observe_flow(
        hooks
            .on_event(StepEvent::ToolResult {
                tool_name,
                tool_call_id: tool_call.call_id.as_deref(),
                internal_call_id,
                args: &args,
                result: &output,
            })
            .await,
    ) {
        return Err(PromptError::prompt_cancelled(
            error_history.to_vec(),
            reason,
        ));
    }

    // Real tool output: parsed (may be multimodal).
    Ok(tool_result_output(
        tool_call.id.clone(),
        tool_call.call_id.clone(),
        output,
    ))
}

impl<M> AgentRunner<M>
where
    M: CompletionModel,
{
    /// Drive the agent loop to completion, returning the aggregated
    /// [`PromptResponse`]. Hooks fire at every observable point; the first hook
    /// to terminate cancels the run.
    pub async fn run(self) -> Result<PromptResponse, PromptError> {
        let agent_span = if tracing::Span::current().is_disabled() {
            info_span!(
                "invoke_agent",
                gen_ai.operation.name = "invoke_agent",
                gen_ai.agent.name = self.agent_name_or_default(),
                gen_ai.system_instructions = self.preamble,
                gen_ai.prompt = tracing::field::Empty,
                gen_ai.completion = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
                gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
                gen_ai.usage.tool_use_prompt_tokens = tracing::field::Empty,
                gen_ai.usage.reasoning_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        if let Some(text) = self.prompt.rag_text() {
            agent_span.record("gen_ai.prompt", text);
        }

        let agent_name_for_span = self.agent_name.clone();

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

        let mut run = self.build_run(history_override);
        let current_span_id: AtomicU64 = AtomicU64::new(0);

        loop {
            match run.next_step()? {
                AgentRunStep::CallModel {
                    prompt,
                    history,
                    turn,
                } => {
                    if self.max_turns > 1 {
                        tracing::info!("Current conversation depth: {}/{}", turn, self.max_turns);
                    }

                    if let Some(reason) = observe_flow(
                        self.hooks
                            .on_event(StepEvent::CompletionCall {
                                prompt: &prompt,
                                history: &history,
                                turn,
                            })
                            .await,
                    ) {
                        return Err(run.cancel_error(reason));
                    }

                    let span = tracing::Span::current();
                    let chat_span = info_span!(
                        target: "rig::agent_chat",
                        parent: &span,
                        "chat",
                        gen_ai.operation.name = "chat",
                        gen_ai.agent.name = agent_name_for_span.as_deref().unwrap_or(UNKNOWN_AGENT_NAME),
                        gen_ai.system_instructions = self.preamble,
                        gen_ai.provider.name = tracing::field::Empty,
                        gen_ai.request.model = tracing::field::Empty,
                        gen_ai.response.id = tracing::field::Empty,
                        gen_ai.response.model = tracing::field::Empty,
                        gen_ai.usage.output_tokens = tracing::field::Empty,
                        gen_ai.usage.input_tokens = tracing::field::Empty,
                        gen_ai.usage.cache_read.input_tokens = tracing::field::Empty,
                        gen_ai.usage.cache_creation.input_tokens = tracing::field::Empty,
                        gen_ai.usage.tool_use_prompt_tokens = tracing::field::Empty,
                        gen_ai.usage.reasoning_tokens = tracing::field::Empty,
                        gen_ai.input.messages = tracing::field::Empty,
                        gen_ai.output.messages = tracing::field::Empty,
                    );

                    let chat_span = if current_span_id.load(Ordering::SeqCst) != 0 {
                        let id = Id::from_u64(current_span_id.load(Ordering::SeqCst));
                        chat_span.follows_from(id).to_owned()
                    } else {
                        chat_span
                    };

                    if let Some(id) = chat_span.id() {
                        current_span_id.store(id.into_u64(), Ordering::SeqCst);
                    };

                    // Pin Tool output mode once committed so later turns stay
                    // consistent even if the per-turn tool set changes (#1928).
                    let committed_output_tool = run.output_tool_name().map(str::to_owned);
                    let prepared_request = build_prepared_completion_request(
                        &self.model,
                        prompt.clone(),
                        &history,
                        self.preamble.as_deref(),
                        &self.static_context,
                        self.temperature,
                        self.max_tokens,
                        self.additional_params.as_ref(),
                        self.tool_choice.as_ref(),
                        &self.tool_server_handle,
                        &self.dynamic_context,
                        self.output_schema.as_ref(),
                        &self.output_mode,
                        committed_output_tool.as_deref(),
                    )
                    .await?;

                    let resp = prepared_request
                        .builder
                        .send()
                        .instrument(chat_span.clone())
                        .await?;

                    run.set_output_tool_name(prepared_request.output_tool_name.clone());

                    let mut outcome = run.model_response(ModelTurn::new(
                        resp.message_id.clone(),
                        resp.choice.clone(),
                        resp.usage,
                        prepared_request.executable_tool_names,
                        prepared_request.allowed_tool_names,
                    ))?;

                    loop {
                        match outcome {
                            ModelTurnOutcome::NeedsResolution(context) => {
                                let flow = self
                                    .hooks
                                    .on_event(StepEvent::InvalidToolCall(&context))
                                    .await;
                                match flow_into_invalid(flow) {
                                    InvalidDecision::Terminate(reason) => {
                                        return Err(run.cancel_error(reason));
                                    }
                                    InvalidDecision::Action(action) => {
                                        outcome = run.resolve_invalid_tool_call(action)?;
                                    }
                                }
                            }
                            ModelTurnOutcome::TurnRetried => break,
                            ModelTurnOutcome::Continue {
                                response_hook_suppressed,
                            } => {
                                if !response_hook_suppressed
                                    && let Some(reason) = observe_flow(
                                        self.hooks
                                            .on_event(StepEvent::CompletionResponse {
                                                prompt: &prompt,
                                                response: &resp,
                                            })
                                            .await,
                                    )
                                {
                                    return Err(run.cancel_error(reason));
                                }
                                break;
                            }
                        }
                    }
                }
                AgentRunStep::CallTools { calls } => {
                    let hooks = &self.hooks;
                    let tool_server = &self.tool_server_handle;
                    let tool_extensions = &self.tool_extensions;
                    // Materialize the diagnostic history once; tools only read it
                    // (verbatim, on a hook-terminate error path), so every tool
                    // future shares a single borrow instead of deep-cloning the
                    // whole conversation per call on the common success path.
                    let full_history_for_errors = run.full_history();
                    let error_history: &[Message] = &full_history_for_errors;

                    let tool_content = stream::iter(calls)
                        .map(|pending| {
                            let tool_span = info_span!(
                                "execute_tool",
                                gen_ai.operation.name = "execute_tool",
                                gen_ai.tool.type = "function",
                                gen_ai.tool.name = tracing::field::Empty,
                                gen_ai.tool.call.id = tracing::field::Empty,
                                gen_ai.tool.call.arguments = tracing::field::Empty,
                                gen_ai.tool.call.result = tracing::field::Empty
                            );

                            let tool_span = if current_span_id.load(Ordering::SeqCst) != 0 {
                                let id = Id::from_u64(current_span_id.load(Ordering::SeqCst));
                                tool_span.follows_from(id).to_owned()
                            } else {
                                tool_span
                            };

                            if let Some(id) = tool_span.id() {
                                current_span_id.store(id.into_u64(), Ordering::SeqCst);
                            };

                            async move {
                                let PendingToolCall {
                                    tool_call,
                                    preresolved_result,
                                    ..
                                } = pending;
                                // Tool calls suppressed by invalid tool-call
                                // recovery come pre-resolved and must not execute.
                                if let Some(result) = preresolved_result {
                                    return Ok(result);
                                }
                                let internal_call_id = crate::id::generate();
                                run_single_tool(
                                    hooks,
                                    tool_server,
                                    tool_extensions,
                                    &tool_call,
                                    &internal_call_id,
                                    error_history,
                                )
                                .await
                            }
                            .instrument(tool_span)
                        })
                        // `buffered` (not `buffer_unordered`) so results land in
                        // tool-call emission order — matching the streaming
                        // driver's sequential order, so both produce the same
                        // message history even at concurrency > 1.
                        .buffered(self.concurrency)
                        .collect::<Vec<Result<UserContent, PromptError>>>()
                        .await
                        .into_iter()
                        .collect::<Result<Vec<_>, _>>()?;

                    run.tool_results(tool_content)?;
                }
                AgentRunStep::Done(response) => {
                    if self.max_turns > 1 {
                        tracing::info!("Depth reached: {}/{}", run.turn(), self.max_turns);
                    }

                    let usage = response.usage;
                    agent_span.record("gen_ai.completion", &response.output);
                    agent_span.record("gen_ai.usage.input_tokens", usage.input_tokens);
                    agent_span.record("gen_ai.usage.output_tokens", usage.output_tokens);
                    agent_span.record(
                        "gen_ai.usage.cache_read.input_tokens",
                        usage.cached_input_tokens,
                    );
                    agent_span.record(
                        "gen_ai.usage.cache_creation.input_tokens",
                        usage.cache_creation_input_tokens,
                    );
                    agent_span.record(
                        "gen_ai.usage.tool_use_prompt_tokens",
                        usage.tool_use_prompt_tokens,
                    );
                    agent_span.record("gen_ai.usage.reasoning_tokens", usage.reasoning_tokens);

                    if let Some((memory, id)) = memory_handle.as_ref()
                        && let Err(err) = memory
                            .append(id, response.messages.clone().unwrap_or_default())
                            .await
                    {
                        tracing::warn!(
                            error = %err,
                            conversation_id = %id,
                            "conversation memory append failed; returning model response anyway"
                        );
                    }

                    return Ok(response);
                }
            }
        }
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
    use crate::agent::hook::{AgentHook, Flow, StepEvent, StepEventKind};
    use crate::agent::prompt_request::streaming::MultiTurnStreamItem;
    use crate::completion::{CompletionModel, Message, PromptError, ToolDefinition};
    use crate::message::{AssistantContent, ToolCall, ToolFunction, UserContent};
    use crate::test_utils::{
        MockAddTool, MockCompletionModel, MockOperationArgs, MockStreamEvent, MockToolError,
        MockTurn,
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
        async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
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
    /// driver. (`run()` uses `buffered`, which preserves call order.)
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

    /// A tool whose first-*called* invocation completes *after* the second, so a
    /// completion-ordered combinator (`buffer_unordered`) would reorder results
    /// while call-ordered `buffered` does not. The first call (in poll/call
    /// order) waits on a gate the second call releases.
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

    /// `run()` must surface tool results in tool-call (emission) order even when
    /// tools complete out of order under concurrency — i.e. it uses `buffered`,
    /// not `buffer_unordered`. (This is what keeps its message history identical
    /// to the sequential streaming driver.)
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
        async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
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
        async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
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
        async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
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
        async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
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
        async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
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
        async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
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
}
