use rig_core::{
    OneOrMany,
    message::{AssistantContent, UserContent},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};

use crate::{
    agent::completion::{PreparedCompletionRequest, build_prepared_completion_request},
    agent::hook::InvalidToolCallAction,
    agent::prompt_request::{assistant_text_from_choice, is_empty_assistant_turn},
    agent::run::{
        AgentRun, AgentRunStep, PendingToolCall,
        streamed::{StreamedResolution, StreamedTurnAssembler, StreamedTurnEvent},
    },
    agent::runner::{
        AgentRunner, CompletionCallOutcome, ModelTurnDecision, ToolExecution, UnaryTurnSource,
        acquire_agent_span, build_chat_span, new_execute_tool_span, observe_action,
        resolve_completion_call, resolve_model_turn_action, run_single_tool,
    },
    executor::ToolExecutor,
    hooks::{HookEntry, Hooks},
    streaming::{StreamedAssistantContent, StreamedUserContent, ToolCallDeltaContent},
};
use futures::{Stream, StreamExt, stream};
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, pin::Pin, sync::Arc};
use tracing_futures::Instrument;

use super::{CompletionCall, PromptResponse, forward_prompt_setters};
use crate::{
    agent::Agent,
    completion::{CompletionError, PromptError},
};
use rig_core::message::{Message, Text};

// The `Send` bound is dropped exactly where `rig-core`'s `WasmCompat*` markers
// go no-op — browser wasm. `rig-core` keys those markers on this same
// predicate, so keep the two in step: a bare `target_arch = "wasm32"` would
// also drop `Send` on WASI, where `rig-core` still requires it.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub type StreamingResult =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem, StreamingError>> + Send>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub type StreamingResult = Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem, StreamingError>>>>;

#[derive(Deserialize, Serialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
#[non_exhaustive]
pub enum MultiTurnStreamItem {
    /// A streamed assistant content item — the content the **model emitted**:
    /// text/reasoning deltas, tool-call deltas, and, when the model turn is
    /// committed, the complete [`StreamedAssistantContent::ToolCall`] for each
    /// tool call Rig routes to execution. Such a call is reported here whether or
    /// not the tool body ultimately runs (a hook skip still reports it);
    /// it is **not** an execution-lifecycle event (see
    /// [`ToolExecutionCommitted`](Self::ToolExecutionCommitted)).
    ///
    /// Two kinds of model tool call are **not** re-emitted as a complete
    /// `ToolCall` item here (their arguments still stream as tool-call deltas):
    /// a call rejected and handled by invalid-tool-call recovery (surfaced via
    /// that recovery path), and a structured-output Tool-mode output-tool call,
    /// which finalizes the run directly — its structured result is surfaced in
    /// the [`FinalResponse`](Self::FinalResponse) rather than as a completed
    /// `ToolCall` item.
    StreamAssistantItem(StreamedAssistantContent),
    /// Confirmation that Rig **executed and committed** a tool call. This is not
    /// a real-time start notification: it is surfaced together with its
    /// `ToolResult` only after the whole batch settles successfully. Use tool
    /// hooks for live host-side start/result observation.
    ///
    /// This item is emitted only for a tool whose body actually ran (it passed
    /// its `ToolCall` hook checks), never for a call dropped by a sibling's
    /// termination, skipped by a hook, or resolved by invalid-call recovery.
    /// Correlate it with the model call and result through `internal_call_id`.
    ToolExecutionCommitted {
        /// The tool call as **executed**: the model's call with any
        /// [`ToolCallAction::Rewrite`](crate::agent::ToolCallAction::Rewrite) hook rewrite
        /// applied (so a redaction rewrite is reflected here, not leaked). The
        /// model's *original* call is reported via
        /// [`StreamAssistantItem`](Self::StreamAssistantItem).
        tool_call: rig_core::message::ToolCall,
        /// Rig-generated id correlating this execution with the model tool call
        /// ([`StreamedAssistantContent::ToolCall::internal_call_id`]) and the
        /// resulting [`StreamedUserContent::ToolResult`].
        internal_call_id: String,
    },
    /// A streamed user content item: the **result** of an executed (or
    /// hook-skipped) tool call. The tool batch commits and surfaces atomically at
    /// every `tool_concurrency` (including the sequential default): results are
    /// surfaced (in call order) only after the whole batch settles successfully —
    /// a run that terminates mid-batch surfaces no successful tool results.
    StreamUserItem(StreamedUserContent),
    /// Details for one successfully completed completion request made by this agent stream.
    ///
    /// This is emitted when a provider call finishes. Usage is the provider's
    /// final usage for that completion request when available; it is not
    /// incremental per streamed token.
    ///
    /// ```rust,ignore
    /// match item {
    ///     MultiTurnStreamItem::CompletionCall(completion_call) => {
    ///         // Zero-valued usage means the provider reported no metrics.
    ///         if completion_call.usage.has_values() {
    ///             let context_tokens = completion_call.usage.input_tokens;
    ///         }
    ///     }
    ///     _ => {}
    /// }
    /// ```
    CompletionCall(CompletionCall),
    /// The completed model turn was rejected by a hook for retry.
    ///
    /// Text and reasoning deltas emitted for this turn were provisional. A
    /// consumer should discard or visually reset output associated with `turn`.
    /// A subsequent attempt is made only if the run's total model-call budget
    /// permits it.
    ModelTurnRetried {
        /// One-based model-call index of the rejected turn.
        turn: usize,
    },
    /// The final result from the stream: the unified [`PromptResponse`] shared
    /// with the blocking surface.
    FinalResponse(PromptResponse),
}

/// Build the unified [`PromptResponse`] for the streaming surface from the
/// final turn's structured content.
fn final_response_from_content(
    content: OneOrMany<AssistantContent>,
    aggregated_usage: crate::completion::Usage,
    completion_calls: Vec<CompletionCall>,
    history: Option<Vec<Message>>,
) -> PromptResponse {
    let mut response = PromptResponse::new(assistant_text_from_choice(&content), aggregated_usage)
        .with_content(content)
        .with_completion_calls(completion_calls);
    response.messages = history;
    response
}

impl MultiTurnStreamItem {
    pub(crate) fn stream_item(item: StreamedAssistantContent) -> Self {
        Self::StreamAssistantItem(item)
    }

    pub fn final_response(
        content: OneOrMany<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
    ) -> Self {
        Self::FinalResponse(final_response_from_content(
            content,
            aggregated_usage,
            Vec::new(),
            None,
        ))
    }

    pub fn final_response_with_history(
        content: OneOrMany<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
        history: Option<Vec<Message>>,
    ) -> Self {
        Self::FinalResponse(final_response_from_content(
            content,
            aggregated_usage,
            Vec::new(),
            history,
        ))
    }

    pub(crate) fn final_response_with_completion_calls(
        content: OneOrMany<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
        completion_calls: Vec<CompletionCall>,
        history: Option<Vec<Message>>,
    ) -> Self {
        Self::FinalResponse(final_response_from_content(
            content,
            aggregated_usage,
            completion_calls,
            history,
        ))
    }
}

/// Drain a provider stream abandoned by invalid tool-call recovery so the
/// reported usage for the recovered completion call is not lost.
async fn drain_stream_usage(
    stream: &mut crate::streaming::StreamingCompletionResponse,
) -> Result<crate::completion::Usage, StreamingError> {
    while let Some(content) = stream.next().await {
        match content {
            Ok(StreamedAssistantContent::Final(final_resp)) => {
                return Ok(final_resp.usage);
            }
            Ok(_) => {}
            Err(err) => return Err(err.into()),
        }
    }

    Ok(crate::completion::Usage::new())
}

pub(crate) fn record_usage_on_span(span: &tracing::Span, usage: crate::completion::Usage) {
    span.record("gen_ai.usage.input_tokens", usage.input_tokens);
    span.record("gen_ai.usage.output_tokens", usage.output_tokens);
    span.record(
        "gen_ai.usage.cache_read.input_tokens",
        usage.cached_input_tokens,
    );
    span.record(
        "gen_ai.usage.cache_creation.input_tokens",
        usage.cache_creation_input_tokens,
    );
    span.record(
        "gen_ai.usage.tool_use_prompt_tokens",
        usage.tool_use_prompt_tokens,
    );
    span.record("gen_ai.usage.reasoning_tokens", usage.reasoning_tokens);
}

/// Build the final streamed content for a finished run (#1928).
///
/// When the finishing turn carries a tool call it is a Tool-mode output-tool
/// call (a real tool call would have routed to `CallTools`, not `Done`). In that
/// case the tool call AND the model's prose are dropped, any reasoning/image
/// content is kept, and `output` is appended as the final text — so the streamed
/// [`PromptResponse::output`] string is the structured output rather than the
/// prose, with no unanswered tool_use, matching the non-streaming `output`. Note
/// this shapes only the surfaced [`PromptResponse::content`]; the persisted
/// message history is built by the state machine (which keeps the prose, like the
/// blocking driver), so `content` and `messages` intentionally differ on prose in
/// this case.
/// Otherwise returns `None` and the caller surfaces the turn's content unchanged.
fn finalize_streamed_choice(
    last_final_choice: &OneOrMany<AssistantContent>,
    output: &str,
) -> Option<OneOrMany<AssistantContent>> {
    let finalized_via_output_tool = last_final_choice
        .iter()
        .any(|item| matches!(item, AssistantContent::ToolCall(_)));
    if !finalized_via_output_tool {
        return None;
    }
    let mut items: Vec<AssistantContent> = last_final_choice
        .iter()
        .filter(|item| {
            !matches!(
                item,
                AssistantContent::ToolCall(_) | AssistantContent::Text(_)
            )
        })
        .cloned()
        .collect();
    items.push(AssistantContent::text(output.to_string()));
    Some(
        OneOrMany::from_iter_optional(items)
            .unwrap_or_else(|| OneOrMany::one(AssistantContent::text(output.to_string()))),
    )
}

#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
    #[error("CompletionError: {0}")]
    Completion(#[from] CompletionError),
    #[error("PromptError: {0}")]
    Prompt(#[from] Box<PromptError>),
}

/// A builder for creating prompt requests with customizable options.
/// Uses generics to track which options have been set during the build process.
///
/// When the agent has no configured `default_max_turns`, the implicit budget is
/// one model call. Use [`.max_turns()`](Self::max_turns) to override the agent's
/// configured or implicit budget; a tool call followed by a model-authored final
/// answer generally requires at least two model calls.
pub struct StreamingPromptRequest {
    /// The hook-aware driver this streaming request configures and runs.
    runner: AgentRunner,
}

impl StreamingPromptRequest {
    /// Create a new `StreamingPromptRequest` from an agent, including its
    /// default hooks.
    pub fn new(agent: Arc<Agent>, prompt: impl Into<Message>) -> StreamingPromptRequest {
        Self::from_agent(agent.as_ref(), prompt)
    }

    /// Create a new StreamingPromptRequest from an agent, cloning the agent's
    /// data and default hook stack.
    pub fn from_agent(agent: &Agent, prompt: impl Into<Message>) -> StreamingPromptRequest {
        StreamingPromptRequest {
            runner: AgentRunner::from_agent(agent, prompt),
        }
    }

    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. Zero emits no model calls; one permits only the
    /// initial call.
    ///
    /// Named to match the blocking
    /// [`PromptRequest::max_turns`](super::PromptRequest::max_turns) and
    /// [`TypedPromptRequest::max_turns`](super::TypedPromptRequest::max_turns)
    /// builders so the same call reads identically on either surface.
    pub fn max_turns(mut self, turns: usize) -> Self {
        self.runner = self.runner.max_turns(turns);
        self
    }

    /// Execute up to `concurrency` of a turn's tool calls at once (1 by default,
    /// i.e. sequential). See [`AgentRunner::tool_concurrency`]: at any
    /// `concurrency` the stream emits the model's `ToolCall` items (call order),
    /// then — atomically, after the whole tool batch settles successfully — the
    /// per-tool `ToolExecutionCommitted` + `ToolResult` items in **call order** (not
    /// completion order). The streamed message history is unchanged at any
    /// `concurrency`.
    pub fn tool_concurrency(mut self, concurrency: usize) -> Self {
        self.runner = self.runner.tool_concurrency(concurrency);
        self
    }

    /// Append a hook to this request's hook stack (on top of any the agent
    /// already carries). Hooks run in registration order; how their results
    /// compose is event-dependent (`CompletionCall` request patches accumulate
    /// and merge, `ToolCall`/`ToolResult` rewrites chain, while model-turn
    /// steering and observe-only/recovery events use first-non-`Continue`-wins). See the
    /// [`hook`](crate::agent::hook) module docs.
    pub fn add_hook(mut self, hook: HookEntry) -> Self {
        self.runner = self.runner.add_hook(hook);
        self
    }

    forward_prompt_setters!(runner);

    async fn send(self) -> StreamingResult {
        self.runner.stream().await
    }
}

/// A boxed, medium-specific item stream for one engine step (model turn or tool
/// batch). Boxed so a generic [`drive_agent`] can forward it without the
/// per-step future leaking into the engine's own (`Send`) inference.
// Same browser-wasm predicate as `StreamingResult` above, for the same reason.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) type DriveStream<'a> =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem, StreamingError>> + Send + 'a>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub(crate) type DriveStream<'a> =
    Pin<Box<dyn Stream<Item = Result<MultiTurnStreamItem, StreamingError>> + 'a>>;

/// One item emitted by the shared engine [`drive_agent`].
///
/// `Item`s are forwarded to a streaming consumer (and ignored by the blocking
/// fold); `Done` carries both the canonical [`PromptResponse`] the blocking
/// surface returns and the medium-specific final stream item the streaming
/// surface yields.
// The large `Item` variant is the per-delta hot path (one per streamed token);
// boxing it to shrink the variant spread would add an allocation per delta,
// which the streaming path is specifically tuned to avoid. `Done` is yielded
// once per run, so the wasted space on that rare variant is irrelevant.
#[allow(clippy::large_enum_variant)]
pub(crate) enum DriveItem {
    /// An intermediate stream item (assistant delta, tool call/result, a
    /// per-call `CompletionCall`, or — last, for the streaming surface — the
    /// final response item).
    Item(MultiTurnStreamItem),
    /// The run finished; carries the canonical response the blocking fold
    /// returns. The streaming surface has already received the final item as the
    /// preceding `Item` and ignores this.
    Done(Box<PromptResponse>),
}

/// The per-medium half of the agent loop: how a turn is fetched from the model,
/// how its tools are executed, and how the run's spans/usage/final item are
/// shaped. The medium-independent outer loop (turn counting, the `CompletionCall`
/// hook, request preparation) lives once in [`drive_agent`]; only the
/// genuinely divergent pieces are behind this enum. Invalid-tool-call recovery
/// is one of them — it lives inside each source's `run_model_turn` (end-of-turn
/// for blocking, mid-stream for streaming), not in `drive_agent`.
#[allow(clippy::large_enum_variant)]
pub(crate) enum TurnSource {
    /// The blocking surface: unary completion calls (see [`UnaryTurnSource`]).
    Unary(UnaryTurnSource),
    /// The streaming surface: provider streams (see [`StreamingTurnSource`]).
    Streaming(StreamingTurnSource),
}

impl TurnSource {
    /// Build this medium's per-turn `chat` span (name + parenting + any
    /// `follows_from` chaining differ between blocking and streaming).
    fn open_chat_span(
        &self,
        runner: &AgentRunner,
        effective_preamble: Option<&str>,
    ) -> tracing::Span {
        match self {
            TurnSource::Unary(source) => source.open_chat_span(runner, effective_preamble),
            TurnSource::Streaming(source) => source.open_chat_span(runner, effective_preamble),
        }
    }

    /// Run one model turn: issue the provider call, feed the result into the
    /// sans-IO machine, and yield any intermediate items. Returning normally
    /// advances the loop; yielding an `Err` terminates the run.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn run_model_turn<'a>(
        &'a mut self,
        runner: &'a AgentRunner,
        turn: usize,
        run: &'a mut AgentRun,
        prepared: PreparedCompletionRequest,
        chat_span: tracing::Span,
        agent_span: &'a tracing::Span,
        prompt: Message,
    ) -> DriveStream<'a> {
        match self {
            TurnSource::Unary(source) => {
                source.run_model_turn(runner, turn, run, prepared, chat_span, agent_span, prompt)
            }
            TurnSource::Streaming(source) => {
                source.run_model_turn(runner, turn, run, prepared, chat_span, agent_span, prompt)
            }
        }
    }

    /// Execute a turn's tool calls, feeding the results into the machine and
    /// yielding any intermediate items.
    fn run_tool_calls<'a>(
        &'a self,
        runner: &'a AgentRunner,
        run: &'a mut AgentRun,
        calls: Vec<PendingToolCall>,
        tool_executor: Arc<ToolExecutor>,
    ) -> DriveStream<'a> {
        match self {
            TurnSource::Unary(source) => source.run_tool_calls(runner, run, calls, tool_executor),
            TurnSource::Streaming(source) => {
                source.run_tool_calls(runner, run, calls, tool_executor)
            }
        }
    }

    /// Record run-level telemetry onto the agent span at `Done`. Gated on
    /// `created_agent_span` so a caller-supplied outer span is never polluted.
    fn record_run_level_telemetry(
        &self,
        agent_span: &tracing::Span,
        response: &PromptResponse,
        created_agent_span: bool,
    ) {
        match self {
            TurnSource::Unary(source) => {
                source.record_run_level_telemetry(agent_span, response, created_agent_span)
            }
            TurnSource::Streaming(source) => {
                source.record_run_level_telemetry(agent_span, response, created_agent_span)
            }
        }
    }

    /// Build the final stream item surfaced at `Done`, or `None` when the
    /// surface discards it (the blocking fold) so the engine skips the work.
    fn final_item(&self, response: &PromptResponse) -> Option<MultiTurnStreamItem> {
        match self {
            TurnSource::Unary(source) => source.final_item(response),
            TurnSource::Streaming(source) => source.final_item(response),
        }
    }
}

/// Convert a [`StreamingError`] back into a [`PromptError`] for the blocking
/// surface ([`AgentRunner::run`]), which folds the shared engine. Lossless:
/// every streaming error originates as one of these.
pub(crate) fn streaming_error_into_prompt(err: StreamingError) -> PromptError {
    match err {
        StreamingError::Completion(err) => PromptError::CompletionError(err),
        StreamingError::Prompt(err) => *err,
    }
}

pub(crate) fn store_error_usage(runner: &AgentRunner, run: &AgentRun) {
    if let Some(usage) = &runner.error_usage {
        *usage.lock().unwrap_or_else(|error| error.into_inner()) = run.usage();
    }
}

/// The single agent drive loop, shared by the blocking and streaming surfaces.
///
/// Owns the medium-independent loop — `next_step` dispatch, the `CompletionCall`
/// hook + request preparation — and delegates the
/// medium-specific model call, tool execution, span shaping and finalization to
/// a [`TurnSource`]. The streaming surface forwards the yielded [`DriveItem`]s;
/// the blocking surface folds them to `Done`.
pub(crate) fn drive_agent(
    runner: AgentRunner,
    mut source: TurnSource,
    mut run: AgentRun,
    agent_span: tracing::Span,
    created_agent_span: bool,
) -> impl Stream<Item = Result<DriveItem, StreamingError>> {
    async_stream::stream! {
        // Set only after a model turn commits successfully and consumed by its
        // immediately following CallTools step. This keeps the sans-IO run state
        // serializable while pinning execution to the definitions sent that turn.
        let mut pending_tool_executor: Option<Arc<ToolExecutor>> = None;

        'outer: loop {
            let step = match run.next_step() {
                Ok(step) => step,
                Err(err) => {
                    store_error_usage(&runner, &run);
                    yield Err(Box::new(err).into());
                    break 'outer;
                }
            };

            match step {
                AgentRunStep::CallModel { prompt, history, turn } => {
                    drop(pending_tool_executor.take());
                    if runner.max_turns > 1 {
                        tracing::info!("Current conversation Turns: {}/{}", turn, runner.max_turns);
                    }
                    let request_patch =
                        match resolve_completion_call(&runner.hooks, &prompt, &history, turn).await {
                            CompletionCallOutcome::Terminate(reason) => {
                                store_error_usage(&runner, &run);
                                yield Err(StreamingError::Prompt(Box::new(run.cancel_error(reason))));
                                break 'outer;
                            }
                            CompletionCallOutcome::Proceed(request_patch) => request_patch,
                        };

                    // Record this turn's base system prompt — the patched-or-baseline
                    // preamble, before any output-mode augmentation the request builder
                    // appends. Borrow rather than clone since it only needs to outlive
                    // span creation.
                    let effective_preamble = request_patch
                        .as_ref()
                        .and_then(|o| o.preamble.as_deref())
                        .or(runner.preamble.as_deref());

                    let chat_span = source.open_chat_span(&runner, effective_preamble);

                    // Pin Tool output mode once committed so later turns stay
                    // consistent even if the per-turn tool set changes (#1928).
                    let committed_output_tool = run.output_tool_name().map(str::to_owned);
                    let mut prepared = match build_prepared_completion_request(
                        &runner.provider,
                        prompt.clone(),
                        &history,
                        runner.preamble.as_deref(),
                        &runner.static_context,
                        runner.temperature,
                        runner.max_tokens,
                        runner.additional_params.as_ref(),
                        runner.record_telemetry_content,
                        runner.tool_choice.as_ref(),
                        &runner.catalog,
                        &runner.tool_executor,
                        runner.output_schema.as_ref(),
                        &runner.output_mode,
                        committed_output_tool.as_deref(),
                        runner.output_tool_description.as_deref(),
                        runner.augment_output_preamble,
                        request_patch.as_ref(),
                    )
                    .await
                    {
                        Ok(prepared) => prepared,
                        Err(err) => {
                            store_error_usage(&runner, &run);
                            yield Err(err.into());
                            break 'outer;
                        }
                    };
                    run.set_output_tool_name(prepared.output_tool_name.clone());
                    let turn_tool_executor = prepared.tool_executor.clone();
                    if runner.record_telemetry_content {
                        let input_messages = prepared.request.messages_for_telemetry();
                        rig_core::telemetry::record_model_input(&chat_span, &input_messages, true);
                        prepared.request.record_telemetry_content = false;
                    }

                    let mut turn_stream = source.run_model_turn(
                        &runner,
                        turn,
                        &mut run,
                        prepared,
                        chat_span,
                        &agent_span,
                        prompt,
                    );
                    let mut turn_error = None;
                    while let Some(item) = turn_stream.next().await {
                        match item {
                            Ok(item) => yield Ok(DriveItem::Item(item)),
                            Err(err) => {
                                turn_error = Some(err);
                                break;
                            }
                        }
                    }
                    drop(turn_stream);
                    if let Some(err) = turn_error {
                        store_error_usage(&runner, &run);
                        yield Err(err);
                        break 'outer;
                    }
                    pending_tool_executor = Some(turn_tool_executor);
                }
                AgentRunStep::CallTools { calls } => {
                    let Some(tool_executor) = pending_tool_executor.take() else {
                        store_error_usage(&runner, &run);
                        yield Err(StreamingError::Completion(CompletionError::ResponseError(
                            "agent requested tool execution without prepared tool records"
                                .to_string(),
                        )));
                        break 'outer;
                    };
                    let mut tool_stream = source.run_tool_calls(
                        &runner,
                        &mut run,
                        calls,
                        tool_executor,
                    );
                    let mut tool_error = None;
                    while let Some(item) = tool_stream.next().await {
                        match item {
                            Ok(item) => yield Ok(DriveItem::Item(item)),
                            Err(err) => {
                                tool_error = Some(err);
                                break;
                            }
                        }
                    }
                    drop(tool_stream);
                    if let Some(err) = tool_error {
                        store_error_usage(&runner, &run);
                        yield Err(err);
                        break 'outer;
                    }
                }
                AgentRunStep::Done(response) => {
                    // Run-completion marker, unifying the blocking and streaming
                    // drivers' run-finished logs into one shared event.
                    tracing::info!(
                        turn = run.turn(),
                        max_turns = runner.max_turns,
                        "Agent run finished"
                    );
                    source.record_run_level_telemetry(&agent_span, &response, created_agent_span);
                    // Build the final item only when the surface forwards it
                    // (streaming). The blocking fold discards it, so its source
                    // returns `None` and the extra full-response clone is skipped.
                    if let Some(final_item) = source.final_item(&response) {
                        yield Ok(DriveItem::Item(final_item));
                    }
                    yield Ok(DriveItem::Done(Box::new(response)));
                    break 'outer;
                }
            }
        }
    }
}

/// Execute a turn's tool calls **atomically per batch**, shared by both surfaces.
///
/// The batch commits and surfaces all-or-nothing:
///
/// - The model tool-call events ([`StreamedAssistantContent::ToolCall`]) are
///   emitted up front — they report what the model emitted at turn commit.
/// - Every tool then runs (sequentially at `tool_concurrency <= 1`, else
///   concurrently bounded by it), with outcomes **collected, not surfaced**.
/// - On the first hook termination / fail-closed error the batch fails fast: no
///   new tool starts, not-yet-started concurrent siblings are dropped,
///   already-started ones are drained, and the deterministic lowest call-index
///   error is surfaced with **no** successful [`ToolExecutionCommitted`] /
///   [`StreamUserItem`](MultiTurnStreamItem::StreamUserItem) items and **no**
///   history commit.
/// - Only if the whole batch settles successfully are the per-tool
///   [`ToolExecutionCommitted`](MultiTurnStreamItem::ToolExecutionCommitted) + result
///   items surfaced (in call order, only for tools whose body actually ran) and
///   the results committed to run history.
///
/// When `forward_items` is `false` (the blocking fold) no stream items are built,
/// but the collect/commit and fail-fast behavior is identical, so `run()` and
/// `stream()` return the same terminal reason. `chain_tool_span` lets the
/// blocking surface chain spans into its linear `follows_from` sequence.
pub(crate) fn drive_tool_calls<'a, F>(
    runner: &'a AgentRunner,
    run: &'a mut AgentRun,
    calls: Vec<PendingToolCall>,
    tool_executor: Arc<ToolExecutor>,
    chain_tool_span: F,
    forward_items: bool,
) -> DriveStream<'a>
where
    F: Fn(tracing::Span) -> tracing::Span + WasmCompatSend + 'a,
{
    // Per-call working state: a stable internal_call_id and the execute span,
    // paired with the model's tool call. `span` is `Span::none()` for a
    // preresolved (invalid-recovery) call, which never executes.
    struct PreparedToolCall {
        tool_call: rig_core::message::ToolCall,
        preresolved_result: Option<UserContent>,
        internal_call_id: String,
        span: tracing::Span,
    }
    // How a settled tool call is surfaced on the stream once the batch succeeds:
    //   - `Executed`: `ToolExecutionCommitted` (with the effective, hook-rewritten
    //     call) + the `ToolResult`.
    //   - `Skipped`: the `ToolResult` only (a `ToolCall` hook returned `Skip`, so
    //     nothing ran — no execution commit — but the model still sees the result).
    //   - `Preresolved`: neither (an invalid-recovery result, already surfaced
    //     during the model turn); committed to history only.
    enum ToolSurface {
        // Boxed to keep this enum small next to the empty `Skipped`/`Preresolved`.
        Executed(Box<rig_core::message::ToolCall>),
        Skipped,
        Preresolved,
    }
    // A collected tool outcome, held (not surfaced or committed) until the whole
    // batch settles.
    struct CollectedToolResult {
        content: UserContent,
        internal_call_id: String,
        surface: ToolSurface,
    }

    Box::pin(async_stream::stream! {
        let full_history_for_errors = run.full_history();
        let call_count = calls.len();

        // Assign each call a stable internal_call_id and, for calls that will
        // actually execute, an execute span. Emit the MODEL tool-call events now,
        // right after the turn committed: these report what the model emitted and
        // are *not* execution-lifecycle events. A preresolved call emits no model
        // tool-call event (its synthetic result was already surfaced during the
        // model turn) and gets no execute span.
        let mut prepared: Vec<PreparedToolCall> = Vec::with_capacity(call_count);
        for pending in calls {
            let internal_call_id = pending.internal_call_id.unwrap_or_else(rig_core::id::generate);
            let (span, preresolved_result) = match pending.preresolved_result {
                Some(result) => (tracing::Span::none(), Some(result)),
                None => {
                    if forward_items {
                        yield Ok(MultiTurnStreamItem::stream_item(
                            StreamedAssistantContent::ToolCall {
                                tool_call: pending.tool_call.clone(),
                                internal_call_id: internal_call_id.clone(),
                            },
                        ));
                    }
                    (chain_tool_span(new_execute_tool_span()), None)
                }
            };
            prepared.push(PreparedToolCall {
                tool_call: pending.tool_call,
                preresolved_result,
                internal_call_id,
                span,
            });
        }

        // Run all tools, COLLECTING outcomes in call order — nothing is surfaced
        // or committed until the whole batch settles (atomic per-batch). On the
        // first hook termination / fail-closed error we stop starting new tools;
        // already-started ones are drained; the lowest call-index error wins; and
        // no successful result is surfaced or committed.
        let mut collected: Vec<Option<CollectedToolResult>> =
            (0..call_count).map(|_| None).collect();
        let mut first_error: Option<(usize, PromptError)> = None;

        if runner.concurrency <= 1 {
            // Sequential: run in call order, fail-fast on the first terminating
            // error so the remaining tools never start.
            for (index, call) in prepared.into_iter().enumerate() {
                let PreparedToolCall { tool_call, preresolved_result, internal_call_id, span } = call;
                if let Some(result) = preresolved_result {
                    if let Some(slot) = collected.get_mut(index) {
                        *slot = Some(CollectedToolResult {
                            content: result,
                            internal_call_id,
                            surface: ToolSurface::Preresolved,
                        });
                    }
                    continue;
                }
                let outcome = run_single_tool(
                    runner,
                    &tool_executor,
                    &tool_call,
                    &internal_call_id,
                    &full_history_for_errors,
                )
                .instrument(span)
                .await;
                match outcome {
                    Ok(outcome) => {
                        let surface = match outcome.execution {
                            ToolExecution::Executed(effective) => ToolSurface::Executed(effective),
                            ToolExecution::Skipped => ToolSurface::Skipped,
                        };
                        if let Some(slot) = collected.get_mut(index) {
                            *slot = Some(CollectedToolResult {
                                content: outcome.content,
                                internal_call_id,
                                surface,
                            });
                        }
                    }
                    Err(err) => {
                        first_error = Some((index, err));
                        break;
                    }
                }
            }
        } else {
            // Concurrent: bounded by `tool_concurrency`. A shared `terminating`
            // flag makes a not-yet-started sibling skip (its side effect never
            // runs) once any sibling terminates — avoiding the Semantic-Kernel
            // fail-open — while already-in-flight siblings are drained so the
            // lowest call-index terminator wins and no task is left detached.
            let terminating = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let unordered = stream::iter(prepared.into_iter().enumerate())
                .map(|(index, call)| {
                    let PreparedToolCall { tool_call, preresolved_result, internal_call_id, span } = call;
                    let tool_executor = &tool_executor;
                    let full_history_for_errors = &full_history_for_errors;
                    let terminating = terminating.clone();
                    async move {
                        if let Some(result) = preresolved_result {
                            return (
                                index,
                                Some(Ok(CollectedToolResult {
                                    content: result,
                                    internal_call_id,
                                    surface: ToolSurface::Preresolved,
                                })),
                            );
                        }
                        // `None` marks a dropped (never-started) sibling.
                        if terminating.load(std::sync::atomic::Ordering::SeqCst) {
                            return (index, None);
                        }
                        let outcome = run_single_tool(
                            runner,
                            tool_executor,
                            &tool_call,
                            &internal_call_id,
                            full_history_for_errors,
                        )
                        .await;
                        let mapped = outcome.map(|o| {
                            let surface = match o.execution {
                                ToolExecution::Executed(effective) => {
                                    ToolSurface::Executed(effective)
                                }
                                ToolExecution::Skipped => ToolSurface::Skipped,
                            };
                            CollectedToolResult {
                                content: o.content,
                                internal_call_id,
                                surface,
                            }
                        });
                        (index, Some(mapped))
                    }
                    .instrument(span)
                })
                .buffer_unordered(runner.concurrency);
            futures::pin_mut!(unordered);

            while let Some((index, outcome)) = unordered.next().await {
                // A dropped sibling records nothing.
                let result = match outcome {
                    Some(result) => result,
                    None => continue,
                };
                match result {
                    Ok(collected_result) => {
                        if let Some(slot) = collected.get_mut(index) {
                            *slot = Some(collected_result);
                        }
                    }
                    Err(err) => {
                        // Fail-fast: stop starting new siblings; keep draining
                        // in-flight ones so the lowest call-index terminator wins.
                        terminating.store(true, std::sync::atomic::Ordering::SeqCst);
                        if first_error.as_ref().is_none_or(|(i, _)| index < *i) {
                            first_error = Some((index, err));
                        }
                    }
                }
            }
        }

        // Settle. On termination: surface only the deterministic error — no
        // execution commit, no result, no history commit (all-or-nothing).
        if let Some((_, err)) = first_error {
            yield Err(StreamingError::Prompt(Box::new(err)));
            return;
        }

        // Success: prepare each call's stream items and results in call order,
        // commit the results, then surface the buffered items. An executed call
        // surfaces `ToolExecutionCommitted`
        // (with the effective, hook-rewritten call) then its `ToolResult`; a
        // hook-skipped call surfaces its `ToolResult` only (nothing ran); a
        // preresolved call surfaces nothing (already surfaced during the model
        // turn) but is still committed. Every non-dropped slot is filled; a
        // dropped slot only occurs after a termination, handled above.
        let mut committed: Vec<UserContent> = Vec::with_capacity(call_count);
        let mut surface_items: Vec<MultiTurnStreamItem> =
            Vec::with_capacity(call_count.saturating_mul(2));
        for slot in collected {
            let CollectedToolResult { content, internal_call_id, surface } = match slot {
                Some(collected_result) => collected_result,
                None => {
                    yield Err(StreamingError::Prompt(Box::new(PromptError::CompletionError(
                        CompletionError::ResponseError(
                            "tool execution finished without producing every result".to_string(),
                        ),
                    ))));
                    return;
                }
            };
            if forward_items {
                // An executed call also surfaces its execution commit; a skipped
                // call surfaces only its result; a preresolved call surfaces
                // nothing here.
                let surface_result = match surface {
                    ToolSurface::Executed(tool_call) => {
                        surface_items.push(MultiTurnStreamItem::ToolExecutionCommitted {
                            tool_call: *tool_call,
                            internal_call_id: internal_call_id.clone(),
                        });
                        true
                    }
                    ToolSurface::Skipped => true,
                    ToolSurface::Preresolved => false,
                };
                if surface_result
                    && let UserContent::ToolResult(tool_result) = &content
                {
                    surface_items.push(MultiTurnStreamItem::StreamUserItem(
                        StreamedUserContent::ToolResult {
                            tool_result: tool_result.clone(),
                            internal_call_id,
                        },
                    ));
                }
            }
            committed.push(content);
        }

        if let Err(err) = run.tool_results(committed) {
            yield Err(Box::new(err).into());
            return;
        }

        for item in surface_items {
            yield Ok(item);
        }
    })
}

/// [`TurnSource`] for the streaming surface: each turn opens a provider stream,
/// drives a [`StreamedTurnAssembler`], and yields assistant/tool deltas.
pub(crate) struct StreamingTurnSource {
    /// The raw provider choice of the most recent turn; the final response
    /// surfaces it as-is, even when canonical reordering was recorded in history.
    last_final_choice: OneOrMany<AssistantContent>,
    last_message_id: Option<String>,
    /// Resolved agent name, kept only for the empty-turn diagnostic warning.
    agent_name: String,
    /// Whether we created the agent span (vs. adopting a caller's ambient span);
    /// gates recording `gen_ai.completion` onto it, matching the blocking source
    /// so neither surface pollutes a caller-supplied span.
    created_agent_span: bool,
    /// Whether sensitive run-level prompt and completion content may be recorded.
    record_telemetry_content: bool,
    /// Hot-path interest gate, computed once: skip building/dispatching the
    /// high-frequency delta events when no hook entry opted into them.
    observes_deltas: bool,
    /// Whether any hook is present — gates building the (history-cloning)
    /// invalid-tool diagnostic context.
    has_hooks: bool,
}

impl StreamingTurnSource {
    pub(crate) fn new(
        hooks: &Hooks,
        agent_name: String,
        created_agent_span: bool,
        record_telemetry_content: bool,
    ) -> Self {
        Self {
            last_final_choice: OneOrMany::one(AssistantContent::text("")),
            last_message_id: None,
            agent_name,
            created_agent_span,
            record_telemetry_content,
            observes_deltas: hooks.observes_deltas(),
            has_hooks: !hooks.is_empty(),
        }
    }
}

impl StreamingTurnSource {
    fn open_chat_span(
        &self,
        runner: &AgentRunner,
        effective_preamble: Option<&str>,
    ) -> tracing::Span {
        build_chat_span!(runner, effective_preamble, "chat_streaming", "chat")
    }

    #[allow(clippy::too_many_arguments)]
    fn run_model_turn<'a>(
        &'a mut self,
        runner: &'a AgentRunner,
        turn: usize,
        run: &'a mut AgentRun,
        prepared: PreparedCompletionRequest,
        chat_span: tracing::Span,
        agent_span: &'a tracing::Span,
        current_prompt: Message,
    ) -> DriveStream<'a> {
        Box::pin(async_stream::stream! {
            let mut stream = match crate::provider::open_stream(
                &runner.provider,
                &runner.rt,
                prepared.request.clone(),
            )
            .instrument(chat_span.clone())
            .await
            {
                Ok(stream) => stream,
                Err(err) => {
                    yield Err(err.into());
                    return;
                }
            };
            // Captured from each completion-call emission so the normalized
            // `ModelTurnFinished` event carries the turn's usage.
            let mut last_usage = crate::completion::Usage::new();

            let mut assembler = StreamedTurnAssembler::new(
                prepared.executable_tool_names.clone(),
                prepared.allowed_tool_names.clone(),
            );
            let mut completion_call_emitted = false;
            let mut turn_abandoned = false;
            let mut provider_final_seen = false;
            let mut pending_final = None;
            // Mirrors the blocking driver's `response_hook_suppressed`: a turn
            // whose invalid tool call was repaired is a recovered turn, so its
            // response-finish hook is suppressed.
            let mut turn_recovered = false;

            // Emit the turn's single `CompletionCall` exactly once, recording its
            // usage onto the chat span and into the run. Defined here (not a free
            // fn) so it captures `completion_call_emitted`/`chat_span`/`run`; the
            // `yield` stays at each call site because `async_stream::stream!`
            // cannot see a `yield` produced inside a nested macro expansion.
            // Returns the item to yield (`Some` the first time, `None` after), or
            // the terminal error to surface.
            macro_rules! emit_completion_call {
                ($usage:expr) => {{
                    let usage = $usage;
                    last_usage = usage;
                    if !completion_call_emitted {
                        if usage.has_values() {
                            record_usage_on_span(&chat_span, usage);
                        }
                        match run.record_streamed_completion_call(usage) {
                            Ok(call) => {
                                completion_call_emitted = true;
                                Ok(Some(MultiTurnStreamItem::CompletionCall(call)))
                            }
                            Err(err) => Err(Box::new(err).into()),
                        }
                    } else {
                        Ok(None)
                    }
                }};
            }

            'turn: while let Some(item) = stream.next().await {
                let item = match item {
                    Ok(item) => item,
                    Err(err) => {
                        yield Err(err.into());
                        return;
                    }
                };
                if provider_final_seen {
                    yield Err(CompletionError::ResponseError(
                        "provider stream emitted visible assistant content after its final response"
                            .to_string(),
                    )
                    .into());
                    return;
                }
                let mut events: VecDeque<StreamedTurnEvent> = match assembler.ingest(&item) {
                    Ok(events) => events.into(),
                    Err(err) => {
                        yield Err(err.into());
                        return;
                    }
                };
                // At most one event per ingested item forwards the item itself;
                // moving it out of the slot avoids a clone per streamed delta.
                let mut item_slot = Some(item);
                while let Some(event) = events.pop_front() {
                    match event {
                        StreamedTurnEvent::EmitIngested => {
                            if self.observes_deltas
                                && let Some(StreamedAssistantContent::Text(text)) =
                                    item_slot.as_ref()
                                && let Some(reason) = observe_action(
                                    runner
                                        .hooks
                                        .dispatch_text_delta(
                                            turn,
                                            &text.text,
                                            assembler.aggregated_text(),
                                        )
                                        .await,
                                )
                            {
                                yield Err(StreamingError::Prompt(Box::new(
                                    run.cancel_error(reason),
                                )));
                                return;
                            }
                            if let Some(item) = item_slot.take() {
                                yield Ok(MultiTurnStreamItem::stream_item(item));
                            }
                        }
                        StreamedTurnEvent::EmitToolCallDelta {
                            id,
                            internal_call_id,
                            content,
                        } => {
                            if self.observes_deltas {
                                let (delta_name, delta_text) = match &content {
                                    ToolCallDeltaContent::Name(name) => (Some(name.as_str()), ""),
                                    ToolCallDeltaContent::Delta(delta) => (None, delta.as_str()),
                                };
                                if let Some(reason) = observe_action(
                                    runner
                                        .hooks
                                        .dispatch_tool_call_delta(
                                            turn,
                                            &id,
                                            &internal_call_id,
                                            delta_name,
                                            delta_text,
                                        )
                                        .await,
                                ) {
                                    yield Err(StreamingError::Prompt(Box::new(
                                        run.cancel_error(reason),
                                    )));
                                    return;
                                }
                            }

                            yield Ok(MultiTurnStreamItem::StreamAssistantItem(
                                StreamedAssistantContent::ToolCallDelta {
                                    id,
                                    internal_call_id,
                                    content,
                                },
                            ));
                        }
                        StreamedTurnEvent::Completed { usage, emit_final } => {
                            match emit_completion_call!(usage) {
                                Ok(Some(item)) => yield Ok(item),
                                Ok(None) => {}
                                Err(err) => {
                                    yield Err(err);
                                    return;
                                }
                            }
                            provider_final_seen = true;

                            if emit_final
                                && matches!(
                                    item_slot.as_ref(),
                                    Some(StreamedAssistantContent::Final(_))
                                )
                            {
                                pending_final = item_slot.take();
                            }
                        }
                        StreamedTurnEvent::InvalidToolCall(invalid) => {
                            let partial = assembler.partial_turn(stream.message_id.clone());
                            // Gated on `has_hooks`: building the diagnostic context
                            // clones the chat history, so an empty stack skips it and
                            // fails fast — identical to the blocking path.
                            let action = if self.has_hooks {
                                let context =
                                    run.streamed_invalid_tool_call_context(&partial, &invalid);
                                runner
                                    .hooks
                                    .dispatch_invalid_tool_call(&context)
                                    .await
                                    .unwrap_or_else(InvalidToolCallAction::fail)
                            } else {
                                InvalidToolCallAction::fail()
                            };

                            let resolution =
                                match run.resolve_streamed_invalid_tool_call(&partial, &invalid, action) {
                                    Ok(resolution) => resolution,
                                    Err(err) => {
                                        yield Err(Box::new(err).into());
                                        return;
                                    }
                                };

                            match resolution {
                                StreamedResolution::Repaired { .. } => {
                                    // Replayed deltas flow through the same event
                                    // handling above; the turn is now recovered, so
                                    // its response-finish hook is suppressed.
                                    turn_recovered = true;
                                    events.extend(assembler.resolve_pending_invalid(&resolution));
                                }
                                StreamedResolution::TurnAbandoned {
                                    ref skipped_tool_result,
                                } => {
                                    let skipped_tool_result = skipped_tool_result.clone();
                                    assembler.resolve_pending_invalid(&resolution);

                                    if let Some(err) = assembler.pending_delta_error() {
                                        yield Err(err.into());
                                        return;
                                    }
                                    let drained_usage = match drain_stream_usage(&mut stream).await {
                                        Ok(usage) => usage,
                                        Err(err) => {
                                            yield Err(err);
                                            return;
                                        }
                                    };
                                    match emit_completion_call!(drained_usage) {
                                        Ok(Some(item)) => yield Ok(item),
                                        Ok(None) => {}
                                        Err(err) => {
                                            yield Err(err);
                                            return;
                                        }
                                    }
                                    if let Some(tool_result) = skipped_tool_result {
                                        yield Ok(MultiTurnStreamItem::StreamUserItem(
                                            StreamedUserContent::ToolResult {
                                                tool_result,
                                                internal_call_id: invalid.internal_call_id.clone(),
                                            },
                                        ));
                                    }
                                    turn_abandoned = true;
                                    break 'turn;
                                }
                            }
                        }
                    }
                }
            }

            if turn_abandoned {
                return;
            }

            if let Some(err) = assembler.pending_delta_error() {
                yield Err(err.into());
                return;
            }

            // Final fallback: no usage was ever learned, so there is nothing to
            // record onto the span and this is the last read of the flag — kept
            // inline (not `emit_completion_call!`) so it doesn't emit a dead
            // `completion_call_emitted = true` write.
            if !completion_call_emitted {
                match run.record_streamed_completion_call(crate::completion::Usage::new()) {
                    Ok(call) => yield Ok(MultiTurnStreamItem::CompletionCall(call)),
                    Err(err) => {
                        yield Err(Box::new(err).into());
                        return;
                    }
                }
            }

            let final_turn_content = stream.choice.clone();
            let streamed_turn = assembler.finish(stream.message_id.clone(), &final_turn_content);
            if pending_final.is_some()
                && !turn_recovered
                && let Some(reason) = observe_action(
                    runner
                        .hooks
                        .dispatch_stream_response_finish(
                            turn,
                            &current_prompt,
                            &streamed_turn.choice,
                            last_usage,
                            streamed_turn.message_id.as_deref(),
                        )
                        .await,
                )
            {
                yield Err(StreamingError::Prompt(Box::new(run.cancel_error(reason))));
                return;
            }
            self.last_message_id = streamed_turn.message_id.clone();
            // The canonical assistant content: `finish` normalizes
            // reasoning/text/tool ordering, so this can differ from the raw
            // `stream.choice` aggregate. `ModelTurnFinished` — the normalized
            // per-turn event — carries this, matching what is recorded into run
            // history; the raw `stream.choice` is kept in `last_final_choice` for
            // the raw/final streaming behavior.
            let canonical_choice = streamed_turn.choice.clone();
            if let Err(err) = run.streamed_turn(streamed_turn) {
                yield Err(Box::new(err).into());
                return;
            }
            // Normalized per-turn event, fired once the turn is parked for
            // acceptance on the streaming surface — including tool-only /
            // reasoning-only turns that fire no `StreamResponseFinish`.
            // Suppressed for recovered turns, mirroring the blocking surface's
            // `Continue` arm.
            if !turn_recovered {
                let action = runner
                    .hooks
                    .dispatch_model_turn(turn, &canonical_choice, last_usage)
                    .await;
                match resolve_model_turn_action(run, action) {
                    Ok(ModelTurnDecision::Advance) => {}
                    Ok(ModelTurnDecision::Retried) => {
                        yield Ok(MultiTurnStreamItem::ModelTurnRetried { turn });
                        return;
                    }
                    Ok(ModelTurnDecision::Terminate(reason)) => {
                        // Before model-turn steering was added, Stop observed
                        // this already completed provider turn: its buffered
                        // final and content telemetry were visible before the
                        // cancellation. Preserve that behavior while Retry
                        // alone suppresses the provisional final.
                        if self.created_agent_span && self.record_telemetry_content {
                            agent_span.record(
                                "gen_ai.completion",
                                assistant_text_from_choice(&canonical_choice),
                            );
                        }
                        rig_core::telemetry::record_model_output(
                            &chat_span,
                            &canonical_choice,
                            runner.record_telemetry_content,
                        );
                        if let Some(item) = pending_final.take() {
                            yield Ok(MultiTurnStreamItem::stream_item(item));
                        }
                        yield Err(StreamingError::Prompt(Box::new(run.cancel_error(reason))));
                        return;
                    }
                    Err(err) => {
                        yield Err(StreamingError::Prompt(Box::new(err)));
                        return;
                    }
                }
            }

            // Only hook-accepted canonical output belongs in content telemetry.
            // Keep caller-owned spans untouched, matching the blocking source.
            if self.created_agent_span && self.record_telemetry_content {
                agent_span.record(
                    "gen_ai.completion",
                    assistant_text_from_choice(&canonical_choice),
                );
            }
            rig_core::telemetry::record_model_output(
                &chat_span,
                &canonical_choice,
                runner.record_telemetry_content,
            );

            if let Some(item) = pending_final {
                yield Ok(MultiTurnStreamItem::stream_item(item));
            }
            self.last_final_choice = final_turn_content;
        })
    }

    fn run_tool_calls<'a>(
        &'a self,
        runner: &'a AgentRunner,
        run: &'a mut AgentRun,
        calls: Vec<PendingToolCall>,
        tool_executor: Arc<ToolExecutor>,
    ) -> DriveStream<'a> {
        // The streaming surface chains nothing onto its tool spans, and forwards
        // the ToolCall/ToolResult items to the consumer.
        drive_tool_calls(runner, run, calls, tool_executor, |span| span, true)
    }

    fn record_run_level_telemetry(
        &self,
        agent_span: &tracing::Span,
        response: &PromptResponse,
        created_agent_span: bool,
    ) {
        if created_agent_span {
            record_usage_on_span(agent_span, response.usage);
        }
    }

    fn final_item(&self, response: &PromptResponse) -> Option<MultiTurnStreamItem> {
        // Tool output mode (#1928): when the finishing turn made the output-tool
        // call, surface the run's structured output as the final content.
        let final_choice = finalize_streamed_choice(&self.last_final_choice, &response.output)
            .unwrap_or_else(|| {
                if is_empty_assistant_turn(&self.last_final_choice) {
                    tracing::warn!(
                        agent_name = self.agent_name.as_str(),
                        message_id = ?self.last_message_id,
                        "Streaming turn completed without assistant text; final response will be empty"
                    );
                }
                self.last_final_choice.clone()
            });
        // Always surface the accumulated messages (parity with the blocking
        // `run()`), regardless of whether the caller supplied input history.
        let final_messages: Option<Vec<Message>> =
            Some(response.messages.clone().unwrap_or_default());
        Some(MultiTurnStreamItem::final_response_with_completion_calls(
            final_choice,
            response.usage,
            response.completion_calls.clone(),
            final_messages,
        ))
    }
}

impl AgentRunner {
    /// Drive the agent loop, streaming assistant content, tool activity, and a
    /// final response. Hooks fire at every observable point, including streamed
    /// text and tool-call deltas.
    ///
    /// Shares the drive loop, run construction, tool execution and fail-closed
    /// hook handling with the blocking [`run`](AgentRunner::run) via
    /// `drive_agent`, so the two behave identically apart from the streamed
    /// delta events.
    pub async fn stream(self) -> StreamingResult {
        let (agent_span, created_agent_span) = acquire_agent_span(
            self.agent_name_or_default(),
            self.preamble.as_deref(),
            self.record_telemetry_content,
        );

        if self.record_telemetry_content
            && let Some(text) = self.prompt.rag_text()
        {
            agent_span.record("gen_ai.prompt", text);
        }

        let run = self.build_run(None);
        let source = TurnSource::Streaming(StreamingTurnSource::new(
            &self.hooks,
            self.agent_name_or_default().to_string(),
            created_agent_span,
            self.record_telemetry_content,
        ));

        // The blocking surface folds this same engine; the streaming surface
        // forwards intermediate items (the final response item is the last one)
        // and ends on `Done`.
        let driver = drive_agent(self, source, run, agent_span.clone(), created_agent_span)
            .filter_map(|item| {
                std::future::ready(match item {
                    Ok(DriveItem::Item(item)) => Some(Ok(item)),
                    Ok(DriveItem::Done(_)) => None,
                    Err(err) => Some(Err(err)),
                })
            });

        Box::pin(driver.instrument(agent_span))
    }
}

impl IntoFuture for StreamingPromptRequest {
    type Output = StreamingResult; // what `.await` returns
    type IntoFuture = WasmBoxedFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        // Wrap send() in a future, because send() returns a stream immediately
        Box::pin(async move { self.send().await })
    }
}

/// Helper function to stream assistant-visible completion output to stdout.
///
/// This helper prints streamed assistant text and reasoning. Streaming metadata
/// events, such as `MultiTurnStreamItem::CompletionCall`, are not printed;
/// metadata is returned on the [`PromptResponse`] via accessors such as
/// [`PromptResponse::completion_calls`]. A model-turn retry prints a visible
/// boundary because text already written to stdout cannot be retracted.
pub async fn stream_to_stdout(
    stream: &mut StreamingResult,
) -> Result<PromptResponse, std::io::Error> {
    let mut final_res = PromptResponse::empty();
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                Text { text, .. },
            ))) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning(
                reasoning,
            ))) => {
                let reasoning = reasoning.display_text();
                print!("{reasoning}");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_res = res;
            }
            Ok(MultiTurnStreamItem::ModelTurnRetried { turn }) => {
                print!("\n[model turn {turn} rejected; retry requested]\nResponse: ");
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Err(err) => {
                eprintln!("Error: {err}");
            }
            _ => {}
        }
    }

    Ok(final_res)
}

#[cfg(test)]
#[allow(irrefutable_let_patterns, unreachable_patterns)]
mod migrated_tests {
    use crate::agent::{InvalidToolCallAction, InvalidToolCallContext, ObservationAction};

    use super::*;
    use crate::agent::prompt_request::mock_support::{
        MockCompletionModel, MockStreamEvent, MockTurn, agent_builder,
    };
    use crate::agent::prompt_request::{TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER, tool_result_output};
    use crate::agent::run::streamed::merge_reasoning_blocks;
    use crate::client::AgentClientExt;
    use crate::completion::{CompletionRequest, Prompt, PromptError, ToolDefinition, Usage};
    use crate::hooks::{HookDecision, HookEntry, HookEvent};
    use crate::streaming::{StreamingPrompt, ToolCallDeltaContent};
    use crate::test_utils::{MockAddTool, MockBarrierTool, MockSubtractTool, MockToolError};
    use crate::tool::PortableTool;
    use futures::{StreamExt, TryStreamExt};
    use rig_core::client::ProviderClient;
    use rig_core::message::{
        AssistantContent, DocumentSourceKind, ImageMediaType, Message, ReasoningContent,
        ToolChoice, ToolResultContent, UserContent,
    };
    use rig_core::providers::anthropic;
    use serde::Deserialize;
    use std::collections::{BTreeSet, HashMap};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tracing::field::{Field, Visit};
    use tracing::{Id, Subscriber};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::{Layer, Registry, registry::LookupSpan};

    fn reasoning(
        id: Option<&str>,
        content: impl IntoIterator<Item = ReasoningContent>,
    ) -> rig_core::message::Reasoning {
        let mut reasoning = rig_core::message::Reasoning::new("");
        reasoning.id = id.map(str::to_string);
        reasoning.content = content.into_iter().collect();
        reasoning
    }

    /// Named hook entry over a synchronous decision function.
    fn hook_entry(
        name: &str,
        decide: impl Fn(HookEvent) -> HookDecision + Send + Sync + 'static,
    ) -> HookEntry {
        HookEntry::new(name, move |event| {
            let decision = decide(event);
            Box::pin(async move { decision })
        })
    }

    fn stop_agent_streaming_before_completion() -> HookEntry {
        hook_entry("stop-before-completion", |event| {
            let HookEvent::BeforeModelCall { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::CompletionCall(crate::agent::CompletionCallAction::stop(
                "agent streaming stopped",
            ))
        })
    }

    #[tokio::test]
    async fn public_streaming_request_constructor_preserves_agent_hooks() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("should not run"),
            MockStreamEvent::final_response(Usage::new()),
        ]]);
        let agent = Arc::new(
            agent_builder(model.clone())
                .add_hook(stop_agent_streaming_before_completion())
                .build(),
        );

        let mut stream = StreamingPromptRequest::new(agent, "go").await;
        let error = stream
            .try_next()
            .await
            .expect_err("the configured agent hook should terminate the stream");

        assert!(matches!(
            error,
            StreamingError::Prompt(error)
                if matches!(*error, PromptError::PromptCancelled { ref reason, .. }
                    if reason == "agent streaming stopped")
        ));
        assert_eq!(model.request_count(), 0);
    }

    #[test]
    fn finalize_streamed_choice_surfaces_output_over_tool_call_and_prose() {
        use rig_core::message::{ToolCall, ToolFunction};

        let output_call = AssistantContent::ToolCall(ToolCall::new(
            "c1".to_string(),
            ToolFunction::new(
                "final_result".to_string(),
                serde_json::json!({"city": "Tokyo"}),
            ),
        ));

        // Prose + output-tool call (#1928): the streamed response text must be
        // the structured output, not the prose, with no orphan tool_use.
        let with_prose = OneOrMany::many(vec![
            AssistantContent::text("Sure, here is the weather:"),
            output_call.clone(),
        ])
        .expect("two items");
        let final_choice = finalize_streamed_choice(&with_prose, r#"{"city":"Tokyo"}"#)
            .expect("a turn with the output-tool call is finalized via it");
        assert_eq!(
            assistant_text_from_choice(&final_choice),
            r#"{"city":"Tokyo"}"#
        );
        assert!(
            !final_choice
                .iter()
                .any(|item| matches!(item, AssistantContent::ToolCall(_))),
            "no unanswered tool_use should remain in the final content"
        );

        // Output-tool call only.
        let only_call = OneOrMany::one(output_call);
        let final_choice = finalize_streamed_choice(&only_call, r#"{"city":"Tokyo"}"#)
            .expect("finalized via output tool");
        assert_eq!(
            assistant_text_from_choice(&final_choice),
            r#"{"city":"Tokyo"}"#
        );

        // A plain-text finalize (no tool call) is left to the caller.
        let text_only = OneOrMany::one(AssistantContent::text(r#"{"city":"Tokyo"}"#));
        assert!(finalize_streamed_choice(&text_only, r#"{"city":"Tokyo"}"#).is_none());
    }

    #[test]
    fn merge_reasoning_blocks_preserves_order_and_signatures() {
        let mut accumulated = Vec::new();
        let first = reasoning(
            Some("rs_1"),
            [ReasoningContent::Text {
                text: "step-1".to_string(),
                signature: Some("sig-1".to_string()),
            }],
        );
        let second = reasoning(
            Some("rs_1"),
            [
                ReasoningContent::Text {
                    text: "step-2".to_string(),
                    signature: Some("sig-2".to_string()),
                },
                ReasoningContent::Summary("summary".to_string()),
            ],
        );

        merge_reasoning_blocks(&mut accumulated, &first);
        merge_reasoning_blocks(&mut accumulated, &second);

        assert_eq!(accumulated.len(), 1);
        let merged = accumulated.first().expect("expected accumulated reasoning");
        assert_eq!(merged.id.as_deref(), Some("rs_1"));
        assert_eq!(merged.content.len(), 3);
        assert!(matches!(
            merged.content.first(),
            Some(ReasoningContent::Text { text, signature: Some(sig) })
                if text == "step-1" && sig == "sig-1"
        ));
        assert!(matches!(
            merged.content.get(1),
            Some(ReasoningContent::Text { text, signature: Some(sig) })
                if text == "step-2" && sig == "sig-2"
        ));
    }

    #[test]
    fn merge_reasoning_blocks_keeps_distinct_ids_as_separate_items() {
        let mut accumulated = vec![reasoning(
            Some("rs_a"),
            [ReasoningContent::Text {
                text: "step-1".to_string(),
                signature: None,
            }],
        )];
        let incoming = reasoning(
            Some("rs_b"),
            [ReasoningContent::Text {
                text: "step-2".to_string(),
                signature: None,
            }],
        );

        merge_reasoning_blocks(&mut accumulated, &incoming);
        assert_eq!(accumulated.len(), 2);
        assert_eq!(
            accumulated.first().and_then(|r| r.id.as_deref()),
            Some("rs_a")
        );
        assert_eq!(
            accumulated.get(1).and_then(|r| r.id.as_deref()),
            Some("rs_b")
        );
    }

    #[test]
    fn merge_reasoning_blocks_keeps_none_ids_separate_items() {
        let mut accumulated = vec![reasoning(
            None,
            [ReasoningContent::Text {
                text: "first".to_string(),
                signature: None,
            }],
        )];
        let incoming = reasoning(
            None,
            [ReasoningContent::Text {
                text: "second".to_string(),
                signature: None,
            }],
        );

        merge_reasoning_blocks(&mut accumulated, &incoming);
        assert_eq!(accumulated.len(), 2);
        assert!(accumulated.first().is_some_and(|reasoning| {
            reasoning.id.is_none()
                && matches!(
                    reasoning.content.first(),
                    Some(ReasoningContent::Text { text, .. }) if text == "first"
                )
        }));
        assert!(accumulated.get(1).is_some_and(|reasoning| {
            reasoning.id.is_none()
                && matches!(
                    reasoning.content.first(),
                    Some(ReasoningContent::Text { text, .. }) if text == "second"
                )
        }));
    }

    #[test]
    fn tool_result_output_preserves_multimodal_tool_output() {
        let instruction = serde_json::json!({
            "instruction": "Use the image part to answer."
        });
        let mut content = rig_core::OneOrMany::one(ToolResultContent::json(instruction.clone()));
        content.push(ToolResultContent::image_base64(
            "base64data==",
            Some(ImageMediaType::PNG),
            None,
        ));
        let user_content = tool_result_output(
            "tool_call_1".to_string(),
            Some("call_1".to_string()),
            crate::tool::ToolOutput::content(content),
        );

        let tool_result = match user_content {
            UserContent::ToolResult(tool_result) => tool_result,
            other => panic!("expected tool result content, got {other:?}"),
        };

        assert_eq!(tool_result.id, "tool_call_1");
        assert_eq!(tool_result.call_id.as_deref(), Some("call_1"));
        assert_eq!(tool_result.content.len(), 2);

        let mut items = tool_result.content.iter();
        match items.next() {
            Some(ToolResultContent::Json { value }) => {
                assert_eq!(value, &instruction);
            }
            other => panic!("expected structured JSON payload first, got {other:?}"),
        }

        match items.next() {
            Some(ToolResultContent::Image(image)) => {
                assert_eq!(image.media_type, Some(ImageMediaType::PNG));
                assert!(matches!(
                    image.data,
                    DocumentSourceKind::Base64(ref data) if data == "base64data=="
                ));
            }
            other => panic!("expected image payload second, got {other:?}"),
        }
    }

    fn validate_follow_up_tool_history(request: &CompletionRequest) -> Result<(), String> {
        let history = request.chat_history.iter().cloned().collect::<Vec<_>>();
        if history.len() != 3 {
            return Err(format!(
                "follow-up request should contain [original user prompt, assistant tool call, user tool result]: {history:?}"
            ));
        }

        if !matches!(
            history.first(),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::Text(text) if text.text == "do tool work"
                )
        ) {
            return Err(format!(
                "follow-up request should begin with the original user prompt: {history:?}"
            ));
        }

        if !matches!(
            history.get(1),
            Some(Message::Assistant { content, .. })
                if matches!(
                    content.first(),
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.call_id.as_deref() == Some("call_1")
                )
        ) {
            return Err(format!(
                "follow-up request is missing the assistant tool call in position 2: {history:?}"
            ));
        }

        if !matches!(
            history.get(2),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::ToolResult(tool_result)
                        if tool_result.id == "tool_call_1"
                            && tool_result.call_id.as_deref() == Some("call_1")
                )
        ) {
            return Err(format!(
                "follow-up request should end with the user tool result: {history:?}"
            ));
        }

        Ok(())
    }

    fn history_contains_tool_call(history: &[Message], tool_name: &str) -> bool {
        history.iter().any(|message| {
            matches!(
                message,
                Message::Assistant { content, .. }
                    if content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.function.name == tool_name
                    ))
            )
        })
    }

    fn history_contains_text(history: &[Message], expected: &str) -> bool {
        history.iter().any(|message| {
            matches!(
                message,
                Message::Assistant { content, .. }
                    if content.iter().any(|item| matches!(
                        item,
                        AssistantContent::Text(text) if text.text == expected
                    ))
            )
        })
    }

    fn assistant_reasoning_precedes_tool_call(
        history: &[Message],
        expected_reasoning: &str,
        tool_name: &str,
    ) -> bool {
        history.iter().any(|message| {
            let Message::Assistant { content, .. } = message else {
                return false;
            };

            let reasoning_index = content.iter().position(|item| {
                matches!(
                    item,
                    AssistantContent::Reasoning(reasoning)
                        if reasoning.content.iter().any(|content| matches!(
                            content,
                            ReasoningContent::Text { text, .. }
                                if text == expected_reasoning
                        ))
                )
            });
            let tool_index = content.iter().position(|item| {
                matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == tool_name
                )
            });

            matches!((reasoning_index, tool_index), (Some(reasoning), Some(tool)) if reasoning < tool)
        })
    }

    fn assistant_reasoning_precedes_text_and_tool_call(
        history: &[Message],
        expected_reasoning: &str,
        expected_text: &str,
        tool_name: &str,
    ) -> bool {
        history.iter().any(|message| {
            let Message::Assistant { content, .. } = message else {
                return false;
            };

            let reasoning_index = content.iter().position(|item| {
                matches!(
                    item,
                    AssistantContent::Reasoning(reasoning)
                        if reasoning.content.iter().any(|content| matches!(
                            content,
                            ReasoningContent::Text { text, .. }
                                if text == expected_reasoning
                        ))
                )
            });
            let text_index = content.iter().position(|item| {
                matches!(
                    item,
                    AssistantContent::Text(text) if text.text == expected_text
                )
            });
            let tool_index = content.iter().position(|item| {
                matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == tool_name
                )
            });

            matches!(
                (reasoning_index, text_index, tool_index),
                (Some(reasoning), Some(text), Some(tool))
                    if reasoning < text && text < tool
            )
        })
    }

    /// Fails the test if the driver ever reaches a tool-call delta, a tool-call
    /// decision point, or the stream-finish observation for an invalid turn.
    /// Registered with `observing_deltas` so the delta assertion is real.
    fn panic_on_unknown_tool_entry() -> HookEntry {
        hook_entry("panic-on-unknown-tool", |event| match event {
            HookEvent::ToolCallDelta { .. } => {
                panic!("unknown tool call delta should fail before delta hooks run")
            }
            HookEvent::ToolCall { .. } => {
                panic!("unknown tool call should fail before tool hooks run")
            }
            HookEvent::StreamResponseFinish { .. } => {
                panic!("unknown tool call should fail before stream finish hooks run")
            }
            _ => HookDecision::Continue,
        })
        .observing_deltas()
    }

    #[derive(Clone)]
    struct CountingAddTool {
        calls: Arc<AtomicU32>,
    }

    #[derive(Clone)]
    struct CountingSubtractTool {
        calls: Arc<AtomicU32>,
    }

    #[derive(Deserialize)]
    struct CountingOperationArgs {
        x: i32,
        y: i32,
    }

    fn arithmetic_tool_definition(name: &str, description: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: description.to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first operand"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second operand"
                    }
                },
                "required": ["x", "y"],
            }),
        }
    }

    impl PortableTool for CountingAddTool {
        const NAME: &'static str = "add";
        type Error = MockToolError;
        type Args = CountingOperationArgs;
        type Output = i32;

        fn description(&self) -> String {
            "Add x and y together".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            arithmetic_tool_definition(Self::NAME, "Add x and y together").parameters
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(args.x + args.y)
        }
    }

    impl PortableTool for CountingSubtractTool {
        const NAME: &'static str = "subtract";
        type Error = MockToolError;
        type Args = CountingOperationArgs;
        type Output = i32;

        fn description(&self) -> String {
            "Subtract y from x".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            arithmetic_tool_definition(Self::NAME, "Subtract y from x").parameters
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(args.x - args.y)
        }
    }

    fn streaming_tool_then_text_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ])
    }

    fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cached_input_tokens: 0,
            cache_creation_input_tokens: 0,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
        }
    }

    #[tokio::test]
    async fn execution_commit_items_are_not_emitted_when_run_commit_fails() {
        let runner = agent_builder(MockCompletionModel::default())
            .build()
            .runner("go");
        let tool_executor = Arc::new(runner.tool_executor.clone());

        let mut run = AgentRun::new("go").max_turns(2);
        assert!(matches!(
            run.next_step().expect("initial model step"),
            AgentRunStep::CallModel { .. }
        ));

        let tool_name = "missing".to_string();
        let advertised = BTreeSet::from([tool_name.clone()]);
        let turn = crate::agent::run::ModelTurn::new(
            None,
            OneOrMany::one(AssistantContent::ToolCall(
                rig_core::message::ToolCall::new(
                    "expected_call".to_string(),
                    rig_core::message::ToolFunction::new(tool_name, serde_json::json!({})),
                ),
            )),
            Usage::new(),
            advertised.clone(),
            advertised,
        );
        assert!(matches!(
            run.model_response(turn)
                .expect("tool turn should be accepted"),
            crate::agent::run::ModelTurnOutcome::Continue { .. }
        ));

        let mut calls = match run.next_step().expect("tool step") {
            AgentRunStep::CallTools { calls } => calls,
            other => panic!("expected tool step, got {other:?}"),
        };
        // Corrupt only the driver's copy so execution settles successfully but
        // `AgentRun` rejects the result before any commit-labelled item escapes.
        calls[0].tool_call.id = "mismatched_call".to_string();

        // The runner carries its own (empty) `Hooks`; the driver no longer needs
        // an ambient hook context, so the mismatched-commit path is exercised
        // directly.
        let mut stream =
            drive_tool_calls(&runner, &mut run, calls, tool_executor, |span| span, true);

        let mut saw_commit = false;
        let mut saw_result = false;
        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::ToolExecutionCommitted { .. }) => saw_commit = true,
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    ..
                })) => saw_result = true,
                Err(_) => saw_error = true,
                _ => {}
            }
        }

        assert!(
            saw_error,
            "the mismatched result must fail run-state commit"
        );
        assert!(!saw_commit, "a failed run-state commit cannot be announced");
        assert!(!saw_result, "an uncommitted result cannot be surfaced");
    }

    #[derive(Clone, Debug, Default)]
    struct CapturedSpan {
        id: u64,
        name: String,
        parent_id: Option<u64>,
        fields: HashMap<String, u64>,
        string_fields: HashMap<String, String>,
        record_counts: HashMap<String, usize>,
    }

    #[derive(Clone, Default)]
    struct CapturedSpans(Arc<Mutex<Vec<CapturedSpan>>>);

    impl CapturedSpans {
        fn clear(&self) {
            if let Ok(mut spans) = self.0.lock() {
                spans.clear();
            }
        }

        fn insert(&self, id: &Id, name: &str, parent_id: Option<u64>) {
            let id = id.into_u64();
            if let Ok(mut spans) = self.0.lock() {
                spans.push(CapturedSpan {
                    id,
                    name: name.to_string(),
                    parent_id,
                    fields: HashMap::new(),
                    string_fields: HashMap::new(),
                    record_counts: HashMap::new(),
                });
            }
        }

        fn record(&self, id: &Id, fields: Vec<CapturedField>) {
            if let Ok(mut spans) = self.0.lock()
                && let Some(span) = spans.iter_mut().rev().find(|span| span.id == id.into_u64())
            {
                for field in fields {
                    match field {
                        CapturedField::Number(name, value) => {
                            *span.record_counts.entry(name.clone()).or_insert(0) += 1;
                            span.fields.insert(name, value);
                        }
                        CapturedField::Text(name, value) => {
                            *span.record_counts.entry(name.clone()).or_insert(0) += 1;
                            span.fields.insert(name.clone(), 0);
                            span.string_fields.insert(name, value);
                        }
                    }
                }
            }
        }

        fn record_strings(&self, id: &Id, fields: Vec<(String, String)>) {
            if let Ok(mut spans) = self.0.lock()
                && let Some(span) = spans.iter_mut().rev().find(|span| span.id == id.into_u64())
            {
                span.string_fields.extend(fields);
            }
        }

        fn snapshot(&self) -> Vec<CapturedSpan> {
            self.0.lock().map(|spans| spans.clone()).unwrap_or_default()
        }
    }

    struct SpanCaptureLayer {
        spans: CapturedSpans,
    }

    impl<S> Layer<S> for SpanCaptureLayer
    where
        S: Subscriber,
        S: for<'lookup> LookupSpan<'lookup>,
    {
        fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            let parent_id = attrs
                .parent()
                .map(Id::into_u64)
                .or_else(|| ctx.current_span().id().map(Id::into_u64));
            self.spans.insert(id, attrs.metadata().name(), parent_id);
            let mut string_fields = Vec::new();
            attrs.record(&mut SpanStringCaptureVisitor {
                fields: &mut string_fields,
            });
            self.spans.record_strings(id, string_fields);
        }

        fn on_record(&self, span: &Id, values: &tracing::span::Record<'_>, _ctx: Context<'_, S>) {
            let mut fields = Vec::new();
            values.record(&mut SpanFieldCaptureVisitor {
                fields: &mut fields,
            });
            self.spans.record(span, fields);
            let mut string_fields = Vec::new();
            values.record(&mut SpanStringCaptureVisitor {
                fields: &mut string_fields,
            });
            self.spans.record_strings(span, string_fields);
        }
    }

    enum CapturedField {
        Number(String, u64),
        Text(String, String),
    }

    struct SpanFieldCaptureVisitor<'a> {
        fields: &'a mut Vec<CapturedField>,
    }

    struct SpanStringCaptureVisitor<'a> {
        fields: &'a mut Vec<(String, String)>,
    }

    impl Visit for SpanStringCaptureVisitor<'_> {
        fn record_str(&mut self, field: &Field, value: &str) {
            self.fields
                .push((field.name().to_string(), value.to_string()));
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.fields
                .push((field.name().to_string(), format!("{value:?}")));
        }
    }

    impl Visit for SpanFieldCaptureVisitor<'_> {
        fn record_u64(&mut self, field: &Field, value: u64) {
            self.fields
                .push(CapturedField::Number(field.name().to_string(), value));
        }

        // Capture the *presence* of non-numeric fields (e.g. `gen_ai.completion`)
        // with a placeholder value so tests can assert whether they were recorded.
        fn record_str(&mut self, field: &Field, value: &str) {
            self.fields.push(CapturedField::Text(
                field.name().to_string(),
                value.to_string(),
            ));
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.fields.push(CapturedField::Text(
                field.name().to_string(),
                format!("{value:?}"),
            ));
        }
    }

    async fn assert_stream_usage_recorded_on_chat_spans(
        agent: crate::agent::Agent,
        prompt: &str,
        max_turns: usize,
        expected_usages: &[Usage],
    ) {
        // Scoped-subscriber tests must not run concurrently; the warm-up
        // below explains the callsite-interest hazard this guards against.
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let spans = CapturedSpans::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            spans: spans.clone(),
        });
        let _default = tracing::subscriber::set_default(subscriber);

        // Span callsites in the driver are shared with every other test in
        // this binary. The FIRST thread to hit a callsite caches its interest
        // from that thread's dispatcher (`Dispatchers::Rebuilder::JustOne`
        // consults `dispatcher::get_default`), so a parallel test without a
        // subscriber can permanently cache `Interest::never` for the very
        // spans this harness asserts on. Defend in two steps, both under the
        // isolation guard: (1) warm the whole driver path from THIS thread so
        // unregistered callsites first-register against this subscriber, then
        // (2) rebuild the interest cache to heal callsites a foreign thread
        // already poisoned.
        let warmup_model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("warmup"),
            MockStreamEvent::final_response(Usage::default()),
        ]]);
        let warmup_agent = agent_builder(warmup_model).build();
        let mut warmup_stream = warmup_agent.stream_prompt("warmup").max_turns(1).await;
        while let Some(item) = warmup_stream
            .try_next()
            .await
            .expect("warmup stream should not error")
        {
            if matches!(item, MultiTurnStreamItem::FinalResponse(_)) {
                break;
            }
        }
        tracing::callsite::rebuild_interest_cache();
        spans.clear();

        let empty_history: &[Message] = &[];
        // Declare the fields the guard protects so a regression (recording onto
        // a caller span) is actually observable, not silently a no-op.
        let outer_span = tracing::info_span!("outer", gen_ai.completion = tracing::field::Empty);

        async {
            let mut stream = agent
                .stream_prompt(prompt)
                .history(empty_history)
                .max_turns(max_turns)
                .await;

            while let Some(item) = stream.try_next().await.expect("stream should not error") {
                if matches!(item, MultiTurnStreamItem::FinalResponse(_)) {
                    break;
                }
            }
        }
        .instrument(outer_span)
        .await;

        let span_snapshot = spans.snapshot();
        let outer_span_id = span_snapshot
            .iter()
            .find(|span| span.name == "outer")
            .map(|span| span.id)
            .expect("outer span should be captured");
        let chat_spans = span_snapshot
            .iter()
            .filter(|span| span.name == "chat_streaming")
            .collect::<Vec<_>>();

        assert_eq!(chat_spans.len(), expected_usages.len());
        assert!(
            span_snapshot.iter().all(|span| span.name != "invoke_agent"),
            "outer span path should not create invoke_agent"
        );

        for (chat_span, expected_usage) in chat_spans.into_iter().zip(expected_usages) {
            assert_eq!(chat_span.parent_id, Some(outer_span_id));
            assert_eq!(
                chat_span
                    .string_fields
                    .get("gen_ai.operation.name")
                    .map(String::as_str),
                Some("chat")
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.input_tokens"),
                Some(&expected_usage.input_tokens)
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.output_tokens"),
                Some(&expected_usage.output_tokens)
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.cache_read.input_tokens"),
                Some(&expected_usage.cached_input_tokens)
            );
            assert_eq!(
                chat_span
                    .fields
                    .get("gen_ai.usage.cache_creation.input_tokens"),
                Some(&expected_usage.cache_creation_input_tokens)
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.tool_use_prompt_tokens"),
                Some(&expected_usage.tool_use_prompt_tokens)
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.reasoning_tokens"),
                Some(&expected_usage.reasoning_tokens)
            );
        }

        let outer_span = span_snapshot
            .iter()
            .find(|span| span.id == outer_span_id)
            .expect("outer span should be present");
        assert!(
            outer_span
                .fields
                .keys()
                .all(|field| !field.starts_with("gen_ai.usage.")),
            "usage should not be recorded onto the caller's outer span"
        );
        assert!(
            !outer_span.fields.contains_key("gen_ai.completion"),
            "gen_ai.completion should not be recorded onto the caller's outer span \
             (parity with the blocking driver)"
        );
    }

    async fn capture_stream_message_telemetry(
        record_telemetry_content: bool,
    ) -> (CapturedSpan, Vec<CompletionRequest>) {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let spans = CapturedSpans::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            spans: spans.clone(),
        });
        let _default = tracing::subscriber::set_default(subscriber);

        let warmup_model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("warmup"),
            MockStreamEvent::final_response(Usage::default()),
        ]]);
        let warmup_agent = agent_builder(warmup_model).build();
        let mut warmup_stream = warmup_agent.stream_prompt("warmup").max_turns(1).await;
        while let Some(item) = warmup_stream
            .try_next()
            .await
            .expect("warmup stream should not error")
        {
            if matches!(item, MultiTurnStreamItem::FinalResponse(_)) {
                break;
            }
        }
        tracing::callsite::rebuild_interest_cache();
        spans.clear();

        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("stream response secret"),
            MockStreamEvent::final_response(Usage::default()),
        ]]);
        let recorded_model = model.clone();
        let builder = agent_builder(model);
        let agent = if record_telemetry_content {
            builder
                .record_content_telemetry(true)
                .context("static stream context secret")
                .build()
        } else {
            builder.context("static stream context secret").build()
        };

        let mut stream = agent
            .stream_prompt("stream prompt secret")
            .max_turns(1)
            .await;
        while let Some(item) = stream.try_next().await.expect("stream should not error") {
            if matches!(item, MultiTurnStreamItem::FinalResponse(_)) {
                break;
            }
        }

        let span = spans
            .snapshot()
            .into_iter()
            .find(|span| span.name == "chat_streaming")
            .expect("chat_streaming span should be captured");
        (span, recorded_model.requests())
    }

    async fn capture_unary_message_telemetry(
        record_telemetry_content: bool,
    ) -> (CapturedSpan, CapturedSpan, Vec<CompletionRequest>) {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let spans = CapturedSpans::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            spans: spans.clone(),
        });
        let _default = tracing::subscriber::set_default(subscriber);

        let warmup_agent = agent_builder(MockCompletionModel::text("warmup")).build();
        warmup_agent
            .prompt("warmup")
            .await
            .expect("warmup prompt should not error");
        tracing::callsite::rebuild_interest_cache();
        spans.clear();

        let model = MockCompletionModel::text("blocking response secret");
        let recorded_model = model.clone();
        let builder = agent_builder(model).preamble("blocking system secret");
        let agent = if record_telemetry_content {
            builder.record_content_telemetry(true).build()
        } else {
            builder.build()
        };

        agent
            .prompt("blocking prompt secret")
            .await
            .expect("prompt should not error");

        let snapshot = spans.snapshot();
        let chat_span = snapshot
            .iter()
            .find(|span| span.name == "chat")
            .cloned()
            .expect("chat span should be captured");
        let agent_span = snapshot
            .into_iter()
            .find(|span| span.name == "invoke_agent")
            .expect("invoke_agent span should be captured");
        (chat_span, agent_span, recorded_model.requests())
    }

    #[tokio::test]
    async fn stream_prompt_message_telemetry_is_opt_in() {
        let (default_span, default_requests) = capture_stream_message_telemetry(false).await;
        assert!(
            !default_span.fields.contains_key("gen_ai.input.messages"),
            "default streaming prompt should not record input message contents"
        );
        assert!(
            !default_span.fields.contains_key("gen_ai.output.messages"),
            "default streaming prompt should not record output message contents"
        );

        assert_eq!(default_requests.len(), 1);
        // Migration note: the provider-request `record_telemetry_content` flag
        // is no longer observable — the closed provider dispatch cannot record
        // raw requests, so the "provider request flag stays cleared" assertion
        // was dropped (the request count above still pins one provider call).

        let (opt_in_span, opt_in_requests) = capture_stream_message_telemetry(true).await;
        let input = opt_in_span
            .string_fields
            .get("gen_ai.input.messages")
            .expect("opt-in should record input messages");
        assert!(input.contains("stream prompt secret"));
        assert!(input.contains("static stream context secret"));
        let output = opt_in_span
            .string_fields
            .get("gen_ai.output.messages")
            .expect("opt-in should record output messages");
        assert!(output.contains("stream response secret"));
        assert_eq!(
            opt_in_span
                .record_counts
                .get("gen_ai.input.messages")
                .copied(),
            Some(1),
            "agent-owned input message telemetry should be recorded once"
        );
        assert_eq!(
            opt_in_span
                .record_counts
                .get("gen_ai.output.messages")
                .copied(),
            Some(1),
            "agent-owned output message telemetry should be recorded once"
        );
        assert_eq!(opt_in_requests.len(), 1);
        // Migration note: the provider-request flag assertion was dropped (see
        // the default-path note above).
    }

    #[tokio::test]
    async fn unary_prompt_message_telemetry_records_accepted_output_when_opted_in() {
        let (default_span, default_agent_span, default_requests) =
            capture_unary_message_telemetry(false).await;
        assert!(
            !default_span.fields.contains_key("gen_ai.input.messages"),
            "default blocking prompt should not record input message contents"
        );
        assert!(
            !default_span.fields.contains_key("gen_ai.output.messages"),
            "default blocking prompt should not record output message contents"
        );
        assert!(
            !default_span
                .string_fields
                .contains_key("gen_ai.system_instructions"),
            "default blocking prompt should not record system instructions"
        );
        assert!(
            !default_agent_span
                .string_fields
                .contains_key("gen_ai.prompt")
        );
        assert!(
            !default_agent_span
                .string_fields
                .contains_key("gen_ai.completion")
        );
        assert_eq!(default_requests.len(), 1);
        // Migration note: the provider-request `record_telemetry_content` flag
        // is no longer observable through the closed provider dispatch; the
        // assertion was dropped (the request count still pins one provider call).

        let (opt_in_span, opt_in_agent_span, opt_in_requests) =
            capture_unary_message_telemetry(true).await;
        let input = opt_in_span
            .string_fields
            .get("gen_ai.input.messages")
            .expect("opt-in should record blocking input messages");
        assert!(input.contains("blocking prompt secret"));
        let output = opt_in_span
            .string_fields
            .get("gen_ai.output.messages")
            .expect("opt-in should record blocking output messages");
        assert!(output.contains("blocking response secret"));
        assert_eq!(
            opt_in_span
                .string_fields
                .get("gen_ai.system_instructions")
                .map(String::as_str),
            Some(r#"[{"type":"text","content":"blocking system secret"}]"#)
        );
        assert_eq!(
            opt_in_agent_span
                .string_fields
                .get("gen_ai.prompt")
                .map(String::as_str),
            Some("blocking prompt secret")
        );
        assert_eq!(
            opt_in_agent_span
                .string_fields
                .get("gen_ai.completion")
                .map(String::as_str),
            Some("blocking response secret")
        );
        assert_eq!(opt_in_requests.len(), 1);
        // Migration note: the provider-request flag assertion was dropped (see
        // the default-path note above).
    }

    async fn capture_tool_content_telemetry(record_telemetry_content: bool) -> CapturedSpan {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let spans = CapturedSpans::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            spans: spans.clone(),
        });
        let _default = tracing::subscriber::set_default(subscriber);

        let warmup = agent_builder(MockCompletionModel::from_turns([
            MockTurn::tool_call("warmup", "add", serde_json::json!({"x": 1, "y": 2})),
            MockTurn::text("done"),
        ]))
        .tool(MockAddTool)
        .build();
        warmup
            .runner("warmup")
            .max_turns(2)
            .run()
            .await
            .expect("warmup tool run should succeed");
        tracing::callsite::rebuild_interest_cache();
        spans.clear();

        let builder = agent_builder(MockCompletionModel::from_turns([
            MockTurn::tool_call(
                "secret-tool-call",
                "add",
                serde_json::json!({"x": 12345, "y": 67890}),
            ),
            MockTurn::text("done"),
        ]))
        .tool(MockAddTool);
        let agent = if record_telemetry_content {
            builder.record_content_telemetry(true).build()
        } else {
            builder.build()
        };
        agent
            .runner("use the tool")
            .max_turns(2)
            .run()
            .await
            .expect("tool run should succeed");

        spans
            .snapshot()
            .into_iter()
            .find(|span| span.name == "execute_tool")
            .expect("execute_tool span should be captured")
    }

    #[tokio::test]
    async fn tool_arguments_and_results_follow_content_telemetry_toggle() {
        let default_span = capture_tool_content_telemetry(false).await;
        assert!(
            !default_span
                .string_fields
                .contains_key("gen_ai.tool.call.arguments")
        );
        assert!(
            !default_span
                .string_fields
                .contains_key("gen_ai.tool.call.result")
        );
        assert_eq!(
            default_span
                .string_fields
                .get("gen_ai.tool.name")
                .map(String::as_str),
            Some("add"),
            "structural tool metadata should remain available"
        );

        let opt_in_span = capture_tool_content_telemetry(true).await;
        assert!(
            opt_in_span
                .string_fields
                .get("gen_ai.tool.call.arguments")
                .is_some_and(|args| args.contains("12345") && args.contains("67890"))
        );
        assert!(
            opt_in_span
                .string_fields
                .get("gen_ai.tool.call.result")
                .is_some_and(|result| result.contains("80235"))
        );
    }

    #[tokio::test]
    async fn streaming_rejected_message_telemetry_does_not_record_output() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let spans = CapturedSpans::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            spans: spans.clone(),
        });
        let _default = tracing::subscriber::set_default(subscriber);

        let warmup_model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("warmup"),
            MockStreamEvent::final_response(Usage::default()),
        ]]);
        let warmup_agent = agent_builder(warmup_model).build();
        let mut warmup_stream = warmup_agent.stream_prompt("warmup").max_turns(1).await;
        while let Some(item) = warmup_stream
            .try_next()
            .await
            .expect("warmup stream should not error")
        {
            if matches!(item, MultiTurnStreamItem::FinalResponse(_)) {
                break;
            }
        }
        tracing::callsite::rebuild_interest_cache();
        spans.clear();

        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("rejected stream output secret"),
            MockStreamEvent::tool_call(
                "tool_call_1",
                "default_api",
                serde_json::json!({"x": 2, "y": 3}),
            ),
            MockStreamEvent::final_response(Usage::default()),
        ]]);
        let agent = agent_builder(model).record_content_telemetry(true).build();

        let mut stream = agent
            .stream_prompt("stream rejection prompt")
            .max_turns(1)
            .await;
        let err = loop {
            match stream.try_next().await {
                Ok(Some(_)) => continue,
                Ok(None) => panic!("rejected stream should error"),
                Err(err) => break err,
            }
        };
        assert!(
            err.to_string().contains("default_api"),
            "expected invalid tool error, got {err}"
        );

        let chat_span = spans
            .snapshot()
            .into_iter()
            .find(|span| span.name == "chat_streaming")
            .expect("chat_streaming span should be captured");
        assert!(
            chat_span.fields.contains_key("gen_ai.input.messages"),
            "opt-in rejected stream should still record input messages"
        );
        assert!(
            !chat_span.fields.contains_key("gen_ai.output.messages"),
            "rejected streaming turn must not record output message contents"
        );
    }

    #[tokio::test]
    async fn unary_repaired_message_telemetry_records_canonical_output() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let spans = CapturedSpans::default();
        let subscriber = Registry::default().with(SpanCaptureLayer {
            spans: spans.clone(),
        });
        let _default = tracing::subscriber::set_default(subscriber);

        let warmup_agent = agent_builder(MockCompletionModel::text("warmup")).build();
        warmup_agent
            .prompt("warmup")
            .await
            .expect("warmup prompt should not error");
        tracing::callsite::rebuild_interest_cache();
        spans.clear();

        let model = MockCompletionModel::new([
            MockTurn::tool_call(
                "tool_call_1",
                "default_api",
                serde_json::json!({"x": 2, "y": 3}),
            ),
            MockTurn::text("done"),
        ]);
        let recorded_model = model.clone();
        let agent = agent_builder(model)
            .record_content_telemetry(true)
            .tool(MockAddTool)
            .build();

        let output = agent
            .prompt("repair tool call")
            .add_hook(repair_default_api_entry())
            .max_turns(3)
            .await
            .expect("repaired tool call should complete");
        assert_eq!(output, "done");

        let output_messages: Vec<String> = spans
            .snapshot()
            .into_iter()
            .filter(|span| span.name == "chat")
            .filter_map(|span| span.string_fields.get("gen_ai.output.messages").cloned())
            .collect();
        assert!(
            output_messages.iter().any(|output| output.contains("add")),
            "repaired accepted output should include canonical tool name: {output_messages:?}"
        );
        assert!(
            !output_messages
                .iter()
                .any(|output| output.contains("default_api")),
            "repaired output telemetry must not serialize stale raw tool name: {output_messages:?}"
        );

        let requests = recorded_model.requests();
        assert_eq!(requests.len(), 2);
        // Migration note: the provider-request `record_telemetry_content` flag
        // is no longer observable through the closed provider dispatch; the
        // "flags stay cleared" assertion was dropped.
    }

    #[test]
    fn completion_calls_stream_item_serializes_and_deserializes_expected_shape() {
        let item: MultiTurnStreamItem =
            MultiTurnStreamItem::CompletionCall(CompletionCall::new(2, usage(3, 4)));

        let value = serde_json::to_value(&item).expect("serialize completion call event");

        assert_eq!(
            value,
            serde_json::json!({
                "type": "completionCall",
                "call_index": 2,
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "total_tokens": 7,
                    "cached_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "tool_use_prompt_tokens": 0,
                    "reasoning_tokens": 0,
                }
            })
        );

        let item: MultiTurnStreamItem =
            serde_json::from_value(value).expect("deserialize completion call event");
        match item {
            MultiTurnStreamItem::CompletionCall(call_usage) => {
                assert_eq!(call_usage, CompletionCall::new(2, usage(3, 4)));
            }
            other => panic!("expected completion call event, got {other:?}"),
        }

        let item: MultiTurnStreamItem =
            MultiTurnStreamItem::CompletionCall(CompletionCall::new(3, Usage::new()));
        let value = serde_json::to_value(&item).expect("serialize missing usage event");

        // Unreported usage serializes as a plain zero-valued object (Usage's
        // documented sentinel for missing provider metrics).
        assert_eq!(
            value,
            serde_json::json!({
                "type": "completionCall",
                "call_index": 3,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cached_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "tool_use_prompt_tokens": 0,
                    "reasoning_tokens": 0,
                }
            })
        );

        // Stream items serialized before the Option encoding was dropped used
        // `"usage": null`; they must still deserialize.
        let legacy: MultiTurnStreamItem = serde_json::from_value(serde_json::json!({
            "type": "completionCall",
            "call_index": 3,
            "usage": null
        }))
        .expect("legacy null-usage event should deserialize");
        match legacy {
            MultiTurnStreamItem::CompletionCall(call) => {
                assert_eq!(call, CompletionCall::new(3, Usage::new()));
            }
            other => panic!("expected completion call event, got {other:?}"),
        }
    }

    #[test]
    fn final_response_serializes_completion_calls_with_missing_usage() {
        let item: MultiTurnStreamItem = MultiTurnStreamItem::final_response_with_completion_calls(
            OneOrMany::one(AssistantContent::text("done")),
            usage(3, 4),
            vec![
                CompletionCall::new(0, Usage::new()),
                CompletionCall::new(1, usage(3, 4)),
            ],
            None,
        );

        if let MultiTurnStreamItem::FinalResponse(response) = &item {
            assert_eq!(response.requests(), 2);
        }

        let value = serde_json::to_value(&item).expect("serialize final response");

        assert_eq!(
            value.get("completion_calls"),
            Some(&serde_json::json!([
                {
                    "call_index": 0,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cached_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "tool_use_prompt_tokens": 0,
                        "reasoning_tokens": 0,
                    }
                },
                {
                    "call_index": 1,
                    "usage": {
                        "input_tokens": 3,
                        "output_tokens": 4,
                        "total_tokens": 7,
                        "cached_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "tool_use_prompt_tokens": 0,
                        "reasoning_tokens": 0,
                    }
                }
            ]))
        );
    }

    fn streaming_text_then_final_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("hello"),
            MockStreamEvent::text(" world"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]])
    }

    fn citation_metadata() -> serde_json::Value {
        serde_json::json!({
            "citations": [{
                "type": "web_search_result_location",
                "cited_text": "Claude Shannon was born in 1916.",
                "url": "https://example.com/shannon",
                "title": "Claude Shannon",
                "encrypted_index": "encrypted-reference"
            }]
        })
    }

    fn streaming_cited_text_then_final_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text_start(Some(citation_metadata())),
            MockStreamEvent::text("cited "),
            MockStreamEvent::text_start(None),
            MockStreamEvent::text("answer"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]])
    }

    fn streaming_cited_text_then_tool_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text_start(Some(citation_metadata())),
                MockStreamEvent::text("I need a tool. "),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ])
    }

    fn streaming_final_only_model() -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([[
            MockStreamEvent::final_response_with_total_tokens(1),
        ]])
    }

    fn terminate_on_stream_finish_entry() -> HookEntry {
        hook_entry("terminate-on-stream-finish", |event| {
            let HookEvent::StreamResponseFinish { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::Observation(ObservationAction::stop("stop after completion call"))
        })
    }

    type RecordedToolCallDelta = (String, String, Option<String>, String);

    fn repair_default_api_entry() -> HookEntry {
        hook_entry("repair-default-api", |event| {
            let HookEvent::InvalidToolCall(context) = event else {
                return HookDecision::Continue;
            };
            assert_eq!(context.tool_name, "default_api");
            HookDecision::InvalidToolCall(InvalidToolCallAction::repair("add"))
        })
    }

    fn retry_default_api_entry() -> HookEntry {
        hook_entry("retry-default-api", |event| {
            let HookEvent::InvalidToolCall(context) = event else {
                return HookDecision::Continue;
            };
            assert_eq!(context.tool_name, "default_api");
            if let Some(args) = context.args.as_deref() {
                assert!(!args.is_empty());
            }
            HookDecision::InvalidToolCall(InvalidToolCallAction::retry("Use the add tool instead"))
        })
    }

    fn skip_default_api_entry() -> HookEntry {
        hook_entry("skip-default-api", |event| {
            let HookEvent::InvalidToolCall(context) = event else {
                return HookDecision::Continue;
            };
            assert_eq!(context.tool_name, "default_api");
            HookDecision::InvalidToolCall(InvalidToolCallAction::skip("default_api was skipped"))
        })
    }

    /// Records every invalid-tool-call context it is asked about, then fails
    /// (preserving the driver's fail-fast behavior).
    #[derive(Clone, Default)]
    struct RecordedInvalidToolCalls(Arc<Mutex<Vec<InvalidToolCallContext>>>);

    impl RecordedInvalidToolCalls {
        fn observed(&self) -> Vec<InvalidToolCallContext> {
            self.0
                .lock()
                .expect("invalid tool context records mutex was poisoned")
                .clone()
        }

        fn entry(&self) -> HookEntry {
            let contexts = self.0.clone();
            hook_entry("record-invalid-tool-call", move |event| {
                let HookEvent::InvalidToolCall(context) = event else {
                    return HookDecision::Continue;
                };
                contexts
                    .lock()
                    .expect("invalid tool context records mutex was poisoned")
                    .push(context);
                HookDecision::InvalidToolCall(InvalidToolCallAction::fail())
            })
        }
    }

    #[derive(Clone, Default)]
    struct RecordedToolCallDeltas(Arc<Mutex<Vec<RecordedToolCallDelta>>>);

    impl RecordedToolCallDeltas {
        fn observed(&self) -> Vec<RecordedToolCallDelta> {
            self.0
                .lock()
                .expect("tool call delta hook records mutex was poisoned")
                .clone()
        }

        /// Delta-observing entry: without `observing_deltas` the callback would
        /// never see a `ToolCallDelta`.
        fn entry(&self) -> HookEntry {
            self.entry_with(ObservationAction::continue_run())
        }

        /// As [`Self::entry`], but answers with `action` after recording (used
        /// to assert hook termination on the delta path).
        fn entry_with(&self, action: ObservationAction) -> HookEntry {
            let deltas = self.0.clone();
            hook_entry("record-tool-call-delta", move |event| {
                let HookEvent::ToolCallDelta {
                    tool_call_id,
                    internal_call_id,
                    tool_name,
                    delta,
                    ..
                } = event
                else {
                    return HookDecision::Continue;
                };
                deltas
                    .lock()
                    .expect("tool call delta hook records mutex was poisoned")
                    .push((tool_call_id, internal_call_id, tool_name, delta));
                HookDecision::Observation(action.clone())
            })
            .observing_deltas()
        }
    }

    #[derive(Clone, Default)]
    struct RecordedTextDeltas(Arc<Mutex<Vec<(String, String)>>>);

    impl RecordedTextDeltas {
        fn observed(&self) -> Vec<(String, String)> {
            self.0
                .lock()
                .expect("text delta hook records mutex was poisoned")
                .clone()
        }

        /// Delta-observing entry: without `observing_deltas` the callback would
        /// never see a `TextDelta`.
        fn entry(&self) -> HookEntry {
            let deltas = self.0.clone();
            hook_entry("record-text-delta", move |event| {
                let HookEvent::TextDelta {
                    delta, aggregated, ..
                } = event
                else {
                    return HookDecision::Continue;
                };
                deltas
                    .lock()
                    .expect("text delta hook records mutex was poisoned")
                    .push((delta, aggregated));
                HookDecision::Observation(ObservationAction::continue_run())
            })
            .observing_deltas()
        }
    }

    fn text_metadata(content: &OneOrMany<AssistantContent>) -> Option<&serde_json::Value> {
        content.iter().find_map(|item| match item {
            AssistantContent::Text(text) => text.additional_params.as_ref(),
            _ => None,
        })
    }

    #[tokio::test]
    async fn stream_prompt_continues_after_tool_call_turn() {
        let model = streaming_tool_then_text_model();
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();
        let empty_history: &[Message] = &[];

        let mut stream = agent
            .stream_prompt("do tool work")
            .history(empty_history)
            .max_turns(3)
            .await;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut saw_final_response = false;
        let mut final_text = String::new();
        let mut final_response_text = None;
        let mut final_history = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    ..
                })) => {
                    saw_tool_result = true;
                }
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => {
                    final_text.push_str(&text.text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    saw_final_response = true;
                    final_response_text = Some(res.output().to_owned());
                    final_history = res.messages().map(|history| history.to_vec());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert!(saw_tool_call);
        assert!(saw_tool_result);
        assert!(saw_final_response);
        assert_eq!(final_text, "done");
        assert_eq!(final_response_text.as_deref(), Some("done"));
        let history = final_history.expect("expected final response history");
        assert!(history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "done"
                ))
        )));
        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        assert!(validate_follow_up_tool_history(&requests[1]).is_ok());
    }

    /// `StreamingPromptRequest::tool_concurrency` reaches the runner: two
    /// barrier-synchronized tools in a streamed turn only finish if they run
    /// concurrently. At `tool_concurrency(2)` the stream completes; sequential
    /// execution would block on the first tool forever, so the timeout asserts
    /// the public builder actually enables concurrency on the streaming path.
    #[tokio::test]
    async fn streaming_prompt_request_tool_concurrency_runs_tools_concurrently() {
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("b1", "barrier_tool", serde_json::json!({})),
                MockStreamEvent::tool_call("b2", "barrier_tool", serde_json::json!({})),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ]);
        let agent = agent_builder(model)
            .tool(MockBarrierTool::new(barrier))
            .build();

        let drive = async {
            let mut stream = agent
                .stream_prompt("hit the barrier twice")
                .max_turns(3)
                .tool_concurrency(2)
                .await;
            while let Some(item) = stream.next().await {
                item.unwrap_or_else(|err| panic!("unexpected streaming error: {err:?}"));
            }
        };

        tokio::time::timeout(Duration::from_secs(5), drive)
            .await
            .expect("streamed tools must run concurrently, not deadlock at the barrier");
    }

    #[tokio::test]
    async fn unknown_tool_call_fails_before_streaming_second_request() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 1, "y": 2}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(panic_on_unknown_tool_entry())
            .max_turns(3)
            .await;
        let mut saw_tool_call = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_tool_call);
        let error = error.expect("unknown model-emitted tool should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_can_repair_streaming_tool_name() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 2, "y": 3}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(repair_default_api_entry())
            .max_turns(3)
            .history(Vec::<Message>::new())
            .await;
        let mut saw_repaired_tool_call = false;
        let mut saw_tool_result = false;
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { tool_call, .. },
                )) => {
                    assert_eq!(tool_call.function.name, "add");
                    saw_repaired_tool_call = true;
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    tool_result,
                    ..
                })) => {
                    assert!(tool_result.content.iter().any(|content| {
                        matches!(
                            content,
                            ToolResultContent::Json { value }
                                if value == &serde_json::json!(5)
                        )
                    }));
                    saw_tool_result = true;
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.output().to_string());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert!(saw_repaired_tool_call);
        assert!(saw_tool_result);
        assert_eq!(final_response_text.as_deref(), Some("done"));
        assert_eq!(recorded.request_count(), 2);
    }

    #[tokio::test]
    async fn invalid_tool_call_context_uses_completed_streaming_tool_call_provider_id() {
        let invalid_hook = RecordedInvalidToolCalls::default();
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 2, "y": 3}),
                )
                .with_call_id("provider_call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(invalid_hook.entry())
            .max_turns(3)
            .await;
        let mut error = None;

        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                error = Some(err);
                break;
            }
        }

        assert!(error.is_some(), "invalid tool should fail");
        assert_eq!(recorded.request_count(), 1);
        let contexts = invalid_hook.observed();
        assert_eq!(contexts.len(), 1);
        let context = &contexts[0];
        assert_eq!(context.tool_name, "default_api");
        assert_eq!(context.tool_call_id.as_deref(), Some("tool_call_1"));
        assert!(context.internal_call_id.is_some());
        assert!(context.is_streaming);
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_skip_emits_streaming_tool_result() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 2, "y": 3}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("continued"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(skip_default_api_entry())
            .max_turns(3)
            .history(Vec::<Message>::new())
            .await;
        let mut skipped_tool_result = None;
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    tool_result,
                    internal_call_id,
                })) => {
                    assert!(!internal_call_id.is_empty());
                    skipped_tool_result = Some(tool_result);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.output().to_string());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let skipped_tool_result =
            skipped_tool_result.expect("skip recovery should emit a synthetic tool result");
        assert_eq!(skipped_tool_result.id, "tool_call_1");
        assert_eq!(skipped_tool_result.call_id.as_deref(), Some("call_1"));
        assert!(skipped_tool_result.content.iter().any(|content| matches!(
            content,
            ToolResultContent::Text(text) if text.text == "default_api was skipped"
        )));
        assert_eq!(final_response_text.as_deref(), Some("continued"));
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let follow_up_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert!(matches!(
            follow_up_history.get(2),
            Some(Message::User { content })
                if content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.id == "tool_call_1"
                            && result.content.iter().any(|content| matches!(
                                content,
                                ToolResultContent::Text(text)
                                    if text.text == "default_api was skipped"
                            ))
                ))
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_retries_mixed_streaming_turn_without_executing_valid_call() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 2, "y": 3}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::tool_call(
                    "tool_call_2",
                    "default_api",
                    serde_json::json!({"x": 4, "y": 5}),
                )
                .with_call_id("call_2"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("retried"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(retry_default_api_entry())
            .max_turns(3)
            .history(Vec::<Message>::new())
            .max_invalid_tool_call_retries(1)
            .await;
        let mut completion_call_events = Vec::new();
        let mut final_response_text = None;
        let mut final_response_usage = Usage::new();
        let mut final_completion_calls = Vec::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(completion_call)) => {
                    completion_call_events.push(completion_call);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.output().to_string());
                    final_response_usage = response.usage();
                    final_completion_calls = response.completion_calls().to_vec();
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(final_response_text.as_deref(), Some("retried"));
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let mut first_usage = Usage::new();
        first_usage.total_tokens = 4;
        let mut second_usage = Usage::new();
        second_usage.total_tokens = 6;
        let expected_completion_calls = vec![
            CompletionCall::new(0, first_usage),
            CompletionCall::new(1, second_usage),
        ];
        assert_eq!(completion_call_events, expected_completion_calls);
        assert_eq!(final_completion_calls, expected_completion_calls);
        assert_eq!(final_response_usage.total_tokens, 10);

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let retry_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert_eq!(retry_history.len(), 3);
        assert!(matches!(
            retry_history.get(1),
            Some(Message::Assistant { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "checking "
                ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_1"
                                && tool_call.function.name == "add"
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_2"
                                && tool_call.function.name == "default_api"
                    ))
        ));
        assert!(matches!(
            retry_history.get(2),
            Some(Message::User { content })
                if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_1"
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    ToolResultContent::Text(text)
                                        if text.text == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                                ))
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_2"
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    ToolResultContent::Text(text)
                                        if text.text == "Use the add tool instead"
                                ))
                    ))
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_skips_mixed_streaming_turn_without_executing_valid_call() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 2, "y": 3}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::tool_call(
                    "tool_call_2",
                    "default_api",
                    serde_json::json!({"x": 4, "y": 5}),
                )
                .with_call_id("call_2"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("continued"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(skip_default_api_entry())
            .max_turns(3)
            .history(Vec::<Message>::new())
            .await;
        let mut skipped_tool_result = None;
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    tool_result,
                    ..
                })) => {
                    skipped_tool_result = Some(tool_result);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.output().to_string());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let skipped_tool_result =
            skipped_tool_result.expect("skip recovery should emit a synthetic tool result");
        assert_eq!(skipped_tool_result.id, "tool_call_2");
        assert_eq!(skipped_tool_result.call_id.as_deref(), Some("call_2"));
        assert_eq!(final_response_text.as_deref(), Some("continued"));
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let follow_up_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert_eq!(follow_up_history.len(), 3);
        assert!(matches!(
            follow_up_history.get(1),
            Some(Message::Assistant { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "checking "
                ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_1"
                                && tool_call.function.name == "add"
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_2"
                                && tool_call.function.name == "default_api"
                    ))
        ));
        assert!(matches!(
            follow_up_history.get(2),
            Some(Message::User { content })
                if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_1"
                                && result.call_id.as_deref() == Some("call_1")
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    ToolResultContent::Text(text)
                                        if text.text == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                                ))
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_2"
                                && result.call_id.as_deref() == Some("call_2")
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    ToolResultContent::Text(text)
                                        if text.text == "default_api was skipped"
                                ))
            ))
        ));
    }

    #[tokio::test]
    async fn invalid_completed_tool_call_skip_preserves_streaming_reasoning_history() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::reasoning("reasoned step").with_reasoning_id("rs_1"),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 2, "y": 3}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("continued"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(skip_default_api_entry())
            .max_turns(3)
            .history(Vec::<Message>::new())
            .await;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let follow_up_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert!(history_contains_text(&follow_up_history, "checking "));
        assert!(assistant_reasoning_precedes_tool_call(
            &follow_up_history,
            "reasoned step",
            "default_api"
        ));
        assert!(
            assistant_reasoning_precedes_text_and_tool_call(
                &follow_up_history,
                "reasoned step",
                "checking ",
                "default_api"
            ),
            "{follow_up_history:?}"
        );
    }

    #[tokio::test]
    async fn invalid_name_delta_retry_preserves_streaming_reasoning_history() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::reasoning_delta(Some("rs_1"), "delta reason"),
                MockStreamEvent::tool_call_arguments_delta(
                    "tool_call_1",
                    "internal_1",
                    r#"{"x":2,"y":3}"#,
                ),
                MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("retried"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(retry_default_api_entry())
            .max_turns(3)
            .history(Vec::<Message>::new())
            .max_invalid_tool_call_retries(1)
            .await;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let retry_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert!(assistant_reasoning_precedes_tool_call(
            &retry_history,
            "delta reason",
            "default_api"
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_skip_resets_streaming_text_delta_state() {
        let text_hook = RecordedTextDeltas::default();
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("stale "),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 2, "y": 3}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("fresh"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(text_hook.entry())
            .add_hook(skip_default_api_entry())
            .max_turns(3)
            .history(Vec::<Message>::new())
            .await;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            text_hook.observed(),
            vec![
                ("stale ".to_string(), "stale ".to_string()),
                ("fresh".to_string(), "fresh".to_string()),
            ]
        );
    }

    #[tokio::test]
    async fn invalid_tool_call_delta_retry_uses_structured_tool_feedback() {
        let delta_hook = RecordedToolCallDeltas::default();
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::reasoning_delta(Some("rs_1"), "diagnostic reason"),
                MockStreamEvent::tool_call(
                    "tool_call_0",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_0"),
                MockStreamEvent::tool_call_arguments_delta(
                    "tool_call_1",
                    "internal_1",
                    r#"{"x":2,"y":3}"#,
                ),
                MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("retried"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(delta_hook.entry())
            .add_hook(retry_default_api_entry())
            .max_turns(3)
            .history(Vec::<Message>::new())
            .max_invalid_tool_call_retries(1)
            .await;
        let mut completion_call_events = Vec::new();
        let mut final_response_text = None;
        let mut final_response_usage = Usage::new();
        let mut final_completion_calls = Vec::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(completion_call)) => {
                    completion_call_events.push(completion_call);
                }
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => panic!("invalid tool-call delta should not be emitted"),
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.output().to_string());
                    final_response_usage = response.usage();
                    final_completion_calls = response.completion_calls().to_vec();
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(final_response_text.as_deref(), Some("retried"));
        assert!(delta_hook.observed().is_empty());
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let mut first_usage = Usage::new();
        first_usage.total_tokens = 4;
        let mut second_usage = Usage::new();
        second_usage.total_tokens = 6;
        let expected_completion_calls = vec![
            CompletionCall::new(0, first_usage),
            CompletionCall::new(1, second_usage),
        ];
        assert_eq!(completion_call_events, expected_completion_calls);
        assert_eq!(final_completion_calls, expected_completion_calls);
        assert_eq!(final_response_usage.total_tokens, 10);

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let retry_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert!(matches!(
            retry_history.get(1),
            Some(Message::Assistant { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "checking "
                ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_0"
                                && tool_call.function.name == "add"
                    ))
                    && content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.function.name == "default_api"
                            && tool_call.function.arguments == serde_json::json!({"x": 2, "y": 3})
                ))
        ));
        assert!(matches!(
            retry_history.get(2),
            Some(Message::User { content })
                if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_0"
                                && result.call_id.as_deref() == Some("call_0")
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    ToolResultContent::Text(text)
                                        if text.text == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                                ))
                    ))
                    && content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.id == "tool_call_1"
                            && result.content.iter().any(|content| matches!(
                                content,
                                ToolResultContent::Text(text)
                                    if text.text == "Use the add tool instead"
                            ))
                ))
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_delta_context_includes_same_turn_history_and_tool_call_id() {
        let invalid_hook = RecordedInvalidToolCalls::default();
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::reasoning_delta(Some("rs_1"), "diagnostic reason"),
                MockStreamEvent::tool_call(
                    "tool_call_0",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_0"),
                MockStreamEvent::tool_call_arguments_delta(
                    "tool_call_1",
                    "internal_1",
                    r#"{"x":2,"y":3}"#,
                ),
                MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(invalid_hook.entry())
            .max_turns(3)
            .await;
        let mut error = None;

        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                error = Some(err);
                break;
            }
        }

        assert!(error.is_some(), "invalid name delta should fail");
        assert_eq!(recorded.request_count(), 1);
        let contexts = invalid_hook.observed();
        assert_eq!(contexts.len(), 1);
        let context = &contexts[0];
        assert_eq!(context.tool_name, "default_api");
        assert_eq!(context.tool_call_id.as_deref(), Some("tool_call_1"));
        assert_eq!(context.internal_call_id.as_deref(), Some("internal_1"));
        assert!(context.is_streaming);
        assert!(history_contains_text(&context.chat_history, "checking "));
        assert!(
            assistant_reasoning_precedes_tool_call(
                &context.chat_history,
                "diagnostic reason",
                "add"
            ),
            "{:?}",
            context.chat_history
        );
        assert!(history_contains_tool_call(&context.chat_history, "add"));
        assert!(history_contains_tool_call(
            &context.chat_history,
            "default_api"
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_delta_retry_resets_streaming_text_delta_state() {
        let text_hook = RecordedTextDeltas::default();
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("stale "),
                MockStreamEvent::tool_call_arguments_delta(
                    "tool_call_1",
                    "internal_1",
                    r#"{"x":2,"y":3}"#,
                ),
                MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("fresh"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(text_hook.entry())
            .add_hook(retry_default_api_entry())
            .max_turns(3)
            .history(Vec::<Message>::new())
            .max_invalid_tool_call_retries(1)
            .await;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            text_hook.observed(),
            vec![
                ("stale ".to_string(), "stale ".to_string()),
                ("fresh".to_string(), "fresh".to_string()),
            ]
        );
    }

    #[tokio::test]
    async fn invalid_tool_call_delta_skip_uses_structured_tool_feedback() {
        let delta_hook = RecordedToolCallDeltas::default();
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::tool_call(
                    "tool_call_0",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_0"),
                MockStreamEvent::tool_call_arguments_delta(
                    "tool_call_1",
                    "internal_1",
                    r#"{"x":2,"y":3}"#,
                ),
                MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("continued"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(delta_hook.entry())
            .add_hook(skip_default_api_entry())
            .max_turns(3)
            .history(Vec::<Message>::new())
            .await;
        let mut skipped_tool_result = None;
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => panic!("invalid tool-call delta should not be emitted"),
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    tool_result,
                    internal_call_id,
                })) => {
                    assert_eq!(internal_call_id, "internal_1");
                    skipped_tool_result = Some(tool_result);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.output().to_string());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let skipped_tool_result =
            skipped_tool_result.expect("skip recovery should emit a synthetic tool result");
        assert_eq!(skipped_tool_result.id, "tool_call_1");
        assert!(skipped_tool_result.call_id.is_none());
        assert!(skipped_tool_result.content.iter().any(|content| matches!(
            content,
            ToolResultContent::Text(text) if text.text == "default_api was skipped"
        )));
        assert_eq!(final_response_text.as_deref(), Some("continued"));
        assert!(delta_hook.observed().is_empty());
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let follow_up_history = requests[1].chat_history.iter().cloned().collect::<Vec<_>>();
        assert!(matches!(
            follow_up_history.get(1),
            Some(Message::Assistant { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "checking "
                ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(tool_call)
                            if tool_call.id == "tool_call_0"
                                && tool_call.function.name == "add"
                    ))
                    && content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.function.name == "default_api"
                            && tool_call.function.arguments == serde_json::json!({"x": 2, "y": 3})
                ))
        ));
        assert!(matches!(
            follow_up_history.get(2),
            Some(Message::User { content })
                if content.iter().filter(|item| matches!(item, UserContent::ToolResult(_))).count() == 2
                    && content.iter().any(|item| matches!(
                        item,
                        UserContent::ToolResult(result)
                            if result.id == "tool_call_0"
                                && result.call_id.as_deref() == Some("call_0")
                                && result.content.iter().any(|content| matches!(
                                    content,
                                    ToolResultContent::Text(text)
                                        if text.text == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
                                ))
                    ))
                    && content.iter().any(|item| matches!(
                    item,
                    UserContent::ToolResult(result)
                        if result.id == "tool_call_1"
                            && result.content.iter().any(|content| matches!(
                                content,
                                ToolResultContent::Text(text)
                                    if text.text == "default_api was skipped"
                            ))
                ))
        ));
    }

    #[tokio::test]
    async fn streaming_retry_budget_exhaustion_history_contains_invalid_tool_call() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 1, "y": 2}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(retry_default_api_entry())
            .max_turns(3)
            .max_invalid_tool_call_retries(0)
            .await;
        let mut error = None;

        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                error = Some(err);
                break;
            }
        }

        let error = error.expect("retry budget exhaustion should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    chat_history,
                    ..
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn streaming_name_delta_retry_budget_exhaustion_history_includes_same_turn_context() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("checking "),
                MockStreamEvent::tool_call(
                    "tool_call_0",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_0"),
                MockStreamEvent::tool_call_arguments_delta(
                    "tool_call_1",
                    "internal_1",
                    r#"{"x":2,"y":3}"#,
                ),
                MockStreamEvent::tool_call_name_delta("tool_call_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(retry_default_api_entry())
            .max_turns(3)
            .max_invalid_tool_call_retries(0)
            .await;
        let mut error = None;

        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                error = Some(err);
                break;
            }
        }

        let error = error.expect("retry budget exhaustion should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    chat_history,
                    ..
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert!(history_contains_text(&chat_history, "checking "));
                    assert!(history_contains_tool_call(&chat_history, "add"));
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn completed_unknown_tool_call_after_text_fails_before_finish_hook_or_later_emit() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::text("thinking "),
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "default_api",
                    serde_json::json!({"x": 1, "y": 2}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the tool")
            .add_hook(panic_on_unknown_tool_entry())
            .max_turns(3)
            .await;
        let mut saw_text = false;
        let mut saw_completion_call = false;
        let mut saw_final_response = false;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(_))) => {
                    saw_text = true;
                }
                Ok(MultiTurnStreamItem::CompletionCall(_)) => {
                    saw_completion_call = true;
                }
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Final(
                    _,
                )))
                | Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                    saw_final_response = true;
                }
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    ..
                })) => {
                    saw_tool_result = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(saw_text);
        assert!(!saw_completion_call);
        assert!(!saw_final_response);
        assert!(!saw_tool_call);
        assert!(!saw_tool_result);
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let error = error.expect("completed unknown tool call should fail immediately");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn mixed_streaming_tool_calls_fail_before_any_tool_execution() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::tool_call(
                    "tool_call_2",
                    "default_api",
                    serde_json::json!({"x": 3, "y": 4}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let mut stream = agent
            .stream_prompt("use tools")
            .add_hook(panic_on_unknown_tool_entry())
            .max_turns(3)
            .await;
        let mut saw_completion_call = false;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(_)) => {
                    saw_completion_call = true;
                }
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    ..
                })) => {
                    saw_tool_result = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_completion_call);
        assert!(!saw_tool_call);
        assert!(!saw_tool_result);
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let error = error.expect("mixed unknown streamed tool call should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn multiple_valid_streaming_tool_calls_execute_after_batch_validation() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let subtract_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::tool_call(
                    "tool_call_2",
                    "subtract",
                    serde_json::json!({"x": 8, "y": 3}),
                )
                .with_call_id("call_2"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .tool(CountingSubtractTool {
                calls: subtract_calls.clone(),
            })
            .build();

        let mut stream = agent.stream_prompt("use tools").max_turns(3).await;
        let mut tool_call_names = Vec::new();
        let mut tool_result_ids = Vec::new();
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { tool_call, .. },
                )) => {
                    tool_call_names.push(tool_call.function.name);
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    tool_result,
                    ..
                })) => {
                    tool_result_ids.push(tool_result.id);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response_text = Some(response.output().to_owned());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            tool_call_names,
            vec!["add".to_string(), "subtract".to_string()]
        );
        assert_eq!(
            tool_result_ids,
            vec!["tool_call_1".to_string(), "tool_call_2".to_string()]
        );
        assert_eq!(add_calls.load(Ordering::SeqCst), 1);
        assert_eq!(subtract_calls.load(Ordering::SeqCst), 1);
        assert_eq!(final_response_text.as_deref(), Some("done"));
        assert_eq!(recorded.request_count(), 2);
    }

    #[tokio::test]
    async fn disallowed_specific_tool_call_fails_before_streaming_second_request() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "subtract",
                    serde_json::json!({"x": 3, "y": 1}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the allowed tool")
            .add_hook(panic_on_unknown_tool_entry())
            .max_turns(3)
            .await;
        let mut saw_tool_call = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_tool_call);
        let error = error.expect("disallowed model-emitted tool should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "subtract");
                    assert_eq!(
                        available_tools,
                        vec!["add".to_string(), "subtract".to_string()]
                    );
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "subtract"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn mixed_specific_tool_calls_fail_before_any_tool_execution() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                ),
                MockStreamEvent::tool_call(
                    "tool_call_2",
                    "subtract",
                    serde_json::json!({"x": 3, "y": 1}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let mut stream = agent
            .stream_prompt("use the allowed tool")
            .add_hook(panic_on_unknown_tool_entry())
            .max_turns(3)
            .await;
        let mut saw_tool_call = false;
        let mut saw_tool_result = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                    ..
                })) => {
                    saw_tool_result = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_tool_call);
        assert!(!saw_tool_result);
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let error = error.expect("mixed disallowed streamed tool call should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "subtract");
                    assert_eq!(
                        available_tools,
                        vec!["add".to_string(), "subtract".to_string()]
                    );
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "subtract"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_choice_none_rejects_streaming_tool_call() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                ),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let mut stream = agent
            .stream_prompt("do not use tools")
            .add_hook(panic_on_unknown_tool_entry())
            .max_turns(3)
            .await;
        let mut saw_tool_call = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCall { .. },
                )) => {
                    saw_tool_call = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_tool_call);
        let error = error.expect("ToolChoice::None should reject returned tool calls");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "add");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert!(allowed_tools.is_empty());
                    assert!(history_contains_tool_call(&chat_history, "add"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_choice_none_rejects_streaming_tool_call_name_delta_before_hook_or_emit() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "add"),
                MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":1}"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let mut stream = agent
            .stream_prompt("do not use tools")
            .add_hook(panic_on_unknown_tool_entry())
            .max_turns(3)
            .await;
        let mut saw_delta = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_delta);
        let error = error.expect("ToolChoice::None should reject returned tool-call deltas");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "add");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert!(allowed_tools.is_empty());
                    assert!(history_contains_tool_call(&chat_history, "add"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn unknown_tool_call_name_delta_fails_before_streaming_delta_hook_or_emit() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "default_api"),
                MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":1}"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("stream a bad tool call")
            .add_hook(panic_on_unknown_tool_entry())
            .max_turns(3)
            .await;
        let mut saw_delta = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_delta);
        let error = error.expect("unknown tool-call name delta should fail");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_call_args_delta_before_unknown_name_fails_before_hook_or_emit() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":1}"),
                MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "default_api"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("stream a bad tool call")
            .add_hook(panic_on_unknown_tool_entry())
            .max_turns(3)
            .await;
        let mut saw_delta = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_delta);
        let error = error.expect("unknown tool-call name should reject buffered args");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "default_api");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert_eq!(allowed_tools, vec!["add".to_string()]);
                    assert!(history_contains_tool_call(&chat_history, "default_api"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_call_args_delta_before_valid_name_buffers_then_emits_in_safe_order() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":"),
            MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "1}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]);
        let hook = RecordedToolCallDeltas::default();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("stream a tool call")
            .add_hook(hook.entry())
            .await;
        let mut stream_deltas = Vec::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    },
                )) => {
                    stream_deltas.push((id, internal_call_id, content));
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            hook.observed(),
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    Some("add".to_string()),
                    String::new()
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    None,
                    "{\"x\":".to_string()
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    None,
                    "1}".to_string()
                ),
            ]
        );
        assert_eq!(
            stream_deltas,
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Name("add".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("{\"x\":".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("1}".to_string())
                ),
            ]
        );
    }

    #[tokio::test]
    async fn tool_call_args_delta_without_name_errors_at_stream_end() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":1}"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("stream an incomplete tool call")
            .add_hook(panic_on_unknown_tool_entry())
            .max_turns(3)
            .await;
        let mut saw_delta = false;
        let mut saw_completion_call = false;
        let mut saw_final_response = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(MultiTurnStreamItem::CompletionCall(_)) => {
                    saw_completion_call = true;
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                    saw_final_response = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_delta);
        assert!(!saw_completion_call);
        assert!(!saw_final_response);
        let error = error.expect("unterminated tool-call args delta should fail");
        match error {
            StreamingError::Completion(CompletionError::ResponseError(message)) => {
                assert!(
                    message.contains("streamed tool call arguments"),
                    "{message}"
                );
                assert!(message.contains("tool_1"), "{message}");
                assert!(message.contains("internal_1"), "{message}");
            }
            other => panic!("expected completion response error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_choice_none_buffers_args_then_rejects_name_without_emit() {
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":1}"),
                MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "add"),
                MockStreamEvent::final_response_with_total_tokens(4),
            ],
            vec![
                MockStreamEvent::text("should not be requested"),
                MockStreamEvent::final_response_with_total_tokens(6),
            ],
        ]);
        let recorded = model.clone();
        let agent = agent_builder(model)
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let mut stream = agent
            .stream_prompt("do not use tools")
            .add_hook(panic_on_unknown_tool_entry())
            .max_turns(3)
            .await;
        let mut saw_delta = false;
        let mut error = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        assert!(!saw_delta);
        let error = error.expect("ToolChoice::None should reject buffered tool-call deltas");
        match error {
            StreamingError::Prompt(err) => match *err {
                PromptError::UnknownToolCall {
                    tool_name,
                    available_tools,
                    allowed_tools,
                    chat_history,
                } => {
                    assert_eq!(tool_name, "add");
                    assert_eq!(available_tools, vec!["add".to_string()]);
                    assert!(allowed_tools.is_empty());
                    assert!(history_contains_tool_call(&chat_history, "add"));
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt streaming error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn stream_prompt_emits_tool_call_deltas_without_hook() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "1}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]);
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent.stream_prompt("stream a tool call").await;
        let mut deltas = Vec::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    },
                )) => {
                    deltas.push((id, internal_call_id, content));
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            deltas,
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Name("add".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("{\"x\":".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("1}".to_string())
                ),
            ]
        );
    }

    #[tokio::test]
    async fn stream_prompt_emits_tool_call_deltas_after_hook_continue() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "1}"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]);
        let hook = RecordedToolCallDeltas::default();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("stream a tool call")
            .add_hook(hook.entry())
            .await;
        let mut stream_deltas = Vec::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    },
                )) => {
                    stream_deltas.push((id, internal_call_id, content));
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            hook.observed(),
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    Some("add".to_string()),
                    String::new()
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    None,
                    "{\"x\":".to_string()
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    None,
                    "1}".to_string()
                ),
            ]
        );
        assert_eq!(
            stream_deltas,
            vec![
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Name("add".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("{\"x\":".to_string())
                ),
                (
                    "tool_1".to_string(),
                    "internal_1".to_string(),
                    ToolCallDeltaContent::Delta("1}".to_string())
                ),
            ]
        );
    }

    #[tokio::test]
    async fn stream_prompt_tool_call_deltas_hook_termination_prevents_delta_emit() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::tool_call_name_delta("tool_1", "internal_1", "add"),
            MockStreamEvent::tool_call_arguments_delta("tool_1", "internal_1", "{\"x\":"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]);
        let hook = RecordedToolCallDeltas::default();
        let agent = agent_builder(model).tool(MockAddTool).build();

        let mut stream = agent
            .stream_prompt("stream a tool call")
            .add_hook(hook.entry_with(ObservationAction::stop("stop on tool call delta")))
            .await;
        let mut saw_delta = false;
        let mut saw_final_response = false;
        let mut error_message = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(
                    StreamedAssistantContent::ToolCallDelta { .. },
                )) => {
                    saw_delta = true;
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                    saw_final_response = true;
                }
                Ok(_) => {}
                Err(err) => {
                    error_message = Some(err.to_string());
                    break;
                }
            }
        }

        assert_eq!(
            hook.observed(),
            vec![(
                "tool_1".to_string(),
                "internal_1".to_string(),
                Some("add".to_string()),
                String::new()
            )]
        );
        assert!(!saw_delta);
        assert!(!saw_final_response);
        assert!(
            error_message
                .as_deref()
                .is_some_and(|message| message.contains("PromptCancelled: stop on tool call delta")),
            "expected hook termination error, got {error_message:?}"
        );
    }

    #[tokio::test]
    async fn stream_prompt_exposes_completion_calls() {
        let first_call_usage = usage(10, 2);
        let second_call_usage = usage(25, 5);
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response(first_call_usage),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response(second_call_usage),
            ],
        ]);
        let agent = agent_builder(model).tool(MockAddTool).build();
        let empty_history: &[Message] = &[];

        let mut stream = agent
            .stream_prompt("do tool work")
            .history(empty_history)
            .max_turns(3)
            .await;
        let mut completion_calls_events = Vec::new();
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(call_usage)) => {
                    completion_calls_events.push(call_usage);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response = Some(response);
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(
            completion_calls_events,
            vec![
                CompletionCall::new(0, first_call_usage),
                CompletionCall::new(1, second_call_usage)
            ]
        );

        let final_response = final_response.expect("expected final response");
        assert_eq!(
            final_response.usage(),
            Usage {
                input_tokens: 35,
                output_tokens: 7,
                total_tokens: 42,
                cached_input_tokens: 0,
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            }
        );
        assert_eq!(
            final_response.completion_calls(),
            &[
                CompletionCall::new(0, first_call_usage),
                CompletionCall::new(1, second_call_usage)
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stream_prompt_records_single_call_usage_on_chat_span_under_outer_span() {
        let call_usage = usage(10, 2);
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response(call_usage),
        ]]);
        let agent = agent_builder(model).build();

        assert_stream_usage_recorded_on_chat_spans(agent, "say done", 1, &[call_usage]).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stream_prompt_records_multi_turn_usage_on_chat_spans_under_outer_span() {
        let first_call_usage = usage(10, 2);
        let second_call_usage = usage(25, 5);
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
                MockStreamEvent::final_response(first_call_usage),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response(second_call_usage),
            ],
        ]);
        let agent = agent_builder(model).tool(MockAddTool).build();

        assert_stream_usage_recorded_on_chat_spans(
            agent,
            "do tool work",
            3,
            &[first_call_usage, second_call_usage],
        )
        .await;
    }

    #[tokio::test]
    async fn stream_prompt_emits_completion_call_before_finish_hook_termination() {
        let call_usage = usage(10, 2);
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response(call_usage),
        ]]);
        let agent = agent_builder(model).build();

        let mut stream = agent
            .stream_prompt("say done")
            .add_hook(terminate_on_stream_finish_entry())
            .await;
        let mut completion_calls = Vec::new();
        let mut saw_error = false;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(completion_call)) => {
                    completion_calls.push(completion_call);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    panic!("unexpected final response after hook termination: {response:?}");
                }
                Ok(_) => {}
                Err(_) => {
                    saw_error = true;
                    break;
                }
            }
        }

        assert_eq!(completion_calls, vec![CompletionCall::new(0, call_usage)]);
        assert!(saw_error);
    }

    #[tokio::test]
    async fn stream_prompt_completion_calls_records_unreported_usage() {
        let second_call_usage = usage(25, 5);
        let model = MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call(
                    "tool_call_1",
                    "add",
                    serde_json::json!({"x": 1, "y": 2}),
                )
                .with_call_id("call_1"),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response(second_call_usage),
            ],
        ]);
        let agent = agent_builder(model).tool(MockAddTool).build();
        let empty_history: &[Message] = &[];

        let mut stream = agent
            .stream_prompt("do tool work")
            .history(empty_history)
            .max_turns(3)
            .await;
        let mut completion_calls_events = Vec::new();
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::CompletionCall(call_usage)) => {
                    completion_calls_events.push(call_usage);
                }
                Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                    final_response = Some(response);
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let expected_usage = vec![
            CompletionCall::new(0, Usage::new()),
            CompletionCall::new(1, second_call_usage),
        ];
        assert_eq!(completion_calls_events, expected_usage);

        let final_response = final_response.expect("expected final response");
        assert_eq!(final_response.completion_calls(), expected_usage.as_slice());
    }

    #[tokio::test]
    async fn final_response_matches_streamed_text_when_provider_final_is_textless() {
        let agent = agent_builder(streaming_text_then_final_model()).build();

        let mut stream = agent.stream_prompt("say hello").await;
        let mut streamed_text = String::new();
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => streamed_text.push_str(&text.text),
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_response_text = Some(res.output().to_owned());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert_eq!(streamed_text, "hello world");
        assert_eq!(final_response_text.as_deref(), Some("hello world"));
    }

    #[tokio::test]
    async fn final_response_preserves_structured_text_metadata() {
        let agent = agent_builder(streaming_cited_text_then_final_model()).build();

        let mut stream = agent.stream_prompt("answer with citations").await;
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_response = Some(res);
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let final_response = final_response.expect("expected final response");
        assert_eq!(final_response.output(), "cited answer");
        let metadata = text_metadata(final_response.content())
            .expect("expected text metadata in final content");
        assert_eq!(
            metadata["citations"][0]["encrypted_index"],
            "encrypted-reference"
        );
    }

    #[tokio::test]
    async fn final_response_history_preserves_structured_text_metadata() {
        let agent = agent_builder(streaming_cited_text_then_final_model()).build();

        let empty_history: &[Message] = &[];
        let mut stream = agent
            .stream_prompt("answer with citations")
            .history(empty_history)
            .await;
        let mut final_response = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_response = Some(res);
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let final_response = final_response.expect("expected final response");
        let history = final_response
            .messages()
            .expect("with_history should include final history");
        let assistant_content = history
            .iter()
            .find_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .expect("expected assistant message in history");
        let metadata =
            text_metadata(assistant_content).expect("expected text metadata in assistant history");
        assert_eq!(
            metadata["citations"][0]["encrypted_index"],
            "encrypted-reference"
        );
    }

    #[tokio::test]
    async fn tool_follow_up_history_preserves_structured_text_metadata() {
        let model = streaming_cited_text_then_tool_model();
        let recorded = model.clone();
        let agent = agent_builder(model).tool(MockAddTool).build();
        let empty_history: &[Message] = &[];

        let mut stream = agent
            .stream_prompt("use a tool with citations")
            .history(empty_history)
            .max_turns(3)
            .await;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(_)) => break,
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let follow_up_history = requests[1].chat_history.iter().collect::<Vec<_>>();
        let assistant_content = follow_up_history
            .iter()
            .find_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .expect("expected assistant message in follow-up history");
        let metadata = text_metadata(assistant_content)
            .expect("expected citation metadata in follow-up assistant history");
        assert_eq!(
            metadata["citations"][0]["encrypted_index"],
            "encrypted-reference"
        );
    }

    #[tokio::test]
    async fn final_response_can_remain_empty_for_truly_textless_turns() {
        let agent = agent_builder(streaming_final_only_model()).build();

        let mut stream = agent.stream_prompt("say nothing").await;
        let mut streamed_text = String::new();
        let mut final_response_text = None;

        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => streamed_text.push_str(&text.text),
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_response_text = Some(res.output().to_owned());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        assert!(streamed_text.is_empty());
        assert_eq!(final_response_text.as_deref(), Some(""));
    }

    /// Background task that logs periodically to detect span leakage.
    /// If span leakage occurs, these logs will be prefixed with `invoke_agent{...}`.
    async fn background_logger(stop: Arc<AtomicBool>, leak_count: Arc<AtomicU32>) {
        let mut interval = tokio::time::interval(Duration::from_millis(50));
        let mut count = 0u32;

        while !stop.load(Ordering::Relaxed) {
            interval.tick().await;
            count += 1;

            tracing::event!(
                target: "background_logger",
                tracing::Level::INFO,
                count = count,
                "Background tick"
            );

            // Check if we're inside an unexpected span
            let current = tracing::Span::current();
            if !current.is_disabled() && !current.is_none() {
                leak_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        tracing::info!(target: "background_logger", total_ticks = count, "Background logger stopped");
    }

    /// Test that span context doesn't leak to concurrent tasks during streaming.
    ///
    /// This test verifies that using `.instrument()` instead of `span.enter()` in
    /// async_stream prevents thread-local span context from leaking to other tasks.
    ///
    /// Uses single-threaded runtime to force all tasks onto the same thread,
    /// making the span leak deterministic (it only occurs when tasks share a thread).
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "This requires an API key"]
    async fn test_span_context_isolation() -> anyhow::Result<()> {
        let stop = Arc::new(AtomicBool::new(false));
        let leak_count = Arc::new(AtomicU32::new(0));

        // Start background logger
        let bg_stop = stop.clone();
        let bg_leak = leak_count.clone();
        let bg_handle = tokio::spawn(async move {
            background_logger(bg_stop, bg_leak).await;
        });

        // Small delay to let background logger start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Make streaming request WITHOUT an outer span so rig creates its own invoke_agent span
        // (rig reuses current span if one exists, so we need to ensure there's no current span)
        let client = anthropic::Client::from_env()?;
        let agent = client
            .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
            .preamble("You are a helpful assistant.")
            .temperature(0.1)
            .max_tokens(100)
            .build();

        let mut stream = agent
            .stream_prompt("Say 'hello world' and nothing else.")
            .await;

        let mut full_content = String::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => {
                    full_content.push_str(&text.text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(_)) => {
                    break;
                }
                Err(e) => {
                    tracing::warn!("Error: {:?}", e);
                    break;
                }
                _ => {}
            }
        }

        tracing::info!("Got response: {:?}", full_content);

        // Stop background logger
        stop.store(true, Ordering::Relaxed);
        bg_handle.await?;

        let leaks = leak_count.load(Ordering::Relaxed);
        anyhow::ensure!(
            leaks == 0,
            "SPAN LEAK DETECTED: Background logger was inside unexpected spans {leaks} times. \
             This indicates that span.enter() is being used inside async_stream instead of .instrument()"
        );

        Ok(())
    }

    /// Test that FinalResponse contains the updated chat history when a starting
    /// history is provided via `.history(..)`.
    ///
    /// This verifies that:
    /// 1. PromptResponse.messages() returns Some when a starting history was provided
    /// 2. The history contains both the user prompt and assistant response
    #[tokio::test]
    #[ignore = "This requires an API key"]
    async fn test_chat_history_in_final_response() -> anyhow::Result<()> {
        use rig_core::message::Message;

        let client = anthropic::Client::from_env()?;
        let agent = client
            .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
            .preamble("You are a helpful assistant. Keep responses brief.")
            .temperature(0.1)
            .max_tokens(50)
            .build();

        // Send streaming request with history
        let empty_history: &[Message] = &[];
        let mut stream = agent
            .stream_prompt("Say 'hello' and nothing else.")
            .history(empty_history)
            .await;

        // Consume the stream and collect FinalResponse
        let mut response_text = String::new();
        let mut final_history = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                    text,
                ))) => {
                    response_text.push_str(&text.text);
                }
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    final_history = res.messages().map(|h| h.to_vec());
                    break;
                }
                Err(e) => {
                    return Err(e.into());
                }
                _ => {}
            }
        }

        let history = final_history
            .ok_or_else(|| anyhow::anyhow!("final response should include history"))?;

        // Should contain at least the user message
        anyhow::ensure!(
            history.iter().any(|m| matches!(m, Message::User { .. })),
            "History should contain the user message"
        );

        // Should contain the assistant response
        anyhow::ensure!(
            history
                .iter()
                .any(|m| matches!(m, Message::Assistant { .. })),
            "History should contain the assistant response"
        );

        tracing::info!(
            "History after streaming: {} messages, response: {:?}",
            history.len(),
            response_text
        );

        Ok(())
    }

    #[tokio::test]
    async fn streaming_final_response_carries_the_committed_transcript() {
        // The host persists a turn by appending `PromptResponse::messages`
        // after the stream completes, so the streaming surface must publish
        // the committed transcript on the final response.
        let agent = agent_builder(streaming_text_then_final_model()).build();

        let mut stream = agent.stream_prompt("hi there").await;

        let mut history_in_final = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    history_in_final = res.messages().map(|h| h.to_vec());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let final_history = history_in_final.expect("PromptResponse.messages should be populated");
        assert_eq!(
            final_history.len(),
            2,
            "user prompt + assistant response in final history: {final_history:?}"
        );
    }
    #[tokio::test]
    async fn streaming_reasoning_without_tools_does_not_duplicate_final_history() {
        let agent = agent_builder(MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("final answer"),
            MockStreamEvent::reasoning("reasoned step").with_reasoning_id("rs_1"),
            MockStreamEvent::final_response_with_total_tokens(3),
        ]]))
        .build();

        let mut stream = agent
            .stream_prompt("think before answering")
            .history(Vec::<Message>::new())
            .await;

        let mut history_in_final = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                    history_in_final = res.messages().map(|h| h.to_vec());
                    break;
                }
                Ok(_) => {}
                Err(err) => panic!("unexpected streaming error: {err:?}"),
            }
        }

        let final_history = history_in_final
            .expect("PromptResponse.messages should be populated when with_history is used");
        assert_eq!(
            final_history.len(),
            2,
            "user prompt + one assistant response in final history: {final_history:?}"
        );

        assert!(matches!(
            final_history.first(),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::Text(text) if text.text == "think before answering"
                )
        ));

        let assistant_messages = final_history
            .iter()
            .filter_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            assistant_messages.len(),
            1,
            "reasoning turn should produce exactly one assistant history message: {final_history:?}"
        );
        let assistant_content = assistant_messages
            .first()
            .expect("expected assistant history message");
        assert!(assistant_content.iter().any(|item| matches!(
            item,
            AssistantContent::Text(text) if text.text == "final answer"
        )));
        assert!(assistant_content.iter().any(|item| matches!(
            item,
            AssistantContent::Reasoning(reasoning)
                if reasoning.id.as_deref() == Some("rs_1")
                    && reasoning.content.iter().any(|content| matches!(
                        content,
                        ReasoningContent::Text { text, .. } if text == "reasoned step"
                    ))
        )));
        let reasoning_index = assistant_content
            .iter()
            .position(|item| matches!(item, AssistantContent::Reasoning(_)))
            .expect("assistant history should contain reasoning");
        let text_index = assistant_content
            .iter()
            .position(|item| matches!(item, AssistantContent::Text(_)))
            .expect("assistant history should contain text");
        assert!(
            reasoning_index < text_index,
            "assistant reasoning must be stored before assistant text: {assistant_content:?}"
        );
    }
}
