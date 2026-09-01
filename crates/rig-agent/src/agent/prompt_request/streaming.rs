use rig_core::id::InternalCallId;
use rig_core::{
    message::{AssistantContent, UserContent},
    telemetry::SpanCombinator,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend},
};

use crate::{
    agent::completion::{PreparedCompletionRequest, build_prepared_completion_request},
    agent::hook::{
        AgentHook, CompletionResponse as CompletionResponseEvent, HookContext, HookStack,
        InvalidToolCallAction, ModelSelection, ModelSelectionAction, ModelTurnFinished,
        ReasoningDelta, RunSettled, RunStart, RunStartAction, SettledOutcome, StepEventKind,
        TextDelta, ToolCallDelta,
    },
    agent::prompt_request::{assistant_text_from_choice, is_empty_assistant_turn},
    agent::run::{
        AgentRun, AgentRunStep, PendingToolCall,
        streamed::{StreamedResolution, StreamedTurnAssembler, StreamedTurnEvent},
    },
    agent::runner::{
        AgentRunner, CompletionCallOutcome, ModelTurnDecision, ToolExecution, append_run_messages,
        build_chat_span, new_execute_tool_span, observe_action, resolve_completion_call,
        resolve_model_turn_action, run_single_tool,
    },
    streaming::{StreamedAssistantContent, StreamedUserContent, ToolCallDeltaContent},
    tool::{ToolContext, server::ToolRegistrySnapshot},
};
use futures::{SinkExt, Stream, StreamExt, channel::mpsc, stream, stream::FusedStream};
use serde::{Deserialize, Serialize};
use std::{collections::VecDeque, pin::Pin, sync::Arc};
use tracing_futures::Instrument;

use super::{CompletionCall, PromptResponse, forward_prompt_setters};
use crate::{
    agent::{Agent, ModelHandle},
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
        internal_call_id: InternalCallId,
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
    ///
    /// Terminal for the run: nothing follows it automatically — no retry,
    /// further turn, or tool execution — so this item is the stream-side
    /// counterpart of the `on_run_settled` hook's success outcome. Error
    /// termination surfaces as the stream's `Err` item instead, which is
    /// equally terminal.
    FinalResponse(PromptResponse),
}

/// Build the unified [`PromptResponse`] for the streaming surface from the
/// final turn's structured content.
fn final_response_from_content(
    content: Vec<AssistantContent>,
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

    /// Build a `FinalResponse` item from final-turn content, applying the
    /// run-finalization shaping of `final_response_from_content` (#1928).
    /// The one public entry point to that shaping, for mocks and adapters
    /// that synthesize final items outside the drive loop.
    pub fn final_response(
        content: Vec<AssistantContent>,
        aggregated_usage: crate::completion::Usage,
    ) -> Self {
        Self::FinalResponse(final_response_from_content(
            content,
            aggregated_usage,
            Vec::new(),
            None,
        ))
    }

    pub(crate) fn final_response_with_completion_calls(
        content: Vec<AssistantContent>,
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
    last_final_choice: &[AssistantContent],
    output: &str,
) -> Option<Vec<AssistantContent>> {
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
    // `items` is non-empty: the output text was just pushed unconditionally.
    items.push(AssistantContent::text(output.to_string()));
    Some(items)
}

#[derive(Debug, thiserror::Error)]
pub enum StreamingError {
    #[error("CompletionError: {0}")]
    Completion(#[from] CompletionError),
    #[error("PromptError: {0}")]
    Prompt(#[from] Box<PromptError>),
}

impl From<rig_core::memory::MemoryError> for StreamingError {
    fn from(err: rig_core::memory::MemoryError) -> Self {
        Self::Prompt(Box::new(PromptError::MemoryError(err)))
    }
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
    pub fn new(agent: &Agent, prompt: impl Into<Message>) -> StreamingPromptRequest {
        Self::from_agent(agent, prompt)
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
    /// compose is event-dependent (model selections and `ToolCall`/`ToolResult` rewrites
    /// chain, `CompletionCall` request patches accumulate and merge, while model-turn
    /// steering and observe-only/recovery events use first-non-`Continue`-wins). See the
    /// [`hook`](crate::agent::hook) module docs.
    pub fn add_hook<H>(mut self, hook: H) -> Self
    where
        H: AgentHook + 'static,
    {
        self.runner = self.runner.add_hook(hook);
        self
    }

    forward_prompt_setters!(runner);

    async fn send(self) -> StreamingResult {
        self.runner.stream().await
    }

    /// Split the configured run into a driving future and a [`RunEvents`]
    /// feed instead of a stream. See [`AgentRunner::run_channel`].
    pub fn run_channel(
        self,
    ) -> (
        impl Future<Output = Result<PromptResponse, PromptError>> + WasmCompatSend,
        RunEvents,
    ) {
        self.runner.run_channel()
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
/// hook, request preparation, memory) lives once in [`drive_agent`]; only the
/// genuinely divergent pieces are behind this trait. Invalid-tool-call recovery
/// is one of them — it lives inside each source's `run_model_turn` (end-of-turn
/// for blocking, mid-stream for streaming), not in `drive_agent`.
pub(crate) trait TurnSource: WasmCompatSend {
    /// Build this medium's per-turn `chat` span (name + parenting + any
    /// `follows_from` chaining differ between blocking and streaming).
    fn open_chat_span(
        &self,
        runner: &AgentRunner,
        effective_preamble: Option<&str>,
    ) -> tracing::Span;

    /// Run one model turn: issue the provider call, feed the result into the
    /// sans-IO machine, and yield any intermediate items. Returning normally
    /// advances the loop; yielding an `Err` terminates the run.
    #[allow(clippy::too_many_arguments)]
    fn run_model_turn<'a>(
        &'a mut self,
        runner: &'a AgentRunner,
        hook_ctx: &'a HookContext,
        run: &'a mut AgentRun,
        prepared: PreparedCompletionRequest,
        chat_span: tracing::Span,
        agent_span: &'a tracing::Span,
        prompt: Message,
    ) -> DriveStream<'a>;

    /// Execute a turn's tool calls, feeding the results into the machine and
    /// yielding any intermediate items.
    fn run_tool_calls<'a>(
        &'a self,
        runner: &'a AgentRunner,
        hook_ctx: &'a HookContext,
        run: &'a mut AgentRun,
        calls: Vec<PendingToolCall>,
        tool_snapshot: Arc<ToolRegistrySnapshot>,
    ) -> DriveStream<'a>;

    /// Record run-level telemetry onto the agent span at `Done`. Gated on
    /// `created_agent_span` so a caller-supplied outer span is never polluted.
    fn record_run_level_telemetry(
        &self,
        agent_span: &tracing::Span,
        response: &PromptResponse,
        created_agent_span: bool,
    );

    /// Build the final stream item surfaced at `Done`, or `None` when the
    /// surface discards it (the blocking fold) so the engine skips the work.
    fn final_item(&self, response: &PromptResponse) -> Option<MultiTurnStreamItem>;
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
        *usage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = run.usage();
    }
}

/// The single agent drive loop, shared by the blocking and streaming surfaces.
///
/// Owns the medium-independent loop — `next_step` dispatch, the `CompletionCall`
/// hook + request preparation, the `Done` memory append — and delegates the
/// medium-specific model call, tool execution, span shaping and finalization to
/// a [`TurnSource`]. The streaming surface forwards the yielded [`DriveItem`]s;
/// the blocking surface folds them to `Done`.
pub(crate) fn drive_agent<S>(
    runner: AgentRunner,
    mut source: S,
    mut run: AgentRun,
    agent_span: tracing::Span,
    created_agent_span: bool,
    memory_handle: Option<(
        Arc<dyn rig_core::memory::ConversationMemory>,
        rig_core::id::ConversationId,
    )>,
    is_streaming: bool,
) -> impl Stream<Item = Result<DriveItem, StreamingError>>
where
    S: TurnSource,
{
    async_stream::stream! {
        // Run-scoped hook context: minted once, shared by every hook event on
        // both surfaces. `is_streaming` records which surface is driving; the
        // per-turn index is advanced on each `CallModel` step below.
        let hook_ctx = HookContext::new(is_streaming, runner.config.name.clone());
        // Seed the entries a resumed run carried, so `HookContext::entries`
        // replays the full record from the first hook event on.
        hook_ctx.seed_entries(run.entries());
        // Flush hook appends into the run's record. Called at every step
        // boundary and in the Done arm, so the run — the serializable record
        // — is current whenever it can be observed. Entries appended by the
        // terminal `on_run_settled` hook are documented as not persisted.
        macro_rules! flush_entries {
            () => {
                for entry in hook_ctx.drain_pending_entries() {
                    run.append_entry(entry);
                }
            };
        }
        // Rendered terminal-error text, set on every error path before its
        // yield so the run-settled hook below can report the outcome after
        // the error itself has been moved into the stream.
        let mut settled_error: Option<String> = None;
        // Set only after a model turn commits successfully and consumed by its
        // immediately following CallTools step. This keeps the sans-IO run state
        // serializable while pinning execution to the definitions sent that turn.
        let mut pending_tool_snapshot: Option<Arc<ToolRegistrySnapshot>> = None;
        // Live routing state stays in the driver, not the serde `AgentRun`. It
        // records the model behind the preceding *issued* attempt: it advances
        // immediately before the selected model's unary or streaming operation
        // is invoked, so a completion-call stop, selection stop, or preparation
        // failure leaves it unchanged while a provider error still counts.
        let mut previous_model: Option<ModelHandle> = None;

        // Pre-run hook: fired once with the initial prompt before any model
        // call. Rewrites chain across the stack in registration order; the
        // first stop wins and terminates the run before any provider work.
        if runner.config.hooks.observes(StepEventKind::RunStart) {
            let action = match run.initial_prompt() {
                Some(prompt) => {
                    runner
                        .config
                        .hooks
                        .on_run_start(
                            &hook_ctx,
                            RunStart {
                                prompt,
                                history: run.input_chat_history(),
                            },
                        )
                        .await
                }
                // A run resumed past its first model call has no pending
                // initial prompt to steer.
                None => RunStartAction::Continue,
            };
            let early_stop = match action {
                RunStartAction::Continue => None,
                RunStartAction::Rewrite(prompt) => run
                    .rewrite_initial_prompt(prompt)
                    .err()
                    .map(|err| StreamingError::Prompt(Box::new(err))),
                RunStartAction::Stop(reason) => {
                    Some(StreamingError::Prompt(Box::new(run.cancel_error(reason))))
                }
            };
            if let Some(err) = early_stop {
                store_error_usage(&runner, &run);
                let reason = err.to_string();
                yield Err(err);
                if runner.config.hooks.observes(StepEventKind::RunSettled) {
                    runner
                        .config
                        .hooks
                        .on_run_settled(
                            &hook_ctx,
                            RunSettled {
                                outcome: SettledOutcome::Error(&reason),
                            },
                        )
                        .await;
                }
                return;
            }
        }

        // Drive one medium-specific step stream: forward its items, and on the
        // first error store error usage, surface it, and end the run. A macro
        // because `yield`/`break 'outer` cannot cross a fn boundary; the loop
        // label is passed in because labels are hygienic across the macro edge.
        macro_rules! drive_step {
            ($label:lifetime, $step_stream:expr) => {{
                let mut step_stream = $step_stream;
                let mut step_error = None;
                while let Some(item) = step_stream.next().await {
                    match item {
                        Ok(item) => yield Ok(DriveItem::Item(item)),
                        Err(err) => {
                            step_error = Some(err);
                            break;
                        }
                    }
                }
                drop(step_stream);
                if let Some(err) = step_error {
                    store_error_usage(&runner, &run);
                    settled_error = Some(err.to_string());
                    yield Err(err);
                    break $label;
                }
            }};
        }

        'outer: loop {
            flush_entries!();
            let step = match run.next_step() {
                Ok(step) => step,
                Err(err) => {
                    store_error_usage(&runner, &run);
                    let err: StreamingError = Box::new(err).into();
                    settled_error = Some(err.to_string());
                    yield Err(err);
                    break 'outer;
                }
            };

            match step {
                AgentRunStep::CallModel { prompt, history, turn } => {
                    drop(pending_tool_snapshot.take());
                    if runner.config.max_turns > 1 {
                        tracing::info!("Current conversation Turns: {}/{}", turn, runner.config.max_turns);
                    }
                    hook_ctx.set_turn(turn);

                    // Completion-call hooks resolve FIRST: a stop here suppresses
                    // model selection entirely, and their merged `RequestPatch`
                    // is handed to the selection hooks below.
                    let request_patch =
                        match resolve_completion_call(&runner.config.hooks, &hook_ctx, &prompt, &history, turn).await {
                            CompletionCallOutcome::Terminate(reason) => {
                                store_error_usage(&runner, &run);
                                let err = StreamingError::Prompt(Box::new(run.cancel_error(reason)));
                                settled_error = Some(err.to_string());
                                yield Err(err);
                                break 'outer;
                            }
                            CompletionCallOutcome::Proceed(request_patch) => request_patch,
                        };

                    // Resolve routing once at the model-call boundary, after the
                    // completion-call hooks proceed. The resulting handle is
                    // cloned into the prepared attempt, so request preparation
                    // inspects the *selected* model's captured capabilities and
                    // the same handle executes the request.
                    let selected_model = match runner.config.hooks.on_model_select(
                        &hook_ctx,
                        ModelSelection {
                            prompt: &prompt,
                            history: &history,
                            request_patch: request_patch.as_ref(),
                            previous_model: previous_model.as_ref(),
                            default_model: &runner.config.model,
                            selected_model: &runner.config.model,
                        },
                    ) {
                        ModelSelectionAction::Continue => runner.config.model.clone(),
                        ModelSelectionAction::Select(model) => model,
                        ModelSelectionAction::Stop(reason) => {
                            store_error_usage(&runner, &run);
                            let err = StreamingError::Prompt(Box::new(run.cancel_error(reason)));
                            settled_error = Some(err.to_string());
                            yield Err(err);
                            break 'outer;
                        }
                    };

                    // Record this turn's base system prompt — the patched-or-baseline
                    // preamble, before any output-mode augmentation the request builder
                    // appends. Borrow rather than clone since it only needs to outlive
                    // span creation.
                    let effective_preamble = request_patch
                        .as_ref()
                        .and_then(|o| o.preamble.as_deref())
                        .or(runner.config.preamble.as_deref());

                    let chat_span = source.open_chat_span(&runner, effective_preamble);

                    // Pin Tool output mode once committed so later turns stay
                    // consistent even if the per-turn tool set changes (#1928).
                    let committed_output_tool = run.output_tool_name().map(str::to_owned);
                    let mut prepared = match build_prepared_completion_request(
                        &runner,
                        &selected_model,
                        prompt.clone(),
                        &history,
                        committed_output_tool.as_deref(),
                        request_patch.as_ref(),
                    )
                    .await
                    {
                        Ok(prepared) => prepared,
                        Err(err) => {
                            store_error_usage(&runner, &run);
                            let err: StreamingError = err.into();
                            settled_error = Some(err.to_string());
                            yield Err(err);
                            break 'outer;
                        }
                    };
                    run.set_output_tool_name(prepared.output_tool_name.clone());
                    let turn_tool_snapshot = prepared.tool_snapshot.clone();
                    // What this request advertises becomes run data, so a
                    // resumed run or another driver can re-pair the calls
                    // that come back with the tools that were offered.
                    run.advertise_tools(turn, std::mem::take(&mut prepared.advertised_tools));
                    if runner.config.record_telemetry_content {
                        let input_messages = prepared.builder.messages_for_telemetry();
                        rig_core::telemetry::record_model_input(&chat_span, &input_messages, true);
                        prepared.builder = prepared.builder.record_content_telemetry(false);
                    }

                    // The attempt is now committed: advance `previous_model`
                    // immediately before the model turn is driven (the
                    // streaming request is issued on first poll of the turn
                    // stream). An issued attempt counts even when
                    // the provider returns an error; every stop/error path
                    // above left `previous_model` untouched.
                    previous_model = Some(selected_model);

                    drive_step!('outer, source.run_model_turn(
                        &runner,
                        &hook_ctx,
                        &mut run,
                        prepared,
                        chat_span,
                        &agent_span,
                        prompt,
                    ));
                    pending_tool_snapshot = Some(turn_tool_snapshot);
                }
                AgentRunStep::CallTools { calls } => {
                    let Some(tool_snapshot) = pending_tool_snapshot.take() else {
                        store_error_usage(&runner, &run);
                        let err = StreamingError::Completion(CompletionError::ResponseError(
                            "agent requested tool execution without a prepared registry snapshot"
                                .to_string(),
                        ));
                        settled_error = Some(err.to_string());
                        yield Err(err);
                        break 'outer;
                    };
                    drive_step!('outer, source.run_tool_calls(
                        &runner,
                        &hook_ctx,
                        &mut run,
                        calls,
                        tool_snapshot,
                    ));
                }
                AgentRunStep::Done(response) => {
                    flush_entries!();
                    // Run-completion marker, unifying the blocking and streaming
                    // drivers' run-finished logs into one shared event.
                    tracing::info!(
                        turn = run.turn(),
                        max_turns = runner.config.max_turns,
                        "Agent run finished"
                    );
                    source.record_run_level_telemetry(&agent_span, &response, created_agent_span);
                    append_run_messages(
                        memory_handle.as_ref(),
                        response.messages.as_deref().unwrap_or_default(),
                    )
                    .await;
                    // The run has settled successfully: nothing follows this
                    // response — the error endings settle after the loop.
                    if runner.config.hooks.observes(StepEventKind::RunSettled) {
                        runner
                            .config
                            .hooks
                            .on_run_settled(
                                &hook_ctx,
                                RunSettled {
                                    outcome: SettledOutcome::Response(&response),
                                },
                            )
                            .await;
                    }
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

        // Terminal settle for the error endings; the success ending settles in
        // the `Done` arm above, so exactly one settle fires per run.
        if let Some(reason) = settled_error
            && runner.config.hooks.observes(StepEventKind::RunSettled)
        {
            runner
                .config
                .hooks
                .on_run_settled(
                    &hook_ctx,
                    RunSettled {
                        outcome: SettledOutcome::Error(&reason),
                    },
                )
                .await;
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
    hook_ctx: &'a HookContext,
    run: &'a mut AgentRun,
    calls: Vec<PendingToolCall>,
    tool_snapshot: Arc<ToolRegistrySnapshot>,
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
        internal_call_id: InternalCallId,
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
        internal_call_id: InternalCallId,
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
            let internal_call_id = pending.internal_call_id.unwrap_or_else(rig_core::id::InternalCallId::new);
            let (span, preresolved_result) = match pending.preresolved_result {
                Some(result) => (tracing::Span::none(), Some(result)),
                None => {
                    if forward_items {
                        yield Ok(MultiTurnStreamItem::stream_item(
                            StreamedAssistantContent::ToolCall {
                                tool_call: pending.tool_call.clone(),
                                internal_call_id,
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

        {
            // Bounded by `tool_concurrency` (`0`/`1` poll strictly in call
            // order, giving sequential fail-fast). A shared `terminating`
            // flag makes a not-yet-started sibling skip (its side effect never
            // runs) once any sibling terminates — avoiding the Semantic-Kernel
            // fail-open — while already-in-flight siblings are drained so the
            // lowest call-index terminator wins and no task is left detached.
            let terminating = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let unordered = stream::iter(prepared.into_iter().enumerate())
                .map(|(index, call)| {
                    let PreparedToolCall { tool_call, preresolved_result, internal_call_id, span } = call;
                    let tool_snapshot = &tool_snapshot;
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
                            hook_ctx,
                            tool_snapshot,
                            &tool_call,
                            internal_call_id,
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
                .buffer_unordered(runner.concurrency.max(1));
            futures::pin_mut!(unordered);

            while let Some((index, outcome)) = unordered.next().await {
                // A dropped sibling records nothing.
                let Some(result) = outcome else { continue };
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
            let Some(CollectedToolResult { content, internal_call_id, surface }) = slot else {
                yield Err(StreamingError::Prompt(Box::new(PromptError::CompletionError(
                    CompletionError::ResponseError(
                        "tool execution finished without producing every result".to_string(),
                    ),
                ))));
                return;
            };
            if forward_items {
                // An executed call also surfaces its execution commit; a skipped
                // call surfaces only its result; a preresolved call surfaces
                // nothing here.
                let surface_result = match surface {
                    ToolSurface::Executed(tool_call) => {
                        surface_items.push(MultiTurnStreamItem::ToolExecutionCommitted {
                            tool_call: *tool_call,
                            internal_call_id,
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
    last_final_choice: Vec<AssistantContent>,
    last_message_id: Option<String>,
    /// Resolved agent name, kept only for the empty-turn diagnostic warning.
    agent_name: String,
    /// Whether we created the agent span (vs. adopting a caller's ambient span);
    /// gates recording `gen_ai.completion` onto it, matching the blocking source
    /// so neither surface pollutes a caller-supplied span.
    created_agent_span: bool,
    /// Whether sensitive run-level prompt and completion content may be recorded.
    record_telemetry_content: bool,
    /// Hot-path interest gates, computed once: skip building/dispatching the
    /// high-frequency delta events when no hook observes them.
    observes_text_delta: bool,
    observes_reasoning_delta: bool,
    observes_tool_call_delta: bool,
    /// Whether any hook is present — gates building the (history-cloning)
    /// invalid-tool diagnostic context.
    has_hooks: bool,
}

impl StreamingTurnSource {
    pub(crate) fn new(
        hooks: &HookStack,
        agent_name: String,
        created_agent_span: bool,
        record_telemetry_content: bool,
    ) -> Self {
        Self {
            // Nothing has streamed yet, so the last final choice is nothing.
            // This was a fabricated empty-text part for want of an empty
            // representation; `is_empty_assistant_turn` treated it as empty
            // anyway, so the two are equivalent — this one is just honest.
            last_final_choice: Vec::new(),
            last_message_id: None,
            agent_name,
            created_agent_span,
            record_telemetry_content,
            observes_text_delta: hooks.observes(StepEventKind::TextDelta),
            observes_reasoning_delta: hooks.observes(StepEventKind::ReasoningDelta),
            observes_tool_call_delta: hooks.observes(StepEventKind::ToolCallDelta),
            has_hooks: !hooks.is_empty(),
        }
    }

    /// Record a completed model turn's canonical output onto the agent and
    /// chat spans. Only self-created agent spans receive `gen_ai.completion`,
    /// so neither surface pollutes a caller-supplied span.
    fn record_turn_telemetry(
        &self,
        agent_span: &tracing::Span,
        chat_span: &tracing::Span,
        choice: &[AssistantContent],
        record_content: bool,
    ) {
        if self.created_agent_span && self.record_telemetry_content {
            agent_span.record("gen_ai.completion", assistant_text_from_choice(choice));
        }
        rig_core::telemetry::record_model_output(chat_span, choice, record_content);
    }
}

impl TurnSource for StreamingTurnSource {
    fn open_chat_span(
        &self,
        runner: &AgentRunner,
        effective_preamble: Option<&str>,
    ) -> tracing::Span {
        build_chat_span!(runner, effective_preamble, "chat_streaming", "chat")
    }

    fn run_model_turn<'a>(
        &'a mut self,
        runner: &'a AgentRunner,
        hook_ctx: &'a HookContext,
        run: &'a mut AgentRun,
        prepared: PreparedCompletionRequest,
        chat_span: tracing::Span,
        agent_span: &'a tracing::Span,
        current_prompt: Message,
    ) -> DriveStream<'a> {
        Box::pin(async_stream::stream! {
            // Bound before the builder is consumed, exactly as the blocking
            // surface does: the cap this attempt was prepared with, patches
            // included. Both surfaces read it from the same carrier, so they
            // cannot report different numbers for the same attempt.
            let attempt_max_tokens = prepared.max_tokens;

            let mut stream = match prepared
                .builder
                .stream()
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
            // A turn whose invalid tool call was repaired is a recovered turn:
            // neither the response hook nor `ModelTurnFinished` fires for it.
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
                    // Same source as identity below: the provider's terminal
                    // record. A path that never saw one yields `None`, which is
                    // "the provider reported no reason" — not "the turn stopped
                    // normally".
                    let reason = stream
                        .response
                        .as_ref()
                        .and_then(|response| response.finish_reason.clone());
                    emit_completion_call!($usage, reason)
                }};
                ($usage:expr, $finish_reason:expr) => {{
                    let usage = $usage;
                    last_usage = usage;
                    if !completion_call_emitted {
                        chat_span.record_token_usage(&usage);
                        // The terminal record (when the provider delivered
                        // one) carries this attempt's identity metadata — and
                        // its captured raw payload, read from the same
                        // terminal so the recorded call carries *this*
                        // attempt's response, never a previous attempt's.
                        match run.record_streamed_completion_call(
                            usage,
                            stream.identity(),
                            $finish_reason,
                            stream
                                .response
                                .as_ref()
                                .map_or(serde_json::Value::Null, |response| response.raw.clone()),
                        ) {
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
                            if self.observes_text_delta
                                && let Some(StreamedAssistantContent::Text(text)) =
                                    item_slot.as_ref()
                                && let Some(reason) = observe_action(
                                    runner
                                        .config.hooks
                                        .on_text_delta(
                                            hook_ctx,
                                            TextDelta {
                                                delta: &text.text,
                                                aggregated: assembler.aggregated_text(),
                                            },
                                        )
                                        .await,
                                )
                            {
                                yield Err(StreamingError::Prompt(Box::new(
                                    run.cancel_error(reason),
                                )));
                                return;
                            }
                            if self.observes_reasoning_delta
                                && let Some(StreamedAssistantContent::ReasoningDelta {
                                    id,
                                    provider_id,
                                    reasoning,
                                }) = item_slot.as_ref()
                            {
                                let Some(aggregated) = assembler.aggregated_reasoning(id) else {
                                    yield Err(CompletionError::ResponseError(format!(
                                        "reasoning delta `{id}` was ingested without a pending aggregate"
                                    ))
                                    .into());
                                    return;
                                };
                                if let Some(reason) = observe_action(
                                    runner
                                        .config.hooks
                                        .on_reasoning_delta(
                                            hook_ctx,
                                            ReasoningDelta {
                                                id,
                                                provider_id: provider_id.as_deref(),
                                                delta: reasoning,
                                                aggregated,
                                            },
                                        )
                                        .await,
                                ) {
                                    yield Err(StreamingError::Prompt(Box::new(
                                        run.cancel_error(reason),
                                    )));
                                    return;
                                }
                            }
                            if let Some(item) = item_slot.take() {
                                yield Ok(MultiTurnStreamItem::stream_item(item));
                            }
                        }
                        StreamedTurnEvent::EmitToolCallDelta {
                            internal_call_id,
                            content,
                        } => {
                            if self.observes_tool_call_delta {
                                let (delta_name, delta_text) = match &content {
                                    ToolCallDeltaContent::Name(name) => (Some(name.as_str()), ""),
                                    ToolCallDeltaContent::Delta(delta) => (None, delta.as_str()),
                                };
                                if let Some(reason) = observe_action(
                                    runner
                                        .config.hooks
                                        .on_tool_call_delta(
                                            hook_ctx,
                                            ToolCallDelta {
                                                internal_call_id,
                                                tool_name: delta_name,
                                                delta: delta_text,
                                            },
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
                                    internal_call_id,
                                    content,
                                },
                            ));
                        }
                        StreamedTurnEvent::Completed {
                            usage,
                            emit_final,
                            finish_reason,
                        } => {
                            match emit_completion_call!(usage, finish_reason) {
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
                                    .config.hooks
                                    .on_invalid_tool_call(hook_ctx, &context)
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
                                    // handling above; the turn is now recovered.
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
                                                tool_result: *tool_result,
                                                internal_call_id: invalid.internal_call_id,
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

            // The provider stream ended without its terminal record. Per the
            // emission contract (`rig_core::streaming`), that absence means
            // truncation and must never be treated as a successful zero-usage
            // completion: reject the turn before any usage fallback, assembly,
            // history mutation, or tool dispatch can occur.
            if !provider_final_seen {
                yield Err(CompletionError::ResponseError(
                    "provider stream ended without a terminal record; treating the turn as truncated"
                        .to_string(),
                )
                .into());
                return;
            }

            if let Some(err) = assembler.pending_delta_error() {
                yield Err(err.into());
                return;
            }

            // Final fallback: no usage was ever learned, so there is nothing to
            // record onto the span (zero usage is the missing-metrics sentinel)
            // and this is the last read of the flag — kept inline (not
            // `emit_completion_call!`) so it doesn't emit a dead
            // `completion_call_emitted = true` write, which `unused_assignments`
            // rejects. Identity comes from the same accessor the macro uses, so
            // `completion_calls` and hook observations agree on this path too.
            if !completion_call_emitted {
                let fallback_finish_reason = stream
                    .response
                    .as_ref()
                    .and_then(|response| response.finish_reason.clone());
                match run.record_streamed_completion_call(
                    crate::completion::Usage::new(),
                    stream.identity(),
                    fallback_finish_reason,
                    stream
                        .response
                        .as_ref()
                        .map_or(serde_json::Value::Null, |response| response.raw.clone()),
                ) {
                    Ok(call) => yield Ok(MultiTurnStreamItem::CompletionCall(call)),
                    Err(err) => {
                        yield Err(Box::new(err).into());
                        return;
                    }
                }
            }

            let final_turn_content = stream.choice.clone();
            let streamed_turn = assembler.finish(stream.message_id.clone(), &final_turn_content);
            // This attempt's identity, read from *this* stream's terminal
            // record (each attempt — including a retry — opens its own
            // stream, so a previous attempt's ids can never leak in). The
            // message id prefers the assembled turn's, which folds in an
            // explicit `MessageId` event; the terminal's ids fill the rest.
            let identity = rig_core::completion::ResponseIdentity {
                message_id: streamed_turn.message_id.clone(),
                ..stream.identity()
            };
            // This attempt's raw payload, from the same terminal record as the
            // identity above — so a retry never observes a previous attempt's
            // response. `Null` when no terminal record arrived.
            let attempt_raw = stream
                .response
                .as_ref()
                .map_or(&serde_json::Value::Null, |response| &response.raw);
            if !turn_recovered
                && let Some(reason) = observe_action(
                    runner
                        .config.hooks
                        .on_completion_response(
                            hook_ctx,
                            CompletionResponseEvent {
                                prompt: &current_prompt,
                                content: &streamed_turn.choice,
                                usage: last_usage,
                                identity: &identity,
                                raw: attempt_raw,
                            },
                        )
                        .await,
                )
            {
                yield Err(StreamingError::Prompt(Box::new(run.cancel_error(reason))));
                return;
            }
            self.last_message_id.clone_from(&streamed_turn.message_id);
            // The canonical assistant content: `finish` normalizes
            // reasoning/text/tool ordering, so this can differ from the raw
            // `stream.choice` aggregate. `ModelTurnFinished` — the normalized
            // per-turn event — carries this, matching what is recorded into run
            // history; the raw `stream.choice` is kept in `last_final_choice` for
            // the raw/final streaming behavior.
            let canonical_choice = streamed_turn.choice.clone();
            // Captured for the same reason as the choice above: `streamed_turn`
            // is moved into run state on the next line, and the per-turn hook
            // fires after that. `FinishReason::Other` carries a `String`, so
            // this is a clone rather than a copy.
            let attempt_finish_reason = streamed_turn.finish_reason.clone();
            if let Err(err) = run.streamed_turn(streamed_turn) {
                yield Err(Box::new(err).into());
                return;
            }
            // Normalized per-turn event, fired once the turn is parked for
            // acceptance. Suppressed for recovered turns.
            if !turn_recovered {
                let action = runner
                    .config.hooks
                    .on_model_turn_finished(
                        hook_ctx,
                        ModelTurnFinished {
                            turn: hook_ctx.turn(),
                            content: &canonical_choice,
                            usage: last_usage,
                            identity: &identity,
                            finish_reason: attempt_finish_reason.as_ref(),
                            max_tokens: attempt_max_tokens,
                            raw: attempt_raw,
                        },
                    )
                    .await;
                match resolve_model_turn_action(run, action) {
                    Ok(ModelTurnDecision::Advance) => {}
                    Ok(ModelTurnDecision::Retried) => {
                        yield Ok(MultiTurnStreamItem::ModelTurnRetried {
                            turn: hook_ctx.turn(),
                        });
                        return;
                    }
                    Ok(ModelTurnDecision::Terminate(reason)) => {
                        // Before model-turn steering was added, Stop observed
                        // this already completed provider turn: its buffered
                        // final and content telemetry were visible before the
                        // cancellation. Preserve that behavior while Retry
                        // alone suppresses the provisional final.
                        self.record_turn_telemetry(
                            agent_span,
                            &chat_span,
                            &canonical_choice,
                            runner.config.record_telemetry_content,
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
            self.record_turn_telemetry(
                agent_span,
                &chat_span,
                &canonical_choice,
                runner.config.record_telemetry_content,
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
        hook_ctx: &'a HookContext,
        run: &'a mut AgentRun,
        calls: Vec<PendingToolCall>,
        tool_snapshot: Arc<ToolRegistrySnapshot>,
    ) -> DriveStream<'a> {
        // The streaming surface chains nothing onto its tool spans, and forwards
        // the ToolCall/ToolResult items to the consumer.
        drive_tool_calls(
            runner,
            hook_ctx,
            run,
            calls,
            tool_snapshot,
            |span| span,
            true,
        )
    }

    fn record_run_level_telemetry(
        &self,
        agent_span: &tracing::Span,
        response: &PromptResponse,
        created_agent_span: bool,
    ) {
        if created_agent_span {
            agent_span.record_token_usage(&response.usage);
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
    /// text and tool-call deltas. Returns the stream after loading any
    /// configured conversation memory.
    ///
    /// Shares the drive loop, run construction, tool execution and fail-closed
    /// hook handling with the blocking [`run`](AgentRunner::run) via
    /// `drive_agent`, so the two behave identically apart from the streamed
    /// delta events.
    pub async fn stream(self) -> StreamingResult {
        let (agent_span, created_agent_span) = self.open_agent_span();

        let (history_override, memory_handle) = match self.resolve_history_and_memory().await {
            Ok(resolved) => resolved,
            Err(err) => {
                let stream = async_stream::stream! {
                    yield Err(StreamingError::from(err));
                };
                // Instrument under the agent span like the success path so
                // a load failure stays tied to invoke_agent.
                return Box::pin(stream.instrument(agent_span));
            }
        };

        let run = self.build_run(history_override);
        let source = StreamingTurnSource::new(
            &self.config.hooks,
            self.agent_name_or_default().to_string(),
            created_agent_span,
            self.config.record_telemetry_content,
        );

        // The blocking surface folds this same engine; the streaming surface
        // forwards intermediate items (the final response item is the last one)
        // and ends on `Done`.
        let driver = drive_agent(
            self,
            source,
            run,
            agent_span.clone(),
            created_agent_span,
            memory_handle,
            true,
        )
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

/// Capacity of the event queue behind [`AgentRunner::run_channel`]: the number
/// of [`MultiTurnStreamItem`]s the run may buffer ahead of the consumer before
/// it parks on back-pressure.
pub const RUN_EVENTS_CAPACITY: usize = 32;

/// Event feed of an agent run started with [`AgentRunner::run_channel`] or
/// [`Agent::run_channel`].
///
/// Every [`MultiTurnStreamItem`] the run would have streamed is delivered here
/// in order, ending with [`MultiTurnStreamItem::FinalResponse`]. The feed is a
/// bounded queue ([`RUN_EVENTS_CAPACITY`]): a slow consumer applies
/// back-pressure to the run instead of losing events. Poll it as a
/// [`Stream`] from async code, or drain it with the non-blocking
/// [`try_next`](RunEvents::try_next) from a synchronous tick — a game loop, a
/// UI frame, an ECS system.
///
/// Dropping the feed does not cancel the run; it simply stops receiving events
/// and the run future still resolves with the final
/// [`PromptResponse`].
#[derive(Debug)]
pub struct RunEvents {
    receiver: mpsc::Receiver<MultiTurnStreamItem>,
}

impl RunEvents {
    /// Take the next buffered event without waiting.
    ///
    /// Returns `None` both when no event is queued yet and once the run has
    /// finished and the feed is drained; use [`is_done`](RunEvents::is_done)
    /// to tell the two apart.
    pub fn try_next(&mut self) -> Option<MultiTurnStreamItem> {
        self.receiver.try_recv().ok()
    }

    /// Whether the run has finished and every event has been taken.
    pub fn is_done(&self) -> bool {
        self.receiver.is_terminated()
    }
}

impl Stream for RunEvents {
    type Item = MultiTurnStreamItem;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        Pin::new(&mut self.receiver).poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.receiver.size_hint()
    }
}

impl FusedStream for RunEvents {
    fn is_terminated(&self) -> bool {
        self.receiver.is_terminated()
    }
}

impl AgentRunner {
    /// Split the run into a driving future and a [`RunEvents`] feed.
    ///
    /// The future performs the whole agent loop — the same engine as
    /// [`run`](AgentRunner::run) and [`stream`](AgentRunner::stream) — and
    /// resolves with the final [`PromptResponse`]; the
    /// feed receives each intermediate [`MultiTurnStreamItem`] as it happens.
    /// Spawn the future on any executor and poll the feed from wherever the
    /// events are consumed; neither side assumes a runtime.
    ///
    /// The feed is bounded ([`RUN_EVENTS_CAPACITY`]); when it is full the run
    /// waits for the consumer rather than dropping events. Dropping the feed
    /// lets the run continue to completion unobserved.
    pub fn run_channel(
        self,
    ) -> (
        impl Future<Output = Result<PromptResponse, PromptError>> + WasmCompatSend,
        RunEvents,
    ) {
        let (mut sender, receiver) = mpsc::channel(RUN_EVENTS_CAPACITY);
        let future = async move {
            let mut stream = self.stream().await;
            let mut response = None;
            let mut forward = true;
            while let Some(item) = stream.next().await {
                let item = item.map_err(streaming_error_into_prompt)?;
                match item {
                    MultiTurnStreamItem::FinalResponse(done) => {
                        if forward {
                            // The consumer is gone; nothing left to forward.
                            let _ = sender
                                .send(MultiTurnStreamItem::FinalResponse(done.clone()))
                                .await;
                        }
                        response = Some(done);
                    }
                    item => {
                        if forward && sender.send(item).await.is_err() {
                            forward = false;
                        }
                    }
                }
            }
            response.ok_or_else(|| {
                PromptError::CompletionError(CompletionError::ResponseError(
                    "agent run ended without producing a final response".to_string(),
                ))
            })
        };
        (future, RunEvents { receiver })
    }
}

impl Agent {
    /// Run `prompt` with the agent's defaults, returning the driving future and
    /// a [`RunEvents`] feed. See [`AgentRunner::run_channel`]; to configure the
    /// run first (history, turn budget, tool context, …), configure a
    /// [`StreamingPromptRequest`] and call its
    /// [`run_channel`](StreamingPromptRequest::run_channel).
    pub fn run_channel<P: Into<Message> + WasmCompatSend>(
        &self,
        prompt: P,
    ) -> (
        impl Future<Output = Result<PromptResponse, PromptError>> + WasmCompatSend + use<P>,
        RunEvents,
    ) {
        AgentRunner::from_agent(self, prompt).run_channel()
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
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Reasoning {
                reasoning,
                ..
            })) => {
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
mod migrated_tests;
