//! The agent engine: one drive loop for both surfaces, plus the two
//! [`TurnSource`]s that fetch a model turn for it.
//!
//! [`drive_agent`] owns everything that does not depend on the medium — step
//! dispatch over the sans-IO [`AgentRun`], the run-start and completion-call
//! hooks, request preparation, model selection, the `Done` memory append.
//! Per turn it hands the prepared request to a [`TurnSource`]:
//! [`UnaryTurnSource`] issues one `completion()` call (the blocking
//! [`AgentRunner::run`]), [`StreamingTurnSource`] opens a provider stream and
//! drives a [`StreamedTurnAssembler`] (the streaming [`AgentRunner::stream`]).
//! Both hand the assembled response to [`settle_model_turn`], which fires the
//! response and model-turn hooks and applies the resulting action, so the two
//! surfaces cannot disagree on how a turn is accepted, retried, or stopped.
//! Tool execution is likewise shared: [`drive_tool_calls`] runs a turn's calls
//! through [`run_single_tool`].
//!
//! The blocking surface folds this engine to its final response; the streaming
//! surface forwards its [`DriveItem`]s.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::{collections::VecDeque, pin::Pin};

use futures::{Stream, StreamExt, stream};
use tracing::{Instrument, span::Id};

use rig_core::{
    completion::{FinishReason, ResponseIdentity},
    memory::ConversationMemory,
    message::{AssistantContent, Message, ToolCall, UserContent},
    streaming::BlockId,
    telemetry::SpanCombinator,
    wasm_compat::WasmCompatSend,
};

use super::{
    ModelHandle,
    completion::{PreparedCompletionRequest, build_prepared_completion_request},
    hook::{
        AgentHook, CompletionCall, CompletionCallAction,
        CompletionResponse as CompletionResponseEvent, HookContext, HookStack,
        InvalidToolCallAction, ModelSelection, ModelSelectionAction, ModelTurnAction,
        ModelTurnFinished, ObservationAction, ReasoningDelta, RequestPatch, RunSettled, RunStart,
        RunStartAction, SettledOutcome, StepEventKind, TextDelta, ToolCall as ToolCallEvent,
        ToolCallAction, ToolCallDelta, ToolResultAction, ToolResultEvent,
    },
    run::{
        AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome, PendingToolCall,
        streamed::{StreamedResolution, StreamedTurnAssembler, StreamedTurnEvent},
    },
    run::{
        response::PromptResponse,
        transcript::{assistant_text_from_choice, is_empty_assistant_turn, tool_result_output},
    },
    runner::AgentRunner,
    streaming::{
        MultiTurnStreamItem, StreamingError, drain_stream_usage, finalize_streamed_choice,
    },
    telemetry::{build_chat_span, new_execute_tool_span},
};
use crate::{
    completion::{CompletionError, PromptError, Usage},
    json_utils,
    streaming::{StreamedAssistantContent, StreamedUserContent, ToolCallDeltaContent},
    tool::{ToolDispatch, ToolResult, server::ToolRegistrySnapshot},
};

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
    // Per-call working state: a stable block_id and the execute span,
    // paired with the model's tool call. `span` is `Span::none()` for a
    // preresolved (invalid-recovery) call, which never executes.
    struct PreparedToolCall {
        tool_call: rig_core::message::ToolCall,
        preresolved_result: Option<UserContent>,
        block_id: BlockId,
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
        block_id: BlockId,
        surface: ToolSurface,
    }

    Box::pin(async_stream::stream! {
        let full_history_for_errors = run.full_history();
        let call_count = calls.len();

        // Assign each call a stable block_id and, for calls that will
        // actually execute, an execute span. Emit the MODEL tool-call events now,
        // right after the turn committed: these report what the model emitted and
        // are *not* execution-lifecycle events. A preresolved call emits no model
        // tool-call event (its synthetic result was already surfaced during the
        // model turn) and gets no execute span.
        let mut prepared: Vec<PreparedToolCall> = Vec::with_capacity(call_count);
        for pending in calls {
            let block_id = pending
                .block_id
                .unwrap_or_else(|| BlockId::wire(pending.tool_call.id.as_str()));
            let (span, preresolved_result) = match pending.preresolved_result {
                Some(result) => (tracing::Span::none(), Some(result)),
                None => {
                    if forward_items {
                        yield Ok(MultiTurnStreamItem::stream_item(
                            StreamedAssistantContent::ToolCall {
                                tool_call: pending.tool_call.clone(),
                                id: block_id.clone(),
                            },
                        ));
                    }
                    (chain_tool_span(new_execute_tool_span()), None)
                }
            };
            prepared.push(PreparedToolCall {
                tool_call: pending.tool_call,
                preresolved_result,
                block_id,
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
                    let PreparedToolCall { tool_call, preresolved_result, block_id, span } = call;
                    let tool_snapshot = &tool_snapshot;
                    let full_history_for_errors = &full_history_for_errors;
                    let terminating = terminating.clone();
                    async move {
                        if let Some(result) = preresolved_result {
                            return (
                                index,
                                Some(Ok(CollectedToolResult {
                                    content: result,
                                    block_id,
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
                            &block_id,
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
                                block_id,
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
            let Some(CollectedToolResult { content, block_id, surface }) = slot else {
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
                            block_id: block_id.clone(),
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
                            id: block_id,
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
            // Bound before the builder is consumed: the cap this attempt was
            // prepared with, completion-call patches included.
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
                            block_id,
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
                                                block_id: &block_id,
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
                                    id: block_id,
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
                            // fails fast.
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
                                                id: invalid.block_id.clone(),
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
            self.last_message_id.clone_from(&streamed_turn.message_id);
            // The canonical assistant content: `finish` normalizes
            // reasoning/text/tool ordering, so this can differ from the raw
            // `stream.choice` aggregate. The hooks and run history see this;
            // the raw `stream.choice` is kept in `last_final_choice` for the
            // raw/final streaming behavior.
            let canonical_choice = streamed_turn.choice.clone();
            // `streamed_turn` is moved into run state on the next line and the
            // hooks fire after that. `FinishReason::Other` carries a `String`,
            // so this is a clone rather than a copy.
            let attempt_finish_reason = streamed_turn.finish_reason.clone();
            if let Err(err) = run.streamed_turn(streamed_turn) {
                yield Err(Box::new(err).into());
                return;
            }
            if !turn_recovered {
                let settlement = settle_model_turn(
                    &runner.config.hooks,
                    hook_ctx,
                    run,
                    AssembledTurn {
                        prompt: &current_prompt,
                        content: &canonical_choice,
                        usage: last_usage,
                        identity: &identity,
                        finish_reason: attempt_finish_reason.as_ref(),
                        max_tokens: attempt_max_tokens,
                        raw: attempt_raw,
                    },
                )
                .await;
                match settlement {
                    Ok(ModelTurnDecision::Advance) => {}
                    Ok(ModelTurnDecision::Retried) => {
                        yield Ok(MultiTurnStreamItem::ModelTurnRetried {
                            turn: hook_ctx.turn(),
                        });
                        return;
                    }
                    Ok(ModelTurnDecision::Terminate(reason)) => {
                        // A stop observes an already completed provider turn:
                        // its buffered final and content telemetry stay
                        // visible before the cancellation. Retry alone
                        // suppresses the provisional final.
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
        // Always surface the accumulated messages, regardless of whether the
        // caller supplied input history.
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

/// One attempt's assembled response, as both drivers hand it to
/// [`settle_model_turn`]: the unary response under `run()`, the finished
/// stream under `stream()`. Every field is this attempt's own — a retry
/// re-enters with a fresh response, so a stale attempt's ids or payload can
/// never be attributed here.
pub(crate) struct AssembledTurn<'a> {
    pub(crate) prompt: &'a Message,
    pub(crate) content: &'a Vec<AssistantContent>,
    pub(crate) usage: Usage,
    pub(crate) identity: &'a ResponseIdentity,
    pub(crate) finish_reason: Option<&'a FinishReason>,
    /// The cap this attempt was prepared with, completion-call patches
    /// included; read off the prepared request, never the agent config.
    pub(crate) max_tokens: Option<u64>,
    pub(crate) raw: &'a serde_json::Value,
}

/// Settle a parked model turn: fire [`AgentHook::on_completion_response`]
/// (observe-only; a stop terminates), then
/// [`AgentHook::on_model_turn_finished`] and apply its action to the sans-IO
/// run. Both drivers call this once per accepted attempt, so retry history,
/// tool-turn rejection, and state transitions cannot diverge by medium. The
/// callers own what happens next: the blocking driver records the accepted
/// turn's telemetry; the streaming driver additionally surfaces or discards
/// the buffered provisional `Final`.
pub(crate) async fn settle_model_turn(
    hooks: &HookStack,
    hook_ctx: &HookContext,
    run: &mut AgentRun,
    turn: AssembledTurn<'_>,
) -> Result<ModelTurnDecision, PromptError> {
    if let Some(reason) = observe_action(
        hooks
            .on_completion_response(
                hook_ctx,
                CompletionResponseEvent {
                    prompt: turn.prompt,
                    content: turn.content,
                    usage: turn.usage,
                    identity: turn.identity,
                    raw: turn.raw,
                },
            )
            .await,
    ) {
        return Ok(ModelTurnDecision::Terminate(reason));
    }
    let action = hooks
        .on_model_turn_finished(
            hook_ctx,
            ModelTurnFinished {
                turn: hook_ctx.turn(),
                content: turn.content,
                usage: turn.usage,
                identity: turn.identity,
                finish_reason: turn.finish_reason,
                max_tokens: turn.max_tokens,
                raw: turn.raw,
            },
        )
        .await;
    match action {
        ModelTurnAction::Continue => Ok(ModelTurnDecision::Advance),
        ModelTurnAction::Retry(request) => {
            run.retry_model_turn(request)?;
            Ok(ModelTurnDecision::Retried)
        }
        ModelTurnAction::Stop(reason) => Ok(ModelTurnDecision::Terminate(reason)),
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
    /// [`ToolExecutionCommitted`](crate::agent::streaming::MultiTurnStreamItem::ToolExecutionCommitted)
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
    block_id: &BlockId,
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
                block_id,
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
                block_id,
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
                            None => run.resolve_unhandled_invalid_tool_call(),
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
                            let identity = resp.identity();
                            let settlement = settle_model_turn(
                                &runner.config.hooks,
                                hook_ctx,
                                run,
                                AssembledTurn {
                                    prompt: &current_prompt,
                                    content: &resp.choice,
                                    usage: resp.usage,
                                    identity: &identity,
                                    finish_reason: attempt_finish_reason.as_ref(),
                                    max_tokens: attempt_max_tokens,
                                    raw: &resp.raw,
                                },
                            )
                            .await;
                            match settlement {
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

#[cfg(test)]
#[allow(irrefutable_let_patterns, unreachable_patterns)]
mod tests;
