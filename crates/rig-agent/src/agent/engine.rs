//! Shared agent drive loop, tool dispatch, and turn settlement. Unary and
//! streaming [`TurnSource`] implementations supply model responses; the engine
//! applies lifecycle policy and advances the sans-I/O [`AgentRun`].

use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use futures::{Stream, StreamExt, stream};
use tracing::{Instrument, span::Id};

use crate::bus::{DispatchOptions, MemoryHandle};
use rig_core::error::ProviderError;
use rig_core::{
    completion::ModelRef,
    effect::{EffectId, EffectKind, Outcome},
    error::{ErrorKind, ErrorReport},
    message::{AssistantContent, Message, ToolCall, ToolFunction, UserContent},
    telemetry::SpanCombinator,
    wasm_compat::{WasmBoxedStream, WasmCompatSend, WasmCompatSync},
};

use super::{
    ModelHandle,
    completion::{PreparedCompletionRequest, build_prepared_completion_request},
    hook::{
        AgentHook, CompletionCallAction, CompletionCallEvent, DispatchAction, DispatchEvent,
        HookContext, HookStack, InvalidToolCallAction, ModelSelection, ModelSelectionAction,
        ModelTurnAction, ModelTurnFinished, ObservationAction, OutcomeAction, OutcomeEvent,
        ReasoningDelta, RequestPatch, RunSettled, RunStart, RunStartAction, SettledOutcome,
        StepEventKind, TextDelta, ToolCallDelta,
    },
    run::{
        AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome, PendingToolCall,
        streamed::{StreamedResolution, StreamedTurnAssembler, StreamedTurnEvent},
    },
    run::{
        response::{MemoryAppend, PromptResponse, finalize_output_tool_choice},
        transcript::{
            assistant_text_from_choice, is_empty_assistant_turn, tool_result_message,
            tool_result_output,
        },
    },
    runner::AgentRunner,
    streaming::MultiTurnStreamItem,
    telemetry::{build_chat_span, new_execute_tool_span},
};
use crate::run::UnhandledInvalidToolCall;
use crate::{
    completion::PromptError,
    streaming::{Item, StreamEvent},
    tool::{ToolCatalog, ToolResult},
};

/// A boxed, medium-specific item stream for one engine step (model turn or tool
/// batch). Boxed so a generic [`drive_agent`] can forward it without the
/// per-step future leaking into the engine's own (`Send`) inference.
pub(crate) type DriveStream<'a> = WasmBoxedStream<'a, Result<MultiTurnStreamItem, PromptError>>;

/// Engine output: stream items for forwarding or a canonical terminal response.
pub(crate) enum DriveItem {
    /// Stream event, including the streaming surface's final response item.
    Item(MultiTurnStreamItem),
    /// The run finished; carries the canonical response the blocking fold
    /// returns. The streaming surface has already received the final item as the
    /// preceding `Item` and ignores this.
    Done(PromptResponse),
}

/// Medium-specific model turns, span chaining, telemetry, and final-item
/// construction. Implementations resolve invalid calls during model ingestion
/// and feed accepted turns back into the run; the engine runs tool calls.
pub(crate) trait TurnSource: WasmCompatSend + WasmCompatSync {
    /// Whether the surface forwards intermediate items. The blocking fold
    /// discards them, so its source skips building them.
    const FORWARDS_ITEMS: bool;

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

    /// Chain a chat or tool execute span into this medium's span sequence.
    fn chain_span(&self, span: tracing::Span) -> tracing::Span;

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

pub(crate) fn store_error_usage(runner: &AgentRunner, run: &AgentRun) {
    if let Some(usage) = &runner.error_usage {
        *usage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = run.usage();
    }
}

/// Drive run steps, hooks, request preparation, and memory append using `source`.
/// Yields stream items and a terminal response, or settles and yields an error.
/// Only engine-owned agent spans receive run-level telemetry.
pub(crate) fn drive_agent<S>(
    runner: AgentRunner,
    mut source: S,
    mut run: AgentRun,
    agent_span: tracing::Span,
    created_agent_span: bool,
    memory_handle: Option<(MemoryHandle, rig_core::id::ConversationId)>,
    hook_ctx: HookContext,
) -> impl Stream<Item = Result<DriveItem, PromptError>>
where
    S: TurnSource,
{
    async_stream::stream! {
        // Hooks must see resumed entries from their first invocation.
        hook_ctx.seed_entries(run.entries());
        // Step-boundary flushing includes hook state in serialization; settlement
        // appends occur after the final flush and remain local.
        macro_rules! flush_entries {
            () => {
                for entry in hook_ctx.drain_pending_entries() {
                    run.append_entry(entry);
                }
            };
        }
        // Settle before yielding an error because consumers need not poll again.
        macro_rules! settle_error {
            ($err:expr) => {{
                if runner.config.hooks.observes(StepEventKind::RunSettled) {
                    let reason = $err.to_string();
                    runner
                        .config
                        .hooks
                        .on_run_settled(
                            &hook_ctx,
                            RunSettled {
                                outcome: SettledOutcome::Error(&reason),
                                messages: Some(run.messages()),
                            },
                        )
                        .await;
                }
            }};
        }
        // Set only after a model turn commits successfully and consumed by its
        // immediately following CallTools step. This keeps the sans-IO run state
        // serializable while pinning execution to the definitions sent that turn.
        let mut pending_tool_snapshot: Option<Arc<ToolCatalog>> = None;
        // Restore routing history so resumed hooks see the last issued model.
        let mut previous_model: Option<ModelRef> = run.previous_model().cloned();

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
                    .err(),
                RunStartAction::Stop(reason) => {
                    Some(run.cancel_error(reason))
                }
            };
            if let Some(err) = early_stop {
                store_error_usage(&runner, &run);
                settle_error!(err);
                yield Err(err);
                return;
            }
        }

        // A macro keeps yield and loop termination in the caller's scope;
        // macro hygiene requires passing the loop label explicitly.
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
                    settle_error!(err);
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
                    settle_error!(err);
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

                    // Selection must see merged request patches and must not run after a stop.
                    let request_patch =
                        match resolve_completion_call(&runner.config.hooks, &hook_ctx, &prompt, &history, turn).await {
                            CompletionCallOutcome::Terminate(reason) => {
                                store_error_usage(&runner, &run);
                                let err = run.cancel_error(reason);
                                settle_error!(err);
                                yield Err(err);
                                break 'outer;
                            }
                            CompletionCallOutcome::Proceed(request_patch) => request_patch,
                        };

                    // Preparation and execution must use the same selected model's capabilities.
                    let default_label = runner.config.model_label();
                    let selected_label = match runner.config.hooks.on_model_select(
                        &hook_ctx,
                        ModelSelection {
                            prompt: &prompt,
                            history: &history,
                            request_patch: request_patch.as_ref(),
                            previous_model: previous_model.as_ref(),
                            default_model: &default_label,
                            selected_model: &default_label,
                        },
                    ) {
                        ModelSelectionAction::Continue => default_label.clone(),
                        ModelSelectionAction::Select(model) => model,
                        ModelSelectionAction::Stop(reason) => {
                            store_error_usage(&runner, &run);
                            let err = run.cancel_error(reason);
                            settle_error!(err);
                            yield Err(err);
                            break 'outer;
                        }
                    };
                    // Bind the typed view now: an unregistered label is a
                    // wiring error, surfaced before any request is built.
                    let selected_model = if selected_label == default_label {
                        runner.config.model_handle()
                    } else {
                        runner.config.model_by_ref(&selected_label)
                    };
                    let selected_model: ModelHandle = match selected_model {
                        Ok(model) => model,
                        Err(report) => {
                            store_error_usage(&runner, &run);
                            let err = PromptError::Report(report);
                            settle_error!(err);
                            yield Err(err);
                            break 'outer;
                        }
                    };

                    // Telemetry uses the effective preamble before output-mode augmentation.
                    let effective_preamble = request_patch
                        .as_ref()
                        .and_then(|o| o.preamble.as_deref())
                        .or(runner.config.preamble.as_deref());

                    let chat_span = source.open_chat_span(&runner, effective_preamble);

                    // Pin output mode across registry changes between turns.
                    let committed_output_tool = run.output_tool_name().map(str::to_owned);
                    let mut prepared = match build_prepared_completion_request(
                        &runner,
                        &hook_ctx,
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
                            let err = PromptError::from(err);
                            settle_error!(err);
                            yield Err(err);
                            break 'outer;
                        }
                    };
                    if let Some(name) = &prepared.output_tool_name {
                        // Refused after the first turn by design: the name is pinned.
                        let _ = run.commit_output_tool_name(name.clone());
                    }
                    let turn_tool_snapshot = prepared.tool_snapshot.clone();
                    // What this request advertises becomes run data, so a
                    // resumed run or another driver can re-pair the calls
                    // that come back with the tools that were offered.
                    run.advertise_tools(turn, std::mem::take(&mut prepared.advertised_tools));
                    if runner.config.record_telemetry_content {
                        let input_messages = std::mem::take(&mut prepared.telemetry_messages);
                        rig_core::telemetry::record_model_input(&chat_span, &input_messages, true);
                    }

                    // Only issued attempts advance routing history, including provider errors.
                    run.set_previous_model(selected_label.clone());
                    previous_model = Some(selected_label);

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
                    // Resume cannot restore registration leases; bind advertised names
                    // to this process's current implementations.
                    if pending_tool_snapshot.is_none()
                        && let Some(advertised) = run.advertised_tools()
                    {
                        let names: Vec<String> = advertised
                            .definitions
                            .iter()
                            .map(|definition| definition.name.to_string())
                            .collect();
                        pending_tool_snapshot = Some(Arc::new(
                            runner.tool_server_handle.snapshot_with_dynamic(&names),
                        ));
                    }
                    let Some(tool_snapshot) = pending_tool_snapshot.take() else {
                        store_error_usage(&runner, &run);
                        let err = PromptError::Provider(ProviderError::Response(
                            "agent requested tool execution without a prepared registry snapshot"
                                .to_string(),
                        ));
                        settle_error!(err);
                        yield Err(err);
                        break 'outer;
                    };
                    drive_step!('outer, drive_tool_calls(
                        &runner,
                        &hook_ctx,
                        &mut run,
                        calls,
                        tool_snapshot,
                        |span| source.chain_span(span),
                        S::FORWARDS_ITEMS,
                    ));
                }
                AgentRunStep::Done(response) => {
                    flush_entries!();
                    tracing::info!(
                        turn = run.turn(),
                        max_turns = runner.config.max_turns,
                        "Agent run finished"
                    );
                    source.record_run_level_telemetry(&agent_span, &response, created_agent_span);
                    // Append failure does not discard the answer; expose acknowledgement
                    // separately without assuming a failed append wrote nothing.
                    let memory_append = append_run_messages(
                        &runner,
                        &hook_ctx,
                        memory_handle.as_ref(),
                        &response.messages,
                    )
                    .await;
                    let response = response.with_memory_append(memory_append);
                    if runner.config.hooks.observes(StepEventKind::RunSettled) {
                        runner
                            .config
                            .hooks
                            .on_run_settled(
                                &hook_ctx,
                                RunSettled {
                                    outcome: SettledOutcome::Response(&response),
                                    messages: Some(run.messages()),
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
                    yield Ok(DriveItem::Done(response));
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
/// - The model tool-call events ([`MultiTurnStreamItem::ToolCall`]) are
///   emitted up front, reporting what the model emitted at turn commit.
/// - Every tool then runs (sequentially at `tool_concurrency <= 1`, else
///   concurrently bounded by it), with outcomes **collected, not surfaced**.
/// - On the first hook termination / fail-closed error the batch fails fast: no
///   new tool starts, not-yet-started concurrent siblings are dropped,
///   already-started ones are drained, and the deterministic lowest call-index
///   error is surfaced with **no** successful [`ToolExecutionCommitted`] /
///   [`ToolResult`](MultiTurnStreamItem::ToolResult) items and **no**
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
    tool_snapshot: Arc<ToolCatalog>,
    chain_tool_span: F,
    forward_items: bool,
) -> DriveStream<'a>
where
    F: Fn(tracing::Span) -> tracing::Span + WasmCompatSend + 'a,
{
    // Per-call working state: the execute span, paired with the model's
    // tool call. `span` is `Span::none()` for a
    // preresolved (invalid-recovery) call, which never executes.
    struct PreparedToolCall {
        tool_call: rig_core::message::ToolCall,
        preresolved_result: Option<UserContent>,
        span: tracing::Span,
    }
    // How a settled tool call is surfaced on the stream once the batch succeeds:
    //   - `Executed`: `ToolExecutionCommitted` (with the effective, hook-rewritten
    //     call) + the `ToolResult`.
    //   - `Skipped`: the `ToolResult` only (a `ToolCall` hook returned `Skip`, so
    //     nothing ran and no execution commit is surfaced, but the model still
    //     sees the result).
    //   - `Preresolved`: neither (an invalid-recovery result, already surfaced
    //     during the model turn); committed to history only.
    enum ToolSurface {
        Executed(rig_core::message::ToolCall),
        Skipped,
        Preresolved,
    }
    // A collected tool outcome, held (not surfaced or committed) until the whole
    // batch settles.
    struct CollectedToolResult {
        content: UserContent,
        surface: ToolSurface,
    }

    Box::pin(async_stream::stream! {
        let full_history_for_errors = run.full_history();
        let call_count = calls.len();

        // Assign each call that will actually execute an execute span. Emit the MODEL tool-call events now,
        // right after the turn committed: these report what the model emitted and
        // are *not* execution-lifecycle events. A preresolved call emits no model
        // tool-call event (its synthetic result was already surfaced during the
        // model turn) and gets no execute span.
        let mut prepared: Vec<PreparedToolCall> = Vec::with_capacity(call_count);
        for pending in calls {
            let (span, preresolved_result) = match pending.preresolved_result {
                Some(result) => (tracing::Span::none(), Some(result)),
                None => {
                    if forward_items {
                        yield Ok(MultiTurnStreamItem::ToolCall {
                            tool_call: pending.tool_call.clone(),
                        });
                    }
                    (chain_tool_span(new_execute_tool_span()), None)
                }
            };
            prepared.push(PreparedToolCall {
                tool_call: pending.tool_call,
                preresolved_result,
                span,
            });
        }

        // Outcomes are collected in call order and nothing is surfaced or
        // committed until the whole batch settles. After the first termination or
        // fail-closed error no new tool starts, started ones are drained, and the
        // lowest call-index error wins.
        let mut collected: Vec<Option<CollectedToolResult>> =
            (0..call_count).map(|_| None).collect();
        let mut first_error: Option<(usize, PromptError)> = None;

        {
            // Bounded by `tool_concurrency` (`0`/`1` poll strictly in call
            // order, giving sequential fail-fast). A shared `terminating`
            // flag makes a not-yet-started sibling skip (its side effect never
            // runs) once any sibling terminates, while in-flight siblings are
            // drained so the lowest call-index terminator wins and no task is left
            // detached.
            let terminating = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let unordered = stream::iter(prepared.into_iter().enumerate())
                .map(|(index, call)| {
                    let PreparedToolCall { tool_call, preresolved_result, span } = call;
                    let tool_snapshot = &tool_snapshot;
                    let full_history_for_errors = &full_history_for_errors;
                    let terminating = terminating.clone();
                    async move {
                        if let Some(result) = preresolved_result {
                            return (
                                index,
                                Some(Ok(CollectedToolResult {
                                    content: result,
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

        // On termination surface only the deterministic error: no execution
        // commit, no result, and no history commit.
        if let Some((_, err)) = first_error {
            yield Err(err);
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
            let Some(CollectedToolResult { content, surface }) = slot else {
                yield Err(PromptError::Provider(ProviderError::Response(
                    "tool execution finished without producing every result".to_string(),
                )));
                return;
            };
            if forward_items {
                // An executed call also surfaces its execution commit; a skipped
                // call surfaces only its result; a preresolved call surfaces
                // nothing here.
                let surface_result = match surface {
                    ToolSurface::Executed(tool_call) => {
                        surface_items.push(MultiTurnStreamItem::ToolExecutionCommitted {
                            tool_call,
                        });
                        true
                    }
                    ToolSurface::Skipped => true,
                    ToolSurface::Preresolved => false,
                };
                if surface_result
                    && let UserContent::ToolResult(tool_result) = &content
                {
                    surface_items.push(MultiTurnStreamItem::ToolResult {
                        tool_result: tool_result.clone(),
                    });
                }
            }
            committed.push(content);
        }

        if let Err(err) = run.tool_results(committed) {
            yield Err(err);
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
    last_response_id: Option<String>,
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
    /// Whether any hook is present. Gates building the history-cloning
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
            last_final_choice: Vec::new(),
            last_response_id: None,
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
    const FORWARDS_ITEMS: bool = true;

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
        _current_prompt: Message,
    ) -> DriveStream<'a> {
        Box::pin(async_stream::stream! {
            // Bound before the builder is consumed: the cap this attempt was
            // prepared with, completion-call patches included.
            let attempt_max_tokens = prepared.max_tokens;

            let dispatched =
                dispatch_completion_stream(runner, hook_ctx, run, &prepared.model, prepared.request)
                    .instrument(chat_span.clone())
                    .await;
            let CompletionDispatch { id: dispatch_id, kind: dispatched_kind, answer: mut stream } =
                match dispatched {
                    Ok(dispatched) => dispatched,
                    Err(err) => {
                        yield Err(err);
                        return;
                    }
                };
            let mut assembler = StreamedTurnAssembler::new(
                prepared.executable_tool_names.clone(),
                prepared.allowed_tool_names.clone(),
            );
            // A turn whose invalid tool call was repaired is a recovered turn:
            // neither the response hook nor `ModelTurnFinished` fires for it.
            let mut turn_recovered = false;
            // A tool call's start and arguments, held until its end validates
            // the call.
            let mut held: Vec<StreamEvent> = Vec::new();

            'turn: while let Some(item) = stream.next().await {
                // A stream error ends the reply. At most one event per item
                // forwards the item itself, so moving it out of the slot
                // avoids cloning every streamed fragment.
                let (mut item_slot, mut events, ended): (
                    Option<Item<StreamEvent>>,
                    VecDeque<StreamedTurnEvent>,
                    bool,
                ) = match item {
                    Ok(item) => match assembler.ingest(&item) {
                        Ok(events) => (Some(item), events.into(), false),
                        Err(err) => {
                            yield Err(err.into());
                            return;
                        }
                    },
                    Err(err) => {
                        yield Err(err.into());
                        return;
                    }
                };
                while let Some(event) = events.pop_front() {
                    match event {
                        StreamedTurnEvent::EmitIngested => {
                            if self.observes_text_delta
                                && let Some(Item::Event(StreamEvent::Text { text, .. })) =
                                    item_slot.as_ref()
                                && let Some(reason) = observe_action(
                                    runner
                                        .config.hooks
                                        .on_text_delta(
                                            hook_ctx,
                                            TextDelta {
                                                delta: text,
                                                aggregated: assembler.aggregated_text(),
                                            },
                                        )
                                        .await,
                                )
                            {
                                // The stop is the run's: the dispatch in flight is
                                // cancelled here, before the error surfaces, so the
                                // record is the same cancel on every transport.
                                drop(stream);
                                yield Err(run.cancel_error(reason));
                                return;
                            }
                            if self.observes_reasoning_delta
                                && let Some(Item::Event(StreamEvent::Reasoning {
                                    part,
                                    text: reasoning,
                                })) = item_slot.as_ref()
                                && let Some(reason) = observe_action(
                                    runner
                                        .config.hooks
                                        .on_reasoning_delta(
                                            hook_ctx,
                                            ReasoningDelta {
                                                part: *part,
                                                delta: reasoning,
                                                aggregated: assembler
                                                    .aggregated_reasoning(part.index())
                                                    .unwrap_or_default(),
                                            },
                                        )
                                        .await,
                                )
                            {
                                // The stop is the run's: the dispatch in flight is
                                // cancelled here, before the error surfaces, so the
                                // record is the same cancel on every transport.
                                drop(stream);
                                yield Err(run.cancel_error(reason));
                                return;
                            }
                            if let Some(item) = item_slot.take() {
                                yield Ok(MultiTurnStreamItem::stream_item(item));
                            }
                        }
                        StreamedTurnEvent::HoldToolCall => {
                            if let Some(Item::Event(event)) = item_slot.take() {
                                held.push(event);
                            }
                        }
                        StreamedTurnEvent::EmitToolCall { call } => {
                            // The call's end is the ingested item; a repaired
                            // call's end carries the repaired name.
                            let end = match item_slot.take() {
                                Some(Item::Event(StreamEvent::End { part, .. })) => Some(part),
                                _ => None,
                            };
                            for event in std::mem::take(&mut held) {
                                if self.observes_tool_call_delta
                                    && let StreamEvent::Arguments { part, json } = &event
                                    && let Some(reason) = observe_action(
                                        runner
                                            .config.hooks
                                            .on_tool_call_delta(
                                                hook_ctx,
                                                ToolCallDelta {
                                                    part: *part,
                                                    call_id: &call.id,
                                                    tool_name: call.function.name.as_str(),
                                                    delta: json,
                                                },
                                            )
                                            .await,
                                    )
                                {
                                    // The stop is the run's: the dispatch in flight is
                                    // cancelled here, before the error surfaces, so the
                                    // record is the same cancel on every transport.
                                    drop(stream);
                                    yield Err(run.cancel_error(reason));
                                    return;
                                }
                                yield Ok(MultiTurnStreamItem::stream_item(Item::Event(event)));
                            }
                            if let Some(part) = end {
                                yield Ok(MultiTurnStreamItem::stream_item(Item::Event(
                                    StreamEvent::End {
                                        part,
                                        content: AssistantContent::ToolCall(call),
                                    },
                                )));
                            }
                        }
                        StreamedTurnEvent::InvalidToolCall(invalid) => {
                            // The rejected call's items are not forwarded.
                            held.clear();
                            let partial = assembler.partial_turn(&stream.partial());
                            // Gated on `has_hooks`: building the diagnostic context
                            // clones the chat history, so an empty stack skips it and
                            // fails fast.
                            let hook_action = if self.has_hooks {
                                let context =
                                    run.streamed_invalid_tool_call_context(&partial, &invalid);
                                runner
                                    .config.hooks
                                    .on_invalid_tool_call(hook_ctx, &context)
                                    .await
                            } else {
                                None
                            };
                            // No hook resolved it: the run's policy, as the
                            // unary surface applies it (`Ignore` drops the call
                            // and goes on; `Fail` fails the run).
                            let resolved = match hook_action {
                                Some(action) => {
                                    run.resolve_streamed_invalid_tool_call(&partial, &invalid, action)
                                }
                                None => match run.unhandled_invalid_tool_call() {
                                    UnhandledInvalidToolCall::Fail => run
                                        .resolve_streamed_invalid_tool_call(
                                            &partial,
                                            &invalid,
                                            InvalidToolCallAction::fail(),
                                        ),
                                    UnhandledInvalidToolCall::Ignore => {
                                        run.ignore_streamed_invalid_tool_call()
                                    }
                                },
                            };
                            let resolution = match resolved {
                                Ok(resolution) => resolution,
                                Err(err) => {
                                    yield Err(err);
                                    return;
                                }
                            };

                            match resolution {
                                StreamedResolution::Ignored => {
                                    assembler.resolve_pending_invalid(&resolution);
                                    item_slot = None;
                                }
                                StreamedResolution::Repaired { .. } => {
                                    // The repaired call flows through the same event
                                    // handling above; the turn is now recovered.
                                    turn_recovered = true;
                                    events.extend(assembler.resolve_pending_invalid(&resolution));
                                }
                                StreamedResolution::TurnAbandoned {
                                    ref skipped_tool_result,
                                } => {
                                    let skipped_tool_result = skipped_tool_result.clone();
                                    assembler.resolve_pending_invalid(&resolution);
                                    // The abandoned reply still reports its usage
                                    // when it ends; one that already ended with the
                                    // rejected call's error has none to record.
                                    if !ended {
                                        let response = match stream.finish().await {
                                            Ok(response) => response,
                                            Err(err) => {
                                                yield Err(err.into());
                                                return;
                                            }
                                        };
                                        chat_span.record_token_usage(&response.usage);
                                        match run.record_streamed_completion_call(
                                            response.usage,
                                            response.identity(),
                                            response.finish_reason(),
                                            response.raw.clone(),
                                        ) {
                                            Ok(call) => {
                                                yield Ok(MultiTurnStreamItem::CompletionCall(call));
                                            }
                                            Err(err) => {
                                                yield Err(err);
                                                return;
                                            }
                                        }
                                    }
                                    if let Some(tool_result) = skipped_tool_result {
                                        yield Ok(MultiTurnStreamItem::ToolResult { tool_result });
                                    }
                                    return;
                                }
                            }
                        }
                    }
                }
                if ended {
                    break 'turn;
                }
            }

            // The reply's end: the response `call` would have returned for it.
            // A reply the provider did not end is truncated, and never a
            // successful zero-usage completion.
            let mut response = match stream.finish().await {
                Ok(response) => response,
                Err(err) => {
                    yield Err(err.into());
                    return;
                }
            };
            chat_span.record_token_usage(&response.usage);
            match run.record_streamed_completion_call(
                response.usage,
                response.identity(),
                response.finish_reason(),
                response.raw.clone(),
            ) {
                Ok(call) => yield Ok(MultiTurnStreamItem::CompletionCall(call)),
                Err(err) => {
                    yield Err(err);
                    return;
                }
            }

            let streamed_turn = assembler.finish(&response);
            self.last_response_id = response.response_id().map(str::to_owned);
            // The hooks and run history see the assembled turn: the
            // response's choice without ignored calls, with repaired names.
            // The final item keeps the provider's choice.
            let mut final_turn_content =
                std::mem::replace(&mut response.choice, streamed_turn.choice.clone());
            if let Err(err) = run.streamed_turn(streamed_turn) {
                yield Err(err);
                return;
            }
            let turn = CompletionDispatch {
                id: dispatch_id,
                kind: dispatched_kind,
                answer: response,
            };
            let settlement = settle_model_turn(
                &runner.config.hooks,
                hook_ctx,
                run,
                &turn,
                attempt_max_tokens,
                turn_recovered,
            )
            .await;
            match settlement {
                Ok(ModelTurnDecision::Advance { replaced }) => {
                    // The run keeps the replacement, and so does the final
                    // item the consumer receives: the fragments it saw were
                    // the provider's, the answer is the hook's.
                    if let Some(choice) = replaced {
                        final_turn_content = choice;
                    }
                }
                Ok(ModelTurnDecision::Retried) => {
                    yield Ok(MultiTurnStreamItem::ModelTurnRetried {
                        turn: hook_ctx.turn(),
                    });
                    return;
                }
                Ok(ModelTurnDecision::Terminate(reason)) => {
                    // A stop observes an already completed provider turn:
                    // its content telemetry stays visible before the
                    // cancellation.
                    self.record_turn_telemetry(
                        agent_span,
                        &chat_span,
                        &turn.answer.choice,
                        runner.config.record_telemetry_content,
                    );
                    yield Err(run.cancel_error(reason));
                    return;
                }
                Err(err) => {
                    yield Err(err);
                    return;
                }
            }

            // Only hook-accepted canonical output belongs in content telemetry.
            self.record_turn_telemetry(
                agent_span,
                &chat_span,
                &turn.answer.choice,
                runner.config.record_telemetry_content,
            );

            self.last_final_choice = final_turn_content;
        })
    }

    fn chain_span(&self, span: tracing::Span) -> tracing::Span {
        span
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
        // In tool output mode, when the finishing turn made the output-tool call,
        // surface the run's structured output as the final content.
        let final_choice = finalize_output_tool_choice(&self.last_final_choice, &response.output())
            .unwrap_or_else(|| {
                if is_empty_assistant_turn(&self.last_final_choice) {
                    tracing::warn!(
                        agent_name = self.agent_name.as_str(),
                        response_id = ?self.last_response_id,
                        "Streaming turn completed without assistant text; final response will be empty"
                    );
                }
                self.last_final_choice.clone()
            });
        Some(
            MultiTurnStreamItem::final_response_with_completion_calls(
                final_choice,
                response.usage,
                response.completion_calls.clone(),
                response.messages.clone(),
            )
            .with_memory_append(response.memory_append.clone()),
        )
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
    /// Accept the turn and advance normally. Carries the content a hook
    /// replaced the turn with, when one did. The streaming surface's final item
    /// follows it.
    Advance {
        replaced: Option<Vec<AssistantContent>>,
    },
    /// The turn was rejected and the run is ready to issue another model call.
    Retried,
    /// Stop the run with the supplied reason.
    Terminate(String),
}

/// Settle a parked model turn: fire [`AgentHook::on_outcome`] for the
/// completion dispatch (a replacement lands on the parked turn, a
/// `Cancelled` replacement terminates), then
/// [`AgentHook::on_model_turn_finished`] and apply its action to the sans-IO
/// run. The outcome hook fires here rather than at the dispatch so that both
/// media fire it in one slot, after the run validated the answer's tool calls
/// (a `recovered` turn fires neither hook, on either medium, and advances)
/// and while the turn can still be replaced. Both drivers call this once per
/// accepted attempt, so retry history, tool-turn rejection, and state
/// transitions cannot diverge by medium. `turn` carries this attempt's
/// response with the choice the hooks see, and `max_tokens` the cap it was
/// prepared with. The callers own what happens next: both record the accepted
/// turn's telemetry, and the streaming driver also keeps the content its
/// final item surfaces.
pub(crate) async fn settle_model_turn(
    hooks: &HookStack,
    hook_ctx: &HookContext,
    run: &mut AgentRun,
    turn: &CompletionDispatch<rig_core::completion::CompletionResponse>,
    max_tokens: Option<u64>,
    recovered: bool,
) -> Result<ModelTurnDecision, PromptError> {
    if recovered {
        return Ok(ModelTurnDecision::Advance { replaced: None });
    }
    let response = &turn.answer;
    let identity = response.identity();
    let finish_reason = response.finish_reason();
    let mut folded = rig_core::completion::CompletionResponse::new(
        response.choice.clone(),
        response.usage,
        response.origin.clone(),
        response.raw.clone(),
    )
    .with_optional_finish_reason(finish_reason.clone());
    folded.native = response.native.clone();
    folded.error = response.error.clone();
    folded.provider_request_id = identity.provider_request_id.clone();
    let outcome: Result<Outcome, ErrorReport> = Ok(Outcome::Completion(folded));
    let mut replaced: Option<Vec<AssistantContent>> = None;
    match hooks
        .on_outcome(
            hook_ctx,
            OutcomeEvent {
                id: turn.id,
                kind: &turn.kind,
                outcome: &outcome,
                turn: hook_ctx.turn(),
                call_id: None,
                context: None,
            },
        )
        .await
    {
        OutcomeAction::Proceed => {}
        OutcomeAction::Replace(Ok(Outcome::Completion(replacement))) => {
            run.replace_accepted_turn_choice(replacement.choice.clone())?;
            replaced = Some(replacement.choice);
        }
        OutcomeAction::Replace(Ok(other)) => {
            return Err(PromptError::Report(wrong_outcome("a completion", &other)));
        }
        OutcomeAction::Replace(Err(report)) => {
            if report.kind == ErrorKind::Cancelled {
                return Ok(ModelTurnDecision::Terminate(report.message));
            }
            return Err(PromptError::Report(report));
        }
    }
    let content = replaced.as_ref().unwrap_or(&response.choice);
    let action = hooks
        .on_model_turn_finished(
            hook_ctx,
            ModelTurnFinished {
                turn: hook_ctx.turn(),
                content,
                usage: response.usage,
                identity: &identity,
                finish_reason: finish_reason.as_ref(),
                max_tokens,
                raw: &response.raw,
            },
        )
        .await;
    match action {
        ModelTurnAction::Continue => Ok(ModelTurnDecision::Advance { replaced }),
        ModelTurnAction::Retry(request) => {
            run.retry_model_turn(request)?;
            Ok(ModelTurnDecision::Retried)
        }
        ModelTurnAction::Stop(reason) => Ok(ModelTurnDecision::Terminate(reason)),
    }
}

/// Outcome of firing the `CompletionCallEvent` hook for a turn.
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
            CompletionCallEvent {
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
/// proceeding on failure. Shared `Done`-arm behavior for both drivers. The
/// append is a `Memory` dispatch at the boundary: observe-only for hooks
/// unless one opts into `MemoryDispatch`. Returns how the append settled
/// for the response, `None` when the run has no memory to append to.
pub(crate) async fn append_run_messages(
    runner: &AgentRunner,
    ctx: &HookContext,
    memory_handle: Option<&(MemoryHandle, rig_core::id::ConversationId)>,
    messages: &[Message],
) -> Option<MemoryAppend> {
    // Clone into an owned vec only when there is a backend to append to, so the
    // common no-memory path pays nothing.
    let (memory, id) = memory_handle?;
    let appended = dispatch_effect(
        &runner.config.hooks,
        ctx,
        runner.config.bus.dispatcher(),
        memory.key(),
        EffectKind::Memory {
            op: rig_core::effect::MemoryOp::Append {
                conversation: id.clone(),
                messages: messages.to_vec(),
            },
        },
    )
    .await
    .and_then(|outcome| match outcome {
        Outcome::Memory(rig_core::effect::MemoryOutcome::Appended) => Ok(()),
        other => Err(wrong_outcome("appended memory", &other)),
    });
    Some(match appended {
        Ok(()) => MemoryAppend::Acknowledged,
        Err(report) => {
            tracing::warn!(
                error = %report,
                conversation_id = %id,
                "conversation memory append failed; surfacing final response anyway"
            );
            MemoryAppend::Failed { report }
        }
    })
}

/// Dispatch any effect through the agent's bus at the dispatch boundary:
/// `on_dispatch` before (a same-family patch or a denial), the bus, then
/// `on_outcome` after (a replacement). The engine's memory and retrieval
/// effects go through here; completions and tool calls have their own
/// entry points because their denials have run-level meaning.
pub(crate) async fn dispatch_effect(
    hooks: &HookStack,
    ctx: &HookContext,
    dispatcher: &crate::bus::Dispatcher,
    key: &rig_core::effect::HandlerKey,
    kind: EffectKind,
) -> Result<Outcome, ErrorReport> {
    let id = dispatcher.mint_id();
    let family = kind.family();
    let kind = match hooks
        .on_dispatch(
            ctx,
            DispatchEvent {
                id,
                kind: &kind,
                turn: ctx.turn(),
                call_id: None,
                context: None,
            },
        )
        .await
    {
        DispatchAction::Proceed => kind,
        DispatchAction::Patch(patched) if patched.family() == family => patched,
        DispatchAction::Patch(other) => return Err(wrong_family_patch(kind.name(), &other)),
        DispatchAction::Deny(report) => return Err(report),
    };
    let outcome = dispatcher
        .dispatch_with(key, kind.clone(), DispatchOptions::default().with_id(id))
        .await;
    match hooks
        .on_outcome(
            ctx,
            OutcomeEvent {
                id,
                kind: &kind,
                outcome: &outcome,
                turn: ctx.turn(),
                call_id: None,
                context: None,
            },
        )
        .await
    {
        OutcomeAction::Proceed => outcome,
        OutcomeAction::Replace(replaced) => replaced,
    }
}

pub(crate) fn wrong_outcome(expected: &str, outcome: &Outcome) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Internal,
        format!(
            "expected {expected}, the handler answered with a {} outcome",
            outcome.family()
        ),
    )
}

/// Whether (and how) a tool call executed, for [`run_single_tool`].
pub(crate) enum ToolExecution {
    /// The tool's body ran. Carries the effective tool call, meaning the model's
    /// call with any [`DispatchAction::Patch`] hook rewrite applied, so the
    /// driver can surface it in the
    /// [`ToolExecutionCommitted`](crate::agent::streaming::MultiTurnStreamItem::ToolExecutionCommitted)
    /// event (what actually ran, not the model's original arguments).
    Executed(ToolCall),
    /// A dispatch hook denied the call ([`DispatchAction::skip`]): the
    /// body did not run, so no execution-commit is surfaced, but the skip result
    /// still reaches the model and is surfaced as a `ToolResult`.
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

/// Execute a single tool call through the dispatch boundary and shape the
/// result. **Shared by the blocking and streaming drivers** so a tool call
/// behaves identically in both: same hook events (`on_dispatch` before,
/// `on_outcome` after), same fail-closed skip/terminate handling, and the
/// same result shaping. A hook's skip becomes [`ToolResult::skipped`], and
/// every result is converted directly into typed message content through
/// [`tool_result_output`] without reparsing text. Records `gen_ai.tool.*` on
/// the current span; `error_history` builds a cancellation error if a hook
/// terminates the run. Returns whether the tool body executed via
/// [`ToolCallOutcome::execution`].
pub(crate) async fn run_single_tool(
    runner: &AgentRunner,
    ctx: &HookContext,
    tool_snapshot: &ToolCatalog,
    tool_call: &ToolCall,
    error_history: &[Message],
) -> Result<ToolCallOutcome, PromptError> {
    let tool_context = &runner.tool_context;
    let record_content = runner.config.record_telemetry_content;
    let tool_name = tool_call.function.name.as_str();
    // Arguments the model sent that are not a JSON object never reach the
    // tool: the model reads why, and calls again.
    if let Some(raw) = &tool_call.function.invalid_arguments {
        let content = tool_result_message(
            tool_call.id.clone(),
            tool_call.function.name.clone(),
            format!(
                "The arguments for tool `{tool_name}` are not a JSON object: {raw}. \
                 Call the tool again with a JSON object as its arguments."
            ),
        );
        return Ok(ToolCallOutcome {
            content,
            execution: ToolExecution::Skipped,
        });
    }
    let args = tool_call.function.arguments_value().to_string();

    let tool_span = tracing::Span::current();
    tool_span.record("gen_ai.tool.name", tool_name);
    tool_span.record(
        "gen_ai.tool.call.id",
        tracing::field::display(&tool_call.id),
    );
    if record_content {
        tool_span.record("gen_ai.tool.call.arguments", &args);
    }

    let ToolCallDispatch {
        result: exec,
        executed,
        context: _dispatch_context,
        args: effective_args,
    } = match dispatch_tool_call(
        runner,
        ctx,
        tool_snapshot,
        tool_name,
        args.clone(),
        &tool_call.id,
        tool_context,
    )
    .await
    {
        Ok(dispatch) => dispatch,
        Err(ToolDispatchAbort::Cancelled(reason)) => {
            return Err(PromptError::cancelled(error_history.to_vec(), reason));
        }
        Err(ToolDispatchAbort::Failed(report)) => return Err(PromptError::Report(report)),
    };

    // A hook patched the arguments: re-record the span so the trace reflects
    // what the tool actually received rather than what the model emitted.
    if effective_args != args {
        if record_content {
            tool_span.record("gen_ai.tool.call.arguments", &effective_args);
        }
        tracing::debug!(
            tool_name = tool_name,
            "tool-call arguments rewritten by a hook"
        );
    }

    // A skip runs nothing and surfaces no execution commit; a real execution
    // carries the effective tool call (the model's call with any patch
    // applied) so a redaction rewrite does not leak.
    let execution = if !executed {
        ToolExecution::Skipped
    } else {
        let mut effective_tool_call = tool_call.clone();
        effective_tool_call.function =
            ToolFunction::parse(tool_call.function.name.clone(), &effective_args);
        ToolExecution::Executed(effective_tool_call)
    };
    // Outcome metadata describes the execution itself, while result content
    // follows the same presentation policy as the model: what the outcome
    // hook let through is what telemetry records.
    record_tool_result(&tool_span, &exec);
    if record_content {
        tool_span.record("gen_ai.tool.call.result", exec.output().render());
    }
    let content = tool_result_output(tool_call.id.clone(), tool_call.function.name.clone(), &exec);
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
    /// streaming surface parents the same tree but does not chain).
    ///
    /// Atomic rather than `Cell` despite being driven by a single sequential
    /// task: the engine passes `chain_span` as a closure into
    /// `drive_tool_calls`, whose returned `DriveStream` is `Send`. That closure
    /// borrows the source, so every `TurnSource` must be `Sync`, which
    /// `AtomicU64` provides and `Cell` does not.
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
}

impl TurnSource for UnaryTurnSource {
    const FORWARDS_ITEMS: bool = false;

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
        _current_prompt: Message,
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

            let dispatched =
                dispatch_completion_response(runner, hook_ctx, run, &prepared.model, prepared.request)
                    .instrument(chat_span.clone())
                    .await;
            let dispatched = match dispatched {
                Ok(dispatched) => dispatched,
                Err(err) => {
                    yield Err(err);
                    return;
                }
            };

            let mut outcome = match run.model_response(ModelTurn::from_response_parts(
                &dispatched.answer,
                prepared.executable_tool_names,
                prepared.allowed_tool_names,
            )) {
                Ok(outcome) => outcome,
                Err(err) => {
                    yield Err(err);
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
                                yield Err(err);
                                return;
                            }
                        };
                    }
                    ModelTurnOutcome::TurnRetried => break,
                    ModelTurnOutcome::Continue {
                        response_hook_suppressed,
                    } => {
                        let settlement = settle_model_turn(
                            &runner.config.hooks,
                            hook_ctx,
                            run,
                            &dispatched,
                            attempt_max_tokens,
                            response_hook_suppressed,
                        )
                        .await;
                        match settlement {
                            Ok(ModelTurnDecision::Advance { .. }) => {}
                            Ok(ModelTurnDecision::Retried) => break,
                            Ok(ModelTurnDecision::Terminate(reason)) => {
                                record_accepted_turn(run);
                                yield Err(run.cancel_error(reason));
                                return;
                            }
                            Err(err) => {
                                yield Err(err);
                                return;
                            }
                        }
                        record_accepted_turn(run);
                        break;
                    }
                }
            }
        })
    }

    fn record_run_level_telemetry(
        &self,
        agent_span: &tracing::Span,
        response: &PromptResponse,
        created_agent_span: bool,
    ) {
        // Record completion and usage onto the agent span only when this run
        // created it, never onto a caller-supplied outer span. The blocking
        // surface additionally records the final completion text.
        if created_agent_span {
            if self.record_telemetry_content {
                agent_span.record("gen_ai.completion", response.output());
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

/// A completion the bus accepted, with the effect it was dispatched as. The
/// outcome hook fires when the turn is settled (after the run validated the
/// answer's tool calls), so the id and the dispatched effect travel with the
/// answer: the unary response or the provider stream.
pub(crate) struct CompletionDispatch<A> {
    pub(crate) id: EffectId,
    pub(crate) kind: EffectKind,
    pub(crate) answer: A,
}

fn wrong_family_patch(expected: &str, kind: &EffectKind) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Internal,
        format!(
            "a hook patched a {expected} dispatch into a `{}` effect",
            kind.name()
        ),
    )
}

/// Fire `on_dispatch` for a completion and return the effect to send, with
/// any same-family patch applied. A `Cancelled` denial cancels the run; any
/// other denial or a wrong-family patch fails it. The outcome hook is not fired
/// here: on either medium it fires in [`settle_model_turn`], once the run has
/// validated the answer's tool calls and parked the turn. That is the slot
/// where a hook can still replace what history keeps, on both media.
async fn completion_effect(
    runner: &AgentRunner,
    ctx: &HookContext,
    run: &AgentRun,
    request: rig_core::completion::CompletionRequest,
    stream: bool,
) -> Result<(EffectId, EffectKind), PromptError> {
    let id = runner.config.bus.dispatcher().mint_id();
    let kind = EffectKind::Completion { request, stream };
    let kind = match runner
        .config
        .hooks
        .on_dispatch(
            ctx,
            DispatchEvent {
                id,
                kind: &kind,
                turn: ctx.turn(),
                call_id: None,
                context: None,
            },
        )
        .await
    {
        DispatchAction::Proceed => kind,
        DispatchAction::Patch(EffectKind::Completion { request, .. }) => {
            EffectKind::Completion { request, stream }
        }
        DispatchAction::Patch(other) => {
            return Err(PromptError::Report(wrong_family_patch(
                "completion",
                &other,
            )));
        }
        DispatchAction::Deny(report) if report.kind == ErrorKind::Cancelled => {
            return Err(run.cancel_error(report.message));
        }
        DispatchAction::Deny(report) => return Err(PromptError::Report(report)),
    };
    Ok((id, kind))
}

/// Dispatch a streaming completion through the agent's bus.
pub(crate) async fn dispatch_completion_stream(
    runner: &AgentRunner,
    ctx: &HookContext,
    run: &AgentRun,
    model: &ModelHandle,
    request: rig_core::completion::CompletionRequest,
) -> Result<CompletionDispatch<rig_core::streaming::CompletionStream>, PromptError> {
    let (id, kind) = completion_effect(runner, ctx, run, request, true).await?;
    let events = runner.config.bus.dispatcher().dispatch_stream_with(
        model.key(),
        kind.clone(),
        DispatchOptions::default().with_id(id),
    );
    Ok(CompletionDispatch {
        id,
        kind,
        answer: crate::bus::wrap_stream(model.label().to_string(), events),
    })
}

/// Dispatch a unary completion through the agent's bus and await its response.
pub(crate) async fn dispatch_completion_response(
    runner: &AgentRunner,
    ctx: &HookContext,
    run: &AgentRun,
    model: &ModelHandle,
    request: rig_core::completion::CompletionRequest,
) -> Result<CompletionDispatch<rig_core::completion::CompletionResponse>, PromptError> {
    let (id, kind) = completion_effect(runner, ctx, run, request, false).await?;
    let outcome = runner
        .config
        .bus
        .dispatcher()
        .dispatch_with(
            model.key(),
            kind.clone(),
            DispatchOptions::default().with_id(id),
        )
        .await;
    match outcome {
        Ok(Outcome::Completion(answer)) => Ok(CompletionDispatch { id, kind, answer }),
        Ok(other) => Err(PromptError::Report(ErrorReport::new(
            ErrorKind::Internal,
            format!(
                "the completion handler answered with a {} outcome",
                other.family()
            ),
        ))),
        Err(report) => Err(PromptError::Report(report)),
    }
}

/// A tool call answered at the dispatch boundary: the result, the context
/// the tool answered with, and the arguments it actually ran with (after
/// any hook's patch).
pub(crate) struct ToolCallDispatch {
    pub(crate) result: ToolResult,
    /// Disposition at the dispatch boundary, before outcome presentation hooks.
    pub(crate) executed: bool,
    pub(crate) context: crate::tool::ToolContext,
    pub(crate) args: String,
}

/// Dispatch a tool call through the agent's bus at the dispatch boundary:
/// `on_dispatch` before (patch the arguments, skip with a reason, or stop),
/// the bus, `on_outcome` after (replace what the run sees, or stop).
/// `Err(reason)` cancels the run; every other failure is the tool result
/// the model sees.
/// Why a tool dispatch produced no result for the model: a hook cancelled
/// the run, or the bus could not serve the call (closed, or the tool's
/// handler gone). This is a failure of the run, not of the tool.
pub(crate) enum ToolDispatchAbort {
    Cancelled(String),
    Failed(ErrorReport),
}

pub(crate) async fn dispatch_tool_call(
    runner: &AgentRunner,
    ctx: &HookContext,
    tool_snapshot: &ToolCatalog,
    tool_name: &str,
    args: String,
    call_id: &rig_core::message::CallId,
    tool_context: &crate::tool::ToolContext,
) -> Result<ToolCallDispatch, ToolDispatchAbort> {
    let hooks = &runner.config.hooks;
    let dispatcher = runner.config.bus.dispatcher();
    let id = dispatcher.mint_id();
    let kind = EffectKind::ToolCall {
        name: tool_name.to_owned(),
        args,
    };
    // The context the tool runs with travels beside the effect, never in it
    // (format 5): the hooks see it on the event, the bus carries it to the
    // tool's sink, and what the tool published comes back the same way.
    let inbound = tool_context.for_dispatch();
    let (kind, denied) = match hooks
        .on_dispatch(
            ctx,
            DispatchEvent {
                id,
                kind: &kind,
                turn: ctx.turn(),
                call_id: Some(call_id),
                context: Some(&inbound),
            },
        )
        .await
    {
        DispatchAction::Proceed => (kind, None),
        DispatchAction::Patch(patched) => {
            match super::hook::validate_dispatch_patch(&kind, &patched) {
                Ok(()) => (patched, None),
                Err(report) => (kind, Some(report)),
            }
        }
        DispatchAction::Deny(report) => {
            if report.kind == ErrorKind::Cancelled {
                return Err(ToolDispatchAbort::Cancelled(report.message));
            }
            tracing::info!(tool_name = tool_name, reason = %report.message, "Tool call rejected");
            // A patch an earlier hook made before the denial is what the
            // skipped result reports.
            let salvaged = ctx.take_salvaged_patch(id);
            let kind = salvaged
                .filter(|patched| super::hook::validate_dispatch_patch(&kind, patched).is_ok())
                .unwrap_or(kind);
            (kind, Some(report))
        }
    };
    let effective_args = match &kind {
        EffectKind::ToolCall { args, .. } => args.clone(),
        _ => String::new(),
    };
    let mut published: Option<crate::tool::ToolContext> = None;
    let mut executed = false;
    let outcome: Result<Outcome, ErrorReport> = match denied {
        Some(report) => Ok(Outcome::ToolResult {
            result: ToolResult::skipped(report.message),
        }),
        None => match tool_snapshot.key(tool_name) {
            Some(key) => {
                let pending = dispatcher.dispatch_with(
                    key.raw(),
                    kind.clone(),
                    DispatchOptions::default()
                        .with_id(id)
                        .with_tool_context(inbound.clone()),
                );
                let published_at = pending.published_context();
                let outcome = pending.await;
                executed =
                    matches!(&outcome, Ok(Outcome::ToolResult { result }) if !result.is_skipped());
                published = published_at.and_then(|published| published.take());
                outcome
            }
            None => Ok(Outcome::ToolResult {
                result: ToolResult::failed(
                    crate::tool::ToolExecutionError::not_found(format!(
                        "no tool named `{tool_name}` is registered"
                    ))
                    .with_model_feedback(format!("tool `{tool_name}` not found")),
                ),
            }),
        },
    };
    let context = published.unwrap_or_else(|| tool_context.for_dispatch());
    let outcome = match hooks
        .on_outcome(
            ctx,
            OutcomeEvent {
                id,
                kind: &kind,
                outcome: &outcome,
                turn: ctx.turn(),
                call_id: Some(call_id),
                context: Some(&context),
            },
        )
        .await
    {
        OutcomeAction::Proceed => outcome,
        OutcomeAction::Replace(replaced) => replaced,
    };
    Ok(match outcome {
        Ok(Outcome::ToolResult { result }) => ToolCallDispatch {
            executed,
            result,
            context,
            args: effective_args,
        },
        Ok(other) => ToolCallDispatch {
            executed,
            result: ToolResult::failed(crate::tool::ToolExecutionError::other(format!(
                "the tool handler answered with a {} outcome",
                other.family()
            ))),
            context: tool_context.for_dispatch(),
            args: effective_args,
        },
        // A hook that observed the result stopped the run.
        Err(report) if report.kind == ErrorKind::Cancelled => {
            return Err(ToolDispatchAbort::Cancelled(report.message));
        }
        // A layer on the tool's key denied the call: the model sees the
        // skipped result, as it does for a hook's denial.
        Err(report) if report.kind == ErrorKind::Denied => {
            tracing::info!(tool_name = tool_name, reason = %report.message, "Tool call denied");
            ToolCallDispatch {
                executed,
                result: ToolResult::skipped(report.message),
                context: tool_context.for_dispatch(),
                args: effective_args,
            }
        }
        // The bus could not serve the call, or a replayer refused it as a
        // divergence: the run fails with the report rather than telling the model
        // its tool failed, since a replay continuing on an answer the record never
        // gave would pass with a different trace.
        Err(report)
            if matches!(
                report.kind,
                ErrorKind::BusClosed | ErrorKind::HandlerUnavailable | ErrorKind::Divergence
            ) =>
        {
            return Err(ToolDispatchAbort::Failed(report));
        }
        Err(report) => ToolCallDispatch {
            executed,
            result: ToolResult::failed(report.into()),
            context: tool_context.for_dispatch(),
            args: effective_args,
        },
    })
}
