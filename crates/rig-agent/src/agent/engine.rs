//! Shared agent drive loop, tool dispatch, and turn settlement. Unary and
//! streaming [`TurnSource`] implementations supply model responses; the engine
//! applies lifecycle policy and advances the sans-I/O [`AgentRun`].

use std::collections::{BTreeMap, VecDeque};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use futures::{Stream, StreamExt, stream};
use tracing::{Instrument, span::Id};

use crate::bus::MemoryHandle;
use rig_core::error::ProviderError;
use rig_core::{
    completion::ModelRef,
    effect::{EffectKind, Outcome},
    error::{ErrorKind, ErrorReport},
    message::{AssistantContent, Message, ToolCall, ToolFunction, ToolName},
    telemetry::SpanCombinator,
    wasm_compat::{WasmBoxedFuture, WasmBoxedStream, WasmCompatSend, WasmCompatSync},
};

use super::{
    ModelHandle,
    completion::{PreparedCompletionRequest, build_prepared_completion_request},
    hook::{
        AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, HookStack,
        InvalidToolCallAction, ModelSelection, ModelSelectionAction, ModelTurnAction,
        ModelTurnFinished, ObservationAction, OutcomeAction, ReasoningDelta, RequestPatch,
        RunSettled, RunStart, RunStartAction, SettledOutcome, StepEventKind, TextDelta,
        ToolCallDelta,
    },
    run::{
        AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome, PendingToolCall, ToolAnswer,
        streamed::{StreamedResolution, StreamedTurnAssembler, StreamedTurnEvent},
    },
    run::{
        response::{CompletionCall, MemoryAppend, PromptResponse, finalize_output_tool_choice},
        transcript::{assistant_text_from_choice, is_empty_assistant_turn},
    },
    runner::AgentRunner,
    telemetry::{build_chat_span, new_execute_tool_span},
};
use crate::run::UnhandledInvalidToolCall;
use crate::{
    completion::PromptError,
    streaming::{Item, Part, StreamEvent},
    tool::{ToolCatalog, ToolResult},
};
use dispatch::{CompletionScope, DispatchScope, model_step};
use projection::{ProjectedItems, committed_stream_items};

mod dispatch;
mod projection;

/// A boxed, medium-specific item stream for one model turn. Boxed so a generic
/// [`drive_agent`] can forward it without the per-step future leaking into the
/// engine's own (`Send`) inference.
pub(crate) type DriveStream<'a> = WasmBoxedStream<'a, Result<Surfaced, PromptError>>;

/// What a model turn surfaces directly. It has no tool variants: tool items
/// reach the stream only as [`ProjectedItems`] of committed history.
pub(crate) enum Surfaced {
    /// See [`MultiTurnStreamItem::StreamAssistantItem`](crate::agent::MultiTurnStreamItem::StreamAssistantItem).
    Provider(Item<StreamEvent>),
    /// See [`MultiTurnStreamItem::CompletionCall`](crate::agent::MultiTurnStreamItem::CompletionCall).
    CompletionCall(CompletionCall),
    /// See [`MultiTurnStreamItem::ModelTurnRetried`](crate::agent::MultiTurnStreamItem::ModelTurnRetried).
    ModelTurnRetried { turn: usize },
    /// See [`MultiTurnStreamItem::FinalResponse`](crate::agent::MultiTurnStreamItem::FinalResponse).
    Final(PromptResponse),
}

/// Engine output: stream items for forwarding (no bare tool item: only
/// [`committed_stream_items`] builds [`ProjectedItems`]) or the terminal response.
pub(crate) enum DriveItem {
    /// A non-tool item, including the streaming surface's final response.
    Surfaced(Surfaced),
    /// The tool items of what the run committed since the last projection.
    Projected(ProjectedItems),
    /// The run finished; carries the canonical response the blocking fold
    /// returns. The streaming surface has already received the final item as
    /// the preceding `Surfaced` and ignores this.
    Done(PromptResponse),
}

/// Medium-specific model turns, span chaining, telemetry, and final-item
/// construction. Implementations resolve invalid calls during model ingestion
/// and feed accepted turns back into the run; the engine runs tool calls.
pub(crate) trait TurnSource: Sized + WasmCompatSend + WasmCompatSync {
    /// Whether this source's completions are dispatched as streams.
    const STREAMS: bool;

    /// Build this medium's per-turn `chat` span (name + parenting + any
    /// `follows_from` chaining differ between blocking and streaming).
    fn open_chat_span(
        &self,
        runner: &AgentRunner,
        effective_preamble: Option<&str>,
    ) -> tracing::Span;

    /// Run one model turn: issue the provider call through the engine's open
    /// `scope`, feed the result into the sans-IO machine, and yield any
    /// intermediate items. Returning normally advances the loop; yielding an
    /// `Err` terminates the run. Only [`model_step`] calls this, and it closes
    /// the scope on either exit.
    fn run_model_turn<'a>(
        &'a mut self,
        runner: &'a AgentRunner,
        hook_ctx: &'a HookContext,
        run: &'a mut AgentRun,
        prepared: PreparedCompletionRequest,
        chat_span: tracing::Span,
        agent_span: &'a tracing::Span,
        scope: &'a mut CompletionScope<Self>,
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
    fn final_item(&self, response: &PromptResponse) -> Option<Surfaced>;
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
        // Every failed exit records usage and settles before yielding the
        // error, because consumers need not poll again, then leaves by `exit`.
        macro_rules! fail {
            ($err:expr, $($exit:tt)+) => {{
                let err: PromptError = $err;
                store_error_usage(&runner, &run);
                if runner.config.hooks.observes(StepEventKind::RunSettled) {
                    let reason = err.to_string();
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
                yield Err(err);
                $($exit)+;
            }};
        }
        // Set only after a model turn commits successfully and consumed by its
        // immediately following CallTools step. This keeps the sans-IO run state
        // serializable while pinning execution to the definitions sent that turn.
        let mut pending_tool_snapshot: Option<Arc<ToolCatalog>> = None;
        // Restore routing history so resumed hooks see the last issued model.
        let mut previous_model: Option<ModelRef> = run.previous_model().cloned();
        // Tool items stream only as the projection of what the run committed
        // past `cursor`, which starts at `projection_start`. `executed` holds
        // the settled batch's slots in commit order; both surfaces check it.
        let mut cursor = run.projection_start();
        let mut executed: Vec<Option<ToolCall>> = Vec::new();
        macro_rules! project {
            ($label:lifetime) => {{
                let committed = run.messages().get(cursor..).unwrap_or_default();
                match committed_stream_items(committed, &std::mem::take(&mut executed)) {
                    Ok(items) => yield Ok(DriveItem::Projected(items)),
                    Err(at) => fail!(run.cancel_error(format!(
                        "agent run driver protocol violation: tool batch \
                         does not line up with its committed results at {at}"
                    )), break $label),
                }
                cursor = run.messages().len();
            }};
        }

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
                fail!(err, return);
            }
        }

        'outer: loop {
            flush_entries!();
            // A failed step is not projected: a failed turn may have committed
            // calls it never answers, and the error carries that history.
            let step = match run.next_step() {
                Ok(step) => step,
                Err(err) => fail!(err, break 'outer),
            };
            project!('outer);

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
                                fail!(run.cancel_error(reason), break 'outer)
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
                            fail!(run.cancel_error(reason), break 'outer)
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
                        Err(report) => fail!(PromptError::Report(report), break 'outer),
                    };

                    // Telemetry uses the effective preamble before output-mode augmentation.
                    let effective_preamble = request_patch
                        .as_ref()
                        .and_then(|o| o.preamble.as_deref())
                        .or(runner.config.preamble.as_deref());

                    let chat_span = source.open_chat_span(&runner, effective_preamble);

                    // Pin output mode across registry changes between turns.
                    let committed_output_tool = run.output_tool_name().map(str::to_owned);
                    let (mut request, mut prepared) = match build_prepared_completion_request(
                        &runner,
                        &hook_ctx,
                        &selected_model,
                        prompt,
                        &history,
                        committed_output_tool.as_deref(),
                        request_patch.as_ref(),
                    )
                    .await
                    {
                        Ok(prepared) => prepared,
                        Err(err) => fail!(PromptError::from(err), break 'outer),
                    };
                    // Checked against the model this call goes to, after
                    // selection: a refused option fails the run before the
                    // call is sent.
                    if let Err(err) = runner
                        .config
                        .check_call_options(&selected_label, &mut request.options)
                    {
                        fail!(PromptError::from(err), break 'outer);
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

                    let mut turn_stream = model_step(
                        &mut source,
                        &runner,
                        &hook_ctx,
                        &mut run,
                        request,
                        prepared,
                        chat_span,
                        &agent_span,
                    );
                    let turn_error = loop {
                        match turn_stream.next().await {
                            Some(Ok(item)) => yield Ok(DriveItem::Surfaced(item)),
                            Some(Err(err)) => break Some(err),
                            None => break None,
                        }
                    };
                    drop(turn_stream);
                    if let Some(err) = turn_error {
                        fail!(err, break 'outer);
                    }
                    // An abandoned turn commits its call and the answers.
                    project!('outer);
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
                        fail!(PromptError::Provider(ProviderError::Response(
                            "agent requested tool execution without a prepared registry snapshot"
                                .to_string(),
                        )), break 'outer);
                    };
                    let settled = drive_tool_calls(
                        &runner,
                        &hook_ctx,
                        &mut run,
                        calls,
                        tool_snapshot,
                        |span| source.chain_span(span),
                        S::STREAMS,
                    )
                    .await;
                    match settled {
                        Ok(ran) => executed = ran,
                        Err(err) => fail!(err, break 'outer),
                    }
                    project!('outer);
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
                        yield Ok(DriveItem::Surfaced(final_item));
                    }
                    yield Ok(DriveItem::Done(response));
                    break 'outer;
                }
            }
        }

    }
}

/// Execute a turn's tool calls **atomically per batch**, shared by both surfaces.
/// It streams nothing: the engine projects what it commits. It returns one
/// slot per committed result, in commit order, holding the effective call when
/// that body ran (a [`ToolExecutionCommitted`](crate::agent::MultiTurnStreamItem::ToolExecutionCommitted)).
///
/// - Every tool runs (sequentially at `tool_concurrency <= 1`, else
///   concurrently bounded by it), with outcomes collected. A malformed call
///   never runs: its invalid-call hook decides its answer.
/// - Once the batch drains, each settled call is answered into the run in
///   call order, so a stop cancels with the settled results kept (see below).
///   A run a host drove to a part-answered batch itself resumes here with only
///   the unanswered calls.
/// - On the first hook termination / fail-closed error the batch fails fast: no
///   new tool starts, not-yet-started concurrent siblings are dropped,
///   already-started ones are drained, and the deterministic lowest call-index
///   error is returned. A termination's history is
///   [`AgentRun::canonical_history`]: the settled results, every other call
///   closed by `transcript::close_pending`.
/// - Only if the whole batch settles successfully does the run commit the
///   results to history, in call order, with any results answered before
///   (a preresolved invalid-recovery result, or one a host already fed).
///
/// `chain_tool_span` lets the blocking surface chain spans into its linear
/// `follows_from` sequence; `is_streaming` is reported to the invalid-call hook.
pub(crate) fn drive_tool_calls<'a, F>(
    runner: &'a AgentRunner,
    hook_ctx: &'a HookContext,
    run: &'a mut AgentRun,
    calls: Vec<PendingToolCall>,
    tool_snapshot: Arc<ToolCatalog>,
    chain_tool_span: F,
    is_streaming: bool,
) -> WasmBoxedFuture<'a, Result<Vec<Option<ToolCall>>, PromptError>>
where
    F: Fn(tracing::Span) -> tracing::Span + WasmCompatSend + 'a,
{
    Box::pin(async move {
        let call_count = calls.len();

        // A malformed call carries the invalid-call context its hook is
        // offered before the call is answered.
        let prepared: Vec<_> = calls
            .into_iter()
            .map(|call| {
                let call = match call {
                    PendingToolCall::Execute(call) => Ok(call),
                    PendingToolCall::Malformed(call) => {
                        let context = run.malformed_context(&call, is_streaming);
                        Err((call, context))
                    }
                };
                (call, chain_tool_span(new_execute_tool_span()))
            })
            .collect();

        // Answers are collected in call order and nothing is committed until
        // the whole batch settles. After the first termination or fail-closed
        // error no new tool starts, started ones are drained, and the lowest
        // call-index error wins.
        let mut collected: Vec<Option<(ToolAnswer, Option<ToolCall>)>> =
            (0..call_count).map(|_| None).collect();
        let mut first_error: Option<(usize, ToolExit)> = None;

        {
            // Bounded by `tool_concurrency` (`0`/`1` poll strictly in call
            // order, giving sequential fail-fast). A shared `terminating`
            // flag makes a not-yet-started sibling skip (its side effect never
            // runs) once any sibling terminates, while in-flight siblings are
            // drained so the lowest call-index terminator wins and no task is left
            // detached.
            let terminating = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let unordered = stream::iter(prepared.into_iter().enumerate())
                .map(|(index, (call, span))| {
                    let tool_snapshot = &tool_snapshot;
                    let terminating = terminating.clone();
                    async move {
                        // `None` marks a dropped (never-started) sibling.
                        if terminating.load(std::sync::atomic::Ordering::SeqCst) {
                            return (index, None);
                        }
                        let outcome = match call {
                            Ok(call) => {
                                run_single_tool(runner, hook_ctx, tool_snapshot, call.tool_call())
                                    .await
                                    .map(|outcome| (call.answer(outcome.result), outcome.executed))
                            }
                            Err((call, context)) => {
                                let action = runner
                                    .config
                                    .hooks
                                    .on_invalid_tool_call(hook_ctx, &context)
                                    .await;
                                Ok((call.answer(action), None))
                            }
                        };
                        (index, Some(outcome))
                    }
                    .instrument(span)
                })
                .buffer_unordered(runner.concurrency.max(1));
            futures::pin_mut!(unordered);

            while let Some((index, outcome)) = unordered.next().await {
                // A dropped sibling records nothing.
                let Some(result) = outcome else { continue };
                match result {
                    Ok(answered) => {
                        // An answer that ends the run fails the batch when applied.
                        if answered.0.ends_run() {
                            terminating.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                        if let Some(slot) = collected.get_mut(index) {
                            *slot = Some(answered);
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

        // Answer the settled calls into the run, so a stop still cancels with
        // them kept. The run applies a run-ending answer after the rest, and
        // one after the lowest error is never applied.
        let mut answers = Vec::with_capacity(call_count);
        let mut ran = Vec::with_capacity(call_count);
        for (index, slot) in collected.into_iter().enumerate() {
            let Some((answer, executed)) = slot else {
                continue;
            };
            if answer.ends_run() && first_error.as_ref().is_some_and(|(i, _)| *i < index) {
                continue;
            }
            ran.push((answer.index, executed));
            answers.push(answer);
        }
        if let Err(err) = run.answer_all(answers) {
            first_error = Some((0, ToolExit::Fail(err)));
        }

        // On termination return only the deterministic error.
        if let Some((_, exit)) = first_error {
            return Err(match exit {
                ToolExit::Stop(reason) => run.cancel_error(reason),
                ToolExit::Fail(err) => err,
            });
        }

        // The batch is committed as one user message holding a result per
        // call, in call order: a call answered here holds its effective call
        // when its body ran, and any other slot (answered before) holds none.
        let committed = match run.messages().last() {
            Some(Message::User { content }) => content.len(),
            _ => 0,
        };
        let mut executed: Vec<Option<ToolCall>> = vec![None; committed];
        for (index, call) in ran {
            if let Some(slot) = executed.get_mut(index) {
                *slot = call;
            }
        }
        Ok(executed)
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
    const STREAMS: bool = true;

    fn open_chat_span(
        &self,
        runner: &AgentRunner,
        effective_preamble: Option<&str>,
    ) -> tracing::Span {
        build_chat_span!(runner, effective_preamble, "chat", "chat")
    }

    fn run_model_turn<'a>(
        &'a mut self,
        runner: &'a AgentRunner,
        hook_ctx: &'a HookContext,
        run: &'a mut AgentRun,
        prepared: PreparedCompletionRequest,
        chat_span: tracing::Span,
        agent_span: &'a tracing::Span,
        scope: &'a mut CompletionScope<Self>,
    ) -> DriveStream<'a> {
        Box::pin(async_stream::stream! {
            // Bound before the builder is consumed: the cap this attempt was
            // prepared with, completion-call patches included.
            let attempt_max_tokens = prepared.max_tokens;

            let stream = chat_span.in_scope(|| scope.dispatch_stream(runner, &prepared.model));
            let mut stream = match stream {
                Ok(stream) => stream,
                Err(err) => {
                    yield Err(err);
                    return;
                }
            };
            let mut assembler = StreamedTurnAssembler::new(prepared.policy);
            // A turn whose invalid tool call was repaired is a recovered turn:
            // neither the response hook nor `ModelTurnFinished` fires for it.
            let mut turn_recovered = false;
            // The start and arguments of each call naming a tool the turn
            // does not allow, by part, held until its end resolves the call.
            let mut held: BTreeMap<Part, Vec<StreamEvent>> = BTreeMap::new();

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
                                yield Ok(Surfaced::Provider(item));
                            }
                        }
                        StreamedTurnEvent::EmitToolCallDelta => {
                            if self.observes_tool_call_delta
                                && let Some(Item::Event(StreamEvent::Arguments { part, json })) =
                                    item_slot.as_ref()
                                && let Some(reason) = observe_action(
                                    runner
                                        .config.hooks
                                        .on_tool_call_delta(
                                            hook_ctx,
                                            ToolCallDelta {
                                                part: *part,
                                                tool_name: assembler
                                                    .streaming_tool_name(part.index())
                                                    .map(ToolName::as_str)
                                                    .unwrap_or_default(),
                                                delta: json,
                                                aggregated: assembler
                                                    .aggregated_arguments(part.index())
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
                                yield Ok(Surfaced::Provider(item));
                            }
                        }
                        StreamedTurnEvent::HoldToolCall => {
                            if let Some(Item::Event(event)) = item_slot.take() {
                                held.entry(event.part()).or_default().push(event);
                            }
                        }
                        StreamedTurnEvent::EmitToolCall { call } => {
                            // The call's end is the ingested item; a repaired
                            // call's end carries the repaired name.
                            let end = match item_slot.take() {
                                Some(Item::Event(StreamEvent::End { part, .. })) => Some(part),
                                _ => None,
                            };
                            // A repaired call's held items stream now, under
                            // the repaired name; a live call holds none.
                            let replay = end
                                .and_then(|part| held.remove(&part))
                                .unwrap_or_default();
                            let mut aggregated = String::new();
                            for mut event in replay {
                                if let StreamEvent::Start { name, .. } = &mut event {
                                    *name = Some(call.function.name.clone());
                                }
                                if let StreamEvent::Arguments { json, .. } = &event {
                                    aggregated.push_str(json);
                                }
                                if self.observes_tool_call_delta
                                    && let StreamEvent::Arguments { part, json } = &event
                                    && let Some(reason) = observe_action(
                                        runner
                                            .config.hooks
                                            .on_tool_call_delta(
                                                hook_ctx,
                                                ToolCallDelta {
                                                    part: *part,
                                                    tool_name: call.function.name.as_str(),
                                                    delta: json,
                                                    aggregated: &aggregated,
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
                                yield Ok(Surfaced::Provider(Item::Event(event)));
                            }
                            if let Some(part) = end {
                                yield Ok(Surfaced::Provider(Item::Event(
                                    StreamEvent::End {
                                        part,
                                        content: AssistantContent::ToolCall(call),
                                    },
                                )));
                            }
                        }
                        StreamedTurnEvent::InvalidToolCall(invalid) => {
                            // The rejected call's items stay held: a repair
                            // replays them, and any other resolution drops them.
                            let rejected = match item_slot.as_ref() {
                                Some(Item::Event(event)) => Some(event.part()),
                                _ => None,
                            };
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
                                    if let Some(part) = rejected {
                                        held.remove(&part);
                                    }
                                    item_slot = None;
                                }
                                StreamedResolution::Repaired { .. } => {
                                    // The repaired call flows through the same event
                                    // handling above; the turn is now recovered.
                                    turn_recovered = true;
                                    events.extend(assembler.resolve_pending_invalid(&resolution));
                                }
                                StreamedResolution::TurnAbandoned => {
                                    assembler.resolve_pending_invalid(&resolution);
                                    // Only provider fragments are held; the turn's
                                    // committed calls and answers stream as the
                                    // engine's projection of history.
                                    held.clear();
                                    // The abandoned reply still reports its usage
                                    // when it ends; one that already ended with the
                                    // rejected call's error has none to record.
                                    if !ended {
                                        match scope.finish_stream(stream, run, &chat_span).await {
                                            Ok((_, call)) => {
                                                yield Ok(Surfaced::CompletionCall(call));
                                            }
                                            Err(err) => {
                                                yield Err(err);
                                                return;
                                            }
                                        }
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
            let response = match scope.finish_stream(stream, run, &chat_span).await {
                Ok((response, call)) => {
                    yield Ok(Surfaced::CompletionCall(call));
                    response
                }
                Err(err) => {
                    yield Err(err);
                    return;
                }
            };

            let streamed_turn = assembler.finish(response);
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
            let settlement = settle_model_turn(
                runner,
                hook_ctx,
                run,
                scope,
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
                    yield Ok(Surfaced::ModelTurnRetried {
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
                        scope.choice(),
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
                scope.choice(),
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

    fn final_item(&self, response: &PromptResponse) -> Option<Surfaced> {
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
        Some(Surfaced::Final(
            PromptResponse::from_content(final_choice, response.usage)
                .with_completion_calls(response.completion_calls.clone())
                .with_messages(response.messages.clone())
                .with_memory_append(response.memory_append.clone()),
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

/// Settle a parked model turn: close the completion `scope` with
/// [`AgentHook::on_outcome`] (a replacement lands on the parked turn, a
/// `Cancelled` replacement terminates), then
/// [`AgentHook::on_model_turn_finished`] and apply its action to the sans-IO
/// run. Both media close an accepted turn in this slot, after the run
/// validated the answer's tool calls and while the turn can still be
/// replaced. A `recovered` turn fires neither hook and advances; the engine
/// closes its scope, observe-only, as it does for every attempt that ends
/// unsettled. Both drivers call this once per accepted attempt, so retry
/// history, tool-turn rejection, and state transitions cannot diverge by
/// medium. `scope` holds this attempt's response with the choice the hooks
/// see, and `max_tokens` the cap it was prepared with. The callers own what
/// happens next: both record the accepted turn's telemetry, and the streaming
/// driver also keeps the content its final item surfaces.
pub(crate) async fn settle_model_turn<M>(
    runner: &AgentRunner,
    hook_ctx: &HookContext,
    run: &mut AgentRun,
    scope: &mut CompletionScope<M>,
    max_tokens: Option<u64>,
    recovered: bool,
) -> Result<ModelTurnDecision, PromptError> {
    if recovered {
        return Ok(ModelTurnDecision::Advance { replaced: None });
    }
    let (action, response) = scope.close(runner, hook_ctx).await?;
    let hooks = &runner.config.hooks;
    let mut replaced: Option<Vec<AssistantContent>> = None;
    match action {
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
    let identity = response.identity();
    let finish_reason = response.finish_reason();
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
        runner,
        ctx,
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
    runner: &AgentRunner,
    ctx: &HookContext,
    key: &rig_core::effect::HandlerKey,
    kind: EffectKind,
) -> Result<Outcome, ErrorReport> {
    let scope = DispatchScope::open(runner, ctx, kind, None)
        .await
        .map_err(|(_, report)| report)?;
    let bus = runner.config.bus.dispatcher();
    let outcome = bus
        .dispatch_with(key, scope.kind().clone(), scope.options())
        .await;
    match scope.close(runner, ctx, &outcome, None).await {
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

/// Why a tool call ended the run instead of answering: a hook stopped it,
/// cancelling with the run's own history, or a failure the run returns.
pub(crate) enum ToolExit {
    Stop(String),
    Fail(PromptError),
}

/// Outcome of [`run_single_tool`]: the tool result plus whether the tool's
/// body ran (and the effective call) or a hook skipped it.
pub(crate) struct ToolCallOutcome {
    /// The tool result delivered to the model (a real output, a redacted
    /// replacement, or a hook skip reason).
    pub result: ToolResult,
    /// The effective call when the tool's body ran: the model's call with any
    /// [`DispatchAction::Patch`] hook rewrite applied, reported as
    /// [`ToolExecutionCommitted`](crate::agent::streaming::MultiTurnStreamItem::ToolExecutionCommitted).
    /// `None` when a hook skipped the call ([`DispatchAction::skip`]).
    pub executed: Option<ToolCall>,
}

/// Execute a single tool call through the dispatch boundary and shape the
/// result. **Shared by the blocking and streaming drivers** so a tool call
/// behaves identically in both: same hook events (`on_dispatch` before,
/// `on_outcome` after), same fail-closed skip/terminate handling, and the
/// same result shaping. A hook's skip becomes [`ToolResult::skipped`].
/// Records `gen_ai.tool.*` on the current span. A hook that terminates the
/// run returns [`ToolExit::Stop`]. Returns whether the tool body executed via
/// [`ToolCallOutcome::executed`].
pub(crate) async fn run_single_tool(
    runner: &AgentRunner,
    ctx: &HookContext,
    tool_snapshot: &ToolCatalog,
    tool_call: &ToolCall,
) -> Result<ToolCallOutcome, ToolExit> {
    let record_content = runner.config.record_telemetry_content;
    let tool_name = tool_call.function.name.as_str();
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
        args: effective_args,
    } = dispatch_tool_call(runner, ctx, tool_snapshot, tool_call, args.clone()).await?;

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
    let executed = executed.then(|| {
        let mut effective_tool_call = tool_call.clone();
        effective_tool_call.function =
            ToolFunction::parse(tool_call.function.name.clone(), &effective_args);
        effective_tool_call
    });
    // Outcome metadata describes the execution itself, while result content
    // follows the same presentation policy as the model: what the outcome
    // hook let through is what telemetry records.
    record_tool_result(&tool_span, &exec);
    if record_content {
        tool_span.record("gen_ai.tool.call.result", exec.output().render());
    }
    Ok(ToolCallOutcome {
        result: exec,
        executed,
    })
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
    const STREAMS: bool = false;

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
        scope: &'a mut CompletionScope<Self>,
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

            let dispatched = scope
                .dispatch_response(runner, &prepared.model)
                .instrument(chat_span.clone())
                .await;
            let response = match dispatched {
                Ok(response) => response,
                Err(err) => {
                    yield Err(err);
                    return;
                }
            };

            let mut outcome = match run.model_response(ModelTurn::from_policy(
                response,
                prepared.policy,
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
                            runner,
                            hook_ctx,
                            run,
                            scope,
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

    fn final_item(&self, _response: &PromptResponse) -> Option<Surfaced> {
        // The blocking surface folds the engine and discards the final item, so
        // building it (an extra full-response clone) is skipped entirely.
        None
    }
}

#[cfg(test)]
#[allow(irrefutable_let_patterns, unreachable_patterns)]
mod tests;

/// A tool call answered at the dispatch boundary: the result, whether the
/// tool's body ran, and the arguments it ran with (after any hook's patch).
pub(crate) struct ToolCallDispatch {
    pub(crate) result: ToolResult,
    /// Disposition at the dispatch boundary, before outcome presentation hooks.
    pub(crate) executed: bool,
    pub(crate) args: String,
}

/// Dispatch a tool call through the agent's bus at the dispatch boundary:
/// `on_dispatch` before (patch the arguments, skip with a reason, or stop),
/// the bus, `on_outcome` after (replace what the run sees, or stop). A stop
/// is [`ToolExit::Stop`], and a bus that cannot serve the call fails the run;
/// every other failure is the tool result the model sees.
pub(crate) async fn dispatch_tool_call(
    runner: &AgentRunner,
    ctx: &HookContext,
    tool_snapshot: &ToolCatalog,
    tool_call: &ToolCall,
    args: String,
) -> Result<ToolCallDispatch, ToolExit> {
    let (tool_name, call_id) = (tool_call.function.name.as_str(), &tool_call.id);
    let kind = EffectKind::ToolCall {
        name: tool_name.to_owned(),
        args,
    };
    // The context the tool runs with travels beside the effect, never in it
    // (format 5): the hooks see it on the event, the bus carries it to the
    // tool's sink, and what the tool published comes back the same way.
    let inbound = runner.tool_context.for_dispatch();
    let (scope, denied) = match DispatchScope::open(runner, ctx, kind, Some((call_id, &inbound)))
        .await
    {
        Ok(scope) => (scope, None),
        Err((_, report)) if report.kind == ErrorKind::Cancelled => {
            return Err(ToolExit::Stop(report.message));
        }
        Err((scope, report)) => {
            tracing::info!(tool_name = tool_name, reason = %report.message, "Tool call rejected");
            (scope, Some(report))
        }
    };
    let args = match scope.kind() {
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
                let pending = runner.config.bus.dispatcher().dispatch_with(
                    key.raw(),
                    scope.kind().clone(),
                    scope.options().with_tool_context(inbound),
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
    let context = published.unwrap_or_else(|| runner.tool_context.for_dispatch());
    let outcome = match scope
        .close(runner, ctx, &outcome, Some((call_id, &context)))
        .await
    {
        OutcomeAction::Proceed => outcome,
        OutcomeAction::Replace(replaced) => replaced,
    };
    let result = match outcome {
        Ok(Outcome::ToolResult { result }) => result,
        Ok(other) => ToolResult::failed(crate::tool::ToolExecutionError::other(format!(
            "the tool handler answered with a {} outcome",
            other.family()
        ))),
        // A hook that observed the result stopped the run.
        Err(report) if report.kind == ErrorKind::Cancelled => {
            return Err(ToolExit::Stop(report.message));
        }
        // A layer on the tool's key denied the call: the model sees the
        // skipped result, as it does for a hook's denial.
        Err(report) if report.kind == ErrorKind::Denied => {
            tracing::info!(tool_name = tool_name, reason = %report.message, "Tool call denied");
            ToolResult::skipped(report.message)
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
            return Err(ToolExit::Fail(PromptError::Report(report)));
        }
        Err(report) => ToolResult::failed(report.into()),
    };
    Ok(ToolCallDispatch {
        result,
        executed,
        args,
    })
}
