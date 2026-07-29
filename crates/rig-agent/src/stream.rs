//! The streaming session driver: an explicit, callback-free state machine.
//!
//! [`AgentStream`] is the streaming sibling of
//! [`AgentSession`](crate::session::AgentSession): the host pulls
//! [`AgentStreamItem`]s with [`AgentStream::next_item`], observes deltas as
//! they arrive, and answers the same decision inboxes (tools, invalid
//! calls, and — policy-gated — request patching and turn acceptance).
//! Backpressure is structural (stop calling `next_item`), and stopping is
//! owning: drop or stop polling and the [`AgentRun`] state remains intact
//! and serializable.

use std::collections::VecDeque;
use std::sync::Arc;

use futures::StreamExt;

use crate::agent::hook::{
    CompletionCallAction, InvalidToolCallAction, ModelTurnAction, RequestPatch, ToolCallAction,
    ToolResultAction,
};
use crate::agent::prepare::{ToolCatalog, prepare_request};
use crate::agent::prompt_request::streaming::record_usage_on_span;
use crate::agent::prompt_request::tool_result_output;
use crate::agent::run::{
    AgentRun, AgentRunStep, DEFAULT_OUTPUT_RETRIES, PartialStreamedTurn, PendingToolCall,
    StreamedInvalidToolCall, StreamedResolution, StreamedTurnAssembler, StreamedTurnEvent,
};
use crate::agent::runner::{
    SessionSpanParams, acquire_agent_span, new_session_chat_streaming_span,
};
use crate::agent::{AgentConfig, InvalidToolCallContext, PromptResponse, UNKNOWN_AGENT_NAME};
use crate::completion::{Message, PromptError, Usage};
use rig_core::OneOrMany;
use rig_core::message::{AssistantContent, ToolCall, UserContent};
use rig_core::streaming::{
    StreamFinal, StreamedAssistantContent, StreamedUserContent, StreamingCompletionResponse,
};
use tracing_futures::Instrument;

use crate::provider::{self, ProviderConfig, Runtime};
use crate::session::SessionPolicy;

/// One item pulled from an [`AgentStream`].
///
/// Deliberately exhaustive, like
/// [`SessionEvent`](crate::session::SessionEvent): a new decision-bearing
/// variant must fail to compile in every streaming host.
///
/// Serializable: every payload is owned data, so an item can be persisted,
/// forwarded over a wire, or replayed by a durable host.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AgentStreamItem {
    /// A streamed assistant item: text/reasoning/tool-call deltas, complete
    /// tool calls (surfaced in call order immediately before
    /// [`Self::ToolCallsReady`], preserving the announce-before-execute
    /// contract), and the provider's terminal
    /// [`StreamFinal`](rig_core::streaming::StreamFinal).
    Assistant(StreamedAssistantContent),
    /// Exactly one per provider call: the recorded completion-call entry.
    CompletionCall(crate::agent::CompletionCall),
    /// `policy.surface_completion_calls` — answer via
    /// [`AgentStream::reply_before_call`] (pre-build, like the session).
    BeforeModelCall {
        /// This turn's prompt message.
        prompt: Message,
        /// The history preceding it.
        history: Vec<Message>,
        /// One-based model-call index.
        turn: usize,
    },
    /// `policy.surface_model_turns` — answer via
    /// [`AgentStream::reply_turn`]; a retry surfaces
    /// [`Self::ModelTurnRetried`] so consumers discard the turn's
    /// provisional deltas.
    TurnFinished {
        /// One-based model-call index.
        turn: usize,
        /// Canonicalized assistant content parked for acceptance.
        content: OneOrMany<AssistantContent>,
        /// Usage reported for the turn.
        usage: Usage,
    },
    /// A turn was rolled back (hook retry or invalid-call recovery); its
    /// earlier deltas were provisional.
    ModelTurnRetried {
        /// One-based index of the retried model call.
        turn: usize,
    },
    /// The model called an unknown/disallowed tool mid-stream. Answer via
    /// [`AgentStream::resolve_invalid`].
    InvalidToolCall(InvalidToolCallContext),
    /// `policy.surface_tool_calls`: one executable call awaiting its
    /// pre-execution decision. Answer via
    /// [`AgentStream::reply_tool_call`] with classic-runner semantics:
    /// `Run` executes as-is, `Rewrite` replaces the arguments, `Skip`
    /// pre-resolves the call as a skipped tool result, `Stop` cancels the
    /// run. Calls carrying a preresolved result pass through without
    /// surfacing.
    ToolCallPending {
        /// The call awaiting the decision.
        call: PendingToolCall,
    },
    /// Execute these calls and answer via
    /// [`AgentStream::provide_tool_results`].
    ToolCallsReady(Vec<PendingToolCall>),
    /// `policy.surface_tool_results`: one provided result awaiting its
    /// post-execution decision. Answer via
    /// [`AgentStream::reply_tool_result`]: `Keep` commits as provided,
    /// `Rewrite` replaces the model-visible presentation, `Stop` cancels
    /// the run. Results preresolved by invalid-call recovery pass through
    /// verbatim without surfacing; skip-gated calls do surface, exactly as
    /// the classic runner fires its tool-result hook for skips.
    ToolResultReady {
        /// The executed (or skipped) tool call, with effective arguments.
        call: ToolCall,
        /// The model-visible result content as provided.
        result: UserContent,
    },
    /// A tool call's result was committed to history (post-batch, call
    /// order).
    ToolExecutionCommitted {
        /// The executed tool call.
        tool_call: ToolCall,
        /// Rig correlation id for its stream items.
        internal_call_id: String,
    },
    /// A committed tool result, in call order.
    User(StreamedUserContent),
    /// The final response; the stream ends after this item.
    Final(PromptResponse),
}

/// The in-flight provider stream plus its sans-IO assembler.
struct ActiveTurn {
    stream: StreamingCompletionResponse,
    assembler: StreamedTurnAssembler,
    turn: usize,
    /// The per-call `chat_streaming` span the provider stream is polled
    /// under, mirroring the classic streaming driver.
    chat_span: tracing::Span,
}

/// The host decision the stream is waiting for.
enum Pending {
    None,
    BeforeCall {
        prompt: Message,
        history: Vec<Message>,
    },
    TurnReply,
    Invalid {
        partial: PartialStreamedTurn,
        invalid: StreamedInvalidToolCall,
    },
    /// `policy.surface_tool_calls`: pre-execution decisions in flight.
    ToolCallGate {
        /// The calls as the model emitted them, for the
        /// announce-before-execute items.
        originals: Vec<PendingToolCall>,
        /// Calls not yet decided, in call order.
        remaining: VecDeque<PendingToolCall>,
        /// Calls already decided (rewrites/skips applied).
        decided: Vec<PendingToolCall>,
        /// The call surfaced and awaiting [`AgentStream::reply_tool_call`].
        current: Option<PendingToolCall>,
        /// Ids skipped via [`ToolCallAction::Skip`].
        skipped_ids: Vec<String>,
    },
    Tools {
        calls: Vec<PendingToolCall>,
        /// Ids skipped via [`ToolCallAction::Skip`] — these still surface
        /// their (synthetic) result under `surface_tool_results`.
        skipped_ids: Vec<String>,
    },
    /// `policy.surface_tool_results`: post-execution decisions in flight.
    ToolResultGate {
        /// One entry per pending call, in call order.
        entries: Vec<ResultGateEntry>,
        /// Results without a matching call, passed through for the run to
        /// validate.
        extras: Vec<UserContent>,
        /// Index of the entry surfaced (when `awaiting`) or the next scan
        /// position.
        cursor: usize,
        /// Whether a [`AgentStreamItem::ToolResultReady`] item awaits
        /// [`AgentStream::reply_tool_result`].
        awaiting: bool,
    },
    Finished,
}

/// One paired call/result awaiting the result-gate decision.
struct ResultGateEntry {
    call: PendingToolCall,
    internal_call_id: String,
    result: Option<UserContent>,
    /// Whether this entry surfaces as [`AgentStreamItem::ToolResultReady`].
    surface: bool,
}

/// The streaming driver for one agent run. See the [module docs](self).
pub struct AgentStream {
    /// The agent's model-free configuration.
    pub config: AgentConfig,
    /// The provider fulfilling model calls.
    pub provider: ProviderConfig,
    /// Tool definitions advertised each turn (the host executes them).
    pub tools: ToolCatalog,
    /// Which decision points surface.
    pub policy: SessionPolicy,
    run: AgentRun,
    rt: Arc<Runtime>,
    next_patch: Option<RequestPatch>,
    pending: Pending,
    active: Option<ActiveTurn>,
    buffered: VecDeque<AgentStreamItem>,
    last_final: Option<StreamFinal>,
    /// The run-level `invoke_agent` span (created, or adopted from the
    /// caller's ambient span), mirroring the classic drivers.
    agent_span: tracing::Span,
    /// Whether this stream created `agent_span` — run-level usage is only
    /// recorded onto a span the stream owns.
    created_agent_span: bool,
}

impl AgentStream {
    /// Create a streaming session for one prompt.
    pub fn new(
        config: AgentConfig,
        provider: ProviderConfig,
        rt: Arc<Runtime>,
        prompt: impl Into<Message>,
    ) -> Self {
        let prompt: Message = prompt.into();
        let (agent_span, created_agent_span) = acquire_agent_span(
            config.name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME),
            config.preamble.as_deref(),
            config.record_telemetry_content,
        );
        if config.record_telemetry_content
            && let Some(text) = prompt.rag_text()
        {
            agent_span.record("gen_ai.prompt", text);
        }
        let run = AgentRun::new(prompt)
            .max_turns(config.max_turns.unwrap_or(1))
            .max_invalid_tool_call_retries(config.max_invalid_tool_call_retries)
            .with_output_validation(
                config
                    .output_schema
                    .as_ref()
                    .map(|schema| schema.as_value().clone()),
                DEFAULT_OUTPUT_RETRIES,
            );
        let run = match config.tool_choice.clone() {
            Some(tool_choice) => run.with_tool_choice(tool_choice),
            None => run,
        };
        Self {
            config,
            provider,
            tools: ToolCatalog::default(),
            policy: SessionPolicy::default(),
            run,
            rt,
            next_patch: None,
            pending: Pending::None,
            active: None,
            buffered: VecDeque::new(),
            last_final: None,
            agent_span,
            created_agent_span,
        }
    }

    /// Set the input chat history preceding the prompt.
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.run = std::mem::replace(&mut self.run, AgentRun::new("")).with_history(history);
        self
    }

    /// Advertise these tool definitions each turn.
    pub fn with_tools(mut self, catalog: ToolCatalog) -> Self {
        self.tools = catalog;
        self
    }

    /// Set the surfacing policy.
    pub fn with_policy(mut self, policy: SessionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Live aggregated usage — valid mid-stream and after failures.
    pub fn usage(&self) -> Usage {
        self.run.usage()
    }

    /// The underlying run state.
    pub fn run_state(&self) -> &AgentRun {
        &self.run
    }

    /// The last provider terminal record, for observation parity.
    pub fn last_response(&self) -> Option<&StreamFinal> {
        self.last_final.as_ref()
    }

    /// Terminate the current provider stream early (the run state stays
    /// consistent; the turn is abandoned on the next poll).
    pub fn close_turn(&mut self) {
        if let Some(active) = &mut self.active {
            active.stream.cancel();
        }
    }

    /// Merge a per-turn request patch consumed by the next model call.
    pub fn patch_next_turn(&mut self, patch: RequestPatch) {
        self.next_patch = Some(match self.next_patch.take() {
            Some(existing) => existing.merge(patch),
            None => patch,
        });
    }

    /// Pull the next item. Returns `None` after [`AgentStreamItem::Final`].
    pub async fn next_item(&mut self) -> Option<Result<AgentStreamItem, PromptError>> {
        loop {
            if let Some(item) = self.buffered.pop_front() {
                return Some(Ok(item));
            }
            match &self.pending {
                Pending::Finished => return None,
                Pending::None => {}
                Pending::ToolCallGate { current: None, .. } => {
                    // Surface the next pre-execution decision (or the
                    // decided batch) into the buffer.
                    match self.step_tool_call_gate() {
                        Ok(()) => continue,
                        Err(error) => return Some(Err(error)),
                    }
                }
                Pending::ToolResultGate {
                    awaiting: false, ..
                } => {
                    // Surface the next post-execution decision, or commit
                    // the batch and continue the run.
                    match self.step_tool_result_gate() {
                        Ok(()) => continue,
                        Err(error) => return Some(Err(error)),
                    }
                }
                Pending::BeforeCall { .. }
                | Pending::TurnReply
                | Pending::Invalid { .. }
                | Pending::Tools { .. }
                | Pending::ToolCallGate { .. }
                | Pending::ToolResultGate { .. } => {
                    return Some(Err(self.run.cancel_error(
                        "next_item called while a decision inbox awaits its answer",
                    )));
                }
            }

            if self.active.is_some() {
                match self.poll_active_turn().await {
                    Ok(()) => continue,
                    Err(error) => return Some(Err(error)),
                }
            }

            match self.begin_next_step().await {
                Ok(()) => continue,
                Err(error) => return Some(Err(error)),
            }
        }
    }

    /// Advance the run machine when no provider stream is open.
    async fn begin_next_step(&mut self) -> Result<(), PromptError> {
        match self.run.next_step()? {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                if self.policy.surface_completion_calls {
                    self.pending = Pending::BeforeCall {
                        prompt: prompt.clone(),
                        history: history.clone(),
                    };
                    self.buffered.push_back(AgentStreamItem::BeforeModelCall {
                        prompt,
                        history,
                        turn,
                    });
                    return Ok(());
                }
                self.open_turn(prompt, history, turn).await
            }
            AgentRunStep::CallTools { calls } => {
                if self.policy.surface_tool_calls {
                    // Per-call decisions first; announce-before-execute
                    // items surface once the whole batch is decided.
                    self.pending = Pending::ToolCallGate {
                        originals: calls.clone(),
                        remaining: calls.into(),
                        decided: Vec::new(),
                        current: None,
                        skipped_ids: Vec::new(),
                    };
                    return Ok(());
                }
                // Announce-before-execute: complete tool calls surface in
                // call order right before the decision point.
                for call in &calls {
                    self.buffered.push_back(AgentStreamItem::Assistant(
                        StreamedAssistantContent::ToolCall {
                            tool_call: call.tool_call.clone(),
                            internal_call_id: call
                                .internal_call_id
                                .clone()
                                .unwrap_or_else(rig_core::id::generate),
                        },
                    ));
                }
                self.buffered
                    .push_back(AgentStreamItem::ToolCallsReady(calls.clone()));
                self.pending = Pending::Tools {
                    calls,
                    skipped_ids: Vec::new(),
                };
                Ok(())
            }
            AgentRunStep::Done(response) => {
                // Run-level telemetry, mirroring the classic streaming
                // driver: usage only, and only onto a span this stream
                // created — never pollute a caller-supplied outer span.
                if self.created_agent_span {
                    record_usage_on_span(&self.agent_span, response.usage);
                }
                self.pending = Pending::Finished;
                self.buffered.push_back(AgentStreamItem::Final(response));
                Ok(())
            }
        }
    }

    /// Build the request and open the provider stream for one turn.
    async fn open_turn(
        &mut self,
        prompt: Message,
        history: Vec<Message>,
        turn: usize,
    ) -> Result<(), PromptError> {
        let patch = self.next_patch.take().unwrap_or_default();
        let prepared = prepare_request(
            &self.config,
            &self.tools,
            self.provider.descriptor().composes_native_output_with_tools,
            prompt,
            &history,
            self.run.output_tool_name(),
            Some(&patch),
        )?;
        self.run
            .set_output_tool_name(prepared.output_tool_name.clone());
        // The per-call `chat_streaming` span — identical shape to the classic
        // streaming driver's, parented into the ambient span tree.
        let chat_span = new_session_chat_streaming_span(
            &SessionSpanParams {
                record_telemetry_content: self.config.record_telemetry_content,
                agent_name: self.config.name.as_deref(),
            },
            self.config.preamble.as_deref(),
        );
        let stream = match provider::open_stream(&self.provider, &self.rt, prepared.request)
            .instrument(chat_span.clone())
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                // Transient provider failure: return to the pre-call state so
                // a later next_item() retries the call instead of wedging the
                // run in AwaitingModel forever.
                self.run.abandon_pending_model_call();
                return Err(PromptError::from(error));
            }
        };
        self.active = Some(ActiveTurn {
            stream,
            assembler: StreamedTurnAssembler::new(
                prepared.executable_tool_names,
                prepared.allowed_tool_names,
            ),
            turn,
            chat_span,
        });
        Ok(())
    }

    /// Poll the open provider stream once, translating assembler events
    /// into buffered items or pending decisions.
    async fn poll_active_turn(&mut self) -> Result<(), PromptError> {
        let Some(active) = &mut self.active else {
            return Ok(());
        };
        let chat_span = active.chat_span.clone();
        match active.stream.next().instrument(chat_span).await {
            Some(Ok(item)) => {
                let events = active.assembler.ingest(&item).map_err(PromptError::from)?;
                for event in events {
                    match event {
                        StreamedTurnEvent::EmitIngested => {
                            self.buffered
                                .push_back(AgentStreamItem::Assistant(item.clone()));
                        }
                        StreamedTurnEvent::EmitToolCallDelta {
                            id,
                            internal_call_id,
                            content,
                        } => {
                            self.buffered.push_back(AgentStreamItem::Assistant(
                                StreamedAssistantContent::ToolCallDelta {
                                    id,
                                    internal_call_id,
                                    content,
                                },
                            ));
                        }
                        StreamedTurnEvent::InvalidToolCall(invalid) => {
                            let partial = active
                                .assembler
                                .partial_turn(active.stream.message_id.clone());
                            let context = self
                                .run
                                .streamed_invalid_tool_call_context(&partial, &invalid);
                            self.pending = Pending::Invalid {
                                partial,
                                invalid: *invalid,
                            };
                            self.buffered
                                .push_back(AgentStreamItem::InvalidToolCall(context));
                            return Ok(());
                        }
                        StreamedTurnEvent::Completed { usage, emit_final } => {
                            let call = self.run.record_streamed_completion_call(usage)?;
                            self.buffered
                                .push_back(AgentStreamItem::CompletionCall(call));
                            if let StreamedAssistantContent::Final(final_record) = &item {
                                self.last_final = Some(final_record.clone());
                                if emit_final {
                                    self.buffered
                                        .push_back(AgentStreamItem::Assistant(item.clone()));
                                }
                            }
                        }
                    }
                }
                Ok(())
            }
            Some(Err(error)) => Err(PromptError::from(error)),
            None => {
                // EOF: assemble and commit the turn, then continue the run.
                let Some(active) = self.active.take() else {
                    return Ok(());
                };
                let ActiveTurn {
                    stream,
                    assembler,
                    turn,
                    chat_span: _,
                } = active;
                let streamed = assembler.finish(stream.message_id.clone(), &stream.choice);
                self.run.streamed_turn(streamed)?;
                if self.policy.surface_model_turns
                    && let Some(content) = self.run.accepted_turn_choice()
                {
                    self.pending = Pending::TurnReply;
                    // Per-turn usage, not the run aggregate: the last recorded
                    // completion call is this turn's (recorded from the
                    // stream's terminal event, or as a zero-usage fallback by
                    // `streamed_turn` when the provider reported none).
                    let usage = self
                        .run
                        .completion_calls()
                        .last()
                        .map_or_else(Usage::new, |call| call.usage);
                    self.buffered.push_back(AgentStreamItem::TurnFinished {
                        turn,
                        content,
                        usage,
                    });
                }
                Ok(())
            }
        }
    }

    /// Pull the next non-tool item, answering every
    /// [`AgentStreamItem::ToolCallsReady`] batch through the executor
    /// (classic tool-loop semantics; see
    /// [`ToolExecutor`](crate::executor::ToolExecutor)). Every other item —
    /// deltas, committed tool results, execution markers, policy-gated
    /// decision points — is returned to the caller unchanged; decision items
    /// still expect their usual answers before the next poll.
    ///
    /// Tool failures stay model-visible as failed tool results, exactly as
    /// the classic loop delivered them.
    pub async fn next_item_with_tools(
        &mut self,
        executor: &crate::executor::ToolExecutor,
    ) -> Option<Result<AgentStreamItem, PromptError>> {
        loop {
            match self.next_item().await? {
                Ok(AgentStreamItem::ToolCallsReady(calls)) => {
                    let batch = executor.execute_batch(&calls).await;
                    if let Err(error) = self.provide_tool_results(batch.results) {
                        return Some(Err(error));
                    }
                }
                other => return Some(other),
            }
        }
    }

    /// Answer [`AgentStreamItem::BeforeModelCall`].
    pub async fn reply_before_call(
        &mut self,
        action: CompletionCallAction,
    ) -> Result<(), PromptError> {
        let Pending::BeforeCall { prompt, history } =
            std::mem::replace(&mut self.pending, Pending::None)
        else {
            return Err(self
                .run
                .cancel_error("reply_before_call without a pending BeforeModelCall item"));
        };
        match action {
            CompletionCallAction::Continue => {}
            CompletionCallAction::Patch(patch) => self.patch_next_turn(patch),
            CompletionCallAction::Stop(reason) => {
                return Err(self.run.cancel_error(reason));
            }
        }
        let turn = self.run.turn();
        self.open_turn(prompt, history, turn).await
    }

    /// Answer [`AgentStreamItem::TurnFinished`].
    pub fn reply_turn(&mut self, action: ModelTurnAction) -> Result<(), PromptError> {
        if !matches!(self.pending, Pending::TurnReply) {
            return Err(self
                .run
                .cancel_error("reply_turn without a pending TurnFinished item"));
        }
        self.pending = Pending::None;
        match action {
            ModelTurnAction::Continue => Ok(()),
            ModelTurnAction::Retry(request) => {
                let turn = self.run.turn();
                self.run.retry_model_turn(request)?;
                self.buffered
                    .push_back(AgentStreamItem::ModelTurnRetried { turn });
                Ok(())
            }
            ModelTurnAction::Stop(reason) => Err(self.run.cancel_error(reason)),
        }
    }

    /// Answer [`AgentStreamItem::InvalidToolCall`].
    pub async fn resolve_invalid(
        &mut self,
        action: InvalidToolCallAction,
    ) -> Result<(), PromptError> {
        let Pending::Invalid { partial, invalid } =
            std::mem::replace(&mut self.pending, Pending::None)
        else {
            return Err(self
                .run
                .cancel_error("resolve_invalid without a pending InvalidToolCall item"));
        };
        let resolution = self
            .run
            .resolve_streamed_invalid_tool_call(&partial, &invalid, action)?;
        match &resolution {
            StreamedResolution::Repaired { .. } => {
                if let Some(active) = &mut self.active {
                    for event in active.assembler.resolve_pending_invalid(&resolution) {
                        if let StreamedTurnEvent::EmitToolCallDelta {
                            id,
                            internal_call_id,
                            content,
                        } = event
                        {
                            self.buffered.push_back(AgentStreamItem::Assistant(
                                StreamedAssistantContent::ToolCallDelta {
                                    id,
                                    internal_call_id,
                                    content,
                                },
                            ));
                        }
                    }
                }
                Ok(())
            }
            StreamedResolution::TurnAbandoned {
                skipped_tool_result,
            } => {
                let turn = self.run.turn();
                if let Some(result) = skipped_tool_result {
                    self.buffered.push_back(AgentStreamItem::User(
                        StreamedUserContent::tool_result(
                            result.clone(),
                            invalid.internal_call_id.clone(),
                        ),
                    ));
                }
                // Drain the abandoned provider stream for its usage record.
                if let Some(mut active) = self.active.take() {
                    let _ = active.assembler.resolve_pending_invalid(&resolution);
                    let mut drained_usage = Usage::new();
                    while let Some(item) = active.stream.next().await {
                        if let Ok(StreamedAssistantContent::Final(final_record)) = item {
                            drained_usage = final_record.usage;
                            self.last_final = Some(final_record);
                        }
                    }
                    match self.run.record_streamed_completion_call(drained_usage) {
                        Ok(call) => {
                            self.buffered
                                .push_back(AgentStreamItem::CompletionCall(call));
                        }
                        Err(error) => {
                            // The run rejected the record (e.g. one was
                            // already recorded for this turn); the drained
                            // usage is dropped rather than double-counted.
                            tracing::debug!(
                                %error,
                                "dropping usage record for abandoned streamed turn"
                            );
                        }
                    }
                }
                self.buffered
                    .push_back(AgentStreamItem::ModelTurnRetried { turn });
                Ok(())
            }
        }
    }

    /// Surface the next pre-execution gate event into the buffer: the next
    /// undecided call as [`AgentStreamItem::ToolCallPending`], or the
    /// announce-before-execute items plus
    /// [`AgentStreamItem::ToolCallsReady`] once every call is decided.
    /// Calls carrying a preresolved result pass through undecided, exactly
    /// as the classic driver returns them verbatim without firing hooks.
    fn step_tool_call_gate(&mut self) -> Result<(), PromptError> {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::ToolCallGate {
                originals,
                mut remaining,
                mut decided,
                current: None,
                skipped_ids,
            } => {
                while let Some(next) = remaining.pop_front() {
                    if next.preresolved_result.is_some() {
                        decided.push(next);
                        continue;
                    }
                    let call = next.clone();
                    self.pending = Pending::ToolCallGate {
                        originals,
                        remaining,
                        decided,
                        current: Some(next),
                        skipped_ids,
                    };
                    self.buffered
                        .push_back(AgentStreamItem::ToolCallPending { call });
                    return Ok(());
                }
                // Announce-before-execute: the model's calls, in call order,
                // right before the decision point.
                for call in &originals {
                    self.buffered.push_back(AgentStreamItem::Assistant(
                        StreamedAssistantContent::ToolCall {
                            tool_call: call.tool_call.clone(),
                            internal_call_id: call
                                .internal_call_id
                                .clone()
                                .unwrap_or_else(rig_core::id::generate),
                        },
                    ));
                }
                self.buffered
                    .push_back(AgentStreamItem::ToolCallsReady(decided.clone()));
                self.pending = Pending::Tools {
                    calls: decided,
                    skipped_ids,
                };
                Ok(())
            }
            other => {
                self.pending = other;
                Err(self
                    .run
                    .cancel_error("tool-call gate advanced without pending decisions"))
            }
        }
    }

    /// Answer [`AgentStreamItem::ToolCallPending`], mirroring the classic
    /// runner's pre-execution semantics (see
    /// [`AgentSession::reply_tool_call`](crate::session::AgentSession::reply_tool_call)).
    ///
    /// # Errors
    /// [`ToolCallAction::Stop`] cancels the run; calling without a pending
    /// [`AgentStreamItem::ToolCallPending`] item is a protocol violation.
    pub fn reply_tool_call(&mut self, action: ToolCallAction) -> Result<(), PromptError> {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::ToolCallGate {
                originals,
                remaining,
                mut decided,
                current: Some(mut call),
                mut skipped_ids,
            } => {
                match action {
                    ToolCallAction::Run => decided.push(call),
                    ToolCallAction::Rewrite(args) => {
                        call.tool_call.function.arguments = args;
                        decided.push(call);
                    }
                    ToolCallAction::Skip(reason) => {
                        // Mirror run_single_tool: the skip becomes a
                        // `ToolResult::skipped` presentation delivered to
                        // the model without executing the body.
                        let skipped = crate::tool::ToolResult::skipped(reason);
                        let content = tool_result_output(
                            call.tool_call.id.clone(),
                            call.tool_call.call_id.clone(),
                            skipped.output().clone(),
                        );
                        skipped_ids.push(call.tool_call.id.clone());
                        call.preresolved_result = Some(content);
                        decided.push(call);
                    }
                    ToolCallAction::Stop(reason) => {
                        return Err(self.run.cancel_error(reason));
                    }
                }
                self.pending = Pending::ToolCallGate {
                    originals,
                    remaining,
                    decided,
                    current: None,
                    skipped_ids,
                };
                Ok(())
            }
            other => {
                self.pending = other;
                Err(self
                    .run
                    .cancel_error("reply_tool_call without a pending ToolCallPending item"))
            }
        }
    }

    /// Step the post-execution gate: surface the next result decision as
    /// [`AgentStreamItem::ToolResultReady`], or commit the batch and buffer
    /// the committed items in call order.
    fn step_tool_result_gate(&mut self) -> Result<(), PromptError> {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::ToolResultGate {
                entries,
                extras,
                cursor,
                awaiting: false,
            } => {
                if let Some((index, call, result)) = entries
                    .iter()
                    .enumerate()
                    .skip(cursor)
                    .find(|(_, entry)| entry.surface)
                    .and_then(|(index, entry)| {
                        entry
                            .result
                            .as_ref()
                            .map(|result| (index, entry.call.tool_call.clone(), result.clone()))
                    })
                {
                    self.pending = Pending::ToolResultGate {
                        entries,
                        extras,
                        cursor: index,
                        awaiting: true,
                    };
                    self.buffered
                        .push_back(AgentStreamItem::ToolResultReady { call, result });
                    return Ok(());
                }
                // Every decision answered: commit, then surface committed
                // items in call order, mirroring the ungated path.
                let results: Vec<UserContent> = entries
                    .iter()
                    .filter_map(|entry| entry.result.clone())
                    .chain(extras)
                    .collect();
                self.run.tool_results(results)?;
                for entry in entries {
                    self.buffered
                        .push_back(AgentStreamItem::ToolExecutionCommitted {
                            tool_call: entry.call.tool_call.clone(),
                            internal_call_id: entry.internal_call_id.clone(),
                        });
                    if let Some(UserContent::ToolResult(result)) = entry.result {
                        self.buffered.push_back(AgentStreamItem::User(
                            StreamedUserContent::tool_result(result, entry.internal_call_id),
                        ));
                    }
                }
                Ok(())
            }
            other => {
                self.pending = other;
                Err(self
                    .run
                    .cancel_error("tool-result gate advanced without pending decisions"))
            }
        }
    }

    /// Answer [`AgentStreamItem::ToolResultReady`], mirroring the classic
    /// runner's post-execution semantics (see
    /// [`AgentSession::reply_tool_result`](crate::session::AgentSession::reply_tool_result)).
    ///
    /// # Errors
    /// [`ToolResultAction::Stop`] cancels the run; calling without a
    /// pending [`AgentStreamItem::ToolResultReady`] item is a protocol
    /// violation.
    pub fn reply_tool_result(&mut self, action: ToolResultAction) -> Result<(), PromptError> {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::ToolResultGate {
                mut entries,
                extras,
                cursor,
                awaiting: true,
            } => {
                match action {
                    ToolResultAction::Keep => {}
                    ToolResultAction::Rewrite(output) => {
                        if let Some(entry) = entries.get_mut(cursor) {
                            entry.result = Some(tool_result_output(
                                entry.call.tool_call.id.clone(),
                                entry.call.tool_call.call_id.clone(),
                                output,
                            ));
                        }
                    }
                    ToolResultAction::Stop(reason) => {
                        return Err(self.run.cancel_error(reason));
                    }
                }
                self.pending = Pending::ToolResultGate {
                    entries,
                    extras,
                    cursor: cursor + 1,
                    awaiting: false,
                };
                Ok(())
            }
            other => {
                self.pending = other;
                Err(self
                    .run
                    .cancel_error("reply_tool_result without a pending ToolResultReady item"))
            }
        }
    }

    /// Answer [`AgentStreamItem::ToolCallsReady`]. Committed results and
    /// execution markers surface on subsequent [`AgentStream::next_item`]
    /// calls, in call order. Under `policy.surface_tool_results` the batch
    /// is not committed yet: each result surfaces as
    /// [`AgentStreamItem::ToolResultReady`] first, and commits once every
    /// decision is answered.
    pub fn provide_tool_results(&mut self, results: Vec<UserContent>) -> Result<(), PromptError> {
        let Pending::Tools { calls, skipped_ids } =
            std::mem::replace(&mut self.pending, Pending::None)
        else {
            return Err(self
                .run
                .cancel_error("provide_tool_results without a pending ToolCallsReady item"));
        };
        if self.policy.surface_tool_results {
            // Pair results with calls by provider id as a multiset
            // (mirroring `AgentRun::tool_results`): duplicate ids pair
            // positionally. Results preresolved by invalid-call recovery
            // pass through verbatim; gate skips do surface, exactly as
            // run_single_tool fires its tool-result hook for a hook skip.
            let mut remaining: Vec<Option<UserContent>> = results.into_iter().map(Some).collect();
            let mut entries = Vec::new();
            for call in calls {
                let matched = remaining.iter_mut().find_map(|slot| {
                    let is_match = slot.as_ref().is_some_and(|content| {
                        matches!(
                            content,
                            UserContent::ToolResult(result) if result.id == call.tool_call.id
                        )
                    });
                    if is_match { slot.take() } else { None }
                });
                let surface = matched.is_some()
                    && (call.preresolved_result.is_none()
                        || skipped_ids.contains(&call.tool_call.id));
                let internal_call_id = call
                    .internal_call_id
                    .clone()
                    .unwrap_or_else(rig_core::id::generate);
                entries.push(ResultGateEntry {
                    call,
                    internal_call_id,
                    result: matched,
                    surface,
                });
            }
            let extras: Vec<UserContent> = remaining.into_iter().flatten().collect();
            self.pending = Pending::ToolResultGate {
                entries,
                extras,
                cursor: 0,
                awaiting: false,
            };
            return Ok(());
        }
        self.run.tool_results(results.clone())?;
        // Post-commit surface items in call order. Results are consumed as a
        // multiset (mirroring `AgentRun::tool_results`): duplicate provider
        // tool-call ids within one turn each surface their own result exactly
        // once, paired positionally.
        let mut remaining: Vec<Option<rig_core::message::ToolResult>> = results
            .into_iter()
            .map(|content| match content {
                UserContent::ToolResult(result) => Some(result),
                _ => None,
            })
            .collect();
        for call in &calls {
            let internal = call
                .internal_call_id
                .clone()
                .unwrap_or_else(rig_core::id::generate);
            self.buffered
                .push_back(AgentStreamItem::ToolExecutionCommitted {
                    tool_call: call.tool_call.clone(),
                    internal_call_id: internal.clone(),
                });
            if let Some(result) = remaining.iter_mut().find_map(|slot| {
                if slot
                    .as_ref()
                    .is_some_and(|result| result.id == call.tool_call.id)
                {
                    slot.take()
                } else {
                    None
                }
            }) {
                self.buffered
                    .push_back(AgentStreamItem::User(StreamedUserContent::tool_result(
                        result, internal,
                    )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::message::{ToolFunction, ToolResult, ToolResultContent};
    use rig_core::streaming::ToolCallDeltaContent;
    use serde_json::json;

    fn tool_call(id: &str) -> ToolCall {
        ToolCall::new(
            id.to_string(),
            ToolFunction::new("add".to_string(), json!({"a": 1, "b": 2})),
        )
    }

    /// Round-trip every serializable payload family through serde and check
    /// the restored item matches structurally.
    #[test]
    fn agent_stream_items_round_trip_through_serde() {
        let items = vec![
            AgentStreamItem::Assistant(StreamedAssistantContent::ToolCallDelta {
                id: "tc_1".to_string(),
                internal_call_id: "internal_tc_1".to_string(),
                content: ToolCallDeltaContent::Name("add".to_string()),
            }),
            AgentStreamItem::BeforeModelCall {
                prompt: Message::user("hello"),
                history: vec![Message::assistant("earlier")],
                turn: 1,
            },
            AgentStreamItem::TurnFinished {
                turn: 1,
                content: OneOrMany::one(AssistantContent::text("answer")),
                usage: Usage::new(),
            },
            AgentStreamItem::ModelTurnRetried { turn: 2 },
            AgentStreamItem::ToolExecutionCommitted {
                tool_call: tool_call("tc_1"),
                internal_call_id: "internal_tc_1".to_string(),
            },
            AgentStreamItem::User(StreamedUserContent::tool_result(
                ToolResult {
                    id: "tc_1".to_string(),
                    call_id: None,
                    content: OneOrMany::one(ToolResultContent::text("3")),
                },
                "internal_tc_1".to_string(),
            )),
        ];
        for item in items {
            let serialized = serde_json::to_string(&item).expect("serialize");
            let restored: AgentStreamItem = serde_json::from_str(&serialized).expect("deserialize");
            let reserialized = serde_json::to_string(&restored).expect("reserialize");
            assert_eq!(serialized, reserialized, "round trip changed {item:?}");
        }
    }

    /// The decision-bearing tool inbox round-trips with its pending calls
    /// intact.
    #[test]
    fn tool_calls_ready_round_trips_pending_calls() {
        let item = AgentStreamItem::ToolCallsReady(vec![PendingToolCall::new(tool_call("tc_1"))]);
        let serialized = serde_json::to_string(&item).expect("serialize");
        let restored: AgentStreamItem = serde_json::from_str(&serialized).expect("deserialize");
        let AgentStreamItem::ToolCallsReady(calls) = restored else {
            panic!("expected ToolCallsReady");
        };
        assert_eq!(calls.len(), 1);
        assert_eq!(calls.first().map(|c| c.tool_call.id.as_str()), Some("tc_1"));
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    use crate::agent::prepare::ToolCatalog;
    use crate::provider::MockScript;
    use rig_core::completion::ToolDefinition;
    use rig_core::message::{Text, ToolFunction};

    fn usage(total: u64) -> Usage {
        let mut usage = Usage::new();
        usage.total_tokens = total;
        usage
    }

    fn tool_loop_script() -> MockScript {
        MockScript::from_responses(Vec::new()).with_streams(vec![
            vec![
                StreamedAssistantContent::ToolCall {
                    tool_call: ToolCall::new(
                        "call_1".to_string(),
                        ToolFunction::new("add".to_string(), serde_json::json!({"a": 1, "b": 2})),
                    ),
                    internal_call_id: "internal_1".to_string(),
                },
                StreamedAssistantContent::Final(StreamFinal::new("mock", usage(3))),
            ],
            vec![
                StreamedAssistantContent::Text(Text::new("done")),
                StreamedAssistantContent::Final(StreamFinal::new("mock", usage(5))),
            ],
        ])
    }

    fn tool_stream(policy: SessionPolicy) -> AgentStream {
        let mut config = AgentConfig::new();
        config.max_turns = Some(3);
        AgentStream::new(
            config,
            ProviderConfig::Mock(tool_loop_script()),
            Arc::new(Runtime::new()),
            "hello",
        )
        .with_tools(ToolCatalog::new(vec![ToolDefinition {
            name: "add".to_string(),
            description: "Adds two numbers".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }]))
        .with_policy(policy)
    }

    async fn next(stream: &mut AgentStream) -> AgentStreamItem {
        stream
            .next_item()
            .await
            .expect("stream not finished")
            .expect("no error")
    }

    /// Pull items until one is not an Assistant/CompletionCall observation.
    async fn next_decision(stream: &mut AgentStream) -> AgentStreamItem {
        loop {
            match next(stream).await {
                AgentStreamItem::Assistant(_) | AgentStreamItem::CompletionCall(_) => {}
                other => return other,
            }
        }
    }

    fn tool_result_for(call: &ToolCall, content: &str) -> UserContent {
        UserContent::tool_result(
            call.id.clone(),
            OneOrMany::one(rig_core::message::ToolResultContent::text(content)),
        )
    }

    #[tokio::test]
    async fn stream_gate_rewrite_then_result_keep_commits_effective_call() {
        let policy = SessionPolicy {
            surface_tool_calls: true,
            surface_tool_results: true,
            ..SessionPolicy::default()
        };
        let mut stream = tool_stream(policy);

        let call = match next_decision(&mut stream).await {
            AgentStreamItem::ToolCallPending { call } => call,
            other => panic!("expected ToolCallPending, got {other:?}"),
        };
        assert_eq!(call.tool_call.function.name, "add");
        stream
            .reply_tool_call(ToolCallAction::rewrite(serde_json::json!({"a": 7, "b": 8})))
            .expect("reply");

        // Announce-before-execute: the model's original call surfaces first.
        match next(&mut stream).await {
            AgentStreamItem::Assistant(StreamedAssistantContent::ToolCall {
                tool_call, ..
            }) => {
                assert_eq!(
                    tool_call.function.arguments,
                    serde_json::json!({"a": 1, "b": 2})
                );
            }
            other => panic!("expected announced ToolCall, got {other:?}"),
        }
        let calls = match next(&mut stream).await {
            AgentStreamItem::ToolCallsReady(calls) => calls,
            other => panic!("expected ToolCallsReady, got {other:?}"),
        };
        let ready = calls.first().expect("one call");
        assert_eq!(
            ready.tool_call.function.arguments,
            serde_json::json!({"a": 7, "b": 8})
        );
        stream
            .provide_tool_results(vec![tool_result_for(&ready.tool_call, "15")])
            .expect("results");

        match next(&mut stream).await {
            AgentStreamItem::ToolResultReady { call, .. } => assert_eq!(call.id, "call_1"),
            other => panic!("expected ToolResultReady, got {other:?}"),
        }
        stream
            .reply_tool_result(ToolResultAction::keep())
            .expect("reply");

        // Committed items carry the effective (rewritten) call.
        match next(&mut stream).await {
            AgentStreamItem::ToolExecutionCommitted { tool_call, .. } => {
                assert_eq!(
                    tool_call.function.arguments,
                    serde_json::json!({"a": 7, "b": 8})
                );
            }
            other => panic!("expected ToolExecutionCommitted, got {other:?}"),
        }
        match next(&mut stream).await {
            AgentStreamItem::User(_) => {}
            other => panic!("expected committed User result, got {other:?}"),
        }
        loop {
            match next(&mut stream).await {
                AgentStreamItem::Final(done) => {
                    assert_eq!(done.output, "done");
                    break;
                }
                AgentStreamItem::Assistant(_) | AgentStreamItem::CompletionCall(_) => {}
                other => panic!("expected Final, got {other:?}"),
            }
        }
        assert!(stream.next_item().await.is_none());
    }

    #[tokio::test]
    async fn stream_gate_skip_surfaces_synthetic_result() {
        let policy = SessionPolicy {
            surface_tool_calls: true,
            surface_tool_results: true,
            ..SessionPolicy::default()
        };
        let mut stream = tool_stream(policy);

        match next_decision(&mut stream).await {
            AgentStreamItem::ToolCallPending { .. } => {}
            other => panic!("expected ToolCallPending, got {other:?}"),
        }
        stream
            .reply_tool_call(ToolCallAction::skip("blocked by policy"))
            .expect("reply");

        let calls = match next_decision(&mut stream).await {
            AgentStreamItem::ToolCallsReady(calls) => calls,
            other => panic!("expected ToolCallsReady, got {other:?}"),
        };
        let ready = calls.first().expect("one call");
        let preresolved = ready
            .preresolved_result
            .clone()
            .expect("skip preresolves the result");
        stream
            .provide_tool_results(vec![preresolved])
            .expect("results");

        // A gate skip still surfaces its (synthetic) result, exactly as the
        // classic runner fires its tool-result hook for a hook skip.
        let result = match next(&mut stream).await {
            AgentStreamItem::ToolResultReady { result, .. } => result,
            other => panic!("expected ToolResultReady, got {other:?}"),
        };
        let rendered = serde_json::to_string(&result).expect("serialize");
        assert!(rendered.contains("blocked by policy"), "got {rendered}");
        stream
            .reply_tool_result(ToolResultAction::keep())
            .expect("reply");

        loop {
            if let AgentStreamItem::Final(done) = next(&mut stream).await {
                assert_eq!(done.output, "done");
                break;
            }
        }
    }
}
