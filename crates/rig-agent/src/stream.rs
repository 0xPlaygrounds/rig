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
//!
//! [`AgentStream::drive`] consumes those decision items through hooks and a
//! tool executor, returning an [`AgentRunStream`] of observation-only
//! [`AgentRunItem`] values.

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::{Stream, StreamExt};

use crate::agent::attempt::ModelCallAttempt;
use crate::agent::hook::{
    CompletionCallAction, InvalidToolCallAction, ModelTurnAction, RequestPatch, ResolvedToolCall,
    ResolvedToolCallDisposition, ToolCallAction, ToolResultAction,
};
use crate::agent::prepare::ToolCatalog;
use crate::agent::response::tool_result_output;
use crate::agent::run::{
    AgentRun, AgentRunStep, DEFAULT_OUTPUT_RETRIES, PartialStreamedTurn, PendingToolCall,
    StreamedInvalidToolCall, StreamedResolution, StreamedTurnAssembler, StreamedTurnEvent,
    ToolResultSubmission,
};
use crate::agent::telemetry::{
    SessionSpanParams, acquire_agent_span, new_session_chat_streaming_span, record_usage_on_span,
};
use crate::agent::{AgentConfig, InvalidToolCallContext, PromptResponse, UNKNOWN_AGENT_NAME};
use crate::completion::{Message, PromptError, Usage};
use rig_core::OneOrMany;
use rig_core::message::{AssistantContent, ToolCall, UserContent};
use rig_core::streaming::{
    CompletionStream, StreamFinal, StreamedAssistantContent, StreamedUserContent,
};
use tracing_futures::Instrument;

use crate::provider::{self, ProviderConfig, Runtime};
use crate::session::SessionPolicy;
use rig_core::wasm_compat::WasmCompatSend;

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
    /// [`StreamFinal`].
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
        /// Provider-assigned message ID for this turn, when present.
        message_id: Option<String>,
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
        /// Stable Rig identity shared with the originating ToolCall event.
        internal_call_id: String,
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

/// One observation emitted by a fully driven [`AgentRunStream`].
///
/// Unlike [`AgentStreamItem`], this enum contains no host decision requests:
/// hooks have already been dispatched and tool calls have already been
/// executed by [`AgentStream::drive`]. The deliberately exhaustive split
/// makes a newly added host decision fail to compile in the driver instead of
/// leaking into a driven stream.
///
/// Every payload is owned and serializable so observations can be persisted,
/// forwarded over a wire, or replayed without borrowing session state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AgentRunItem {
    /// A streamed assistant item: text/reasoning/tool-call deltas, complete
    /// tool calls, and the provider's terminal [`StreamFinal`].
    Assistant(StreamedAssistantContent),
    /// Exactly one per provider call: the recorded completion-call entry.
    CompletionCall(crate::agent::CompletionCall),
    /// A turn was rolled back; its earlier deltas were provisional.
    ModelTurnRetried {
        /// One-based index of the retried model call.
        turn: usize,
    },
    /// A tool call's result was committed to history.
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

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type ErasedAgentRunStream =
    Pin<Box<dyn Stream<Item = Result<AgentRunItem, PromptError>> + Send + 'static>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type ErasedAgentRunStream =
    Pin<Box<dyn Stream<Item = Result<AgentRunItem, PromptError>> + 'static>>;

/// A fully driven agent run exposed as a stream of owned observation events.
///
/// Unlike the host-driven [`AgentStream`], this stream dispatches hooks and
/// executes tools for the caller and yields [`AgentRunItem`] rather than the
/// host protocol's [`AgentStreamItem`]. It is already pinned internally, so
/// callers can use [`Self::next`] without importing [`StreamExt`], pinning it,
/// or naming an `Unpin` bound. It remains a [`Stream`] for combinator
/// interoperability.
///
/// Call [`AgentRunStream::into_final_response`] when intermediate events are
/// not needed and only the committed terminal response matters.
#[must_use = "streams do nothing unless polled"]
pub struct AgentRunStream {
    inner: ErasedAgentRunStream,
}

impl AgentRunStream {
    fn new<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<AgentRunItem, PromptError>> + WasmCompatSend + 'static,
    {
        Self {
            inner: Box::pin(stream),
        }
    }

    /// Pull the next observation without requiring [`StreamExt`] or caller
    /// pinning.
    ///
    /// The suspended generator remains stored in this handle when the
    /// temporary future returned here is dropped, so an in-flight hook or tool
    /// operation resumes on the next poll instead of being reconstructed.
    pub async fn next(&mut self) -> Option<Result<AgentRunItem, PromptError>> {
        StreamExt::next(&mut self.inner).await
    }

    /// Consume the stream and return its committed terminal response.
    ///
    /// Intermediate assistant, tool, retry, and completion-call observations
    /// are discarded. Hooks are still dispatched and tools are still executed
    /// by the fully driven run. A yielded [`PromptError`] is returned
    /// immediately; a stream that ends without [`AgentRunItem::Final`]
    /// returns [`PromptError::StreamEndedWithoutFinalResponse`].
    pub async fn into_final_response(mut self) -> Result<PromptResponse, PromptError> {
        while let Some(item) = self.next().await {
            if let AgentRunItem::Final(response) = item? {
                return Ok(response);
            }
        }

        Err(PromptError::StreamEndedWithoutFinalResponse)
    }
}

impl Stream for AgentRunStream {
    type Item = Result<AgentRunItem, PromptError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next(cx)
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[allow(dead_code)]
fn _assert_agent_run_stream_is_send() {
    fn assert_send<T: Send>() {}

    assert_send::<AgentRunStream>();
}

// Browser-wasm execution may retain worker-local JavaScript state inside the
// suspended generator. This lives in ordinary module code because wasm CI runs
// `cargo check`, which does not compile `#[cfg(test)]` modules.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[allow(dead_code)]
fn _assert_agent_run_stream_accepts_worker_local_generator() {
    let worker_local = std::rc::Rc::new(());
    let stream = futures::stream::once(async move {
        std::future::pending::<()>().await;
        drop(worker_local);
        Ok(AgentRunItem::Final(PromptResponse::empty()))
    });

    let _stream = AgentRunStream::new(stream);
}

/// The in-flight provider stream plus its sans-IO assembler.
struct ActiveTurn {
    /// Request inputs and provisional run mutations owned until commit.
    attempt: ModelCallAttempt,
    stream: CompletionStream,
    assembler: StreamedTurnAssembler,
    /// The per-call `chat_streaming` span the provider stream is polled
    /// under, mirroring the classic streaming driver.
    chat_span: tracing::Span,
    /// Whether the provider already emitted its terminal record.
    provider_final_seen: bool,
    /// Last committed terminal record before this provisional attempt. If the
    /// attempt rolls back after yielding a final record, restore this value so
    /// observation state does not point at a failed turn.
    previous_last_final: Option<StreamFinal>,
    /// A retry/skip decision is kept provisional until the discarded provider
    /// stream reaches checked natural exhaustion.
    recovery: Option<PendingStreamRecovery>,
}

struct PendingStreamRecovery {
    run: AgentRun,
    resolution: StreamedResolution,
    turn: usize,
    internal_call_id: String,
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
        /// Calls not yet decided, in call order.
        remaining: VecDeque<PendingToolCall>,
        /// Calls already decided (rewrites/skips applied).
        decided: Vec<PendingToolCall>,
        /// The call surfaced and awaiting [`AgentStream::reply_tool_call`].
        current: Option<PendingToolCall>,
        /// Rig identities skipped via [`ToolCallAction::Skip`].
        skipped_internal_ids: Vec<String>,
    },
    Tools {
        calls: Vec<PendingToolCall>,
        /// Rig identities skipped via [`ToolCallAction::Skip`] — these still
        /// surface their synthetic result under `surface_tool_results`.
        skipped_internal_ids: Vec<String>,
    },
    /// `policy.surface_tool_results`: post-execution decisions in flight.
    ToolResultGate {
        /// One entry per pending call, in call order.
        entries: Vec<ResultGateEntry>,
        /// Results without a matching call, passed through for the run to
        /// validate.
        extras: Vec<ToolResultSubmission>,
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
    /// Exact logical attempt retained across preparation/open/stream failure
    /// or cancellation until the next poll reissues it.
    retry_attempt: Option<ModelCallAttempt>,
    pending: Pending,
    active: Option<ActiveTurn>,
    buffered: VecDeque<AgentStreamItem>,
    last_final: Option<StreamFinal>,
    /// The finished turn's `chat_streaming` span, parked until the turn's
    /// verdict is known so a retried (provisional) turn records no output
    /// telemetry — exactly as the classic streaming driver suppressed it.
    turn_chat_span: Option<tracing::Span>,
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
            retry_attempt: None,
            pending: Pending::None,
            active: None,
            buffered: VecDeque::new(),
            last_final: None,
            turn_chat_span: None,
            agent_span,
            created_agent_span,
        }
    }

    /// Set the input chat history preceding the prompt.
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.run.set_history(history);
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

    /// Terminate and abandon the current provider attempt immediately.
    ///
    /// Continuing to poll starts a fresh attempt with the retained request
    /// patch. Already-observed deltas were provisional and may be repeated.
    pub fn close_turn(&mut self) {
        self.abandon_active_turn();
    }

    /// Merge a per-turn request patch consumed by the next model call.
    pub fn patch_next_turn(&mut self, patch: RequestPatch) {
        if let Some(attempt) = &mut self.retry_attempt {
            attempt.merge_patch(patch);
            return;
        }
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
        let mut retry_attempt = self.retry_attempt.take();
        if let Some(attempt) = &mut retry_attempt {
            attempt.make_retryable(&mut self.run);
        }
        match self.run.next_step()? {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                if let Some(mut attempt) = retry_attempt {
                    attempt.reissue(turn);
                    return self.open_attempt(attempt).await;
                }
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
            AgentRunStep::CallTools { mut calls } => {
                for call in &mut calls {
                    call.ensure_internal_call_id();
                }
                if self.policy.surface_tool_calls {
                    // Per-call decisions first; announce-before-execute
                    // items surface once the whole batch is decided.
                    self.pending = Pending::ToolCallGate {
                        remaining: calls.into(),
                        decided: Vec::new(),
                        current: None,
                        skipped_internal_ids: Vec::new(),
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
                    skipped_internal_ids: Vec::new(),
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
        let attempt = ModelCallAttempt::begin(prompt, history, turn, &mut self.next_patch);
        self.open_attempt(attempt).await
    }

    async fn open_attempt(&mut self, mut attempt: ModelCallAttempt) -> Result<(), PromptError> {
        let mut prepared = match attempt.prepare(
            &self.config,
            &self.tools,
            self.provider.descriptor().composes_native_output_with_tools,
            self.run.output_tool_name(),
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                attempt.make_retryable(&mut self.run);
                self.retry_attempt = Some(attempt);
                return Err(error.into());
            }
        };
        // The per-call `chat_streaming` span — identical shape to the classic
        // streaming driver's, parented into the ambient span tree.
        let chat_span = new_session_chat_streaming_span(
            &SessionSpanParams {
                agent_name: self.config.name.as_deref(),
            },
            &prepared.request,
        );
        // Content telemetry is recorded onto the agent's own `chat_streaming`
        // span and suppressed on the provider's, exactly as the classic
        // streaming driver did.
        if self.config.record_telemetry_content {
            let input_messages = prepared.request.messages_for_telemetry();
            rig_core::telemetry::record_model_input(&chat_span, &input_messages, true);
            prepared.request.record_telemetry_content = false;
        }
        attempt.mark_in_flight();
        self.retry_attempt = Some(attempt);
        let stream_result = provider::open_stream(&self.provider, &self.rt, prepared.request)
            .instrument(chat_span.clone())
            .await;
        let Some(mut attempt) = self.retry_attempt.take() else {
            self.run.abandon_pending_model_call();
            return Err(self.run.cancel_error(
                "model-call attempt ownership disappeared while provider stream was opening",
            ));
        };
        let stream = match stream_result {
            Ok(stream) => stream,
            Err(error) => {
                attempt.make_retryable(&mut self.run);
                self.retry_attempt = Some(attempt);
                return Err(PromptError::from(error));
            }
        };
        let previous_last_final = self.last_final.take();
        self.active = Some(ActiveTurn {
            attempt,
            stream,
            assembler: StreamedTurnAssembler::new(
                prepared.executable_tool_names,
                prepared.allowed_tool_names,
            ),
            chat_span,
            provider_final_seen: false,
            previous_last_final,
            recovery: None,
        });
        Ok(())
    }

    /// Poll the open provider stream once, translating assembler events
    /// into buffered items or pending decisions.
    async fn poll_active_turn(&mut self) -> Result<(), PromptError> {
        let result = self.poll_active_turn_once().await;
        if result.is_err() {
            self.abandon_active_turn();
        }
        result
    }

    /// Poll one item from the active attempt. The outer wrapper owns rollback
    /// so every protocol, assembler, and provider error follows one path.
    async fn poll_active_turn_once(&mut self) -> Result<(), PromptError> {
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.recovery.is_some())
        {
            return self.poll_active_recovery_once().await;
        }
        let Some(active) = &mut self.active else {
            return Ok(());
        };
        let chat_span = active.chat_span.clone();
        match active.stream.next().instrument(chat_span.clone()).await {
            Some(Ok(item)) => {
                if active.provider_final_seen {
                    // Any provider item after the terminal record violates the
                    // stream protocol, including metadata-only items.
                    return Err(PromptError::CompletionError(
                        rig_core::completion::CompletionError::ResponseError(
                            "provider stream emitted an item after its final response".to_string(),
                        ),
                    ));
                }
                let events = active.assembler.ingest(&item).map_err(PromptError::from)?;
                let mut saw_provider_final = false;
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
                            let partial = active.assembler.partial_turn(active.stream.message_id());
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
                        StreamedTurnEvent::Completed {
                            usage: _,
                            emit_final,
                        } => {
                            saw_provider_final = true;
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
                if saw_provider_final && let Some(active) = &mut self.active {
                    active.provider_final_seen = true;
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
                    attempt,
                    stream,
                    assembler,
                    chat_span,
                    provider_final_seen: _,
                    previous_last_final: _,
                    recovery: _,
                } = active;
                let response = match stream.into_response() {
                    Ok(response) => response,
                    Err(error) => {
                        attempt.abandon(&mut self.run, &mut self.next_patch);
                        return Err(PromptError::CompletionError(
                            rig_core::completion::CompletionError::ResponseError(error.to_string()),
                        ));
                    }
                };
                let turn = attempt.turn();
                let usage = response.usage;
                let message_id = response.message_id.clone();
                let streamed = assembler.finish(response.message_id, response.choice, usage);
                // Exactly one CompletionCall item per model call: when the
                // provider never yielded a terminal record, `streamed_turn`
                // records the no-usage fallback and the item is surfaced here.
                let recorded_calls = self.run.completion_calls().len();
                attempt.commit_streamed(&mut self.run, &mut self.next_patch, streamed)?;
                record_usage_on_span(&chat_span, usage);
                if let Some(call) = self.run.completion_calls().get(recorded_calls).copied() {
                    self.buffered
                        .push_back(AgentStreamItem::CompletionCall(call));
                }
                if self.policy.surface_model_turns && self.run.accepted_turn_choice().is_some() {
                    // The verdict is the host's: park the span so a retried
                    // turn records no output telemetry.
                    self.turn_chat_span = Some(chat_span.clone());
                } else {
                    self.record_turn_output(&chat_span);
                }
                if self.policy.surface_model_turns
                    && let Some(content) = self.run.accepted_turn_choice()
                {
                    self.pending = Pending::TurnReply;
                    // Per-turn usage, not the run aggregate: the last recorded
                    // completion call is this turn's (recorded from the
                    // stream's terminal event, or as a zero-usage fallback by
                    // `streamed_turn` when the provider reported none).
                    self.buffered.push_back(AgentStreamItem::TurnFinished {
                        turn,
                        content,
                        usage,
                        message_id,
                    });
                }
                Ok(())
            }
        }
    }

    /// Drain one item from a deliberately abandoned invalid-call turn. The
    /// candidate recovery run remains provisional until checked natural EOF;
    /// cancellation keeps this state on the driver, and any provider failure
    /// rolls the original attempt back through the ordinary retry path.
    async fn poll_active_recovery_once(&mut self) -> Result<(), PromptError> {
        let Some(active) = &mut self.active else {
            return Ok(());
        };
        let chat_span = active.chat_span.clone();
        match active.stream.next().instrument(chat_span.clone()).await {
            Some(Ok(StreamedAssistantContent::Final(final_record))) => {
                self.last_final = Some(final_record);
                Ok(())
            }
            Some(Ok(_)) => Ok(()),
            Some(Err(error)) => Err(PromptError::from(error)),
            None => {
                let Some(active) = self.active.take() else {
                    return Ok(());
                };
                let ActiveTurn {
                    mut attempt,
                    stream,
                    assembler: _,
                    chat_span,
                    provider_final_seen: _,
                    previous_last_final: _,
                    recovery,
                } = active;
                let response = match stream.into_response() {
                    Ok(response) => response,
                    Err(error) => {
                        attempt.make_retryable(&mut self.run);
                        self.retry_attempt = Some(attempt);
                        return Err(PromptError::CompletionError(
                            rig_core::completion::CompletionError::ResponseError(error.to_string()),
                        ));
                    }
                };
                let Some(PendingStreamRecovery {
                    mut run,
                    resolution,
                    turn,
                    internal_call_id,
                }) = recovery
                else {
                    attempt.make_retryable(&mut self.run);
                    self.retry_attempt = Some(attempt);
                    return Err(self.run.cancel_error(
                        "invalid-call recovery drain lost its provisional transition",
                    ));
                };
                let call = match run.record_streamed_completion_call(response.usage) {
                    Ok(call) => call,
                    Err(error) => {
                        attempt.make_retryable(&mut self.run);
                        self.retry_attempt = Some(attempt);
                        return Err(error);
                    }
                };
                attempt.commit_recovered(&mut run);
                self.run = run;
                record_usage_on_span(&chat_span, response.usage);

                let StreamedResolution::TurnAbandoned {
                    skipped_tool_result,
                } = resolution
                else {
                    return Err(self.run.cancel_error(
                        "invalid-call recovery drain completed without an abandoned turn",
                    ));
                };
                if let Some(result) = skipped_tool_result {
                    self.buffered.push_back(AgentStreamItem::User(
                        StreamedUserContent::tool_result(result, internal_call_id),
                    ));
                }
                self.buffered
                    .push_back(AgentStreamItem::CompletionCall(call));
                self.buffered
                    .push_back(AgentStreamItem::ModelTurnRetried { turn });
                Ok(())
            }
        }
    }

    /// Drop transport and assembler state, restore the attempt patch, and
    /// return the run to the same pre-call state used by both drivers.
    fn abandon_active_turn(&mut self) {
        if let Some(mut active) = self.active.take() {
            active.stream.cancel();
            self.last_final = active.previous_last_final.take();
            active.attempt.make_retryable(&mut self.run);
            if let Some(later) = self.next_patch.take() {
                active.attempt.merge_patch(later);
            }
            self.retry_attempt = Some(active.attempt);
            // Decision inboxes and buffered observations created by this
            // attempt are provisional just like its assembler state.
            self.pending = Pending::None;
            self.buffered.clear();
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
                    if let Err(error) = self.provide_tool_results(batch.into_submissions()) {
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

    /// Record the accepted turn's content telemetry onto its
    /// `chat_streaming` span, gated on `record_telemetry_content` like the
    /// classic streaming driver.
    fn record_turn_output(&self, chat_span: &tracing::Span) {
        if self.config.record_telemetry_content
            && let Some(choice) = self.run.accepted_turn_choice()
        {
            rig_core::telemetry::record_model_output(chat_span, &choice, true);
        }
    }

    /// Answer [`AgentStreamItem::TurnFinished`].
    pub fn reply_turn(&mut self, action: ModelTurnAction) -> Result<(), PromptError> {
        if !matches!(self.pending, Pending::TurnReply) {
            return Err(self
                .run
                .cancel_error("reply_turn without a pending TurnFinished item"));
        }
        self.pending = Pending::None;
        let chat_span = self.turn_chat_span.take();
        match action {
            ModelTurnAction::Continue => {
                if let Some(span) = &chat_span {
                    self.record_turn_output(span);
                }
                Ok(())
            }
            ModelTurnAction::Retry(request) => {
                let turn = self.run.turn();
                self.run.retry_model_turn(request)?;
                self.buffered
                    .push_back(AgentStreamItem::ModelTurnRetried { turn });
                Ok(())
            }
            ModelTurnAction::Stop(reason) => {
                if let Some(span) = &chat_span {
                    self.record_turn_output(span);
                }
                Err(self.run.cancel_error(reason))
            }
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
        // Retry/skip mutates history and retry accounting. Apply it to a
        // candidate run first; the live run remains AwaitingModel until the
        // discarded provider stream proves natural exhaustion.
        let mut candidate_run = self.run.clone();
        let resolution =
            match candidate_run.resolve_streamed_invalid_tool_call(&partial, &invalid, action) {
                Ok(resolution) => resolution,
                Err(error) => {
                    self.run = candidate_run;
                    return Err(error);
                }
            };
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
                skipped_tool_result: _,
            } => {
                let turn = self.run.turn();
                let Some(active) = &mut self.active else {
                    return Err(self
                        .run
                        .cancel_error("invalid-call recovery lost its active provider attempt"));
                };
                let _ = active.assembler.resolve_pending_invalid(&resolution);
                active.recovery = Some(PendingStreamRecovery {
                    run: candidate_run,
                    resolution,
                    turn,
                    internal_call_id: invalid.internal_call_id,
                });
                while self
                    .active
                    .as_ref()
                    .is_some_and(|active| active.recovery.is_some())
                {
                    self.poll_active_turn().await?;
                }
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
                mut remaining,
                mut decided,
                current: None,
                skipped_internal_ids,
            } => {
                while let Some(next) = remaining.pop_front() {
                    if next.preresolved_result.is_some() {
                        decided.push(next);
                        continue;
                    }
                    let call = next.clone();
                    self.pending = Pending::ToolCallGate {
                        remaining,
                        decided,
                        current: Some(next),
                        skipped_internal_ids,
                    };
                    self.buffered
                        .push_back(AgentStreamItem::ToolCallPending { call });
                    return Ok(());
                }
                // Announce-before-execute: the model's calls, in call order,
                // right before the decision point.
                for call in &decided {
                    let mut announced_call = call
                        .original_tool_call
                        .as_deref()
                        .cloned()
                        .unwrap_or_else(|| call.tool_call.clone());
                    // Argument rewrites are execution policy, so the
                    // assistant item keeps provider arguments. Invalid-name
                    // repair is different: consumers must see the executable
                    // repaired name. Project only that name so a later
                    // argument rewrite cannot leak into provider output.
                    announced_call.function.name = call.tool_call.function.name.clone();
                    self.buffered.push_back(AgentStreamItem::Assistant(
                        StreamedAssistantContent::ToolCall {
                            tool_call: announced_call,
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
                    skipped_internal_ids,
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
        let original_call = match &self.pending {
            Pending::ToolCallGate {
                current: Some(call),
                ..
            } => call.tool_call.clone(),
            _ => {
                return Err(self
                    .run
                    .cancel_error("reply_tool_call without a pending ToolCallPending item"));
            }
        };
        self.reply_resolved_tool_call(ResolvedToolCall::from_action(original_call, action))
    }

    /// Apply one fully composed hook resolution without separating its
    /// effective arguments from a terminal skip.
    fn reply_resolved_tool_call(
        &mut self,
        resolution: ResolvedToolCall,
    ) -> Result<(), PromptError> {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::ToolCallGate {
                remaining,
                mut decided,
                current: Some(mut call),
                mut skipped_internal_ids,
            } => {
                let (effective_call, disposition) = resolution.into_parts();
                call.original_tool_call
                    .get_or_insert_with(|| Box::new(call.tool_call.clone()));
                call.tool_call = effective_call;
                match disposition {
                    ResolvedToolCallDisposition::Run => decided.push(call),
                    ResolvedToolCallDisposition::Skip(reason) => {
                        // Mirror run_single_tool: the skip becomes a
                        // `ToolResult::skipped` presentation delivered to
                        // the model without executing the body.
                        let skipped = crate::tool::ToolResult::skipped(reason.clone());
                        let content = tool_result_output(
                            call.tool_call.id.clone(),
                            call.tool_call.call_id.clone(),
                            skipped.output().clone(),
                        );
                        skipped_internal_ids.push(call.ensure_internal_call_id().to_owned());
                        call.preresolved_result = Some(content);
                        call.preresolved_disposition =
                            Some(crate::agent::run::PreresolvedToolDisposition::Skipped { reason });
                        decided.push(call);
                    }
                    ResolvedToolCallDisposition::Stop(reason) => {
                        return Err(self.run.cancel_error(reason));
                    }
                }
                self.pending = Pending::ToolCallGate {
                    remaining,
                    decided,
                    current: None,
                    skipped_internal_ids,
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
                if let Some((index, call, internal_call_id, result)) = entries
                    .iter()
                    .enumerate()
                    .skip(cursor)
                    .find(|(_, entry)| entry.surface)
                    .and_then(|(index, entry)| {
                        entry.result.as_ref().map(|result| {
                            (
                                index,
                                entry.call.tool_call.clone(),
                                entry.internal_call_id.clone(),
                                result.clone(),
                            )
                        })
                    })
                {
                    self.pending = Pending::ToolResultGate {
                        entries,
                        extras,
                        cursor: index,
                        awaiting: true,
                    };
                    self.buffered.push_back(AgentStreamItem::ToolResultReady {
                        call,
                        internal_call_id,
                        result,
                    });
                    return Ok(());
                }
                // Every decision answered: commit, then surface committed
                // items in call order, mirroring the ungated path.
                let submissions: Vec<ToolResultSubmission> = entries
                    .iter()
                    .filter_map(|entry| {
                        entry.result.clone().map(|result| {
                            ToolResultSubmission::new(entry.internal_call_id.clone(), result)
                        })
                    })
                    .chain(extras)
                    .collect();
                self.run.tool_result_submissions(submissions)?;
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

    /// Answer [`AgentStreamItem::ToolCallsReady`] with identity-bearing
    /// [`ToolResultSubmission`] records. Use each
    /// [`PendingToolCall::internal_call_id`] as the join key; provider IDs may
    /// duplicate and submissions may arrive in any order. Committed results
    /// and execution markers surface on subsequent [`AgentStream::next_item`]
    /// calls, in call order. Under `policy.surface_tool_results` the batch is
    /// not committed yet: each result surfaces as
    /// [`AgentStreamItem::ToolResultReady`] first, and commits once every
    /// decision is answered.
    pub fn provide_tool_results(
        &mut self,
        submissions: Vec<ToolResultSubmission>,
    ) -> Result<(), PromptError> {
        let Pending::Tools {
            calls,
            skipped_internal_ids,
        } = std::mem::replace(&mut self.pending, Pending::None)
        else {
            return Err(self
                .run
                .cancel_error("provide_tool_results without a pending ToolCallsReady item"));
        };
        if self.policy.surface_tool_results {
            // Pair by stable Rig identity. Provider IDs are payload and may
            // duplicate. Results preresolved by invalid-call recovery
            // pass through verbatim; gate skips do surface, exactly as
            // run_single_tool fires its tool-result hook for a hook skip.
            let mut remaining: Vec<Option<ToolResultSubmission>> =
                submissions.into_iter().map(Some).collect();
            let mut entries = Vec::new();
            for call in calls {
                let matched = remaining.iter_mut().find_map(|slot| {
                    let is_match = slot.as_ref().is_some_and(|submission| {
                        call.internal_call_id.as_deref()
                            == Some(submission.internal_call_id.as_str())
                    });
                    if is_match { slot.take() } else { None }
                });
                let surface = matched.is_some()
                    && (call.preresolved_result.is_none()
                        || call
                            .internal_call_id
                            .as_ref()
                            .is_some_and(|id| skipped_internal_ids.contains(id)));
                let internal_call_id = call
                    .internal_call_id
                    .clone()
                    .unwrap_or_else(rig_core::id::generate);
                entries.push(ResultGateEntry {
                    call,
                    internal_call_id,
                    result: matched.map(|submission| submission.result),
                    surface,
                });
            }
            let extras: Vec<ToolResultSubmission> = remaining.into_iter().flatten().collect();
            self.pending = Pending::ToolResultGate {
                entries,
                extras,
                cursor: 0,
                awaiting: false,
            };
            return Ok(());
        }
        self.run.tool_result_submissions(submissions.clone())?;
        // Post-commit surface items in call order, joined only by the unique
        // Rig invocation identity.
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
            if let Some(UserContent::ToolResult(result)) = submissions
                .iter()
                .find(|submission| submission.internal_call_id == internal)
                .map(|submission| submission.result.clone())
            {
                self.buffered
                    .push_back(AgentStreamItem::User(StreamedUserContent::tool_result(
                        result, internal,
                    )));
            }
        }
        Ok(())
    }

    /// Accept executor-owned records without rejoining results to calls by a
    /// provider ID that is allowed to be duplicated.
    fn provide_tool_execution_records(
        &mut self,
        records: &[crate::executor::ToolExecutionRecord],
    ) -> Result<(), PromptError> {
        let Pending::Tools {
            calls,
            skipped_internal_ids,
        } = std::mem::replace(&mut self.pending, Pending::None)
        else {
            return Err(self.run.cancel_error(
                "provide_tool_execution_records without a pending ToolCallsReady item",
            ));
        };
        if calls.len() != records.len()
            || calls.iter().zip(records).any(|(call, record)| {
                call.internal_call_id.as_deref() != Some(record.internal_call_id.as_str())
            })
        {
            return Err(self
                .run
                .cancel_error("tool execution records do not match the pending indexed batch"));
        }

        if self.policy.surface_tool_results {
            let entries = calls
                .into_iter()
                .zip(records)
                .map(|(call, record)| {
                    let surface = call.preresolved_result.is_none()
                        || skipped_internal_ids.contains(&record.internal_call_id);
                    ResultGateEntry {
                        call,
                        internal_call_id: record.internal_call_id.clone(),
                        result: Some(record.result.clone()),
                        surface,
                    }
                })
                .collect();
            self.pending = Pending::ToolResultGate {
                entries,
                extras: Vec::new(),
                cursor: 0,
                awaiting: false,
            };
            return Ok(());
        }

        self.run.tool_result_submissions(
            records
                .iter()
                .map(|record| {
                    ToolResultSubmission::new(
                        record.internal_call_id.clone(),
                        record.result.clone(),
                    )
                })
                .collect(),
        )?;
        for (call, record) in calls.iter().zip(records) {
            self.buffered
                .push_back(AgentStreamItem::ToolExecutionCommitted {
                    tool_call: record.effective_call.clone(),
                    internal_call_id: record.internal_call_id.clone(),
                });
            if let UserContent::ToolResult(result) = &record.result {
                self.buffered
                    .push_back(AgentStreamItem::User(StreamedUserContent::tool_result(
                        result.clone(),
                        record.internal_call_id.clone(),
                    )));
            }
            debug_assert_eq!(
                call.internal_call_id.as_deref(),
                Some(record.internal_call_id.as_str())
            );
        }
        Ok(())
    }
}

impl AgentStream {
    /// Drive this stream with the classic streaming runtime semantics:
    /// dispatch `hooks` at every surfaced decision point, answer tool
    /// batches through `executor`, and forward the observable items.
    ///
    /// The returned stream yields exactly what the deleted classic streaming
    /// surface yielded — assistant deltas and complete tool calls, per-call
    /// [`AgentRunItem::CompletionCall`] records, tool execution/result items,
    /// [`AgentRunItem::ModelTurnRetried`], and the terminal
    /// [`AgentRunItem::Final`]. Policy-gated decision items are consumed by
    /// the hook dispatch rather than surfaced, so a consumer sees a pure
    /// observation stream.
    ///
    /// Errors terminate the stream: the error item is the last one.
    pub fn drive(
        mut self,
        hooks: crate::hooks::Hooks,
        executor: Option<crate::executor::ToolExecutor>,
    ) -> AgentRunStream {
        // Only entries that opted in observe the hot-path deltas.
        let observes_deltas = hooks.observes_deltas();
        AgentRunStream::new(async_stream::stream! {
            // The turn prompt surfaced on `BeforeModelCall`, replayed into the
            // response-finish observation (classic parity).
            let mut turn_prompt: Option<Message> = None;
            // Aggregated text for the in-flight turn, matching the classic
            // assembler's `aggregated_text()`.
            let mut aggregated = String::new();
            // The in-flight batch's structured results and per-call spans, so
            // the post-execution decision point carries the classic
            // classification and records post-hook result telemetry.
            let mut batch_records: Vec<crate::executor::ToolExecutionRecord> = Vec::new();
            loop {
                let item = match self.next_item().await {
                    None => break,
                    Some(Ok(item)) => item,
                    Some(Err(error)) => {
                        yield Err(error);
                        break;
                    }
                };
                match item {
                    AgentStreamItem::BeforeModelCall { prompt, history, turn } => {
                        aggregated.clear();
                        let action =
                            hooks.dispatch_completion_call(turn, &prompt, &history).await;
                        turn_prompt = Some(prompt);
                        if let Err(error) = self.reply_before_call(action).await {
                            yield Err(error);
                            break;
                        }
                    }
                    AgentStreamItem::TurnFinished {
                        turn,
                        content,
                        usage,
                        message_id,
                    } => {
                        // The provider's terminal record observation fires
                        // first, then the normalized per-turn verdict.
                        let observed_prompt =
                            turn_prompt.clone().unwrap_or_else(|| Message::from(""));
                        if let crate::agent::ObservationAction::Stop(reason) = hooks
                            .dispatch_stream_response_finish(
                                turn,
                                &observed_prompt,
                                &content,
                                usage,
                                message_id.as_deref(),
                            )
                            .await
                        {
                            if let Err(error) = self.reply_turn(ModelTurnAction::Stop(reason)) {
                                yield Err(error);
                            }
                            break;
                        }
                        let action = hooks.dispatch_model_turn(turn, &content, usage).await;
                        if let Err(error) = self.reply_turn(action) {
                            yield Err(error);
                            break;
                        }
                    }
                    AgentStreamItem::InvalidToolCall(context) => {
                        let action = hooks
                            .dispatch_invalid_tool_call(&context)
                            .await
                            // Preserve the classic default: fail fast.
                            .unwrap_or_else(InvalidToolCallAction::fail);
                        if let Err(error) = self.resolve_invalid(action).await {
                            yield Err(error);
                            break;
                        }
                    }
                    AgentStreamItem::ToolCallPending { call } => {
                        let internal_call_id = call
                            .internal_call_id
                            .clone()
                            .unwrap_or_else(rig_core::id::generate);
                        let resolution = hooks
                            .dispatch_tool_call(&call.tool_call, &internal_call_id)
                            .await;
                        if let Err(error) = self.reply_resolved_tool_call(resolution) {
                            yield Err(error);
                            break;
                        }
                    }
                    AgentStreamItem::ToolCallsReady(calls) => {
                        match &executor {
                            Some(executor) => {
                                let batch = executor.execute_batch(&calls).await;
                                batch_records = batch.records;
                                if let Err(error) =
                                    self.provide_tool_execution_records(&batch_records)
                                {
                                    yield Err(error);
                                    break;
                                }
                                if !self.policy.surface_tool_results {
                                    batch_records.clear();
                                }
                            }
                            None => {
                                let results = match crate::session::preresolved_only_results(&calls) {
                                    Ok(results) => results,
                                    Err(error) => {
                                        yield Err(error);
                                        break;
                                    }
                                };
                                if let Err(error) = self.provide_tool_results(results) {
                                    yield Err(error);
                                    break;
                                }
                                batch_records.clear();
                            }
                        }
                    }
                    AgentStreamItem::ToolResultReady {
                        call,
                        internal_call_id,
                        result,
                    } => {
                        // The executor's structured result when the body ran,
                        // so the classic classification (failed/skipped/
                        // denied, error kind, HTTP status) survives; otherwise
                        // reconstruct it from the committed content.
                        let execution = batch_records
                            .iter()
                            .find(|record| record.internal_call_id == internal_call_id);
                        let raw = execution
                            .map(|record| record.raw_result.clone())
                            .unwrap_or_else(|| crate::executor::raw_tool_result(&result));
                        let action = hooks
                            .dispatch_tool_result(&call, &internal_call_id, &raw)
                            .await;
                        // Result telemetry once, post-hook (the executor
                        // defers it for this driver), so a redaction rewrite is
                        // never preceded by the raw value.
                        if self.config.record_telemetry_content
                            && let Some(span) = execution.and_then(|record| record.span.as_ref())
                        {
                            let rendered = match &action {
                                ToolResultAction::Rewrite(output) => Some(output.render()),
                                ToolResultAction::Keep => Some(raw.output().render()),
                                ToolResultAction::Stop(_) => None,
                            };
                            if let Some(rendered) = rendered {
                                span.record("gen_ai.tool.call.result", rendered);
                            }
                        }
                        if let Err(error) = self.reply_tool_result(action) {
                            yield Err(error);
                            break;
                        }
                    }
                    AgentStreamItem::Assistant(content) => {
                        // Delta observation, gated exactly as the classic
                        // driver gated it: only when an entry opted in.
                        if observes_deltas {
                            let turn = self.run_state().turn();
                            let observation = match &content {
                                StreamedAssistantContent::Text(text) => {
                                    aggregated.push_str(&text.text);
                                    hooks
                                        .dispatch_text_delta(turn, &text.text, &aggregated)
                                        .await
                                }
                                StreamedAssistantContent::ToolCallDelta {
                                    id,
                                    internal_call_id,
                                    content,
                                } => {
                                    let (name, delta) = match content {
                                        crate::streaming::ToolCallDeltaContent::Name(name) => {
                                            (Some(name.as_str()), "")
                                        }
                                        crate::streaming::ToolCallDeltaContent::Delta(delta) => {
                                            (None, delta.as_str())
                                        }
                                    };
                                    hooks
                                        .dispatch_tool_call_delta(
                                            turn,
                                            id,
                                            internal_call_id,
                                            name,
                                            delta,
                                        )
                                        .await
                                }
                                _ => crate::agent::ObservationAction::Continue,
                            };
                            if let crate::agent::ObservationAction::Stop(reason) = observation {
                                yield Err(self.run_state().cancel_error(reason));
                                break;
                            }
                        } else if let StreamedAssistantContent::Text(text) = &content {
                            aggregated.push_str(&text.text);
                        }
                        // The provider's terminal record also drives the
                        // stream-finish observation.
                        if let StreamedAssistantContent::Final(final_record) = &content {
                            let observation = hooks.dispatch_stream_finish(final_record).await;
                            if let crate::agent::ObservationAction::Stop(reason) = observation {
                                let error = self.run_state().cancel_error(reason);
                                yield Ok(AgentRunItem::Assistant(content));
                                yield Err(error);
                                break;
                            }
                        }
                        yield Ok(AgentRunItem::Assistant(content));
                    }
                    AgentStreamItem::CompletionCall(call) => {
                        yield Ok(AgentRunItem::CompletionCall(call));
                    }
                    AgentStreamItem::ModelTurnRetried { turn } => {
                        yield Ok(AgentRunItem::ModelTurnRetried { turn });
                    }
                    AgentStreamItem::ToolExecutionCommitted {
                        tool_call,
                        internal_call_id,
                    } => {
                        yield Ok(AgentRunItem::ToolExecutionCommitted {
                            tool_call,
                            internal_call_id,
                        });
                    }
                    AgentStreamItem::User(content) => {
                        yield Ok(AgentRunItem::User(content));
                    }
                    AgentStreamItem::Final(response) => {
                        yield Ok(AgentRunItem::Final(response));
                    }
                }
            }
        })
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

    fn host_item_kind(item: &AgentStreamItem) -> &'static str {
        match item {
            AgentStreamItem::Assistant(_) => "assistant",
            AgentStreamItem::CompletionCall(_) => "completion_call",
            AgentStreamItem::BeforeModelCall { .. } => "before_model_call",
            AgentStreamItem::TurnFinished { .. } => "turn_finished",
            AgentStreamItem::ModelTurnRetried { .. } => "model_turn_retried",
            AgentStreamItem::InvalidToolCall(_) => "invalid_tool_call",
            AgentStreamItem::ToolCallPending { .. } => "tool_call_pending",
            AgentStreamItem::ToolCallsReady(_) => "tool_calls_ready",
            AgentStreamItem::ToolResultReady { .. } => "tool_result_ready",
            AgentStreamItem::ToolExecutionCommitted { .. } => "tool_execution_committed",
            AgentStreamItem::User(_) => "user",
            AgentStreamItem::Final(_) => "final",
        }
    }

    fn run_item_kind(item: &AgentRunItem) -> &'static str {
        match item {
            AgentRunItem::Assistant(_) => "assistant",
            AgentRunItem::CompletionCall(_) => "completion_call",
            AgentRunItem::ModelTurnRetried { .. } => "model_turn_retried",
            AgentRunItem::ToolExecutionCommitted { .. } => "tool_execution_committed",
            AgentRunItem::User(_) => "user",
            AgentRunItem::Final(_) => "final",
        }
    }

    #[test]
    fn host_and_driven_item_matches_are_deliberately_exhaustive() {
        assert_eq!(
            host_item_kind(&AgentStreamItem::ModelTurnRetried { turn: 1 }),
            "model_turn_retried"
        );
        assert_eq!(
            run_item_kind(&AgentRunItem::ModelTurnRetried { turn: 1 }),
            "model_turn_retried"
        );
    }

    #[test]
    fn driven_observations_preserve_the_host_protocol_serde_shape() {
        let assistant = StreamedAssistantContent::ToolCallDelta {
            id: "tc_1".to_string(),
            internal_call_id: "internal_tc_1".to_string(),
            content: ToolCallDeltaContent::Name("add".to_string()),
        };
        let completion_call = crate::agent::CompletionCall::new(2, Usage::new());
        let committed_call = tool_call("tc_1");
        let user = StreamedUserContent::tool_result(
            ToolResult {
                id: "tc_1".to_string(),
                call_id: None,
                content: OneOrMany::one(ToolResultContent::text("3")),
            },
            "internal_tc_1".to_string(),
        );
        let response = PromptResponse::empty();

        let pairs = [
            (
                AgentStreamItem::Assistant(assistant.clone()),
                AgentRunItem::Assistant(assistant),
            ),
            (
                AgentStreamItem::CompletionCall(completion_call),
                AgentRunItem::CompletionCall(completion_call),
            ),
            (
                AgentStreamItem::ModelTurnRetried { turn: 2 },
                AgentRunItem::ModelTurnRetried { turn: 2 },
            ),
            (
                AgentStreamItem::ToolExecutionCommitted {
                    tool_call: committed_call.clone(),
                    internal_call_id: "internal_tc_1".to_string(),
                },
                AgentRunItem::ToolExecutionCommitted {
                    tool_call: committed_call,
                    internal_call_id: "internal_tc_1".to_string(),
                },
            ),
            (
                AgentStreamItem::User(user.clone()),
                AgentRunItem::User(user),
            ),
            (
                AgentStreamItem::Final(response.clone()),
                AgentRunItem::Final(response),
            ),
        ];

        for (host, driven) in pairs {
            assert_eq!(
                serde_json::to_value(host).expect("serialize host item"),
                serde_json::to_value(driven).expect("serialize driven item")
            );
        }
    }

    fn assert_agent_run_stream_shape<T>()
    where
        T: Stream<Item = Result<AgentRunItem, PromptError>> + Unpin,
    {
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    fn assert_send<T: Send>() {}

    #[test]
    fn agent_run_stream_has_a_directly_pollable_concrete_shape() {
        assert_agent_run_stream_shape::<AgentRunStream>();

        #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
        assert_send::<AgentRunStream>();
    }

    mod inherent_next_without_stream_ext {
        use crate::stream::{AgentRunItem, AgentRunStream};

        #[tokio::test]
        async fn polls_without_the_extension_trait_or_caller_pinning() {
            let mut stream = AgentRunStream::new(futures::stream::iter([Ok(
                AgentRunItem::ModelTurnRetried { turn: 2 },
            )]));

            assert!(matches!(
                stream.next().await,
                Some(Ok(AgentRunItem::ModelTurnRetried { turn: 2 }))
            ));
            assert!(stream.next().await.is_none());
        }
    }

    #[tokio::test]
    async fn into_final_response_discards_observations_and_preserves_response() {
        let mut usage = Usage::new();
        usage.total_tokens = 7;
        let expected = PromptResponse::new("done", usage)
            .with_messages(vec![Message::user("go"), Message::assistant("done")])
            .with_output_tool_calls(2);
        let expected_json = serde_json::to_value(expected.clone()).expect("serialize response");
        let stream = AgentRunStream::new(futures::stream::iter([
            Ok(AgentRunItem::ModelTurnRetried { turn: 1 }),
            Ok(AgentRunItem::Final(expected)),
        ]));

        let response = stream.into_final_response().await.expect("final response");

        assert_eq!(
            serde_json::to_value(response.clone()).expect("serialize returned response"),
            expected_json
        );
        assert_eq!(response.output_tool_calls(), 2);
    }

    #[tokio::test]
    async fn into_final_response_propagates_prompt_errors() {
        let stream =
            AgentRunStream::new(futures::stream::iter([Err(PromptError::PromptCancelled {
                chat_history: vec![Message::user("go")],
                reason: "operator stop".to_string(),
            })]));

        let error = stream
            .into_final_response()
            .await
            .expect_err("stream should fail");

        assert!(matches!(
            error,
            PromptError::PromptCancelled { reason, .. } if reason == "operator stop"
        ));
    }

    #[tokio::test]
    async fn into_final_response_rejects_a_stream_without_final() {
        let stream = AgentRunStream::new(futures::stream::empty());

        let error = stream
            .into_final_response()
            .await
            .expect_err("missing final response should fail");

        assert!(matches!(
            error,
            PromptError::StreamEndedWithoutFinalResponse
        ));
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
                message_id: Some("msg_1".to_owned()),
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

    fn submission_for(call: &PendingToolCall, result: UserContent) -> ToolResultSubmission {
        ToolResultSubmission::new(
            call.internal_call_id.clone().expect("durable internal id"),
            result,
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
            .provide_tool_results(vec![submission_for(
                ready,
                tool_result_for(&ready.tool_call, "15"),
            )])
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
            .provide_tool_results(vec![submission_for(ready, preresolved)])
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

/// Ports of the deleted classic streaming driver's behavioral corpus onto
/// [`AgentStream::drive`], which is now the single streaming loop.
#[cfg(test)]
mod migrated_streaming_tests {
    use super::*;
    use crate::agent::hook::ObservationAction;
    use crate::agent::response::{TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER, tool_result_output};
    use crate::agent::run::streamed::merge_reasoning_blocks;
    use crate::agent::{AgentBuilder, CompletionCall, InvalidToolCallContext};
    use crate::hooks::{HookDecision, HookEntry, HookEvent};
    use crate::provider::MockScript;
    use crate::test_utils::{MockAddTool, MockBarrierTool, MockSubtractTool, MockToolError};
    use crate::tool::PortableTool;
    use rig_core::completion::{CompletionRequest, ToolDefinition};
    use rig_core::message::{
        DocumentSourceKind, ImageMediaType, Reasoning, ReasoningContent, Text, ToolChoice,
        ToolFunction, ToolResult, ToolResultContent,
    };
    use rig_core::streaming::ToolCallDeltaContent;
    use serde::Deserialize;
    use serde_json::json;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
    use std::time::Duration;

    // ---------- scripting helpers ----------

    fn script(turns: Vec<Vec<StreamedAssistantContent>>) -> MockScript {
        MockScript::from_responses(Vec::new()).with_streams(turns)
    }

    #[test]
    fn agent_and_runner_stream_run_return_agent_run_stream() {
        let agent = agent_builder(script(Vec::new())).build();
        let _: AgentRunStream = agent.stream_run("go");

        let agent = agent_builder(script(Vec::new())).build();
        let _: AgentRunStream = agent.runner("go").stream_run();
    }

    #[tokio::test]
    async fn driven_stream_is_exhausted_after_its_final_response() {
        let agent = agent_builder(script(vec![vec![text_item("done"), final_tokens(1)]])).build();
        let mut stream = agent.stream_run("go");

        loop {
            match stream.next().await {
                Some(Ok(AgentRunItem::Final(response))) => {
                    assert_eq!(response.output, "done");
                    break;
                }
                Some(Ok(_)) => {}
                Some(Err(error)) => panic!("driven run failed before its final response: {error}"),
                None => panic!("driven run ended without a final response"),
            }
        }

        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn driven_stream_is_exhausted_after_its_terminal_error() {
        let agent = agent_builder(script(Vec::new()))
            .add_hook(stop_before_completion_entry())
            .build();
        let mut stream = agent.stream_run("go");

        loop {
            match stream.next().await {
                Some(Err(PromptError::PromptCancelled { reason, .. })) => {
                    assert_eq!(reason, "agent streaming stopped");
                    break;
                }
                Some(Err(error)) => panic!("unexpected terminal error: {error}"),
                Some(Ok(AgentRunItem::Final(_))) => {
                    panic!("driven run completed instead of yielding its hook error")
                }
                Some(Ok(_)) => {}
                None => panic!("driven run ended without its hook error"),
            }
        }

        assert!(stream.next().await.is_none());
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn dropping_next_preserves_the_in_flight_hook_future() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let polls = Arc::new(AtomicUsize::new(0));
        let hook_invocations = Arc::clone(&invocations);
        let hook_polls = Arc::clone(&polls);
        let hook = HookEntry::new("pending-once", move |event| {
            let targeted = matches!(event, HookEvent::BeforeModelCall { .. });
            if targeted {
                hook_invocations.fetch_add(1, Ordering::SeqCst);
            }
            let hook_polls = Arc::clone(&hook_polls);
            let mut pending_once = targeted;
            futures::future::poll_fn(move |cx| {
                if pending_once {
                    pending_once = false;
                    hook_polls.fetch_add(1, Ordering::SeqCst);
                    cx.waker().wake_by_ref();
                    Poll::Pending
                } else {
                    if targeted {
                        hook_polls.fetch_add(1, Ordering::SeqCst);
                    }
                    Poll::Ready(HookDecision::Continue)
                }
            })
        });
        let agent = agent_builder(script(vec![vec![text_item("done"), final_tokens(1)]]))
            .add_hook(hook)
            .build();
        let mut stream = agent.stream_run("go");

        {
            let next = stream.next();
            futures::pin_mut!(next);
            assert!(futures::poll!(next).is_pending());
        }

        let mut finals = 0;
        while let Some(item) = stream.next().await {
            if matches!(item.expect("driven item"), AgentRunItem::Final(_)) {
                finals += 1;
            }
        }

        assert_eq!(invocations.load(Ordering::SeqCst), 1);
        assert_eq!(polls.load(Ordering::SeqCst), 2);
        assert_eq!(finals, 1);
    }

    fn text_item(text: &str) -> StreamedAssistantContent {
        StreamedAssistantContent::Text(Text::new(text))
    }

    fn cited_text_item(text: &str, metadata: serde_json::Value) -> StreamedAssistantContent {
        StreamedAssistantContent::Text(Text {
            text: text.to_string(),
            additional_params: Some(metadata),
        })
    }

    fn call_item(id: &str, name: &str, args: serde_json::Value) -> StreamedAssistantContent {
        StreamedAssistantContent::ToolCall {
            tool_call: ToolCall::new(id.to_string(), ToolFunction::new(name.to_string(), args)),
            internal_call_id: format!("ic-{id}"),
        }
    }

    fn call_item_with_call_id(
        id: &str,
        call_id: &str,
        name: &str,
        args: serde_json::Value,
    ) -> StreamedAssistantContent {
        StreamedAssistantContent::ToolCall {
            tool_call: ToolCall::new(id.to_string(), ToolFunction::new(name.to_string(), args))
                .with_call_id(call_id.to_string()),
            internal_call_id: format!("ic-{id}"),
        }
    }

    fn name_delta(id: &str, internal: &str, name: &str) -> StreamedAssistantContent {
        StreamedAssistantContent::ToolCallDelta {
            id: id.to_string(),
            internal_call_id: internal.to_string(),
            content: ToolCallDeltaContent::Name(name.to_string()),
        }
    }

    fn args_delta(id: &str, internal: &str, args: &str) -> StreamedAssistantContent {
        StreamedAssistantContent::ToolCallDelta {
            id: id.to_string(),
            internal_call_id: internal.to_string(),
            content: ToolCallDeltaContent::Delta(args.to_string()),
        }
    }

    fn reasoning_block(id: Option<&str>, text: &str) -> StreamedAssistantContent {
        let mut reasoning = Reasoning::new(text);
        reasoning.id = id.map(str::to_string);
        StreamedAssistantContent::Reasoning(reasoning)
    }

    fn reasoning_delta_item(id: Option<&str>, text: &str) -> StreamedAssistantContent {
        StreamedAssistantContent::ReasoningDelta {
            id: id.map(str::to_string),
            reasoning: text.to_string(),
        }
    }

    fn total_usage(total_tokens: u64) -> Usage {
        let mut usage = Usage::new();
        usage.total_tokens = total_tokens;
        usage
    }

    fn split_usage(input_tokens: u64, output_tokens: u64) -> Usage {
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

    fn final_tokens(total_tokens: u64) -> StreamedAssistantContent {
        StreamedAssistantContent::Final(StreamFinal::new("mock", total_usage(total_tokens)))
    }

    fn final_usage(usage: Usage) -> StreamedAssistantContent {
        StreamedAssistantContent::Final(StreamFinal::new("mock", usage))
    }

    #[tokio::test]
    async fn close_after_complete_tool_call_discards_the_uncommitted_batch() {
        let script = script(vec![
            vec![
                call_item("call_1", "add", json!({"a": 1, "b": 2})),
                final_tokens(9),
            ],
            vec![text_item("recovered"), final_tokens(5)],
        ]);
        let mut driver = AgentStream::new(
            AgentConfig::new(),
            ProviderConfig::Mock(script),
            Arc::new(Runtime::new()),
            "go",
        )
        .with_tools(ToolCatalog::new(vec![ToolDefinition {
            name: "add".to_string(),
            description: "add".to_string(),
            parameters: json!({"type": "object"}),
        }]));

        let AgentRunStep::CallModel {
            prompt,
            history,
            turn,
        } = driver.run.next_step().expect("first model call")
        else {
            panic!("expected CallModel");
        };
        driver
            .open_turn(prompt, history, turn)
            .await
            .expect("stream opens");
        driver
            .poll_active_turn_once()
            .await
            .expect("complete tool call is provisional");
        assert_eq!(
            driver
                .active
                .as_ref()
                .expect("active attempt")
                .assembler
                .partial_turn(None)
                .pending_tool_calls
                .len(),
            1
        );

        driver.close_turn();
        assert!(driver.active.is_none());
        assert_eq!(driver.run.turn(), 0);
        assert_eq!(driver.usage().total_tokens, 0);
        assert!(driver.run_state().completion_calls().is_empty());
        assert_eq!(driver.run_state().messages().len(), 1);

        let mut saw_tool_gate = false;
        let mut final_output = None;
        while let Some(item) = driver.next_item().await {
            match item.expect("retry succeeds") {
                AgentStreamItem::ToolCallsReady(_) => saw_tool_gate = true,
                AgentStreamItem::Final(response) => final_output = Some(response.output),
                _ => {}
            }
        }
        assert!(!saw_tool_gate);
        assert_eq!(final_output.as_deref(), Some("recovered"));
        assert_eq!(driver.usage().total_tokens, 5);
    }

    /// Any provider item after its terminal record is malformed and rejected
    /// rather than being folded into the completed turn.
    #[tokio::test]
    async fn provider_item_after_the_final_is_rejected() {
        let agent = AgentBuilder::new(ProviderConfig::Mock(script(vec![vec![
            text_item("before the final"),
            final_tokens(7),
            text_item("stray content after the final"),
        ]])))
        .build();

        let mut stream = agent.stream_run("go");
        let mut error = None;
        while let Some(item) = stream.next().await {
            if let Err(failure) = item {
                error = Some(failure);
                break;
            }
        }
        let error = error.expect("a malformed provider stream must fail the run");
        assert!(
            error
                .to_string()
                .contains("provider stream emitted an item after its final response"),
            "got {error}"
        );
    }

    fn agent_builder(script: MockScript) -> AgentBuilder {
        AgentBuilder::new(ProviderConfig::Mock(script))
    }

    fn citation_metadata() -> serde_json::Value {
        json!({
            "citations": [{
                "type": "web_search_result_location",
                "cited_text": "Claude Shannon was born in 1916.",
                "url": "https://example.com/shannon",
                "title": "Claude Shannon",
                "encrypted_index": "encrypted-reference"
            }]
        })
    }

    // ---------- collection helpers ----------

    /// Everything one driven stream produced, in order.
    #[derive(Default)]
    struct Collected {
        items: Vec<AgentRunItem>,
        error: Option<PromptError>,
    }

    impl Collected {
        fn tool_calls(&self) -> Vec<ToolCall> {
            self.items
                .iter()
                .filter_map(|item| match item {
                    AgentRunItem::Assistant(StreamedAssistantContent::ToolCall {
                        tool_call,
                        ..
                    }) => Some(tool_call.clone()),
                    _ => None,
                })
                .collect()
        }

        fn deltas(&self) -> Vec<(String, String, ToolCallDeltaContent)> {
            self.items
                .iter()
                .filter_map(|item| match item {
                    AgentRunItem::Assistant(StreamedAssistantContent::ToolCallDelta {
                        id,
                        internal_call_id,
                        content,
                    }) => Some((id.clone(), internal_call_id.clone(), content.clone())),
                    _ => None,
                })
                .collect()
        }

        fn streamed_text(&self) -> String {
            self.items
                .iter()
                .filter_map(|item| match item {
                    AgentRunItem::Assistant(StreamedAssistantContent::Text(text)) => {
                        Some(text.text.as_str())
                    }
                    _ => None,
                })
                .collect()
        }

        fn tool_results(&self) -> Vec<(ToolResult, String)> {
            self.items
                .iter()
                .filter_map(|item| match item {
                    AgentRunItem::User(StreamedUserContent::ToolResult {
                        tool_result,
                        internal_call_id,
                    }) => Some((tool_result.clone(), internal_call_id.clone())),
                    _ => None,
                })
                .collect()
        }

        fn completion_calls(&self) -> Vec<CompletionCall> {
            self.items
                .iter()
                .filter_map(|item| match item {
                    AgentRunItem::CompletionCall(call) => Some(*call),
                    _ => None,
                })
                .collect()
        }

        fn saw_provider_final(&self) -> bool {
            self.items.iter().any(|item| {
                matches!(
                    item,
                    AgentRunItem::Assistant(StreamedAssistantContent::Final(_))
                )
            })
        }

        fn final_response(&self) -> Option<&PromptResponse> {
            self.items.iter().find_map(|item| match item {
                AgentRunItem::Final(response) => Some(response),
                _ => None,
            })
        }

        fn expect_final(&self) -> &PromptResponse {
            self.final_response()
                .unwrap_or_else(|| panic!("expected a final response, error was {:?}", self.error))
        }

        fn expect_error(&self) -> &PromptError {
            self.error.as_ref().expect("expected a streaming error")
        }
    }

    async fn collect(
        stream: impl futures::Stream<Item = Result<AgentRunItem, PromptError>>,
    ) -> Collected {
        futures::pin_mut!(stream);
        let mut collected = Collected::default();
        while let Some(item) = stream.next().await {
            match item {
                Ok(item) => collected.items.push(item),
                Err(error) => {
                    collected.error = Some(error);
                    break;
                }
            }
        }
        collected
    }

    // ---------- hook helpers ----------

    fn hook_entry(
        name: &str,
        decide: impl Fn(HookEvent) -> HookDecision + Send + Sync + 'static,
    ) -> HookEntry {
        HookEntry::sync(name, decide)
    }

    fn stop_before_completion_entry() -> HookEntry {
        hook_entry("stop-before-completion", |event| {
            let HookEvent::BeforeModelCall { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::CompletionCall(CompletionCallAction::stop("agent streaming stopped"))
        })
    }

    fn repair_default_api_entry() -> HookEntry {
        hook_entry("repair-default-api", |event| {
            let HookEvent::InvalidToolCall(context) = event else {
                return HookDecision::Continue;
            };
            assert_eq!(context.tool_name, "default_api");
            HookDecision::InvalidToolCall(InvalidToolCallAction::repair("add"))
        })
    }

    fn rewrite_add_args_entry(arguments: serde_json::Value) -> HookEntry {
        hook_entry("rewrite-add-args", move |event| {
            let HookEvent::ToolCall { call, .. } = event else {
                return HookDecision::Continue;
            };
            assert_eq!(call.function.name, "add");
            HookDecision::ToolCall(ToolCallAction::rewrite(arguments.clone()))
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

    fn terminate_on_stream_finish_entry() -> HookEntry {
        hook_entry("terminate-on-stream-finish", |event| {
            let HookEvent::StreamResponseFinish { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::Observation(ObservationAction::stop("stop after completion call"))
        })
    }

    /// Fails the test if the driver reaches a tool-call delta, a tool-call
    /// decision point, or the stream-finish observation for an invalid turn.
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

    #[derive(Clone, Default)]
    struct RecordedInvalidToolCalls(Arc<Mutex<Vec<InvalidToolCallContext>>>);

    impl RecordedInvalidToolCalls {
        fn observed(&self) -> Vec<InvalidToolCallContext> {
            self.0.lock().expect("mutex").clone()
        }

        fn entry(&self) -> HookEntry {
            let contexts = self.0.clone();
            hook_entry("record-invalid-tool-call", move |event| {
                let HookEvent::InvalidToolCall(context) = event else {
                    return HookDecision::Continue;
                };
                contexts.lock().expect("mutex").push((*context).clone());
                HookDecision::InvalidToolCall(InvalidToolCallAction::fail())
            })
        }
    }

    type RecordedToolCallDelta = (String, String, Option<String>, String);

    #[derive(Clone, Default)]
    struct RecordedToolCallDeltas(Arc<Mutex<Vec<RecordedToolCallDelta>>>);

    impl RecordedToolCallDeltas {
        fn observed(&self) -> Vec<RecordedToolCallDelta> {
            self.0.lock().expect("mutex").clone()
        }

        fn entry(&self) -> HookEntry {
            self.entry_with(ObservationAction::continue_run())
        }

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
                deltas.lock().expect("mutex").push((
                    tool_call_id.to_string(),
                    internal_call_id.to_string(),
                    tool_name.map(|name| name.to_string()),
                    delta.to_string(),
                ));
                HookDecision::Observation(action.clone())
            })
            .observing_deltas()
        }
    }

    #[derive(Clone, Default)]
    struct RecordedTextDeltas(Arc<Mutex<Vec<(String, String)>>>);

    impl RecordedTextDeltas {
        fn observed(&self) -> Vec<(String, String)> {
            self.0.lock().expect("mutex").clone()
        }

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
                    .expect("mutex")
                    .push((delta.to_string(), aggregated.to_string()));
                HookDecision::Observation(ObservationAction::continue_run())
            })
            .observing_deltas()
        }
    }

    // ---------- counting tools ----------

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

    fn arithmetic_parameters() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "The first operand"},
                "y": {"type": "number", "description": "The second operand"}
            },
            "required": ["x", "y"],
        })
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
            arithmetic_parameters()
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
            arithmetic_parameters()
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(args.x - args.y)
        }
    }

    // ---------- history assertion helpers ----------

    fn history_of(request: &CompletionRequest) -> Vec<Message> {
        request.chat_history.iter().cloned().collect()
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
                            ReasoningContent::Text { text, .. } if text == expected_reasoning
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
            matches!((reasoning_index, tool_index), (Some(r), Some(t)) if r < t)
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
                            ReasoningContent::Text { text, .. } if text == expected_reasoning
                        ))
                )
            });
            let text_index = content.iter().position(
                |item| matches!(item, AssistantContent::Text(text) if text.text == expected_text),
            );
            let tool_index = content.iter().position(|item| {
                matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == tool_name
                )
            });
            matches!(
                (reasoning_index, text_index, tool_index),
                (Some(r), Some(t), Some(c)) if r < t && t < c
            )
        })
    }

    fn text_metadata(content: &OneOrMany<AssistantContent>) -> Option<&serde_json::Value> {
        content.iter().find_map(|item| match item {
            AssistantContent::Text(text) => text.additional_params.as_ref(),
            _ => None,
        })
    }

    fn expect_unknown_tool_call(error: &PromptError) -> (&str, &[String], &[String], &[Message]) {
        match error {
            PromptError::UnknownToolCall {
                tool_name,
                available_tools,
                allowed_tools,
                chat_history,
            } => (tool_name, available_tools, allowed_tools, chat_history),
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
    }

    // ---------- sans-IO unit ports ----------

    #[test]
    fn merge_reasoning_blocks_preserves_order_and_signatures() {
        let mut accumulated = Vec::new();
        let mut first = Reasoning::new("");
        first.id = Some("rs_1".to_string());
        first.content = vec![ReasoningContent::Text {
            text: "step-1".to_string(),
            signature: Some("sig-1".to_string()),
        }];
        let mut second = Reasoning::new("");
        second.id = Some("rs_1".to_string());
        second.content = vec![
            ReasoningContent::Text {
                text: "step-2".to_string(),
                signature: Some("sig-2".to_string()),
            },
            ReasoningContent::Summary("summary".to_string()),
        ];

        merge_reasoning_blocks(&mut accumulated, &first);
        merge_reasoning_blocks(&mut accumulated, &second);

        assert_eq!(accumulated.len(), 1);
        let merged = accumulated.first().expect("accumulated reasoning");
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
        let mut first = Reasoning::new("");
        first.id = Some("rs_a".to_string());
        first.content = vec![ReasoningContent::Text {
            text: "step-1".to_string(),
            signature: None,
        }];
        let mut incoming = Reasoning::new("");
        incoming.id = Some("rs_b".to_string());
        incoming.content = vec![ReasoningContent::Text {
            text: "step-2".to_string(),
            signature: None,
        }];
        let mut accumulated = vec![first];

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
        let mut first = Reasoning::new("");
        first.content = vec![ReasoningContent::Text {
            text: "first".to_string(),
            signature: None,
        }];
        let mut incoming = Reasoning::new("");
        incoming.content = vec![ReasoningContent::Text {
            text: "second".to_string(),
            signature: None,
        }];
        let mut accumulated = vec![first];

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
        let instruction = json!({"instruction": "Use the image part to answer."});
        let mut content = OneOrMany::one(ToolResultContent::json(instruction.clone()));
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
            Some(ToolResultContent::Json { value }) => assert_eq!(value, &instruction),
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

    /// The per-call record's wire shape (and its tolerance of legacy
    /// `"usage": null`) is what durable hosts persist.
    #[test]
    fn completion_calls_stream_item_serializes_and_deserializes_expected_shape() {
        let call = CompletionCall::new(2, split_usage(3, 4));
        let value = serde_json::to_value(call).expect("serialize completion call");
        assert_eq!(
            value,
            json!({
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
        let restored: CompletionCall =
            serde_json::from_value(value).expect("deserialize completion call");
        assert_eq!(restored, call);

        // Unreported usage is the zero-valued sentinel, and payloads written
        // before the `Option<Usage>` encoding was dropped still load.
        let unreported = CompletionCall::new(3, Usage::new());
        let value = serde_json::to_value(unreported).expect("serialize missing usage");
        assert_eq!(value["usage"]["total_tokens"], 0);
        let legacy: CompletionCall =
            serde_json::from_value(json!({"call_index": 3, "usage": null}))
                .expect("legacy null-usage record should deserialize");
        assert_eq!(legacy, unreported);

        // The same record travels as a stream item.
        let item = AgentRunItem::CompletionCall(call);
        let restored: AgentRunItem =
            serde_json::from_str(&serde_json::to_string(&item).expect("serialize item"))
                .expect("deserialize item");
        assert!(matches!(
            restored,
            AgentRunItem::CompletionCall(restored) if restored == call
        ));
    }

    #[test]
    fn final_response_serializes_completion_calls_with_missing_usage() {
        let mut response = PromptResponse::new("done".to_string(), split_usage(3, 4));
        response.completion_calls = vec![
            CompletionCall::new(0, Usage::new()),
            CompletionCall::new(1, split_usage(3, 4)),
        ];
        let item = AgentRunItem::Final(response);
        let value = serde_json::to_value(&item).expect("serialize final response");
        let final_value = value.get("Final").expect("final payload");

        assert_eq!(
            final_value.get("completion_calls"),
            Some(&json!([
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

    // ---------- driven-stream behavior ----------

    /// Hooks configured on the agent reach the driven stream: a
    /// before-call stop cancels before any provider call.
    #[tokio::test]
    async fn public_streaming_request_constructor_preserves_agent_hooks() {
        let script = script(vec![vec![text_item("should not run"), final_tokens(0)]]);
        let agent = agent_builder(script.clone())
            .add_hook(stop_before_completion_entry())
            .build();

        let collected = collect(agent.stream_run("go")).await;

        assert!(matches!(
            collected.expect_error(),
            PromptError::PromptCancelled { reason, .. } if reason == "agent streaming stopped"
        ));
        assert_eq!(script.calls(), 0);
    }

    /// A tool batch commits atomically: when the run rejects the results,
    /// neither the execution markers nor the results are surfaced.
    #[tokio::test]
    async fn execution_commit_items_are_not_emitted_when_run_commit_fails() {
        let mut config = AgentConfig::new();
        config.max_turns = Some(3);
        let mut stream = AgentStream::new(
            config,
            ProviderConfig::Mock(script(vec![
                vec![
                    call_item("tool_call_1", "add", json!({"x": 1, "y": 2})),
                    final_tokens(4),
                ],
                vec![text_item("done"), final_tokens(6)],
            ])),
            Arc::new(Runtime::new()),
            "do tool work",
        )
        .with_tools(ToolCatalog::new(vec![ToolDefinition {
            name: "add".to_string(),
            description: "Adds".to_string(),
            parameters: arithmetic_parameters(),
        }]));

        let calls = loop {
            match stream.next_item().await.expect("item").expect("no error") {
                AgentStreamItem::ToolCallsReady(calls) => break calls,
                _ => continue,
            }
        };
        let call = calls.first().expect("one call");
        // A result whose id does not match the pending call: execution
        // settled, but the run rejects the commit.
        let mismatched = UserContent::tool_result(
            "mismatched_call".to_string(),
            OneOrMany::one(ToolResultContent::text("3")),
        );
        let error = stream
            .provide_tool_results(vec![ToolResultSubmission::new(
                call.internal_call_id.clone().expect("durable internal id"),
                mismatched,
            )])
            .expect_err("the mismatched result must fail run-state commit");
        assert!(!format!("{error}").is_empty());
        assert_eq!(call.tool_call.id, "tool_call_1");

        // Nothing commit-labelled escapes after the rejected commit.
        let mut saw_commit = false;
        let mut saw_result = false;
        while let Some(item) = stream.next_item().await {
            match item {
                Ok(AgentStreamItem::ToolExecutionCommitted { .. }) => saw_commit = true,
                Ok(AgentStreamItem::User(_)) => saw_result = true,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        assert!(!saw_commit, "a failed run-state commit cannot be announced");
        assert!(!saw_result, "an uncommitted result cannot be surfaced");
    }

    #[tokio::test]
    async fn stream_prompt_continues_after_tool_call_turn() {
        let script = script(vec![
            vec![
                call_item_with_call_id("tool_call_1", "call_1", "add", json!({"x": 1, "y": 2})),
                final_tokens(4),
            ],
            vec![text_item("done"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("do tool work")
                .history(Vec::<Message>::new())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert_eq!(collected.tool_calls().len(), 1);
        assert_eq!(collected.tool_results().len(), 1);
        assert_eq!(collected.streamed_text(), "done");
        let final_response = collected.expect_final();
        assert_eq!(final_response.output, "done");
        let history = final_response
            .messages
            .as_ref()
            .expect("final response history");
        assert!(history_contains_text(history, "done"));

        // The follow-up request carries [prompt, assistant tool call, result].
        let requests = script.requests();
        assert_eq!(requests.len(), 2);
        let follow_up = history_of(&requests[1]);
        assert_eq!(follow_up.len(), 3, "{follow_up:?}");
        assert!(matches!(
            follow_up.first(),
            Some(Message::User { content })
                if matches!(content.first(), UserContent::Text(text) if text.text == "do tool work")
        ));
        assert!(matches!(
            follow_up.get(1),
            Some(Message::Assistant { content, .. })
                if matches!(
                    content.first(),
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.call_id.as_deref() == Some("call_1")
                )
        ));
        assert!(matches!(
            follow_up.get(2),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::ToolResult(result)
                        if result.id == "tool_call_1"
                            && result.call_id.as_deref() == Some("call_1")
                )
        ));
    }

    /// `tool_concurrency` reaches the streaming executor: two
    /// barrier-synchronized tools only finish if they run concurrently.
    #[tokio::test]
    async fn streaming_prompt_request_tool_concurrency_runs_tools_concurrently() {
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let script = script(vec![
            vec![
                call_item("b1", "barrier_tool", json!({})),
                call_item("b2", "barrier_tool", json!({})),
                final_tokens(0),
            ],
            vec![text_item("done"), final_tokens(0)],
        ]);
        let agent = agent_builder(script)
            .tool(MockBarrierTool::new(barrier))
            .build();

        let drive = async {
            let collected = collect(
                agent
                    .runner("hit the barrier twice")
                    .max_turns(3)
                    .tool_concurrency(2)
                    .stream_run(),
            )
            .await;
            assert!(collected.error.is_none(), "{:?}", collected.error);
            assert_eq!(collected.expect_final().output, "done");
        };

        tokio::time::timeout(Duration::from_secs(5), drive)
            .await
            .expect("streamed tools must run concurrently, not deadlock at the barrier");
    }

    #[tokio::test]
    async fn multiple_valid_streaming_tool_calls_execute_after_batch_validation() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let subtract_calls = Arc::new(AtomicU32::new(0));
        let script = script(vec![
            vec![
                call_item_with_call_id("tool_call_1", "call_1", "add", json!({"x": 1, "y": 2})),
                call_item_with_call_id(
                    "tool_call_2",
                    "call_2",
                    "subtract",
                    json!({"x": 8, "y": 3}),
                ),
                final_tokens(4),
            ],
            vec![text_item("done"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .tool(CountingSubtractTool {
                calls: subtract_calls.clone(),
            })
            .build();

        let collected = collect(agent.runner("use tools").max_turns(3).stream_run()).await;

        assert_eq!(
            collected
                .tool_calls()
                .into_iter()
                .map(|call| call.function.name)
                .collect::<Vec<_>>(),
            vec!["add".to_string(), "subtract".to_string()]
        );
        assert_eq!(
            collected
                .tool_results()
                .into_iter()
                .map(|(result, _)| result.id)
                .collect::<Vec<_>>(),
            vec!["tool_call_1".to_string(), "tool_call_2".to_string()]
        );
        assert_eq!(add_calls.load(Ordering::SeqCst), 1);
        assert_eq!(subtract_calls.load(Ordering::SeqCst), 1);
        assert_eq!(collected.expect_final().output, "done");
        assert_eq!(script.calls(), 2);
    }

    #[tokio::test]
    async fn unknown_tool_call_fails_before_streaming_second_request() {
        let script = script(vec![
            vec![
                call_item("tool_call_1", "default_api", json!({"x": 1, "y": 2})),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(panic_on_unknown_tool_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert!(collected.tool_calls().is_empty());
        let (tool_name, available, allowed, history) =
            expect_unknown_tool_call(collected.expect_error());
        assert_eq!(tool_name, "default_api");
        assert_eq!(available, ["add".to_string()]);
        assert_eq!(allowed, ["add".to_string()]);
        assert!(history_contains_tool_call(history, "default_api"));
        assert_eq!(script.calls(), 1);
    }

    #[tokio::test]
    async fn completed_unknown_tool_call_after_text_fails_before_finish_hook_or_later_emit() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let script = script(vec![
            vec![
                text_item("thinking "),
                call_item("tool_call_1", "default_api", json!({"x": 1, "y": 2})),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(panic_on_unknown_tool_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert_eq!(collected.streamed_text(), "thinking ");
        assert!(collected.completion_calls().is_empty());
        assert!(!collected.saw_provider_final());
        assert!(collected.final_response().is_none());
        assert!(collected.tool_calls().is_empty());
        assert!(collected.tool_results().is_empty());
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let (tool_name, available, allowed, history) =
            expect_unknown_tool_call(collected.expect_error());
        assert_eq!(tool_name, "default_api");
        assert_eq!(available, ["add".to_string()]);
        assert_eq!(allowed, ["add".to_string()]);
        assert!(history_contains_tool_call(history, "default_api"));
        assert_eq!(script.calls(), 1);
    }

    #[tokio::test]
    async fn mixed_streaming_tool_calls_fail_before_any_tool_execution() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let script = script(vec![
            vec![
                call_item_with_call_id("tool_call_1", "call_1", "add", json!({"x": 1, "y": 2})),
                call_item("tool_call_2", "default_api", json!({"x": 3, "y": 4})),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let collected = collect(
            agent
                .runner("use tools")
                .add_hook(panic_on_unknown_tool_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert!(collected.completion_calls().is_empty());
        assert!(collected.tool_calls().is_empty());
        assert!(collected.tool_results().is_empty());
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let (tool_name, _, _, history) = expect_unknown_tool_call(collected.expect_error());
        assert_eq!(tool_name, "default_api");
        assert!(history_contains_tool_call(history, "default_api"));
        assert_eq!(script.calls(), 1);
    }

    #[tokio::test]
    async fn disallowed_specific_tool_call_fails_before_streaming_second_request() {
        let script = script(vec![
            vec![
                call_item("tool_call_1", "subtract", json!({"x": 3, "y": 1})),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let collected = collect(
            agent
                .runner("use the allowed tool")
                .add_hook(panic_on_unknown_tool_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert!(collected.tool_calls().is_empty());
        let (tool_name, available, allowed, history) =
            expect_unknown_tool_call(collected.expect_error());
        assert_eq!(tool_name, "subtract");
        assert_eq!(available, ["add".to_string(), "subtract".to_string()]);
        assert_eq!(allowed, ["add".to_string()]);
        assert!(history_contains_tool_call(history, "subtract"));
        assert_eq!(script.calls(), 1);
    }

    #[tokio::test]
    async fn mixed_specific_tool_calls_fail_before_any_tool_execution() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let script = script(vec![
            vec![
                call_item("tool_call_1", "add", json!({"x": 1, "y": 2})),
                call_item("tool_call_2", "subtract", json!({"x": 3, "y": 1})),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let collected = collect(
            agent
                .runner("use the allowed tool")
                .add_hook(panic_on_unknown_tool_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert!(collected.tool_calls().is_empty());
        assert!(collected.tool_results().is_empty());
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let (tool_name, available, allowed, history) =
            expect_unknown_tool_call(collected.expect_error());
        assert_eq!(tool_name, "subtract");
        assert_eq!(available, ["add".to_string(), "subtract".to_string()]);
        assert_eq!(allowed, ["add".to_string()]);
        assert!(history_contains_tool_call(history, "subtract"));
        assert_eq!(script.calls(), 1);
    }

    #[tokio::test]
    async fn tool_choice_none_rejects_streaming_tool_call() {
        let script = script(vec![
            vec![
                call_item("tool_call_1", "add", json!({"x": 1, "y": 2})),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let collected = collect(
            agent
                .runner("do not use tools")
                .add_hook(panic_on_unknown_tool_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert!(collected.tool_calls().is_empty());
        let (tool_name, available, allowed, history) =
            expect_unknown_tool_call(collected.expect_error());
        assert_eq!(tool_name, "add");
        assert_eq!(available, ["add".to_string()]);
        assert!(allowed.is_empty());
        assert!(history_contains_tool_call(history, "add"));
        assert_eq!(script.calls(), 1);
    }

    // ---------- invalid tool-call recovery ----------

    #[tokio::test]
    async fn invalid_tool_call_hook_can_repair_streaming_tool_name() {
        let script = script(vec![
            vec![
                call_item("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
                final_tokens(4),
            ],
            vec![text_item("done"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(repair_default_api_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        let calls = collected.tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls.first().map(|call| call.function.name.as_str()),
            Some("add"),
            "the repaired name must be the one announced"
        );
        let results = collected.tool_results();
        assert_eq!(results.len(), 1);
        assert!(results[0].0.content.iter().any(|content| matches!(
            content,
            ToolResultContent::Json { value } if value == &json!(5)
        )));
        assert_eq!(collected.expect_final().output, "done");
        assert_eq!(script.calls(), 2);
    }

    #[tokio::test]
    async fn repaired_name_with_argument_rewrite_announces_provider_arguments() {
        let original_args = json!({"x": 2, "y": 3});
        let rewritten_args = json!({"x": 20, "y": 30});
        let script = script(vec![
            vec![
                call_item("tool_call_1", "default_api", original_args.clone()),
                final_tokens(4),
            ],
            vec![text_item("done"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(repair_default_api_entry())
                .add_hook(rewrite_add_args_entry(rewritten_args.clone()))
                .max_turns(3)
                .stream_run(),
        )
        .await;

        let announced = collected.tool_calls();
        assert_eq!(announced.len(), 1);
        assert_eq!(announced[0].function.name, "add");
        assert_eq!(announced[0].function.arguments, original_args);

        let committed = collected.items.iter().find_map(|item| match item {
            AgentRunItem::ToolExecutionCommitted { tool_call, .. } => Some(tool_call),
            _ => None,
        });
        let committed = committed.expect("effective execution commit");
        assert_eq!(committed.function.name, "add");
        assert_eq!(committed.function.arguments, rewritten_args);
        assert_eq!(collected.expect_final().output, "done");
        assert_eq!(script.calls(), 2);
    }

    #[tokio::test]
    async fn invalid_tool_call_context_uses_completed_streaming_tool_call_provider_id() {
        let invalid_hook = RecordedInvalidToolCalls::default();
        let script = script(vec![vec![
            call_item_with_call_id(
                "tool_call_1",
                "provider_call_1",
                "default_api",
                json!({"x": 2, "y": 3}),
            ),
            final_tokens(4),
        ]]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(invalid_hook.entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        collected.expect_error();
        assert_eq!(script.calls(), 1);
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
        let script = script(vec![
            vec![
                call_item_with_call_id(
                    "tool_call_1",
                    "call_1",
                    "default_api",
                    json!({"x": 2, "y": 3}),
                ),
                final_tokens(4),
            ],
            vec![text_item("continued"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(skip_default_api_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        let (skipped, internal_call_id) = collected
            .tool_results()
            .into_iter()
            .next()
            .expect("skip recovery should emit a synthetic tool result");
        assert!(!internal_call_id.is_empty());
        assert_eq!(skipped.id, "tool_call_1");
        assert_eq!(skipped.call_id.as_deref(), Some("call_1"));
        assert!(skipped.content.iter().any(|content| matches!(
            content,
            ToolResultContent::Text(text) if text.text == "default_api was skipped"
        )));
        assert_eq!(collected.expect_final().output, "continued");
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);

        let requests = script.requests();
        assert_eq!(requests.len(), 2);
        let follow_up = history_of(&requests[1]);
        assert!(matches!(
            follow_up.get(2),
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
        let script = script(vec![
            vec![
                text_item("checking "),
                call_item_with_call_id("tool_call_1", "call_1", "add", json!({"x": 2, "y": 3})),
                call_item_with_call_id(
                    "tool_call_2",
                    "call_2",
                    "default_api",
                    json!({"x": 4, "y": 5}),
                ),
                final_tokens(4),
            ],
            vec![text_item("retried"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(retry_default_api_entry())
                .max_turns(3)
                .max_invalid_tool_call_retries(1)
                .stream_run(),
        )
        .await;

        let final_response = collected.expect_final();
        assert_eq!(final_response.output, "retried");
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let expected_calls = vec![
            CompletionCall::new(0, total_usage(4)),
            CompletionCall::new(1, total_usage(6)),
        ];
        assert_eq!(collected.completion_calls(), expected_calls);
        assert_eq!(final_response.completion_calls, expected_calls);
        assert_eq!(final_response.usage.total_tokens, 10);

        let requests = script.requests();
        assert_eq!(requests.len(), 2);
        let retry_history = history_of(&requests[1]);
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
                        AssistantContent::ToolCall(call)
                            if call.id == "tool_call_1" && call.function.name == "add"
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(call)
                            if call.id == "tool_call_2" && call.function.name == "default_api"
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
        let script = script(vec![
            vec![
                text_item("checking "),
                call_item_with_call_id("tool_call_1", "call_1", "add", json!({"x": 2, "y": 3})),
                call_item_with_call_id(
                    "tool_call_2",
                    "call_2",
                    "default_api",
                    json!({"x": 4, "y": 5}),
                ),
                final_tokens(4),
            ],
            vec![text_item("continued"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(skip_default_api_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        let (skipped, _) = collected
            .tool_results()
            .into_iter()
            .next()
            .expect("skip recovery should emit a synthetic tool result");
        assert_eq!(skipped.id, "tool_call_2");
        assert_eq!(skipped.call_id.as_deref(), Some("call_2"));
        assert_eq!(collected.expect_final().output, "continued");
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);

        let requests = script.requests();
        assert_eq!(requests.len(), 2);
        let follow_up = history_of(&requests[1]);
        assert_eq!(follow_up.len(), 3);
        assert!(matches!(
            follow_up.get(2),
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
        let script = script(vec![
            vec![
                text_item("checking "),
                reasoning_block(Some("rs_1"), "reasoned step"),
                call_item("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
                final_tokens(4),
            ],
            vec![text_item("continued"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(skip_default_api_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;
        assert!(collected.error.is_none(), "{:?}", collected.error);

        let requests = script.requests();
        assert_eq!(requests.len(), 2);
        let follow_up = history_of(&requests[1]);
        assert!(history_contains_text(&follow_up, "checking "));
        assert!(assistant_reasoning_precedes_tool_call(
            &follow_up,
            "reasoned step",
            "default_api"
        ));
        assert!(
            assistant_reasoning_precedes_text_and_tool_call(
                &follow_up,
                "reasoned step",
                "checking ",
                "default_api"
            ),
            "{follow_up:?}"
        );
    }

    #[tokio::test]
    async fn invalid_name_delta_retry_preserves_streaming_reasoning_history() {
        let script = script(vec![
            vec![
                reasoning_delta_item(Some("rs_1"), "delta reason"),
                args_delta("tool_call_1", "internal_1", r#"{"x":2,"y":3}"#),
                name_delta("tool_call_1", "internal_1", "default_api"),
                final_tokens(4),
            ],
            vec![text_item("retried"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(retry_default_api_entry())
                .max_turns(3)
                .max_invalid_tool_call_retries(1)
                .stream_run(),
        )
        .await;
        assert!(collected.error.is_none(), "{:?}", collected.error);

        let requests = script.requests();
        assert_eq!(requests.len(), 2);
        assert!(assistant_reasoning_precedes_tool_call(
            &history_of(&requests[1]),
            "delta reason",
            "default_api"
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_skip_resets_streaming_text_delta_state() {
        let text_hook = RecordedTextDeltas::default();
        let script = script(vec![
            vec![
                text_item("stale "),
                call_item("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
                final_tokens(4),
            ],
            vec![text_item("fresh"), final_tokens(6)],
        ]);
        let agent = agent_builder(script).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(text_hook.entry())
                .add_hook(skip_default_api_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;
        assert!(collected.error.is_none(), "{:?}", collected.error);

        assert_eq!(
            text_hook.observed(),
            vec![
                ("stale ".to_string(), "stale ".to_string()),
                ("fresh".to_string(), "fresh".to_string()),
            ],
            "the abandoned turn's aggregated text must not leak into the retry"
        );
    }

    #[tokio::test]
    async fn invalid_tool_call_delta_retry_uses_structured_tool_feedback() {
        let delta_hook = RecordedToolCallDeltas::default();
        let add_calls = Arc::new(AtomicU32::new(0));
        let script = script(vec![
            vec![
                text_item("checking "),
                reasoning_delta_item(Some("rs_1"), "diagnostic reason"),
                call_item_with_call_id("tool_call_0", "call_0", "add", json!({"x": 1, "y": 2})),
                args_delta("tool_call_1", "internal_1", r#"{"x":2,"y":3}"#),
                name_delta("tool_call_1", "internal_1", "default_api"),
                final_tokens(4),
            ],
            vec![text_item("retried"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(delta_hook.entry())
                .add_hook(retry_default_api_entry())
                .max_turns(3)
                .max_invalid_tool_call_retries(1)
                .stream_run(),
        )
        .await;

        let final_response = collected.expect_final();
        assert_eq!(final_response.output, "retried");
        assert!(
            collected.deltas().is_empty(),
            "an invalid tool-call delta must never be emitted"
        );
        assert!(delta_hook.observed().is_empty());
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);
        let expected_calls = vec![
            CompletionCall::new(0, total_usage(4)),
            CompletionCall::new(1, total_usage(6)),
        ];
        assert_eq!(collected.completion_calls(), expected_calls);
        assert_eq!(final_response.completion_calls, expected_calls);
        assert_eq!(final_response.usage.total_tokens, 10);

        let requests = script.requests();
        assert_eq!(requests.len(), 2);
        let retry_history = history_of(&requests[1]);
        assert!(matches!(
            retry_history.get(1),
            Some(Message::Assistant { content, .. })
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text == "checking "
                ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(call)
                            if call.id == "tool_call_0" && call.function.name == "add"
                    ))
                    && content.iter().any(|item| matches!(
                        item,
                        AssistantContent::ToolCall(call)
                            if call.id == "tool_call_1"
                                && call.function.name == "default_api"
                                && call.function.arguments == json!({"x": 2, "y": 3})
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
        let script = script(vec![vec![
            text_item("checking "),
            reasoning_delta_item(Some("rs_1"), "diagnostic reason"),
            call_item_with_call_id("tool_call_0", "call_0", "add", json!({"x": 1, "y": 2})),
            args_delta("tool_call_1", "internal_1", r#"{"x":2,"y":3}"#),
            name_delta("tool_call_1", "internal_1", "default_api"),
            final_tokens(4),
        ]]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(invalid_hook.entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        collected.expect_error();
        assert_eq!(script.calls(), 1);
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
        let script = script(vec![
            vec![
                text_item("stale "),
                args_delta("tool_call_1", "internal_1", r#"{"x":2,"y":3}"#),
                name_delta("tool_call_1", "internal_1", "default_api"),
                final_tokens(4),
            ],
            vec![text_item("fresh"), final_tokens(6)],
        ]);
        let agent = agent_builder(script).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(text_hook.entry())
                .add_hook(retry_default_api_entry())
                .max_turns(3)
                .max_invalid_tool_call_retries(1)
                .stream_run(),
        )
        .await;
        assert!(collected.error.is_none(), "{:?}", collected.error);

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
        let script = script(vec![
            vec![
                text_item("checking "),
                call_item_with_call_id("tool_call_0", "call_0", "add", json!({"x": 1, "y": 2})),
                args_delta("tool_call_1", "internal_1", r#"{"x":2,"y":3}"#),
                name_delta("tool_call_1", "internal_1", "default_api"),
                final_tokens(4),
            ],
            vec![text_item("continued"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(delta_hook.entry())
                .add_hook(skip_default_api_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert!(
            collected.deltas().is_empty(),
            "an invalid tool-call delta must never be emitted"
        );
        let (skipped, internal_call_id) = collected
            .tool_results()
            .into_iter()
            .next()
            .expect("skip recovery should emit a synthetic tool result");
        assert_eq!(internal_call_id, "internal_1");
        assert_eq!(skipped.id, "tool_call_1");
        assert!(skipped.call_id.is_none());
        assert!(skipped.content.iter().any(|content| matches!(
            content,
            ToolResultContent::Text(text) if text.text == "default_api was skipped"
        )));
        assert_eq!(collected.expect_final().output, "continued");
        assert!(delta_hook.observed().is_empty());
        assert_eq!(add_calls.load(Ordering::SeqCst), 0);

        let requests = script.requests();
        assert_eq!(requests.len(), 2);
        let follow_up = history_of(&requests[1]);
        assert!(matches!(
            follow_up.get(2),
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
        let script = script(vec![
            vec![
                call_item("tool_call_1", "default_api", json!({"x": 1, "y": 2})),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(retry_default_api_entry())
                .max_turns(3)
                .max_invalid_tool_call_retries(0)
                .stream_run(),
        )
        .await;

        let (tool_name, _, _, history) = expect_unknown_tool_call(collected.expect_error());
        assert_eq!(tool_name, "default_api");
        assert!(history_contains_tool_call(history, "default_api"));
        assert_eq!(script.calls(), 1);
    }

    #[tokio::test]
    async fn streaming_name_delta_retry_budget_exhaustion_history_includes_same_turn_context() {
        let script = script(vec![
            vec![
                text_item("checking "),
                call_item_with_call_id("tool_call_0", "call_0", "add", json!({"x": 1, "y": 2})),
                args_delta("tool_call_1", "internal_1", r#"{"x":2,"y":3}"#),
                name_delta("tool_call_1", "internal_1", "default_api"),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("use the tool")
                .add_hook(retry_default_api_entry())
                .max_turns(3)
                .max_invalid_tool_call_retries(0)
                .stream_run(),
        )
        .await;

        let (tool_name, _, _, history) = expect_unknown_tool_call(collected.expect_error());
        assert_eq!(tool_name, "default_api");
        assert!(history_contains_text(history, "checking "));
        assert!(history_contains_tool_call(history, "add"));
        assert!(history_contains_tool_call(history, "default_api"));
        assert_eq!(script.calls(), 1);
    }

    // ---------- tool-call delta emission and gating ----------

    #[tokio::test]
    async fn tool_choice_none_rejects_streaming_tool_call_name_delta_before_hook_or_emit() {
        let script = script(vec![
            vec![
                name_delta("tool_1", "internal_1", "add"),
                args_delta("tool_1", "internal_1", "{\"x\":1}"),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let collected = collect(
            agent
                .runner("do not use tools")
                .add_hook(panic_on_unknown_tool_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert!(collected.deltas().is_empty());
        let (tool_name, available, allowed, history) =
            expect_unknown_tool_call(collected.expect_error());
        assert_eq!(tool_name, "add");
        assert_eq!(available, ["add".to_string()]);
        assert!(allowed.is_empty());
        assert!(history_contains_tool_call(history, "add"));
        assert_eq!(script.calls(), 1);
    }

    #[tokio::test]
    async fn unknown_tool_call_name_delta_fails_before_streaming_delta_hook_or_emit() {
        let script = script(vec![
            vec![
                name_delta("tool_1", "internal_1", "default_api"),
                args_delta("tool_1", "internal_1", "{\"x\":1}"),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("stream a bad tool call")
                .add_hook(panic_on_unknown_tool_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert!(collected.deltas().is_empty());
        let (tool_name, available, allowed, history) =
            expect_unknown_tool_call(collected.expect_error());
        assert_eq!(tool_name, "default_api");
        assert_eq!(available, ["add".to_string()]);
        assert_eq!(allowed, ["add".to_string()]);
        assert!(history_contains_tool_call(history, "default_api"));
        assert_eq!(script.calls(), 1);
    }

    #[tokio::test]
    async fn tool_call_args_delta_before_unknown_name_fails_before_hook_or_emit() {
        let script = script(vec![
            vec![
                args_delta("tool_1", "internal_1", "{\"x\":1}"),
                name_delta("tool_1", "internal_1", "default_api"),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("stream a bad tool call")
                .add_hook(panic_on_unknown_tool_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert!(
            collected.deltas().is_empty(),
            "buffered arguments must not escape once the name is rejected"
        );
        let (tool_name, _, _, history) = expect_unknown_tool_call(collected.expect_error());
        assert_eq!(tool_name, "default_api");
        assert!(history_contains_tool_call(history, "default_api"));
        assert_eq!(script.calls(), 1);
    }

    /// Arguments streamed before the name are buffered, then replayed after
    /// the validated name — never before it.
    #[tokio::test]
    async fn tool_call_args_delta_before_valid_name_buffers_then_emits_in_safe_order() {
        let hook = RecordedToolCallDeltas::default();
        let script = script(vec![vec![
            args_delta("tool_1", "internal_1", "{\"x\":"),
            name_delta("tool_1", "internal_1", "add"),
            args_delta("tool_1", "internal_1", "1}"),
            final_tokens(3),
        ]]);
        let agent = agent_builder(script).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("stream a tool call")
                .add_hook(hook.entry())
                .stream_run(),
        )
        .await;

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
            collected.deltas(),
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
        let script = script(vec![
            vec![
                args_delta("tool_1", "internal_1", "{\"x\":1}"),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("stream an incomplete tool call")
                .add_hook(panic_on_unknown_tool_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert!(collected.deltas().is_empty());
        assert!(collected.final_response().is_none());
        let message = collected.expect_error().to_string();
        assert!(
            message.contains("streamed tool call arguments"),
            "{message}"
        );
        assert!(message.contains("tool_1"), "{message}");
        assert!(message.contains("internal_1"), "{message}");
        assert_eq!(script.calls(), 1);
    }

    #[tokio::test]
    async fn tool_choice_none_buffers_args_then_rejects_name_without_emit() {
        let script = script(vec![
            vec![
                args_delta("tool_1", "internal_1", "{\"x\":1}"),
                name_delta("tool_1", "internal_1", "add"),
                final_tokens(4),
            ],
            vec![text_item("should not be requested"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone())
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let collected = collect(
            agent
                .runner("do not use tools")
                .add_hook(panic_on_unknown_tool_entry())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert!(collected.deltas().is_empty());
        let (tool_name, available, allowed, history) =
            expect_unknown_tool_call(collected.expect_error());
        assert_eq!(tool_name, "add");
        assert_eq!(available, ["add".to_string()]);
        assert!(allowed.is_empty());
        assert!(history_contains_tool_call(history, "add"));
        assert_eq!(script.calls(), 1);
    }

    #[tokio::test]
    async fn stream_prompt_emits_tool_call_deltas_without_hook() {
        let script = script(vec![vec![
            name_delta("tool_1", "internal_1", "add"),
            args_delta("tool_1", "internal_1", "{\"x\":"),
            args_delta("tool_1", "internal_1", "1}"),
            final_tokens(3),
        ]]);
        let agent = agent_builder(script).tool(MockAddTool).build();

        let collected = collect(agent.stream_run("stream a tool call")).await;

        assert_eq!(
            collected.deltas(),
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
        let hook = RecordedToolCallDeltas::default();
        let script = script(vec![vec![
            name_delta("tool_1", "internal_1", "add"),
            args_delta("tool_1", "internal_1", "{\"x\":"),
            args_delta("tool_1", "internal_1", "1}"),
            final_tokens(3),
        ]]);
        let agent = agent_builder(script).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("stream a tool call")
                .add_hook(hook.entry())
                .stream_run(),
        )
        .await;

        assert_eq!(hook.observed().len(), 3);
        assert_eq!(collected.deltas().len(), 3);
        assert_eq!(
            collected.deltas().first().map(|(_, _, content)| content),
            Some(&ToolCallDeltaContent::Name("add".to_string()))
        );
    }

    /// A delta hook that stops the run prevents the delta it observed from
    /// ever reaching the consumer.
    #[tokio::test]
    async fn stream_prompt_tool_call_deltas_hook_termination_prevents_delta_emit() {
        let hook = RecordedToolCallDeltas::default();
        let script = script(vec![vec![
            name_delta("tool_1", "internal_1", "add"),
            args_delta("tool_1", "internal_1", "{\"x\":"),
            final_tokens(3),
        ]]);
        let agent = agent_builder(script).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("stream a tool call")
                .add_hook(hook.entry_with(ObservationAction::stop("stop on tool call delta")))
                .stream_run(),
        )
        .await;

        assert_eq!(
            hook.observed(),
            vec![(
                "tool_1".to_string(),
                "internal_1".to_string(),
                Some("add".to_string()),
                String::new()
            )]
        );
        assert!(collected.deltas().is_empty());
        assert!(collected.final_response().is_none());
        let message = collected.expect_error().to_string();
        assert!(
            message.contains("stop on tool call delta"),
            "expected hook termination error, got {message}"
        );
    }

    // ---------- per-call completion records ----------

    #[tokio::test]
    async fn stream_prompt_exposes_completion_calls() {
        let first = split_usage(10, 2);
        let second = split_usage(25, 5);
        let script = script(vec![
            vec![
                call_item_with_call_id("tool_call_1", "call_1", "add", json!({"x": 1, "y": 2})),
                final_usage(first),
            ],
            vec![text_item("done"), final_usage(second)],
        ]);
        let agent = agent_builder(script).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("do tool work")
                .history(Vec::<Message>::new())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        let expected = vec![
            CompletionCall::new(0, first),
            CompletionCall::new(1, second),
        ];
        assert_eq!(collected.completion_calls(), expected);
        let final_response = collected.expect_final();
        assert_eq!(final_response.usage, split_usage(35, 7));
        assert_eq!(final_response.completion_calls, expected);
    }

    /// The record for a turn is emitted before a stream-finish hook stops the
    /// run, so its usage is never lost.
    #[tokio::test]
    async fn stream_prompt_emits_completion_call_before_finish_hook_termination() {
        let call_usage = split_usage(10, 2);
        let script = script(vec![vec![text_item("done"), final_usage(call_usage)]]);
        let agent = agent_builder(script).build();

        let collected = collect(
            agent
                .runner("say done")
                .add_hook(terminate_on_stream_finish_entry())
                .stream_run(),
        )
        .await;

        assert_eq!(
            collected.completion_calls(),
            vec![CompletionCall::new(0, call_usage)]
        );
        assert!(collected.final_response().is_none());
        collected.expect_error();
    }

    /// A turn whose provider never reported usage still emits exactly one
    /// record, with the zero-valued sentinel.
    #[tokio::test]
    async fn stream_prompt_completion_calls_records_unreported_usage() {
        let second = split_usage(25, 5);
        let script = script(vec![
            vec![call_item_with_call_id(
                "tool_call_1",
                "call_1",
                "add",
                json!({"x": 1, "y": 2}),
            )],
            vec![text_item("done"), final_usage(second)],
        ]);
        let agent = agent_builder(script).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("do tool work")
                .history(Vec::<Message>::new())
                .max_turns(3)
                .stream_run(),
        )
        .await;

        let expected = vec![
            CompletionCall::new(0, Usage::new()),
            CompletionCall::new(1, second),
        ];
        assert_eq!(collected.completion_calls(), expected);
        assert_eq!(collected.expect_final().completion_calls, expected);
    }

    // ---------- final-response shaping and history parity ----------

    #[tokio::test]
    async fn final_response_matches_streamed_text_when_provider_final_is_textless() {
        let script = script(vec![vec![
            text_item("hello"),
            text_item(" world"),
            final_tokens(3),
        ]]);
        let agent = agent_builder(script).build();

        let collected = collect(agent.stream_run("say hello")).await;

        assert_eq!(collected.streamed_text(), "hello world");
        assert_eq!(collected.expect_final().output, "hello world");
    }

    #[tokio::test]
    async fn final_response_preserves_structured_text_metadata() {
        let script = script(vec![vec![
            cited_text_item("cited ", citation_metadata()),
            text_item("answer"),
            final_tokens(3),
        ]]);
        let agent = agent_builder(script).build();

        let collected = collect(agent.stream_run("answer with citations")).await;

        let final_response = collected.expect_final();
        assert_eq!(final_response.output, "cited answer");
        let metadata =
            text_metadata(&final_response.content).expect("text metadata in final content");
        assert_eq!(
            metadata["citations"][0]["encrypted_index"],
            "encrypted-reference"
        );
    }

    #[tokio::test]
    async fn final_response_history_preserves_structured_text_metadata() {
        let script = script(vec![vec![
            cited_text_item("cited ", citation_metadata()),
            text_item("answer"),
            final_tokens(3),
        ]]);
        let agent = agent_builder(script).build();

        let collected = collect(
            agent
                .runner("answer with citations")
                .history(Vec::<Message>::new())
                .stream_run(),
        )
        .await;

        let history = collected
            .expect_final()
            .messages
            .as_ref()
            .expect("final history")
            .clone();
        let assistant_content = history
            .iter()
            .find_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .expect("assistant message in history");
        let metadata =
            text_metadata(assistant_content).expect("text metadata in assistant history");
        assert_eq!(
            metadata["citations"][0]["encrypted_index"],
            "encrypted-reference"
        );
    }

    #[tokio::test]
    async fn tool_follow_up_history_preserves_structured_text_metadata() {
        let script = script(vec![
            vec![
                cited_text_item("I need a tool. ", citation_metadata()),
                call_item_with_call_id("tool_call_1", "call_1", "add", json!({"x": 1, "y": 2})),
                final_tokens(4),
            ],
            vec![text_item("done"), final_tokens(6)],
        ]);
        let agent = agent_builder(script.clone()).tool(MockAddTool).build();

        let collected = collect(
            agent
                .runner("use a tool with citations")
                .history(Vec::<Message>::new())
                .max_turns(3)
                .stream_run(),
        )
        .await;
        assert!(collected.error.is_none(), "{:?}", collected.error);

        let requests = script.requests();
        assert_eq!(requests.len(), 2);
        let follow_up = history_of(&requests[1]);
        let assistant_content = follow_up
            .iter()
            .find_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .expect("assistant message in follow-up history");
        let metadata = text_metadata(assistant_content)
            .expect("citation metadata in follow-up assistant history");
        assert_eq!(
            metadata["citations"][0]["encrypted_index"],
            "encrypted-reference"
        );
    }

    /// A truly textless turn stays empty rather than inventing content.
    #[tokio::test]
    async fn final_response_can_remain_empty_for_truly_textless_turns() {
        let script = script(vec![vec![final_tokens(1)]]);
        let agent = agent_builder(script).build();

        let collected = collect(agent.stream_run("say nothing")).await;

        assert!(collected.streamed_text().is_empty());
        assert_eq!(collected.expect_final().output, "");
    }

    #[tokio::test]
    async fn streaming_final_response_carries_the_committed_transcript() {
        let script = script(vec![vec![
            text_item("hello"),
            text_item(" world"),
            final_tokens(3),
        ]]);
        let agent = agent_builder(script).build();

        let collected = collect(agent.stream_run("hi there")).await;

        let history = collected
            .expect_final()
            .messages
            .as_ref()
            .expect("PromptResponse.messages should be populated")
            .clone();
        assert_eq!(
            history.len(),
            2,
            "user prompt + assistant response in final history: {history:?}"
        );
    }

    #[tokio::test]
    async fn streaming_reasoning_without_tools_does_not_duplicate_final_history() {
        let script = script(vec![vec![
            text_item("final answer"),
            reasoning_block(Some("rs_1"), "reasoned step"),
            final_tokens(3),
        ]]);
        let agent = agent_builder(script).build();

        let collected = collect(
            agent
                .runner("think before answering")
                .history(Vec::<Message>::new())
                .stream_run(),
        )
        .await;

        let history = collected
            .expect_final()
            .messages
            .as_ref()
            .expect("PromptResponse.messages should be populated")
            .clone();
        assert_eq!(
            history.len(),
            2,
            "user prompt + one assistant response in final history: {history:?}"
        );
        assert!(matches!(
            history.first(),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::Text(text) if text.text == "think before answering"
                )
        ));

        let assistant_messages = history
            .iter()
            .filter_map(|message| match message {
                Message::Assistant { content, .. } => Some(content),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(assistant_messages.len(), 1, "{history:?}");
        let assistant_content = assistant_messages.first().expect("assistant message");
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
            .expect("reasoning in assistant history");
        let text_index = assistant_content
            .iter()
            .position(|item| matches!(item, AssistantContent::Text(_)))
            .expect("text in assistant history");
        assert!(
            reasoning_index < text_index,
            "assistant reasoning must be stored before assistant text: {assistant_content:?}"
        );
    }

    /// Tool-mode structured output: a turn mixing prose with the output-tool
    /// call finalizes as the structured payload, and leaves no unanswered
    /// `tool_use` in the committed history (#1928).
    #[tokio::test]
    async fn finalize_streamed_choice_surfaces_output_over_tool_call_and_prose() {
        let script = script(vec![vec![
            text_item("Sure, here is the weather:"),
            call_item("c1", "final_result", json!({"city": "Tokyo"})),
            final_tokens(3),
        ]]);
        let agent = agent_builder(script)
            .output_schema_raw(schemars::json_schema!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }))
            .output_mode(crate::agent::run::OutputMode::Tool)
            .build();

        let collected = collect(agent.stream_run("weather in Tokyo")).await;

        let final_response = collected.expect_final();
        assert_eq!(final_response.output, r#"{"city":"Tokyo"}"#);
        assert!(
            !final_response
                .content
                .iter()
                .any(|item| matches!(item, AssistantContent::ToolCall(_))),
            "no unanswered tool_use should remain in the final content"
        );
        let history = final_response
            .messages
            .as_ref()
            .expect("final history")
            .clone();
        assert!(
            history.iter().all(|message| match message {
                Message::Assistant { content, .. } => !content
                    .iter()
                    .any(|item| matches!(item, AssistantContent::ToolCall(_))),
                _ => true,
            }),
            "the output-tool call must not survive as an orphan tool_use: {history:?}"
        );
    }

    // ---------- telemetry ----------

    #[derive(Clone, Debug, Default)]
    struct CapturedSpan {
        id: u64,
        name: String,
        parent_id: Option<u64>,
        fields: std::collections::HashMap<String, u64>,
        string_fields: std::collections::HashMap<String, String>,
        record_counts: std::collections::HashMap<String, usize>,
    }

    #[derive(Clone, Default)]
    struct CapturedSpans(Arc<Mutex<Vec<CapturedSpan>>>);

    impl CapturedSpans {
        fn clear(&self) {
            self.0.lock().expect("spans").clear();
        }

        fn insert(&self, id: &tracing::Id, name: &str, parent_id: Option<u64>) {
            self.0.lock().expect("spans").push(CapturedSpan {
                id: id.into_u64(),
                name: name.to_string(),
                parent_id,
                ..CapturedSpan::default()
            });
        }

        fn record(&self, id: &tracing::Id, fields: Vec<CapturedField>) {
            let mut spans = self.0.lock().expect("spans");
            if let Some(span) = spans.iter_mut().rev().find(|span| span.id == id.into_u64()) {
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

        fn record_strings(&self, id: &tracing::Id, fields: Vec<(String, String)>) {
            let mut spans = self.0.lock().expect("spans");
            if let Some(span) = spans.iter_mut().rev().find(|span| span.id == id.into_u64()) {
                span.string_fields.extend(fields);
            }
        }

        fn snapshot(&self) -> Vec<CapturedSpan> {
            self.0.lock().expect("spans").clone()
        }
    }

    enum CapturedField {
        Number(String, u64),
        Text(String, String),
    }

    struct SpanCaptureLayer {
        spans: CapturedSpans,
    }

    impl<S> tracing_subscriber::Layer<S> for SpanCaptureLayer
    where
        S: tracing::Subscriber,
        S: for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>,
    {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            id: &tracing::Id,
            ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let parent_id = attrs
                .parent()
                .map(tracing::Id::into_u64)
                .or_else(|| ctx.current_span().id().map(tracing::Id::into_u64));
            self.spans.insert(id, attrs.metadata().name(), parent_id);
            let mut string_fields = Vec::new();
            attrs.record(&mut SpanStringCaptureVisitor {
                fields: &mut string_fields,
            });
            self.spans.record_strings(id, string_fields);
        }

        fn on_record(
            &self,
            span: &tracing::Id,
            values: &tracing::span::Record<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
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

    struct SpanFieldCaptureVisitor<'a> {
        fields: &'a mut Vec<CapturedField>,
    }

    struct SpanStringCaptureVisitor<'a> {
        fields: &'a mut Vec<(String, String)>,
    }

    impl tracing::field::Visit for SpanStringCaptureVisitor<'_> {
        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.fields
                .push((field.name().to_string(), value.to_string()));
        }

        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.fields
                .push((field.name().to_string(), format!("{value:?}")));
        }
    }

    impl tracing::field::Visit for SpanFieldCaptureVisitor<'_> {
        fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
            self.fields
                .push(CapturedField::Number(field.name().to_string(), value));
        }

        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.fields.push(CapturedField::Text(
                field.name().to_string(),
                value.to_string(),
            ));
        }

        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.fields.push(CapturedField::Text(
                field.name().to_string(),
                format!("{value:?}"),
            ));
        }
    }

    /// Install a span-capturing subscriber, warming the driver's span
    /// callsites against it first: the FIRST thread to hit a callsite caches
    /// its interest, so a parallel test without a subscriber can otherwise
    /// permanently cache `Interest::never` for the very spans asserted on.
    async fn warmed_span_capture() -> (CapturedSpans, tracing::subscriber::DefaultGuard) {
        use tracing_subscriber::layer::SubscriberExt;

        let spans = CapturedSpans::default();
        let subscriber = tracing_subscriber::Registry::default().with(SpanCaptureLayer {
            spans: spans.clone(),
        });
        let guard = tracing::subscriber::set_default(subscriber);

        let warmup =
            agent_builder(script(vec![vec![text_item("warmup"), final_tokens(0)]])).build();
        let collected = collect(warmup.stream_run("warmup")).await;
        assert!(collected.error.is_none());
        let unary = agent_builder(script(vec![vec![text_item("warmup"), final_tokens(0)]])).build();
        let _ = unary.runner("warmup").run().await;
        tracing::callsite::rebuild_interest_cache();
        spans.clear();
        (spans, guard)
    }

    async fn assert_stream_usage_recorded_on_chat_spans(
        agent: crate::agent::Agent,
        prompt: &str,
        max_turns: usize,
        expected_usages: &[Usage],
    ) {
        let (spans, _default) = warmed_span_capture().await;
        // Declare the field the outer-span guard protects, so a regression
        // (recording onto a caller span) is observable rather than a no-op.
        let outer_span = tracing::info_span!("outer", gen_ai.completion = tracing::field::Empty);

        async {
            let collected = collect(agent.runner(prompt).max_turns(max_turns).stream_run()).await;
            assert!(collected.error.is_none(), "{:?}", collected.error);
        }
        .instrument(outer_span)
        .await;

        let snapshot = spans.snapshot();
        let outer_id = snapshot
            .iter()
            .find(|span| span.name == "outer")
            .map(|span| span.id)
            .expect("outer span should be captured");
        let chat_spans = snapshot
            .iter()
            .filter(|span| span.name == "chat_streaming")
            .collect::<Vec<_>>();

        assert_eq!(chat_spans.len(), expected_usages.len());
        assert!(
            snapshot.iter().all(|span| span.name != "invoke_agent"),
            "outer span path should not create invoke_agent"
        );
        for (chat_span, expected) in chat_spans.into_iter().zip(expected_usages) {
            assert_eq!(chat_span.parent_id, Some(outer_id));
            assert_eq!(
                chat_span
                    .string_fields
                    .get("gen_ai.operation.name")
                    .map(String::as_str),
                Some("chat")
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.input_tokens"),
                Some(&expected.input_tokens)
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.output_tokens"),
                Some(&expected.output_tokens)
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.cache_read.input_tokens"),
                Some(&expected.cached_input_tokens)
            );
            assert_eq!(
                chat_span
                    .fields
                    .get("gen_ai.usage.cache_creation.input_tokens"),
                Some(&expected.cache_creation_input_tokens)
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.tool_use_prompt_tokens"),
                Some(&expected.tool_use_prompt_tokens)
            );
            assert_eq!(
                chat_span.fields.get("gen_ai.usage.reasoning_tokens"),
                Some(&expected.reasoning_tokens)
            );
        }

        let outer = snapshot
            .iter()
            .find(|span| span.id == outer_id)
            .expect("outer span");
        assert!(
            outer
                .fields
                .keys()
                .all(|field| !field.starts_with("gen_ai.usage.")),
            "usage should not be recorded onto the caller's outer span"
        );
        assert!(
            !outer.fields.contains_key("gen_ai.completion"),
            "gen_ai.completion should not be recorded onto the caller's outer span"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stream_prompt_records_single_call_usage_on_chat_span_under_outer_span() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let call_usage = split_usage(10, 2);
        let agent = agent_builder(script(vec![vec![
            text_item("done"),
            final_usage(call_usage),
        ]]))
        .build();

        assert_stream_usage_recorded_on_chat_spans(agent, "say done", 1, &[call_usage]).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stream_prompt_records_multi_turn_usage_on_chat_spans_under_outer_span() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let first = split_usage(10, 2);
        let second = split_usage(25, 5);
        let agent = agent_builder(script(vec![
            vec![
                call_item_with_call_id("tool_call_1", "call_1", "add", json!({"x": 1, "y": 2})),
                final_usage(first),
            ],
            vec![text_item("done"), final_usage(second)],
        ]))
        .tool(MockAddTool)
        .build();

        assert_stream_usage_recorded_on_chat_spans(agent, "do tool work", 3, &[first, second])
            .await;
    }

    async fn capture_stream_message_telemetry(record_telemetry_content: bool) -> Vec<CapturedSpan> {
        let (spans, _default) = warmed_span_capture().await;
        let builder = agent_builder(script(vec![vec![
            text_item("stream response secret"),
            final_tokens(0),
        ]]))
        .context("static stream context secret");
        let agent = if record_telemetry_content {
            builder.record_content_telemetry(true).build()
        } else {
            builder.build()
        };

        let collected = collect(agent.stream_run("stream prompt secret")).await;
        assert!(collected.error.is_none(), "{:?}", collected.error);
        spans.snapshot()
    }

    /// Streaming content telemetry is opt-in: the default run records no
    /// message contents, and the opt-in run records the turn's input and
    /// accepted output exactly once on its own `chat_streaming` span.
    #[tokio::test]
    async fn stream_prompt_message_telemetry_is_opt_in() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;

        let default_spans = capture_stream_message_telemetry(false).await;
        let default_chat = default_spans
            .iter()
            .find(|span| span.name == "chat_streaming")
            .expect("chat_streaming span");
        assert!(
            !default_chat.fields.contains_key("gen_ai.input.messages"),
            "default streaming prompt should not record input message contents"
        );
        assert!(
            !default_chat.fields.contains_key("gen_ai.output.messages"),
            "default streaming prompt should not record output message contents"
        );
        assert!(
            default_spans
                .iter()
                .filter(|span| span.name == "invoke_agent")
                .all(|span| !span.string_fields.contains_key("gen_ai.prompt")),
            "default streaming prompt should not record the prompt"
        );

        let opt_in_spans = capture_stream_message_telemetry(true).await;
        let opt_in_chat = opt_in_spans
            .iter()
            .find(|span| span.name == "chat_streaming")
            .expect("chat_streaming span");
        let input = opt_in_chat
            .string_fields
            .get("gen_ai.input.messages")
            .expect("opt-in should record input messages");
        assert!(input.contains("stream prompt secret"), "{input}");
        assert!(input.contains("static stream context secret"), "{input}");
        let output = opt_in_chat
            .string_fields
            .get("gen_ai.output.messages")
            .expect("opt-in should record output messages");
        assert!(output.contains("stream response secret"), "{output}");
        assert_eq!(
            opt_in_chat
                .record_counts
                .get("gen_ai.input.messages")
                .copied(),
            Some(1),
            "input message telemetry should be recorded once"
        );
        assert_eq!(
            opt_in_chat
                .record_counts
                .get("gen_ai.output.messages")
                .copied(),
            Some(1),
            "output message telemetry should be recorded once"
        );
        assert!(
            opt_in_spans
                .iter()
                .filter(|span| span.name == "invoke_agent")
                .any(|span| span
                    .string_fields
                    .get("gen_ai.prompt")
                    .is_some_and(|prompt| prompt.contains("stream prompt secret"))),
            "opt-in should record the run-level prompt"
        );
    }

    /// A streamed turn rejected mid-stream (invalid tool call) records its
    /// input but never its provisional output.
    #[tokio::test]
    async fn streaming_rejected_message_telemetry_does_not_record_output() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let (spans, _default) = warmed_span_capture().await;

        let agent = agent_builder(script(vec![vec![
            text_item("rejected stream output secret"),
            call_item("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            final_tokens(0),
        ]]))
        .record_content_telemetry(true)
        .build();

        let collected = collect(agent.stream_run("stream rejection prompt")).await;
        let message = collected.expect_error().to_string();
        assert!(
            message.contains("default_api"),
            "expected invalid tool error, got {message}"
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

    /// Blocking-driver parity for the same toggle: content telemetry is
    /// opt-in on the `chat` span and the run-level `invoke_agent` span.
    #[tokio::test]
    async fn unary_prompt_message_telemetry_records_accepted_output_when_opted_in() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;

        async fn capture(record: bool) -> Vec<CapturedSpan> {
            let (spans, _default) = warmed_span_capture().await;
            let builder = agent_builder(MockScript::from_responses(vec![
                rig_core::completion::CompletionResponse::new(
                    OneOrMany::one(AssistantContent::text("blocking response secret")),
                    Usage::new(),
                    "mock",
                ),
            ]))
            .preamble("blocking system secret");
            let agent = if record {
                builder.record_content_telemetry(true).build()
            } else {
                builder.build()
            };
            agent
                .runner("blocking prompt secret")
                .run()
                .await
                .expect("prompt should not error");
            spans.snapshot()
        }

        let default_spans = capture(false).await;
        let default_chat = default_spans
            .iter()
            .find(|span| span.name == "chat")
            .expect("chat span");
        assert!(!default_chat.fields.contains_key("gen_ai.input.messages"));
        assert!(!default_chat.fields.contains_key("gen_ai.output.messages"));
        assert!(
            !default_chat
                .string_fields
                .contains_key("gen_ai.system_instructions")
        );
        let default_agent_span = default_spans
            .iter()
            .find(|span| span.name == "invoke_agent")
            .expect("invoke_agent span");
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

        let opt_in_spans = capture(true).await;
        let opt_in_chat = opt_in_spans
            .iter()
            .find(|span| span.name == "chat")
            .expect("chat span");
        assert!(
            opt_in_chat
                .string_fields
                .get("gen_ai.input.messages")
                .is_some_and(|input| input.contains("blocking prompt secret"))
        );
        assert!(
            opt_in_chat
                .string_fields
                .get("gen_ai.output.messages")
                .is_some_and(|output| output.contains("blocking response secret"))
        );
        assert_eq!(
            opt_in_chat
                .string_fields
                .get("gen_ai.system_instructions")
                .map(String::as_str),
            Some(r#"[{"type":"text","content":"blocking system secret"}]"#)
        );
        let opt_in_agent_span = opt_in_spans
            .iter()
            .find(|span| span.name == "invoke_agent")
            .expect("invoke_agent span");
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
    }

    /// A repaired tool call's telemetry carries the canonical name, never the
    /// rejected raw one.
    #[tokio::test]
    async fn unary_repaired_message_telemetry_records_canonical_output() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let (spans, _default) = warmed_span_capture().await;

        let script = MockScript::from_responses(vec![
            rig_core::completion::CompletionResponse::new(
                OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                    "tool_call_1".to_string(),
                    ToolFunction::new("default_api".to_string(), json!({"x": 2, "y": 3})),
                ))),
                Usage::new(),
                "mock",
            ),
            rig_core::completion::CompletionResponse::new(
                OneOrMany::one(AssistantContent::text("done")),
                Usage::new(),
                "mock",
            ),
        ]);
        let agent = agent_builder(script.clone())
            .record_content_telemetry(true)
            .tool(MockAddTool)
            .build();

        let response = agent
            .runner("repair tool call")
            .add_hook(repair_default_api_entry())
            .max_turns(3)
            .run()
            .await
            .expect("repaired tool call should complete");
        assert_eq!(response.output, "done");

        let output_messages: Vec<String> = spans
            .snapshot()
            .into_iter()
            .filter(|span| span.name == "chat")
            .filter_map(|span| span.string_fields.get("gen_ai.output.messages").cloned())
            .collect();
        // The accepted continuation is recorded; the repaired turn's own chat
        // span records no output in the data-oriented blocking driver (it
        // returns for the invalid-call resolution before recording), so the
        // surviving invariant asserted here is that no stale raw tool name
        // ever reaches telemetry.
        assert!(
            output_messages.iter().any(|output| output.contains("done")),
            "the accepted continuation's output should be recorded: {output_messages:?}"
        );
        assert!(
            !output_messages
                .iter()
                .any(|output| output.contains("default_api")),
            "repaired output telemetry must not serialize the stale raw tool name: {output_messages:?}"
        );
        assert_eq!(script.calls(), 2);
    }

    /// Tool spans always carry structural metadata; the argument/result
    /// payloads follow the content-telemetry toggle and stay unrecorded by
    /// default.
    #[tokio::test]
    async fn tool_arguments_and_results_follow_content_telemetry_toggle() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let (spans, _default) = warmed_span_capture().await;

        let script = MockScript::from_responses(vec![
            rig_core::completion::CompletionResponse::new(
                OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                    "secret-tool-call".to_string(),
                    ToolFunction::new("add".to_string(), json!({"x": 12345, "y": 67890})),
                ))),
                Usage::new(),
                "mock",
            ),
            rig_core::completion::CompletionResponse::new(
                OneOrMany::one(AssistantContent::text("done")),
                Usage::new(),
                "mock",
            ),
        ]);
        let agent = agent_builder(script).tool(MockAddTool).build();
        agent
            .runner("use the tool")
            .max_turns(2)
            .run()
            .await
            .expect("tool run should succeed");

        let tool_span = spans
            .snapshot()
            .into_iter()
            .find(|span| span.name == "execute_tool")
            .expect("execute_tool span should be captured");
        assert_eq!(
            tool_span
                .string_fields
                .get("gen_ai.tool.name")
                .map(String::as_str),
            Some("add"),
            "structural tool metadata should remain available"
        );
        assert_eq!(
            tool_span
                .string_fields
                .get("gen_ai.tool.call.id")
                .map(String::as_str),
            Some("secret-tool-call")
        );
        assert!(
            tool_span
                .string_fields
                .contains_key("gen_ai.tool.call.outcome")
        );
        // Sensitive payloads stay off the span unless a driver opts in; the
        // data-oriented executor never records them.
        assert!(
            !tool_span
                .string_fields
                .contains_key("gen_ai.tool.call.arguments")
        );
        assert!(
            !tool_span
                .string_fields
                .contains_key("gen_ai.tool.call.result")
        );
    }

    /// Span context must not leak into concurrent tasks: the driver uses
    /// `.instrument()` rather than entering spans across await points.
    #[tokio::test(flavor = "current_thread")]
    async fn test_span_context_isolation() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let leaks = Arc::new(AtomicU32::new(0));

        let background = {
            let stop = stop.clone();
            let leaks = leaks.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_millis(5));
                while !stop.load(Ordering::Relaxed) {
                    interval.tick().await;
                    let current = tracing::Span::current();
                    if !current.is_disabled() && !current.is_none() {
                        leaks.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        };

        let script = script(vec![vec![
            text_item("hello"),
            text_item(" world"),
            final_tokens(3),
        ]]);
        let agent = agent_builder(script).build();
        let collected = collect(agent.stream_run("say hello world")).await;
        assert_eq!(collected.expect_final().output, "hello world");

        stop.store(true, Ordering::Relaxed);
        background.await.expect("background task");
        assert_eq!(
            leaks.load(Ordering::Relaxed),
            0,
            "the driver's spans must not leak into concurrent tasks"
        );
    }
}
