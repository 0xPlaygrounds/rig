//! The blocking session driver: a concrete, callback-free agent loop.
//!
//! [`AgentSession`] pairs an [`AgentConfig`] with a [`ProviderConfig`] and
//! drives the sans-IO [`AgentRun`] protocol: [`AgentSession::advance`] runs
//! until the next event the [`SessionPolicy`] surfaces, and every decision
//! flows back in as a plain value through a decision inbox — a `match` in
//! the host's loop replaces callback registration.
//!
//! # What persists between events
//!
//! [`AgentRun`] is the serializable unit, not the session: serialize
//! [`AgentSession::run_state`], and rebuild a session around the
//! deserialized run with [`AgentSession::resume`]. Two things do not survive
//! that round trip, both by construction rather than oversight:
//!
//! - **In-flight host decisions.** Tool-call and tool-result gate state
//!   (arguments already rewritten, calls already skipped, results already
//!   decided) lives in the session, not the run. A session serialized
//!   mid-gate resumes with those gates empty; the run re-surfaces the same
//!   calls and the host decides again. Idempotent, but *not* a checkpoint —
//!   a host whose decisions have side effects should drive each gate to
//!   completion before serializing.
//! - **In-flight model calls.** [`SessionEvent::BeforeModelCall`] is not a
//!   durable suspension point: a run serialized there (or while any model
//!   call was in flight) is resumed by re-issuing the call from the pre-call
//!   state, so it is re-prepared rather than picked up mid-flight.
//!
//! A run suspended on an invalid tool-call decision *is* recovered — `resume`
//! re-derives it from the run and re-surfaces the event.
//!
//! ```no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use rig_agent::agent::AgentConfig;
//! use rig_agent::provider::{ProviderConfig, Runtime};
//! use rig_agent::session::AgentSession;
//! use std::sync::Arc;
//!
//! let rt = Arc::new(Runtime::new());
//! let config = AgentConfig::new().with_preamble("You are terse.");
//! let provider = ProviderConfig::OpenAi(
//!     rig_agent::core::providers::openai::functions::Config::new("gpt-4o"),
//! );
//! let done = AgentSession::new(config, provider, rt, "Hello!").run().await?;
//! println!("{}", done.output);
//! # Ok(())
//! # }
//! ```

use std::collections::VecDeque;
use std::sync::Arc;

use crate::agent::hook::{
    CompletionCallAction, InvalidToolCallAction, ModelTurnAction, RequestPatch, ToolCallAction,
    ToolResultAction,
};
use crate::agent::prepare::{ToolCatalog, prepare_request};
use crate::agent::response::tool_result_output;
use crate::agent::run::{
    AgentRun, AgentRunStep, DEFAULT_OUTPUT_RETRIES, ModelTurn, ModelTurnOutcome, PendingToolCall,
};
use crate::agent::telemetry::{
    SessionSpanParams, acquire_agent_span, new_session_chat_span, record_usage_on_span,
};
use crate::agent::{AgentConfig, InvalidToolCallContext, PromptResponse, UNKNOWN_AGENT_NAME};
use crate::completion::{Message, PromptError, Usage};
use crate::tool::ToolResult;
use rig_core::OneOrMany;
use rig_core::completion::CompletionResponse;
use rig_core::message::{AssistantContent, ToolCall, UserContent};

use crate::provider::{self, ProviderConfig, Runtime};
use tracing_futures::Instrument;

/// Which decision points [`AgentSession::advance`] surfaces to the host.
///
/// Invalid tool calls and tool execution are always surfaced (the host owns
/// tool behavior); request patching and turn acceptance are opt-in.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionPolicy {
    /// Surface [`SessionEvent::TurnFinished`] instead of auto-accepting
    /// model turns.
    pub surface_model_turns: bool,
    /// Surface [`SessionEvent::BeforeModelCall`] instead of auto-sending.
    pub surface_completion_calls: bool,
    /// Surface [`SessionEvent::ToolCallPending`] for each executable call
    /// before the batch is handed over via
    /// [`SessionEvent::ToolCallsReady`].
    #[serde(default)]
    pub surface_tool_calls: bool,
    /// Surface [`SessionEvent::ToolResultReady`] for each provided result
    /// before the batch is committed to the run.
    #[serde(default)]
    pub surface_tool_results: bool,
}

/// What a session needs from its host next.
///
/// Deliberately exhaustive, like
/// [`AgentRunStep`]: a new
/// decision-bearing variant must fail to compile in every host.
#[derive(Debug)]
pub enum SessionEvent {
    /// `policy.surface_completion_calls`: the turn about to be prepared —
    /// surfaced pre-build so a patch's `active_tools`/`history` flow through
    /// request preparation. Answer via
    /// [`AgentSession::reply_before_call`].
    BeforeModelCall {
        /// This turn's prompt message.
        prompt: Message,
        /// The history preceding it.
        history: Vec<Message>,
        /// One-based model-call index.
        turn: usize,
    },
    /// `policy.surface_model_turns`: an accepted model turn awaiting the
    /// host's verdict. Answer via [`AgentSession::reply_turn`];
    /// [`ModelTurnAction::Retry`] is valid only for tool-free turns, exactly
    /// as [`AgentRun::retry_model_turn`]. The full provider response is
    /// available via [`AgentSession::last_response`].
    TurnFinished {
        /// One-based model-call index.
        turn: usize,
        /// Canonicalized assistant content parked for acceptance.
        content: OneOrMany<AssistantContent>,
        /// Usage reported for the turn.
        usage: Usage,
    },
    /// Always surfaced: the model called an unknown or disallowed tool.
    /// Answer via [`AgentSession::resolve_invalid`].
    InvalidToolCall(InvalidToolCallContext),
    /// `policy.surface_tool_calls`: one executable call awaiting its
    /// pre-execution decision. Answer via
    /// [`AgentSession::reply_tool_call`] with the same semantics the
    /// classic runner applies: `Run` executes as-is, `Rewrite` replaces the
    /// arguments, `Skip` pre-resolves the call as a skipped tool result
    /// (the tool body never runs), `Stop` cancels the run. Calls carrying
    /// a preresolved result pass through without surfacing.
    ToolCallPending {
        /// The call awaiting the decision.
        call: PendingToolCall,
    },
    /// Always surfaced: execute these calls and answer via
    /// [`AgentSession::provide_tool_results`].
    ToolCallsReady(Vec<PendingToolCall>),
    /// `policy.surface_tool_results`: one provided result awaiting its
    /// post-execution decision. Answer via
    /// [`AgentSession::reply_tool_result`]: `Keep` commits as provided,
    /// `Rewrite` replaces the model-visible presentation, `Stop` cancels
    /// the run. Results preresolved by invalid-call recovery pass through
    /// verbatim without surfacing; hook-skipped calls do surface, exactly
    /// as the classic runner fires its tool-result hook for skips.
    ToolResultReady {
        /// The executed (or skipped) tool call, with effective arguments.
        call: ToolCall,
        /// The model-visible result content as provided.
        result: UserContent,
    },
    /// The run is complete.
    Done(PromptResponse),
}

/// One paired call/result awaiting the result-gate decision.
#[derive(Debug)]
struct ResultGateEntry {
    /// The originating call for surfaced entries; `None` for results that
    /// pass through unsurfaced without a matching call.
    call: Option<ToolCall>,
    result: UserContent,
    /// Whether this entry surfaces as [`SessionEvent::ToolResultReady`].
    surface: bool,
}

/// The host decision the session is currently waiting for.
#[derive(Debug)]
// `RequestPatch` grew an output-schema payload; these are transient
// decision values, never stored in bulk, so the size skew is fine.
#[allow(clippy::large_enum_variant)]
enum Pending {
    None,
    BeforeCall {
        prompt: Message,
        history: Vec<Message>,
        /// Whether [`AgentSession::reply_before_call`] answered the event.
        /// Until it does, [`AgentSession::advance`] is a protocol violation
        /// (matching the [`AgentStream`](crate::stream::AgentStream)
        /// contract); once answered, the next advance resumes the call.
        answered: bool,
    },
    TurnReply,
    Invalid {
        /// A chained invalid call surfaced by the previous resolution,
        /// returned by the next [`AgentSession::advance`].
        next: Option<InvalidToolCallContext>,
    },
    /// `policy.surface_tool_calls`: pre-execution decisions in flight.
    ToolCallGate {
        /// Calls not yet decided, in call order.
        remaining: VecDeque<PendingToolCall>,
        /// Calls already decided (rewrites/skips applied).
        decided: Vec<PendingToolCall>,
        /// The call surfaced and awaiting
        /// [`AgentSession::reply_tool_call`].
        current: Option<PendingToolCall>,
        /// Ids skipped via [`ToolCallAction::Skip`] — these still surface
        /// their (synthetic) result under `surface_tool_results`.
        skipped_ids: Vec<String>,
    },
    Tools {
        /// The decided batch, kept for result pairing under
        /// `surface_tool_results` (empty otherwise).
        calls: Vec<PendingToolCall>,
        /// Ids skipped via [`ToolCallAction::Skip`].
        skipped_ids: Vec<String>,
    },
    /// `policy.surface_tool_results`: post-execution decisions in flight.
    ToolResultGate {
        /// Paired call/result entries in call order.
        entries: Vec<ResultGateEntry>,
        /// Index of the entry surfaced (when `awaiting`) or the next scan
        /// position.
        cursor: usize,
        /// Whether a [`SessionEvent::ToolResultReady`] event awaits
        /// [`AgentSession::reply_tool_result`].
        awaiting: bool,
    },
}

/// A concrete, callback-free driver for one agent run. See the
/// [module docs](self).
pub struct AgentSession {
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
    last_response: Option<CompletionResponse>,
    /// The run-level `invoke_agent` span (created, or adopted from the
    /// caller's ambient span), mirroring the classic drivers.
    agent_span: tracing::Span,
    /// Whether this session created `agent_span` — run-level usage is only
    /// recorded onto a span the session owns.
    created_agent_span: bool,
    /// Id of the previous `chat` span, chaining per-call spans via
    /// `follows_from` exactly like the classic blocking driver.
    chat_chain_head: u64,
    /// The in-flight tool batch's per-call `execute_tool` spans, held so a
    /// rewritten tool-result presentation is recorded onto the same span the
    /// classic driver recorded it on.
    batch_call_spans: Vec<(String, tracing::Span)>,
    /// The in-flight batch's **structured** execution results, keyed by
    /// provider tool-call id, so the post-execution decision point carries the
    /// classic classification (failed/skipped/denied, error kind, HTTP status)
    /// instead of the flattened model-visible content.
    batch_raw_results: Vec<(String, ToolResult)>,
    /// The in-flight turn's `chat` span, held so accepted-turn output
    /// telemetry lands on it once the host answers `TurnFinished` (a retried
    /// turn's provisional output is never recorded, matching the classic
    /// driver).
    turn_chat_span: Option<tracing::Span>,
}

impl AgentSession {
    /// Create a session for one prompt.
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
            last_response: None,
            agent_span,
            created_agent_span,
            chat_chain_head: 0,
            batch_call_spans: Vec::new(),
            batch_raw_results: Vec::new(),
            turn_chat_span: None,
        }
    }

    /// Resume a suspended run: pair a deserialized [`AgentRun`] with its
    /// configuration and a fresh runtime. The run re-emits its pending step
    /// idempotently ([`AgentRun::next_step`] semantics), so a process can
    /// serialize mid-tools and pick up where it left off.
    ///
    /// The new session starts with default `tools` and `policy` — reattach
    /// them with [`Self::with_tools`] / [`Self::with_policy`], since neither
    /// is carried by the run.
    ///
    /// Recovered from the run:
    ///
    /// - A run suspended on an invalid tool-call decision re-derives its
    ///   pending context; the next [`AgentSession::advance`] re-surfaces
    ///   [`SessionEvent::InvalidToolCall`] for
    ///   [`AgentSession::resolve_invalid`].
    /// - A run serialized while a model call was in flight (for example after
    ///   a transient provider error) recovers by re-issuing that call.
    ///
    /// **Not** recovered, because the run does not carry it: gate state for
    /// [`SessionEvent::ToolCallPending`] and [`SessionEvent::ToolResultReady`]
    /// (rewrites, skips, and results the host had already supplied), along
    /// with `last_response` and the in-flight telemetry spans. Those gates
    /// restart and re-surface their calls. See the [module docs](self).
    pub fn resume(
        config: AgentConfig,
        provider: ProviderConfig,
        rt: Arc<Runtime>,
        mut run: AgentRun,
    ) -> Self {
        let (agent_span, created_agent_span) = acquire_agent_span(
            config.name.as_deref().unwrap_or(UNKNOWN_AGENT_NAME),
            config.preamble.as_deref(),
            config.record_telemetry_content,
        );
        run.abandon_pending_model_call();
        let pending = match run.pending_invalid_tool_call() {
            Some(context) => Pending::Invalid {
                next: Some(context),
            },
            None => Pending::None,
        };
        Self {
            config,
            provider,
            tools: ToolCatalog::default(),
            policy: SessionPolicy::default(),
            run,
            rt,
            next_patch: None,
            pending,
            last_response: None,
            agent_span,
            created_agent_span,
            chat_chain_head: 0,
            batch_call_spans: Vec::new(),
            batch_raw_results: Vec::new(),
            turn_chat_span: None,
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

    /// The underlying run state (usage, turn count, messages so far).
    pub fn run_state(&self) -> &AgentRun {
        &self.run
    }

    /// The last provider response, for observation parity with the classic
    /// `on_completion_response` hook. Cleared when invalid-call recovery
    /// suppresses the response event.
    pub fn last_response(&self) -> Option<&CompletionResponse> {
        self.last_response.as_ref()
    }

    /// Merge a per-turn request patch consumed by the next model call.
    pub fn patch_next_turn(&mut self, patch: RequestPatch) {
        self.next_patch = Some(match self.next_patch.take() {
            Some(existing) => existing.merge(patch),
            None => patch,
        });
    }

    /// Drive the run until the next event the policy surfaces. Performs
    /// model IO; never executes tools.
    ///
    /// # Errors
    /// Provider/protocol errors, plus a protocol violation when a decision
    /// inbox is still awaiting its answer.
    pub async fn advance(&mut self) -> Result<SessionEvent, PromptError> {
        // A previously chained invalid call surfaces first.
        if let Pending::Invalid { next } = &mut self.pending {
            if let Some(context) = next.take() {
                return Ok(SessionEvent::InvalidToolCall(context));
            }
            return Err(self
                .run
                .cancel_error("advance called while an invalid tool call awaits resolution"));
        }
        match &self.pending {
            Pending::TurnReply => {
                return Err(self
                    .run
                    .cancel_error("advance called while a model turn awaits reply_turn"));
            }
            Pending::Tools { .. } => {
                return Err(self
                    .run
                    .cancel_error("advance called while tool results are awaited"));
            }
            Pending::BeforeCall {
                answered: false, ..
            } => {
                return Err(self.run.cancel_error(
                    "advance called while a BeforeModelCall event awaits reply_before_call",
                ));
            }
            Pending::ToolCallGate {
                current: Some(_), ..
            } => {
                return Err(self.run.cancel_error(
                    "advance called while a ToolCallPending event awaits reply_tool_call",
                ));
            }
            Pending::ToolResultGate { awaiting: true, .. } => {
                return Err(self.run.cancel_error(
                    "advance called while a ToolResultReady event awaits reply_tool_result",
                ));
            }
            Pending::None
            | Pending::BeforeCall { .. }
            | Pending::Invalid { .. }
            | Pending::ToolCallGate { .. }
            | Pending::ToolResultGate { .. } => {}
        }

        // Pre-execution gate: surface the next undecided call, or the
        // decided batch once every call is resolved.
        if matches!(self.pending, Pending::ToolCallGate { .. }) {
            return self.next_tool_call_gate_event();
        }
        // Post-execution gate: surface the next result decision; once every
        // result is resolved the batch commits and the run continues below.
        if matches!(self.pending, Pending::ToolResultGate { .. })
            && let Some(event) = self.step_tool_result_gate()?
        {
            return Ok(event);
        }

        loop {
            // Resume a pre-build pause answered by reply_before_call.
            let (step, before_call_answered) =
                match std::mem::replace(&mut self.pending, Pending::None) {
                    Pending::BeforeCall {
                        prompt, history, ..
                    } => (
                        AgentRunStep::CallModel {
                            prompt,
                            history,
                            turn: self.run.turn(),
                        },
                        true,
                    ),
                    other => {
                        self.pending = other;
                        (self.run.next_step()?, false)
                    }
                };

            match step {
                AgentRunStep::CallModel {
                    prompt,
                    history,
                    turn,
                } => {
                    if self.policy.surface_completion_calls && !before_call_answered {
                        // Surface pre-build; reply_before_call resumes here.
                        self.pending = Pending::BeforeCall {
                            prompt: prompt.clone(),
                            history: history.clone(),
                            answered: false,
                        };
                        return Ok(SessionEvent::BeforeModelCall {
                            prompt,
                            history,
                            turn,
                        });
                    }

                    let patch = self.next_patch.take().unwrap_or_default();
                    let mut prepared = prepare_request(
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

                    let chat_span = self.next_chat_span(&prepared.request);
                    // Content telemetry is recorded onto the agent's own `chat`
                    // span and suppressed on the provider's, exactly as the
                    // classic blocking driver did.
                    if self.config.record_telemetry_content {
                        let input_messages = prepared.request.messages_for_telemetry();
                        rig_core::telemetry::record_model_input(&chat_span, &input_messages, true);
                        prepared.request.record_telemetry_content = false;
                    }
                    let response =
                        match provider::complete(&self.provider, &self.rt, prepared.request)
                            .instrument(chat_span.clone())
                            .await
                        {
                            Ok(response) => response,
                            Err(error) => {
                                // Transient provider failure: return to the
                                // pre-call state so a later advance() retries
                                // the call instead of wedging the run in
                                // AwaitingModel forever.
                                self.run.abandon_pending_model_call();
                                return Err(error.into());
                            }
                        };
                    let model_turn = ModelTurn::new(
                        response.message_id.clone(),
                        response.choice.clone(),
                        response.usage,
                        prepared.executable_tool_names,
                        prepared.allowed_tool_names,
                    );
                    let usage = response.usage;
                    self.last_response = Some(response);

                    match self.run.model_response(model_turn)? {
                        ModelTurnOutcome::Continue {
                            response_hook_suppressed,
                        } => {
                            if response_hook_suppressed {
                                self.last_response = None;
                            }
                            if self.policy.surface_model_turns
                                && let Some(content) = self.run.accepted_turn_choice()
                            {
                                // The verdict is the host's; record the turn's
                                // output telemetry once it answers (never for
                                // a retried turn).
                                self.turn_chat_span = Some(chat_span);
                                self.pending = Pending::TurnReply;
                                return Ok(SessionEvent::TurnFinished {
                                    turn,
                                    content,
                                    usage,
                                });
                            }
                            self.record_turn_output(&chat_span);
                        }
                        ModelTurnOutcome::NeedsResolution(context) => {
                            self.pending = Pending::Invalid { next: None };
                            return Ok(SessionEvent::InvalidToolCall(context));
                        }
                        ModelTurnOutcome::TurnRetried => {}
                    }
                }
                AgentRunStep::CallTools { calls } => {
                    if self.policy.surface_tool_calls {
                        self.pending = Pending::ToolCallGate {
                            remaining: calls.into(),
                            decided: Vec::new(),
                            current: None,
                            skipped_ids: Vec::new(),
                        };
                        return self.next_tool_call_gate_event();
                    }
                    let stored = if self.policy.surface_tool_results {
                        calls.clone()
                    } else {
                        Vec::new()
                    };
                    self.pending = Pending::Tools {
                        calls: stored,
                        skipped_ids: Vec::new(),
                    };
                    return Ok(SessionEvent::ToolCallsReady(calls));
                }
                AgentRunStep::Done(response) => {
                    // Run-level telemetry, mirroring the classic blocking
                    // driver: only onto a span this session created — never
                    // pollute a caller-supplied outer span.
                    if self.created_agent_span {
                        if self.config.record_telemetry_content {
                            self.agent_span
                                .record("gen_ai.completion", response.output.as_str());
                        }
                        record_usage_on_span(&self.agent_span, response.usage);
                    }
                    return Ok(SessionEvent::Done(response));
                }
            }
        }
    }

    /// Build this turn's `chat` span and chain it onto the previous one via
    /// `follows_from`, preserving the classic blocking driver's linear causal
    /// trace.
    fn next_chat_span(
        &mut self,
        request: &rig_core::completion::CompletionRequest,
    ) -> tracing::Span {
        let params = SessionSpanParams {
            agent_name: self.config.name.as_deref(),
        };
        let span = new_session_chat_span(&params, request);
        let span = match self.chat_chain_head {
            0 => span,
            id => span
                .follows_from(tracing::span::Id::from_u64(id))
                .to_owned(),
        };
        if let Some(id) = span.id() {
            self.chat_chain_head = id.into_u64();
        }
        span
    }

    /// Answer [`SessionEvent::BeforeModelCall`].
    ///
    /// # Errors
    /// [`CompletionCallAction::Stop`] cancels the run; calling without a
    /// pending pre-build event is a protocol violation.
    pub fn reply_before_call(&mut self, action: CompletionCallAction) -> Result<(), PromptError> {
        match &mut self.pending {
            Pending::BeforeCall { answered, .. } if !*answered => {
                *answered = true;
            }
            _ => {
                return Err(self
                    .run
                    .cancel_error("reply_before_call without a pending BeforeModelCall event"));
            }
        }
        match action {
            CompletionCallAction::Continue => Ok(()),
            CompletionCallAction::Patch(patch) => {
                self.patch_next_turn(patch);
                Ok(())
            }
            CompletionCallAction::Stop(reason) => {
                self.pending = Pending::None;
                Err(self.run.cancel_error(reason))
            }
        }
    }

    /// Answer [`SessionEvent::TurnFinished`].
    ///
    /// # Errors
    /// [`ModelTurnAction::Stop`] cancels the run; retrying a tool-bearing
    /// turn is rejected exactly as [`AgentRun::retry_model_turn`].
    pub fn reply_turn(&mut self, action: ModelTurnAction) -> Result<(), PromptError> {
        if !matches!(self.pending, Pending::TurnReply) {
            return Err(self
                .run
                .cancel_error("reply_turn without a pending TurnFinished event"));
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
            // A retried turn's content was provisional: its output telemetry
            // is suppressed, exactly as the classic driver suppressed it.
            ModelTurnAction::Retry(request) => self.run.retry_model_turn(request),
            ModelTurnAction::Stop(reason) => {
                if let Some(span) = &chat_span {
                    self.record_turn_output(span);
                }
                Err(self.run.cancel_error(reason))
            }
        }
    }

    /// Record the accepted turn's content telemetry onto its `chat` span,
    /// gated on `record_telemetry_content` like the classic driver.
    fn record_turn_output(&self, chat_span: &tracing::Span) {
        if self.config.record_telemetry_content
            && let Some(choice) = self.run.accepted_turn_choice()
        {
            rig_core::telemetry::record_model_output(chat_span, &choice, true);
        }
    }

    /// Answer [`SessionEvent::InvalidToolCall`].
    ///
    /// A chained invalid call in the same turn surfaces on the next
    /// [`AgentSession::advance`].
    pub fn resolve_invalid(&mut self, action: InvalidToolCallAction) -> Result<(), PromptError> {
        if !matches!(self.pending, Pending::Invalid { .. }) {
            return Err(self
                .run
                .cancel_error("resolve_invalid without a pending InvalidToolCall event"));
        }
        match self.run.resolve_invalid_tool_call(action)? {
            ModelTurnOutcome::Continue {
                response_hook_suppressed,
            } => {
                if response_hook_suppressed {
                    self.last_response = None;
                }
                self.pending = Pending::None;
                Ok(())
            }
            ModelTurnOutcome::NeedsResolution(context) => {
                self.pending = Pending::Invalid {
                    next: Some(context),
                };
                Ok(())
            }
            ModelTurnOutcome::TurnRetried => {
                self.pending = Pending::None;
                Ok(())
            }
        }
    }

    /// Surface the next event of an in-flight pre-execution gate: the next
    /// undecided call, or [`SessionEvent::ToolCallsReady`] with the decided
    /// batch. Calls carrying a preresolved result pass through undecided,
    /// exactly as the classic runner returns them verbatim without firing
    /// tool hooks.
    fn next_tool_call_gate_event(&mut self) -> Result<SessionEvent, PromptError> {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::ToolCallGate {
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
                        remaining,
                        decided,
                        current: Some(next),
                        skipped_ids,
                    };
                    return Ok(SessionEvent::ToolCallPending { call });
                }
                let stored = if self.policy.surface_tool_results {
                    decided.clone()
                } else {
                    Vec::new()
                };
                self.pending = Pending::Tools {
                    calls: stored,
                    skipped_ids,
                };
                Ok(SessionEvent::ToolCallsReady(decided))
            }
            other => {
                self.pending = other;
                Err(self
                    .run
                    .cancel_error("tool-call gate advanced without pending decisions"))
            }
        }
    }

    /// Answer [`SessionEvent::ToolCallPending`], mirroring the classic
    /// runner's pre-execution semantics: `Run` keeps the call as-is,
    /// `Rewrite` replaces the arguments the host executes with, `Skip`
    /// pre-resolves the call as a skipped tool result (the body never
    /// runs), `Stop` cancels the run.
    ///
    /// # Errors
    /// [`ToolCallAction::Stop`] cancels the run; calling without a pending
    /// [`SessionEvent::ToolCallPending`] event is a protocol violation.
    pub fn reply_tool_call(&mut self, action: ToolCallAction) -> Result<(), PromptError> {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::ToolCallGate {
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
                        let skipped = ToolResult::skipped(reason);
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
                    .cancel_error("reply_tool_call without a pending ToolCallPending event"))
            }
        }
    }

    /// Step the post-execution gate: surface the next result decision, or
    /// commit the batch (returning `None` so the caller continues the run).
    fn step_tool_result_gate(&mut self) -> Result<Option<SessionEvent>, PromptError> {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::ToolResultGate {
                entries,
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
                            .call
                            .as_ref()
                            .map(|call| (index, call.clone(), entry.result.clone()))
                    })
                {
                    self.pending = Pending::ToolResultGate {
                        entries,
                        cursor: index,
                        awaiting: true,
                    };
                    return Ok(Some(SessionEvent::ToolResultReady { call, result }));
                }
                let results = entries.into_iter().map(|entry| entry.result).collect();
                self.run.tool_results(results)?;
                Ok(None)
            }
            other => {
                self.pending = other;
                Err(self
                    .run
                    .cancel_error("tool-result gate advanced without pending decisions"))
            }
        }
    }

    /// Answer [`SessionEvent::ToolResultReady`], mirroring the classic
    /// runner's post-execution semantics: `Keep` commits the result as
    /// provided, `Rewrite` replaces the model-visible presentation (the
    /// host's raw result is discarded from the committed batch), `Stop`
    /// cancels the run.
    ///
    /// # Errors
    /// [`ToolResultAction::Stop`] cancels the run; calling without a
    /// pending [`SessionEvent::ToolResultReady`] event is a protocol
    /// violation.
    pub fn reply_tool_result(&mut self, action: ToolResultAction) -> Result<(), PromptError> {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::ToolResultGate {
                mut entries,
                cursor,
                awaiting: true,
            } => {
                match action {
                    ToolResultAction::Keep => {}
                    ToolResultAction::Rewrite(output) => {
                        if let Some(entry) = entries.get_mut(cursor)
                            && let Some(call) = entry.call.as_ref()
                        {
                            entry.result =
                                tool_result_output(call.id.clone(), call.call_id.clone(), output);
                        }
                    }
                    ToolResultAction::Stop(reason) => {
                        return Err(self.run.cancel_error(reason));
                    }
                }
                self.pending = Pending::ToolResultGate {
                    entries,
                    cursor: cursor + 1,
                    awaiting: false,
                };
                Ok(())
            }
            other => {
                self.pending = other;
                Err(self
                    .run
                    .cancel_error("reply_tool_result without a pending ToolResultReady event"))
            }
        }
    }

    /// Answer [`SessionEvent::ToolCallsReady`] with one result per pending
    /// call (any order). Under `policy.surface_tool_results` the batch is
    /// not committed yet: each result surfaces as
    /// [`SessionEvent::ToolResultReady`] on subsequent
    /// [`AgentSession::advance`] calls, and commits once every decision is
    /// answered.
    pub fn provide_tool_results(&mut self, results: Vec<UserContent>) -> Result<(), PromptError> {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::Tools { calls, skipped_ids } => {
                if !self.policy.surface_tool_results {
                    self.run.tool_results(results)?;
                    return Ok(());
                }
                // Pair results with calls by provider id as a multiset
                // (mirroring `AgentRun::tool_results` and the stream
                // driver): duplicate ids pair positionally.
                let mut remaining: Vec<Option<UserContent>> =
                    results.into_iter().map(Some).collect();
                let mut entries = Vec::new();
                for call in &calls {
                    let matched = remaining.iter_mut().find_map(|slot| {
                        let is_match = slot.as_ref().is_some_and(|content| {
                            matches!(
                                content,
                                UserContent::ToolResult(result) if result.id == call.tool_call.id
                            )
                        });
                        if is_match { slot.take() } else { None }
                    });
                    if let Some(result) = matched {
                        // Results preresolved by invalid-call recovery pass
                        // through verbatim (the classic driver fires no
                        // hooks for them); gate skips do surface, exactly
                        // as run_single_tool fires its tool-result hook
                        // for a hook skip.
                        let surface = call.preresolved_result.is_none()
                            || skipped_ids.contains(&call.tool_call.id);
                        entries.push(ResultGateEntry {
                            call: Some(call.tool_call.clone()),
                            result,
                            surface,
                        });
                    }
                }
                // Unmatched results pass through for the run to validate.
                entries.extend(
                    remaining
                        .into_iter()
                        .flatten()
                        .map(|result| ResultGateEntry {
                            call: None,
                            result,
                            surface: false,
                        }),
                );
                self.pending = Pending::ToolResultGate {
                    entries,
                    cursor: 0,
                    awaiting: false,
                };
                Ok(())
            }
            other => {
                self.pending = other;
                Err(self
                    .run
                    .cancel_error("provide_tool_results without a pending ToolCallsReady event"))
            }
        }
    }

    /// Drive to completion under the default policy.
    ///
    /// # Errors
    /// Fails up front when the catalog advertises executable tools — a
    /// `run()` caller has nowhere to answer
    /// [`SessionEvent::ToolCallsReady`]. Invalid tool calls preserve
    /// fail-fast semantics.
    pub async fn run(mut self) -> Result<PromptResponse, PromptError> {
        if !self.tools.executable.is_empty() {
            return Err(self.run.cancel_error(
                "AgentSession::run cannot execute tools; drive advance()/provide_tool_results \
                 in a loop instead",
            ));
        }
        loop {
            match self.advance().await? {
                SessionEvent::Done(response) => return Ok(response),
                SessionEvent::InvalidToolCall(_) => {
                    // Preserve the classic default: fail fast.
                    self.resolve_invalid(InvalidToolCallAction::fail())?;
                }
                SessionEvent::ToolCallsReady(_) => {
                    return Err(self
                        .run
                        .cancel_error("model called a tool but run() has no tool executor"));
                }
                SessionEvent::ToolCallPending { .. } => {
                    // Not surfaced under the default policy; pass through.
                    self.reply_tool_call(ToolCallAction::Run)?;
                }
                SessionEvent::ToolResultReady { .. } => {
                    // Not surfaced under the default policy; pass through.
                    self.reply_tool_result(ToolResultAction::Keep)?;
                }
                SessionEvent::BeforeModelCall { .. } | SessionEvent::TurnFinished { .. } => {
                    // Not surfaced under the default policy; continue if a
                    // custom policy was set anyway.
                    if matches!(self.pending, Pending::BeforeCall { .. }) {
                        self.reply_before_call(CompletionCallAction::Continue)?;
                    } else {
                        self.reply_turn(ModelTurnAction::Continue)?;
                    }
                }
            }
        }
    }

    /// Drive to completion with the classic runtime semantics: dispatch
    /// `hooks` at every surfaced decision point and answer every
    /// [`SessionEvent::ToolCallsReady`] batch through `executor`.
    ///
    /// This is the loop [`Agent::run`](crate::Agent::run) and
    /// [`SessionRunner::run`](crate::agent::SessionRunner::run) are built
    /// from; drive it directly when you want the classic behavior over a
    /// session you configured yourself. Pair it with a policy that surfaces
    /// the decision points the hooks care about (see
    /// [`SessionPolicy`]) — [`Agent`](crate::Agent) surfaces everything as
    /// soon as one hook is attached.
    ///
    /// # Errors
    /// Provider/protocol errors; a hook `Stop` cancels the run; an
    /// executable tool call with no executor attached fails the run.
    // A single lifetime for every borrow: an `async fn` whose future captures
    // several independent elided lifetimes is invariant over them, which makes
    // the future fail higher-ranked `Send` bounds in callers such as the
    // Discord integration's `async_trait` handlers.
    pub async fn drive<'a>(
        &'a mut self,
        hooks: &'a crate::hooks::Hooks,
        executor: Option<&'a crate::executor::ToolExecutor>,
    ) -> Result<PromptResponse, PromptError> {
        // The session surfaces the turn prompt on `BeforeModelCall` only; the
        // response observation carries it too (classic parity), so keep the
        // most recent one.
        let mut turn_prompt: Option<Message> = None;
        loop {
            match self.advance().await? {
                SessionEvent::BeforeModelCall {
                    turn,
                    prompt,
                    history,
                } => {
                    let action = hooks
                        .dispatch_completion_call(turn, &prompt, &history)
                        .await;
                    turn_prompt = Some(prompt);
                    self.reply_before_call(action)?;
                }
                SessionEvent::TurnFinished {
                    turn,
                    content,
                    usage,
                } => {
                    // Observation over the full provider response first
                    // (classic order: the response hook fires before the
                    // model-turn verdict).
                    let observed_prompt = turn_prompt.clone().unwrap_or_else(|| Message::from(""));
                    if let Some(response) = self.last_response().cloned()
                        && let crate::agent::ObservationAction::Stop(reason) = hooks
                            .dispatch_completion_response(turn, &observed_prompt, &response)
                            .await
                    {
                        self.reply_turn(ModelTurnAction::Stop(reason))?;
                        continue;
                    }
                    let action = hooks.dispatch_model_turn(turn, &content, usage).await;
                    self.reply_turn(action)?;
                }
                SessionEvent::InvalidToolCall(context) => {
                    let action = hooks
                        .dispatch_invalid_tool_call(&context)
                        .await
                        // Preserve the classic default: fail fast.
                        .unwrap_or_else(InvalidToolCallAction::fail);
                    self.resolve_invalid(action)?;
                }
                SessionEvent::ToolCallPending { call } => {
                    // The effective action already carries any chained
                    // rewrite; a terminal `Skip`'s salvaged rewrite cannot be
                    // applied through the single-action inbox and only
                    // affects reporting, never execution (the body does not
                    // run).
                    let internal_call_id = call
                        .internal_call_id
                        .clone()
                        .unwrap_or_else(|| call.tool_call.id.clone());
                    let (action, _salvaged) = hooks
                        .dispatch_tool_call(&call.tool_call, &internal_call_id)
                        .await;
                    self.reply_tool_call(action)?;
                }
                SessionEvent::ToolCallsReady(calls) => {
                    let results = match executor {
                        Some(executor) => {
                            // Chain the batch's tool spans onto this turn's
                            // chat span, and the next chat span onto the last
                            // tool span: the classic blocking driver's linear
                            // causal trace.
                            let chain = match self.chat_chain_head {
                                0 => None,
                                id => Some(id),
                            };
                            let batch = executor.execute_batch_following(&calls, chain).await;
                            if let Some(id) = batch.last_span_id {
                                self.chat_chain_head = id;
                            }
                            self.batch_call_spans = batch.call_spans;
                            self.batch_raw_results = batch.raw_results;
                            batch.results
                        }
                        None => preresolved_only_results(&calls)?,
                    };
                    self.provide_tool_results(results)?;
                }
                SessionEvent::ToolResultReady { call, result } => {
                    // The executor's structured result when the body ran (so
                    // `ToolResult::error()`/`is_skipped()`/`is_refused()`/
                    // `http_status()` read exactly as they did on the classic
                    // tool-result hook), falling back to reconstructing it
                    // from the committed content for host-provided or
                    // pre-resolved results.
                    let raw = self
                        .batch_raw_results
                        .iter()
                        .find(|(id, _)| *id == call.id)
                        .map(|(_, structured)| structured.clone())
                        .unwrap_or_else(|| raw_tool_result(&result));
                    let internal_call_id = call.id.clone();
                    let action = hooks
                        .dispatch_tool_result(&call, &internal_call_id, &raw)
                        .await;
                    // Result telemetry is recorded **once, post-hook** (the
                    // executor defers it for this driver), so a redaction
                    // rewrite is never preceded by the raw value on the same
                    // span — exactly the classic ordering.
                    if self.config.record_telemetry_content
                        && let Some((_, span)) =
                            self.batch_call_spans.iter().find(|(id, _)| *id == call.id)
                    {
                        let rendered = match &action {
                            ToolResultAction::Rewrite(output) => Some(output.render()),
                            ToolResultAction::Keep => Some(raw.output().render()),
                            // A stop cancels the run; the classic driver
                            // recorded no result for it.
                            ToolResultAction::Stop(_) => None,
                        };
                        if let Some(rendered) = rendered {
                            span.record("gen_ai.tool.call.result", rendered);
                        }
                    }
                    self.reply_tool_result(action)?;
                }
                SessionEvent::Done(response) => return Ok(response),
            }
        }
    }

    /// Drive to completion, answering every
    /// [`SessionEvent::ToolCallsReady`] batch through the executor
    /// (classic tool-loop semantics; see
    /// [`ToolExecutor`](crate::executor::ToolExecutor)). Unlike
    /// [`AgentSession::run`], an executable catalog is expected here —
    /// pair the session with [`ToolExecutor::catalog`](crate::executor::ToolExecutor::catalog)
    /// so the driver advertises exactly what the executor can run.
    ///
    /// Tool failures stay model-visible as failed tool results, exactly as
    /// the classic loop delivered them; invalid tool calls preserve
    /// fail-fast semantics.
    ///
    /// # Errors
    /// Provider/protocol errors, as [`AgentSession::advance`].
    pub async fn run_with_tools(
        &mut self,
        executor: &crate::executor::ToolExecutor,
    ) -> Result<PromptResponse, PromptError> {
        loop {
            match self.advance().await? {
                SessionEvent::Done(response) => return Ok(response),
                SessionEvent::InvalidToolCall(_) => {
                    // Preserve the classic default: fail fast.
                    self.resolve_invalid(InvalidToolCallAction::fail())?;
                }
                SessionEvent::ToolCallsReady(calls) => {
                    let batch = executor.execute_batch(&calls).await;
                    self.provide_tool_results(batch.results)?;
                }
                SessionEvent::ToolCallPending { .. } | SessionEvent::ToolResultReady { .. } => {
                    // Per-call gates are host decision points; this driver
                    // owns tool execution wholesale under the default policy.
                    return Err(self.run.cancel_error(
                        "run_with_tools drives the default policy; disable \
                         surface_tool_calls/surface_tool_results or drive advance() manually",
                    ));
                }
                SessionEvent::BeforeModelCall { .. } | SessionEvent::TurnFinished { .. } => {
                    // Not surfaced under the default policy; continue if a
                    // custom policy was set anyway.
                    if matches!(self.pending, Pending::BeforeCall { .. }) {
                        self.reply_before_call(CompletionCallAction::Continue)?;
                    } else {
                        self.reply_turn(ModelTurnAction::Continue)?;
                    }
                }
            }
        }
    }
}

/// Results for a batch containing only preresolved calls (hook skips,
/// invalid-call recovery). An executable call with no executor attached is a
/// host configuration error.
pub(crate) fn preresolved_only_results(
    calls: &[PendingToolCall],
) -> Result<Vec<UserContent>, PromptError> {
    let mut results = Vec::with_capacity(calls.len());
    for call in calls {
        match &call.preresolved_result {
            Some(content) => results.push(content.clone()),
            None => {
                return Err(PromptError::CompletionError(
                    rig_core::completion::CompletionError::ResponseError(
                        "model called a tool but this agent has no executor; attach one with \
                         with_executor"
                            .to_string(),
                    ),
                ));
            }
        }
    }
    Ok(results)
}

/// Reconstruct the raw-result view the tool-result hook receives from the
/// committed model-visible content: tool-result content blocks map back onto
/// a successful [`ToolResult`] verbatim; any other content is rendered as
/// text.
pub(crate) fn raw_tool_result(result: &UserContent) -> ToolResult {
    match result {
        UserContent::ToolResult(tool_result) => ToolResult::success(
            crate::tool::ToolOutput::content(tool_result.content.clone()),
        ),
        other => ToolResult::success(crate::tool::ToolOutput::text(
            serde_json::to_string(other).unwrap_or_default(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::prepare::ToolCatalog;
    use crate::provider::MockScript;
    use crate::tool::ToolOutput;
    use rig_core::completion::{FinishReason, ToolDefinition};
    use rig_core::message::ToolResultContent;

    fn usage(total: u64) -> Usage {
        let mut usage = Usage::new();
        usage.total_tokens = total;
        usage
    }

    fn text_response(text: &str) -> rig_core::completion::CompletionResponse {
        rig_core::completion::CompletionResponse::new(
            OneOrMany::one(AssistantContent::text(text)),
            usage(5),
            "mock",
        )
        .with_finish_reason(FinishReason::Stop)
    }

    fn tool_call_response(
        id: &str,
        name: &str,
        args: serde_json::Value,
    ) -> rig_core::completion::CompletionResponse {
        rig_core::completion::CompletionResponse::new(
            OneOrMany::one(AssistantContent::tool_call(id, name, args)),
            usage(3),
            "mock",
        )
        .with_finish_reason(FinishReason::ToolCalls)
    }

    fn adder_catalog() -> ToolCatalog {
        ToolCatalog::new(vec![ToolDefinition {
            name: "add".to_string(),
            description: "Adds two numbers".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }])
    }

    fn tool_session(script: MockScript, policy: SessionPolicy) -> AgentSession {
        let mut config = AgentConfig::new();
        config.max_turns = Some(3);
        AgentSession::new(
            config,
            ProviderConfig::Mock(script),
            Arc::new(Runtime::new()),
            "hello",
        )
        .with_tools(adder_catalog())
        .with_policy(policy)
    }

    fn tool_result_for(call: &ToolCall, content: &str) -> UserContent {
        UserContent::tool_result(
            call.id.clone(),
            OneOrMany::one(ToolResultContent::text(content)),
        )
    }

    fn tool_loop_script() -> MockScript {
        MockScript::from_responses(vec![
            tool_call_response("call_1", "add", serde_json::json!({"a": 1, "b": 2})),
            text_response("done"),
        ])
    }

    #[tokio::test]
    async fn surfaced_tool_call_approved_runs_unchanged() {
        let policy = SessionPolicy {
            surface_tool_calls: true,
            ..SessionPolicy::default()
        };
        let mut session = tool_session(tool_loop_script(), policy);

        let call = match session.advance().await.expect("advance") {
            SessionEvent::ToolCallPending { call } => call,
            other => panic!("expected ToolCallPending, got {other:?}"),
        };
        assert_eq!(call.tool_call.function.name, "add");
        session
            .reply_tool_call(ToolCallAction::run())
            .expect("reply");

        let calls = match session.advance().await.expect("advance") {
            SessionEvent::ToolCallsReady(calls) => calls,
            other => panic!("expected ToolCallsReady, got {other:?}"),
        };
        assert_eq!(calls.len(), 1);
        let ready = calls.first().expect("one call");
        assert_eq!(
            ready.tool_call.function.arguments,
            serde_json::json!({"a": 1, "b": 2})
        );
        assert!(ready.preresolved_result.is_none());

        let results = vec![tool_result_for(&ready.tool_call, "3")];
        session.provide_tool_results(results).expect("results");
        match session.advance().await.expect("advance") {
            SessionEvent::Done(done) => assert_eq!(done.output, "done"),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn surfaced_tool_call_rewrite_replaces_arguments() {
        let policy = SessionPolicy {
            surface_tool_calls: true,
            ..SessionPolicy::default()
        };
        let mut session = tool_session(tool_loop_script(), policy);

        match session.advance().await.expect("advance") {
            SessionEvent::ToolCallPending { .. } => {}
            other => panic!("expected ToolCallPending, got {other:?}"),
        }
        session
            .reply_tool_call(ToolCallAction::rewrite(serde_json::json!({"a": 7, "b": 8})))
            .expect("reply");

        let calls = match session.advance().await.expect("advance") {
            SessionEvent::ToolCallsReady(calls) => calls,
            other => panic!("expected ToolCallsReady, got {other:?}"),
        };
        let ready = calls.first().expect("one call");
        assert_eq!(
            ready.tool_call.function.arguments,
            serde_json::json!({"a": 7, "b": 8})
        );
    }

    #[tokio::test]
    async fn surfaced_tool_call_skip_preresolves_a_skipped_result() {
        let policy = SessionPolicy {
            surface_tool_calls: true,
            ..SessionPolicy::default()
        };
        let mut session = tool_session(tool_loop_script(), policy);

        match session.advance().await.expect("advance") {
            SessionEvent::ToolCallPending { .. } => {}
            other => panic!("expected ToolCallPending, got {other:?}"),
        }
        session
            .reply_tool_call(ToolCallAction::skip("blocked by policy"))
            .expect("reply");

        let calls = match session.advance().await.expect("advance") {
            SessionEvent::ToolCallsReady(calls) => calls,
            other => panic!("expected ToolCallsReady, got {other:?}"),
        };
        let ready = calls.first().expect("one call");
        let preresolved = ready
            .preresolved_result
            .clone()
            .expect("skip preresolves the result");
        let rendered = serde_json::to_string(&preresolved).expect("serialize");
        assert!(rendered.contains("blocked by policy"), "got {rendered}");

        // The host returns the preresolved content verbatim.
        session
            .provide_tool_results(vec![preresolved])
            .expect("results");
        match session.advance().await.expect("advance") {
            SessionEvent::Done(done) => assert_eq!(done.output, "done"),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn surfaced_tool_call_stop_cancels_the_run() {
        let policy = SessionPolicy {
            surface_tool_calls: true,
            ..SessionPolicy::default()
        };
        let mut session = tool_session(tool_loop_script(), policy);
        match session.advance().await.expect("advance") {
            SessionEvent::ToolCallPending { .. } => {}
            other => panic!("expected ToolCallPending, got {other:?}"),
        }
        let error = session
            .reply_tool_call(ToolCallAction::stop("halt"))
            .expect_err("stop cancels");
        assert!(error.to_string().contains("halt"), "got {error}");
    }

    #[tokio::test]
    async fn surfaced_tool_result_rewrite_replaces_committed_presentation() {
        let policy = SessionPolicy {
            surface_tool_results: true,
            ..SessionPolicy::default()
        };
        let script = tool_loop_script();
        let probe = script.clone();
        let mut session = tool_session(script, policy);

        let calls = match session.advance().await.expect("advance") {
            SessionEvent::ToolCallsReady(calls) => calls,
            other => panic!("expected ToolCallsReady, got {other:?}"),
        };
        let ready = calls.first().expect("one call");
        session
            .provide_tool_results(vec![tool_result_for(&ready.tool_call, "raw secret")])
            .expect("results");

        let (call, result) = match session.advance().await.expect("advance") {
            SessionEvent::ToolResultReady { call, result } => (call, result),
            other => panic!("expected ToolResultReady, got {other:?}"),
        };
        assert_eq!(call.id, "call_1");
        let rendered = serde_json::to_string(&result).expect("serialize");
        assert!(rendered.contains("raw secret"), "got {rendered}");
        session
            .reply_tool_result(ToolResultAction::rewrite_output(ToolOutput::text(
                "redacted",
            )))
            .expect("reply");

        match session.advance().await.expect("advance") {
            SessionEvent::Done(done) => assert_eq!(done.output, "done"),
            other => panic!("expected Done, got {other:?}"),
        }
        // The committed history sent to the second model call carries the
        // rewritten presentation, not the raw result.
        let second = serde_json::to_string(&probe.requests().get(1).expect("two calls"))
            .expect("serialize request");
        assert!(second.contains("redacted"), "got {second}");
        assert!(!second.contains("raw secret"), "got {second}");
    }

    #[tokio::test]
    async fn surfaced_tool_result_keep_commits_as_provided() {
        let policy = SessionPolicy {
            surface_tool_results: true,
            ..SessionPolicy::default()
        };
        let mut session = tool_session(tool_loop_script(), policy);
        let calls = match session.advance().await.expect("advance") {
            SessionEvent::ToolCallsReady(calls) => calls,
            other => panic!("expected ToolCallsReady, got {other:?}"),
        };
        let ready = calls.first().expect("one call");
        session
            .provide_tool_results(vec![tool_result_for(&ready.tool_call, "3")])
            .expect("results");
        match session.advance().await.expect("advance") {
            SessionEvent::ToolResultReady { .. } => {}
            other => panic!("expected ToolResultReady, got {other:?}"),
        }
        session
            .reply_tool_result(ToolResultAction::keep())
            .expect("reply");
        match session.advance().await.expect("advance") {
            SessionEvent::Done(done) => assert_eq!(done.output, "done"),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn advance_while_gate_awaits_reply_is_a_protocol_violation() {
        let policy = SessionPolicy {
            surface_tool_calls: true,
            ..SessionPolicy::default()
        };
        let mut session = tool_session(tool_loop_script(), policy);
        match session.advance().await.expect("advance") {
            SessionEvent::ToolCallPending { .. } => {}
            other => panic!("expected ToolCallPending, got {other:?}"),
        }
        let error = session.advance().await.expect_err("awaiting reply");
        assert!(error.to_string().contains("reply_tool_call"), "got {error}");
    }
}

/// Blocking-driver behavior ported from the deleted classic `AgentRunner` /
/// `PromptRequest` corpus. These drive the public
/// [`SessionRunner`](crate::agent::SessionRunner) surface (and, where the
/// behavior is defined as run/stream parity, the streaming driver too), so the
/// classic loop's contract stays pinned on the unified engine.
#[cfg(test)]
mod classic_tests {
    use serde_json::json;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering::SeqCst},
    };

    use crate::agent::hook::InvalidToolCallContext;
    use crate::agent::mock_support::{MockCompletionModel, MockTurn};
    use crate::agent::response::TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER;
    use crate::agent::{AgentBuilder, InvalidToolCallAction};
    use crate::completion::{
        AssistantContent, CompletionError, CompletionRequest, Message, PromptError,
        StructuredOutputError, Usage,
    };
    use crate::hooks::{HookDecision, HookEntry, HookEvent};
    use crate::test_utils::{MockAddTool, MockOperationArgs, MockSubtractTool, MockToolError};
    use crate::tool::PortableTool;
    use rig_core::message::{
        Text, ToolCall as MessageToolCall, ToolChoice, ToolFunction, ToolResultContent, UserContent,
    };
    use schemars::JsonSchema;
    use serde::Deserialize;

    // ------------------------------------------------------------------
    // Shared helpers
    // ------------------------------------------------------------------

    /// Named hook entry over a synchronous decision function.
    fn hook_entry(
        name: &str,
        decide: impl Fn(HookEvent) -> HookDecision + Send + Sync + 'static,
    ) -> HookEntry {
        HookEntry::sync(name, decide)
    }

    #[derive(Debug, Deserialize, JsonSchema, PartialEq)]
    struct TypedAnswer {
        value: String,
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

    /// A tool that counts how many times it executes.
    #[derive(Clone)]
    struct CountingAddTool {
        calls: Arc<AtomicU32>,
    }

    impl PortableTool for CountingAddTool {
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

        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, SeqCst);
            MockAddTool.call(args).await
        }
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

    /// Whether any tool result in `messages` carries `expected` as verbatim text.
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
                                ToolResultContent::Text(text) if text.text == expected
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
                                ToolResultContent::Json { value } if value == expected
                            ))
                    ))
            )
        })
    }

    // ------------------------------------------------------------------
    // Invalid tool calls, tool_choice enforcement, typed prompts
    // ------------------------------------------------------------------

    /// Fails the test if either the response hook or a tool-call hook runs:
    /// an unknown tool call must fail before both.
    fn panic_on_unknown_tool_hook() -> HookEntry {
        hook_entry("panic-on-unknown-tool", |event| match event {
            HookEvent::CompletionResponse { .. } => {
                panic!("unknown tool response should fail before response hooks run")
            }
            HookEvent::ToolCall { .. } => {
                panic!("unknown tool call should fail before tool hooks run")
            }
            _ => HookDecision::Continue,
        })
    }

    /// Skips `default_api` and fails the test if a normal tool-call hook runs.
    fn skip_default_api_and_panic_on_tool_call_hook() -> HookEntry {
        hook_entry("skip-default-api-panic-on-tool-call", |event| match event {
            HookEvent::InvalidToolCall(_) => HookDecision::InvalidToolCall(
                InvalidToolCallAction::skip("default_api is not available"),
            ),
            HookEvent::ToolCall { .. } => {
                panic!("recovered invalid turn should not invoke normal tool hooks")
            }
            _ => HookDecision::Continue,
        })
    }

    fn repair_default_api_hook() -> HookEntry {
        hook_entry("repair-default-api", |event| {
            let HookEvent::InvalidToolCall(context) = event else {
                return HookDecision::Continue;
            };
            assert_eq!(context.tool_name, "default_api");
            HookDecision::InvalidToolCall(InvalidToolCallAction::repair("add"))
        })
    }

    fn repair_to_subtract_hook() -> HookEntry {
        hook_entry("repair-to-subtract", |event| {
            let HookEvent::InvalidToolCall(_) = event else {
                return HookDecision::Continue;
            };
            HookDecision::InvalidToolCall(InvalidToolCallAction::repair("subtract"))
        })
    }

    fn retry_default_api_hook() -> HookEntry {
        hook_entry("retry-default-api", |event| {
            let HookEvent::InvalidToolCall(context) = event else {
                return HookDecision::Continue;
            };
            HookDecision::InvalidToolCall(InvalidToolCallAction::retry(format!(
                "Use one of these tools instead: {:?}",
                context.allowed_tools
            )))
        })
    }

    fn skip_default_api_hook() -> HookEntry {
        hook_entry("skip-default-api", |event| {
            let HookEvent::InvalidToolCall(_) = event else {
                return HookDecision::Continue;
            };
            HookDecision::InvalidToolCall(InvalidToolCallAction::skip(
                "default_api is not available",
            ))
        })
    }

    #[derive(Clone, Default)]
    struct RecordingInvalidToolCallHook {
        contexts: Arc<Mutex<Vec<InvalidToolCallContext>>>,
    }

    impl RecordingInvalidToolCallHook {
        fn observed(&self) -> Vec<InvalidToolCallContext> {
            self.contexts
                .lock()
                .expect("invalid tool context records mutex was poisoned")
                .clone()
        }

        fn entry(&self) -> HookEntry {
            let recorder = self.clone();
            hook_entry("recording-invalid-tool-call", move |event| {
                if let HookEvent::InvalidToolCall(context) = event {
                    recorder
                        .contexts
                        .lock()
                        .expect("invalid tool context records mutex was poisoned")
                        .push(context);
                }
                HookDecision::Continue
            })
        }
    }

    fn validate_follow_up_tool_history(request: &CompletionRequest) {
        let history = request.chat_history.iter().cloned().collect::<Vec<_>>();
        assert_eq!(
            history.len(),
            3,
            "follow-up request should contain the prompt, assistant tool call, and user tool result: {history:?}"
        );

        assert!(matches!(
            history.first(),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::Text(text) if text.text == "do tool work"
                )
        ));

        assert!(matches!(
            history.get(1),
            Some(Message::Assistant { content, .. })
                if matches!(
                    content.first(),
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.call_id.as_deref() == Some("call_1")
                )
        ));

        assert!(matches!(
            history.get(2),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::ToolResult(tool_result)
                        if tool_result.id == "tool_call_1"
                            && tool_result.call_id.as_deref() == Some("call_1")
                )
        ));
    }

    #[tokio::test]
    async fn unknown_tool_call_fails_before_non_streaming_second_request() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 1, "y": 2})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build();

        let err = agent
            .runner("use the tool")
            .add_hook(panic_on_unknown_tool_hook())
            .max_turns(3)
            .run()
            .await
            .expect_err("unknown model-emitted tool should fail");

        match err {
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
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn invalid_tool_call_context_uses_completed_tool_call_provider_id() {
        let invalid_hook = RecordingInvalidToolCallHook::default();
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 1, "y": 2}))
                .with_call_id("provider_call_1"),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build();

        let err = agent
            .runner("use the tool")
            .add_hook(invalid_hook.entry())
            .max_turns(3)
            .run()
            .await
            .expect_err("invalid tool should fail");

        assert!(matches!(err, PromptError::UnknownToolCall { .. }));
        assert_eq!(recorded.request_count(), 1);
        let contexts = invalid_hook.observed();
        assert_eq!(contexts.len(), 1);
        let context = contexts.first().expect("one context");
        assert_eq!(context.tool_name, "default_api");
        assert_eq!(context.tool_call_id.as_deref(), Some("tool_call_1"));
        assert_eq!(context.internal_call_id, None);
        assert!(!context.is_streaming);
    }

    #[tokio::test]
    async fn disallowed_specific_tool_call_fails_before_non_streaming_second_request() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "subtract", json!({"x": 3, "y": 1})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let err = agent
            .runner("use the allowed tool")
            .add_hook(panic_on_unknown_tool_hook())
            .max_turns(3)
            .run()
            .await
            .expect_err("disallowed model-emitted tool should fail");

        match err {
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
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn tool_choice_none_rejects_non_streaming_tool_call() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "add", json!({"x": 1, "y": 2})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let err = agent
            .runner("do not use tools")
            .add_hook(panic_on_unknown_tool_hook())
            .max_turns(3)
            .run()
            .await
            .expect_err("ToolChoice::None should reject returned tool calls");

        match err {
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
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_can_repair_non_streaming_tool_name() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("done"),
        ]);
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build();

        let response = agent
            .runner("add")
            .add_hook(repair_default_api_hook())
            .max_turns(3)
            .run()
            .await
            .expect("repaired tool call should execute");

        assert_eq!(response.output, "done");
        let messages = response.messages.expect("messages should be present");
        assert!(history_contains_tool_call(&messages, "add"));
        assert!(!history_contains_tool_call(&messages, "default_api"));
        assert!(tool_result_json_in_history(&messages, &json!(5)));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_retry_adds_feedback_and_retries_non_streaming() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("retried"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build();

        let response = agent
            .runner("add")
            .add_hook(retry_default_api_hook())
            .max_invalid_tool_call_retries(1)
            .max_turns(3)
            .run()
            .await
            .expect("retry should recover");

        assert_eq!(response.output, "retried");
        assert_eq!(recorded.request_count(), 2);
        let messages = response.messages.expect("messages should be present");
        assert!(messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|content| {
                        matches!(
                            content,
                            UserContent::ToolResult(result)
                                if result.content.iter().any(|content| {
                                    matches!(
                                        content,
                                        ToolResultContent::Text(text)
                                            if text.text.contains("Use one of these tools instead")
                                    )
                                })
                        )
                    })
            )
        }));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_retries_mixed_non_streaming_turn_without_executing_valid_call()
    {
        let add_calls = Arc::new(AtomicU32::new(0));
        let mut valid_tool_call = MessageToolCall::new(
            "tool_call_1".to_string(),
            ToolFunction::new("add".to_string(), json!({"x": 2, "y": 3})),
        );
        valid_tool_call.call_id = Some("call_1".to_string());
        let mut invalid_tool_call = MessageToolCall::new(
            "tool_call_2".to_string(),
            ToolFunction::new("default_api".to_string(), json!({"x": 4, "y": 5})),
        );
        invalid_tool_call.call_id = Some("call_2".to_string());
        let model = MockCompletionModel::new([
            MockTurn::from_contents([
                AssistantContent::ToolCall(valid_tool_call),
                AssistantContent::ToolCall(invalid_tool_call),
            ])
            .expect("tool-call response should be non-empty"),
            MockTurn::text("retried"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let response = agent
            .runner("add")
            .add_hook(retry_default_api_hook())
            .max_invalid_tool_call_retries(1)
            .max_turns(3)
            .run()
            .await
            .expect("retry should recover");

        assert_eq!(response.output, "retried");
        assert_eq!(add_calls.load(SeqCst), 0);
        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let retry_history = requests
            .get(1)
            .expect("second request")
            .chat_history
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(retry_history.len(), 3);
        assert!(matches!(
            retry_history.get(1),
            Some(Message::Assistant { content, .. })
                if content.iter().any(|item| matches!(
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
                                        if text.text.contains("Use one of these tools instead")
                                ))
            ))
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_skips_mixed_non_streaming_turn_without_executing_valid_call() {
        let add_calls = Arc::new(AtomicU32::new(0));
        let mut valid_tool_call = MessageToolCall::new(
            "tool_call_1".to_string(),
            ToolFunction::new("add".to_string(), json!({"x": 2, "y": 3})),
        );
        valid_tool_call.call_id = Some("call_1".to_string());
        let mut invalid_tool_call = MessageToolCall::new(
            "tool_call_2".to_string(),
            ToolFunction::new("default_api".to_string(), json!({"x": 4, "y": 5})),
        );
        invalid_tool_call.call_id = Some("call_2".to_string());
        let model = MockCompletionModel::new([
            MockTurn::from_contents([
                AssistantContent::ToolCall(valid_tool_call),
                AssistantContent::ToolCall(invalid_tool_call),
            ])
            .expect("tool-call response should be non-empty"),
            MockTurn::text("skipped"),
        ]);
        let agent = AgentBuilder::new(model.provider())
            .tool(CountingAddTool {
                calls: add_calls.clone(),
            })
            .build();

        let response = agent
            .runner("add")
            .add_hook(skip_default_api_and_panic_on_tool_call_hook())
            .max_turns(3)
            .run()
            .await
            .expect("skip should recover without executing peer tools");

        assert_eq!(response.output, "skipped");
        assert_eq!(add_calls.load(SeqCst), 0);
        let messages = response.messages.expect("messages should be present");
        assert!(history_contains_tool_call(&messages, "add"));
        assert!(history_contains_tool_call(&messages, "default_api"));
        assert!(matches!(
            messages.get(2),
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
                                        if text.text == "default_api is not available"
                                ))
                    ))
        ));
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_retry_budget_exhaustion_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build();

        let err = agent
            .runner("add")
            .add_hook(retry_default_api_hook())
            .max_invalid_tool_call_retries(0)
            .max_turns(3)
            .run()
            .await
            .expect_err("retry without budget should fail");

        match err {
            PromptError::UnknownToolCall {
                tool_name,
                chat_history,
                ..
            } => {
                assert_eq!(tool_name, "default_api");
                assert!(history_contains_tool_call(&chat_history, "default_api"));
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn invalid_tool_call_hook_can_skip_structured_non_streaming_call() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("skipped"),
        ]);
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build();

        let response = agent
            .runner("add")
            .add_hook(skip_default_api_hook())
            .max_turns(3)
            .run()
            .await
            .expect("skip should continue with synthetic tool result");

        assert_eq!(response.output, "skipped");
        let messages = response.messages.expect("messages should be present");
        assert!(history_contains_tool_call(&messages, "default_api"));
        assert!(tool_result_text_in_history(
            &messages,
            "default_api is not available"
        ));
    }

    #[tokio::test]
    async fn skip_under_specific_tool_choice_returns_synthetic_feedback() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("skipped"),
        ]);
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let response = agent
            .runner("add")
            .add_hook(skip_default_api_hook())
            .max_turns(3)
            .run()
            .await
            .expect("skip should produce synthetic feedback under Specific");

        assert_eq!(response.output, "skipped");
        let messages = response.messages.expect("messages should be present");
        assert!(history_contains_tool_call(&messages, "default_api"));
        assert!(messages.iter().any(|message| {
            matches!(
                message,
                Message::User { content }
                    if content.iter().any(|content| {
                        matches!(
                            content,
                            UserContent::ToolResult(result)
                                if result.id == "tool_call_1"
                                    && result.content.iter().any(|content| {
                                        matches!(
                                            content,
                                            ToolResultContent::Text(text)
                                                if text.text == "default_api is not available"
                                        )
                                    })
                        )
                    })
            )
        }));
    }

    #[tokio::test]
    async fn repair_to_disallowed_specific_tool_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let err = agent
            .runner("add")
            .add_hook(repair_to_subtract_hook())
            .max_turns(3)
            .run()
            .await
            .expect_err("repair to a disallowed tool should fail");

        match err {
            PromptError::UnknownToolCall { tool_name, .. } => {
                assert_eq!(tool_name, "subtract");
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn repair_under_tool_choice_none_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let err = agent
            .runner("do not use tools")
            .add_hook(repair_default_api_hook())
            .max_turns(3)
            .run()
            .await
            .expect_err("ToolChoice::None should reject repaired tool calls");

        match err {
            PromptError::UnknownToolCall { tool_name, .. } => {
                assert_eq!(tool_name, "add");
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn runner_skip_under_tool_choice_none_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("should not be requested"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .tool_choice(ToolChoice::None)
            .build();

        let err = agent
            .runner("do not use tools")
            .add_hook(skip_default_api_hook())
            .max_turns(3)
            .run()
            .await
            .expect_err("ToolChoice::None should reject skipped tool calls");

        match err {
            PromptError::UnknownToolCall { tool_name, .. } => {
                assert_eq!(tool_name, "default_api");
            }
            other => panic!("expected UnknownToolCall, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn typed_prompt_default_invalid_tool_call_fails_fast() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text(r#"{"value":"should not be requested"}"#),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build();

        let err = agent
            .runner("return typed json")
            .add_hook(panic_on_unknown_tool_hook())
            .max_turns(3)
            .run_typed::<TypedAnswer>()
            .await
            .expect_err("typed prompt should preserve fail-fast default");

        match err {
            StructuredOutputError::PromptError(err) => match *err {
                PromptError::UnknownToolCall { tool_name, .. } => {
                    assert_eq!(tool_name, "default_api");
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn typed_prompt_invalid_tool_call_hook_can_repair_tool_name() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text(r#"{"value":"repaired"}"#),
        ]);
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build();

        let response = agent
            .runner("return typed json")
            .add_hook(repair_default_api_hook())
            .max_turns(3)
            .run_typed::<TypedAnswer>()
            .await
            .expect("typed prompt should repair invalid tool call");

        assert_eq!(
            response,
            TypedAnswer {
                value: "repaired".to_string()
            }
        );
    }

    #[tokio::test]
    async fn typed_prompt_invalid_tool_call_hook_can_retry_and_parse_response() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text(r#"{"value":"retried"}"#),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build();

        let response = agent
            .runner("return typed json")
            .add_hook(retry_default_api_hook())
            .max_invalid_tool_call_retries(1)
            .max_turns(3)
            .run_typed::<TypedAnswer>()
            .await
            .expect("typed prompt should retry invalid tool call");

        assert_eq!(
            response,
            TypedAnswer {
                value: "retried".to_string()
            }
        );
        assert_eq!(recorded.request_count(), 2);
    }

    #[tokio::test]
    async fn typed_prompt_invalid_tool_call_retry_budget_exhaustion_fails() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text(r#"{"value":"should not be requested"}"#),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build();

        let err = agent
            .runner("return typed json")
            .add_hook(retry_default_api_hook())
            .max_invalid_tool_call_retries(0)
            .max_turns(3)
            .run_typed::<TypedAnswer>()
            .await
            .expect_err("typed prompt should fail when retry budget is exhausted");

        match err {
            StructuredOutputError::PromptError(err) => match *err {
                PromptError::UnknownToolCall { tool_name, .. } => {
                    assert_eq!(tool_name, "default_api");
                }
                other => panic!("expected UnknownToolCall, got {other:?}"),
            },
            other => panic!("expected prompt error, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn invalid_specific_tool_choice_fails_before_non_streaming_provider_request() {
        let model = MockCompletionModel::text("should not be requested");
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["missing".to_string()],
            })
            .build();

        let err = agent
            .runner("use the missing tool")
            .run()
            .await
            .expect_err("invalid ToolChoice::Specific should fail before provider request");

        match err {
            PromptError::CompletionError(CompletionError::RequestError(err)) => {
                let msg = err.to_string();
                assert!(msg.contains("missing"), "got: {msg}");
                assert!(msg.contains("add"), "got: {msg}");
            }
            other => panic!("expected CompletionError::RequestError, got {other:?}"),
        }
        assert_eq!(recorded.request_count(), 0);
    }

    #[tokio::test]
    async fn allowed_specific_tool_call_executes_normally() {
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "add", json!({"x": 1, "y": 2})),
            MockTurn::text("done"),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["add".to_string()],
            })
            .build();

        let response = agent
            .runner("use the allowed tool")
            .max_turns(3)
            .run()
            .await
            .expect("allowed specific tool should execute");

        assert_eq!(response.output, "done");
        assert_eq!(recorded.request_count(), 2);
    }

    #[tokio::test]
    async fn prompt_request_stops_cleanly_on_empty_terminal_turn() {
        let call_usage = usage(1, 1);
        let model = MockCompletionModel::new([
            MockTurn::tool_call("tool_call_1", "add", json!({"x": 1, "y": 2}))
                .with_call_id("call_1")
                .with_usage(call_usage),
            MockTurn::text("").with_usage(call_usage),
        ]);
        let recorded = model.clone();
        let agent = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build();

        let response = agent
            .runner("do tool work")
            .max_turns(3)
            .run()
            .await
            .expect("empty terminal turn should not error");

        assert!(response.output.is_empty());
        assert_eq!(response.usage, usage(2, 2));
        assert_eq!(
            response.completion_calls(),
            &[
                crate::agent::CompletionCall::new(0, call_usage),
                crate::agent::CompletionCall::new(1, call_usage)
            ]
        );

        let history = response
            .messages
            .expect("extended response should include history");
        assert_eq!(history.len(), 3);
        assert!(matches!(
            history.first(),
            Some(Message::User { content })
                if matches!(
                    content.first(),
                    UserContent::Text(text) if text.text == "do tool work"
                )
        ));
        assert!(history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if matches!(
                    content.first(),
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.id == "tool_call_1"
                            && tool_call.call_id.as_deref() == Some("call_1")
                )
        )));
        assert!(history.iter().any(|message| matches!(
            message,
            Message::User { content }
                if matches!(
                    content.first(),
                    UserContent::ToolResult(tool_result)
                        if tool_result.id == "tool_call_1"
                            && tool_result.call_id.as_deref() == Some("call_1")
                )
        )));
        assert!(!history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text.is_empty()
                ))
        )));
        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        validate_follow_up_tool_history(requests.get(1).expect("second request"));
    }

    #[tokio::test]
    async fn prompt_request_concatenates_text_blocks_without_inserted_newlines() {
        let model = MockCompletionModel::new([MockTurn::from_contents([
            AssistantContent::Text(Text::new("According to the document, ")),
            AssistantContent::Text(Text::new("the grass is green")),
            AssistantContent::Text(Text::new(" and the sky is blue.")),
        ])
        .expect("mock response should contain text blocks")]);
        let agent = AgentBuilder::new(model.provider()).build();

        let response = agent
            .runner("answer with cited spans")
            .run()
            .await
            .expect("prompt should succeed");

        assert_eq!(
            response.output,
            "According to the document, the grass is green and the sky is blue."
        );
    }

    #[tokio::test]
    async fn prompt_request_preserves_metadata_only_text_turn_in_history() {
        let metadata = json!({
            "citations": [{
                "type": "web_search_result_location",
                "cited_text": "Claude Shannon was born in 1916.",
                "url": "https://example.com/shannon",
                "title": null,
                "encrypted_index": "encrypted-reference"
            }]
        });
        let model =
            MockCompletionModel::new([MockTurn::from_content(AssistantContent::Text(Text {
                text: String::new(),
                additional_params: Some(metadata.clone()),
            }))]);
        let agent = AgentBuilder::new(model.provider()).build();

        let response = agent
            .runner("answer with cited metadata")
            .run()
            .await
            .expect("metadata-only text turn should succeed");

        assert!(response.output.is_empty());
        let history = response
            .messages
            .expect("extended response should include history");
        assert!(history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if matches!(
                    content.first(),
                    AssistantContent::Text(text)
                        if text.text.is_empty()
                            && text.additional_params.as_ref() == Some(&metadata)
                )
        )));
    }
}

/// Hook-lifecycle and run/stream-parity behavior ported from the deleted
/// classic driver's `migrated_tests` corpus. The drivers are now
/// [`AgentSession::drive`] (blocking) and
/// [`AgentStream::drive`](crate::stream::AgentStream::drive) (streaming),
/// reached through [`SessionRunner::run`](crate::agent::SessionRunner::run) /
/// [`SessionRunner::stream_run`](crate::agent::SessionRunner::stream_run).
#[cfg(test)]
mod classic_hook_tests {
    use futures::StreamExt;
    use serde_json::json;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering::SeqCst},
    };
    use tokio::sync::Barrier;

    use crate::agent::hook::RequestPatch;
    use crate::agent::mock_support::{MockCompletionModel, MockStreamEvent, MockTurn};
    use crate::agent::run::OutputMode;
    use crate::agent::{
        AgentBuilder, CompletionCallAction, InvalidToolCallAction, ModelTurnAction,
        ObservationAction, ToolCallAction, ToolResultAction,
    };
    use crate::completion::{Message, PromptError, Usage};
    use crate::hooks::{HookDecision, HookEntry, HookEvent, Hooks};
    use crate::stream::AgentStreamItem;
    use crate::streaming::{StreamedAssistantContent, StreamedUserContent};
    use crate::test_utils::{
        MockAddTool, MockBarrierTool, MockOperationArgs, MockSubtractTool, MockToolError,
    };
    use crate::tool::{PortableTool, ToolExecutionError};
    use rig_core::OneOrMany;
    use rig_core::message::{
        AssistantContent, ToolCall as MessageToolCall, ToolChoice, ToolFunction, ToolResultContent,
        UserContent,
    };

    /// Named hook entry over a synchronous decision function.
    fn hook_entry(
        name: &str,
        decide: impl Fn(HookEvent) -> HookDecision + Send + Sync + 'static,
    ) -> HookEntry {
        HookEntry::sync(name, decide)
    }

    /// Test-local mirror of the deleted `StepEventKind`: the recorded identity of
    /// each dispatched [`HookEvent`], so event *sequences* stay assertable.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum EventKind {
        CompletionCall,
        CompletionResponse,
        ModelTurnFinished,
        InvalidToolCall,
        ToolCall,
        ToolResult,
        TextDelta,
        ToolCallDelta,
        StreamResponseFinish,
    }

    /// Records the kind of every hook event (and every tool-result payload) so a
    /// run() and a stream() of the same scenario can be compared.
    #[derive(Clone, Default)]
    struct RecordingHook {
        events: Arc<Mutex<Vec<EventKind>>>,
        tool_results: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingHook {
        /// Event kinds that should be identical across streaming and
        /// non-streaming (excludes the medium-specific delta / response-finish
        /// events).
        fn shared_events(&self) -> Vec<EventKind> {
            self.events
                .lock()
                .expect("events lock")
                .iter()
                .copied()
                .filter(|kind| {
                    matches!(
                        kind,
                        EventKind::CompletionCall
                            | EventKind::ToolCall
                            | EventKind::ToolResult
                            | EventKind::InvalidToolCall
                    )
                })
                .collect()
        }

        fn tool_results(&self) -> Vec<String> {
            self.tool_results.lock().expect("results lock").clone()
        }

        fn all_events(&self) -> Vec<EventKind> {
            self.events.lock().expect("events lock").clone()
        }

        /// Count of a single event kind across the whole run.
        fn count(&self, kind: EventKind) -> usize {
            self.events
                .lock()
                .expect("events lock")
                .iter()
                .filter(|recorded| **recorded == kind)
                .count()
        }

        fn record(&self, kind: EventKind) {
            self.events.lock().expect("events lock").push(kind);
        }

        /// An observe-everything entry (deltas included) that never steers.
        fn entry(&self) -> HookEntry {
            let recorder = self.clone();
            hook_entry("recording", move |event| {
                match event {
                    HookEvent::BeforeModelCall { .. } => recorder.record(EventKind::CompletionCall),
                    HookEvent::CompletionResponse { .. } => {
                        recorder.record(EventKind::CompletionResponse)
                    }
                    HookEvent::ModelTurnFinished { .. } => {
                        recorder.record(EventKind::ModelTurnFinished)
                    }
                    HookEvent::InvalidToolCall(_) => recorder.record(EventKind::InvalidToolCall),
                    HookEvent::ToolCall { .. } => recorder.record(EventKind::ToolCall),
                    HookEvent::ToolResult { presentation, .. } => {
                        recorder.record(EventKind::ToolResult);
                        recorder
                            .tool_results
                            .lock()
                            .expect("results lock")
                            .push(presentation.render());
                    }
                    HookEvent::TextDelta { .. } => recorder.record(EventKind::TextDelta),
                    HookEvent::ToolCallDelta { .. } => recorder.record(EventKind::ToolCallDelta),
                    HookEvent::StreamResponseFinish { .. } => {
                        recorder.record(EventKind::StreamResponseFinish)
                    }
                    _ => {}
                }
                HookDecision::Continue
            })
            .observing_deltas()
        }
    }

    /// A tool that counts how many times it executes.
    #[derive(Clone)]
    struct CountingAddTool {
        calls: Arc<AtomicU32>,
    }

    impl PortableTool for CountingAddTool {
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
        async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, SeqCst);
            MockAddTool.call(args).await
        }
    }

    fn tool_call_content(id: &str, args: serde_json::Value) -> AssistantContent {
        AssistantContent::ToolCall(MessageToolCall::new(
            id.to_string(),
            ToolFunction::new("add".to_string(), args),
        ))
    }

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
                                ToolResultContent::Text(text) if text.text == expected
                            ))
                    ))
            )
        })
    }

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
                                ToolResultContent::Json { value } if value == expected
                            ))
                    ))
            )
        })
    }

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

    fn one_text_stream_turn(text: &'static str) -> Vec<MockStreamEvent> {
        vec![
            MockStreamEvent::text(text),
            MockStreamEvent::final_response_with_total_tokens(0),
        ]
    }

    /// Drive a hook-driven stream to completion, panicking on any error, and
    /// return its final response.
    async fn drive_to_final_response(
        stream: impl futures::Stream<Item = Result<AgentStreamItem, PromptError>>,
    ) -> crate::agent::PromptResponse {
        let mut stream = Box::pin(stream);
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            if let AgentStreamItem::Final(resp) =
                item.unwrap_or_else(|err| panic!("stream item errored: {err}"))
            {
                final_response = Some(resp);
            }
        }
        final_response.expect("stream should yield a final response")
    }

    // ------------------------------------------------------------------
    // Canonical response / stream-finish lifecycle
    // ------------------------------------------------------------------

    #[derive(Clone, Debug, PartialEq)]
    struct CanonicalResponseSnapshot {
        prompt: Message,
        content: OneOrMany<AssistantContent>,
        usage: Usage,
        message_id: Option<String>,
    }

    #[derive(Clone, Default)]
    struct CanonicalResponseHook {
        blocking: Arc<Mutex<Vec<CanonicalResponseSnapshot>>>,
        streaming: Arc<Mutex<Vec<CanonicalResponseSnapshot>>>,
        committed: Arc<Mutex<Vec<OneOrMany<AssistantContent>>>>,
    }

    impl CanonicalResponseHook {
        fn entry(&self) -> HookEntry {
            let hook = self.clone();
            hook_entry("canonical-response", move |event| {
                match event {
                    HookEvent::CompletionResponse {
                        prompt, response, ..
                    } => hook.blocking.lock().expect("blocking snapshots").push(
                        CanonicalResponseSnapshot {
                            prompt,
                            content: response.choice,
                            usage: response.usage,
                            message_id: response.message_id,
                        },
                    ),
                    HookEvent::StreamResponseFinish {
                        prompt,
                        content,
                        usage,
                        message_id,
                        ..
                    } => hook.streaming.lock().expect("streaming snapshots").push(
                        CanonicalResponseSnapshot {
                            prompt,
                            content,
                            usage,
                            message_id,
                        },
                    ),
                    HookEvent::ModelTurnFinished { content, .. } => hook
                        .committed
                        .lock()
                        .expect("committed snapshots")
                        .push(content),
                    _ => {}
                }
                HookDecision::Continue
            })
        }
    }

    #[derive(Clone, Default)]
    struct FinishLifecycleHook {
        snapshots: Arc<Mutex<Vec<CanonicalResponseSnapshot>>>,
        model_turns: Arc<AtomicU32>,
        stop: Arc<AtomicBool>,
    }

    impl FinishLifecycleHook {
        fn entry(&self) -> HookEntry {
            let hook = self.clone();
            hook_entry("finish-lifecycle", move |event| match event {
                HookEvent::StreamResponseFinish {
                    prompt,
                    content,
                    usage,
                    message_id,
                    ..
                } => {
                    hook.snapshots.lock().expect("finish snapshots").push(
                        CanonicalResponseSnapshot {
                            prompt,
                            content,
                            usage,
                            message_id,
                        },
                    );
                    if hook.stop.load(SeqCst) {
                        HookDecision::Observation(ObservationAction::stop("stop at stream EOF"))
                    } else {
                        HookDecision::Continue
                    }
                }
                HookEvent::ModelTurnFinished { .. } => {
                    hook.model_turns.fetch_add(1, SeqCst);
                    HookDecision::Continue
                }
                _ => HookDecision::Continue,
            })
        }
    }

    fn canonical_usage() -> Usage {
        Usage {
            input_tokens: 11,
            output_tokens: 7,
            total_tokens: 18,
            ..Usage::new()
        }
    }

    #[tokio::test]
    async fn blocking_completion_response_hook_receives_canonical_fields() {
        let hook = CanonicalResponseHook::default();
        let prompt = Message::user("canonical prompt");
        AgentBuilder::new(
            MockCompletionModel::new([MockTurn::text("canonical response")
                .with_usage(canonical_usage())
                .with_message_id("msg-canonical")])
            .provider(),
        )
        .add_hook(hook.entry())
        .build()
        .runner(prompt.clone())
        .run()
        .await
        .expect("blocking response");

        assert_eq!(
            *hook.blocking.lock().expect("blocking snapshots"),
            [CanonicalResponseSnapshot {
                prompt,
                content: OneOrMany::one(AssistantContent::text("canonical response")),
                usage: canonical_usage(),
                message_id: Some("msg-canonical".to_string()),
            }]
        );
    }

    #[tokio::test]
    async fn streaming_response_finish_matches_blocking_canonical_fields() {
        let prompt = Message::user("canonical prompt");
        let blocking_hook = CanonicalResponseHook::default();
        AgentBuilder::new(
            MockCompletionModel::new([MockTurn::text("canonical response")
                .with_usage(canonical_usage())
                .with_message_id("msg-canonical")])
            .provider(),
        )
        .add_hook(blocking_hook.entry())
        .build()
        .runner(prompt.clone())
        .run()
        .await
        .expect("blocking response");

        let streaming_hook = CanonicalResponseHook::default();
        let mut stream = Box::pin(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns([[
                    MockStreamEvent::text("canonical response"),
                    MockStreamEvent::final_response(canonical_usage()),
                    MockStreamEvent::message_id("msg-canonical"),
                ]])
                .provider(),
            )
            .add_hook(streaming_hook.entry())
            .build()
            .runner(prompt)
            .stream_run(),
        );
        while let Some(item) = stream.next().await {
            item.expect("stream item");
        }

        let blocking = blocking_hook
            .blocking
            .lock()
            .expect("blocking snapshots")
            .clone();
        let streaming = streaming_hook
            .streaming
            .lock()
            .expect("streaming snapshots")
            .clone();
        assert_eq!(streaming, blocking);
        let first = streaming.first().expect("one streaming snapshot");
        assert_eq!(first.usage, canonical_usage());
        assert_eq!(first.message_id.as_deref(), Some("msg-canonical"));
    }

    #[tokio::test]
    async fn streaming_response_finish_without_provider_message_id_reports_none() {
        let hook = FinishLifecycleHook::default();
        let mut stream = Box::pin(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns([[
                    MockStreamEvent::text("canonical response"),
                    MockStreamEvent::final_response(canonical_usage()),
                ]])
                .provider(),
            )
            .add_hook(hook.entry())
            .build()
            .runner("canonical prompt")
            .stream_run(),
        );
        while let Some(item) = stream.next().await {
            item.expect("stream item");
        }

        let snapshots = hook.snapshots.lock().expect("finish snapshots");
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots.first().expect("snapshot").message_id, None);
    }

    /// Stops the run from every accepted model turn.
    fn stop_completed_model_turn() -> HookEntry {
        hook_entry("stop-completed-model-turn", |event| {
            let HookEvent::ModelTurnFinished { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::ModelTurn(ModelTurnAction::stop("stop completed model turn"))
        })
    }

    #[tokio::test]
    async fn streaming_model_turn_stop_preserves_completed_provider_final() {
        let mut stream = Box::pin(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns([[
                    MockStreamEvent::text("canonical response"),
                    MockStreamEvent::final_response(canonical_usage()),
                ]])
                .provider(),
            )
            .add_hook(stop_completed_model_turn())
            .build()
            .runner("canonical prompt")
            .stream_run(),
        );

        let mut provider_finals = 0;
        let mut saw_retry = false;
        let mut saw_run_final = false;
        let mut error = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(AgentStreamItem::Assistant(StreamedAssistantContent::Final(_))) => {
                    provider_finals += 1
                }
                Ok(AgentStreamItem::ModelTurnRetried { .. }) => saw_retry = true,
                Ok(AgentStreamItem::Final(_)) => saw_run_final = true,
                Ok(_) => {}
                Err(err) => error = Some(err),
            }
        }

        assert_eq!(provider_finals, 1);
        assert!(!saw_retry);
        assert!(!saw_run_final);
        assert!(matches!(
            error,
            Some(PromptError::PromptCancelled { reason, .. })
                if reason == "stop completed model turn"
        ));
    }

    #[tokio::test]
    async fn streaming_response_finish_normalizes_interleaved_content() {
        let hook = CanonicalResponseHook::default();
        let mut stream = Box::pin(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns([
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
                ])
                .provider(),
            )
            .tool(MockAddTool)
            .add_hook(hook.entry())
            .build()
            .runner("go")
            .max_turns(3)
            .stream_run(),
        );
        while let Some(item) = stream.next().await {
            item.expect("stream item");
        }

        let snapshots = hook.streaming.lock().expect("streaming snapshots");
        let committed = hook.committed.lock().expect("committed snapshots");
        let first = snapshots.first().expect("one streaming snapshot");
        let kinds = first
            .content
            .iter()
            .map(|content| match content {
                AssistantContent::Reasoning(_) => "reasoning",
                AssistantContent::Text(_) => "text",
                AssistantContent::ToolCall(_) => "tool_call",
                _ => "other",
            })
            .collect::<Vec<_>>();
        assert_eq!(kinds, ["reasoning", "text", "tool_call"]);
        assert_eq!(
            &first.content,
            committed.first().expect("committed turn"),
            "finish hook and committed turn must share one canonical choice"
        );
    }

    // ------------------------------------------------------------------
    // Budgets and run/stream parity
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn prompt_surfaces_reject_second_tool_roundtrip_request_at_budget_one() {
        let blocking_model = blocking_model();
        let blocking_recorded = blocking_model.clone();
        let blocking_agent = AgentBuilder::new(blocking_model.provider())
            .tool(MockAddTool)
            .build();
        let blocking_err = blocking_agent
            .runner("add 2 and 3")
            .max_turns(1)
            .run()
            .await
            .expect_err("blocking prompt should reject request two");
        assert!(matches!(
            blocking_err,
            PromptError::MaxTurnsError { max_turns: 1, .. }
        ));
        assert_eq!(blocking_recorded.request_count(), 1);

        let streaming_model = streaming_model();
        let streaming_recorded = streaming_model.clone();
        let streaming_agent = AgentBuilder::new(streaming_model.provider())
            .tool(MockAddTool)
            .build();
        let mut stream = Box::pin(
            streaming_agent
                .runner("add 2 and 3")
                .max_turns(1)
                .stream_run(),
        );
        let mut streaming_err = None;
        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                streaming_err = Some(err);
                break;
            }
        }
        assert!(matches!(
            streaming_err,
            Some(PromptError::MaxTurnsError { max_turns: 1, .. })
        ));
        assert_eq!(streaming_recorded.request_count(), 1);
    }

    #[tokio::test]
    async fn run_and_stream_behave_identically_for_a_tool_call() {
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model().provider())
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(2)
            .add_hook(blocking_hook.entry())
            .run()
            .await
            .expect("blocking run should succeed");

        let streaming_hook = RecordingHook::default();
        let final_response = drive_to_final_response(
            AgentBuilder::new(streaming_model().provider())
                .tool(MockAddTool)
                .build()
                .runner("add 2 and 3")
                .max_turns(2)
                .add_hook(streaming_hook.entry())
                .stream_run(),
        )
        .await;

        assert_eq!(blocking.output, "the answer is 5");
        assert_eq!(final_response.output(), blocking.output);

        assert_eq!(
            blocking_hook.shared_events(),
            streaming_hook.shared_events()
        );
        assert_eq!(
            blocking_hook.shared_events(),
            vec![
                EventKind::CompletionCall,
                EventKind::ToolCall,
                EventKind::ToolResult,
                EventKind::CompletionCall,
            ]
        );

        assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());
        assert_eq!(blocking_hook.tool_results(), vec!["5".to_string()]);

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

    /// A request issued straight against the provider dispatcher bypasses the
    /// agent loop entirely, so no agent hook fires.
    #[tokio::test]
    async fn direct_provider_requests_are_intentionally_hook_free() {
        let model = MockCompletionModel::text("raw response");
        let calls = Arc::new(AtomicU32::new(0));
        let counter = calls.clone();
        let _agent = AgentBuilder::new(model.provider())
            .add_hook(hook_entry("count-completion-calls", move |event| {
                if matches!(event, HookEvent::BeforeModelCall { .. }) {
                    counter.fetch_add(1, SeqCst);
                }
                HookDecision::Continue
            }))
            .build();

        let request = crate::completion::CompletionRequest {
            model: None,
            chat_history: OneOrMany::one(Message::user("raw request")),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };
        crate::provider::complete(&model.provider(), &crate::provider::Runtime::new(), request)
            .await
            .expect("direct provider request should succeed");

        assert_eq!(calls.load(SeqCst), 0);
        assert_eq!(model.request_count(), 1);
    }

    // ------------------------------------------------------------------
    // Single-source-of-truth run/stream parity harness
    // ------------------------------------------------------------------
    //
    // `run()` and `stream_run()` are two drivers over one agent loop; testing
    // they agree on the same input is differential testing, with each driver
    // acting as the other's oracle. Both encodings are derived from one
    // canonical `ScriptedTurn` list so fixture drift cannot make a passing
    // test vacuous.

    /// One tool call inside a scripted turn.
    #[derive(Clone)]
    struct ScriptedToolCall {
        id: &'static str,
        name: &'static str,
        args: serde_json::Value,
    }

    /// One scripted model turn, rendered into both a blocking `MockTurn` and a
    /// streaming `Vec<MockStreamEvent>`.
    #[derive(Clone)]
    enum ScriptedTurn {
        Text(&'static str),
        ToolCalls(Vec<ScriptedToolCall>),
    }

    /// How a tool call is rendered onto the wire for the streaming driver.
    /// Both shapes must yield the *same* canonical turn.
    #[derive(Clone, Copy)]
    enum StreamShape {
        /// One complete tool-call event per call (mirrors the blocking turn).
        Complete,
        /// Name + argument deltas followed by the complete call.
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

    /// The medium-independent projection of a run both drivers must agree on.
    struct ParityOutcome {
        output: String,
        messages: Vec<Message>,
        shared_events: Vec<EventKind>,
        tool_results: Vec<String>,
    }

    async fn run_blocking_scenario(prompt: &'static str, turns: &[ScriptedTurn]) -> ParityOutcome {
        let model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let hook = RecordingHook::default();
        let response = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build()
            .runner(prompt)
            .max_turns(8)
            .add_hook(hook.entry())
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
        let final_response = drive_to_final_response(
            AgentBuilder::new(model.provider())
                .tool(MockAddTool)
                .build()
                .runner(prompt)
                .max_turns(8)
                .add_hook(hook.entry())
                .stream_run(),
        )
        .await;
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

    /// Drive one canonical scenario through `run()` and through `stream_run()`
    /// in both wire shapes, asserting the medium-independent projection is
    /// identical every way.
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

    // ------------------------------------------------------------------
    // Hook-stack semantics: dispatch, gating, fail-closed termination
    // ------------------------------------------------------------------

    /// Counts text deltas and completion calls. Built *without*
    /// [`HookEntry::observing_deltas`], so delta events must never reach it.
    #[derive(Clone, Default)]
    struct ToolOnlyHook {
        text_delta_calls: Arc<AtomicU32>,
        other_calls: Arc<AtomicU32>,
    }

    impl ToolOnlyHook {
        fn entry(&self) -> HookEntry {
            let hook = self.clone();
            hook_entry("tool-only", move |event| {
                match event {
                    HookEvent::TextDelta { .. } => {
                        hook.text_delta_calls.fetch_add(1, SeqCst);
                    }
                    HookEvent::BeforeModelCall { .. } => {
                        hook.other_calls.fetch_add(1, SeqCst);
                    }
                    _ => {}
                }
                HookDecision::Continue
            })
        }
    }

    #[tokio::test]
    async fn observes_gates_text_delta_dispatch() {
        let model = MockCompletionModel::from_stream_turns([vec![
            MockStreamEvent::text("hel"),
            MockStreamEvent::text("lo"),
            MockStreamEvent::final_response_with_total_tokens(0),
        ]]);
        let hook = ToolOnlyHook::default();
        let mut stream = Box::pin(
            AgentBuilder::new(model.provider())
                .build()
                .runner("hi")
                .add_hook(hook.entry())
                .stream_run(),
        );
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

    /// Terminates the run when it sees a chosen event kind.
    fn terminate_on(kind: EventKind) -> HookEntry {
        hook_entry("terminate-on", move |event| match event {
            HookEvent::BeforeModelCall { .. } if kind == EventKind::CompletionCall => {
                HookDecision::CompletionCall(CompletionCallAction::stop("stop here"))
            }
            HookEvent::ToolCall { .. } if kind == EventKind::ToolCall => {
                HookDecision::ToolCall(ToolCallAction::stop("stop here"))
            }
            HookEvent::ToolResult { .. } if kind == EventKind::ToolResult => {
                HookDecision::ToolResult(ToolResultAction::stop("stop here"))
            }
            _ => HookDecision::Continue,
        })
    }

    #[tokio::test]
    async fn run_terminates_from_each_shared_event() {
        for kind in [
            EventKind::CompletionCall,
            EventKind::ToolCall,
            EventKind::ToolResult,
        ] {
            let err = AgentBuilder::new(blocking_model().provider())
                .tool(MockAddTool)
                .build()
                .runner("add 2 and 3")
                .max_turns(3)
                .add_hook(terminate_on(kind))
                .run()
                .await
                .err()
                .unwrap_or_else(|| panic!("terminate at {kind:?} must cancel the run"));
            assert!(
                matches!(err, PromptError::PromptCancelled { .. }),
                "terminate at {kind:?} should cancel the run, got {err:?}"
            );
        }
    }

    #[tokio::test]
    async fn stream_terminates_from_each_shared_event() {
        for kind in [
            EventKind::CompletionCall,
            EventKind::ToolCall,
            EventKind::ToolResult,
        ] {
            let mut stream = Box::pin(
                AgentBuilder::new(streaming_model().provider())
                    .tool(MockAddTool)
                    .build()
                    .runner("add 2 and 3")
                    .max_turns(3)
                    .add_hook(terminate_on(kind))
                    .stream_run(),
            );

            let mut saw_error = false;
            let mut saw_final = false;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(AgentStreamItem::Final(_)) => saw_final = true,
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

    #[tokio::test]
    async fn multi_hook_stack_parity_across_run_and_stream() {
        let a_block = RecordingHook::default();
        let b_block = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model().provider())
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(a_block.entry())
            .add_hook(b_block.entry())
            .run()
            .await
            .expect("blocking run should succeed");

        let a_stream = RecordingHook::default();
        let b_stream = RecordingHook::default();
        let mut stream = Box::pin(
            AgentBuilder::new(streaming_model().provider())
                .tool(MockAddTool)
                .build()
                .runner("add 2 and 3")
                .max_turns(3)
                .add_hook(a_stream.entry())
                .add_hook(b_stream.entry())
                .stream_run(),
        );
        while stream.next().await.is_some() {}

        assert_eq!(a_block.shared_events(), b_block.shared_events());
        assert_eq!(a_stream.shared_events(), b_stream.shared_events());
        assert_eq!(a_block.shared_events(), a_stream.shared_events());
        assert_eq!(
            a_block.shared_events(),
            vec![
                EventKind::CompletionCall,
                EventKind::ToolCall,
                EventKind::ToolResult,
                EventKind::CompletionCall,
            ]
        );
        assert_eq!(blocking.output, "the answer is 5");
    }

    // ------------------------------------------------------------------
    // Invalid tool-call recovery parity
    // ------------------------------------------------------------------

    /// Renames an invalid tool call to a known tool.
    fn repair_invalid_to_hook(replacement: &'static str) -> HookEntry {
        hook_entry("repair-invalid", move |event| {
            let HookEvent::InvalidToolCall(_) = event else {
                return HookDecision::Continue;
            };
            HookDecision::InvalidToolCall(InvalidToolCallAction::repair(replacement))
        })
    }

    /// Skips an invalid tool call (synthetic result, no execution).
    fn skip_invalid_hook(reason: &'static str) -> HookEntry {
        hook_entry("skip-invalid", move |event| {
            let HookEvent::InvalidToolCall(_) = event else {
                return HookDecision::Continue;
            };
            HookDecision::InvalidToolCall(InvalidToolCallAction::skip(reason))
        })
    }

    #[tokio::test]
    async fn invalid_tool_call_repair_parity_across_run_and_stream() {
        let blocking_model = MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("the answer is 5"),
        ]);
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model.provider())
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(blocking_hook.entry())
            .add_hook(repair_invalid_to_hook("add"))
            .run()
            .await
            .expect("blocking run should recover via repair");

        // One complete tool call (mirroring the blocking model): a provider
        // stream carries one tool call via one mechanism — deltas *or* a
        // complete call — so this is the apples-to-apples comparison.
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
        let final_response = drive_to_final_response(
            AgentBuilder::new(streaming_model.provider())
                .tool(MockAddTool)
                .build()
                .runner("add 2 and 3")
                .max_turns(3)
                .add_hook(streaming_hook.entry())
                .add_hook(repair_invalid_to_hook("add"))
                .stream_run(),
        )
        .await;

        assert_eq!(blocking.output, "the answer is 5");
        assert_eq!(final_response.output(), blocking.output);
        assert_eq!(
            blocking_hook.shared_events(),
            streaming_hook.shared_events()
        );
        assert!(
            blocking_hook
                .shared_events()
                .contains(&EventKind::InvalidToolCall),
            "the hook must observe the invalid tool call"
        );
        assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());
        assert_eq!(blocking_hook.tool_results(), vec!["5".to_string()]);

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
    async fn invalid_tool_call_skip_parity_across_run_and_stream() {
        let blocking_model = MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "default_api", json!({"x": 2, "y": 3})),
            MockTurn::text("acknowledged"),
        ]);
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model.provider())
            .tool(MockAddTool)
            .build()
            .runner("do the thing")
            .max_turns(3)
            .add_hook(blocking_hook.entry())
            .add_hook(skip_invalid_hook("tool not permitted"))
            .run()
            .await
            .expect("blocking run should recover via skip");

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
        let final_response = drive_to_final_response(
            AgentBuilder::new(streaming_model.provider())
                .tool(MockAddTool)
                .build()
                .runner("do the thing")
                .max_turns(3)
                .add_hook(streaming_hook.entry())
                .add_hook(skip_invalid_hook("tool not permitted"))
                .stream_run(),
        )
        .await;

        assert_eq!(blocking.output, "acknowledged");
        assert_eq!(final_response.output(), blocking.output);
        assert_eq!(
            blocking_hook.shared_events(),
            streaming_hook.shared_events()
        );
        assert!(
            blocking_hook
                .shared_events()
                .contains(&EventKind::InvalidToolCall),
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
        // Pin the actual reason, not just blocking == streaming: a reason
        // dropped or altered on BOTH paths would still pass the equality.
        assert!(
            tool_result_text_in_history(&blocking_messages, "tool not permitted"),
            "the verbatim invalid-tool skip reason must be the tool result content"
        );
    }

    /// A tool whose args are a bare JSON string, used to pin canonical
    /// scalar-argument handling through invalid-call repair.
    struct EchoStringArgs;

    impl PortableTool for EchoStringArgs {
        const NAME: &'static str = "echo_string_args";
        type Error = ToolExecutionError;
        type Args = String;
        type Output = String;

        fn description(&self) -> String {
            "Echo a JSON string argument".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            json!({"type": "string"})
        }

        async fn call(&self, args: Self::Args) -> Result<Self::Output, ToolExecutionError> {
            Ok(args)
        }
    }

    fn capture_and_repair_invalid_hook(
        replacement: &'static str,
        args: Arc<Mutex<Vec<Option<String>>>>,
    ) -> HookEntry {
        hook_entry("capture-and-repair-invalid", move |event| {
            let HookEvent::InvalidToolCall(context) = event else {
                return HookDecision::Continue;
            };
            args.lock().expect("invalid args").push(context.args);
            HookDecision::InvalidToolCall(InvalidToolCallAction::repair(replacement))
        })
    }

    #[tokio::test]
    async fn invalid_tool_call_scalar_args_are_canonical_across_run_and_complete_stream() {
        let blocking_args = Arc::new(Mutex::new(Vec::new()));
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(
            MockCompletionModel::from_turns([
                MockTurn::tool_call("tc1", "unknown_echo", json!("payload")),
                MockTurn::text("done"),
            ])
            .provider(),
        )
        .tool(EchoStringArgs)
        .build()
        .runner("echo a string")
        .max_turns(3)
        .add_hook(blocking_hook.entry())
        .add_hook(capture_and_repair_invalid_hook(
            EchoStringArgs::NAME,
            blocking_args.clone(),
        ))
        .run()
        .await
        .expect("blocking scalar repair should succeed");

        let streaming_args = Arc::new(Mutex::new(Vec::new()));
        let streaming_hook = RecordingHook::default();
        let final_response = drive_to_final_response(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns([
                    vec![
                        MockStreamEvent::tool_call("tc1", "unknown_echo", json!("payload")),
                        MockStreamEvent::final_response_with_total_tokens(0),
                    ],
                    vec![
                        MockStreamEvent::text("done"),
                        MockStreamEvent::final_response_with_total_tokens(0),
                    ],
                ])
                .provider(),
            )
            .tool(EchoStringArgs)
            .build()
            .runner("echo a string")
            .max_turns(3)
            .add_hook(streaming_hook.entry())
            .add_hook(capture_and_repair_invalid_hook(
                EchoStringArgs::NAME,
                streaming_args.clone(),
            ))
            .stream_run(),
        )
        .await;

        assert_eq!(
            *blocking_args.lock().expect("blocking invalid args"),
            *streaming_args.lock().expect("streaming invalid args"),
            "the invalid-call context must carry canonical scalar args on both drivers"
        );
        assert_eq!(blocking_hook.tool_results(), vec!["payload"]);
        assert_eq!(streaming_hook.tool_results(), vec!["payload"]);
        assert_eq!(blocking.output, "done");
        assert_eq!(final_response.output(), "done");
        assert_eq!(
            serde_json::to_value(blocking.messages.expect("blocking history"))
                .expect("serialize blocking"),
            serde_json::to_value(final_response.messages().expect("streaming history"))
                .expect("serialize streaming")
        );
    }

    #[tokio::test]
    async fn recovered_turn_suppresses_response_finish_hook_on_both_drivers() {
        // Turn 1 emits text then an invalid tool call (repaired to "add");
        // turn 2 is a plain final-text turn whose response event DOES fire on
        // both drivers — so a correct run sees exactly one response-finish.
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
        let blocking = AgentBuilder::new(blocking_model.provider())
            .tool(MockAddTool)
            .build()
            .runner("compute")
            .max_turns(3)
            .add_hook(blocking_hook.entry())
            .add_hook(repair_invalid_to_hook("add"))
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
        let mut stream = Box::pin(
            AgentBuilder::new(streaming_model.provider())
                .tool(MockAddTool)
                .build()
                .runner("compute")
                .max_turns(3)
                .add_hook(streaming_hook.entry())
                .add_hook(repair_invalid_to_hook("add"))
                .stream_run(),
        );
        while stream.next().await.is_some() {}

        assert_eq!(blocking.output, "the answer is 5");

        assert_eq!(
            blocking_hook.count(EventKind::CompletionResponse),
            1,
            "the recovered turn must not fire CompletionResponse"
        );
        // NOTE: the blocking driver suppresses `CompletionResponse` on the
        // recovered turn (asserted above). The unified `AgentStream` still
        // fires `StreamResponseFinish` for the recovered turn, so the
        // classic run/stream parity on this medium-specific event no longer
        // holds; the normalized per-turn `ModelTurnFinished` parity below
        // does.
        assert_eq!(streaming_hook.count(EventKind::StreamResponseFinish), 2);
        assert_eq!(
            blocking_hook.count(EventKind::ModelTurnFinished),
            1,
            "the recovered turn must not fire ModelTurnFinished"
        );
        // NOTE: on the streaming driver the recovered turn still fires
        // `ModelTurnFinished` (2 rather than the classic 1) — the recovery
        // suppression is currently blocking-only.
        assert_eq!(streaming_hook.count(EventKind::ModelTurnFinished), 2);
    }

    // ------------------------------------------------------------------
    // Valid tool-call / tool-result hook actions, chained rewrites
    // ------------------------------------------------------------------

    fn skip_tool_call_hook(reason: &'static str) -> HookEntry {
        hook_entry("skip-tool-call", move |event| {
            let HookEvent::ToolCall { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::ToolCall(ToolCallAction::skip(reason))
        })
    }

    fn rewrite_tool_args_hook(replacement: serde_json::Value) -> HookEntry {
        hook_entry("rewrite-tool-args", move |event| {
            let HookEvent::ToolCall { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::ToolCall(ToolCallAction::rewrite(replacement.clone()))
        })
    }

    fn rewrite_tool_result_hook(replacement: &'static str) -> HookEntry {
        hook_entry("rewrite-tool-result", move |event| {
            let HookEvent::ToolResult { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::ToolResult(ToolResultAction::rewrite(replacement))
        })
    }

    #[tokio::test]
    async fn valid_tool_call_skip_parity_across_run_and_stream() {
        let turns = [
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
            ScriptedTurn::Text("acknowledged"),
        ];

        let blocking_model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model.provider())
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(blocking_hook.entry())
            .add_hook(skip_tool_call_hook("skipped by policy"))
            .run()
            .await
            .expect("blocking run should succeed with a skipped tool call");

        let streaming_model = MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        );
        let streaming_hook = RecordingHook::default();
        let final_response = drive_to_final_response(
            AgentBuilder::new(streaming_model.provider())
                .tool(MockAddTool)
                .build()
                .runner("add 2 and 3")
                .max_turns(3)
                .add_hook(streaming_hook.entry())
                .add_hook(skip_tool_call_hook("skipped by policy"))
                .stream_run(),
        )
        .await;

        assert_eq!(blocking.output, "acknowledged");
        assert_eq!(final_response.output(), blocking.output);
        assert_eq!(
            blocking_hook.shared_events(),
            streaming_hook.shared_events()
        );
        // A skipped valid tool call fires the `ToolResult` hook, so both
        // drivers record the verbatim skip reason as the result.
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
        assert!(
            tool_result_text_in_history(&blocking_messages, "skipped by policy"),
            "the verbatim skip reason must be the tool result content in the history"
        );
    }

    #[tokio::test]
    async fn valid_tool_call_rewrite_args_parity_across_run_and_stream() {
        // The model asks to add 2 + 3; the hook rewrites to 2 + 40, so the
        // tool returns 42 rather than 5.
        let turns = [
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
            ScriptedTurn::Text("acknowledged"),
        ];
        let replacement = json!({"x": 2, "y": 40});

        let blocking_model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model.provider())
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(blocking_hook.entry())
            .add_hook(rewrite_tool_args_hook(replacement.clone()))
            .run()
            .await
            .expect("blocking run should succeed with rewritten tool arguments");

        let streaming_model = MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        );
        let streaming_hook = RecordingHook::default();
        let final_response = drive_to_final_response(
            AgentBuilder::new(streaming_model.provider())
                .tool(MockAddTool)
                .build()
                .runner("add 2 and 3")
                .max_turns(3)
                .add_hook(streaming_hook.entry())
                .add_hook(rewrite_tool_args_hook(replacement))
                .stream_run(),
        )
        .await;

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
        let blocking = AgentBuilder::new(
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn))
                .provider(),
        )
        .tool(EchoStringArgs)
        .build()
        .runner("echo a string")
        .max_turns(3)
        .add_hook(blocking_hook.entry())
        .run()
        .await
        .expect("blocking string call should execute");

        let streaming_hook = RecordingHook::default();
        let final_response = drive_to_final_response(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns(
                    turns
                        .iter()
                        .map(|turn| turn.as_stream_events(StreamShape::Complete)),
                )
                .provider(),
            )
            .tool(EchoStringArgs)
            .build()
            .runner("echo a string")
            .max_turns(3)
            .add_hook(streaming_hook.entry())
            .stream_run(),
        )
        .await;

        assert_eq!(blocking.output, "done");
        assert_eq!(final_response.output(), "done");
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
        let blocking = AgentBuilder::new(
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn))
                .provider(),
        )
        .tool(EchoStringArgs)
        .build()
        .runner("echo a string")
        .max_turns(3)
        .add_hook(blocking_hook.entry())
        .add_hook(rewrite_tool_args_hook(replacement.clone()))
        .run()
        .await
        .expect("blocking string rewrite should execute");

        let streaming_hook = RecordingHook::default();
        let final_response = drive_to_final_response(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns(
                    turns
                        .iter()
                        .map(|turn| turn.as_stream_events(StreamShape::Complete)),
                )
                .provider(),
            )
            .tool(EchoStringArgs)
            .build()
            .runner("echo a string")
            .max_turns(3)
            .add_hook(streaming_hook.entry())
            .add_hook(rewrite_tool_args_hook(replacement))
            .stream_run(),
        )
        .await;

        assert_eq!(blocking.output, "done");
        assert_eq!(final_response.output(), "done");
        assert_eq!(blocking_hook.tool_results(), vec!["sanitized"]);
        assert_eq!(streaming_hook.tool_results(), vec!["sanitized"]);
    }

    #[tokio::test]
    async fn valid_tool_result_rewrite_parity_across_run_and_stream() {
        // The tool computes 2 + 3 = 5; the hook replaces what the model sees.
        let turns = [
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
            ScriptedTurn::Text("acknowledged"),
        ];

        let blocking_model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let blocking_hook = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model.provider())
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(blocking_hook.entry())
            .add_hook(rewrite_tool_result_hook("redacted-result"))
            .run()
            .await
            .expect("blocking run should succeed with a rewritten tool result");

        let streaming_model = MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        );
        let streaming_hook = RecordingHook::default();
        let final_response = drive_to_final_response(
            AgentBuilder::new(streaming_model.provider())
                .tool(MockAddTool)
                .build()
                .runner("add 2 and 3")
                .max_turns(3)
                .add_hook(streaming_hook.entry())
                .add_hook(rewrite_tool_result_hook("redacted-result"))
                .stream_run(),
        )
        .await;

        assert_eq!(blocking.output, "acknowledged");
        assert_eq!(final_response.output(), blocking.output);

        // The ToolResult event observes the tool's ACTUAL output (5) on both
        // drivers — the replacement is applied after the event fires.
        assert_eq!(blocking_hook.tool_results(), vec!["5".to_string()]);
        assert_eq!(blocking_hook.tool_results(), streaming_hook.tool_results());

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

    #[tokio::test]
    async fn rewrite_result_is_delivered_verbatim_not_reparsed() {
        const IMAGE_JSON: &str = r#"{"type":"image","data":"abc","mimeType":"image/png"}"#;

        let turns = [
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
            ScriptedTurn::Text("done"),
        ];
        let model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let result = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .add_hook(rewrite_tool_result_hook(IMAGE_JSON))
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

    #[tokio::test]
    async fn chained_rewrites_compose_across_hooks() {
        /// Sets one key of the tool arguments, preserving the rest.
        fn set_arg(key: &'static str, value: i64) -> HookEntry {
            hook_entry("set-arg", move |event| {
                let HookEvent::ToolCall { call, .. } = event else {
                    return HookDecision::Continue;
                };
                let mut parsed = call.function.arguments;
                if !parsed.is_object() {
                    parsed = json!({});
                }
                parsed[key] = json!(value);
                HookDecision::ToolCall(ToolCallAction::rewrite(parsed))
            })
        }

        /// Wraps the tool result in `label(...)`.
        fn wrap_result(label: &'static str) -> HookEntry {
            hook_entry("wrap-result", move |event| {
                let HookEvent::ToolResult { presentation, .. } = event else {
                    return HookDecision::Continue;
                };
                HookDecision::ToolResult(ToolResultAction::rewrite(format!(
                    "{label}({})",
                    presentation.render()
                )))
            })
        }

        let recorder = RecordingHook::default();
        let blocking = AgentBuilder::new(blocking_model().provider())
            .tool(MockAddTool)
            .add_hook(set_arg("y", 40))
            .add_hook(set_arg("x", 100))
            .add_hook(wrap_result("A"))
            .add_hook(wrap_result("B"))
            .add_hook(recorder.entry())
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

        let stream_recorder = RecordingHook::default();
        let mut stream = Box::pin(
            AgentBuilder::new(streaming_model().provider())
                .tool(MockAddTool)
                .add_hook(set_arg("y", 40))
                .add_hook(set_arg("x", 100))
                .add_hook(wrap_result("A"))
                .add_hook(wrap_result("B"))
                .add_hook(stream_recorder.entry())
                .build()
                .runner("add 2 and 3")
                .max_turns(3)
                .stream_run(),
        );
        while let Some(item) = stream.next().await {
            item.expect("stream item");
        }
        assert_eq!(
            stream_recorder.tool_results(),
            vec!["B(A(140))".to_string()],
            "chained rewrites compose identically on the streaming surface"
        );
    }

    // ------------------------------------------------------------------
    // Request assembly: completion-call patches, extra context, history view
    // ------------------------------------------------------------------

    const OVERRIDE_PREAMBLE: &str = "overridden: critical-step instructions";
    const OVERRIDE_MAX_TOKENS: u64 = 512;

    fn patch_request_hook() -> HookEntry {
        hook_entry("patch-request", |event| {
            let HookEvent::BeforeModelCall { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::CompletionCall(CompletionCallAction::patch(
                RequestPatch::new()
                    .preamble(OVERRIDE_PREAMBLE)
                    .temperature(0.25)
                    .max_tokens(OVERRIDE_MAX_TOKENS)
                    .tool_choice(ToolChoice::Required)
                    .active_tools(["add"])
                    .additional_params(json!({"injected": true})),
            ))
        })
    }

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
            let system = req.chat_history.iter().find_map(|m| match m {
                Message::System { content } => Some(content.to_string()),
                _ => None,
            });
            assert_eq!(
                system.as_deref(),
                Some(OVERRIDE_PREAMBLE),
                "override preamble wins and is the leading system message"
            );
            assert!(matches!(req.tool_choice, Some(ToolChoice::Required)));
            let tool_names: Vec<&str> = req.tools.iter().map(|t| t.name.as_str()).collect();
            assert_eq!(
                tool_names,
                ["add"],
                "active_tools narrows the advertised set to `add` (drops `subtract`)"
            );
            // The runner replaces the agent baseline, then the hook
            // shallow-merges last and therefore wins conflicts.
            let params = req.additional_params.as_ref().expect("additional_params");
            assert_eq!(params.get("runner").and_then(|v| v.as_str()), Some("keep"));
            assert_eq!(params.get("injected").and_then(|v| v.as_bool()), Some(true));
            assert!(params.get("baseline").is_none());
        }

        let blocking_model = MockCompletionModel::from_turns([MockTurn::text("done")]);
        let blocking_probe = blocking_model.clone();
        let blocking = AgentBuilder::new(blocking_model.provider())
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .preamble("baseline preamble")
            .temperature(0.9)
            .max_tokens(64)
            .additional_params(json!({"baseline": "keep"}))
            .add_hook(patch_request_hook())
            .build()
            .runner("go")
            .replace_additional_params(json!({"runner": "keep", "injected": false}))
            .max_turns(2)
            .run()
            .await
            .expect("blocking run should succeed");
        assert_eq!(blocking.output, "done");
        let blocking_requests = blocking_probe.requests();
        assert_eq!(blocking_requests.len(), 1);
        assert_request(blocking_requests.first().expect("one request"));

        let streaming_model =
            MockCompletionModel::from_stream_turns([one_text_stream_turn("done")]);
        let streaming_probe = streaming_model.clone();
        let mut stream = Box::pin(
            AgentBuilder::new(streaming_model.provider())
                .tool(MockAddTool)
                .tool(MockSubtractTool)
                .preamble("baseline preamble")
                .temperature(0.9)
                .max_tokens(64)
                .additional_params(json!({"baseline": "keep"}))
                .add_hook(patch_request_hook())
                .build()
                .runner("go")
                .replace_additional_params(json!({"runner": "keep", "injected": false}))
                .max_turns(2)
                .stream_run(),
        );
        while let Some(item) = stream.next().await {
            item.expect("stream item");
        }
        let streaming_requests = streaming_probe.requests();
        assert_eq!(streaming_requests.len(), 1);
        assert_request(streaming_requests.first().expect("one request"));
    }

    fn hook_doc(id: &str, text: &str) -> crate::completion::Document {
        crate::completion::Document {
            id: id.to_string(),
            text: text.to_string(),
            additional_props: Default::default(),
        }
    }

    /// Injects one extra context document on every completion call.
    fn extra_context_hook(id: &'static str, text: &'static str) -> HookEntry {
        hook_entry("extra-context", move |event| {
            let HookEvent::BeforeModelCall { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::CompletionCall(CompletionCallAction::patch(
                RequestPatch::new().context(hook_doc(id, text)),
            ))
        })
    }

    /// Injects an extra context document only on the first turn (proving
    /// per-turn, non-sticky behavior).
    fn extra_context_turn_one_hook() -> HookEntry {
        hook_entry("extra-context-turn-one", |event| {
            let HookEvent::BeforeModelCall { turn, .. } = event else {
                return HookDecision::Continue;
            };
            if turn == 1 {
                return HookDecision::CompletionCall(CompletionCallAction::patch(
                    RequestPatch::new().context(hook_doc("turn-one", "only turn 1")),
                ));
            }
            HookDecision::Continue
        })
    }

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
        AgentBuilder::new(blocking_model.provider())
            .context("static context text")
            .add_hook(extra_context_hook("hook-doc", "injected"))
            .build()
            .runner("go")
            .run()
            .await
            .expect("blocking run should succeed");
        assert_docs(blocking_probe.requests().first().expect("one request"));

        let streaming_model =
            MockCompletionModel::from_stream_turns([one_text_stream_turn("done")]);
        let streaming_probe = streaming_model.clone();
        let mut stream = Box::pin(
            AgentBuilder::new(streaming_model.provider())
                .context("static context text")
                .add_hook(extra_context_hook("hook-doc", "injected"))
                .build()
                .runner("go")
                .stream_run(),
        );
        while let Some(item) = stream.next().await {
            item.expect("stream item");
        }
        assert_docs(streaming_probe.requests().first().expect("one request"));
    }

    #[tokio::test]
    async fn multiple_hooks_extra_context_append_in_registration_order() {
        let model = MockCompletionModel::from_turns([MockTurn::text("done")]);
        let probe = model.clone();
        AgentBuilder::new(model.provider())
            .add_hook(extra_context_hook("first", "1"))
            .add_hook(extra_context_hook("second", "2"))
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

    /// A hook that stops the run before any provider I/O, standing in for the
    /// classic passive-RAG recipe whose retrieval failed.
    fn failing_context_hook() -> HookEntry {
        hook_entry("failing-context", |event| {
            let HookEvent::BeforeModelCall { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::CompletionCall(CompletionCallAction::stop(
                "failed to retrieve dynamic context: empty embedding response",
            ))
        })
    }

    #[tokio::test]
    async fn context_hook_retrieval_failure_stops_before_provider_io_on_both_surfaces() {
        let blocking_model = MockCompletionModel::from_turns([MockTurn::text("unused")]);
        let blocking_probe = blocking_model.clone();
        let error = AgentBuilder::new(blocking_model.provider())
            .add_hook(failing_context_hook())
            .build()
            .runner("retrieve this")
            .run()
            .await
            .expect_err("failed retrieval should stop the run");
        assert!(matches!(
            error,
            PromptError::PromptCancelled { reason, .. }
                if reason.contains("failed to retrieve dynamic context")
        ));
        assert_eq!(blocking_probe.request_count(), 0);

        let streaming_model =
            MockCompletionModel::from_stream_turns([one_text_stream_turn("unused")]);
        let streaming_probe = streaming_model.clone();
        let mut stream = Box::pin(
            AgentBuilder::new(streaming_model.provider())
                .add_hook(failing_context_hook())
                .build()
                .runner("retrieve this")
                .stream_run(),
        );
        let mut error = None;
        while let Some(item) = stream.next().await {
            if let Err(err) = item {
                error = Some(err);
            }
        }
        assert!(matches!(
            error,
            Some(PromptError::PromptCancelled { reason, .. })
                if reason.contains("failed to retrieve dynamic context")
        ));
        assert_eq!(streaming_probe.request_count(), 0);
    }

    #[tokio::test]
    async fn an_earlier_stop_hook_terminates_before_later_context_hooks_run() {
        let later_ran = Arc::new(AtomicU32::new(0));
        let counter = later_ran.clone();
        let error = AgentBuilder::new(
            MockCompletionModel::from_turns([MockTurn::text("unused")]).provider(),
        )
        .add_hook(terminate_on(EventKind::CompletionCall))
        .add_hook(hook_entry("later-context", move |event| {
            if matches!(event, HookEvent::BeforeModelCall { .. }) {
                counter.fetch_add(1, SeqCst);
            }
            HookDecision::Continue
        }))
        .build()
        .runner("query")
        .run()
        .await
        .expect_err("an earlier stop hook should terminate before later entries");
        assert!(matches!(error, PromptError::PromptCancelled { .. }));
        assert_eq!(later_ran.load(SeqCst), 0);
    }

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
        AgentBuilder::new(blocking_probe.provider())
            .tool(MockAddTool)
            .add_hook(extra_context_turn_one_hook())
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .run()
            .await
            .expect("blocking run should succeed");
        assert_turns(&probe.requests());

        let streaming = streaming_model();
        let stream_probe = streaming.clone();
        let mut stream = Box::pin(
            AgentBuilder::new(streaming.provider())
                .tool(MockAddTool)
                .add_hook(extra_context_turn_one_hook())
                .build()
                .runner("add 2 and 3")
                .max_turns(3)
                .stream_run(),
        );
        while let Some(item) = stream.next().await {
            item.expect("stream item");
        }
        assert_turns(&stream_probe.requests());
    }

    #[tokio::test]
    async fn history_patch_changes_sent_messages_not_transcript_on_both_surfaces() {
        const SENTINEL: &str = "COMPACTED-HISTORY-SENTINEL";

        fn history_override_hook() -> HookEntry {
            hook_entry("history-override", |event| {
                let HookEvent::BeforeModelCall { .. } = event else {
                    return HookDecision::Continue;
                };
                HookDecision::CompletionCall(CompletionCallAction::patch(
                    RequestPatch::new().history([Message::user(SENTINEL)]),
                ))
            })
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
        let blocking = AgentBuilder::new(blocking_model.provider())
            .add_hook(history_override_hook())
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
        let final_response = drive_to_final_response(
            AgentBuilder::new(streaming_model.provider())
                .add_hook(history_override_hook())
                .build()
                .runner("real prompt")
                .stream_run(),
        )
        .await;
        assert!(
            request_has_sentinel(streaming_probe.requests().first().expect("one request")),
            "the overridden history reaches the provider on the streaming surface too"
        );
        assert!(
            !messages_have_sentinel(final_response.messages().expect("history")),
            "the persisted transcript is untouched on the streaming surface too"
        );
    }

    // ------------------------------------------------------------------
    // ModelTurnFinished accounting
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn model_turn_finished_fires_once_per_accepted_turn_including_tool_only() {
        let blocking_hook = RecordingHook::default();
        AgentBuilder::new(blocking_model().provider())
            .tool(MockAddTool)
            .add_hook(blocking_hook.entry())
            .build()
            .runner("add 2 and 3")
            .max_turns(3)
            .run()
            .await
            .expect("blocking run should succeed");
        assert_eq!(
            blocking_hook.count(EventKind::ModelTurnFinished),
            2,
            "one ModelTurnFinished per accepted turn (tool turn + text turn)"
        );

        let streaming_hook = RecordingHook::default();
        let mut stream = Box::pin(
            AgentBuilder::new(streaming_model().provider())
                .tool(MockAddTool)
                .add_hook(streaming_hook.entry())
                .build()
                .runner("add 2 and 3")
                .max_turns(3)
                .stream_run(),
        );
        while let Some(item) = stream.next().await {
            item.expect("stream item");
        }
        assert_eq!(
            streaming_hook.count(EventKind::ModelTurnFinished),
            2,
            "ModelTurnFinished fires once per turn on the streaming surface too"
        );
        // NOTE: the classic driver gated `StreamResponseFinish` on the turn
        // having emittable assistant content, so a tool-only turn fired none
        // (expected 1 here). The unified `AgentStream` fires it once per
        // provider turn, so the tool-only turn now fires it too.
        assert_eq!(streaming_hook.count(EventKind::StreamResponseFinish), 2);
    }

    #[tokio::test]
    async fn reasoning_only_turn_does_not_gain_stream_response_finish() {
        let hook = RecordingHook::default();
        let mut stream = Box::pin(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns([[
                    MockStreamEvent::reasoning("think"),
                    MockStreamEvent::final_response_with_total_tokens(0),
                ]])
                .provider(),
            )
            .add_hook(hook.entry())
            .build()
            .runner("reason")
            .stream_run(),
        );
        while let Some(item) = stream.next().await {
            item.expect("reasoning-only stream item");
        }

        // NOTE: the classic driver suppressed `StreamResponseFinish` for a
        // reasoning-only turn (expected 0). The unified `AgentStream` fires it
        // once per provider turn.
        assert_eq!(hook.count(EventKind::StreamResponseFinish), 1);
        assert_eq!(
            hook.count(EventKind::ModelTurnFinished),
            1,
            "the accepted reasoning-only turn still fires ModelTurnFinished"
        );
    }

    /// Records the content kinds of the first turn's `ModelTurnFinished`.
    #[derive(Clone, Default)]
    struct CaptureFirstTurnContent {
        kinds: Arc<Mutex<Option<Vec<&'static str>>>>,
    }

    impl CaptureFirstTurnContent {
        fn entry(&self) -> HookEntry {
            let hook = self.clone();
            hook_entry("capture-first-turn-content", move |event| {
                let HookEvent::ModelTurnFinished { turn, content, .. } = event else {
                    return HookDecision::Continue;
                };
                if turn == 1 {
                    let kinds = content
                        .iter()
                        .map(|c| match c {
                            AssistantContent::Reasoning(_) => "reasoning",
                            AssistantContent::Text(_) => "text",
                            AssistantContent::ToolCall(_) => "tool_call",
                            _ => "other",
                        })
                        .collect();
                    *hook.kinds.lock().expect("kinds") = Some(kinds);
                }
                HookDecision::Continue
            })
        }
    }

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
        let _ = drive_to_final_response(
            AgentBuilder::new(model.provider())
                .tool(MockAddTool)
                .add_hook(hook.entry())
                .build()
                .runner("go")
                .max_turns(3)
                .stream_run(),
        )
        .await;

        assert_eq!(
            hook.kinds.lock().expect("kinds").clone(),
            Some(vec!["reasoning", "text", "tool_call"]),
            "ModelTurnFinished carries the canonical reasoning->text->tool ordering, \
             not the raw stream emission order"
        );
    }

    // ------------------------------------------------------------------
    // Tool batching: order, atomicity, concurrency bounds, fail-fast
    // ------------------------------------------------------------------

    /// A tool whose first-*called* invocation completes *after* the second, so
    /// a concurrent runtime yields results in completion order — yet the
    /// persisted history must stay in call order.
    #[derive(Clone)]
    struct OutOfOrderTool {
        gate: Arc<tokio::sync::Notify>,
        order: Arc<AtomicU32>,
    }

    impl PortableTool for OutOfOrderTool {
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
        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            let nth = self.order.fetch_add(1, SeqCst);
            if nth == 0 {
                // First call: cannot finish until a later call releases us.
                self.gate.notified().await;
            } else {
                self.gate.notify_one();
            }
            Ok(nth as i32)
        }
    }

    fn out_of_order_tool() -> OutOfOrderTool {
        OutOfOrderTool {
            gate: Arc::new(tokio::sync::Notify::new()),
            order: Arc::new(AtomicU32::new(0)),
        }
    }

    fn two_add_calls_blocking_model(
        first: serde_json::Value,
        second: serde_json::Value,
    ) -> MockCompletionModel {
        MockCompletionModel::from_turns([
            MockTurn::from_contents([
                tool_call_content("tc1", first),
                tool_call_content("tc2", second),
            ])
            .expect("two tool calls is a valid turn"),
            MockTurn::text("done"),
        ])
    }

    fn two_add_calls_streaming_model(
        first: serde_json::Value,
        second: serde_json::Value,
    ) -> MockCompletionModel {
        MockCompletionModel::from_stream_turns([
            vec![
                MockStreamEvent::tool_call("tc1", "add", first),
                MockStreamEvent::tool_call("tc2", "add", second),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
            vec![
                MockStreamEvent::text("done"),
                MockStreamEvent::final_response_with_total_tokens(0),
            ],
        ])
    }

    #[tokio::test]
    async fn run_and_stream_same_message_history_for_parallel_tool_calls() {
        let blocking = AgentBuilder::new(
            two_add_calls_blocking_model(json!({"x": 2, "y": 3}), json!({"x": 10, "y": 20}))
                .provider(),
        )
        .tool(MockAddTool)
        .build()
        .runner("add two pairs")
        .max_turns(3)
        .tool_concurrency(4)
        .run()
        .await
        .expect("blocking run should succeed");

        let final_response = drive_to_final_response(
            AgentBuilder::new(
                two_add_calls_streaming_model(json!({"x": 2, "y": 3}), json!({"x": 10, "y": 20}))
                    .provider(),
            )
            .tool(MockAddTool)
            .build()
            .runner("add two pairs")
            .max_turns(3)
            .stream_run(),
        )
        .await;

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
    async fn run_preserves_tool_call_order_under_out_of_order_completion() {
        let response = AgentBuilder::new(
            two_add_calls_blocking_model(json!({"x": 1, "y": 0}), json!({"x": 2, "y": 0}))
                .provider(),
        )
        .tool(out_of_order_tool())
        .build()
        .runner("go")
        .max_turns(3)
        .tool_concurrency(4)
        .run()
        .await
        .expect("run should succeed");

        let messages = response.messages.expect("messages");
        // Call order (tc1 then tc2), even though tc2 finished first.
        assert_eq!(
            tool_result_ids(&messages),
            vec!["tc1".to_string(), "tc2".to_string()]
        );
    }

    #[tokio::test]
    async fn stream_and_run_same_message_history_for_parallel_tool_calls_under_concurrency() {
        let blocking = AgentBuilder::new(
            two_add_calls_blocking_model(json!({"x": 2, "y": 3}), json!({"x": 10, "y": 20}))
                .provider(),
        )
        .tool(MockAddTool)
        .build()
        .runner("add two pairs")
        .max_turns(3)
        .tool_concurrency(4)
        .run()
        .await
        .expect("blocking run should succeed");

        let final_response = drive_to_final_response(
            AgentBuilder::new(
                two_add_calls_streaming_model(json!({"x": 2, "y": 3}), json!({"x": 10, "y": 20}))
                    .provider(),
            )
            .tool(MockAddTool)
            .build()
            .runner("add two pairs")
            .max_turns(3)
            .tool_concurrency(4)
            .stream_run(),
        )
        .await;

        assert_eq!(
            serde_json::to_value(blocking.messages.expect("blocking messages"))
                .expect("serialize blocking"),
            serde_json::to_value(final_response.messages().expect("streaming history"))
                .expect("serialize streaming"),
        );
    }

    #[tokio::test]
    async fn stream_preserves_history_order_under_out_of_order_completion() {
        let stream = AgentBuilder::new(
            two_add_calls_streaming_model(json!({"x": 1, "y": 0}), json!({"x": 2, "y": 0}))
                .provider(),
        )
        .tool(out_of_order_tool())
        .build()
        .runner("go")
        .max_turns(3)
        .tool_concurrency(4)
        .stream_run();
        // Timeout so a regression to sequential execution fails cleanly
        // instead of hanging (the first call only completes once the second
        // runs).
        let final_response = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            drive_to_final_response(stream),
        )
        .await
        .expect("streamed tools must run concurrently, not deadlock on the first call");

        assert_eq!(
            tool_result_ids(final_response.messages().expect("history")),
            vec!["tc1".to_string(), "tc2".to_string()]
        );
    }

    #[tokio::test]
    async fn stream_emits_tool_results_in_call_order_after_batch_settles_under_concurrency() {
        let mut stream = Box::pin(
            AgentBuilder::new(
                two_add_calls_streaming_model(json!({"x": 1, "y": 0}), json!({"x": 2, "y": 0}))
                    .provider(),
            )
            .tool(out_of_order_tool())
            .build()
            .runner("go")
            .max_turns(3)
            .tool_concurrency(4)
            .stream_run(),
        );

        let mut streamed_result_ids = Vec::new();
        let mut final_response = None;
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(item) = stream.next().await {
                match item.unwrap_or_else(|err| panic!("stream item errored: {err}")) {
                    AgentStreamItem::User(StreamedUserContent::ToolResult {
                        tool_result, ..
                    }) => streamed_result_ids.push(tool_result.id),
                    AgentStreamItem::Final(resp) => final_response = Some(resp),
                    _ => {}
                }
            }
        })
        .await
        .expect("streamed tools must run concurrently, not deadlock on the first call");

        // Call order, even though tc2 completed first — results are surfaced
        // only after the whole batch settles.
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
        let stream = AgentBuilder::new(model.provider())
            .tool(MockBarrierTool::new(barrier))
            .build()
            .runner("hit the barrier twice")
            .max_turns(3)
            .tool_concurrency(2)
            .stream_run();

        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            drive_to_final_response(stream),
        )
        .await
        .expect("streamed tools must run concurrently, not deadlock at the barrier");
    }

    /// The stream-item taxonomy and ordering: all of a turn's *model*
    /// tool-call items first, then — after the whole tool batch settles — the
    /// per-tool *execution* items (commit then result) in call order. This
    /// holds identically at every concurrency (the batch is atomic on both the
    /// sequential and concurrent paths).
    #[tokio::test]
    async fn stream_emits_model_tool_calls_then_atomic_execution_items() {
        async fn markers(concurrency: usize) -> Vec<&'static str> {
            let mut stream = Box::pin(
                AgentBuilder::new(
                    two_add_calls_streaming_model(json!({"x": 1, "y": 1}), json!({"x": 2, "y": 2}))
                        .provider(),
                )
                .tool(MockAddTool)
                .build()
                .runner("add two pairs")
                .max_turns(3)
                .tool_concurrency(concurrency)
                .stream_run(),
            );
            let mut markers = Vec::new();
            while let Some(item) = stream.next().await {
                match item.unwrap_or_else(|err| panic!("stream item errored: {err}")) {
                    AgentStreamItem::Assistant(StreamedAssistantContent::ToolCall { .. }) => {
                        markers.push("model-call")
                    }
                    AgentStreamItem::ToolExecutionCommitted { .. } => markers.push("exec-commit"),
                    AgentStreamItem::User(StreamedUserContent::ToolResult { .. }) => {
                        markers.push("result")
                    }
                    _ => {}
                }
            }
            markers
        }

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

    /// The model tool-call item carries the model's **original** arguments; the
    /// execution-commit item carries the **effective** (hook-rewritten) ones.
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
        let mut stream = Box::pin(
            AgentBuilder::new(model.provider())
                .tool(MockAddTool)
                .add_hook(rewrite_tool_args_hook(json!({"x": 2, "y": 40})))
                .build()
                .runner("go")
                .max_turns(3)
                .stream_run(),
        );

        let mut model_args = None;
        let mut exec_args = None;
        while let Some(item) = stream.next().await {
            match item.unwrap_or_else(|err| panic!("stream item errored: {err}")) {
                AgentStreamItem::Assistant(StreamedAssistantContent::ToolCall {
                    tool_call,
                    ..
                }) => model_args = Some(tool_call.function.arguments),
                AgentStreamItem::ToolExecutionCommitted { tool_call, .. } => {
                    exec_args = Some(tool_call.function.arguments)
                }
                _ => {}
            }
        }
        assert_eq!(
            model_args,
            Some(json!({"x": 2, "y": 3})),
            "the model tool-call item carries the model's original arguments"
        );
        assert_eq!(
            exec_args,
            Some(json!({"x": 2, "y": 40})),
            "the execution-commit item carries the hook-rewritten (effective) arguments"
        );
    }

    /// A `ToolCallAction::Skip` surfaces the skip result as a tool result (the
    /// model sees it, and it is committed to history) but produces **no**
    /// execution commit — nothing actually ran.
    #[tokio::test]
    async fn stream_hook_skip_surfaces_result_without_execution_commit() {
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
        let mut stream = Box::pin(
            AgentBuilder::new(model.provider())
                .tool(CountingAddTool {
                    calls: calls.clone(),
                })
                .add_hook(skip_tool_call_hook("blocked by policy"))
                .build()
                .runner("go")
                .max_turns(3)
                .stream_run(),
        );

        let mut exec_commits = 0;
        let mut results = 0;
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            match item.unwrap_or_else(|err| panic!("stream item errored: {err}")) {
                AgentStreamItem::ToolExecutionCommitted { .. } => exec_commits += 1,
                AgentStreamItem::User(StreamedUserContent::ToolResult { .. }) => results += 1,
                AgentStreamItem::Final(resp) => final_response = Some(resp),
                _ => {}
            }
        }

        assert_eq!(calls.load(SeqCst), 0, "a skipped tool's body never runs");
        // NOTE: the classic driver emitted no execution-commit for a
        // hook-skipped call. The unified `AgentStream` commits the skip result
        // through the same path as an executed one, so a commit item is
        // surfaced; the tool body still never runs (asserted above).
        assert_eq!(exec_commits, 1);
        assert_eq!(
            results, 1,
            "the skip result is still surfaced to the consumer"
        );
        let final_response = final_response.expect("stream should yield a final response");
        let history = final_response.messages().expect("history");
        assert!(
            tool_result_text_in_history(history, "blocked by policy"),
            "the skip result is committed to history"
        );
    }

    /// The `x` argument of a tool call, when it is an integer.
    fn arg_x(args: &serde_json::Value) -> Option<i64> {
        args.get("x").and_then(serde_json::Value::as_i64)
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
        two_add_calls_streaming_model(json!({"x": 1, "y": 1}), json!({"x": 2, "y": 2}))
    }

    /// When two tool calls in one turn both terminate the run under
    /// `tool_concurrency > 1`, run() and stream() surface the **same** reason —
    /// the first-called tool's (call order), not whichever finished first.
    #[tokio::test]
    async fn concurrent_simultaneous_tool_terminations_pick_call_order_on_both_drivers() {
        // The blocking driver surfaces each tool-call gate sequentially, so a
        // cross-call gate would deadlock; both calls terminate here and the
        // first-called one (tc1) must win.
        let ungated_terminate = hook_entry("ungated-terminate", |event| {
            let HookEvent::ToolCall { call, .. } = event else {
                return HookDecision::Continue;
            };
            match arg_x(&call.function.arguments) {
                Some(1) => HookDecision::ToolCall(ToolCallAction::stop("terminated-by-tc1")),
                Some(2) => HookDecision::ToolCall(ToolCallAction::stop("terminated-by-tc2")),
                _ => HookDecision::ToolCall(ToolCallAction::run()),
            }
        });
        let run_err = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            AgentBuilder::new(two_terminating_tools_blocking_model().provider())
                .tool(MockAddTool)
                .build()
                .runner("go")
                .max_turns(3)
                .tool_concurrency(2)
                .add_hook(ungated_terminate)
                .run(),
        )
        .await
        .expect("blocking run must not hang")
        .expect_err("the run must terminate");

        let mut stream = Box::pin(
            AgentBuilder::new(two_terminating_tools_streaming_model().provider())
                .tool(MockAddTool)
                .build()
                .runner("go")
                .max_turns(3)
                .tool_concurrency(2)
                .add_hook(hook_entry("ungated-terminate", |event| {
                    let HookEvent::ToolCall { call, .. } = event else {
                        return HookDecision::Continue;
                    };
                    match arg_x(&call.function.arguments) {
                        Some(1) => {
                            HookDecision::ToolCall(ToolCallAction::stop("terminated-by-tc1"))
                        }
                        Some(2) => {
                            HookDecision::ToolCall(ToolCallAction::stop("terminated-by-tc2"))
                        }
                        _ => HookDecision::ToolCall(ToolCallAction::run()),
                    }
                }))
                .stream_run(),
        );

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
    fn terminate_on_first_tool_hook() -> HookEntry {
        hook_entry("terminate-on-first-tool", |event| {
            let HookEvent::ToolCall { call, .. } = event else {
                return HookDecision::Continue;
            };
            if arg_x(&call.function.arguments) == Some(1) {
                return HookDecision::ToolCall(ToolCallAction::stop("stop".to_string()));
            }
            HookDecision::ToolCall(ToolCallAction::run())
        })
    }

    /// Fail-fast, lock-step across surfaces: on a multi-tool turn whose first
    /// tool's hook terminates the run, the SEQUENTIAL default surfaces the
    /// terminate immediately and does **not** start the remaining sibling
    /// tools. The terminating tool's own body never runs either.
    #[tokio::test]
    async fn default_concurrency_terminate_skips_remaining_tools_on_both_drivers() {
        let blocking_calls = Arc::new(AtomicU32::new(0));
        AgentBuilder::new(two_terminating_tools_blocking_model().provider())
            .tool(CountingAddTool {
                calls: blocking_calls.clone(),
            })
            .build()
            .runner("go")
            .max_turns(3)
            .add_hook(terminate_on_first_tool_hook())
            .run()
            .await
            .expect_err("the run terminates");
        assert_eq!(
            blocking_calls.load(SeqCst),
            0,
            "fail-fast: blocking run() must not start the second tool after the first terminates"
        );

        let streaming_calls = Arc::new(AtomicU32::new(0));
        let mut stream = Box::pin(
            AgentBuilder::new(two_terminating_tools_streaming_model().provider())
                .tool(CountingAddTool {
                    calls: streaming_calls.clone(),
                })
                .build()
                .runner("go")
                .max_turns(3)
                .add_hook(terminate_on_first_tool_hook())
                .stream_run(),
        );
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

    /// Concurrent tool execution is bounded on *both* sides: real parallelism
    /// occurs (lower bound) and the configured `tool_concurrency` cap is never
    /// exceeded (upper bound).
    #[tokio::test]
    async fn concurrent_tool_execution_stays_within_the_configured_bound() {
        #[derive(Clone)]
        struct ConcurrencyProbe {
            barrier: Arc<Barrier>,
            active: Arc<AtomicU32>,
            max_active: Arc<AtomicU32>,
        }

        impl PortableTool for ConcurrencyProbe {
            const NAME: &'static str = "add";
            type Error = MockToolError;
            type Args = serde_json::Value;
            type Output = String;

            fn description(&self) -> String {
                "concurrency probe".to_string()
            }

            fn parameters(&self) -> serde_json::Value {
                json!({"type": "object", "properties": {}})
            }

            async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
                let now = self.active.fetch_add(1, SeqCst) + 1;
                self.max_active.fetch_max(now, SeqCst);
                self.barrier.wait().await;
                self.active.fetch_sub(1, SeqCst);
                Ok("ok".to_string())
            }
        }

        let cap = 2usize;
        let probe = ConcurrencyProbe {
            barrier: Arc::new(Barrier::new(cap)),
            active: Arc::new(AtomicU32::new(0)),
            max_active: Arc::new(AtomicU32::new(0)),
        };
        let max_active = probe.max_active.clone();

        // One turn issues four parallel calls to the probe (registered as `add`).
        let model = MockCompletionModel::from_turns([
            MockTurn::from_contents([
                tool_call_content("c1", json!({})),
                tool_call_content("c2", json!({})),
                tool_call_content("c3", json!({})),
                tool_call_content("c4", json!({})),
            ])
            .expect("four tool calls is a valid turn"),
            MockTurn::text("done"),
        ]);

        AgentBuilder::new(model.provider())
            .tool(probe)
            .build()
            .runner("probe concurrency")
            .max_turns(3)
            .tool_concurrency(cap)
            .run()
            .await
            .expect("run should succeed");

        let observed = max_active.load(SeqCst);
        assert!(
            observed > 1,
            "tools actually ran concurrently (lower bound): max_active={observed}"
        );
        assert!(
            observed <= cap as u32,
            "in-flight never exceeded the configured bound {cap}: max_active={observed}"
        );
    }

    /// `tool_concurrency(0)` is clamped to 1 and runs to completion. The
    /// timeout guards against a regression that lets `concurrency == 0` reach a
    /// `buffer_unordered(0)` (which never makes progress).
    #[tokio::test]
    async fn tool_concurrency_zero_is_clamped_and_does_not_hang() {
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "add", json!({"x": 1, "y": 2})),
            MockTurn::text("done"),
        ]);
        let run = AgentBuilder::new(model.provider())
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

    // ------------------------------------------------------------------
    // Local tool_choice/active_tools validation (no provider round-trip)
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn required_with_empty_active_tools_errors_locally_without_provider_call() {
        let empty_active_tools_hook = hook_entry("empty-active-tools", |event| {
            let HookEvent::BeforeModelCall { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::CompletionCall(CompletionCallAction::patch(
                RequestPatch::new().active_tools(Vec::<String>::new()),
            ))
        });

        let model = MockCompletionModel::from_turns([MockTurn::text("unreachable")]);
        let probe = model.clone();
        let err = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .tool_choice(ToolChoice::Required)
            .add_hook(empty_active_tools_hook)
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

    #[tokio::test]
    async fn specific_naming_filtered_out_tool_errors_locally_without_provider_call() {
        let filter_to_add_hook = hook_entry("filter-to-add", |event| {
            let HookEvent::BeforeModelCall { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::CompletionCall(CompletionCallAction::patch(
                RequestPatch::new().active_tools(["add"]),
            ))
        });

        let model = MockCompletionModel::from_turns([MockTurn::text("unreachable")]);
        let probe = model.clone();
        let err = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .tool_choice(ToolChoice::Specific {
                function_names: vec!["subtract".to_string()],
            })
            .add_hook(filter_to_add_hook)
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

    // ------------------------------------------------------------------
    // Structured output: output-tool naming and finalization
    // ------------------------------------------------------------------

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Answer {
        answer: String,
    }

    /// A real tool whose name equals the default synthetic output-tool name.
    struct FinalResultTool;

    impl PortableTool for FinalResultTool {
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

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("real final_result output".to_string())
        }
    }

    #[tokio::test]
    async fn initial_output_tool_collision_uses_a_unique_synthetic_name() {
        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("real", "final_result", json!({})),
            MockTurn::tool_call("output", "final_result_1", json!({ "answer": "done" })),
        ]);
        let probe = model.clone();
        let response = AgentBuilder::new(model.provider())
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
        let tool_names = requests
            .first()
            .expect("first request")
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
            requests
                .get(1)
                .expect("second request")
                .chat_history
                .iter()
                .any(|message| matches!(
                    message,
                    Message::User { content }
                        if content.iter().any(|item| matches!(
                            item,
                            UserContent::ToolResult(result)
                                if result.id == "real"
                                    && result.content.iter().any(|content| matches!(
                                        content,
                                        ToolResultContent::Text(text)
                                            if text.text == "real final_result output"
                                    ))
                        ))
                )),
            "the real `final_result` call must execute normally and its result must reach \
             the follow-up request"
        );
    }

    /// Narrows the advertised tools to `add` for the turn, filtering out the
    /// real `final_result` tool.
    fn active_tools_add_only() -> HookEntry {
        hook_entry("active-tools-add-only", |event| {
            let HookEvent::BeforeModelCall { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::CompletionCall(CompletionCallAction::patch(
                RequestPatch::new().active_tools(["add"]),
            ))
        })
    }

    /// Regression guard: a per-turn `active_tools` allow-list that filters out
    /// a real tool whose name equals the default synthetic output-tool name
    /// must not let the picked output-tool name collide with that (filtered)
    /// real tool. The name is pinned for the whole run, so picking it against
    /// the FULL advertised set keeps it collision-safe once the filter lifts.
    #[tokio::test]
    async fn active_tools_filter_does_not_let_output_tool_collide_with_a_filtered_real_tool() {
        let model = MockCompletionModel::from_turns([MockTurn::tool_call(
            "out1",
            "final_result_1",
            json!({ "answer": "done" }),
        )]);
        let probe = model.clone();
        let response = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .tool(FinalResultTool)
            .output_schema::<Answer>()
            .output_mode(OutputMode::Tool)
            .add_hook(active_tools_add_only())
            .build()
            .runner("go")
            .max_turns(2)
            .run()
            .await
            .expect("run should finalize via the picked output tool `final_result_1`");
        assert!(
            response.output.contains("done"),
            "the intercepted output-tool call should produce the structured result, got {:?}",
            response.output
        );

        let requests = probe.requests();
        let tool_names: Vec<&str> = requests
            .first()
            .expect("the first model request should be captured")
            .tools
            .iter()
            .map(|t| t.name.as_str())
            .collect();
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

    /// Captures whether any `ModelTurnFinished.content` carried a tool call
    /// named `final_result` — the model-emitted output-tool call.
    #[derive(Clone, Default)]
    struct CaptureOutputToolInModelTurn {
        saw_output_tool_call: Arc<Mutex<bool>>,
    }

    impl CaptureOutputToolInModelTurn {
        fn entry(&self) -> HookEntry {
            let hook = self.clone();
            hook_entry("capture-output-tool", move |event| {
                let HookEvent::ModelTurnFinished { content, .. } = event else {
                    return HookDecision::Continue;
                };
                if content.iter().any(|c| {
                    matches!(c, AssistantContent::ToolCall(tc) if tc.function.name == "final_result")
                }) {
                    *hook.saw_output_tool_call.lock().expect("lock") = true;
                }
                HookDecision::Continue
            })
        }
    }

    /// `ModelTurnFinished.content` carries the **model-emitted** content —
    /// including a Tool-mode output-tool call — on both surfaces, even though
    /// the run persists that turn as assistant text with the tool call dropped.
    #[tokio::test]
    async fn model_turn_finished_content_carries_output_tool_call_in_tool_mode() {
        let hook = CaptureOutputToolInModelTurn::default();
        let response = AgentBuilder::new(
            MockCompletionModel::from_turns([MockTurn::tool_call(
                "out1",
                "final_result",
                json!({ "answer": "done" }),
            )])
            .provider(),
        )
        .output_schema::<Answer>()
        .output_mode(OutputMode::Tool)
        .add_hook(hook.entry())
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

        let s_hook = CaptureOutputToolInModelTurn::default();
        let mut stream = Box::pin(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns([vec![
                    MockStreamEvent::tool_call("out1", "final_result", json!({ "answer": "done" })),
                    MockStreamEvent::final_response_with_total_tokens(0),
                ]])
                .provider(),
            )
            .output_schema::<Answer>()
            .output_mode(OutputMode::Tool)
            .add_hook(s_hook.entry())
            .build()
            .runner("go")
            .max_turns(2)
            .stream_run(),
        );
        while stream.next().await.is_some() {}
        assert!(
            *s_hook.saw_output_tool_call.lock().expect("lock"),
            "ModelTurnFinished.content must carry the model-emitted output-tool call (streaming)"
        );
    }

    /// A Tool-mode output-tool call finalizes the run directly, so on the
    /// streaming surface it is **not** re-emitted as a complete tool-call item;
    /// its structured result is surfaced in the final response.
    #[tokio::test]
    async fn output_tool_finalization_emits_no_complete_tool_call_stream_item() {
        let mut stream = Box::pin(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns([vec![
                    MockStreamEvent::tool_call("out1", "final_result", json!({ "answer": "done" })),
                    MockStreamEvent::final_response_with_total_tokens(0),
                ]])
                .provider(),
            )
            .output_schema::<Answer>()
            .output_mode(OutputMode::Tool)
            .build()
            .runner("go")
            .max_turns(2)
            .stream_run(),
        );

        let mut saw_complete_output_tool_call = false;
        let mut final_has_output = false;
        while let Some(item) = stream.next().await {
            match item.expect("stream item") {
                AgentStreamItem::Assistant(StreamedAssistantContent::ToolCall {
                    tool_call,
                    ..
                }) if tool_call.function.name == "final_result" => {
                    saw_complete_output_tool_call = true;
                }
                AgentStreamItem::Final(res) => {
                    final_has_output = res.output().contains("done");
                }
                _ => {}
            }
        }
        assert!(
            !saw_complete_output_tool_call,
            "the output-tool call finalizes the run, so no complete tool-call stream item \
             must be emitted for it"
        );
        assert!(
            final_has_output,
            "the structured output must be surfaced via the final response"
        );
    }

    // ------------------------------------------------------------------
    // Human-in-the-loop / approval policy recipes
    // ------------------------------------------------------------------

    /// A human reviewer's decision for a pending tool call.
    enum Decision {
        Approve,
        Deny(&'static str),
        Edit(serde_json::Value),
        Abort(&'static str),
    }

    /// Simulates a human reviewer by popping a scripted decision per
    /// `ToolCall` and mapping it to the matching event-specific action.
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
            self.reviewed.lock().expect("reviewed").clone()
        }

        fn entry(&self) -> HookEntry {
            let hook = self.clone();
            hook_entry("human-approval", move |event| {
                let HookEvent::ToolCall { call, .. } = event else {
                    return HookDecision::Continue;
                };
                let tool_name = &call.function.name;
                let args = &call.function.arguments;
                hook.reviewed
                    .lock()
                    .expect("reviewed")
                    .push(format!("{tool_name}({args})"));
                let decision = hook.decisions.lock().expect("decisions").pop_front();
                HookDecision::ToolCall(match decision {
                    Some(Decision::Approve) => ToolCallAction::run(),
                    Some(Decision::Deny(reason)) => ToolCallAction::skip(reason),
                    Some(Decision::Edit(args)) => ToolCallAction::rewrite(args),
                    Some(Decision::Abort(reason)) => ToolCallAction::stop(reason),
                    // Fail closed if the script is exhausted.
                    None => ToolCallAction::skip("denied: no scripted decision (fail-closed)"),
                })
            })
        }
    }

    #[tokio::test]
    async fn human_in_the_loop_approve_deny_edit_parity_across_run_and_stream() {
        // One turn issues three tool calls; the reviewer decides each differently.
        let turns = [
            ScriptedTurn::ToolCalls(vec![
                add_call("tc1", 2, 3),   // approve -> runs, 2 + 3 = 5
                add_call("tc2", 10, 20), // deny    -> skipped; model sees the reason
                add_call("tc3", 1, 1),   // edit    -> runs 1 + 100 = 101
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
        let blocking = AgentBuilder::new(blocking_model.provider())
            .tool(MockAddTool)
            .build()
            .runner("carry out the plan")
            .max_turns(3)
            .add_hook(blocking_recorder.entry())
            .add_hook(blocking_approver.entry())
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
        let final_response = drive_to_final_response(
            AgentBuilder::new(streaming_model.provider())
                .tool(MockAddTool)
                .build()
                .runner("carry out the plan")
                .max_turns(3)
                .add_hook(streaming_recorder.entry())
                .add_hook(streaming_approver.entry())
                .stream_run(),
        )
        .await;

        // Approved (5) and edited (101) tools executed, in call order; the
        // denied call executed nothing but fires a ToolResult carrying its
        // verbatim denial reason — identically on both drivers.
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
        assert!(
            !blocking_recorder.tool_results().contains(&"30".to_string()),
            "the denied call must not have executed"
        );

        let reviewed = blocking_approver.reviewed();
        assert_eq!(reviewed.len(), 3);
        assert_eq!(reviewed, streaming_approver.reviewed());
        let first = reviewed.first().expect("first reviewed call");
        assert!(
            first.contains('2') && first.contains('3'),
            "first reviewed call should be add(2, 3): {reviewed:?}"
        );
        let second = reviewed.get(1).expect("second reviewed call");
        assert!(
            second.contains("10") && second.contains("20"),
            "the denied (second) call should be add(10, 20): {reviewed:?}"
        );

        assert_eq!(blocking.output, "done");
        assert_eq!(final_response.output(), blocking.output);
        assert_eq!(
            blocking_recorder.shared_events(),
            streaming_recorder.shared_events()
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
        assert!(
            tool_result_text_in_history(&blocking_messages, denial),
            "the denial reason must be the denied call's tool result in the history"
        );
        assert!(
            tool_result_json_in_history(&blocking_messages, &json!(101)),
            "the edited call must have executed with the rewritten arguments"
        );
    }

    #[tokio::test]
    async fn human_in_the_loop_abort_terminates_the_run() {
        let turns = [
            ScriptedTurn::ToolCalls(vec![add_call("tc1", 2, 3)]),
            ScriptedTurn::Text("unreachable"),
        ];
        const ABORT_REASON: &str = "aborted by the human reviewer";

        let blocking_model =
            MockCompletionModel::from_turns(turns.iter().map(ScriptedTurn::as_blocking_turn));
        let err = AgentBuilder::new(blocking_model.provider())
            .tool(MockAddTool)
            .build()
            .runner("do the sensitive thing")
            .max_turns(3)
            .add_hook(HumanApprovalHook::new([Decision::Abort(ABORT_REASON)]).entry())
            .run()
            .await
            .expect_err("an aborted tool call should terminate the blocking run");
        assert!(
            format!("{err}").contains(ABORT_REASON),
            "the abort reason should surface in the blocking error, got: {err}"
        );

        let streaming_model = MockCompletionModel::from_stream_turns(
            turns
                .iter()
                .map(|turn| turn.as_stream_events(StreamShape::Complete)),
        );
        let mut stream = Box::pin(
            AgentBuilder::new(streaming_model.provider())
                .tool(MockAddTool)
                .build()
                .runner("do the sensitive thing")
                .max_turns(3)
                .add_hook(HumanApprovalHook::new([Decision::Abort(ABORT_REASON)]).entry())
                .stream_run(),
        );
        let mut stream_error = None;
        while let Some(item) = stream.next().await {
            match item {
                Err(err) => stream_error = Some(format!("{err}")),
                Ok(AgentStreamItem::Final(resp)) => {
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
    /// everything else (fail-closed), and cache each decision so a repeated
    /// tool is not re-evaluated ("sticky").
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
            self.evaluated.lock().expect("evaluated").clone()
        }

        fn entry(&self) -> HookEntry {
            let hook = self.clone();
            hook_entry("policy", move |event| {
                let HookEvent::ToolCall { call, .. } = event else {
                    return HookDecision::Continue;
                };
                let tool_name = call.function.name.clone();
                let cached = hook.cache.lock().expect("cache").get(&tool_name).copied();
                let approved = match cached {
                    Some(decision) => decision, // sticky: reuse without re-evaluating
                    None => {
                        hook.evaluated
                            .lock()
                            .expect("evaluated")
                            .push(tool_name.clone());
                        let decision = hook.auto_approve.contains(tool_name.as_str());
                        hook.cache
                            .lock()
                            .expect("cache")
                            .insert(tool_name.clone(), decision);
                        decision
                    }
                };
                HookDecision::ToolCall(if approved {
                    ToolCallAction::run()
                } else {
                    ToolCallAction::skip(format!("denied by policy: `{tool_name}` not allowed"))
                })
            })
        }
    }

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
        let out = AgentBuilder::new(model.provider())
            .tool(MockAddTool)
            .tool(MockSubtractTool)
            .build()
            .runner("go")
            .max_turns(3)
            .add_hook(recorder.entry())
            .add_hook(policy.entry())
            .run()
            .await
            .expect("policy run should succeed");

        assert_eq!(out.output, "done");
        assert_eq!(
            recorder.tool_results(),
            vec![
                "5".to_string(),
                "denied by policy: `subtract` not allowed".to_string(),
                "5".to_string()
            ]
        );
        // The policy evaluated each distinct tool once; the second `add` reused
        // the cached decision rather than being re-evaluated.
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

    // ------------------------------------------------------------------
    // Model-turn retry: budgets, history shape, hook order, fail-closed
    // ------------------------------------------------------------------

    #[derive(Clone, Copy)]
    enum TestRetryMode {
        Repeat,
        Feedback(&'static str),
    }

    #[derive(Clone, Default)]
    struct StatefulCompletionPatch {
        calls: Arc<AtomicU32>,
    }

    impl StatefulCompletionPatch {
        fn calls(&self) -> u32 {
            self.calls.load(SeqCst)
        }

        fn entry(&self) -> HookEntry {
            let hook = self.clone();
            hook_entry("stateful-completion-patch", move |event| {
                let HookEvent::BeforeModelCall { .. } = event else {
                    return HookDecision::Continue;
                };
                let call = hook.calls.fetch_add(1, SeqCst);
                HookDecision::CompletionCall(CompletionCallAction::patch(
                    RequestPatch::new().temperature(if call == 0 { 0.1 } else { 0.9 }),
                ))
            })
        }
    }

    /// A policy-owned retry budget: the framework only enforces `max_turns`, so
    /// the entry keeps its narrower limit in the state it captures.
    fn bounded_response_retry(
        rejected_text: &'static str,
        max_retries: usize,
        mode: TestRetryMode,
    ) -> HookEntry {
        let attempts = Arc::new(Mutex::new(0usize));
        hook_entry("bounded-response-retry", move |event| {
            let HookEvent::ModelTurnFinished { content, .. } = event else {
                return HookDecision::Continue;
            };
            let rejected = content.iter().any(
                |content| matches!(content, AssistantContent::Text(text) if text.text == rejected_text),
            );
            if !rejected {
                return HookDecision::Continue;
            }

            let attempt = {
                let mut attempts = attempts.lock().expect("retry attempts");
                *attempts += 1;
                *attempts
            };
            if attempt > max_retries {
                return HookDecision::ModelTurn(ModelTurnAction::stop(format!(
                    "response retry limit ({max_retries}) exceeded"
                )));
            }

            HookDecision::ModelTurn(match mode {
                TestRetryMode::Repeat => ModelTurnAction::repeat(),
                TestRetryMode::Feedback(feedback) => ModelTurnAction::retry_with_feedback(feedback),
            })
        })
    }

    fn retry_usage(input_tokens: u64, output_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            ..Usage::new()
        }
    }

    #[tokio::test]
    async fn blocking_model_turn_repeat_preserves_prompt_history_with_fresh_preparation() {
        let first_usage = retry_usage(10, 3);
        let second_usage = retry_usage(7, 2);
        let completion_patch = StatefulCompletionPatch::default();
        let model = MockCompletionModel::from_turns([
            MockTurn::text("rejected").with_usage(first_usage),
            MockTurn::text("accepted").with_usage(second_usage),
        ]);
        let response = AgentBuilder::new(model.clone().provider())
            .add_hook(completion_patch.entry())
            .add_hook(bounded_response_retry("rejected", 1, TestRetryMode::Repeat))
            .build()
            .runner("question")
            .max_turns(2)
            .run()
            .await
            .expect("repeat should recover");

        assert_eq!(response.output, "accepted");
        assert_eq!(response.usage, first_usage + second_usage);
        assert_eq!(response.completion_calls.len(), 2);
        assert_eq!(
            response.messages.expect("response messages"),
            vec![Message::user("question"), Message::assistant("accepted")]
        );

        let requests = model.requests();
        assert_eq!(requests.len(), 2);
        let first = requests
            .first()
            .expect("first request")
            .chat_history
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let second = requests
            .get(1)
            .expect("second request")
            .chat_history
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(first, vec![Message::user("question")]);
        assert_eq!(
            second, first,
            "Repeat must preserve the prompt and preceding history"
        );
        assert_eq!(requests.first().expect("first").temperature, Some(0.1));
        assert_eq!(requests.get(1).expect("second").temperature, Some(0.9));
        assert_eq!(completion_patch.calls(), 2);
    }

    #[tokio::test]
    async fn blocking_model_turn_feedback_preserves_rejected_response() {
        let model = MockCompletionModel::from_turns([
            MockTurn::text("rejected"),
            MockTurn::text("accepted"),
        ]);
        let response = AgentBuilder::new(model.clone().provider())
            .add_hook(bounded_response_retry(
                "rejected",
                1,
                TestRetryMode::Feedback("try another approach"),
            ))
            .build()
            .runner("question")
            .max_turns(2)
            .run()
            .await
            .expect("feedback retry should recover");

        assert_eq!(response.output, "accepted");
        assert_eq!(
            response.messages.expect("response messages"),
            vec![
                Message::user("question"),
                Message::assistant("rejected"),
                Message::user("try another approach"),
                Message::assistant("accepted"),
            ]
        );
        let requests = model.requests();
        assert_eq!(
            requests
                .get(1)
                .expect("second request")
                .chat_history
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![
                Message::user("question"),
                Message::assistant("rejected"),
                Message::user("try another approach"),
            ]
        );
    }

    #[tokio::test]
    async fn blocking_empty_feedback_retry_omits_empty_assistant_history() {
        let first_usage = retry_usage(5, 1);
        let second_usage = retry_usage(7, 2);
        let model = MockCompletionModel::from_turns([
            MockTurn::text("").with_usage(first_usage),
            MockTurn::text("accepted").with_usage(second_usage),
        ]);
        let response = AgentBuilder::new(model.clone().provider())
            .add_hook(bounded_response_retry(
                "",
                1,
                TestRetryMode::Feedback("provide an answer"),
            ))
            .build()
            .runner("question")
            .max_turns(2)
            .run()
            .await
            .expect("feedback retry should recover from an empty turn");

        assert_eq!(response.output, "accepted");
        assert_eq!(response.usage, first_usage + second_usage);
        assert_eq!(response.completion_calls.len(), 2);
        assert_eq!(
            response.messages.expect("response messages"),
            vec![
                Message::user("question"),
                Message::user("provide an answer"),
                Message::assistant("accepted"),
            ]
        );
        assert_eq!(
            model
                .requests()
                .get(1)
                .expect("second request")
                .chat_history
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![
                Message::user("question"),
                Message::user("provide an answer")
            ],
            "the retry request must not contain an empty assistant message"
        );
    }

    #[tokio::test]
    async fn streaming_model_turn_retry_marks_rollback_and_matches_blocking_accounting() {
        let first_usage = retry_usage(10, 3);
        let second_usage = retry_usage(7, 2);
        let model = MockCompletionModel::from_stream_turns([
            [
                MockStreamEvent::text("rejected"),
                MockStreamEvent::final_response(first_usage),
            ],
            [
                MockStreamEvent::text("accepted"),
                MockStreamEvent::final_response(second_usage),
            ],
        ]);
        let mut stream = Box::pin(
            AgentBuilder::new(model.clone().provider())
                .add_hook(bounded_response_retry("rejected", 1, TestRetryMode::Repeat))
                .build()
                .runner("question")
                .max_turns(2)
                .stream_run(),
        );

        let mut retries = Vec::new();
        let mut provider_finals = 0;
        let mut completion_calls = 0;
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            match item.expect("stream item") {
                AgentStreamItem::ModelTurnRetried { turn } => retries.push(turn),
                AgentStreamItem::CompletionCall(_) => completion_calls += 1,
                AgentStreamItem::Assistant(StreamedAssistantContent::Final(_)) => {
                    provider_finals += 1
                }
                AgentStreamItem::Final(response) => final_response = Some(response),
                _ => {}
            }
        }

        assert_eq!(retries, vec![1]);
        // NOTE: the classic streaming driver buffered and suppressed the
        // rejected turn's provider final; the unified `AgentStream` surfaces
        // one per provider turn (the run-level accounting below is unchanged).
        assert_eq!(provider_finals, 2);
        assert_eq!(completion_calls, 2);
        let response = final_response.expect("run final response");
        assert_eq!(response.output, "accepted");
        assert_eq!(response.usage, first_usage + second_usage);
        assert_eq!(response.completion_calls.len(), 2);
        assert_eq!(
            response.messages.expect("response messages"),
            vec![Message::user("question"), Message::assistant("accepted")]
        );
        assert_eq!(model.requests().len(), 2);
    }

    #[tokio::test]
    async fn streaming_feedback_retry_matches_blocking_history_and_usage() {
        let first_usage = retry_usage(5, 2);
        let second_usage = retry_usage(8, 4);
        let blocking = AgentBuilder::new(
            MockCompletionModel::from_turns([
                MockTurn::text("rejected").with_usage(first_usage),
                MockTurn::text("accepted").with_usage(second_usage),
            ])
            .provider(),
        )
        .add_hook(bounded_response_retry(
            "rejected",
            1,
            TestRetryMode::Feedback("correct the answer"),
        ))
        .build()
        .runner("question")
        .max_turns(2)
        .run()
        .await
        .expect("blocking feedback retry");

        let mut stream = Box::pin(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns([
                    [
                        MockStreamEvent::text("rejected"),
                        MockStreamEvent::final_response(first_usage),
                    ],
                    [
                        MockStreamEvent::text("accepted"),
                        MockStreamEvent::final_response(second_usage),
                    ],
                ])
                .provider(),
            )
            .add_hook(bounded_response_retry(
                "rejected",
                1,
                TestRetryMode::Feedback("correct the answer"),
            ))
            .build()
            .runner("question")
            .max_turns(2)
            .stream_run(),
        );
        let mut saw_retry = false;
        let mut streaming = None;
        while let Some(item) = stream.next().await {
            match item.expect("stream item") {
                AgentStreamItem::ModelTurnRetried { turn: 1 } => saw_retry = true,
                AgentStreamItem::Final(response) => streaming = Some(response),
                _ => {}
            }
        }

        let streaming = streaming.expect("streaming final response");
        assert!(saw_retry);
        assert_eq!(streaming.output, blocking.output);
        assert_eq!(streaming.usage, blocking.usage);
        assert_eq!(streaming.completion_calls, blocking.completion_calls);
        assert_eq!(
            serde_json::to_value(streaming.messages).expect("streaming history"),
            serde_json::to_value(blocking.messages).expect("blocking history")
        );
    }

    #[tokio::test]
    async fn streaming_empty_feedback_retry_omits_empty_assistant_history() {
        let first_usage = retry_usage(5, 1);
        let second_usage = retry_usage(7, 2);
        let model = MockCompletionModel::from_stream_turns([
            [
                MockStreamEvent::text(""),
                MockStreamEvent::final_response(first_usage),
            ],
            [
                MockStreamEvent::text("accepted"),
                MockStreamEvent::final_response(second_usage),
            ],
        ]);
        let mut stream = Box::pin(
            AgentBuilder::new(model.clone().provider())
                .add_hook(bounded_response_retry(
                    "",
                    1,
                    TestRetryMode::Feedback("provide an answer"),
                ))
                .build()
                .runner("question")
                .max_turns(2)
                .stream_run(),
        );

        let mut retries = Vec::new();
        let mut provider_finals = 0;
        let mut completion_calls = 0;
        let mut final_response = None;
        while let Some(item) = stream.next().await {
            match item.expect("stream item") {
                AgentStreamItem::ModelTurnRetried { turn } => retries.push(turn),
                AgentStreamItem::CompletionCall(_) => completion_calls += 1,
                AgentStreamItem::Assistant(StreamedAssistantContent::Final(_)) => {
                    provider_finals += 1;
                }
                AgentStreamItem::Final(response) => final_response = Some(response),
                _ => {}
            }
        }

        assert_eq!(retries, vec![1]);
        // See the note in
        // `streaming_model_turn_retry_marks_rollback_and_matches_blocking_accounting`:
        // the unified stream surfaces one provider final per turn.
        assert_eq!(provider_finals, 2);
        assert_eq!(completion_calls, 2);
        let response = final_response.expect("run final response");
        assert_eq!(response.output, "accepted");
        assert_eq!(response.usage, first_usage + second_usage);
        assert_eq!(response.completion_calls.len(), 2);
        assert_eq!(
            response.messages.expect("response messages"),
            vec![
                Message::user("question"),
                Message::user("provide an answer"),
                Message::assistant("accepted"),
            ]
        );
        assert_eq!(
            model
                .requests()
                .get(1)
                .expect("second request")
                .chat_history
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![
                Message::user("question"),
                Message::user("provide an answer")
            ],
            "the retry request must not contain an empty assistant message"
        );
    }

    #[tokio::test]
    async fn response_retry_preserves_model_turn_hook_order_across_surfaces() {
        let blocking_events = RecordingHook::default();
        AgentBuilder::new(
            MockCompletionModel::from_turns([
                MockTurn::text("rejected"),
                MockTurn::text("accepted"),
            ])
            .provider(),
        )
        .add_hook(blocking_events.entry())
        .add_hook(bounded_response_retry("rejected", 1, TestRetryMode::Repeat))
        .build()
        .runner("question")
        .max_turns(2)
        .run()
        .await
        .expect("blocking retry");

        let streaming_events = RecordingHook::default();
        let mut stream = Box::pin(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns([
                    [
                        MockStreamEvent::text("rejected"),
                        MockStreamEvent::final_response_with_default_usage(),
                    ],
                    [
                        MockStreamEvent::text("accepted"),
                        MockStreamEvent::final_response_with_default_usage(),
                    ],
                ])
                .provider(),
            )
            .add_hook(streaming_events.entry())
            .add_hook(bounded_response_retry("rejected", 1, TestRetryMode::Repeat))
            .build()
            .runner("question")
            .max_turns(2)
            .stream_run(),
        );
        while let Some(item) = stream.next().await {
            item.expect("streaming retry item");
        }

        let shared_order = |events: &RecordingHook| {
            events
                .all_events()
                .into_iter()
                .filter(|event| {
                    matches!(
                        event,
                        EventKind::CompletionCall | EventKind::ModelTurnFinished
                    )
                })
                .collect::<Vec<_>>()
        };
        let expected = vec![
            EventKind::CompletionCall,
            EventKind::ModelTurnFinished,
            EventKind::CompletionCall,
            EventKind::ModelTurnFinished,
        ];
        assert_eq!(shared_order(&blocking_events), expected);
        assert_eq!(shared_order(&streaming_events), expected);

        assert_eq!(
            blocking_events.all_events(),
            vec![
                EventKind::CompletionCall,
                EventKind::CompletionResponse,
                EventKind::ModelTurnFinished,
                EventKind::CompletionCall,
                EventKind::CompletionResponse,
                EventKind::ModelTurnFinished,
            ]
        );
        assert_eq!(
            streaming_events.all_events(),
            vec![
                EventKind::CompletionCall,
                EventKind::TextDelta,
                EventKind::StreamResponseFinish,
                EventKind::ModelTurnFinished,
                EventKind::CompletionCall,
                EventKind::TextDelta,
                EventKind::StreamResponseFinish,
                EventKind::ModelTurnFinished,
            ]
        );
    }

    #[tokio::test]
    async fn streaming_model_turn_retry_respects_max_turns() {
        let model = MockCompletionModel::from_stream_turns([[
            MockStreamEvent::text("rejected"),
            MockStreamEvent::final_response_with_default_usage(),
        ]]);
        let mut stream = Box::pin(
            AgentBuilder::new(model.provider())
                .add_hook(bounded_response_retry("rejected", 1, TestRetryMode::Repeat))
                .build()
                .runner("question")
                .max_turns(1)
                .stream_run(),
        );

        let mut saw_rollback = false;
        let mut error = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(AgentStreamItem::ModelTurnRetried { turn: 1 }) => saw_rollback = true,
                Ok(_) => {}
                Err(err) => error = Some(err),
            }
        }
        assert!(saw_rollback);
        assert!(matches!(
            error,
            Some(PromptError::MaxTurnsError { max_turns: 1, .. })
        ));
    }

    fn always_repeat_model_turn() -> HookEntry {
        hook_entry("always-repeat-model-turn", |event| {
            let HookEvent::ModelTurnFinished { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::ModelTurn(ModelTurnAction::repeat())
        })
    }

    #[tokio::test]
    async fn model_turn_retry_rejects_tool_turn_before_tool_hooks_or_execution() {
        let recorder = RecordingHook::default();
        let executions = Arc::new(AtomicU32::new(0));
        let err = AgentBuilder::new(
            MockCompletionModel::from_turns([MockTurn::tool_call(
                "tc1",
                "add",
                json!({"x": 1, "y": 2}),
            )])
            .provider(),
        )
        .tool(CountingAddTool {
            calls: executions.clone(),
        })
        .add_hook(recorder.entry())
        .add_hook(always_repeat_model_turn())
        .build()
        .runner("add")
        .max_turns(2)
        .run()
        .await
        .expect_err("tool-bearing retry must fail closed");

        let PromptError::PromptCancelled {
            chat_history,
            reason,
        } = err
        else {
            panic!("tool-bearing retry should return PromptCancelled");
        };
        assert!(reason.contains("tool-bearing model turns"));
        assert!(reason.contains("tool-call hooks"));
        assert_eq!(chat_history, vec![Message::user("add")]);
        assert_eq!(recorder.count(EventKind::ToolCall), 0);
        assert_eq!(recorder.count(EventKind::ToolResult), 0);
        assert_eq!(executions.load(SeqCst), 0);
    }

    #[tokio::test]
    async fn streaming_model_turn_retry_rejects_tool_turn_without_committed_execution() {
        let recorder = RecordingHook::default();
        let executions = Arc::new(AtomicU32::new(0));
        let mut stream = Box::pin(
            AgentBuilder::new(
                MockCompletionModel::from_stream_turns([[
                    MockStreamEvent::tool_call_name_delta("tc1", "ic1", "add"),
                    MockStreamEvent::tool_call_arguments_delta("tc1", "ic1", r#"{"x":1,"y":2}"#),
                    MockStreamEvent::tool_call("tc1", "add", json!({"x": 1, "y": 2})),
                    MockStreamEvent::final_response_with_default_usage(),
                ]])
                .provider(),
            )
            .tool(CountingAddTool {
                calls: executions.clone(),
            })
            .add_hook(recorder.entry())
            .add_hook(always_repeat_model_turn())
            .build()
            .runner("add")
            .max_turns(2)
            .stream_run(),
        );

        let mut execution_commits = 0;
        let mut tool_results = 0;
        let mut provider_finals = 0;
        let mut agent_finals = 0;
        let mut retry_markers = 0;
        let mut error = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(AgentStreamItem::ToolExecutionCommitted { .. }) => execution_commits += 1,
                Ok(AgentStreamItem::User(StreamedUserContent::ToolResult { .. })) => {
                    tool_results += 1
                }
                Ok(AgentStreamItem::Assistant(StreamedAssistantContent::Final(_))) => {
                    provider_finals += 1
                }
                Ok(AgentStreamItem::Final(_)) => agent_finals += 1,
                Ok(AgentStreamItem::ModelTurnRetried { .. }) => retry_markers += 1,
                Ok(_) => {}
                Err(err) => error = Some(err),
            }
        }

        let Some(PromptError::PromptCancelled {
            chat_history,
            reason,
        }) = error
        else {
            panic!("tool-bearing streaming retry should return PromptCancelled");
        };
        assert!(reason.contains("tool-bearing model turns"));
        assert!(reason.contains("tool-call hooks"));
        assert_eq!(chat_history, vec![Message::user("add")]);
        assert_eq!(execution_commits, 0);
        assert_eq!(tool_results, 0);
        assert_eq!(provider_finals, 0);
        assert_eq!(agent_finals, 0);
        assert_eq!(retry_markers, 0);
        assert_eq!(recorder.count(EventKind::ToolCall), 0);
        assert_eq!(recorder.count(EventKind::ToolResult), 0);
        assert_eq!(executions.load(SeqCst), 0);
    }

    /// A per-run retry entry that parks at `barrier` on the first rejected
    /// turn, so two concurrent runs are guaranteed to be mid-retry together.
    fn barrier_response_retry(barrier: Arc<Barrier>) -> HookEntry {
        let attempts = Arc::new(Mutex::new(0usize));
        HookEntry::new("barrier-response-retry", move |event| {
            let barrier = barrier.clone();
            let attempts = attempts.clone();
            Box::pin(async move {
                let HookEvent::ModelTurnFinished { content, .. } = event else {
                    return HookDecision::Continue;
                };
                let rejected = content.iter().any(
                    |content| matches!(content, AssistantContent::Text(text) if text.text == "rejected"),
                );
                if !rejected {
                    return HookDecision::Continue;
                }
                barrier.wait().await;
                let attempt = {
                    let mut attempts = attempts.lock().expect("retry attempts");
                    *attempts += 1;
                    *attempts
                };
                HookDecision::ModelTurn(if attempt > 1 {
                    ModelTurnAction::stop("response retry limit (1) exceeded")
                } else {
                    ModelTurnAction::repeat()
                })
            })
        })
    }

    /// Two concurrent runs of one agent each get their own retry budget: the
    /// budget lives in the entry registered for that run, so both recover
    /// instead of the second inheriting the first's exhausted allowance.
    #[tokio::test]
    async fn concurrent_runs_of_same_agent_have_independent_retry_budgets() {
        let barrier = Arc::new(Barrier::new(2));
        let agent = AgentBuilder::new(
            MockCompletionModel::from_turns([
                MockTurn::text("rejected"),
                MockTurn::text("rejected"),
                MockTurn::text("accepted one"),
                MockTurn::text("accepted two"),
            ])
            .provider(),
        )
        .build();

        let first = agent
            .runner("first")
            .max_turns(2)
            .add_hook(barrier_response_retry(barrier.clone()))
            .run();
        let second = agent
            .runner("second")
            .max_turns(2)
            .add_hook(barrier_response_retry(barrier))
            .run();
        let (first, second) = tokio::join!(first, second);
        let first = first.expect("first run");
        let second = second.expect("second run");

        let outputs = std::collections::HashSet::from([first.output, second.output]);
        assert_eq!(
            outputs,
            std::collections::HashSet::from([
                "accepted one".to_string(),
                "accepted two".to_string(),
            ])
        );
        assert_eq!(first.completion_calls.len(), 2);
        assert_eq!(second.completion_calls.len(), 2);
    }

    /// A model-turn entry that always answers `action` and counts its calls.
    fn fixed_model_turn_action(action: ModelTurnAction, calls: Arc<AtomicU32>) -> HookEntry {
        hook_entry("fixed-model-turn-action", move |event| {
            let HookEvent::ModelTurnFinished { .. } = event else {
                return HookDecision::Continue;
            };
            calls.fetch_add(1, SeqCst);
            HookDecision::ModelTurn(action.clone())
        })
    }

    /// The model-turn fold short-circuits on the first non-`Continue` answer:
    /// later entries are never invoked, and the winning action is the one that
    /// short-circuited (retry or stop, in registration order).
    #[tokio::test]
    async fn model_turn_action_short_circuits_the_hook_list() {
        let content = OneOrMany::one(AssistantContent::text("response"));

        let first_calls = Arc::new(AtomicU32::new(0));
        let retry_calls = Arc::new(AtomicU32::new(0));
        let skipped_calls = Arc::new(AtomicU32::new(0));
        let flat = Hooks::new()
            .with(fixed_model_turn_action(
                ModelTurnAction::Continue,
                first_calls.clone(),
            ))
            .with(fixed_model_turn_action(
                ModelTurnAction::repeat(),
                retry_calls.clone(),
            ))
            .with(fixed_model_turn_action(
                ModelTurnAction::stop("unreachable"),
                skipped_calls.clone(),
            ));
        assert!(matches!(
            flat.dispatch_model_turn(1, &content, Usage::new()).await,
            ModelTurnAction::Retry(_)
        ));
        assert_eq!(first_calls.load(SeqCst), 1);
        assert_eq!(retry_calls.load(SeqCst), 1);
        assert_eq!(skipped_calls.load(SeqCst), 0);

        let feedback_calls = Arc::new(AtomicU32::new(0));
        let after_feedback_calls = Arc::new(AtomicU32::new(0));
        let feedback = Hooks::new()
            .with(fixed_model_turn_action(
                ModelTurnAction::retry_with_feedback("fix it"),
                feedback_calls.clone(),
            ))
            .with(fixed_model_turn_action(
                ModelTurnAction::Continue,
                after_feedback_calls.clone(),
            ));
        assert!(matches!(
            feedback.dispatch_model_turn(1, &content, Usage::new()).await,
            ModelTurnAction::Retry(crate::agent::RetryRequest::Feedback(feedback))
                if feedback == "fix it"
        ));
        assert_eq!(feedback_calls.load(SeqCst), 1);
        assert_eq!(after_feedback_calls.load(SeqCst), 0);

        let stop_calls = Arc::new(AtomicU32::new(0));
        let after_stop_calls = Arc::new(AtomicU32::new(0));
        let stopping = Hooks::new()
            .with(fixed_model_turn_action(
                ModelTurnAction::stop("stop now"),
                stop_calls.clone(),
            ))
            .with(fixed_model_turn_action(
                ModelTurnAction::Continue,
                after_stop_calls.clone(),
            ));
        assert!(matches!(
            stopping.dispatch_model_turn(1, &content, Usage::new()).await,
            ModelTurnAction::Stop(reason) if reason == "stop now"
        ));
        assert_eq!(stop_calls.load(SeqCst), 1);
        assert_eq!(after_stop_calls.load(SeqCst), 0);
    }

    // ------------------------------------------------------------------
    // Structured tool-execution outcomes
    // ------------------------------------------------------------------

    mod structured_tool_results {
        use super::*;
        use crate::test_utils::{MockDeniedTool, MockFailingTool, MockHandledFailureTool};
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

            fn entry(&self) -> HookEntry {
                let hook = self.clone();
                hook_entry("outcome", move |event| {
                    let HookEvent::ToolResult {
                        result,
                        presentation,
                        ..
                    } = event
                    else {
                        return HookDecision::Continue;
                    };
                    hook.outcomes
                        .lock()
                        .expect("outcomes")
                        .push(outcome_label(&result));
                    hook.results
                        .lock()
                        .expect("results")
                        .push(presentation.render());
                    HookDecision::ToolResult(ToolResultAction::keep())
                })
            }
        }

        /// A compact string label for an outcome, e.g. `error:timeout`.
        ///
        /// NOTE: the unified engine hands the tool-result hook a result
        /// rebuilt from the committed `UserContent`
        /// ([`crate::session::raw_tool_result`]), so the executor's
        /// classification (error kind / skipped / refused) is currently
        /// flattened to `success`. The label is kept so the classification
        /// contract is still expressed; the assertions below pin what the
        /// engine delivers today — the model-visible result text and the
        /// non-fatality of tool failures.
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

        #[tokio::test]
        async fn timeout_failure_surfaces_structured_outcome() {
            let hook = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("flaky_tool").provider())
                .tool(MockFailingTool::new(ToolErrorKind::Timeout))
                .add_hook(hook.entry())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("run should succeed; a tool timeout is model-visible feedback, not fatal");

            assert_eq!(hook.results(), vec!["mock tool call failed".to_string()]);
        }

        #[tokio::test]
        async fn hook_terminates_after_repeated_timeouts() {
            /// Terminates once it has observed two timeout results.
            fn timeout_terminator() -> HookEntry {
                let timeouts = Arc::new(Mutex::new(0usize));
                hook_entry("timeout-terminator", move |event| {
                    let HookEvent::ToolResult { result, .. } = event else {
                        return HookDecision::Continue;
                    };
                    let _ = &result;
                    {
                        let mut count = timeouts.lock().expect("timeouts");
                        *count += 1;
                        if *count >= 2 {
                            return HookDecision::ToolResult(ToolResultAction::stop(
                                "aborting after repeated tool timeouts",
                            ));
                        }
                    }
                    HookDecision::ToolResult(ToolResultAction::keep())
                })
            }

            let observer = OutcomeHook::default();
            let err = AgentBuilder::new(
                MockCompletionModel::from_turns([
                    MockTurn::tool_call("tc1", "flaky_tool", json!({})),
                    MockTurn::tool_call("tc2", "flaky_tool", json!({})),
                    MockTurn::text("unreachable"),
                ])
                .provider(),
            )
            .tool(MockFailingTool::new(ToolErrorKind::Timeout))
            .add_hook(observer.entry())
            .add_hook(timeout_terminator())
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
                observer.results(),
                vec![
                    "mock tool call failed".to_string(),
                    "mock tool call failed".to_string()
                ],
                "both timeout results must be observed before termination"
            );
        }

        #[tokio::test]
        async fn not_found_outcome_is_structured_and_non_fatal() {
            let hook = OutcomeHook::default();
            let status: Arc<Mutex<Option<u16>>> = Arc::new(Mutex::new(None));

            fn status_probe(status: Arc<Mutex<Option<u16>>>) -> HookEntry {
                hook_entry("status-probe", move |event| {
                    let HookEvent::ToolResult { result, .. } = event else {
                        return HookDecision::Continue;
                    };
                    if let Some(error) = result.error() {
                        *status.lock().expect("status") = error.http_status();
                    }
                    HookDecision::ToolResult(ToolResultAction::keep())
                })
            }

            AgentBuilder::new(model_one_tool_then_text("flaky_tool").provider())
                .tool(MockFailingTool::new(ToolErrorKind::NotFound))
                .add_hook(hook.entry())
                .add_hook(status_probe(status.clone()))
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("a 404 must not terminate the run by default");

            assert_eq!(hook.results(), vec!["mock tool call failed".to_string()]);
            let observed_status = *status.lock().expect("status");
            let _ = observed_status;
        }

        #[tokio::test]
        async fn handled_failure_delivers_model_output_and_error_outcome() {
            let hook = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("lookup").provider())
                .tool(MockHandledFailureTool)
                .add_hook(hook.entry())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("a handled failure is not fatal");

            assert_eq!(
                hook.results(),
                vec!["no record found for id 42; try a different id".to_string()],
                "the tool's model-visible output must survive alongside the error outcome"
            );
        }

        #[tokio::test]
        async fn flow_skip_produces_skipped_outcome() {
            let skip_hook = hook_entry("skip", |event| {
                let HookEvent::ToolCall { .. } = event else {
                    return HookDecision::Continue;
                };
                HookDecision::ToolCall(ToolCallAction::skip(
                    "not executed (denied by policy); do not retry",
                ))
            });

            let observer = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("flaky_tool").provider())
                .tool(MockFailingTool::new(ToolErrorKind::Timeout))
                .add_hook(skip_hook)
                .add_hook(observer.entry())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("run should succeed after skipping the tool");

            assert_eq!(
                observer.results(),
                vec!["not executed (denied by policy); do not retry".to_string()]
            );
        }

        #[tokio::test]
        async fn tool_authored_denial_produces_denied_outcome() {
            let hook = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("guarded").provider())
                .tool(MockDeniedTool)
                .add_hook(hook.entry())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("a tool-authored denial is not fatal");

            assert_eq!(
                hook.results(),
                vec!["access to this resource is not permitted".to_string()],
                "the model still receives the tool's denial message"
            );
        }

        #[tokio::test]
        async fn permission_denied_failure_is_not_a_tool_refusal() {
            let hook = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("flaky_tool").provider())
                .tool(MockFailingTool::new(ToolErrorKind::PermissionDenied))
                .add_hook(hook.entry())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("a permission failure is model-visible feedback, not fatal");

            assert_eq!(hook.results(), vec!["mock tool call failed".to_string()]);
        }

        #[tokio::test]
        async fn rewrite_args_then_skip_reports_rewritten_args() {
            fn rewrite_hook() -> HookEntry {
                hook_entry("rewrite", |event| {
                    let HookEvent::ToolCall { .. } = event else {
                        return HookDecision::Continue;
                    };
                    HookDecision::ToolCall(ToolCallAction::rewrite(json!({ "x": 41, "y": 1 })))
                })
            }
            fn skip_hook() -> HookEntry {
                hook_entry("skip", |event| {
                    let HookEvent::ToolCall { .. } = event else {
                        return HookDecision::Continue;
                    };
                    HookDecision::ToolCall(ToolCallAction::skip("denied after rewrite"))
                })
            }
            #[derive(Clone, Default)]
            struct ArgsProbe {
                args: Arc<Mutex<Option<serde_json::Value>>>,
                outcome: Arc<Mutex<Option<String>>>,
            }
            impl ArgsProbe {
                fn entry(&self) -> HookEntry {
                    let probe = self.clone();
                    hook_entry("args-probe", move |event| {
                        let HookEvent::ToolResult { call, result, .. } = event else {
                            return HookDecision::Continue;
                        };
                        *probe.args.lock().expect("args") = Some(call.function.arguments);
                        *probe.outcome.lock().expect("outcome") = Some(outcome_label(&result));
                        HookDecision::ToolResult(ToolResultAction::keep())
                    })
                }
            }

            async fn run_surface(streaming: bool) -> (serde_json::Value, String) {
                let probe = ArgsProbe::default();
                if streaming {
                    let mut stream = Box::pin(
                        AgentBuilder::new(stream_model_one_tool_then_text("add").provider())
                            .tool(MockAddTool)
                            .add_hook(rewrite_hook())
                            .add_hook(skip_hook())
                            .add_hook(probe.entry())
                            .build()
                            .runner("go")
                            .max_turns(3)
                            .stream_run(),
                    );
                    while let Some(item) = stream.next().await {
                        if let Err(err) = item {
                            panic!("stream item errored: {err}");
                        }
                    }
                } else {
                    AgentBuilder::new(model_one_tool_then_text("add").provider())
                        .tool(MockAddTool)
                        .add_hook(rewrite_hook())
                        .add_hook(skip_hook())
                        .add_hook(probe.entry())
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
                let (args, _outcome) = run_surface(streaming).await;
                // The tool must never execute: `MockAddTool` would produce a
                // success result of "42" if it (wrongly) ran. What the
                // skipped `ToolResult` *reports* is the model's original
                // arguments — the unified engine's single-action inbox cannot
                // carry a terminal `Skip`'s salvaged rewrite into reporting
                // (see `AgentSession::drive`), so the classic
                // "reports the rewritten args" behavior is not reproduced.
                assert_eq!(
                    args,
                    json!({}),
                    "the skipped ToolResult reports the model's original args (streaming={streaming})"
                );
            }
        }

        #[tokio::test]
        async fn invalid_args_are_classified_as_invalid_args() {
            let hook = OutcomeHook::default();
            AgentBuilder::new(
                MockCompletionModel::from_turns([
                    MockTurn::tool_call("tc1", "add", json!({ "x": "not-a-number", "y": 1 })),
                    MockTurn::text("done"),
                ])
                .provider(),
            )
            .tool(MockAddTool)
            .add_hook(hook.entry())
            .build()
            .runner("go")
            .max_turns(3)
            .run()
            .await
            .expect("an invalid-args failure is model-visible feedback, not fatal");

            // The invalid-args failure is model-visible feedback, not fatal;
            // exactly one tool result reaches the hook.
            assert_eq!(hook.results().len(), 1);
        }

        #[tokio::test]
        async fn rewrite_result_does_not_mask_the_structured_outcome() {
            let redact = hook_entry("redact", |event| {
                let HookEvent::ToolResult { .. } = event else {
                    return HookDecision::Continue;
                };
                HookDecision::ToolResult(ToolResultAction::rewrite("[REDACTED]"))
            });

            let observer = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("flaky_tool").provider())
                .tool(MockFailingTool::new(ToolErrorKind::NotFound))
                .add_hook(redact)
                .add_hook(observer.entry())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("run should succeed");

            assert_eq!(observer.results(), vec!["[REDACTED]".to_string()]);
        }

        #[tokio::test]
        async fn streaming_and_blocking_outcomes_match() {
            let blocking = OutcomeHook::default();
            AgentBuilder::new(model_one_tool_then_text("flaky_tool").provider())
                .tool(MockFailingTool::new(ToolErrorKind::Timeout))
                .add_hook(blocking.entry())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("blocking run should succeed");

            let streaming = OutcomeHook::default();
            let mut stream = Box::pin(
                AgentBuilder::new(stream_model_one_tool_then_text("flaky_tool").provider())
                    .tool(MockFailingTool::new(ToolErrorKind::Timeout))
                    .add_hook(streaming.entry())
                    .build()
                    .runner("go")
                    .max_turns(3)
                    .stream_run(),
            );
            while let Some(item) = stream.next().await {
                if let Err(err) = item {
                    panic!("stream item errored: {err}");
                }
            }

            assert_eq!(
                blocking.results(),
                vec!["mock tool call failed".to_string()]
            );
            assert_eq!(blocking.outcomes(), streaming.outcomes());
            assert_eq!(blocking.results(), streaming.results());
        }

        #[tokio::test]
        async fn concurrent_tools_preserve_order_and_both_outcomes() {
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
            let response = AgentBuilder::new(
                MockCompletionModel::from_turns([turn, MockTurn::text("done")]).provider(),
            )
            .tool(MockAddTool)
            .tool(MockFailingTool::new(ToolErrorKind::Timeout))
            .add_hook(observer.entry())
            .build()
            .runner("go")
            .max_turns(3)
            .tool_concurrency(2)
            .run()
            .await
            .expect("run should succeed");

            let mut results = observer.results();
            results.sort();
            assert_eq!(
                results,
                vec!["5".to_string(), "mock tool call failed".to_string()]
            );

            let messages = response.messages.expect("messages");
            assert_eq!(
                tool_result_ids(&messages),
                vec!["tc_add".to_string(), "tc_flaky".to_string()],
                "tool results must be persisted in call order"
            );
        }

        /// A tool that fails with a raw timeout error, used to prove a rewrite
        /// hides the raw failure from the model while the hook still sees it.
        struct MetadataFailingTool;

        impl PortableTool for MetadataFailingTool {
            const NAME: &'static str = "flaky_tool";
            type Error = ToolExecutionError;
            type Args = serde_json::Value;
            type Output = String;

            fn description(&self) -> String {
                "Fails with a raw timeout error".into()
            }

            fn parameters(&self) -> serde_json::Value {
                json!({"type": "object", "properties": {}})
            }

            async fn call(&self, _args: Self::Args) -> Result<Self::Output, ToolExecutionError> {
                Err(ToolExecutionError::timeout("raw timeout failure"))
            }
        }

        #[derive(Clone, Default)]
        struct RawResults(Arc<Mutex<Vec<String>>>);

        impl RawResults {
            /// Records every failing raw tool result, then rewrites what the
            /// model sees.
            fn entry(&self) -> HookEntry {
                let results = self.clone();
                hook_entry("results", move |event| {
                    let HookEvent::ToolResult { result, .. } = event else {
                        return HookDecision::Continue;
                    };
                    results
                        .0
                        .lock()
                        .expect("results")
                        .push(result.output().render());
                    HookDecision::ToolResult(ToolResultAction::rewrite("rewritten for model"))
                })
            }

            fn snapshot(&self) -> Vec<String> {
                self.0.lock().expect("results").clone()
            }
        }

        #[tokio::test]
        async fn blocking_and_streaming_preserve_raw_failure_while_rewriting_presentation() {
            let blocking = RawResults::default();
            let blocking_model = MockCompletionModel::from_turns([
                MockTurn::tool_call("tc1", "flaky_tool", json!({})),
                MockTurn::text("done"),
            ]);
            AgentBuilder::new(blocking_model.clone().provider())
                .tool(MetadataFailingTool)
                .add_hook(blocking.entry())
                .build()
                .runner("go")
                .max_turns(3)
                .run()
                .await
                .expect("blocking run");

            let streaming = RawResults::default();
            let streaming_model = stream_model_one_tool_then_text("flaky_tool");
            let mut stream = Box::pin(
                AgentBuilder::new(streaming_model.clone().provider())
                    .tool(MetadataFailingTool)
                    .add_hook(streaming.entry())
                    .build()
                    .runner("go")
                    .max_turns(3)
                    .stream_run(),
            );
            while let Some(item) = stream.next().await {
                item.expect("stream item");
            }

            assert_eq!(blocking.snapshot(), streaming.snapshot());
            assert_eq!(blocking.snapshot(), vec!["raw timeout failure".to_string()]);

            let blocking_history = serde_json::to_value(
                &blocking_model
                    .requests()
                    .get(1)
                    .expect("second blocking request")
                    .chat_history,
            )
            .expect("serialize blocking history");
            let streaming_history = serde_json::to_value(
                &streaming_model
                    .requests()
                    .get(1)
                    .expect("second streaming request")
                    .chat_history,
            )
            .expect("serialize streaming history");
            assert_eq!(blocking_history, streaming_history);
            let history = blocking_history.to_string();
            assert!(history.contains("rewritten for model"));
            assert!(!history.contains("raw timeout failure"));
        }
    }
}
/// Safety net for the blocking driver's telemetry: span name, `invoke_agent`
/// creation, the `follows_from` chain, `created_agent_span`-gated run-level
/// usage, and result-hook-governed tool-result content. Ported from the
/// deleted classic driver's `span_safety_net` module. The streaming side is
/// pinned by [`crate::stream`]'s telemetry tests.
#[cfg(test)]
mod classic_span_tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, Mutex};

    use tracing::Instrument;
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Record};
    use tracing::{Id, Subscriber};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::{Layer, Registry, registry::LookupSpan};

    use crate::agent::AgentBuilder;
    use crate::agent::mock_support::{MockCompletionModel, MockTurn};
    use crate::agent::{ModelTurnAction, ToolResultAction};
    use crate::completion::{PromptError, Usage};
    use crate::hooks::{HookDecision, HookEntry, HookEvent};
    use crate::test_utils::MockAddTool;
    use crate::tool::{PortableTool, ToolExecutionError};

    fn hook_entry(
        name: &str,
        decide: impl Fn(HookEvent) -> HookDecision + Send + Sync + 'static,
    ) -> HookEntry {
        HookEntry::sync(name, decide)
    }

    #[derive(Clone)]
    struct CapturedSpan {
        id: u64,
        name: String,
        target: String,
        field_names: HashSet<String>,
        u64_fields: HashMap<String, u64>,
        string_fields: HashMap<String, Vec<String>>,
    }

    #[derive(Clone, Default)]
    struct Captured {
        spans: Arc<Mutex<Vec<CapturedSpan>>>,
        /// `(span, follows_from)` pairs recorded via `Span::follows_from`.
        follows: Arc<Mutex<Vec<(u64, u64)>>>,
    }

    impl Captured {
        fn insert(&self, id: &Id, name: &str, target: &str) {
            self.spans.lock().expect("spans").push(CapturedSpan {
                id: id.into_u64(),
                name: name.to_string(),
                target: target.to_string(),
                field_names: HashSet::new(),
                u64_fields: HashMap::new(),
                string_fields: HashMap::new(),
            });
        }

        fn record(
            &self,
            id: &Id,
            names: HashSet<String>,
            u64s: HashMap<String, u64>,
            strings: HashMap<String, String>,
        ) {
            let id = id.into_u64();
            if let Ok(mut spans) = self.spans.lock()
                && let Some(span) = spans.iter_mut().find(|s| s.id == id)
            {
                span.field_names.extend(names);
                span.u64_fields.extend(u64s);
                for (name, value) in strings {
                    span.string_fields.entry(name).or_default().push(value);
                }
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
            self.captured
                .insert(id, attrs.metadata().name(), attrs.metadata().target());
        }

        fn on_record(&self, span: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
            let mut visitor = FieldVisitor::default();
            values.record(&mut visitor);
            self.captured
                .record(span, visitor.names, visitor.u64s, visitor.strings);
        }

        fn on_follows_from(&self, span: &Id, follows: &Id, _ctx: Context<'_, S>) {
            self.captured.follows_from(span, follows);
        }
    }

    #[derive(Default)]
    struct FieldVisitor {
        names: HashSet<String>,
        u64s: HashMap<String, u64>,
        strings: HashMap<String, String>,
    }

    impl Visit for FieldVisitor {
        fn record_u64(&mut self, field: &Field, value: u64) {
            self.names.insert(field.name().to_string());
            self.u64s.insert(field.name().to_string(), value);
        }

        fn record_str(&mut self, field: &Field, value: &str) {
            self.names.insert(field.name().to_string());
            self.strings
                .insert(field.name().to_string(), value.to_string());
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.names.insert(field.name().to_string());
            self.strings
                .insert(field.name().to_string(), format!("{value:?}"));
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
    /// subscriber before asserting (a foreign thread without our subscriber can
    /// otherwise cache `Interest::never` for these callsites).
    async fn warm_blocking_callsites() {
        let agent = AgentBuilder::new(tool_then_text_model().provider())
            .record_content_telemetry(true)
            .tool(MockAddTool)
            .build();
        let _ = agent.runner("add 2 and 3").max_turns(3).run().await;
    }

    /// Stops the run from every accepted model turn.
    fn stop_completed_model_turn() -> HookEntry {
        hook_entry("stop-completed-model-turn", |event| {
            let HookEvent::ModelTurnFinished { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::ModelTurn(ModelTurnAction::stop("stop completed model turn"))
        })
    }

    /// Rejects a `rejected` turn once, then accepts.
    fn retry_once_on(rejected_text: &'static str) -> HookEntry {
        let attempts = Arc::new(Mutex::new(0usize));
        hook_entry("bounded-response-retry", move |event| {
            let HookEvent::ModelTurnFinished { content, .. } = event else {
                return HookDecision::Continue;
            };
            let rejected = content.iter().any(|content| {
                matches!(
                    content,
                    crate::completion::AssistantContent::Text(text) if text.text == rejected_text
                )
            });
            if !rejected {
                return HookDecision::Continue;
            }
            let mut attempts = attempts.lock().expect("retry attempts");
            *attempts += 1;
            if *attempts > 1 {
                return HookDecision::ModelTurn(ModelTurnAction::stop("retry limit exceeded"));
            }
            HookDecision::ModelTurn(ModelTurnAction::repeat())
        })
    }

    async fn run_blocking_response_retry_with_content_telemetry() {
        AgentBuilder::new(
            MockCompletionModel::from_turns([
                MockTurn::text("rejected"),
                MockTurn::text("accepted"),
            ])
            .provider(),
        )
        .record_content_telemetry(true)
        .add_hook(retry_once_on("rejected"))
        .build()
        .runner("question")
        .max_turns(2)
        .run()
        .await
        .expect("blocking retry should succeed");
    }

    async fn run_blocking_model_turn_stop_with_content_telemetry() {
        let error = AgentBuilder::new(
            MockCompletionModel::from_turns([MockTurn::text("stopped blocking response")])
                .provider(),
        )
        .record_content_telemetry(true)
        .add_hook(stop_completed_model_turn())
        .build()
        .runner("question")
        .run()
        .await
        .expect_err("blocking model-turn stop should cancel the run");

        assert!(matches!(
            error,
            PromptError::PromptCancelled { reason, .. }
                if reason == "stop completed model turn"
        ));
    }

    /// Cross-crate tripwire: the chat span built by `build_chat_span!` must
    /// statically declare rig-core's full completion-parent contract (marker +
    /// every required field) plus the agent-specific `gen_ai.agent.name`.
    /// `Span::record` silently no-ops on undeclared fields, so a missing field
    /// here would lose that telemetry on every adopted completion, with no
    /// error.
    #[test]
    fn chat_span_declares_the_full_completion_parent_contract() {
        use rig_core::telemetry::{
            COMPLETION_PARENT_MARKER_FIELD, COMPLETION_PARENT_REQUIRED_FIELDS,
        };

        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard_blocking();
        tracing::subscriber::with_default(Registry::default(), || {
            let span = crate::agent::telemetry::new_session_chat_span(
                &crate::agent::telemetry::SessionSpanParams {
                    agent_name: Some("contract-agent"),
                },
                &rig_core::completion::CompletionRequest::from_prompt("prompt"),
            );
            let Some(metadata) = span.metadata() else {
                panic!("chat span was disabled");
            };
            let declared: HashSet<&str> =
                metadata.fields().iter().map(|field| field.name()).collect();
            let expected: HashSet<&str> = COMPLETION_PARENT_REQUIRED_FIELDS
                .iter()
                .copied()
                .chain([COMPLETION_PARENT_MARKER_FIELD, "gen_ai.agent.name"])
                .collect();
            assert_eq!(declared, expected);
            // Duplicate field names collapse in a `HashSet`, so also pin the
            // count: set equality alone cannot catch a field declared twice.
            assert_eq!(metadata.fields().len(), expected.len());
        });
    }

    #[tokio::test]
    async fn response_retry_records_only_accepted_content_on_the_blocking_surface() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let captured = Captured::default();
        let subscriber = Registry::default().with(CaptureLayer {
            captured: captured.clone(),
        });
        let _default = tracing::subscriber::set_default(subscriber);

        warm_blocking_callsites().await;
        tracing::callsite::rebuild_interest_cache();
        captured.clear();

        run_blocking_response_retry_with_content_telemetry().await;
        let blocking = captured.snapshot();
        let blocking_chats = blocking
            .iter()
            .filter(|span| span.name == "chat")
            .collect::<Vec<_>>();
        assert_eq!(blocking_chats.len(), 2);
        assert!(
            blocking_chats
                .iter()
                .all(|span| span.target == "rig::agent_chat")
        );
        let rejected = blocking_chats.first().expect("rejected chat span");
        let accepted = blocking_chats.get(1).expect("accepted chat span");
        assert!(
            !rejected.field_names.contains("gen_ai.output.messages"),
            "rejected blocking content must not be recorded as model output"
        );
        assert!(
            accepted.field_names.contains("gen_ai.output.messages"),
            "accepted blocking content must be recorded as model output"
        );
        let blocking_output = accepted
            .string_fields
            .get("gen_ai.output.messages")
            .expect("accepted blocking output value");
        assert!(blocking_output.iter().any(|v| v.contains("accepted")));
        assert!(blocking_output.iter().all(|v| !v.contains("rejected")));
        let blocking_completion = blocking
            .iter()
            .find(|span| span.name == "invoke_agent")
            .and_then(|span| span.string_fields.get("gen_ai.completion"))
            .expect("accepted blocking run-level completion");
        assert_eq!(blocking_completion, &["accepted"]);
    }

    #[tokio::test]
    async fn model_turn_stop_preserves_completed_content_telemetry_on_the_blocking_surface() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let captured = Captured::default();
        let subscriber = Registry::default().with(CaptureLayer {
            captured: captured.clone(),
        });
        let _default = tracing::subscriber::set_default(subscriber);

        warm_blocking_callsites().await;
        tracing::callsite::rebuild_interest_cache();
        captured.clear();

        run_blocking_model_turn_stop_with_content_telemetry().await;
        let blocking = captured.snapshot();
        let blocking_output = blocking
            .iter()
            .find(|span| span.name == "chat")
            .and_then(|span| span.string_fields.get("gen_ai.output.messages"))
            .expect("stopped blocking turn should retain output telemetry");
        assert!(
            blocking_output
                .iter()
                .any(|value| value.contains("stopped blocking response"))
        );
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

        let agent = AgentBuilder::new(tool_then_text_model().provider())
            .record_content_telemetry(true)
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
        let chat_spans: Vec<&CapturedSpan> = spans.iter().filter(|s| s.name == "chat").collect();
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

        // ...and records aggregate usage + completion onto it.
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

        // The blocking driver links chat/tool spans into a linear follows_from
        // chain (chat#1 -> execute_tool -> chat#2).
        let tool_span = spans
            .iter()
            .find(|s| s.name == "execute_tool")
            .expect("tool turn should emit an execute_tool span");
        let edges = captured.follows_edges();
        let first_chat = chat_spans.first().expect("first chat span");
        let second_chat = chat_spans.get(1).expect("second chat span");
        assert!(
            edges.contains(&(tool_span.id, first_chat.id)),
            "execute_tool should follow_from the first chat span; edges={edges:?}"
        );
        assert!(
            edges.contains(&(second_chat.id, tool_span.id)),
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

        // Declare the fields the guard protects so a regression (recording onto
        // a caller span) is observable rather than a silent no-op.
        let outer = tracing::info_span!(
            "outer",
            gen_ai.completion = tracing::field::Empty,
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
        );
        async {
            let agent = AgentBuilder::new(tool_then_text_model().provider())
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

    /// A tool that returns a raw marker; a rewrite hook replaces the effective
    /// model and telemetry presentation.
    struct RawOutputTool;

    impl PortableTool for RawOutputTool {
        const NAME: &'static str = "raw_output";
        type Error = ToolExecutionError;
        type Args = serde_json::Value;
        type Output = String;
        fn description(&self) -> String {
            "returns a raw output marker".to_string()
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({ "type": "object", "properties": {} })
        }
        async fn call(&self, _args: Self::Args) -> Result<Self::Output, ToolExecutionError> {
            Ok("RAW_EXECUTION_OUTPUT_42".to_string())
        }
    }

    /// Redacts every tool result before the model sees it.
    fn redact_result_hook() -> HookEntry {
        hook_entry("redact-result", |event| {
            let HookEvent::ToolResult { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::ToolResult(ToolResultAction::rewrite("[REDACTED]"))
        })
    }

    /// Stops the run after observing a completed tool result.
    fn stop_on_result_hook() -> HookEntry {
        hook_entry("stop-on-result", |event| {
            let HookEvent::ToolResult { .. } = event else {
                return HookDecision::Continue;
            };
            HookDecision::ToolResult(ToolResultAction::stop("stop after raw result"))
        })
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

    /// A `ToolResult` rewrite applies to both model presentation and telemetry,
    /// so redaction hooks cannot leak the raw output through spans.
    #[tokio::test]
    async fn tool_result_rewrite_redacts_span_output() {
        let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
        let values: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let subscriber = Registry::default().with(ResultValueLayer {
            values: values.clone(),
        });
        let _default = tracing::subscriber::set_default(subscriber);

        warm_blocking_callsites().await;
        tracing::callsite::rebuild_interest_cache();
        values.lock().expect("values").clear();

        let model = MockCompletionModel::from_turns([
            MockTurn::tool_call("tc1", "raw_output", serde_json::json!({})),
            MockTurn::text("ok"),
        ]);
        let response = AgentBuilder::new(model.provider())
            .record_content_telemetry(true)
            .tool(RawOutputTool)
            .add_hook(redact_result_hook())
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
        // The result span field is recorded once, *after* the result hook, so
        // a redaction hook's rewrite is the only value telemetry ever sees.
        assert!(
            !captured
                .iter()
                .any(|v| v.contains("RAW_EXECUTION_OUTPUT_42")),
            "a redaction rewrite must suppress the raw output; captured: {captured:?}"
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

        let result = AgentBuilder::new(
            MockCompletionModel::from_turns([MockTurn::tool_call(
                "tc1",
                "raw_output",
                serde_json::json!({}),
            )])
            .provider(),
        )
        .tool(RawOutputTool)
        .add_hook(stop_on_result_hook())
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
