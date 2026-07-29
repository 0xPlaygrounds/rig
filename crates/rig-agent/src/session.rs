//! The blocking session driver: a concrete, callback-free agent loop.
//!
//! [`AgentSession`] pairs an [`AgentConfig`] with a [`ProviderConfig`] and
//! drives the sans-IO [`AgentRun`] protocol: [`AgentSession::advance`] runs
//! until the next event the [`SessionPolicy`] surfaces, and every decision
//! flows back in as a plain value through a decision inbox — a `match` in
//! the host's loop replaces callback registration. The whole session except
//! its [`Runtime`] handle is serializable between events. One caveat:
//! [`SessionEvent::BeforeModelCall`] is not a durable suspension point — a
//! run serialized there (or while any model call was in flight) is resumed
//! by [`AgentSession::resume`] re-issuing the model call from the pre-call
//! state, so the call is re-prepared rather than picked up mid-flight.
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
use crate::agent::prompt_request::streaming::record_usage_on_span;
use crate::agent::prompt_request::tool_result_output;
use crate::agent::run::{
    AgentRun, AgentRunStep, DEFAULT_OUTPUT_RETRIES, ModelTurn, ModelTurnOutcome, PendingToolCall,
};
use crate::agent::runner::{SessionSpanParams, acquire_agent_span, new_session_chat_span};
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
/// [`AgentRunStep`](crate::agent::run::AgentRunStep): a new
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
        }
    }

    /// Resume a suspended run: pair a deserialized [`AgentRun`] with its
    /// configuration and a fresh runtime. The run re-emits its pending step
    /// idempotently ([`AgentRun::next_step`] semantics), so a process can
    /// serialize mid-tools and pick up exactly where it left off.
    ///
    /// A run suspended on an invalid tool-call decision re-derives its
    /// pending context: the next [`AgentSession::advance`] re-surfaces
    /// [`SessionEvent::InvalidToolCall`] for [`AgentSession::resolve_invalid`].
    /// A run serialized while a model call was in flight (for example after
    /// a transient provider error) recovers by re-issuing that call.
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

                    let chat_span = self.next_chat_span();
                    let response =
                        match provider::complete(&self.provider, &self.rt, prepared.request)
                            .instrument(chat_span)
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
                                self.pending = Pending::TurnReply;
                                return Ok(SessionEvent::TurnFinished {
                                    turn,
                                    content,
                                    usage,
                                });
                            }
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
    fn next_chat_span(&mut self) -> tracing::Span {
        let params = SessionSpanParams {
            record_telemetry_content: self.config.record_telemetry_content,
            agent_name: self.config.name.as_deref(),
        };
        let span = new_session_chat_span(&params, self.config.preamble.as_deref());
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
        match action {
            ModelTurnAction::Continue => Ok(()),
            ModelTurnAction::Retry(request) => self.run.retry_model_turn(request),
            ModelTurnAction::Stop(reason) => Err(self.run.cancel_error(reason)),
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
