//! The blocking session driver: a concrete, callback-free agent loop.
//!
//! [`AgentSession`] pairs an [`AgentConfig`] with a [`ProviderConfig`] and
//! drives the sans-IO [`AgentRun`] protocol: [`AgentSession::advance`] runs
//! until the next event the [`SessionPolicy`] surfaces, and every decision
//! flows back in as a plain value through a decision inbox — a `match` in
//! the host's loop replaces callback registration. The whole session except
//! its [`Runtime`] handle is serializable between events.
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

use std::sync::Arc;

use crate::agent::hook::{
    CompletionCallAction, InvalidToolCallAction, ModelTurnAction, RequestPatch,
};
use crate::agent::prepare::{ToolCatalog, prepare_request};
use crate::agent::run::{
    AgentRun, AgentRunStep, DEFAULT_OUTPUT_RETRIES, ModelTurn, ModelTurnOutcome, PendingToolCall,
};
use crate::agent::{AgentConfig, InvalidToolCallContext, PromptResponse};
use crate::completion::{Message, PromptError, Usage};
use rig_core::OneOrMany;
use rig_core::completion::CompletionResponse;
use rig_core::message::{AssistantContent, UserContent};

use crate::provider::{self, ProviderConfig, Runtime};

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
    /// Always surfaced: execute these calls and answer via
    /// [`AgentSession::provide_tool_results`].
    ToolCallsReady(Vec<PendingToolCall>),
    /// The run is complete.
    Done(PromptResponse),
}

/// The host decision the session is currently waiting for.
#[derive(Debug)]
enum Pending {
    None,
    BeforeCall {
        prompt: Message,
        history: Vec<Message>,
    },
    TurnReply,
    Invalid {
        /// A chained invalid call surfaced by the previous resolution,
        /// returned by the next [`AgentSession::advance`].
        next: Option<InvalidToolCallContext>,
    },
    Tools,
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
}

impl AgentSession {
    /// Create a session for one prompt.
    pub fn new(
        config: AgentConfig,
        provider: ProviderConfig,
        rt: Arc<Runtime>,
        prompt: impl Into<Message>,
    ) -> Self {
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
        }
    }

    /// Resume a suspended run: pair a deserialized [`AgentRun`] with its
    /// configuration and a fresh runtime. The run re-emits its pending step
    /// idempotently ([`AgentRun::next_step`] semantics), so a process can
    /// serialize mid-tools and pick up exactly where it left off.
    pub fn resume(
        config: AgentConfig,
        provider: ProviderConfig,
        rt: Arc<Runtime>,
        run: AgentRun,
    ) -> Self {
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
            Pending::Tools => {
                return Err(self
                    .run
                    .cancel_error("advance called while tool results are awaited"));
            }
            Pending::None | Pending::BeforeCall { .. } | Pending::Invalid { .. } => {}
        }

        loop {
            // Resume a pre-build pause answered by reply_before_call.
            let (step, before_call_answered) =
                match std::mem::replace(&mut self.pending, Pending::None) {
                    Pending::BeforeCall { prompt, history } => (
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

                    let response =
                        provider::complete(&self.provider, &self.rt, prepared.request).await?;
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
                    self.pending = Pending::Tools;
                    return Ok(SessionEvent::ToolCallsReady(calls));
                }
                AgentRunStep::Done(response) => {
                    return Ok(SessionEvent::Done(response));
                }
            }
        }
    }

    /// Answer [`SessionEvent::BeforeModelCall`].
    ///
    /// # Errors
    /// [`CompletionCallAction::Stop`] cancels the run; calling without a
    /// pending pre-build event is a protocol violation.
    pub fn reply_before_call(&mut self, action: CompletionCallAction) -> Result<(), PromptError> {
        if !matches!(self.pending, Pending::BeforeCall { .. }) {
            return Err(self
                .run
                .cancel_error("reply_before_call without a pending BeforeModelCall event"));
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

    /// Answer [`SessionEvent::ToolCallsReady`] with one result per pending
    /// call (any order).
    pub fn provide_tool_results(
        &mut self,
        results: Vec<UserContent>,
    ) -> Result<(), PromptError> {
        if !matches!(self.pending, Pending::Tools) {
            return Err(self
                .run
                .cancel_error("provide_tool_results without a pending ToolCallsReady event"));
        }
        self.run.tool_results(results)?;
        self.pending = Pending::None;
        Ok(())
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
