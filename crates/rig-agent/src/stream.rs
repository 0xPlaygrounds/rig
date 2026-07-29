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
    CompletionCallAction, InvalidToolCallAction, ModelTurnAction, RequestPatch,
};
use crate::agent::prepare::{ToolCatalog, prepare_request};
use crate::agent::run::{
    AgentRun, AgentRunStep, DEFAULT_OUTPUT_RETRIES, PartialStreamedTurn, PendingToolCall,
    StreamedInvalidToolCall, StreamedResolution, StreamedTurnAssembler, StreamedTurnEvent,
};
use crate::agent::{AgentConfig, InvalidToolCallContext, PromptResponse};
use crate::completion::{Message, PromptError, Usage};
use rig_core::OneOrMany;
use rig_core::message::{AssistantContent, ToolCall, UserContent};
use rig_core::streaming::{
    StreamFinal, StreamedAssistantContent, StreamedUserContent, StreamingCompletionResponse,
};

use crate::session::SessionPolicy;
use crate::provider::{self, ProviderConfig, Runtime};

/// One item pulled from an [`AgentStream`].
///
/// Deliberately exhaustive, like
/// [`SessionEvent`](crate::session::SessionEvent): a new decision-bearing
/// variant must fail to compile in every streaming host.
#[derive(Debug, Clone)]
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
    /// Execute these calls and answer via
    /// [`AgentStream::provide_tool_results`].
    ToolCallsReady(Vec<PendingToolCall>),
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
}

/// The host decision the stream is waiting for.
enum Pending {
    None,
    BeforeCall { prompt: Message, history: Vec<Message> },
    TurnReply,
    Invalid {
        partial: PartialStreamedTurn,
        invalid: StreamedInvalidToolCall,
    },
    Tools { calls: Vec<PendingToolCall> },
    Finished,
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
}

impl AgentStream {
    /// Create a streaming session for one prompt.
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
            active: None,
            buffered: VecDeque::new(),
            last_final: None,
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
                Pending::BeforeCall { .. }
                | Pending::TurnReply
                | Pending::Invalid { .. }
                | Pending::Tools { .. } => {
                    return Some(Err(self
                        .run
                        .cancel_error("next_item called while a decision inbox awaits its answer")));
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
                self.pending = Pending::Tools { calls };
                Ok(())
            }
            AgentRunStep::Done(response) => {
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
        let stream = provider::open_stream(&self.provider, &self.rt, prepared.request)
            .await
            .map_err(PromptError::from)?;
        self.active = Some(ActiveTurn {
            stream,
            assembler: StreamedTurnAssembler::new(
                prepared.executable_tool_names,
                prepared.allowed_tool_names,
            ),
            turn,
        });
        Ok(())
    }

    /// Poll the open provider stream once, translating assembler events
    /// into buffered items or pending decisions.
    async fn poll_active_turn(&mut self) -> Result<(), PromptError> {
        let Some(active) = &mut self.active else {
            return Ok(());
        };
        match active.stream.next().await {
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
                            let partial =
                                active.assembler.partial_turn(active.stream.message_id.clone());
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
                            let call = self
                                .run
                                .record_streamed_completion_call(usage)?;
                            self.buffered
                                .push_back(AgentStreamItem::CompletionCall(call));
                            if let StreamedAssistantContent::Final(final_record) = &item {
                                self.last_final = Some(final_record.clone());
                                if emit_final {
                                    self.buffered.push_back(AgentStreamItem::Assistant(
                                        item.clone(),
                                    ));
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
                } = active;
                let streamed = assembler.finish(stream.message_id.clone(), &stream.choice);
                self.run.streamed_turn(streamed)?;
                if self.policy.surface_model_turns
                    && let Some(content) = self.run.accepted_turn_choice()
                {
                    self.pending = Pending::TurnReply;
                    self.buffered.push_back(AgentStreamItem::TurnFinished {
                        turn,
                        content,
                        usage: self.run.usage(),
                    });
                }
                Ok(())
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
                    if let Ok(call) = self.run.record_streamed_completion_call(drained_usage) {
                        self.buffered
                            .push_back(AgentStreamItem::CompletionCall(call));
                    }
                }
                self.buffered
                    .push_back(AgentStreamItem::ModelTurnRetried { turn });
                Ok(())
            }
        }
    }

    /// Answer [`AgentStreamItem::ToolCallsReady`]. Committed results and
    /// execution markers surface on subsequent [`AgentStream::next_item`]
    /// calls, in call order.
    pub fn provide_tool_results(
        &mut self,
        results: Vec<UserContent>,
    ) -> Result<(), PromptError> {
        let Pending::Tools { calls } = std::mem::replace(&mut self.pending, Pending::None) else {
            return Err(self
                .run
                .cancel_error("provide_tool_results without a pending ToolCallsReady item"));
        };
        self.run.tool_results(results.clone())?;
        // Post-commit surface items in call order.
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
            if let Some(result) = results.iter().find_map(|content| match content {
                UserContent::ToolResult(result) if result.id == call.tool_call.id => {
                    Some(result.clone())
                }
                _ => None,
            }) {
                self.buffered.push_back(AgentStreamItem::User(
                    StreamedUserContent::tool_result(result, internal),
                ));
            }
        }
        Ok(())
    }
}
