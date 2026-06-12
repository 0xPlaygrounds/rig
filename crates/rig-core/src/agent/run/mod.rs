//! A sans-IO, steppable, serializable state machine for the agent prompt loop.
//!
//! [`AgentRun`] owns every *decision* the agent loop makes — turn counting,
//! tool-call validation, invalid tool-call recovery, chat-history threading,
//! usage aggregation and final response construction — without performing any
//! IO itself. A driver advances the machine by calling [`AgentRun::next_step`]
//! and acting on the returned [`AgentRunStep`]:
//!
//! - [`AgentRunStep::CallModel`]: send a completion request to the model and
//!   feed the result back via [`AgentRun::model_response`].
//! - [`AgentRunStep::CallTools`]: execute the listed tool calls (with whatever
//!   concurrency the driver chooses) and feed the results back via
//!   [`AgentRun::tool_results`].
//! - [`AgentRunStep::Done`]: the run is complete.
//!
//! Because the machine never awaits anything, it is runtime-agnostic and the
//! whole run state is `Serialize + Deserialize`: a driver can serialize a run
//! between steps (for example while tool calls are pending), persist it, and
//! resume it later in another process. Note that serialized run state embeds
//! the full conversation accumulated so far — persisting it inherits whatever
//! sensitivity the conversation content has.
//!
//! [`crate::completion::Prompt::prompt`] on [`crate::agent::Agent`] drives
//! this machine internally; the same machine can be driven by hand for custom
//! control flow:
//!
//! ```rust,no_run
//! use rig_core::agent::run::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut run = AgentRun::new("What is 2+2?").max_turns(3);
//! loop {
//!     match run.next_step()? {
//!         AgentRunStep::CallModel { prompt, history, .. } => {
//!             // Send `prompt` + `history` to a model, then:
//!             // run.model_response(ModelTurn { ... })?;
//!             # let _ = (prompt, history);
//!             # break;
//!         }
//!         AgentRunStep::CallTools { calls } => {
//!             // Execute `calls`, then: run.tool_results(results)?;
//!             # let _ = calls;
//!         }
//!         AgentRunStep::Done(response) => {
//!             println!("{}", response.output);
//!             break;
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```

pub mod streamed;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    OneOrMany,
    agent::prompt_request::{
        CompletionCall, PromptResponse, TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER,
        assistant_text_from_choice, build_full_history, build_history_for_request,
        hooks::{InvalidToolCallContext, InvalidToolCallHookAction},
        invalid_tool_retry_user_message, is_empty_assistant_turn, tool_result_user_content,
    },
    completion::{Message, PromptError, Usage},
    json_utils,
    message::{AssistantContent, ToolCall, ToolChoice, ToolResult, ToolResultContent, UserContent},
};

pub use streamed::{
    PartialStreamedTurn, StreamedInvalidToolCall, StreamedResolution, StreamedTurn,
    StreamedTurnAssembler, StreamedTurnEvent,
};

/// What a driver must do next to advance an [`AgentRun`].
///
/// Deliberately exhaustive: a driver must handle every step, so adding a
/// variant is a breaking change by design.
#[derive(Debug, Clone)]
pub enum AgentRunStep {
    /// Send a completion request to the model and feed the result back via
    /// [`AgentRun::model_response`].
    CallModel {
        /// The prompt message for this turn (the latest message in the run).
        prompt: Message,
        /// The chat history preceding `prompt`: the caller-provided input
        /// history followed by messages accumulated by earlier turns.
        history: Vec<Message>,
        /// One-based index of this model call within the run.
        turn: usize,
    },
    /// Execute these tool calls and feed the results back via
    /// [`AgentRun::tool_results`].
    CallTools {
        /// The tool calls of the current assistant turn, in emission order.
        calls: Vec<PendingToolCall>,
    },
    /// The run is complete.
    Done(PromptResponse),
}

/// One tool call awaiting execution by the driver.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PendingToolCall {
    /// The tool call emitted by the model (with any repaired tool name applied).
    pub tool_call: ToolCall,
    /// Pre-resolved result for tool calls suppressed by invalid tool-call
    /// recovery. When set, the driver must return this content as the tool
    /// result without executing the tool or invoking tool hooks.
    pub preresolved_result: Option<UserContent>,
}

/// A completed model turn fed back to [`AgentRun::model_response`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelTurn {
    /// Provider-assigned assistant message ID, when available.
    pub message_id: Option<String>,
    /// The assistant content returned by the model.
    pub choice: OneOrMany<AssistantContent>,
    /// Token usage reported by the provider for this completion request.
    pub usage: Usage,
    /// Executable Rig tools advertised to the provider for this turn.
    pub executable_tool_names: BTreeSet<String>,
    /// Tools allowed by the active [`ToolChoice`] for this turn.
    pub allowed_tool_names: BTreeSet<String>,
}

impl ModelTurn {
    /// Create a model turn from response parts and the tool names advertised
    /// for the turn.
    pub fn new(
        message_id: Option<String>,
        choice: OneOrMany<AssistantContent>,
        usage: Usage,
        executable_tool_names: BTreeSet<String>,
        allowed_tool_names: BTreeSet<String>,
    ) -> Self {
        Self {
            message_id,
            choice,
            usage,
            executable_tool_names,
            allowed_tool_names,
        }
    }
}

/// Result of feeding a model turn (or an invalid tool-call resolution) into
/// the machine.
///
/// Deliberately exhaustive: a driver must handle every outcome, so adding a
/// variant is a breaking change by design.
#[derive(Debug)]
pub enum ModelTurnOutcome {
    /// The turn was accepted. Unless `response_hook_suppressed` is set, the
    /// driver should run its completion-response hook now, then call
    /// [`AgentRun::next_step`].
    ///
    /// `response_hook_suppressed` is set when invalid tool-call recovery
    /// (repair or skip) modified the turn, matching the agent loop's behavior
    /// of not invoking `on_completion_response` for recovered turns.
    Continue {
        /// Whether the driver should suppress its completion-response hook.
        response_hook_suppressed: bool,
    },
    /// The model emitted a tool call that is unknown or disallowed for this
    /// turn. The driver must decide how to recover (typically by asking its
    /// invalid tool-call hook) and answer via
    /// [`AgentRun::resolve_invalid_tool_call`].
    NeedsResolution(InvalidToolCallContext),
    /// The turn was rolled back with corrective feedback appended to the
    /// history. Call [`AgentRun::next_step`] to obtain the retry
    /// [`AgentRunStep::CallModel`].
    TurnRetried,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResolvingState {
    message_id: Option<String>,
    /// The unmodified model output, used for diagnostic histories and retry
    /// messages (repairs are never reflected in those).
    original_choice: OneOrMany<AssistantContent>,
    /// Working copy of the assistant content; repairs rename tool calls here.
    items: Vec<AssistantContent>,
    /// Index of the next item to validate.
    next_index: usize,
    executable_tool_names: BTreeSet<String>,
    allowed_tool_names: BTreeSet<String>,
    /// Synthetic tool results for skipped tool calls, keyed by tool call ID.
    skipped: BTreeMap<String, UserContent>,
    recovered: bool,
    any_skipped: bool,
    has_tool_calls: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TurnState {
    message_id: Option<String>,
    items: Vec<AssistantContent>,
    has_tool_calls: bool,
    skipped: BTreeMap<String, UserContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RunState {
    /// Ready to emit [`AgentRunStep::CallModel`].
    PreparingRequest,
    /// Waiting for [`AgentRun::model_response`].
    AwaitingModel,
    /// Scanning the model turn's tool calls for validity; may be waiting for
    /// [`AgentRun::resolve_invalid_tool_call`].
    ResolvingToolCalls(Box<ResolvingState>),
    /// The turn was accepted; ready to emit [`AgentRunStep::CallTools`] or
    /// [`AgentRunStep::Done`].
    AwaitingAdvance(Box<TurnState>),
    /// Waiting for [`AgentRun::tool_results`] for these pending tool calls.
    /// Carrying the calls in the state keeps a serialized run self-contained:
    /// a resumed process re-obtains them from [`AgentRun::next_step`].
    ExecutingTools(Vec<PendingToolCall>),
    /// Terminal: the run completed successfully.
    Done(Box<PromptResponse>),
    /// Terminal: the run returned an error.
    Failed,
}

/// The sans-IO agent loop state machine. See the [module docs](self) for the
/// driving protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRun {
    max_turns: usize,
    max_invalid_tool_call_retries: usize,
    tool_choice: Option<ToolChoice>,
    chat_history: Option<Vec<Message>>,
    new_messages: Vec<Message>,
    current_turn: usize,
    usage: Usage,
    completion_calls: Vec<CompletionCall>,
    completion_call_index: usize,
    invalid_tool_call_retries: usize,
    /// Set while a streamed turn rollback awaits its completion-call record;
    /// see [`AgentRun::record_streamed_completion_call`].
    #[serde(default)]
    rollback_pending: bool,
    state: RunState,
}

impl AgentRun {
    /// Create a run for one prompt with no input history, no multi-turn depth
    /// and no invalid tool-call retries.
    pub fn new(prompt: impl Into<Message>) -> Self {
        Self {
            max_turns: 0,
            max_invalid_tool_call_retries: 0,
            tool_choice: None,
            chat_history: None,
            new_messages: vec![prompt.into()],
            current_turn: 0,
            usage: Usage::new(),
            completion_calls: Vec::new(),
            completion_call_index: 0,
            invalid_tool_call_retries: 0,
            rollback_pending: false,
            state: RunState::PreparingRequest,
        }
    }

    /// Set the input chat history preceding the prompt.
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.chat_history = Some(history);
        self
    }

    /// Set the maximum multi-turn depth. Exceeding it makes
    /// [`AgentRun::next_step`] return [`PromptError::MaxTurnsError`].
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Set the retry budget for [`InvalidToolCallHookAction::Retry`]
    /// resolutions. Invalid tool-call retries also consume multi-turn depth.
    pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
        self.max_invalid_tool_call_retries = retries;
        self
    }

    /// Set the tool choice active for this run. Used to reject
    /// [`InvalidToolCallHookAction::Skip`] resolutions under
    /// [`ToolChoice::None`] and reported in invalid tool-call contexts.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Aggregated token usage across all completed model calls so far.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Number of model calls emitted so far (including retries).
    pub fn turn(&self) -> usize {
        self.current_turn
    }

    /// Details for each completed model call so far.
    pub fn completion_calls(&self) -> &[CompletionCall] {
        &self.completion_calls
    }

    /// Messages accumulated by this run (the prompt plus all assistant turns
    /// and tool results), excluding the input history.
    pub fn messages(&self) -> &[Message] {
        &self.new_messages
    }

    /// The full conversation: input history followed by [`Self::messages`].
    pub fn full_history(&self) -> Vec<Message> {
        build_full_history(self.chat_history.as_deref(), self.new_messages.clone())
    }

    /// Whether the run reached [`AgentRunStep::Done`].
    pub fn is_done(&self) -> bool {
        matches!(self.state, RunState::Done(_))
    }

    /// The final response once the run is done, without cloning it.
    /// [`AgentRun::next_step`] in the done state returns an owned clone
    /// (including the full accumulated message history); prefer this when
    /// only inspecting the result.
    pub fn response(&self) -> Option<&PromptResponse> {
        match &self.state {
            RunState::Done(response) => Some(response),
            _ => None,
        }
    }

    /// Build the cancellation error a driver should return when one of its
    /// hooks terminates the run, carrying the current full history.
    pub fn cancel_error(&self, reason: impl Into<String>) -> PromptError {
        PromptError::prompt_cancelled(self.full_history(), reason)
    }

    /// The invalid tool call currently awaiting
    /// [`AgentRun::resolve_invalid_tool_call`], if any. Useful to re-derive
    /// the resolution context after deserializing a suspended run.
    pub fn pending_invalid_tool_call(&self) -> Option<InvalidToolCallContext> {
        let RunState::ResolvingToolCalls(resolving) = &self.state else {
            return None;
        };
        let AssistantContent::ToolCall(tool_call) = resolving.items.get(resolving.next_index)?
        else {
            return None;
        };
        if resolving
            .allowed_tool_names
            .contains(&tool_call.function.name)
        {
            return None;
        }

        Some(InvalidToolCallContext {
            tool_name: tool_call.function.name.clone(),
            tool_call_id: Some(tool_call.id.clone()),
            internal_call_id: None,
            args: Some(json_utils::value_to_json_string(
                &tool_call.function.arguments,
            )),
            available_tools: resolving.executable_tool_names.iter().cloned().collect(),
            allowed_tools: resolving.allowed_tool_names.iter().cloned().collect(),
            tool_choice: self.tool_choice.clone(),
            chat_history: self.diagnostic_history(resolving),
            is_streaming: false,
        })
    }

    /// Advance the machine and return the next action for the driver.
    ///
    /// # Errors
    /// - [`PromptError::MaxTurnsError`] when the multi-turn depth is exhausted.
    /// - [`PromptError::PromptCancelled`] when the machine is driven out of
    ///   protocol (for example, calling this while a model response is
    ///   pending).
    pub fn next_step(&mut self) -> Result<AgentRunStep, PromptError> {
        match std::mem::replace(&mut self.state, RunState::Failed) {
            RunState::PreparingRequest => {
                let Some((prompt_ref, history_for_turn)) = self.new_messages.split_last() else {
                    return Err(PromptError::prompt_cancelled(
                        self.full_history(),
                        "prompt loop lost its pending prompt",
                    ));
                };
                let prompt = prompt_ref.clone();

                if self.current_turn > self.max_turns + 1 {
                    return Err(PromptError::MaxTurnsError {
                        max_turns: self.max_turns,
                        chat_history: self.full_history().into(),
                        prompt: prompt.into(),
                    });
                }

                let history =
                    build_history_for_request(self.chat_history.as_deref(), history_for_turn);
                self.current_turn += 1;
                self.rollback_pending = false;
                self.state = RunState::AwaitingModel;
                Ok(AgentRunStep::CallModel {
                    prompt,
                    history,
                    turn: self.current_turn,
                })
            }
            RunState::AwaitingAdvance(turn_state) => {
                let TurnState {
                    message_id,
                    items,
                    has_tool_calls,
                    skipped,
                } = *turn_state;
                let Some(choice) = OneOrMany::from_iter_optional(items.clone()) else {
                    return Err(PromptError::prompt_cancelled(
                        self.full_history(),
                        "model turn lost its assistant content",
                    ));
                };

                if !is_empty_assistant_turn(&choice) {
                    self.new_messages.push(Message::Assistant {
                        id: message_id,
                        content: choice.clone(),
                    });
                }

                if has_tool_calls {
                    let calls: Vec<PendingToolCall> = items
                        .iter()
                        .filter_map(|item| match item {
                            AssistantContent::ToolCall(tool_call) => Some(PendingToolCall {
                                tool_call: tool_call.clone(),
                                preresolved_result: skipped.get(&tool_call.id).cloned(),
                            }),
                            _ => None,
                        })
                        .collect();
                    self.state = RunState::ExecutingTools(calls.clone());
                    Ok(AgentRunStep::CallTools { calls })
                } else {
                    let response =
                        PromptResponse::new(assistant_text_from_choice(&choice), self.usage)
                            .with_messages(self.new_messages.clone())
                            .with_completion_calls(self.completion_calls.clone());
                    self.state = RunState::Done(Box::new(response.clone()));
                    Ok(AgentRunStep::Done(response))
                }
            }
            RunState::ExecutingTools(calls) => {
                // Idempotent, like Done: a process resuming a serialized run
                // re-obtains the pending tool calls from the state itself.
                let step = AgentRunStep::CallTools {
                    calls: calls.clone(),
                };
                self.state = RunState::ExecutingTools(calls);
                Ok(step)
            }
            RunState::Done(response) => {
                let step = AgentRunStep::Done((*response).clone());
                self.state = RunState::Done(response);
                Ok(step)
            }
            state @ (RunState::AwaitingModel | RunState::ResolvingToolCalls(_)) => {
                let reason = match &state {
                    RunState::AwaitingModel => {
                        "next_step called while a model response is pending; feed it via model_response first"
                    }
                    _ => {
                        "next_step called while an invalid tool-call resolution is pending; answer it via resolve_invalid_tool_call first"
                    }
                };
                self.state = state;
                Err(self.protocol_violation(reason))
            }
            RunState::Failed => Err(self.protocol_violation(
                "next_step called after the run already failed or was misdriven",
            )),
        }
    }

    /// Feed the model's response for the pending [`AgentRunStep::CallModel`].
    ///
    /// Records the completion call and aggregates usage, then validates the
    /// turn's tool calls against the advertised tool names. See
    /// [`ModelTurnOutcome`] for what the driver must do next.
    pub fn model_response(&mut self, turn: ModelTurn) -> Result<ModelTurnOutcome, PromptError> {
        if !matches!(self.state, RunState::AwaitingModel) {
            return Err(
                self.protocol_violation("model_response called without a pending CallModel step")
            );
        }

        self.completion_calls
            .push(CompletionCall::from_reported_usage(
                self.completion_call_index,
                turn.usage,
            ));
        self.completion_call_index += 1;
        self.usage += turn.usage;

        let items: Vec<AssistantContent> = turn.choice.iter().cloned().collect();
        let has_tool_calls = items
            .iter()
            .any(|item| matches!(item, AssistantContent::ToolCall(_)));

        self.state = RunState::ResolvingToolCalls(Box::new(ResolvingState {
            message_id: turn.message_id,
            original_choice: turn.choice,
            items,
            next_index: 0,
            executable_tool_names: turn.executable_tool_names,
            allowed_tool_names: turn.allowed_tool_names,
            skipped: BTreeMap::new(),
            recovered: false,
            any_skipped: false,
            has_tool_calls,
        }));

        self.advance_resolution()
    }

    /// Answer a pending [`ModelTurnOutcome::NeedsResolution`].
    ///
    /// Applies the agent loop's recovery semantics:
    /// - [`InvalidToolCallHookAction::Fail`] fails the run with
    ///   [`PromptError::UnknownToolCall`].
    /// - [`InvalidToolCallHookAction::Retry`] rolls the turn back with
    ///   corrective feedback while budget remains, consuming multi-turn depth.
    /// - [`InvalidToolCallHookAction::Repair`] renames the tool call; the
    ///   repaired name is revalidated against the allowed tools.
    /// - [`InvalidToolCallHookAction::Skip`] records a synthetic tool result
    ///   and suppresses execution of every tool call in the turn. Rejected
    ///   under [`ToolChoice::None`].
    pub fn resolve_invalid_tool_call(
        &mut self,
        action: InvalidToolCallHookAction,
    ) -> Result<ModelTurnOutcome, PromptError> {
        // Take the resolving state; rejection paths below restore it so an
        // out-of-protocol call does not corrupt a drivable run.
        let mut resolving = match std::mem::replace(&mut self.state, RunState::Failed) {
            RunState::ResolvingToolCalls(resolving) => resolving,
            other => {
                self.state = other;
                return Err(self.protocol_violation(
                    "resolve_invalid_tool_call called without a pending invalid tool call",
                ));
            }
        };
        let tool_call = match resolving.items.get(resolving.next_index) {
            Some(AssistantContent::ToolCall(tool_call))
                if !resolving
                    .allowed_tool_names
                    .contains(&tool_call.function.name) =>
            {
                tool_call.clone()
            }
            _ => {
                self.state = RunState::ResolvingToolCalls(resolving);
                return Err(self.protocol_violation(
                    "resolve_invalid_tool_call called without a pending invalid tool call",
                ));
            }
        };

        let diagnostic_history = self.diagnostic_history(&resolving);
        let executable_tool_names: Vec<String> =
            resolving.executable_tool_names.iter().cloned().collect();
        let allowed_tool_names: Vec<String> =
            resolving.allowed_tool_names.iter().cloned().collect();

        match action {
            InvalidToolCallHookAction::Fail => Err(PromptError::UnknownToolCall {
                tool_name: tool_call.function.name,
                available_tools: executable_tool_names,
                allowed_tools: allowed_tool_names,
                chat_history: Box::new(diagnostic_history),
            }),
            InvalidToolCallHookAction::Retry { feedback } => {
                if self.invalid_tool_call_retries >= self.max_invalid_tool_call_retries {
                    return Err(PromptError::UnknownToolCall {
                        tool_name: tool_call.function.name,
                        available_tools: executable_tool_names,
                        allowed_tools: allowed_tool_names,
                        chat_history: Box::new(diagnostic_history),
                    });
                }
                self.invalid_tool_call_retries += 1;

                self.new_messages.push(Message::Assistant {
                    id: resolving.message_id.clone(),
                    content: resolving.original_choice.clone(),
                });
                let Some(user_message) = invalid_tool_retry_user_message(
                    &resolving.original_choice,
                    &tool_call.id,
                    feedback,
                ) else {
                    return Err(PromptError::prompt_cancelled(
                        diagnostic_history,
                        "invalid tool call retry produced no retry messages",
                    ));
                };
                self.new_messages.push(user_message);
                self.state = RunState::PreparingRequest;
                Ok(ModelTurnOutcome::TurnRetried)
            }
            InvalidToolCallHookAction::Repair { tool_name } => {
                if !allowed_tool_names.contains(&tool_name) {
                    return Err(PromptError::UnknownToolCall {
                        tool_name,
                        available_tools: executable_tool_names,
                        allowed_tools: allowed_tool_names,
                        chat_history: Box::new(diagnostic_history),
                    });
                }
                if let Some(AssistantContent::ToolCall(tool_call)) =
                    resolving.items.get_mut(resolving.next_index)
                {
                    tool_call.function.name = tool_name;
                }
                resolving.recovered = true;
                self.state = RunState::ResolvingToolCalls(resolving);
                self.advance_resolution()
            }
            InvalidToolCallHookAction::Skip { reason } => {
                if matches!(self.tool_choice, Some(ToolChoice::None)) {
                    return Err(PromptError::UnknownToolCall {
                        tool_name: tool_call.function.name,
                        available_tools: executable_tool_names,
                        allowed_tools: allowed_tool_names,
                        chat_history: Box::new(diagnostic_history),
                    });
                }
                let user_content = if let Some(call_id) = tool_call.call_id.clone() {
                    UserContent::tool_result_with_call_id(
                        tool_call.id.clone(),
                        call_id,
                        OneOrMany::one(reason.into()),
                    )
                } else {
                    UserContent::tool_result(tool_call.id.clone(), OneOrMany::one(reason.into()))
                };
                resolving.skipped.insert(tool_call.id.clone(), user_content);
                resolving.recovered = true;
                resolving.any_skipped = true;
                resolving.next_index += 1;
                self.state = RunState::ResolvingToolCalls(resolving);
                self.advance_resolution()
            }
        }
    }

    /// Feed the tool results for the pending [`AgentRunStep::CallTools`].
    ///
    /// Results may be in any order; they are appended as a single user
    /// message, matching what providers expect for parallel tool calls. Each
    /// result must be a tool result answering one of the pending calls, and
    /// every pending call must be answered — exactly what providers require
    /// to accept the next request.
    pub fn tool_results(&mut self, results: Vec<UserContent>) -> Result<(), PromptError> {
        let RunState::ExecutingTools(pending) = &self.state else {
            return Err(
                self.protocol_violation("tool_results called without a pending CallTools step")
            );
        };
        // Match results against pending calls by tool call ID as a multiset,
        // so duplicate provider IDs within one turn stay answerable.
        let mut unanswered: Vec<String> = pending
            .iter()
            .map(|call| call.tool_call.id.clone())
            .collect();

        if results.is_empty() {
            self.state = RunState::Failed;
            return Err(PromptError::prompt_cancelled(
                self.full_history(),
                "tool execution produced no tool results",
            ));
        }
        for result in &results {
            let UserContent::ToolResult(tool_result) = result else {
                return Err(self.protocol_violation(
                    "tool_results received content that is not a tool result",
                ));
            };
            let Some(index) = unanswered.iter().position(|id| *id == tool_result.id) else {
                return Err(self.protocol_violation(&format!(
                    "tool_results received a result for unknown or already-answered tool call id `{}`",
                    tool_result.id
                )));
            };
            unanswered.swap_remove(index);
        }
        if !unanswered.is_empty() {
            return Err(self.protocol_violation(&format!(
                "tool_results left pending tool call id(s) unanswered: {unanswered:?}"
            )));
        }

        let Some(content) = OneOrMany::from_iter_optional(results) else {
            self.state = RunState::Failed;
            return Err(PromptError::prompt_cancelled(
                self.full_history(),
                "tool execution produced no tool results",
            ));
        };

        self.new_messages.push(Message::User { content });
        self.state = RunState::PreparingRequest;
        Ok(())
    }

    /// Scan forward for the next invalid tool call; finish the turn when the
    /// scan completes.
    fn advance_resolution(&mut self) -> Result<ModelTurnOutcome, PromptError> {
        let mut resolving = match std::mem::replace(&mut self.state, RunState::Failed) {
            RunState::ResolvingToolCalls(resolving) => resolving,
            other => {
                self.state = other;
                return Err(self.protocol_violation(
                    "internal: advance_resolution outside of tool-call resolution",
                ));
            }
        };
        while let Some(item) = resolving.items.get(resolving.next_index) {
            match item {
                AssistantContent::ToolCall(tool_call)
                    if !resolving
                        .allowed_tool_names
                        .contains(&tool_call.function.name) =>
                {
                    break;
                }
                _ => resolving.next_index += 1,
            }
        }

        if resolving.next_index < resolving.items.len() {
            self.state = RunState::ResolvingToolCalls(resolving);
            return match self.pending_invalid_tool_call() {
                Some(context) => Ok(ModelTurnOutcome::NeedsResolution(context)),
                None => Err(self.protocol_violation(
                    "internal: pending invalid tool call could not be derived",
                )),
            };
        }

        let ResolvingState {
            message_id,
            items,
            mut skipped,
            recovered,
            any_skipped,
            has_tool_calls,
            ..
        } = *resolving;

        // When any tool call was skipped, none of the turn's tool calls
        // execute: peers get a synthetic "not executed" result.
        if any_skipped {
            for item in &items {
                if let AssistantContent::ToolCall(tool_call) = item {
                    skipped.entry(tool_call.id.clone()).or_insert_with(|| {
                        tool_result_user_content(
                            tool_call.id.clone(),
                            tool_call.call_id.clone(),
                            TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_string(),
                        )
                    });
                }
            }
        }

        self.state = RunState::AwaitingAdvance(Box::new(TurnState {
            message_id,
            items,
            has_tool_calls,
            skipped,
        }));
        Ok(ModelTurnOutcome::Continue {
            response_hook_suppressed: recovered,
        })
    }

    // ── Streamed-turn entry points ──────────────────────────────────────
    // Paired with [`streamed::StreamedTurnAssembler`]; see that module's
    // docs for the full driving protocol.

    /// Record one provider completion call for a streamed turn.
    ///
    /// Streamed turns learn usage from the provider's final stream event —
    /// including for turns abandoned by invalid tool-call recovery, where the
    /// stream is drained for usage after the rollback — so recording is
    /// decoupled from turn ingestion. Valid while a model response is pending
    /// or between a turn rollback and the next [`AgentRunStep::CallModel`];
    /// aggregates `usage` into the run total.
    pub fn record_streamed_completion_call(
        &mut self,
        usage: Option<Usage>,
    ) -> Result<CompletionCall, PromptError> {
        let recordable = matches!(self.state, RunState::AwaitingModel)
            || (matches!(self.state, RunState::PreparingRequest) && self.rollback_pending);
        if !recordable {
            return Err(self.protocol_violation(
                "record_streamed_completion_call called without a pending or rolled-back CallModel step",
            ));
        }

        let call = CompletionCall::new(self.completion_call_index, usage);
        self.completion_call_index += 1;
        self.completion_calls.push(call);
        if let Some(usage) = usage {
            self.usage += usage;
        }
        Ok(call)
    }

    /// The recovery-hook context for an invalid tool call surfaced
    /// mid-stream by a [`streamed::StreamedTurnAssembler`].
    pub fn streamed_invalid_tool_call_context(
        &self,
        partial: &PartialStreamedTurn,
        invalid: &StreamedInvalidToolCall,
    ) -> InvalidToolCallContext {
        InvalidToolCallContext {
            tool_name: invalid.tool_call.function.name.clone(),
            tool_call_id: Some(invalid.tool_call.id.clone()),
            internal_call_id: Some(invalid.internal_call_id.clone()),
            args: invalid.args.clone(),
            available_tools: invalid.executable_tool_names.iter().cloned().collect(),
            allowed_tools: invalid.allowed_tool_names.iter().cloned().collect(),
            tool_choice: self.tool_choice.clone(),
            chat_history: self
                .streamed_diagnostic_history(partial, Some(invalid.tool_call.clone())),
            is_streaming: true,
        }
    }

    /// Resolve an invalid tool call surfaced mid-stream.
    ///
    /// Applies the same recovery semantics as
    /// [`AgentRun::resolve_invalid_tool_call`], but rollback messages are
    /// assembled from the partial streamed turn — exactly what the model has
    /// produced so far — and a successful retry or skip abandons the turn
    /// (see [`StreamedResolution`]) instead of finishing it.
    pub fn resolve_streamed_invalid_tool_call(
        &mut self,
        partial: &PartialStreamedTurn,
        invalid: &StreamedInvalidToolCall,
        action: InvalidToolCallHookAction,
    ) -> Result<StreamedResolution, PromptError> {
        if !matches!(self.state, RunState::AwaitingModel) {
            return Err(self.protocol_violation(
                "resolve_streamed_invalid_tool_call called without a pending CallModel step",
            ));
        }

        let diagnostic_history =
            self.streamed_diagnostic_history(partial, Some(invalid.tool_call.clone()));
        let executable_tool_names: Vec<String> =
            invalid.executable_tool_names.iter().cloned().collect();
        let allowed_tool_names: Vec<String> = invalid.allowed_tool_names.iter().cloned().collect();

        match action {
            InvalidToolCallHookAction::Fail => {
                self.state = RunState::Failed;
                Err(PromptError::UnknownToolCall {
                    tool_name: invalid.tool_call.function.name.clone(),
                    available_tools: executable_tool_names,
                    allowed_tools: allowed_tool_names,
                    chat_history: Box::new(diagnostic_history),
                })
            }
            InvalidToolCallHookAction::Retry { feedback } => {
                if self.invalid_tool_call_retries >= self.max_invalid_tool_call_retries {
                    self.state = RunState::Failed;
                    return Err(PromptError::UnknownToolCall {
                        tool_name: invalid.tool_call.function.name.clone(),
                        available_tools: executable_tool_names,
                        allowed_tools: allowed_tool_names,
                        chat_history: Box::new(diagnostic_history),
                    });
                }
                self.invalid_tool_call_retries += 1;

                let Some((assistant_message, user_message)) =
                    partial.rollback_messages(invalid.tool_call.clone(), feedback)
                else {
                    self.state = RunState::Failed;
                    return Err(PromptError::prompt_cancelled(
                        diagnostic_history,
                        "invalid tool call retry produced no retry messages",
                    ));
                };
                self.new_messages.push(assistant_message);
                self.new_messages.push(user_message);
                self.rollback_pending = true;
                self.state = RunState::PreparingRequest;
                Ok(StreamedResolution::TurnAbandoned {
                    skipped_tool_result: None,
                })
            }
            InvalidToolCallHookAction::Repair { tool_name } => {
                if !invalid.allowed_tool_names.contains(&tool_name) {
                    self.state = RunState::Failed;
                    return Err(PromptError::UnknownToolCall {
                        tool_name,
                        available_tools: executable_tool_names,
                        allowed_tools: allowed_tool_names,
                        chat_history: Box::new(diagnostic_history),
                    });
                }
                Ok(StreamedResolution::Repaired { tool_name })
            }
            InvalidToolCallHookAction::Skip { reason } => {
                if matches!(self.tool_choice, Some(ToolChoice::None)) {
                    self.state = RunState::Failed;
                    return Err(PromptError::UnknownToolCall {
                        tool_name: invalid.tool_call.function.name.clone(),
                        available_tools: executable_tool_names,
                        allowed_tools: allowed_tool_names,
                        chat_history: Box::new(diagnostic_history),
                    });
                }

                let skipped_tool_result = ToolResult {
                    id: invalid.tool_call.id.clone(),
                    call_id: invalid.tool_call.call_id.clone(),
                    content: ToolResultContent::from_tool_output(reason.clone()),
                };
                let Some((assistant_message, user_message)) =
                    partial.rollback_messages(invalid.tool_call.clone(), reason)
                else {
                    self.state = RunState::Failed;
                    return Err(PromptError::prompt_cancelled(
                        diagnostic_history,
                        "invalid tool call skip produced no recovery messages",
                    ));
                };
                self.new_messages.push(assistant_message);
                self.new_messages.push(user_message);
                self.rollback_pending = true;
                self.state = RunState::PreparingRequest;
                Ok(StreamedResolution::TurnAbandoned {
                    skipped_tool_result: Some(skipped_tool_result),
                })
            }
        }
    }

    /// Feed the assembled streamed turn for the pending
    /// [`AgentRunStep::CallModel`].
    ///
    /// Remaining tool calls are validated fail-fast — mid-stream resolution
    /// already had recovery-hook access — and the turn then advances through
    /// [`AgentRun::next_step`] exactly like a non-streamed one.
    pub fn streamed_turn(&mut self, turn: StreamedTurn) -> Result<(), PromptError> {
        if !matches!(self.state, RunState::AwaitingModel) {
            return Err(
                self.protocol_violation("streamed_turn called without a pending CallModel step")
            );
        }

        let items: Vec<AssistantContent> = turn.choice.iter().cloned().collect();
        let has_tool_calls = items
            .iter()
            .any(|item| matches!(item, AssistantContent::ToolCall(_)));

        for item in &items {
            let AssistantContent::ToolCall(tool_call) = item else {
                continue;
            };
            if !turn.allowed_tool_names.contains(&tool_call.function.name) {
                let mut diagnostic_messages = self.new_messages.clone();
                if !is_empty_assistant_turn(&turn.choice) {
                    diagnostic_messages.push(Message::Assistant {
                        id: turn.message_id.clone(),
                        content: turn.choice.clone(),
                    });
                }
                let diagnostic_history =
                    build_full_history(self.chat_history.as_deref(), diagnostic_messages);
                self.state = RunState::Failed;
                return Err(PromptError::UnknownToolCall {
                    tool_name: tool_call.function.name.clone(),
                    available_tools: turn.executable_tool_names.iter().cloned().collect(),
                    allowed_tools: turn.allowed_tool_names.iter().cloned().collect(),
                    chat_history: Box::new(diagnostic_history),
                });
            }
        }

        self.state = RunState::AwaitingAdvance(Box::new(TurnState {
            message_id: turn.message_id,
            items,
            has_tool_calls,
            skipped: BTreeMap::new(),
        }));
        Ok(())
    }

    /// Diagnostic history for a streamed turn: the run's messages plus the
    /// partial assistant turn under inspection.
    fn streamed_diagnostic_history(
        &self,
        partial: &PartialStreamedTurn,
        current_tool_call: Option<ToolCall>,
    ) -> Vec<Message> {
        let mut messages = self.new_messages.clone();
        if let Some(assistant) = partial.assistant_message(current_tool_call) {
            messages.push(assistant);
        }
        build_full_history(self.chat_history.as_deref(), messages)
    }

    /// History used for invalid tool-call diagnostics: the run's messages plus
    /// the unmodified assistant turn under inspection.
    fn diagnostic_history(&self, resolving: &ResolvingState) -> Vec<Message> {
        let mut diagnostic_messages = self.new_messages.clone();
        diagnostic_messages.push(Message::Assistant {
            id: resolving.message_id.clone(),
            content: resolving.original_choice.clone(),
        });
        build_full_history(self.chat_history.as_deref(), diagnostic_messages)
    }

    fn protocol_violation(&self, reason: &str) -> PromptError {
        PromptError::prompt_cancelled(
            self.full_history(),
            format!("agent run driver protocol violation: {reason}"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ToolFunction, ToolResultContent};
    use serde_json::json;

    fn tool_names(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
        Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            ..Usage::new()
        }
    }

    fn text_turn(text: &str) -> ModelTurn {
        ModelTurn::new(
            None,
            OneOrMany::one(AssistantContent::text(text)),
            Usage::new(),
            tool_names(&["add"]),
            tool_names(&["add"]),
        )
    }

    fn tool_call(id: &str, name: &str) -> AssistantContent {
        AssistantContent::ToolCall(ToolCall::new(
            id.to_string(),
            ToolFunction::new(name.to_string(), json!({"x": 1})),
        ))
    }

    fn tool_call_turn(id: &str, name: &str) -> ModelTurn {
        ModelTurn::new(
            None,
            OneOrMany::one(tool_call(id, name)),
            Usage::new(),
            tool_names(&["add"]),
            tool_names(&["add"]),
        )
    }

    fn tool_result(id: &str, output: &str) -> UserContent {
        UserContent::tool_result(
            id.to_string(),
            ToolResultContent::from_tool_output(output.to_string()),
        )
    }

    fn expect_call_model(run: &mut AgentRun) -> (Message, Vec<Message>, usize) {
        match run.next_step().expect("next_step should succeed") {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => (prompt, history, turn),
            step => panic!("expected CallModel, got {step:?}"),
        }
    }

    fn expect_call_tools(run: &mut AgentRun) -> Vec<PendingToolCall> {
        match run.next_step().expect("next_step should succeed") {
            AgentRunStep::CallTools { calls } => calls,
            step => panic!("expected CallTools, got {step:?}"),
        }
    }

    fn expect_done(run: &mut AgentRun) -> PromptResponse {
        match run.next_step().expect("next_step should succeed") {
            AgentRunStep::Done(response) => response,
            step => panic!("expected Done, got {step:?}"),
        }
    }

    fn expect_continue(outcome: ModelTurnOutcome) -> bool {
        match outcome {
            ModelTurnOutcome::Continue {
                response_hook_suppressed,
            } => response_hook_suppressed,
            outcome => panic!("expected Continue, got {outcome:?}"),
        }
    }

    fn expect_needs_resolution(outcome: ModelTurnOutcome) -> InvalidToolCallContext {
        match outcome {
            ModelTurnOutcome::NeedsResolution(context) => context,
            outcome => panic!("expected NeedsResolution, got {outcome:?}"),
        }
    }

    #[test]
    fn text_only_run_completes_in_one_turn() {
        let mut run = AgentRun::new("hello");

        let (prompt, history, turn) = expect_call_model(&mut run);
        assert_eq!(prompt, Message::user("hello"));
        assert!(history.is_empty());
        assert_eq!(turn, 1);

        let suppressed = expect_continue(
            run.model_response(text_turn("hi there"))
                .expect("model_response should succeed"),
        );
        assert!(!suppressed);

        let response = expect_done(&mut run);
        assert_eq!(response.output, "hi there");
        let messages = response.messages.expect("messages should be recorded");
        assert_eq!(messages.len(), 2);
        assert!(run.is_done());
    }

    #[test]
    fn input_history_prefixes_request_history() {
        let mut run = AgentRun::new("question")
            .with_history(vec![Message::user("earlier"), Message::assistant("reply")]);

        let (_, history, _) = expect_call_model(&mut run);
        assert_eq!(
            history,
            vec![Message::user("earlier"), Message::assistant("reply")]
        );

        expect_continue(
            run.model_response(text_turn("answer"))
                .expect("model_response should succeed"),
        );
        let response = expect_done(&mut run);
        // Returned messages exclude the input history.
        assert_eq!(
            response
                .messages
                .expect("messages should be recorded")
                .len(),
            2
        );
    }

    #[test]
    fn tool_roundtrip_threads_history_and_usage() {
        let mut run = AgentRun::new("add things").max_turns(2);

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add").with_usage_for_test(usage(10, 5)))
                .expect("model_response should succeed"),
        );

        let calls = expect_call_tools(&mut run);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_call.function.name, "add");
        assert!(calls[0].preresolved_result.is_none());

        run.tool_results(vec![tool_result("call_1", "2")])
            .expect("tool_results should succeed");

        let (prompt, history, turn) = expect_call_model(&mut run);
        assert_eq!(turn, 2);
        // The tool-result user message becomes the new prompt; the assistant
        // turn is part of the history.
        assert!(matches!(prompt, Message::User { .. }));
        assert_eq!(history.len(), 2);

        expect_continue(
            run.model_response(text_turn("the answer is 2").with_usage_for_test(usage(20, 7)))
                .expect("model_response should succeed"),
        );

        let response = expect_done(&mut run);
        assert_eq!(response.output, "the answer is 2");
        assert_eq!(response.usage, usage(30, 12));
        assert_eq!(response.completion_calls.len(), 2);
        assert_eq!(response.completion_calls[0].call_index, 0);
        assert_eq!(response.completion_calls[0].usage, Some(usage(10, 5)));
        assert_eq!(response.completion_calls[1].usage, Some(usage(20, 7)));
        // prompt, assistant tool call, tool result, final assistant text
        assert_eq!(
            response
                .messages
                .expect("messages should be recorded")
                .len(),
            4
        );
    }

    #[test]
    fn parallel_tool_calls_surface_in_emission_order() {
        let mut run = AgentRun::new("do both").max_turns(2);

        expect_call_model(&mut run);
        let turn = ModelTurn::new(
            None,
            OneOrMany::many(vec![tool_call("call_1", "add"), tool_call("call_2", "add")])
                .expect("two items"),
            Usage::new(),
            tool_names(&["add"]),
            tool_names(&["add"]),
        );
        expect_continue(
            run.model_response(turn)
                .expect("model_response should succeed"),
        );

        let calls = expect_call_tools(&mut run);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool_call.id, "call_1");
        assert_eq!(calls[1].tool_call.id, "call_2");

        // Results fed out of order still land in one user message.
        run.tool_results(vec![tool_result("call_2", "b"), tool_result("call_1", "a")])
            .expect("tool_results should succeed");
        let messages = run.messages();
        assert!(matches!(
            messages.last(),
            Some(Message::User { content }) if content.len() == 2
        ));
    }

    #[test]
    fn max_turns_exhaustion_returns_max_turns_error() {
        let mut run = AgentRun::new("loop forever");

        for turn_id in ["call_1", "call_2"] {
            expect_call_model(&mut run);
            expect_continue(
                run.model_response(tool_call_turn(turn_id, "add"))
                    .expect("model_response should succeed"),
            );
            expect_call_tools(&mut run);
            run.tool_results(vec![tool_result(turn_id, "0")])
                .expect("tool_results should succeed");
        }

        let err = run.next_step().expect_err("depth should be exhausted");
        assert!(matches!(
            err,
            PromptError::MaxTurnsError { max_turns: 0, .. }
        ));
    }

    #[test]
    fn invalid_tool_call_fail_returns_unknown_tool_call() {
        let mut run = AgentRun::new("call something");

        expect_call_model(&mut run);
        let context = expect_needs_resolution(
            run.model_response(tool_call_turn("call_1", "unknown"))
                .expect("model_response should succeed"),
        );
        assert_eq!(context.tool_name, "unknown");
        assert_eq!(context.available_tools, vec!["add".to_string()]);
        assert!(!context.is_streaming);
        // Diagnostic history includes the rejected assistant turn.
        assert_eq!(context.chat_history.len(), 2);

        let err = run
            .resolve_invalid_tool_call(InvalidToolCallHookAction::fail())
            .expect_err("fail action should error");
        assert!(matches!(
            err,
            PromptError::UnknownToolCall { tool_name, .. } if tool_name == "unknown"
        ));
    }

    #[test]
    fn invalid_tool_call_retry_rolls_back_with_feedback() {
        let mut run = AgentRun::new("call something")
            .max_turns(2)
            .max_invalid_tool_call_retries(1);

        expect_call_model(&mut run);
        expect_needs_resolution(
            run.model_response(tool_call_turn("call_1", "unknown"))
                .expect("model_response should succeed"),
        );
        let outcome = run
            .resolve_invalid_tool_call(InvalidToolCallHookAction::retry("use add instead"))
            .expect("retry should be accepted");
        assert!(matches!(outcome, ModelTurnOutcome::TurnRetried));

        // The rolled-back turn appended the assistant message and feedback.
        assert_eq!(run.messages().len(), 3);
        let (prompt, _, turn) = expect_call_model(&mut run);
        assert_eq!(turn, 2);
        assert!(matches!(
            prompt,
            Message::User { ref content }
                if matches!(content.first(), UserContent::ToolResult(_))
        ));

        // Budget of one: a second retry fails with UnknownToolCall.
        expect_needs_resolution(
            run.model_response(tool_call_turn("call_2", "unknown"))
                .expect("model_response should succeed"),
        );
        let err = run
            .resolve_invalid_tool_call(InvalidToolCallHookAction::retry("again"))
            .expect_err("budget exhausted");
        assert!(matches!(err, PromptError::UnknownToolCall { .. }));
    }

    #[test]
    fn invalid_tool_call_repair_renames_and_suppresses_response_hook() {
        let mut run = AgentRun::new("call something").max_turns(2);

        expect_call_model(&mut run);
        expect_needs_resolution(
            run.model_response(tool_call_turn("call_1", "default_api"))
                .expect("model_response should succeed"),
        );
        let suppressed = expect_continue(
            run.resolve_invalid_tool_call(InvalidToolCallHookAction::repair("add"))
                .expect("repair should be accepted"),
        );
        assert!(suppressed);

        let calls = expect_call_tools(&mut run);
        assert_eq!(calls[0].tool_call.function.name, "add");
        assert!(calls[0].preresolved_result.is_none());
    }

    #[test]
    fn invalid_tool_call_repair_to_disallowed_name_fails() {
        let mut run = AgentRun::new("call something");

        expect_call_model(&mut run);
        expect_needs_resolution(
            run.model_response(tool_call_turn("call_1", "unknown"))
                .expect("model_response should succeed"),
        );
        let err = run
            .resolve_invalid_tool_call(InvalidToolCallHookAction::repair("also_unknown"))
            .expect_err("repair to disallowed name should fail");
        assert!(matches!(
            err,
            PromptError::UnknownToolCall { tool_name, .. } if tool_name == "also_unknown"
        ));
    }

    #[test]
    fn invalid_tool_call_skip_suppresses_all_peer_executions() {
        let mut run = AgentRun::new("call things").max_turns(2);

        expect_call_model(&mut run);
        let turn = ModelTurn::new(
            None,
            OneOrMany::many(vec![
                tool_call("call_1", "unknown"),
                tool_call("call_2", "add"),
            ])
            .expect("two items"),
            Usage::new(),
            tool_names(&["add"]),
            tool_names(&["add"]),
        );
        expect_needs_resolution(
            run.model_response(turn)
                .expect("model_response should succeed"),
        );
        let suppressed = expect_continue(
            run.resolve_invalid_tool_call(InvalidToolCallHookAction::skip("not available"))
                .expect("skip should be accepted"),
        );
        assert!(suppressed);

        let calls = expect_call_tools(&mut run);
        assert_eq!(calls.len(), 2);
        // Both the skipped call and its valid peer carry preresolved results.
        assert!(calls.iter().all(|call| call.preresolved_result.is_some()));
    }

    #[test]
    fn skip_under_tool_choice_none_fails() {
        let mut run = AgentRun::new("call something").with_tool_choice(ToolChoice::None);

        expect_call_model(&mut run);
        expect_needs_resolution(
            run.model_response(ModelTurn::new(
                None,
                OneOrMany::one(tool_call("call_1", "add")),
                Usage::new(),
                tool_names(&["add"]),
                BTreeSet::new(),
            ))
            .expect("model_response should succeed"),
        );
        let err = run
            .resolve_invalid_tool_call(InvalidToolCallHookAction::skip("nope"))
            .expect_err("skip under ToolChoice::None should fail");
        assert!(matches!(err, PromptError::UnknownToolCall { .. }));
    }

    #[test]
    fn empty_tool_results_cancel_the_run() {
        let mut run = AgentRun::new("call something").max_turns(2);

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add"))
                .expect("model_response should succeed"),
        );
        expect_call_tools(&mut run);

        let err = run
            .tool_results(Vec::new())
            .expect_err("empty results should cancel");
        assert!(matches!(
            err,
            PromptError::PromptCancelled { reason, .. }
                if reason.contains("tool execution produced no tool results")
        ));
    }

    #[test]
    fn out_of_protocol_calls_are_rejected_without_corrupting_state() {
        let mut run = AgentRun::new("hello");

        let err = run
            .tool_results(vec![tool_result("call_1", "x")])
            .expect_err("no CallTools pending");
        assert!(matches!(err, PromptError::PromptCancelled { .. }));

        // The run is still drivable after a rejected out-of-protocol call.
        expect_call_model(&mut run);
        let err = run
            .next_step()
            .expect_err("model response is pending, next_step must be rejected");
        assert!(matches!(err, PromptError::PromptCancelled { .. }));
        expect_continue(
            run.model_response(text_turn("hi"))
                .expect("model_response should still succeed"),
        );
        assert_eq!(expect_done(&mut run).output, "hi");
    }

    #[test]
    fn done_step_is_idempotent() {
        let mut run = AgentRun::new("hello");
        expect_call_model(&mut run);
        expect_continue(
            run.model_response(text_turn("hi"))
                .expect("model_response should succeed"),
        );
        assert_eq!(expect_done(&mut run).output, "hi");
        assert_eq!(expect_done(&mut run).output, "hi");
    }

    #[test]
    fn serialized_run_alone_carries_pending_tool_calls() {
        let mut run = AgentRun::new("add things").max_turns(2);
        expect_call_model(&mut run);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add"))
                .expect("model_response should succeed"),
        );
        expect_call_tools(&mut run);

        // A fresh process receives only the serialized run: the pending tool
        // calls must be recoverable from the state itself.
        let serialized = serde_json::to_string(&run).expect("mid-run state should serialize");
        drop(run);
        let mut resumed: AgentRun =
            serde_json::from_str(&serialized).expect("mid-run state should deserialize");

        let calls = expect_call_tools(&mut resumed);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_call.function.name, "add");
        // Re-emission is idempotent while results are pending.
        let calls_again = expect_call_tools(&mut resumed);
        assert_eq!(calls_again[0].tool_call.id, calls[0].tool_call.id);

        // Answer using only IDs learned from the re-emitted step.
        let results = calls
            .iter()
            .map(|call| tool_result(&call.tool_call.id, "2"))
            .collect::<Vec<_>>();
        resumed
            .tool_results(results)
            .expect("tool_results should succeed");
        expect_call_model(&mut resumed);
        expect_continue(
            resumed
                .model_response(text_turn("done"))
                .expect("model_response should succeed"),
        );
        assert_eq!(expect_done(&mut resumed).output, "done");
    }

    #[test]
    fn tool_results_validates_against_pending_calls() {
        let drive_to_pending_tools = || {
            let mut run = AgentRun::new("add things").max_turns(2);
            expect_call_model(&mut run);
            expect_continue(
                run.model_response(tool_call_turn("call_1", "add"))
                    .expect("model_response should succeed"),
            );
            expect_call_tools(&mut run);
            run
        };

        // A result for an unknown call ID is rejected without corrupting the run.
        let mut run = drive_to_pending_tools();
        let err = run
            .tool_results(vec![tool_result("call_unknown", "2")])
            .expect_err("unknown tool call id must be rejected");
        assert!(matches!(err, PromptError::PromptCancelled { .. }));
        run.tool_results(vec![tool_result("call_1", "2")])
            .expect("valid results should still be accepted after a rejection");

        // Leaving a pending call unanswered is rejected.
        let mut run = drive_to_pending_tools();
        let err = run
            .tool_results(vec![tool_result("call_1", "2"), tool_result("call_1", "3")])
            .expect_err("answering one call twice must be rejected");
        assert!(matches!(err, PromptError::PromptCancelled { .. }));

        // Non-tool-result content is rejected.
        let mut run = drive_to_pending_tools();
        let err = run
            .tool_results(vec![UserContent::text("not a tool result")])
            .expect_err("non-tool-result content must be rejected");
        assert!(matches!(err, PromptError::PromptCancelled { .. }));
    }

    #[test]
    fn serde_round_trip_mid_run_resumes_identically() {
        let drive_to_pending_tools = || {
            let mut run = AgentRun::new("add things").max_turns(2);
            expect_call_model(&mut run);
            expect_continue(
                run.model_response(
                    tool_call_turn("call_1", "add").with_usage_for_test(usage(10, 5)),
                )
                .expect("model_response should succeed"),
            );
            expect_call_tools(&mut run);
            run
        };

        let finish = |mut run: AgentRun| {
            run.tool_results(vec![tool_result("call_1", "2")])
                .expect("tool_results should succeed");
            expect_call_model(&mut run);
            expect_continue(
                run.model_response(text_turn("done").with_usage_for_test(usage(3, 4)))
                    .expect("model_response should succeed"),
            );
            expect_done(&mut run)
        };

        let uninterrupted = finish(drive_to_pending_tools());

        let suspended = drive_to_pending_tools();
        let serialized = serde_json::to_string(&suspended).expect("mid-run state should serialize");
        let restored: AgentRun =
            serde_json::from_str(&serialized).expect("mid-run state should deserialize");
        let resumed = finish(restored);

        assert_eq!(resumed.output, uninterrupted.output);
        assert_eq!(resumed.usage, uninterrupted.usage);
        assert_eq!(resumed.completion_calls, uninterrupted.completion_calls);
        // Compare messages by their serialized form: deserializing a message
        // normalizes absent `additional_params` to an empty map, which is
        // semantically identical and serializes identically.
        assert_eq!(
            serde_json::to_value(&resumed.messages).expect("messages should serialize"),
            serde_json::to_value(&uninterrupted.messages).expect("messages should serialize"),
        );
    }

    #[test]
    fn pending_invalid_tool_call_survives_serde_round_trip() {
        let mut run = AgentRun::new("call something");
        expect_call_model(&mut run);
        let context = expect_needs_resolution(
            run.model_response(tool_call_turn("call_1", "unknown"))
                .expect("model_response should succeed"),
        );

        let serialized = serde_json::to_string(&run).expect("state should serialize");
        let restored: AgentRun =
            serde_json::from_str(&serialized).expect("state should deserialize");
        let restored_context = restored
            .pending_invalid_tool_call()
            .expect("pending resolution should survive serialization");
        assert_eq!(restored_context.tool_name, context.tool_name);
        assert_eq!(
            restored_context.chat_history.len(),
            context.chat_history.len()
        );
    }

    impl ModelTurn {
        fn with_usage_for_test(mut self, usage: Usage) -> Self {
            self.usage = usage;
            self
        }
    }
}
