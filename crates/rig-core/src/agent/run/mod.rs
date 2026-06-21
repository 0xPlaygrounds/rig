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
//! sensitivity the conversation content has — and the serialization format
//! carries no cross-version stability guarantee yet: resume with the same rig
//! version that suspended the run.
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

pub mod output_mode;
pub mod streamed;

pub use output_mode::OutputMode;

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

/// Default number of times Tool output mode re-prompts the model for valid
/// structured output before finalizing best-effort (see #1928). Mirrors
/// pydantic-ai's default output-retry budget of 1.
pub(crate) const DEFAULT_OUTPUT_RETRIES: usize = 1;

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
    /// Rig-generated identifier correlating this call's stream items, when
    /// the call arrived via a streamed turn. Persisted with the run state so
    /// a resumed process keeps emitting the IDs consumers already saw in
    /// tool-call deltas. Drivers generate a fresh ID when absent.
    #[serde(default)]
    pub internal_call_id: Option<String>,
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
    /// `(tool_call_id, internal_call_id)` pairs for streamed turns, in
    /// emission order; empty for non-streamed turns.
    #[serde(default)]
    internal_call_ids: Vec<(String, String)>,
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
    /// Name of the synthetic output tool when the agent uses Tool output mode
    /// (see #1928). A model turn calling this tool finalizes the run with the
    /// call's arguments as the response, instead of executing it as a tool.
    #[serde(default)]
    output_tool_name: Option<String>,
    /// JSON schema the Tool-mode output must satisfy, used to re-prompt on
    /// missing required fields before finalizing best-effort (#1928).
    #[serde(default)]
    output_schema: Option<serde_json::Value>,
    /// Budget for re-prompting the model in Tool output mode when it finalizes
    /// without calling the output tool, or calls it with arguments missing
    /// required fields. Exhausting it finalizes best-effort.
    #[serde(default)]
    max_output_retries: usize,
    #[serde(default)]
    output_retries: usize,
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
    /// Set once the current streamed model turn's completion call has been
    /// recorded, rejecting duplicate records; reset when the next
    /// [`AgentRunStep::CallModel`] is emitted.
    #[serde(default)]
    streamed_completion_call_recorded: bool,
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
            output_tool_name: None,
            output_schema: None,
            max_output_retries: 0,
            output_retries: 0,
            chat_history: None,
            new_messages: vec![prompt.into()],
            current_turn: 0,
            usage: Usage::new(),
            completion_calls: Vec::new(),
            completion_call_index: 0,
            invalid_tool_call_retries: 0,
            rollback_pending: false,
            streamed_completion_call_recorded: false,
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

    /// Configure Tool output-mode validation (#1928): the JSON schema the
    /// output-tool arguments should satisfy, and how many times to re-prompt the
    /// model — when it finalizes without calling the output tool, or calls it
    /// with arguments missing required fields — before finalizing best-effort.
    pub fn with_output_validation(
        mut self,
        output_schema: Option<serde_json::Value>,
        max_output_retries: usize,
    ) -> Self {
        self.output_schema = output_schema;
        self.max_output_retries = max_output_retries;
        self
    }

    /// Top-level `required` schema fields absent from the output-tool arguments.
    /// A lightweight structural check (not full JSON Schema validation): empty
    /// when there is no schema, no `required` array, or every required field is
    /// present. Non-object arguments (e.g. `null`) count every required field as
    /// missing.
    fn missing_required_output_fields(&self, args: &serde_json::Value) -> Vec<String> {
        let Some(required) = self
            .output_schema
            .as_ref()
            .and_then(|schema| schema.get("required"))
            .and_then(|required| required.as_array())
        else {
            return Vec::new();
        };
        let object = args.as_object();
        required
            .iter()
            .filter_map(|field| field.as_str())
            .filter(|field| object.is_none_or(|object| !object.contains_key(*field)))
            .map(str::to_owned)
            .collect()
    }

    /// Whether `text` already parses as a JSON object satisfying the output
    /// schema's required fields — i.e. it is acceptable structured output even
    /// though the model returned it as plain text instead of an output-tool call.
    fn text_satisfies_output_schema(&self, text: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(text.trim())
            .ok()
            .is_some_and(|value| self.missing_required_output_fields(&value).is_empty())
    }

    /// Whether the run may re-prompt for valid Tool-mode output: budget remains
    /// AND a retry turn would not immediately exceed [`AgentRun::max_turns`]
    /// (otherwise we finalize best-effort rather than surface a max-turns error).
    fn can_reprompt_for_output(&self) -> bool {
        self.output_retries < self.max_output_retries && self.current_turn <= self.max_turns + 1
    }

    /// Roll the run back to re-prompt for valid output (#1928). The caller must
    /// have already appended the assistant turn and the corrective feedback
    /// message to the history. Consumes one output-retry, then emits the retry
    /// [`AgentRunStep::CallModel`].
    fn reprompt_for_output(&mut self) -> Result<AgentRunStep, PromptError> {
        self.output_retries += 1;
        self.state = RunState::PreparingRequest;
        self.next_step()
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

    /// Set the synthetic output-tool name for Tool output mode (see #1928).
    /// When a model turn calls this tool, the run finalizes with the call's
    /// arguments (serialized JSON) as the response.
    pub fn with_output_tool_name(mut self, name: impl Into<String>) -> Self {
        self.output_tool_name = Some(name.into());
        self
    }

    /// Set (or clear) the output-tool name in place. The driver resolves the
    /// name from the prepared request inside the run loop, where the agent's
    /// tool set (and thus the resolved output mode) is known.
    pub(crate) fn set_output_tool_name(&mut self, name: Option<String>) {
        // The name is committed once and pinned for the whole run, so the
        // request the driver builds each turn stays consistent with the
        // intercept (and a tool set that shifts mid-run cannot flip the mode).
        if self.output_tool_name.is_none() {
            self.output_tool_name = name;
        }
    }

    /// The synthetic output-tool name committed for this run, if any. The driver
    /// passes this back when preparing later turns so Tool output mode stays
    /// pinned even if the per-turn tool set changes (see #1928).
    pub(crate) fn output_tool_name(&self) -> Option<&str> {
        self.output_tool_name.as_deref()
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
                self.streamed_completion_call_recorded = false;
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
                    mut internal_call_ids,
                } = *turn_state;
                let Some(choice) = OneOrMany::from_iter_optional(items.clone()) else {
                    return Err(PromptError::prompt_cancelled(
                        self.full_history(),
                        "model turn lost its assistant content",
                    ));
                };

                // Tool output mode (#1928): a call to the synthetic output tool
                // finalizes the run with the call's arguments as the response,
                // instead of executing it as a tool. First match wins; any
                // sibling tool calls in the same turn are dropped.
                if has_tool_calls
                    && let Some(output_tool_name) = self.output_tool_name.clone()
                    && let Some(tool_call) = items.iter().find_map(|item| match item {
                        AssistantContent::ToolCall(tc) if tc.function.name == output_tool_name => {
                            Some(tc)
                        }
                        _ => None,
                    })
                {
                    let args = tool_call.function.arguments.clone();
                    let tool_call_id = tool_call.id.clone();
                    let output = json_utils::value_to_json_string(&args);

                    // Validate the output against the schema's required fields and
                    // re-prompt while budget remains, so a model that omits fields
                    // gets a chance to fix it before we finalize best-effort.
                    let missing = self.missing_required_output_fields(&args);
                    if !missing.is_empty() && self.can_reprompt_for_output() {
                        self.new_messages.push(Message::Assistant {
                            id: message_id,
                            content: choice.clone(),
                        });
                        let feedback = format!(
                            "The `{output_tool_name}` arguments were missing required field(s): \
                             {}. Call `{output_tool_name}` again with every required field.",
                            missing.join(", ")
                        );
                        if let Some(user_message) =
                            invalid_tool_retry_user_message(&choice, &tool_call_id, feedback)
                        {
                            self.new_messages.push(user_message);
                        }
                        return self.reprompt_for_output();
                    }

                    // Finalize. The turn is persisted as the assistant's final
                    // *text* (keeping any reasoning, dropping every tool call)
                    // rather than the raw output-tool call. Otherwise the saved
                    // history would carry an unanswered tool_use, which providers
                    // reject when the conversation is replayed on a later turn.
                    let mut final_items: Vec<AssistantContent> = items
                        .iter()
                        .filter(|item| !matches!(item, AssistantContent::ToolCall(_)))
                        .cloned()
                        .collect();
                    final_items.push(AssistantContent::text(output.clone()));
                    if let Some(content) = OneOrMany::from_iter_optional(final_items) {
                        self.new_messages.push(Message::Assistant {
                            id: message_id,
                            content,
                        });
                    }

                    let response = PromptResponse::new(output, self.usage)
                        .with_messages(self.new_messages.clone())
                        .with_completion_calls(self.completion_calls.clone());
                    self.state = RunState::Done(Box::new(response.clone()));
                    return Ok(AgentRunStep::Done(response));
                }

                if !is_empty_assistant_turn(&choice) {
                    self.new_messages.push(Message::Assistant {
                        id: message_id,
                        content: choice.clone(),
                    });
                }

                if has_tool_calls {
                    // The model is making progress with real tools, so reset the
                    // output-retry budget: it is per finalization attempt, not a
                    // single per-run allowance an early stray turn could burn
                    // before the model genuinely needs to produce output (#1928).
                    self.output_retries = 0;
                    let calls: Vec<PendingToolCall> = items
                        .iter()
                        .filter_map(|item| match item {
                            AssistantContent::ToolCall(tool_call) => {
                                // Consume pairs positionally so duplicate
                                // provider IDs within one turn stay
                                // distinguishable.
                                let internal_call_id = internal_call_ids
                                    .iter()
                                    .position(|(id, _)| *id == tool_call.id)
                                    .map(|index| internal_call_ids.remove(index).1);
                                Some(PendingToolCall {
                                    tool_call: tool_call.clone(),
                                    preresolved_result: skipped.get(&tool_call.id).cloned(),
                                    internal_call_id,
                                })
                            }
                            _ => None,
                        })
                        .collect();
                    self.state = RunState::ExecutingTools(calls.clone());
                    Ok(AgentRunStep::CallTools { calls })
                } else {
                    // Tool output mode (#1928): the model produced a final text
                    // answer without calling the output tool. Re-prompt while
                    // budget remains so it returns structured output; the
                    // assistant text was already appended above, so just add the
                    // corrective feedback. Empty turns finalize best-effort.
                    //
                    // But if the text already *is* valid output (parses as JSON
                    // with every required field), accept it rather than wasting a
                    // turn — the model answered correctly, just via the wrong
                    // channel.
                    if let Some(output_tool_name) = self.output_tool_name.clone()
                        && !is_empty_assistant_turn(&choice)
                        && self.can_reprompt_for_output()
                        && !self.text_satisfies_output_schema(&assistant_text_from_choice(&choice))
                    {
                        let feedback = format!(
                            "Provide your final answer by calling the `{output_tool_name}` tool \
                             with the structured result as its arguments, not as plain text."
                        );
                        self.new_messages.push(Message::user(feedback));
                        return self.reprompt_for_output();
                    }

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
        if self.streamed_completion_call_recorded {
            return Err(self.protocol_violation(
                "model_response called after record_streamed_completion_call for the same turn; feed streamed turns via streamed_turn",
            ));
        }

        self.completion_calls
            .push(CompletionCall::new(self.completion_call_index, turn.usage));
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

        // `results` is non-empty (checked above), so construction succeeds.
        let Some(content) = OneOrMany::from_iter_optional(results) else {
            return Err(
                self.protocol_violation("internal: tool results vanished during validation")
            );
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
            internal_call_ids: Vec::new(),
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
    /// aggregates `usage` into the run total. Zero-valued usage means the
    /// provider reported no usage metrics.
    pub fn record_streamed_completion_call(
        &mut self,
        usage: Usage,
    ) -> Result<CompletionCall, PromptError> {
        let recordable = matches!(self.state, RunState::AwaitingModel)
            || (matches!(self.state, RunState::PreparingRequest) && self.rollback_pending);
        if !recordable {
            return Err(self.protocol_violation(
                "record_streamed_completion_call called without a pending or rolled-back CallModel step",
            ));
        }
        if self.streamed_completion_call_recorded {
            return Err(self.protocol_violation(
                "record_streamed_completion_call called twice for the same model turn",
            ));
        }
        self.streamed_completion_call_recorded = true;

        let call = CompletionCall::new(self.completion_call_index, usage);
        self.completion_call_index += 1;
        self.completion_calls.push(call);
        self.usage += usage;
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

        // Guarantee exactly one CompletionCall per model call: drivers that
        // never learned usage (no record before the turn completed) still get
        // the call recorded, with no reported usage.
        if !self.streamed_completion_call_recorded {
            self.completion_calls.push(CompletionCall::new(
                self.completion_call_index,
                Usage::new(),
            ));
            self.completion_call_index += 1;
            self.streamed_completion_call_recorded = true;
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
            internal_call_ids: turn.internal_call_ids,
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
        assert_eq!(response.completion_calls[0].usage, usage(10, 5));
        assert_eq!(response.completion_calls[1].usage, usage(20, 7));
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
    fn model_response_rejected_after_streamed_completion_call_record() {
        let mut run = AgentRun::new("hello");
        expect_call_model(&mut run);
        run.record_streamed_completion_call(Usage::new())
            .expect("record should succeed");

        let err = run
            .model_response(text_turn("hi"))
            .expect_err("mixed streamed/non-streamed ingestion must be rejected");
        assert!(matches!(err, PromptError::PromptCancelled { .. }));
        // No duplicate completion call was appended.
        assert_eq!(run.completion_calls().len(), 1);
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
    fn agent_run_deserializes_pre_monoid_suspended_state() {
        // Fixture captured from rig before CompletionCall.usage dropped its
        // Option encoding, suspended at ExecutingTools with a null-usage
        // completion call. It must deserialize and resume.
        let fixture = r#"{"max_turns":2,"max_invalid_tool_call_retries":0,"tool_choice":null,"chat_history":null,"new_messages":[{"role":"user","content":[{"type":"text","text":"add things"}]},{"role":"assistant","id":null,"content":[{"id":"call_1","call_id":null,"function":{"name":"add","arguments":{"x":1}},"signature":null,"additional_params":null}]}],"current_turn":1,"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"cached_input_tokens":0,"cache_creation_input_tokens":0,"tool_use_prompt_tokens":0,"reasoning_tokens":0},"completion_calls":[{"call_index":0,"usage":null}],"completion_call_index":1,"invalid_tool_call_retries":0,"rollback_pending":false,"streamed_completion_call_recorded":false,"state":{"ExecutingTools":[{"tool_call":{"id":"call_1","call_id":null,"function":{"name":"add","arguments":{"x":1}},"signature":null,"additional_params":null},"preresolved_result":null,"internal_call_id":null}]}}"#;

        let mut restored: AgentRun =
            serde_json::from_str(fixture).expect("old-format suspended run should deserialize");
        assert_eq!(restored.completion_calls()[0].usage, Usage::new());

        let calls = expect_call_tools(&mut restored);
        assert_eq!(calls.len(), 1);
        restored
            .tool_results(vec![tool_result("call_1", "2")])
            .expect("tool_results should succeed");
        expect_call_model(&mut restored);
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

    /// A turn calling `name`, advertising it as an allowed-but-not-executable
    /// tool (the shape Tool output mode produces — see #1928).
    fn output_tool_turn(id: &str, name: &str) -> ModelTurn {
        ModelTurn::new(
            None,
            OneOrMany::one(tool_call(id, name)),
            Usage::new(),
            tool_names(&["add"]),
            tool_names(&["add", name]),
        )
    }

    /// Every assistant tool call in `messages` must have a matching user tool
    /// result — an unanswered tool_use is rejected by providers on replay.
    fn assert_no_orphan_tool_use(messages: &[Message]) {
        let mut answered = BTreeSet::new();
        for message in messages {
            if let Message::User { content } = message {
                for item in content.iter() {
                    if let UserContent::ToolResult(result) = item {
                        answered.insert(result.id.clone());
                    }
                }
            }
        }
        for message in messages {
            if let Message::Assistant { content, .. } = message {
                for item in content.iter() {
                    if let AssistantContent::ToolCall(call) = item {
                        assert!(
                            answered.contains(&call.id),
                            "assistant tool_call {:?} has no matching tool_result in history",
                            call.id
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn output_tool_call_finalizes_run_with_arguments() {
        let mut run = AgentRun::new("summarize").with_output_tool_name("final_result");

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(output_tool_turn("call_1", "final_result"))
                .expect("model_response should succeed"),
        );

        // The output tool is not executed; its arguments become the run output.
        let response = expect_done(&mut run);
        assert_eq!(response.output, r#"{"x":1}"#);
        assert!(run.is_done());

        // The finalizing turn is persisted as assistant text, not as the raw
        // output-tool call, so the saved history has no dangling tool_use.
        let messages = response.messages.expect("messages should be recorded");
        assert_no_orphan_tool_use(&messages);
        assert!(matches!(
            messages.last(),
            Some(Message::Assistant { content, .. })
                if assistant_text_from_choice(content) == r#"{"x":1}"#
        ));
    }

    #[test]
    fn output_tool_call_wins_over_sibling_real_tool_calls() {
        let mut run = AgentRun::new("do it")
            .max_turns(2)
            .with_output_tool_name("final_result");

        expect_call_model(&mut run);
        // The model emits a real tool call *and* the output tool in one turn;
        // the output-tool intercept wins and the real call is never executed.
        let turn = ModelTurn::new(
            None,
            OneOrMany::many(vec![
                tool_call("call_1", "add"),
                tool_call("call_2", "final_result"),
            ])
            .expect("two items"),
            Usage::new(),
            tool_names(&["add"]),
            tool_names(&["add", "final_result"]),
        );
        expect_continue(
            run.model_response(turn)
                .expect("model_response should succeed"),
        );

        let response = expect_done(&mut run);
        assert_eq!(response.output, r#"{"x":1}"#);
        assert!(run.is_done());

        // Both the sibling `add` call and the output-tool call are dropped from
        // the persisted assistant message, leaving no unanswered tool_use.
        let messages = response.messages.expect("messages should be recorded");
        assert_no_orphan_tool_use(&messages);
        assert!(
            messages.iter().all(|message| match message {
                Message::Assistant { content, .. } => !content
                    .iter()
                    .any(|item| matches!(item, AssistantContent::ToolCall(_))),
                _ => true,
            }),
            "no assistant tool calls should survive in the finalized history"
        );
    }

    #[test]
    fn real_tool_calls_still_execute_when_output_tool_unused() {
        // With an output tool configured but only real tools called, the run
        // proceeds to tool execution as normal (the intercept must not fire).
        let mut run = AgentRun::new("add things")
            .max_turns(2)
            .with_output_tool_name("final_result");

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add"))
                .expect("model_response should succeed"),
        );

        let calls = expect_call_tools(&mut run);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_call.function.name, "add");
    }

    fn required_field_schema(field: &str) -> serde_json::Value {
        json!({
            "type": "object",
            "required": [field],
            "properties": { field: { "type": "string" } },
        })
    }

    #[test]
    fn tool_mode_reprompts_when_output_tool_not_called() {
        // #1928: in Tool mode the model finalized with plain text instead of
        // calling the output tool, so the run re-prompts (within budget).
        let mut run = AgentRun::new("summarize")
            .max_turns(3)
            .with_output_tool_name("final_result")
            .with_output_validation(Some(required_field_schema("summary")), 1);

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(text_turn("here is the answer"))
                .expect("model_response should succeed"),
        );

        // Instead of finalizing, the run emits a second CallModel with corrective
        // feedback naming the output tool.
        let (prompt, _history, turn) = expect_call_model(&mut run);
        assert_eq!(turn, 2);
        let prompt_json = serde_json::to_string(&prompt).expect("prompt should serialize");
        assert!(
            prompt_json.contains("final_result"),
            "re-prompt feedback should name the output tool: {prompt_json}"
        );
        assert!(!run.is_done());
    }

    #[test]
    fn tool_mode_reprompts_when_output_args_missing_required_fields() {
        // #1928: the output tool was called but its arguments omit a required
        // field, so the run re-prompts rather than finalizing invalid output.
        let mut run = AgentRun::new("summarize")
            .max_turns(3)
            .with_output_tool_name("final_result")
            // `output_tool_turn` calls with args {"x":1}; require a different key.
            .with_output_validation(Some(required_field_schema("summary")), 1);

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(output_tool_turn("call_1", "final_result"))
                .expect("model_response should succeed"),
        );

        let (_prompt, _history, turn) = expect_call_model(&mut run);
        assert_eq!(turn, 2);
        assert!(!run.is_done());
    }

    #[test]
    fn tool_mode_accepts_valid_json_text_without_reprompting() {
        // The model returned valid structured output as plain text instead of an
        // output-tool call — accept it rather than wasting a turn re-prompting.
        let mut run = AgentRun::new("summarize")
            .max_turns(3)
            .with_output_tool_name("final_result")
            .with_output_validation(Some(required_field_schema("summary")), 1);

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(text_turn(r#"{"summary":"all good"}"#))
                .expect("model_response should succeed"),
        );

        let response = expect_done(&mut run);
        assert_eq!(response.output, r#"{"summary":"all good"}"#);
        assert!(run.is_done());
    }

    #[test]
    fn tool_mode_finalizes_best_effort_when_output_retry_budget_exhausted() {
        // With no retry budget, invalid output finalizes best-effort (the caller
        // validates) rather than looping — and history stays free of orphan
        // tool_use.
        let mut run = AgentRun::new("summarize")
            .max_turns(3)
            .with_output_tool_name("final_result")
            .with_output_validation(Some(required_field_schema("summary")), 0);

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(output_tool_turn("call_1", "final_result"))
                .expect("model_response should succeed"),
        );

        let response = expect_done(&mut run);
        assert_eq!(response.output, r#"{"x":1}"#);
        let messages = response.messages.expect("messages should be recorded");
        assert_no_orphan_tool_use(&messages);
    }

    #[test]
    fn set_output_tool_name_is_idempotent_and_only_fills_when_unset() {
        // A pre-set name (e.g. via `with_output_tool_name`) is never overwritten,
        // keeping a resumed run deterministic.
        let mut run = AgentRun::new("x").with_output_tool_name("first");
        run.set_output_tool_name(Some("second".to_string()));
        run.set_output_tool_name(None);
        assert_eq!(run.output_tool_name.as_deref(), Some("first"));

        // When unset, the first non-None value fills it.
        let mut run = AgentRun::new("x");
        run.set_output_tool_name(None);
        assert_eq!(run.output_tool_name, None);
        run.set_output_tool_name(Some("filled".to_string()));
        assert_eq!(run.output_tool_name.as_deref(), Some("filled"));
    }

    impl ModelTurn {
        fn with_usage_for_test(mut self, usage: Usage) -> Self {
            self.usage = usage;
            self
        }
    }
}
