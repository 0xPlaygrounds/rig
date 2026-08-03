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
//! `AgentRun` deliberately contains no model, tool registry, memory backend, or
//! hook stack. Hand-driving it is a low-level provider integration: the caller
//! owns all IO and any lifecycle policy. To execute a configured [`Agent`](crate::agent::Agent)
//! with its hooks, tools, retrieval, and memory, use
//! [`Agent::runner`](crate::agent::Agent::runner); constructing an `AgentRun`
//! directly is not an alternate way to execute an `Agent`.
//!
//! [`crate::Agent::prompt`] and
//! [`Agent::runner`](crate::agent::Agent::runner) drive this machine internally;
//! the same machine can be driven by hand for custom provider control flow:
//!
//! ```rust,no_run
//! use rig_agent::agent::run::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome};
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
//!             // Execute each call, then submit its result with the call's
//!             // `internal_call_id` via `run.tool_result_submissions(...)`.
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

use rig_core::{
    OneOrMany,
    message::{AssistantContent, ToolCall, ToolChoice, ToolResult, ToolResultContent, UserContent},
};

use crate::{
    agent::hook::{InvalidToolCallAction, InvalidToolCallContext, RetryRequest},
    agent::response::{
        CompletionCall, PromptResponse, TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER,
        assistant_text_from_choice, build_full_history, build_history_for_request,
        invalid_tool_retry_user_message, is_empty_assistant_turn, tool_result_message,
    },
    completion::{Message, PromptError, Usage},
    json_utils,
};

pub use streamed::{
    PartialStreamedTurn, StreamedInvalidToolCall, StreamedResolution, StreamedTurn,
    StreamedTurnAssembler, StreamedTurnEvent,
};

/// Build the canonical "the model called a tool that isn't available" error.
/// The identical shape is raised from every recovery-rejection path
/// (`resolve_invalid_tool_call`, `resolve_streamed_invalid_tool_call`) and the
/// streamed fail-fast in `streamed_turn`; this collapses the copied struct
/// literal to one place while leaving each caller's control flow untouched.
fn unknown_tool_call_error(
    tool_name: String,
    available_tools: Vec<String>,
    allowed_tools: Vec<String>,
    chat_history: Vec<Message>,
) -> PromptError {
    PromptError::UnknownToolCall {
        tool_name,
        available_tools,
        allowed_tools,
        chat_history: Box::new(chat_history),
    }
}

/// Default number of times Tool output mode re-prompts the model for valid
/// structured output before finalizing best-effort (see #1928). Mirrors
/// pydantic-ai's default output-retry budget of 1.
pub const DEFAULT_OUTPUT_RETRIES: usize = 1;

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

/// Whether a tool invocation actually executed.
///
/// This is carried independently from the model-visible result because a tool
/// body may itself return a skipped-classified [`crate::tool::ToolResult`],
/// while a policy skip produces a durable result without invoking the body.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolInvocationDisposition {
    /// Rig invoked the registered tool body.
    Executed,
    /// A host supplied the result for execution performed outside Rig.
    #[default]
    ExternallyExecuted,
    /// No tool body ran; the result was resolved before or instead of
    /// execution.
    NotExecuted {
        /// Policy or recovery reason, when one is available.
        reason: Option<String>,
    },
}

impl ToolInvocationDisposition {
    /// Whether this invocation represents actual local or external execution
    /// and should therefore emit an execution observation.
    pub fn execution_committed(&self) -> bool {
        matches!(self, Self::Executed | Self::ExternallyExecuted)
    }

    pub(crate) fn not_executed(reason: impl Into<Option<String>>) -> Self {
        Self::NotExecuted {
            reason: reason.into(),
        }
    }
}

/// One result resolved before tool execution, positionally aligned with its
/// assistant content item.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreresolvedToolResult {
    result: UserContent,
    disposition: ToolInvocationDisposition,
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
    /// Original model-emitted call retained while hook rewrites change
    /// [`Self::tool_call`]. Serialized so forwarded/replayed inboxes keep the
    /// original/effective distinction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) original_tool_call: Option<Box<ToolCall>>,
    /// Data-only disposition for a pre-resolved decision. The visible result
    /// carries presentation; this field preserves execution semantics through
    /// serialization and every later result gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invocation_disposition: Option<ToolInvocationDisposition>,
}

impl PendingToolCall {
    /// A pending call for `tool_call` with no preresolved result and no
    /// streamed internal call ID.
    pub fn new(tool_call: ToolCall) -> Self {
        Self {
            tool_call,
            preresolved_result: None,
            internal_call_id: None,
            original_tool_call: None,
            invocation_disposition: None,
        }
    }

    /// Attach a preresolved result the driver must return as this call's tool
    /// result without executing the tool.
    pub fn with_preresolved_result(mut self, result: UserContent) -> Self {
        self.preresolved_result = Some(result);
        self.invocation_disposition = Some(ToolInvocationDisposition::not_executed(None));
        self
    }

    /// Return the stable Rig identity for this invocation, generating it once
    /// when a non-streamed call first reaches a driver.
    pub fn ensure_internal_call_id(&mut self) -> &str {
        self.internal_call_id
            .get_or_insert_with(rig_core::id::generate)
    }
}

/// One host-supplied result tied to Rig's unique invocation identity.
///
/// Provider tool-call IDs are payload and may repeat within a batch. Drivers
/// therefore accept results through this record instead of reconstructing an
/// invocation from [`rig_core::message::ToolResult::id`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolResultSubmission {
    /// The [`PendingToolCall::internal_call_id`] being answered.
    pub internal_call_id: String,
    /// The model-visible tool result for that invocation.
    pub result: UserContent,
    /// Whether producing this result involved an actual tool execution.
    #[serde(default)]
    pub disposition: ToolInvocationDisposition,
}

impl ToolResultSubmission {
    /// Associate an externally executed `result` with one pending Rig
    /// invocation.
    ///
    /// The host is asserting that execution occurred outside Rig, so streaming
    /// drivers emit the same execution observation as for a locally executed
    /// tool body. Results already resolved by Rig policy carry their
    /// [`ToolInvocationDisposition::NotExecuted`] disposition internally.
    pub fn new(internal_call_id: impl Into<String>, result: UserContent) -> Self {
        Self {
            internal_call_id: internal_call_id.into(),
            result,
            disposition: ToolInvocationDisposition::ExternallyExecuted,
        }
    }

    pub(crate) fn with_disposition(
        internal_call_id: impl Into<String>,
        result: UserContent,
        disposition: ToolInvocationDisposition,
    ) -> Self {
        Self {
            internal_call_id: internal_call_id.into(),
            result,
            disposition,
        }
    }
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

/// Canonical accepted model turn owned by the shared run state.
///
/// Unary and streaming drivers surface their normalized turn policy from this
/// record instead of reconstructing the boundary from provider-local state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AcceptedModelTurn {
    /// One-based model-call index within the run.
    pub turn: usize,
    /// Provider-assigned assistant message ID, when available.
    pub message_id: Option<String>,
    /// Canonical content after invalid-call repair and validation.
    pub content: OneOrMany<AssistantContent>,
    /// Usage reported for this provider call.
    pub usage: Usage,
    /// Whether this medium's provider-response observation is suppressed.
    pub response_hook_suppressed: bool,
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
    Continue(AcceptedModelTurn),
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
    usage: Usage,
    /// Working copy of the assistant content; repairs rename tool calls here.
    items: Vec<AssistantContent>,
    /// Original calls aligned with `items`; populated only where repair makes
    /// the effective call differ from provider output.
    original_calls: Vec<Option<ToolCall>>,
    /// Index of the next item to validate.
    next_index: usize,
    executable_tool_names: BTreeSet<String>,
    allowed_tool_names: BTreeSet<String>,
    /// Synthetic results positionally aligned with `items`. Provider IDs are
    /// payload and may duplicate.
    skipped: Vec<Option<PreresolvedToolResult>>,
    recovered: bool,
    any_skipped: bool,
    has_tool_calls: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TurnState {
    accepted: AcceptedModelTurn,
    /// Whether a driver has already applied the normalized Continue verdict.
    /// Kept in the run so a checkpoint between replying and `next_step` does
    /// not surface the steering boundary twice after resume.
    #[serde(default = "model_turn_verdict_pending_default")]
    verdict_pending: bool,
    has_tool_calls: bool,
    /// Synthetic results positionally aligned with `items`.
    skipped: Vec<Option<PreresolvedToolResult>>,
    /// Original provider calls aligned with the tool-call subsequence.
    original_tool_calls: Vec<Option<ToolCall>>,
    /// Rig identities positionally aligned with the streamed tool-call
    /// subsequence; empty for non-streamed turns.
    #[serde(default)]
    internal_call_ids: Vec<String>,
}

const fn model_turn_verdict_pending_default() -> bool {
    true
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
    /// Create a run for one prompt with no input history, a one-model-call
    /// budget, and no invalid tool-call retries.
    pub fn new(prompt: impl Into<Message>) -> Self {
        Self {
            max_turns: 1,
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
        self.set_history(history);
        self
    }

    /// Replace the input chat history preceding the prompt.
    pub fn set_history(&mut self, history: Vec<Message>) {
        self.chat_history = Some(history);
    }

    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. A budget of zero emits no model calls. Exceeding
    /// the budget makes [`AgentRun::next_step`] return
    /// [`PromptError::MaxTurnsError`].
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

    /// Whether the run may re-prompt for valid Tool-mode output: both the
    /// output-retry budget and the total model-call budget must remain.
    /// Otherwise, finalize best-effort rather than surface a max-turns error.
    fn can_reprompt_for_output(&self) -> bool {
        self.output_retries < self.max_output_retries && self.current_turn < self.max_turns
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

    /// Set the retry budget for [`InvalidToolCallAction::Retry`]
    /// resolutions. Invalid tool-call retries also consume the total model-call
    /// budget.
    pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
        self.max_invalid_tool_call_retries = retries;
        self
    }

    /// Set the tool choice active for this run. Used to reject
    /// [`InvalidToolCallAction::Skip`] resolutions under
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
    pub fn set_output_tool_name(&mut self, name: Option<String>) {
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
    pub fn output_tool_name(&self) -> Option<&str> {
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

    /// Canonical accepted model turn awaiting advancement.
    ///
    /// `Some` only between [`AgentRun::model_response`] accepting a turn and
    /// the next [`AgentRun::next_step`] — the window where a driver surfaces
    /// its model-turn-finished decision point.
    pub fn accepted_turn(&self) -> Option<&AcceptedModelTurn> {
        let RunState::AwaitingAdvance(turn) = &self.state else {
            return None;
        };

        Some(&turn.accepted)
    }

    /// Accepted turn whose normalized Continue verdict has not yet been
    /// applied by a driver.
    ///
    /// Unlike [`Self::accepted_turn`], this becomes `None` after
    /// [`Self::continue_model_turn`] even if the run has not yet emitted its
    /// next step. That distinction makes the steering boundary resume-durable.
    pub fn pending_accepted_turn(&self) -> Option<&AcceptedModelTurn> {
        let RunState::AwaitingAdvance(turn) = &self.state else {
            return None;
        };

        turn.verdict_pending.then_some(&turn.accepted)
    }

    /// Persist the driver's Continue verdict without advancing the run.
    ///
    /// Drivers call this after accepting `ModelTurnFinished`. The separate
    /// transition allows a serialized run to distinguish an unanswered turn
    /// from one answered immediately before a process restart.
    pub fn continue_model_turn(&mut self) -> Result<(), PromptError> {
        let RunState::AwaitingAdvance(turn) = &mut self.state else {
            return Err(self.protocol_violation(
                "continue_model_turn called without an accepted turn awaiting advancement",
            ));
        };
        if !turn.verdict_pending {
            return Err(self.protocol_violation(
                "continue_model_turn called after the turn verdict was already applied",
            ));
        }
        turn.verdict_pending = false;
        Ok(())
    }

    /// Canonical content for the accepted model turn awaiting advancement.
    pub fn accepted_turn_choice(&self) -> Option<OneOrMany<AssistantContent>> {
        self.accepted_turn().map(|turn| turn.content.clone())
    }

    /// Reject the accepted, tool-free model turn and prepare another model call.
    ///
    /// [`RetryRequest::Repeat`] discards the rejected assistant response and
    /// reuses the same prompt and preceding history with fresh request
    /// preparation. [`RetryRequest::Feedback`] records the rejected response
    /// followed by corrective user feedback. Canonical empty assistant turns
    /// are omitted from history, matching normal turn advancement. Both modes
    /// preserve completion-call and usage accounting, and the next call consumes
    /// the existing total model-call budget.
    ///
    /// Tool-bearing turns cannot be retried through this operation because
    /// preserving them without matching tool results would create invalid
    /// provider-visible history. Use tool-call hooks to steer those turns.
    pub fn retry_model_turn(&mut self, request: RetryRequest) -> Result<(), PromptError> {
        let turn = match std::mem::replace(&mut self.state, RunState::Failed) {
            RunState::AwaitingAdvance(turn) => turn,
            other => {
                self.state = other;
                return Err(self.protocol_violation(
                    "retry_model_turn called without an accepted turn awaiting advancement",
                ));
            }
        };

        if turn.has_tool_calls {
            return Err(PromptError::prompt_cancelled(
                self.full_history(),
                "model-turn retry does not support tool-bearing model turns; use tool-call hooks instead",
            ));
        }

        match request {
            RetryRequest::Repeat => {}
            RetryRequest::Feedback(feedback) => {
                let content = turn.accepted.content;
                if !is_empty_assistant_turn(&content) {
                    self.new_messages.push(Message::Assistant {
                        id: turn.accepted.message_id,
                        content,
                    });
                }
                self.new_messages.push(Message::user(feedback));
            }
        }

        self.state = RunState::PreparingRequest;
        Ok(())
    }

    /// Recover from a failed provider call: transition a pending
    /// [`AgentRunStep::CallModel`] back to the pre-call state so the next
    /// [`AgentRun::next_step`] re-issues the call instead of returning a
    /// protocol violation. Refunds the model-call budget the failed attempt
    /// consumed (the provider never answered, so no turn happened).
    ///
    /// Returns `false` (and changes nothing) when no model call is pending.
    pub fn abandon_pending_model_call(&mut self) -> bool {
        if !matches!(self.state, RunState::AwaitingModel) {
            return false;
        }
        self.current_turn = self.current_turn.saturating_sub(1);
        self.rollback_pending = false;
        self.streamed_completion_call_recorded = false;
        self.state = RunState::PreparingRequest;
        true
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
            args: Some(json_utils::serialize_json_value(
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
    /// - [`PromptError::MaxTurnsError`] when the total model-call budget is exhausted.
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

                if self.current_turn >= self.max_turns {
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
                    accepted,
                    verdict_pending: _,
                    has_tool_calls,
                    skipped,
                    original_tool_calls,
                    internal_call_ids,
                } = *turn_state;
                let AcceptedModelTurn {
                    message_id,
                    content: choice,
                    ..
                } = accepted;
                let items: Vec<AssistantContent> = choice.iter().cloned().collect();

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
                    let output_tool_calls = items
                        .iter()
                        .filter(|item| {
                            matches!(
                                item,
                                AssistantContent::ToolCall(tc)
                                    if tc.function.name == output_tool_name
                            )
                        })
                        .count();
                    let args = tool_call.function.arguments.clone();
                    let tool_call_id = tool_call.id.clone();
                    let output = json_utils::serialize_json_value(&args);

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
                    let final_content = OneOrMany::from_iter_optional(final_items);
                    if let Some(content) = final_content.clone() {
                        self.new_messages.push(Message::Assistant {
                            id: message_id,
                            content,
                        });
                    }

                    let mut response = PromptResponse::new(output, self.usage)
                        .with_messages(self.new_messages.clone())
                        .with_completion_calls(self.completion_calls.clone())
                        .with_output_tool_calls(output_tool_calls);
                    if let Some(content) = final_content {
                        response = response.with_content(content);
                    }
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
                    let mut internal_call_ids = internal_call_ids.into_iter();
                    let mut original_tool_calls = original_tool_calls.into_iter();
                    let mut claimed_internal_ids = BTreeSet::new();
                    let calls: Vec<PendingToolCall> = items
                        .iter()
                        .enumerate()
                        .filter_map(|(item_index, item)| match item {
                            AssistantContent::ToolCall(tool_call) => {
                                let original_tool_call = original_tool_calls.next().flatten();
                                // Identity is aligned by batch position;
                                // provider IDs are payload and may duplicate.
                                // Generate missing or duplicate Rig IDs before
                                // storing ExecutingTools so serialization and
                                // every resumed driver observe the same value.
                                let internal_call_id = match internal_call_ids.next() {
                                    Some(id) if claimed_internal_ids.insert(id.clone()) => id,
                                    Some(_) | None => loop {
                                        let id = rig_core::id::generate();
                                        if claimed_internal_ids.insert(id.clone()) {
                                            break id;
                                        }
                                    },
                                };
                                let preresolved =
                                    skipped.get(item_index).and_then(|result| result.clone());
                                Some(PendingToolCall {
                                    tool_call: tool_call.clone(),
                                    preresolved_result: preresolved
                                        .as_ref()
                                        .map(|result| result.result.clone()),
                                    internal_call_id: Some(internal_call_id),
                                    original_tool_call: original_tool_call.map(Box::new),
                                    invocation_disposition: preresolved
                                        .map(|result| result.disposition),
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
                            .with_completion_calls(self.completion_calls.clone())
                            .with_content(choice.clone());
                    self.state = RunState::Done(Box::new(response.clone()));
                    Ok(AgentRunStep::Done(response))
                }
            }
            RunState::ExecutingTools(calls) => {
                // Idempotent, like Done: a process resuming a serialized run
                // re-obtains the pending tool calls from the state itself.
                // Repair legacy or manually constructed state before cloning
                // it so every public result boundary has durable Rig identity.
                let mut calls = calls;
                let mut claimed = BTreeSet::new();
                for call in &mut calls {
                    let keep = call
                        .internal_call_id
                        .as_ref()
                        .is_some_and(|id| claimed.insert(id.clone()));
                    if !keep {
                        loop {
                            let id = rig_core::id::generate();
                            if claimed.insert(id.clone()) {
                                call.internal_call_id = Some(id);
                                break;
                            }
                        }
                    }
                }
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

        self.record_completion_call(turn.usage);

        let items: Vec<AssistantContent> = turn.choice.iter().cloned().collect();
        let has_tool_calls = items
            .iter()
            .any(|item| matches!(item, AssistantContent::ToolCall(_)));

        let skipped = vec![None; items.len()];
        let original_calls = vec![None; items.len()];
        self.state = RunState::ResolvingToolCalls(Box::new(ResolvingState {
            message_id: turn.message_id,
            original_choice: turn.choice,
            usage: turn.usage,
            items,
            original_calls,
            next_index: 0,
            executable_tool_names: turn.executable_tool_names,
            allowed_tool_names: turn.allowed_tool_names,
            skipped,
            recovered: false,
            any_skipped: false,
            has_tool_calls,
        }));

        self.advance_resolution()
    }

    /// Record one provider completion call: assign it the next call index,
    /// push it, and aggregate its usage into the run total. The single home for
    /// this accounting arithmetic, shared by the non-streamed and streamed
    /// ingestion paths. Callers own the once-per-turn `streamed_completion_call_recorded`
    /// guard/flag; this helper never touches it, so it cannot be mistaken for
    /// "a completion call happened" and re-introduce a double count.
    fn record_completion_call(&mut self, usage: Usage) -> CompletionCall {
        let call = CompletionCall::new(self.completion_call_index, usage);
        self.completion_call_index += 1;
        self.completion_calls.push(call);
        self.usage += usage;
        call
    }

    /// Park an accepted model turn in [`RunState::AwaitingAdvance`]. Both the
    /// non-streamed (`advance_resolution`) and streamed (`streamed_turn`)
    /// ingestion paths converge here, differing only in the positional
    /// `skipped` results and the streamed `internal_call_ids`.
    fn finalize_turn(
        &mut self,
        accepted: AcceptedModelTurn,
        has_tool_calls: bool,
        skipped: Vec<Option<PreresolvedToolResult>>,
        original_tool_calls: Vec<Option<ToolCall>>,
        internal_call_ids: Vec<String>,
    ) {
        self.state = RunState::AwaitingAdvance(Box::new(TurnState {
            accepted,
            verdict_pending: true,
            has_tool_calls,
            skipped,
            original_tool_calls,
            internal_call_ids,
        }));
    }

    /// Answer a pending [`ModelTurnOutcome::NeedsResolution`].
    ///
    /// Applies the agent loop's recovery semantics:
    /// - [`InvalidToolCallAction::Fail`] fails the run with
    ///   [`PromptError::UnknownToolCall`].
    /// - [`InvalidToolCallAction::Retry`] rolls the turn back with
    ///   corrective feedback while budget remains, consuming the total
    ///   model-call budget.
    /// - [`InvalidToolCallAction::Repair`] renames the tool call; the
    ///   repaired name is revalidated against the allowed tools.
    /// - [`InvalidToolCallAction::Stop`] cancels the run with
    ///   `PromptError::prompt_cancelled` and the supplied reason.
    /// - [`InvalidToolCallAction::Skip`] records a synthetic tool result
    ///   and suppresses execution of every tool call in the turn. Rejected
    ///   under [`ToolChoice::None`].
    /// - [`InvalidToolCallAction::Ignore`] drops the call from the turn with
    ///   no model-visible feedback, letting a sibling output-tool call still
    ///   finalize the turn (the extraction protocol's policy).
    pub fn resolve_invalid_tool_call(
        &mut self,
        action: InvalidToolCallAction,
    ) -> Result<ModelTurnOutcome, PromptError> {
        if matches!(action, InvalidToolCallAction::Ignore) {
            return self.ignore_invalid_tool_call();
        }
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
            // Handled by the early return above; the state has already been
            // taken here, so restore it and delegate.
            InvalidToolCallAction::Ignore => {
                self.state = RunState::ResolvingToolCalls(resolving);
                self.ignore_invalid_tool_call()
            }
            InvalidToolCallAction::Fail => Err(unknown_tool_call_error(
                tool_call.function.name,
                executable_tool_names,
                allowed_tool_names,
                diagnostic_history,
            )),
            InvalidToolCallAction::Retry { feedback } => {
                if self.invalid_tool_call_retries >= self.max_invalid_tool_call_retries {
                    return Err(unknown_tool_call_error(
                        tool_call.function.name,
                        executable_tool_names,
                        allowed_tool_names,
                        diagnostic_history,
                    ));
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
            InvalidToolCallAction::Repair { tool_name } => {
                if !allowed_tool_names.contains(&tool_name) {
                    return Err(unknown_tool_call_error(
                        tool_name,
                        executable_tool_names,
                        allowed_tool_names,
                        diagnostic_history,
                    ));
                }
                if let Some(AssistantContent::ToolCall(tool_call)) =
                    resolving.items.get_mut(resolving.next_index)
                {
                    let Some(original_slot) =
                        resolving.original_calls.get_mut(resolving.next_index)
                    else {
                        self.state = RunState::ResolvingToolCalls(resolving);
                        return Err(self.protocol_violation(
                            "internal: repaired call lost its original positional slot",
                        ));
                    };
                    *original_slot = Some(tool_call.clone());
                    tool_call.function.name = tool_name;
                }
                resolving.recovered = true;
                self.state = RunState::ResolvingToolCalls(resolving);
                self.advance_resolution()
            }
            InvalidToolCallAction::Stop { reason } => {
                self.state = RunState::Failed;
                Err(PromptError::prompt_cancelled(diagnostic_history, reason))
            }
            InvalidToolCallAction::Skip { reason } => {
                if matches!(self.tool_choice, Some(ToolChoice::None)) {
                    return Err(unknown_tool_call_error(
                        tool_call.function.name,
                        executable_tool_names,
                        allowed_tool_names,
                        diagnostic_history,
                    ));
                }
                let user_content = if let Some(call_id) = tool_call.call_id.clone() {
                    UserContent::tool_result_with_call_id(
                        tool_call.id.clone(),
                        call_id,
                        OneOrMany::one(reason.clone().into()),
                    )
                } else {
                    UserContent::tool_result(
                        tool_call.id.clone(),
                        OneOrMany::one(reason.clone().into()),
                    )
                };
                let Some(slot) = resolving.skipped.get_mut(resolving.next_index) else {
                    self.state = RunState::ResolvingToolCalls(resolving);
                    return Err(self.protocol_violation(
                        "internal: invalid-call result lost its positional slot",
                    ));
                };
                *slot = Some(PreresolvedToolResult {
                    result: user_content,
                    disposition: ToolInvocationDisposition::not_executed(reason),
                });
                resolving.recovered = true;
                resolving.any_skipped = true;
                resolving.next_index += 1;
                self.state = RunState::ResolvingToolCalls(resolving);
                self.advance_resolution()
            }
        }
    }

    /// Discard the pending invalid tool call without marking the turn as
    /// recovered.
    ///
    /// Reached through [`InvalidToolCallAction::Ignore`]. Keeping it distinct
    /// from [`InvalidToolCallAction::Skip`] preserves the extraction
    /// protocol's response semantics: unrelated calls disappear, a sibling
    /// output call can still finalize the turn, and response observers still
    /// receive the canonical response fields.
    pub(crate) fn ignore_invalid_tool_call(&mut self) -> Result<ModelTurnOutcome, PromptError> {
        let mut resolving = match std::mem::replace(&mut self.state, RunState::Failed) {
            RunState::ResolvingToolCalls(resolving) => resolving,
            other => {
                self.state = other;
                return Err(self.protocol_violation(
                    "ignore_invalid_tool_call called without a pending invalid tool call",
                ));
            }
        };

        match resolving.items.get(resolving.next_index) {
            Some(AssistantContent::ToolCall(tool_call))
                if !resolving
                    .allowed_tool_names
                    .contains(&tool_call.function.name) => {}
            _ => {
                self.state = RunState::ResolvingToolCalls(resolving);
                return Err(self.protocol_violation(
                    "ignore_invalid_tool_call called without a pending invalid tool call",
                ));
            }
        }

        resolving.items.remove(resolving.next_index);
        resolving.skipped.remove(resolving.next_index);
        resolving.original_calls.remove(resolving.next_index);
        resolving.has_tool_calls = resolving
            .items
            .iter()
            .any(|item| matches!(item, AssistantContent::ToolCall(_)));
        if resolving.items.is_empty() {
            resolving.items.push(AssistantContent::text(""));
            resolving.skipped.push(None);
            resolving.original_calls.push(None);
        }
        self.state = RunState::ResolvingToolCalls(resolving);
        self.advance_resolution()
    }

    /// Feed unambiguous provider-ID-keyed tool results for the pending
    /// [`AgentRunStep::CallTools`].
    ///
    /// This convenience path is available only when provider call IDs are
    /// unique within the batch. For duplicate provider IDs, use
    /// [`Self::tool_result_submissions`] so each result carries Rig's unique
    /// invocation identity.
    pub fn tool_results(&mut self, results: Vec<UserContent>) -> Result<(), PromptError> {
        let RunState::ExecutingTools(pending) = &self.state else {
            return Err(
                self.protocol_violation("tool_results called without a pending CallTools step")
            );
        };
        let mut provider_ids = BTreeSet::new();
        if pending
            .iter()
            .any(|call| !provider_ids.insert(call.tool_call.id.clone()))
        {
            return Err(self.protocol_violation(
                "tool_results cannot associate duplicate provider call IDs; use \
                 tool_result_submissions with each PendingToolCall internal_call_id",
            ));
        }
        let mut submissions = Vec::with_capacity(results.len());
        for result in results {
            let UserContent::ToolResult(tool_result) = result else {
                return Err(self.protocol_violation(
                    "tool_results received content that is not a tool result",
                ));
            };
            let Some(call) = pending
                .iter()
                .find(|call| call.tool_call.id == tool_result.id)
            else {
                return Err(self.protocol_violation(&format!(
                    "tool_results received a result for unknown tool call id `{}`",
                    tool_result.id
                )));
            };
            let Some(internal_call_id) = call.internal_call_id.clone() else {
                return Err(self.protocol_violation(
                    "pending tool call has no Rig internal identity; call next_step before submitting results",
                ));
            };
            submissions.push(ToolResultSubmission::new(
                internal_call_id,
                UserContent::ToolResult(tool_result),
            ));
        }
        self.tool_result_submissions(submissions)
    }

    /// Feed results keyed by Rig's unique invocation identity.
    ///
    /// Submissions may be in any order. They are validated and joined by
    /// identity, then committed in the pending calls' assistant source order
    /// as one provider-compatible user message.
    /// Every pending invocation must be answered exactly once, and the
    /// embedded provider result `id` and `call_id` must still match that
    /// invocation's provider correlation fields.
    pub fn tool_result_submissions(
        &mut self,
        submissions: Vec<ToolResultSubmission>,
    ) -> Result<(), PromptError> {
        let RunState::ExecutingTools(pending) = &self.state else {
            return Err(self.protocol_violation(
                "tool_result_submissions called without a pending CallTools step",
            ));
        };
        if submissions.is_empty() {
            self.state = RunState::Failed;
            return Err(PromptError::prompt_cancelled(
                self.full_history(),
                "tool execution produced no tool results",
            ));
        }

        let mut answered = BTreeSet::new();
        let mut results_by_id = BTreeMap::new();
        for submission in submissions {
            let UserContent::ToolResult(tool_result) = &submission.result else {
                return Err(self.protocol_violation(
                    "tool_result_submissions received content that is not a tool result",
                ));
            };
            let Some(call) = pending.iter().find(|call| {
                call.internal_call_id.as_deref() == Some(submission.internal_call_id.as_str())
            }) else {
                return Err(self.protocol_violation(&format!(
                    "tool_result_submissions received unknown Rig internal call id `{}`",
                    submission.internal_call_id
                )));
            };
            if tool_result.id != call.tool_call.id {
                return Err(self.protocol_violation(&format!(
                    "tool result for Rig internal call id `{}` carries provider id `{}` instead of `{}`",
                    submission.internal_call_id, tool_result.id, call.tool_call.id
                )));
            }
            if tool_result.call_id != call.tool_call.call_id {
                return Err(self.protocol_violation(&format!(
                    "tool result for Rig internal call id `{}` carries provider call_id {:?} instead of {:?}",
                    submission.internal_call_id, tool_result.call_id, call.tool_call.call_id
                )));
            }
            if !answered.insert(submission.internal_call_id.clone()) {
                return Err(self.protocol_violation(&format!(
                    "tool_result_submissions answered Rig internal call id `{}` more than once",
                    submission.internal_call_id
                )));
            }
            results_by_id.insert(submission.internal_call_id, submission.result);
        }

        let mut unanswered = Vec::new();
        let mut results = Vec::with_capacity(pending.len());
        for call in pending {
            let Some(internal_call_id) = &call.internal_call_id else {
                return Err(self.protocol_violation(
                    "pending tool call has no Rig internal identity; call next_step before submitting results",
                ));
            };
            if !answered.contains(internal_call_id) {
                unanswered.push(internal_call_id.clone());
            } else if let Some(result) = results_by_id.remove(internal_call_id) {
                results.push(result);
            } else {
                return Err(self.protocol_violation(
                    "internal: validated tool result lost its Rig invocation identity",
                ));
            }
        }
        if !unanswered.is_empty() {
            return Err(self.protocol_violation(&format!(
                "tool_result_submissions left Rig internal call id(s) unanswered: {unanswered:?}"
            )));
        }

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
            usage,
            original_calls,
            mut skipped,
            recovered,
            any_skipped,
            has_tool_calls,
            ..
        } = *resolving;

        // When any tool call was skipped, none of the turn's tool calls
        // execute: peers get a synthetic "not executed" result.
        if any_skipped {
            for (item, slot) in items.iter().zip(&mut skipped) {
                if let AssistantContent::ToolCall(tool_call) = item
                    && slot.is_none()
                {
                    let reason = TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_string();
                    *slot = Some(PreresolvedToolResult {
                        result: tool_result_message(
                            tool_call.id.clone(),
                            tool_call.call_id.clone(),
                            reason.clone(),
                        ),
                        disposition: ToolInvocationDisposition::not_executed(reason),
                    });
                }
            }
        }

        let original_tool_calls = items
            .iter()
            .enumerate()
            .filter(|(_, item)| matches!(item, AssistantContent::ToolCall(_)))
            .map(|(index, _)| original_calls.get(index).cloned().flatten())
            .collect();
        let content = OneOrMany::from_iter_optional(items).ok_or_else(|| {
            self.protocol_violation("internal: accepted model turn lost its assistant content")
        })?;
        let accepted = AcceptedModelTurn {
            turn: self.current_turn,
            message_id,
            content,
            usage,
            response_hook_suppressed: recovered,
        };
        self.finalize_turn(
            accepted.clone(),
            has_tool_calls,
            skipped,
            original_tool_calls,
            Vec::new(),
        );
        Ok(ModelTurnOutcome::Continue(accepted))
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

        Ok(self.record_completion_call(usage))
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
        action: InvalidToolCallAction,
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
            InvalidToolCallAction::Fail => {
                self.state = RunState::Failed;
                Err(unknown_tool_call_error(
                    invalid.tool_call.function.name.clone(),
                    executable_tool_names,
                    allowed_tool_names,
                    diagnostic_history,
                ))
            }
            // The streamed path has already emitted the call downstream, so a
            // silent drop is not expressible; `Skip` is the streaming
            // equivalent (it records synthetic feedback instead).
            InvalidToolCallAction::Ignore => {
                self.state = RunState::Failed;
                Err(PromptError::prompt_cancelled(
                    diagnostic_history,
                    "InvalidToolCallAction::Ignore is not supported while streaming; use Skip",
                ))
            }
            InvalidToolCallAction::Retry { feedback } => {
                if self.invalid_tool_call_retries >= self.max_invalid_tool_call_retries {
                    self.state = RunState::Failed;
                    return Err(unknown_tool_call_error(
                        invalid.tool_call.function.name.clone(),
                        executable_tool_names,
                        allowed_tool_names,
                        diagnostic_history,
                    ));
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
            InvalidToolCallAction::Repair { tool_name } => {
                if !invalid.allowed_tool_names.contains(&tool_name) {
                    self.state = RunState::Failed;
                    return Err(unknown_tool_call_error(
                        tool_name,
                        executable_tool_names,
                        allowed_tool_names,
                        diagnostic_history,
                    ));
                }
                Ok(StreamedResolution::Repaired { tool_name })
            }
            InvalidToolCallAction::Stop { reason } => {
                self.state = RunState::Failed;
                Err(PromptError::prompt_cancelled(diagnostic_history, reason))
            }
            InvalidToolCallAction::Skip { reason } => {
                if matches!(self.tool_choice, Some(ToolChoice::None)) {
                    self.state = RunState::Failed;
                    return Err(unknown_tool_call_error(
                        invalid.tool_call.function.name.clone(),
                        executable_tool_names,
                        allowed_tool_names,
                        diagnostic_history,
                    ));
                }

                // Synthetic skip reason: emit verbatim text, matching the
                // non-streamed `resolve_invalid_tool_call` skip path (parity) and
                // avoiding re-parsing a rejection message as structured output.
                let skipped_tool_result = ToolResult {
                    id: invalid.tool_call.id.clone(),
                    call_id: invalid.tool_call.call_id.clone(),
                    content: OneOrMany::one(ToolResultContent::text(reason.clone())),
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
    pub fn streamed_turn(&mut self, turn: StreamedTurn) -> Result<AcceptedModelTurn, PromptError> {
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
                return Err(unknown_tool_call_error(
                    tool_call.function.name.clone(),
                    turn.executable_tool_names.iter().cloned().collect(),
                    turn.allowed_tool_names.iter().cloned().collect(),
                    diagnostic_history,
                ));
            }
        }

        // Commit usage only after the assembled turn validates. Older manual
        // drivers may already have recorded the provider-final usage; the
        // transactional driver carries it on `StreamedTurn` and records here.
        if !self.streamed_completion_call_recorded {
            self.record_completion_call(turn.usage);
            self.streamed_completion_call_recorded = true;
        }

        let skipped = vec![None; items.len()];
        let accepted = AcceptedModelTurn {
            turn: self.current_turn,
            message_id: turn.message_id,
            content: turn.choice,
            usage: turn.usage,
            response_hook_suppressed: false,
        };
        self.finalize_turn(
            accepted.clone(),
            has_tool_calls,
            skipped,
            turn.original_tool_calls,
            turn.internal_call_ids,
        );
        Ok(accepted)
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
    use rig_core::message::{ToolFunction, ToolResultContent};
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

    fn tool_call_with_call_id(id: &str, call_id: &str, name: &str) -> AssistantContent {
        AssistantContent::ToolCall(
            ToolCall::new(
                id.to_string(),
                ToolFunction::new(name.to_string(), json!({"x": 1})),
            )
            .with_call_id(call_id.to_string()),
        )
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
            OneOrMany::one(ToolResultContent::text(output)),
        )
    }

    fn tool_result_with_call_id(id: &str, call_id: &str, output: &str) -> UserContent {
        UserContent::tool_result_with_call_id(
            id.to_string(),
            call_id.to_string(),
            OneOrMany::one(ToolResultContent::text(output)),
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
            ModelTurnOutcome::Continue(accepted) => accepted.response_hook_suppressed,
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
    fn repeated_model_turn_reuses_prompt_without_recording_rejected_response() {
        let first_usage = usage(10, 3);
        let second_usage = usage(7, 2);
        let mut run = AgentRun::new("question").max_turns(2);

        let (first_prompt, first_history, first_turn) = expect_call_model(&mut run);
        assert_eq!(first_prompt, Message::user("question"));
        assert!(first_history.is_empty());
        assert_eq!(first_turn, 1);
        expect_continue(
            run.model_response(text_turn("rejected").with_usage_for_test(first_usage))
                .expect("first response"),
        );

        run.retry_model_turn(RetryRequest::Repeat)
            .expect("repeat should be accepted");
        let (second_prompt, second_history, second_turn) = expect_call_model(&mut run);
        assert_eq!(second_prompt, Message::user("question"));
        assert!(second_history.is_empty());
        assert_eq!(second_turn, 2);
        assert_eq!(run.messages(), &[Message::user("question")]);

        expect_continue(
            run.model_response(text_turn("accepted").with_usage_for_test(second_usage))
                .expect("second response"),
        );
        let response = expect_done(&mut run);
        assert_eq!(response.output, "accepted");
        assert_eq!(response.usage, first_usage + second_usage);
        assert_eq!(response.completion_calls.len(), 2);
        let messages = response.messages.expect("response history");
        assert_eq!(messages.len(), 2);
        assert!(!format!("{messages:?}").contains("rejected"));
    }

    #[test]
    fn feedback_retry_records_rejected_response_and_corrective_prompt() {
        let mut run = AgentRun::new("question").max_turns(2);

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(text_turn("rejected"))
                .expect("first response"),
        );
        run.retry_model_turn(RetryRequest::Feedback("try another approach".to_string()))
            .expect("feedback retry should be accepted");

        let (prompt, history, turn) = expect_call_model(&mut run);
        assert_eq!(prompt, Message::user("try another approach"));
        assert_eq!(turn, 2);
        assert_eq!(
            history,
            vec![Message::user("question"), Message::assistant("rejected")]
        );
    }

    #[test]
    fn repeated_model_turn_consumes_existing_max_turns_budget() {
        let mut run = AgentRun::new("question");

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(text_turn("rejected"))
                .expect("first response"),
        );
        run.retry_model_turn(RetryRequest::Repeat)
            .expect("state transition itself should succeed");

        let err = run.next_step().expect_err("second call must exceed budget");
        assert!(matches!(
            err,
            PromptError::MaxTurnsError { max_turns: 1, .. }
        ));
        assert_eq!(run.completion_calls().len(), 1);
    }

    #[test]
    fn model_turn_retry_rejects_tool_calls_without_advancing_to_execution() {
        let mut run = AgentRun::new("add things").max_turns(2);

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add"))
                .expect("tool response"),
        );
        let err = run
            .retry_model_turn(RetryRequest::Feedback("do not call tools".to_string()))
            .expect_err("tool-bearing retries must fail closed");

        let PromptError::PromptCancelled {
            chat_history,
            reason,
        } = err
        else {
            panic!("tool-bearing retry should return PromptCancelled");
        };
        assert!(reason.contains("tool-bearing model turns"));
        assert!(reason.contains("tool-call hooks"));
        assert_eq!(chat_history, vec![Message::user("add things")]);
        assert!(run.next_step().is_err(), "failed run cannot execute tools");
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
    fn max_turns_zero_rejects_initial_model_call() {
        let mut run = AgentRun::new("do not call").max_turns(0);

        let err = run
            .next_step()
            .expect_err("zero budget should emit no call");
        assert!(matches!(
            err,
            PromptError::MaxTurnsError { max_turns: 0, .. }
        ));
        assert_eq!(run.turn(), 0);
    }

    #[test]
    fn new_implicitly_allows_one_model_call_and_rejects_tool_continuation() {
        let mut run = AgentRun::new("add things");

        let (_, _, turn) = expect_call_model(&mut run);
        assert_eq!(turn, 1);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add"))
                .expect("model_response should succeed"),
        );
        expect_call_tools(&mut run);
        run.tool_results(vec![tool_result("call_1", "2")])
            .expect("tool_results should succeed");

        let err = run
            .next_step()
            .expect_err("second model call should exceed budget");
        assert!(matches!(
            err,
            PromptError::MaxTurnsError { max_turns: 1, .. }
        ));
        assert_eq!(run.turn(), 1);
    }

    #[test]
    fn max_turns_n_allows_exactly_n_model_calls() {
        let mut run = AgentRun::new("loop").max_turns(3);

        for (expected_turn, call_id) in [(1, "call_1"), (2, "call_2"), (3, "call_3")] {
            let (_, _, turn) = expect_call_model(&mut run);
            assert_eq!(turn, expected_turn);
            expect_continue(
                run.model_response(tool_call_turn(call_id, "add"))
                    .expect("model_response should succeed"),
            );
            expect_call_tools(&mut run);
            run.tool_results(vec![tool_result(call_id, "0")])
                .expect("tool_results should succeed");
        }

        let err = run
            .next_step()
            .expect_err("fourth model call should exceed budget");
        assert!(matches!(
            err,
            PromptError::MaxTurnsError { max_turns: 3, .. }
        ));
        assert_eq!(run.turn(), 3);
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
            .resolve_invalid_tool_call(InvalidToolCallAction::fail())
            .expect_err("fail action should error");
        assert!(matches!(
            err,
            PromptError::UnknownToolCall { tool_name, .. } if tool_name == "unknown"
        ));
    }

    #[test]
    fn invalid_tool_call_stop_leaves_run_terminal() {
        let mut run = AgentRun::new("call something");

        expect_call_model(&mut run);
        expect_needs_resolution(
            run.model_response(tool_call_turn("call_1", "unknown"))
                .expect("model_response should succeed"),
        );
        let err = run
            .resolve_invalid_tool_call(InvalidToolCallAction::stop("operator stop"))
            .expect_err("stop should cancel the run");
        assert!(matches!(
            err,
            PromptError::PromptCancelled { reason, .. } if reason == "operator stop"
        ));

        let err = run
            .next_step()
            .expect_err("a stopped run must remain terminal");
        assert!(matches!(
            err,
            PromptError::PromptCancelled { reason, .. }
                if reason.contains("next_step called after the run already failed")
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
            .resolve_invalid_tool_call(InvalidToolCallAction::retry("use add instead"))
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
            .resolve_invalid_tool_call(InvalidToolCallAction::retry("again"))
            .expect_err("budget exhausted");
        assert!(matches!(err, PromptError::UnknownToolCall { .. }));
    }

    #[test]
    fn invalid_tool_call_retry_cannot_emit_call_past_total_budget() {
        let mut run = AgentRun::new("call something")
            .max_turns(1)
            .max_invalid_tool_call_retries(1);

        expect_call_model(&mut run);
        expect_needs_resolution(
            run.model_response(tool_call_turn("call_1", "unknown"))
                .expect("model_response should succeed"),
        );
        let outcome = run
            .resolve_invalid_tool_call(InvalidToolCallAction::retry("use add instead"))
            .expect("retry resolution should be accepted");
        assert!(matches!(outcome, ModelTurnOutcome::TurnRetried));
        assert_eq!(run.completion_calls().len(), 1);

        let err = run
            .next_step()
            .expect_err("retry must not emit a second model call");
        assert!(matches!(
            err,
            PromptError::MaxTurnsError { max_turns: 1, .. }
        ));
        assert_eq!(run.turn(), 1);
    }

    #[tokio::test]
    async fn invalid_tool_call_repair_renames_and_suppresses_response_hook() {
        let mut run = AgentRun::new("call something").max_turns(2);

        expect_call_model(&mut run);
        expect_needs_resolution(
            run.model_response(tool_call_turn("call_1", "default_api"))
                .expect("model_response should succeed"),
        );
        let suppressed = expect_continue(
            run.resolve_invalid_tool_call(InvalidToolCallAction::repair("add"))
                .expect("repair should be accepted"),
        );
        assert!(suppressed);

        let calls = expect_call_tools(&mut run);
        assert_eq!(calls[0].tool_call.function.name, "add");
        assert_eq!(
            calls[0]
                .original_tool_call
                .as_deref()
                .map(|call| call.function.name.as_str()),
            Some("default_api")
        );
        assert!(calls[0].preresolved_result.is_none());

        let executor =
            crate::executor::ToolExecutor::new().register(crate::tool::PortableDynamicTool::new(
                "add",
                "test repair audit",
                json!({"type": "object"}),
                |args| async move {
                    Ok::<_, crate::tool::ToolExecutionError>(crate::tool::ToolOutput::json(args))
                },
            ));
        let batch = executor.execute_batch(&calls).await;
        assert_eq!(batch.records.len(), 1);
        assert_eq!(batch.records[0].original_call.function.name, "default_api");
        assert_eq!(batch.records[0].effective_call.function.name, "add");
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
            .resolve_invalid_tool_call(InvalidToolCallAction::repair("also_unknown"))
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
            run.resolve_invalid_tool_call(InvalidToolCallAction::skip("not available"))
                .expect("skip should be accepted"),
        );
        assert!(suppressed);

        let calls = expect_call_tools(&mut run);
        assert_eq!(calls.len(), 2);
        // Both the skipped call and its valid peer carry preresolved results.
        assert!(calls.iter().all(|call| call.preresolved_result.is_some()));
    }

    #[test]
    fn invalid_skip_keeps_duplicate_provider_ids_positionally_correlated() {
        let mut run = AgentRun::new("call things").max_turns(2);

        expect_call_model(&mut run);
        let turn = ModelTurn::new(
            None,
            OneOrMany::many(vec![
                tool_call_with_call_id("duplicate", "valid-call", "add"),
                tool_call_with_call_id("duplicate", "invalid-call", "unknown"),
            ])
            .expect("two items"),
            Usage::new(),
            tool_names(&["add"]),
            tool_names(&["add"]),
        );
        expect_needs_resolution(run.model_response(turn).expect("model response"));
        expect_continue(
            run.resolve_invalid_tool_call(InvalidToolCallAction::skip("invalid reason"))
                .expect("skip should be accepted"),
        );

        let calls = expect_call_tools(&mut run);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool_call.id, "duplicate");
        assert_eq!(calls[1].tool_call.id, "duplicate");
        assert_eq!(calls[0].tool_call.call_id.as_deref(), Some("valid-call"));
        assert_eq!(calls[1].tool_call.call_id.as_deref(), Some("invalid-call"));

        let Some(UserContent::ToolResult(valid_peer_result)) = &calls[0].preresolved_result else {
            panic!("valid peer should have a positional synthetic result");
        };
        let Some(UserContent::ToolResult(invalid_result)) = &calls[1].preresolved_result else {
            panic!("invalid call should retain its own synthetic result");
        };
        assert_eq!(valid_peer_result.call_id.as_deref(), Some("valid-call"));
        assert_eq!(invalid_result.call_id.as_deref(), Some("invalid-call"));
        assert!(
            serde_json::to_string(valid_peer_result)
                .expect("result serializes")
                .contains(TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER)
        );
        assert!(
            serde_json::to_string(invalid_result)
                .expect("result serializes")
                .contains("invalid reason")
        );
        assert!(matches!(
            &calls[0].invocation_disposition,
            Some(ToolInvocationDisposition::NotExecuted { reason: Some(reason) })
                if reason == TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER
        ));
        assert!(matches!(
            &calls[1].invocation_disposition,
            Some(ToolInvocationDisposition::NotExecuted { reason: Some(reason) })
                if reason == "invalid reason"
        ));
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
            .resolve_invalid_tool_call(InvalidToolCallAction::skip("nope"))
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
    fn unary_duplicate_provider_ids_keep_internal_ids_across_serde_resume() {
        let mut run = AgentRun::new("add twice").max_turns(2);
        expect_call_model(&mut run);
        let turn = ModelTurn::new(
            None,
            OneOrMany::many(vec![
                tool_call_with_call_id("duplicate", "provider-call-a", "add"),
                tool_call_with_call_id("duplicate", "provider-call-b", "add"),
            ])
            .expect("two calls"),
            Usage::new(),
            tool_names(&["add"]),
            tool_names(&["add"]),
        );
        expect_continue(run.model_response(turn).expect("model response"));

        let before = expect_call_tools(&mut run);
        let before_ids = before
            .iter()
            .map(|call| call.internal_call_id.clone().expect("internal id"))
            .collect::<Vec<_>>();
        assert_ne!(before_ids[0], before_ids[1]);

        let serialized = serde_json::to_string(&run).expect("pending run serializes");
        let mut restored: AgentRun =
            serde_json::from_str(&serialized).expect("pending run deserializes");
        let after = expect_call_tools(&mut restored);
        let after_ids = after
            .iter()
            .map(|call| call.internal_call_id.clone().expect("internal id"))
            .collect::<Vec<_>>();

        assert_eq!(after_ids, before_ids);
        assert_eq!(after[0].tool_call.id, "duplicate");
        assert_eq!(after[1].tool_call.id, "duplicate");
        assert_eq!(
            after[0].tool_call.call_id.as_deref(),
            Some("provider-call-a")
        );
        assert_eq!(
            after[1].tool_call.call_id.as_deref(),
            Some("provider-call-b")
        );

        let legacy_error = restored
            .tool_results(vec![
                tool_result_with_call_id("duplicate", "provider-call-a", "3"),
                tool_result_with_call_id("duplicate", "provider-call-b", "7"),
            ])
            .expect_err("legacy provider-ID joining must reject duplicate IDs");
        assert!(matches!(legacy_error, PromptError::PromptCancelled { .. }));

        let mismatch = restored
            .tool_result_submissions(vec![
                ToolResultSubmission::new(
                    after_ids[1].clone(),
                    tool_result_with_call_id("duplicate", "provider-call-a", "7"),
                ),
                ToolResultSubmission::new(
                    after_ids[0].clone(),
                    tool_result_with_call_id("duplicate", "provider-call-b", "3"),
                ),
            ])
            .expect_err("swapped provider call_id values must be rejected");
        assert!(matches!(mismatch, PromptError::PromptCancelled { .. }));

        restored
            .tool_result_submissions(vec![
                ToolResultSubmission::new(
                    after_ids[1].clone(),
                    tool_result_with_call_id("duplicate", "provider-call-b", "7"),
                ),
                ToolResultSubmission::new(
                    after_ids[0].clone(),
                    tool_result_with_call_id("duplicate", "provider-call-a", "3"),
                ),
            ])
            .expect("Rig identities make reverse-ordered duplicate-ID results unambiguous");
        let Some(Message::User { content }) = restored.messages().last() else {
            panic!("tool results should commit as one user message");
        };
        let rendered = content
            .iter()
            .map(|item| serde_json::to_string(item).expect("result serializes"))
            .collect::<Vec<_>>();
        assert!(rendered[0].contains('3'));
        assert!(rendered[1].contains('7'));
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
    fn serde_round_trip_at_exhausted_budget_preserves_boundary() {
        let mut run = AgentRun::new("add things").max_turns(1);
        expect_call_model(&mut run);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add"))
                .expect("model_response should succeed"),
        );
        expect_call_tools(&mut run);
        run.tool_results(vec![tool_result("call_1", "2")])
            .expect("tool_results should succeed");

        let serialized = serde_json::to_string(&run).expect("exhausted run should serialize");
        let mut restored: AgentRun =
            serde_json::from_str(&serialized).expect("exhausted run should deserialize");
        assert_eq!(restored.completion_calls().len(), 1);
        let err = restored
            .next_step()
            .expect_err("restored run must not emit a second model call");
        assert!(matches!(
            err,
            PromptError::MaxTurnsError { max_turns: 1, .. }
        ));
        assert_eq!(restored.turn(), 1);
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

    fn output_tool_turn_with_args(id: &str, name: &str, arguments: serde_json::Value) -> ModelTurn {
        ModelTurn::new(
            None,
            OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                id.to_string(),
                ToolFunction::new(name.to_string(), arguments),
            ))),
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
    fn scalar_output_tool_call_is_serialized_as_reparseable_json() {
        let mut run = AgentRun::new("summarize").with_output_tool_name("final_result");

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(output_tool_turn_with_args(
                "call_1",
                "final_result",
                json!("complete"),
            ))
            .expect("model_response should succeed"),
        );

        let response = expect_done(&mut run);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&response.output)
                .expect("scalar output must remain valid JSON"),
            json!("complete")
        );
        assert_eq!(response.output, r#""complete""#);

        let messages = response.messages.expect("messages should be recorded");
        assert_no_orphan_tool_use(&messages);
        assert!(matches!(
            messages.last(),
            Some(Message::Assistant { content, .. })
                if assistant_text_from_choice(content) == r#""complete""#
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
            .max_turns(2)
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
            .max_turns(2)
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
    fn tool_mode_finalizes_best_effort_when_model_call_budget_exhausted() {
        let mut run = AgentRun::new("summarize")
            .max_turns(1)
            .with_output_tool_name("final_result")
            .with_output_validation(Some(required_field_schema("summary")), 1);

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(text_turn("invalid output"))
                .expect("model_response should succeed"),
        );

        let response = expect_done(&mut run);
        assert_eq!(response.output, "invalid output");
        assert_eq!(run.turn(), 1);
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

    /// Durable human-in-the-loop: the run is serialized while tool calls are
    /// pending, reconstructed from JSON (as a separate process / request would),
    /// and only then does the human decision land — approve one call, deny the
    /// other. The resumed-from-bytes run accepts those results and continues to
    /// completion, proving approval can happen out-of-process / arbitrarily later.
    /// This is the state-machine foundation for `examples/agent_with_durable_approval`.
    #[test]
    fn durable_human_in_the_loop_approval_survives_serialize_resume() {
        let mut run = AgentRun::new("pay two invoices").max_turns(3);
        let (_, _, turn) = expect_call_model(&mut run);
        assert_eq!(turn, 1);

        // Turn 1: the model emits two tool calls.
        let two_calls =
            OneOrMany::many([tool_call("c1", "add"), tool_call("c2", "add")]).expect("two calls");
        let outcome = run
            .model_response(ModelTurn::new(
                None,
                two_calls,
                Usage::new(),
                tool_names(&["add"]),
                tool_names(&["add"]),
            ))
            .expect("model_response");
        expect_continue(outcome);

        // CallTools is now pending. Serialize the run (a durable checkpoint) and
        // reconstruct it from the bytes — nothing live crosses this boundary.
        let checkpoint = serde_json::to_string(&run).expect("serialize suspended run");
        let mut resumed: AgentRun = serde_json::from_str(&checkpoint).expect("deserialize run");

        // The resumed run re-emits the pending calls purely from its own state.
        let calls = expect_call_tools(&mut resumed);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].tool_call.id, "c1");
        assert_eq!(calls[1].tool_call.id, "c2");

        // The human decision lands only after the resume: approve c1 (real
        // result), deny c2 (the reason becomes the tool result the model sees).
        resumed
            .tool_results(vec![
                tool_result("c1", "approved-result"),
                tool_result("c2", "denied by reviewer: second payment not authorized"),
            ])
            .expect("tool_results on the resumed run");

        // Both decisions are recorded in the resumed run's persisted state.
        let after = serde_json::to_string(&resumed).expect("serialize resumed run");
        assert!(
            after.contains("approved-result"),
            "the approved call's result must be in the resumed run state"
        );
        assert!(
            after.contains("denied by reviewer: second payment not authorized"),
            "the denied call's reason must be in the resumed run state"
        );

        // Turn 2: the model wraps up; the run completes from the resumed state.
        let (_, _, turn2) = expect_call_model(&mut resumed);
        assert_eq!(turn2, 2);
        expect_continue(
            resumed
                .model_response(text_turn("done"))
                .expect("model_response 2"),
        );
        let response = expect_done(&mut resumed);
        assert_eq!(response.output, "done");
    }

    #[test]
    fn abandon_pending_model_call_reissues_call_model_without_burning_budget() {
        let mut run = AgentRun::new("hello");

        // No model call pending yet: a no-op.
        assert!(!run.abandon_pending_model_call());

        let (_, _, turn) = expect_call_model(&mut run);
        assert_eq!(turn, 1);

        // The provider call failed: recover instead of wedging in
        // AwaitingModel forever.
        assert!(run.abandon_pending_model_call());

        // next_step re-issues the same call; max_turns is 1 and the failed
        // attempt was refunded, so this must not be a MaxTurnsError.
        let (prompt, _, turn) = expect_call_model(&mut run);
        assert_eq!(prompt, Message::user("hello"));
        assert_eq!(turn, 1);

        expect_continue(run.model_response(text_turn("hi")).expect("model_response"));
        assert_eq!(expect_done(&mut run).output, "hi");
        assert_eq!(run.completion_calls().len(), 1);
    }
}
