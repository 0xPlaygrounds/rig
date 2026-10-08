//! Serializable, sans-I/O state machine for model turns, tool recovery, and history.
//!
//! Drivers act on [`AgentRunStep`] and supply responses or tool results before
//! advancing. A run owns no models, tools, memory backends, or hooks; drivers
//! provide those services and lifecycle policy.
//!
//! ```
//! use rig_agent::run::{AgentRun, AgentRunStep};
//! let mut run = AgentRun::new("What is 2+2?").max_turns(3);
//! assert!(matches!(run.next_step()?, AgentRunStep::CallModel { turn: 1, .. }));
//! # Ok::<(), rig_agent::run::PromptError>(())
//! ```

mod calls;
mod committed;
pub mod output;
pub mod patch;
pub mod prepare;
pub mod spec;
pub use spec::UnhandledInvalidToolCall;
pub mod transcript;

mod batch;
use batch::{ToolBatch, ToolSlot};

pub use calls::{ExecCall, MalformedCall, PendingToolCall, ToolAnswer};
pub use committed::{CommittedItem, project};
pub use output::OutputMode;
pub use patch::RequestPatch;
pub use prepare::{PrepareError, PreparedRequest, prepare_request};
pub use spec::RunSpec;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use rig_core::completion::{CompletionResponse, FinishReason, ToolDefinition};
use rig_core::error::ProviderError;

use rig_core::message::{AssistantContent, AssistantMessage, ToolCall, ToolName, UserContent};
use rig_core::transcript::tool_result_output;

use calls::AnswerKind;

use rig_core::completion::{Message, ResponseIdentity, Usage};
pub mod policy;
pub mod response;
pub mod streamed;

pub use policy::{
    InvalidToolCallAction, InvalidToolCallContext, InvalidToolCallReason, RetryRequest, TurnPolicy,
};
pub use response::{CanonicalHistory, CompletionCall, MemoryAppend, PromptError, PromptResponse};
use rig_core::completion::message::turn_failure;
use rig_core::json_utils;
use rig_core::structured_output;
use transcript::{
    TranscriptError, assistant_message, assistant_text_from_choice, assistant_turn,
    build_full_history, build_history_for_request, invalid_tool_retry_user_message,
    is_empty_assistant_turn, tool_result_message, validate_canonical,
};

pub use streamed::{
    PartialStreamedTurn, StreamedInvalidToolCall, StreamedResolution, StreamedTurn,
    StreamedTurnAssembler, StreamedTurnEvent,
};

enum ValidatedInvalidToolCallAction {
    Retry { feedback: String },
    Repair { tool_name: String },
    Skip { reason: String },
}

/// Required driver action to advance an [`AgentRun`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentRunStep {
    /// Send a completion request to the model and feed the result back via
    /// [`AgentRun::model_response`]. Emitted again for the same attempt
    /// after a restart; see [`AgentRun::next_step`] for what a re-issue
    /// requires of the driver.
    CallModel {
        /// The prompt message for this turn (the latest message in the run).
        prompt: Message,
        /// The chat history preceding `prompt`: the caller-provided input
        /// history followed by messages accumulated by earlier turns.
        history: Vec<Message>,
        /// One-based index of this model call within the run.
        turn: usize,
    },
    /// Execute these tool calls and feed each answer back via
    /// [`AgentRun::answer`] (or several at once via [`AgentRun::answer_all`]).
    CallTools {
        /// The current assistant turn's unanswered tool calls, in emission order.
        calls: Vec<PendingToolCall>,
    },
    /// The run is complete.
    Done(PromptResponse),
}

/// A completed model turn fed back to [`AgentRun::model_response`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTurn {
    /// The turn's origin, stop and provider message, without content.
    pub head: AssistantMessage,
    /// Provider-assigned response-scoped ID, when available.
    pub response_id: Option<String>,
    /// The provider's transport request id for this attempt, when reported.
    pub provider_request_id: Option<String>,
    /// The assistant content returned by the model.
    pub choice: Vec<AssistantContent>,
    /// Token usage reported by the provider for this completion request.
    pub usage: Usage,
    /// The tool policy this attempt was prepared with.
    pub policy: TurnPolicy,
    /// Provider-reported terminal reason for this attempt, when available.
    pub finish_reason: Option<FinishReason>,
    /// This attempt's decoded provider response, recorded on its [`CompletionCall`].
    pub raw: serde_json::Value,
}

impl ModelTurn {
    /// Convert a response using the same attempt's prepared policy and the
    /// response's normalized finish reason. `prepared` must describe this call.
    pub fn from_response(resp: &CompletionResponse, prepared: &prepare::PreparedRequest) -> Self {
        Self::from_policy(resp, prepared.policy.clone())
    }

    /// [`from_response`](Self::from_response) for a driver that carries the
    /// turn's [`TurnPolicy`] by value instead of the whole prepared request.
    /// `policy` must be the [`PreparedRequest::policy`](prepare::PreparedRequest::policy)
    /// of this attempt: it alone carries the turn's tool choice to the Skip
    /// gate and the invalid-call context.
    pub fn from_policy(resp: &CompletionResponse, policy: TurnPolicy) -> Self {
        Self::new(
            resp.head(),
            resp.choice.clone(),
            resp.usage,
            policy,
            resp.raw.clone(),
        )
        .with_identity(
            resp.response_id().map(str::to_owned),
            resp.provider_request_id.clone(),
        )
        .with_finish_reason(resp.finish_reason())
    }

    /// Create a model turn from response parts, the policy the turn was
    /// prepared with, and the provider's own response `raw` (see
    /// [`Self::raw`]).
    pub fn new(
        head: AssistantMessage,
        choice: Vec<AssistantContent>,
        usage: Usage,
        policy: TurnPolicy,
        raw: serde_json::Value,
    ) -> Self {
        Self {
            head,
            response_id: None,
            provider_request_id: None,
            choice,
            usage,
            policy,
            finish_reason: None,
            raw,
        }
    }

    /// Attach the remaining response identity metadata this attempt reported.
    pub fn with_identity(
        mut self,
        response_id: Option<String>,
        provider_request_id: Option<String>,
    ) -> Self {
        self.response_id = response_id;
        self.provider_request_id = provider_request_id;
        self
    }

    /// Attach the terminal finish reason this attempt reported.
    pub fn with_finish_reason(mut self, finish_reason: Option<FinishReason>) -> Self {
        self.finish_reason = finish_reason;
        self
    }
}

/// Driver action after ingesting a model turn or resolving an invalid call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelTurnOutcome {
    /// The turn was accepted. Unless `response_hook_suppressed` is set, the
    /// driver should run its completion-response hook now, then call
    /// [`AgentRun::next_step`].
    ///
    /// `response_hook_suppressed` is set when invalid tool-call recovery
    /// (repair or skip) modified the turn, matching the agent loop's behavior
    /// of not firing the completion's outcome hook (`on_outcome`) for
    /// recovered turns.
    Continue {
        /// Whether the driver should suppress the completion's outcome hook.
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
    head: AssistantMessage,
    /// The unmodified model output, used for diagnostic histories and retry
    /// messages (repairs are never reflected in those).
    original_choice: Vec<AssistantContent>,
    /// Working copy of the assistant content; repairs rename tool calls here.
    items: Vec<AssistantContent>,
    /// Index of the next item to validate.
    next_index: usize,
    policy: TurnPolicy,
    /// Synthetic results keyed by content position so repeated call IDs cannot
    /// assign a skipped result to a different call.
    skipped: BTreeMap<usize, UserContent>,
    recovered: bool,
    any_skipped: bool,
    has_tool_calls: bool,
}

/// The invalid tool call resolution is currently parked on, if any: the item
/// at `next_index` when it is a tool call outside the allowed set.
fn pending_invalid_call(resolving: &ResolvingState) -> Option<&ToolCall> {
    match resolving.items.get(resolving.next_index) {
        Some(AssistantContent::ToolCall(tool_call))
            if !resolving.policy.allows(tool_call.function.name.as_str()) =>
        {
            Some(tool_call)
        }
        _ => None,
    }
}

fn has_tool_calls(items: &[AssistantContent]) -> bool {
    items
        .iter()
        .any(|item| matches!(item, AssistantContent::ToolCall(_)))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TurnState {
    head: AssistantMessage,
    items: Vec<AssistantContent>,
    has_tool_calls: bool,
    /// Keyed by position in `items` (see `ResolvingState::skipped`).
    skipped: BTreeMap<usize, UserContent>,
    /// Kept until the turn's calls are answered, so an invalid-call context
    /// raised in the tool step reports the policy the turn was judged by.
    policy: TurnPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RunState {
    /// Ready to emit [`AgentRunStep::CallModel`]. `rollback_owed` is set by a
    /// streamed rollback whose completion call is still to be recorded; see
    /// [`AgentRun::record_streamed_completion_call`].
    PreparingRequest { rollback_owed: bool },
    /// Waiting for [`AgentRun::model_response`] or a streamed turn;
    /// `recorded` once the streamed attempt's completion call is recorded.
    AwaitingModel { recorded: bool },
    /// Scanning the model turn's tool calls for validity; may be waiting for
    /// [`AgentRun::resolve_invalid_tool_call`].
    ResolvingToolCalls(ResolvingState),
    /// The turn was accepted; ready to emit [`AgentRunStep::CallTools`] or
    /// [`AgentRunStep::Done`].
    AwaitingAdvance(TurnState),
    /// Waiting for results for these tool calls. Carrying the calls and the
    /// answers so far keeps a serialized run self-contained: a resumed
    /// process re-obtains the unanswered calls from [`AgentRun::next_step`].
    ExecutingTools(ToolBatch, TurnPolicy),
    /// Terminal: the run completed successfully.
    Done(PromptResponse),
    /// Terminal: the run returned an error.
    Failed,
}

impl RunState {
    const READY: Self = Self::PreparingRequest {
        rollback_owed: false,
    };
}

/// The sans-IO agent loop state machine. See the [module docs](self) for the
/// driving protocol.
/// The persisted envelope is versioned: [`RUN_FORMAT`] is written on every
/// serialization and checked on every deserialization, and an unknown key
/// is refused. State written by another format is refused by name rather
/// than loaded with defaults filled in.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentRun {
    /// The envelope format ([`RUN_FORMAT`]).
    #[serde(deserialize_with = "run_format")]
    format: u32,
    max_turns: usize,
    max_invalid_tool_call_retries: usize,
    /// See [`RunSpec::max_consecutive_malformed_tool_calls`].
    max_consecutive_malformed_tool_calls: Option<usize>,
    /// See [`RunSpec::unhandled_invalid_tool_call`].
    unhandled_invalid_tool_call: UnhandledInvalidToolCall,
    /// Synthetic output-tool name, pinned by the first turn whose policy
    /// names one. Its first call finalizes with arguments as output instead
    /// of executing tools.
    output_tool_name: Option<String>,
    /// Schema whose top-level required fields are checked before finalization.
    output_schema: Option<serde_json::Value>,
    /// Budget for re-prompting the model in Tool output mode when it finalizes
    /// without calling the output tool, or calls it with arguments missing
    /// required fields. Exhausting it finalizes best-effort.
    max_output_retries: usize,
    output_retries: usize,
    chat_history: Option<Vec<Message>>,
    /// Append-only: see [`project`].
    #[serde(rename = "new_messages")]
    history: committed::CommittedLog,
    current_turn: usize,
    usage: Usage,
    completion_calls: Vec<CompletionCall>,
    completion_call_index: usize,
    invalid_tool_call_retries: usize,
    /// Consecutive tool steps that answered at least one call whose
    /// arguments are not a JSON object.
    malformed_tool_call_retries: usize,
    /// The model behind the run's preceding issued completion attempt, as
    /// the driver advances it immediately before the attempt is issued
    /// (a stop or a preparation failure leaves it unchanged; a provider
    /// error still counts). Persisted so a resumed run's model-selection
    /// hook sees the model the run last asked, as a fresh run's would,
    /// rather than a run that has asked none.
    previous_model: Option<rig_core::completion::ModelRef>,
    /// The tool definitions the driver advertised to the model for a turn,
    /// recorded with [`AgentRun::advertise_tools`]. Protocol data, so a second
    /// driver (or a resumed run) can re-pair tool calls with what was offered.
    turn_tools: Option<TurnTools>,
    /// Append-only host records, stored verbatim and excluded from provider requests.
    entries: Vec<RunEntry>,
    state: RunState,
}

/// The [`AgentRun`] envelope format this crate writes and reads.
pub const RUN_FORMAT: u32 = 4;

/// Deserialize the envelope's `format`, refusing any other than
/// [`RUN_FORMAT`] by name so a run persisted by another rig is never loaded
/// with defaults filled in.
fn run_format<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<u32, D::Error> {
    let format = u32::deserialize(deserializer)?;
    if format == RUN_FORMAT {
        Ok(format)
    } else {
        Err(serde::de::Error::custom(format!(
            "resume refused: the run is format {format}, this rig reads format {RUN_FORMAT}"
        )))
    }
}

/// Host record stored verbatim in serialized run state and excluded from provider
/// requests. The driver reconstructs hook state from entries on resume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunEntry {
    /// Hook-chosen namespace (e.g. `"approval"`, `"retry_budget"`).
    /// Unregistered and unvalidated: the kind string is the whole contract
    /// and the collision boundary, so hooks should pick specific names.
    pub kind: String,
    /// Model-call index at append time: zero before the first call, then one-based.
    /// Hosts may place timestamps in [`value`](Self::value).
    pub turn: usize,
    /// The appended value, verbatim JSON. `Value::Null` marker entries are
    /// legitimate.
    pub value: serde_json::Value,
}

/// Tool definitions advertised for one model call, recorded by the driver before
/// execution. Recording definitions does not bind or execute tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnTools {
    /// One-based model-call index the definitions were advertised for
    /// (the `turn` of the matching [`AgentRunStep::CallModel`]).
    pub turn: usize,
    /// The definitions sent with that request, in advertised order.
    pub definitions: Vec<ToolDefinition>,
}

impl TurnTools {
    /// Whether a tool of this name was advertised.
    pub fn contains(&self, tool_name: &str) -> bool {
        self.definitions.iter().any(|d| d.name == tool_name)
    }
}

impl AgentRun {
    /// Create a run for one prompt with no input history, a one-model-call
    /// budget, and no invalid tool-call retries.
    pub fn new(prompt: impl Into<Message>) -> Self {
        Self {
            format: RUN_FORMAT,
            max_turns: 1,
            max_invalid_tool_call_retries: 0,
            max_consecutive_malformed_tool_calls: None,
            unhandled_invalid_tool_call: UnhandledInvalidToolCall::Fail,
            output_tool_name: None,
            output_schema: None,
            max_output_retries: 0,
            output_retries: 0,
            chat_history: None,
            history: committed::CommittedLog::new(prompt.into()),
            current_turn: 0,
            usage: Usage::default(),
            completion_calls: Vec::new(),
            completion_call_index: 0,
            invalid_tool_call_retries: 0,
            malformed_tool_call_retries: 0,
            previous_model: None,
            turn_tools: None,
            entries: Vec::new(),
            state: RunState::READY,
        }
    }

    /// The prompt this run will send on its first model call, while the run
    /// has not yet started. `None` once the first
    /// [`AgentRunStep::CallModel`] has been emitted (the prompt is then run
    /// history, no longer pending input).
    pub fn initial_prompt(&self) -> Option<&Message> {
        (self.current_turn == 0 && matches!(self.state, RunState::PreparingRequest { .. }))
            .then(|| self.history.last())
            .flatten()
    }

    /// Replace the pending prompt before the run starts.
    ///
    /// This is the run-start steering point: a driver's pre-run hook may
    /// rewrite the user prompt here, before any model call. Valid only while
    /// [`initial_prompt`](Self::initial_prompt) is `Some`; once the first
    /// [`AgentRunStep::CallModel`] has been emitted the prompt is committed
    /// and rewriting returns [`PromptError::Cancelled`].
    pub(crate) fn rewrite_initial_prompt(
        &mut self,
        prompt: impl Into<Message>,
    ) -> Result<(), PromptError> {
        let started =
            self.current_turn != 0 || !matches!(self.state, RunState::PreparingRequest { .. });
        if let (false, Some(slot)) = (started, self.history.unstarted_prompt()) {
            *slot = prompt.into();
            return Ok(());
        }
        Err(self.cancel_error("the initial prompt can only be rewritten before the run starts"))
    }

    /// Append one host record without interpreting or validating its contents.
    pub fn append_entry(&mut self, entry: RunEntry) {
        self.entries.push(entry);
    }

    /// Every appended [`RunEntry`], in append order.
    pub fn entries(&self) -> &[RunEntry] {
        &self.entries
    }

    /// The entries of one `kind`, in append order.
    pub fn entries_of<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a RunEntry> {
        self.entries.iter().filter(move |entry| entry.kind == kind)
    }

    /// The most recently appended entry of `kind`, if present.
    pub fn last_entry_of(&self, kind: &str) -> Option<&RunEntry> {
        self.entries.iter().rev().find(|entry| entry.kind == kind)
    }

    /// Record the tool definitions advertised to the model for `turn` (the
    /// `turn` of the [`AgentRunStep::CallModel`] being served). Replaces any
    /// earlier record; the run keeps only the latest turn's advertisement.
    pub fn advertise_tools(&mut self, turn: usize, definitions: Vec<ToolDefinition>) {
        self.turn_tools = Some(TurnTools { turn, definitions });
    }

    /// The tools advertised for the most recent model call, if the driver
    /// recorded them. A [`CallTools`](AgentRunStep::CallTools) step whose
    /// calls name tools outside this set means the driver and the run have
    /// desynchronized (a registry that changed under a resumed run, say); a
    /// driver that wants that guard checks here before dispatching.
    pub fn advertised_tools(&self) -> Option<&TurnTools> {
        self.turn_tools.as_ref()
    }

    /// Set the input chat history preceding the prompt.
    pub fn with_history(mut self, history: Vec<Message>) -> Self {
        self.chat_history = Some(history);
        self
    }

    /// Set the total model-call budget, including the initial call and every
    /// retry or continuation. A budget of zero emits no model calls. Exceeding
    /// the budget makes [`AgentRun::next_step`] return
    /// [`PromptError::MaxTurns`].
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Set the output schema and retry budget for missing output-tool calls,
    /// output-tool arguments that are not a JSON object, or required fields.
    /// Validation checks only top-level required field presence; exhausting
    /// either the output or model-call budget finalizes best-effort, except
    /// that arguments that are not a JSON object fail the run.
    pub fn with_output_validation(
        mut self,
        output_schema: Option<serde_json::Value>,
        max_output_retries: usize,
    ) -> Self {
        self.output_schema = output_schema;
        self.max_output_retries = max_output_retries;
        self
    }

    /// The input chat history this run was created with, empty when none was
    /// set. This is the history preceding the initial prompt, not the run's
    /// accumulated messages.
    pub fn input_chat_history(&self) -> &[Message] {
        self.chat_history.as_deref().unwrap_or_default()
    }

    /// Whether the run may re-prompt for valid Tool-mode output: both the
    /// output-retry budget and the total model-call budget must remain.
    /// Otherwise, finalize best-effort rather than surface a max-turns error.
    fn can_reprompt_for_output(&self) -> bool {
        self.output_retries < self.max_output_retries && self.current_turn < self.max_turns
    }

    /// Roll the run back to re-prompt for valid output. The caller must
    /// have already appended the assistant turn and the corrective feedback
    /// message to the history. Consumes one output-retry, then emits the retry
    /// [`AgentRunStep::CallModel`].
    fn reprompt_for_output(&mut self) -> Result<AgentRunStep, PromptError> {
        self.output_retries += 1;
        self.state = RunState::READY;
        self.next_step()
    }

    /// Set the retry budget for [`InvalidToolCallAction::Retry`]
    /// resolutions. Invalid tool-call retries also consume the total model-call
    /// budget.
    pub fn max_invalid_tool_call_retries(mut self, retries: usize) -> Self {
        self.max_invalid_tool_call_retries = retries;
        self
    }

    /// Set how many consecutive turns may answer a tool call whose arguments
    /// are not a JSON object before the run fails. `None`, the default, sets
    /// no limit. See [`RunSpec::max_consecutive_malformed_tool_calls`].
    pub fn max_consecutive_malformed_tool_calls(mut self, limit: impl Into<Option<usize>>) -> Self {
        self.max_consecutive_malformed_tool_calls = limit.into();
        self
    }

    /// Set what the run does with an invalid tool call no hook resolves. See
    /// [`UnhandledInvalidToolCall`].
    pub fn with_unhandled_invalid_tool_call(mut self, policy: UnhandledInvalidToolCall) -> Self {
        self.unhandled_invalid_tool_call = policy;
        self
    }

    /// Pin the synthetic output-tool name for Tool output mode up front.
    ///
    /// This only seeds [`output_tool_name`](Self::output_tool_name), which the
    /// driver passes to [`prepare_request`] as `committed_output_tool` for
    /// later turns. It does not by itself make a call to the tool finalize the
    /// run: a turn intercepts the call only when its [`TurnPolicy`] names the
    /// tool, as `prepared.policy` does when `prepare_request` resolved Tool
    /// output mode, or a hand-built `TurnPolicy::new(.., Some(name))`. Under a
    /// policy that does not name it, a call to the tool is an invalid
    /// (unknown) tool call.
    pub fn with_output_tool_name(mut self, name: impl Into<String>) -> Self {
        self.output_tool_name = Some(name.into());
        self
    }

    /// The synthetic output-tool name pinned for this run, if any: set up
    /// front, or by the first model turn whose policy names one. The driver
    /// passes this back when preparing later turns so Tool output mode stays
    /// pinned even if the per-turn tool set changes.
    pub fn output_tool_name(&self) -> Option<&str> {
        self.output_tool_name.as_deref()
    }

    /// Pin the output tool `policy` names unless one is already pinned. The
    /// first name wins and is never unpinned, so a tool set that shifts
    /// mid-run cannot flip the output mode.
    fn pin_output_tool(&mut self, policy: &TurnPolicy) {
        if self.output_tool_name.is_none() {
            self.output_tool_name = policy.output_tool().map(str::to_owned);
        }
    }

    /// The policy for an invalid tool call no hook resolves.
    pub(crate) fn unhandled_invalid_tool_call(&self) -> UnhandledInvalidToolCall {
        self.unhandled_invalid_tool_call
    }

    /// Aggregated token usage across all completed model calls so far.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Number of model calls emitted so far (including retries).
    pub fn turn(&self) -> usize {
        self.current_turn
    }

    /// Model of the preceding issued attempt, as recorded by the driver.
    pub fn previous_model(&self) -> Option<&rig_core::completion::ModelRef> {
        self.previous_model.as_ref()
    }

    /// Record the model an attempt is about to be issued to; the driver
    /// calls this immediately before the model turn is driven.
    pub fn set_previous_model(&mut self, model: rig_core::completion::ModelRef) {
        self.previous_model = Some(model);
    }

    /// Every completion call the run made so far, in order.
    pub fn completion_calls(&self) -> &[CompletionCall] {
        &self.completion_calls
    }

    /// Messages accumulated by this run (the prompt plus all assistant turns
    /// and tool results), excluding the input history. While tools run, the
    /// last turn's calls are still unanswered here: this is the view to
    /// display, and [`Self::full_history`] the one to resume from.
    pub fn messages(&self) -> &[Message] {
        &self.history
    }

    /// Where a driver starts to [`project`] [`messages`](Self::messages): the
    /// end, or the message holding a pending [`AgentRunStep::CallTools`]
    /// batch, so a resumed run announces the calls whose results it streams.
    pub fn projection_start(&self) -> usize {
        let pending = matches!(self.state, RunState::ExecutingTools(..));
        let holder = |message: &Message| pending && matches!(message, Message::Assistant(_));
        (self.history.iter().rposition(holder)).unwrap_or(self.history.len())
    }

    /// Canonical content for the accepted model turn awaiting advancement.
    pub fn accepted_turn_choice(&self) -> Option<Vec<AssistantContent>> {
        let RunState::AwaitingAdvance(turn) = &self.state else {
            return None;
        };

        if turn.items.is_empty() {
            return None;
        }
        Some(turn.items.clone())
    }

    /// Replace accepted content before it enters history. Returns a cancellation
    /// error without an accepted turn or if either turn contains tool calls.
    pub fn replace_accepted_turn_choice(
        &mut self,
        choice: Vec<AssistantContent>,
    ) -> Result<(), PromptError> {
        let replacement_has_tool_calls = choice
            .iter()
            .any(|item| matches!(item, AssistantContent::ToolCall(_)));
        let parked_has_tool_calls = match &self.state {
            RunState::AwaitingAdvance(turn) => turn.has_tool_calls,
            _ => {
                return Err(self.protocol_violation(
                    "replace_accepted_turn_choice called without an accepted turn awaiting advancement",
                ));
            }
        };
        if parked_has_tool_calls || replacement_has_tool_calls {
            return Err(self.cancel_error(
                "a completion outcome replacement does not support tool-bearing model turns; patch or deny the tool dispatches instead",
            ));
        }
        if let RunState::AwaitingAdvance(turn) = &mut self.state {
            turn.items = choice;
        }
        Ok(())
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
            return Err(self.fail(
                "model-turn retry does not support tool-bearing model turns; use tool-call hooks instead",
            ));
        }

        match request {
            RetryRequest::Repeat => {}
            RetryRequest::Feedback(feedback) => {
                // Feedback may retry an empty answer, but empty assistant messages
                // must not enter provider history.
                self.history.commit(assistant_turn(turn.head, turn.items));
                self.history.commit([Message::user(feedback)]);
            }
        }

        self.state = RunState::READY;
        Ok(())
    }

    /// The full conversation: input history followed by [`Self::messages`],
    /// with a tool batch the host has not finished answering closed: its
    /// results so far, then the other calls closed by
    /// [`close_pending`](rig_core::transcript::close_pending) (a pre-resolved
    /// result stands), so it resumes as a canonical transcript.
    pub fn full_history(&self) -> Vec<Message> {
        let mut history = build_full_history(self.chat_history.as_deref(), self.history.to_vec());
        history.extend(self.closure());
        history
    }

    /// The user message closing the batch the host was told to run, while
    /// it has not answered every call: the results so far, then the rest
    /// closed.
    fn closure(&self) -> Option<Message> {
        let RunState::ExecutingTools(batch, _) = &self.state else {
            return None;
        };
        Some(batch.interrupt())
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

    /// [`Self::full_history`] closed for cancellation by
    /// [`CanonicalHistory::close`]: mid tool batch, the results so far
    /// follow it with the rest closed, and every call still unanswered in
    /// the unchecked input history is closed as well.
    pub fn canonical_history(&self) -> CanonicalHistory {
        CanonicalHistory::close(self.full_history())
    }

    /// Build the cancellation error a driver should return when one of its
    /// hooks terminates the run, carrying [`Self::canonical_history`].
    pub fn cancel_error(&self, reason: impl Into<String>) -> PromptError {
        PromptError::cancelled(self.canonical_history(), reason)
    }

    /// [`Self::cancel_error`], then fail the run, its history ending as the
    /// error's does.
    fn fail(&mut self, reason: impl Into<String>) -> PromptError {
        let error = self.cancel_error(reason);
        let closure = self.closure();
        self.history.commit(closure);
        self.state = RunState::Failed;
        error
    }

    /// The [`AgentRunStep::CallModel`] for the current turn.
    fn call_model_step(&self) -> Result<AgentRunStep, PromptError> {
        let Some((prompt, history)) = self.history.split_last() else {
            return Err(self.cancel_error("prompt loop lost its pending prompt"));
        };
        Ok(AgentRunStep::CallModel {
            prompt: prompt.clone(),
            history: build_history_for_request(self.chat_history.as_deref(), history),
            turn: self.current_turn,
        })
    }

    /// The invalid tool call currently awaiting
    /// [`AgentRun::resolve_invalid_tool_call`], if any. Useful to re-derive
    /// the resolution context after deserializing a suspended run.
    pub fn pending_invalid_tool_call(&self) -> Option<InvalidToolCallContext> {
        let RunState::ResolvingToolCalls(resolving) = &self.state else {
            return None;
        };
        let tool_call = pending_invalid_call(resolving)?;

        Some(resolving.policy.invalid_call_context(
            tool_call,
            Some(tool_call.function.arguments_value().to_string()),
            self.diagnostic_history(resolving),
            false,
            resolving.policy.name_reason(tool_call),
        ))
    }

    /// Advance the machine and return the next action for the driver.
    ///
    /// Idempotent while awaiting a model response or tool results: the step
    /// is emitted again, a model call without consuming a turn and a tool
    /// step with only its unanswered calls. A persisted run resumes from here
    /// in any non-terminal state except a pending invalid tool-call
    /// resolution.
    ///
    /// A re-issued [`AgentRunStep::CallModel`] is the same attempt: it carries
    /// the same `turn`, and the run cannot tell its response from a late one of
    /// the attempt it replaces. Before re-issuing, the driver drops (or
    /// drains) the previous model call or stream, so nothing of it reaches
    /// [`Self::model_response`], [`Self::record_streamed_completion_call`] or
    /// [`Self::streamed_turn`]. Re-issues are not budgeted: they consume no
    /// turn of [`Self::max_turns`], and an attempt that never reached
    /// `model_response` or `record_streamed_completion_call` is not in
    /// [`Self::usage`]. A driver that re-issues in a loop bounds it itself.
    ///
    /// # Errors
    /// - [`PromptError::MaxTurns`] when the total model-call budget is exhausted.
    /// - [`PromptError::Cancelled`] when the machine is driven out of
    ///   protocol (for example, calling this while an invalid tool-call
    ///   resolution is pending).
    pub fn next_step(&mut self) -> Result<AgentRunStep, PromptError> {
        match std::mem::replace(&mut self.state, RunState::Failed) {
            // Re-issuing a pending attempt consumes no turn; the interrupted
            // attempt stays recorded and billed if it was.
            state @ (RunState::PreparingRequest { .. } | RunState::AwaitingModel { .. }) => {
                if let RunState::PreparingRequest { .. } = state {
                    if self.current_turn >= self.max_turns
                        && let Some(prompt) = self.history.last()
                    {
                        return Err(PromptError::MaxTurns {
                            max_turns: self.max_turns,
                            chat_history: self.full_history(),
                            prompt: prompt.clone(),
                        });
                    }
                    self.current_turn += 1;
                }
                let step = self.call_model_step()?;
                self.state = RunState::AwaitingModel { recorded: false };
                Ok(step)
            }
            RunState::AwaitingAdvance(turn_state) => {
                let TurnState {
                    head,
                    items,
                    has_tool_calls,
                    mut skipped,
                    policy,
                } = turn_state;
                // A failed turn runs no tool and finalizes no output; reasoning
                // alone is not an answer. A failed turn's calls never run, and
                // the turn stays in the run's messages for display, as pi
                // keeps it.
                let finish = self
                    .completion_calls
                    .last()
                    .and_then(|call| call.finish_reason.as_ref());
                if let Some(message) = turn_failure(&items, head.stop.as_ref(), finish) {
                    if has_tool_calls {
                        self.history.commit(assistant_turn(head, items));
                    }
                    return Err(ProviderError::Response(message).into());
                }

                // The first output-tool call is the answer, not executable work;
                // sibling calls must not run after finalization.
                if has_tool_calls
                    && let Some(output_tool_name) = policy.output_tool().map(str::to_owned)
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
                    let args = tool_call.function.arguments_value();
                    let tool_call_id = tool_call.id.clone();
                    let output = json_utils::serialize_json_value(&args);

                    // Arguments that are not a JSON object are no answer:
                    // the model is told why while the budget lasts.
                    if let Some(raw) = &tool_call.function.invalid_arguments {
                        if self.can_reprompt_for_output() {
                            self.history.commit(assistant_message(head, items.clone()));
                            let feedback = rig_core::transcript::invalid_arguments_feedback(
                                &output_tool_name,
                                raw,
                            );
                            if let Some(user_message) =
                                invalid_tool_retry_user_message(&items, &tool_call_id, &feedback)
                            {
                                self.history.commit([user_message]);
                            }
                            return self.reprompt_for_output();
                        }
                        return Err(ProviderError::Response(format!(
                            "the output tool `{output_tool_name}` was called with arguments \
                             that are not a JSON object: {raw}"
                        ))
                        .into());
                    }

                    let missing = self
                        .output_schema
                        .as_ref()
                        .map(|schema| structured_output::missing_required_fields(schema, &args))
                        .unwrap_or_default();
                    if !missing.is_empty() && self.can_reprompt_for_output() {
                        self.history.commit(assistant_message(head, items.clone()));
                        let feedback =
                            structured_output::reprompt_missing_fields(&output_tool_name, &missing);
                        if let Some(user_message) =
                            invalid_tool_retry_user_message(&items, &tool_call_id, &feedback)
                        {
                            self.history.commit([user_message]);
                        }
                        return self.reprompt_for_output();
                    }

                    // Store output as text without tool calls so resumed history
                    // cannot contain unanswered calls.
                    let mut final_items: Vec<AssistantContent> = items
                        .iter()
                        .filter(|item| !matches!(item, AssistantContent::ToolCall(_)))
                        .cloned()
                        .collect();
                    final_items.push(AssistantContent::text(output.clone()));
                    self.history
                        .commit(assistant_message(head, final_items.clone()));

                    let content = response::finalize_output_tool_choice(&items, &output)
                        .unwrap_or_else(|| vec![AssistantContent::text(output)]);
                    return Ok(self.finish(content, output_tool_calls));
                }

                let slots: Option<Vec<ToolSlot>> = has_tool_calls.then(|| {
                    items
                        .iter()
                        .enumerate()
                        .filter_map(|(index, item)| match item {
                            AssistantContent::ToolCall(tool_call) => Some(ToolSlot::classify(
                                tool_call.clone(),
                                skipped.remove(&index),
                            )),
                            _ => None,
                        })
                        .collect()
                });
                // A turn past the malformed-call limit never enters history.
                if let Some(slots) = &slots {
                    self.count_malformed_tool_calls(slots)?;
                }
                // Empty turns may succeed but cannot form provider history entries.
                self.history.commit(assistant_turn(head, items.clone()));

                if let Some(slots) = slots {
                    // Output retries are budgeted per finalization attempt, not per run.
                    self.output_retries = 0;
                    self.state = RunState::ExecutingTools(ToolBatch { slots }, policy);
                    // A batch answered in full (a whole-turn skip) is
                    // committed at once and the run moves on to the model.
                    self.answer_all([])?;
                    self.next_step()
                } else {
                    // Accept schema-compatible JSON text without requiring a tool call;
                    // other nonempty answers may consume an output retry.
                    if let Some(output_tool_name) = policy.output_tool()
                        && !is_empty_assistant_turn(&items)
                        && self.can_reprompt_for_output()
                        && !structured_output::text_satisfies_schema(
                            self.output_schema.as_ref(),
                            &assistant_text_from_choice(&items),
                        )
                    {
                        self.history.commit([Message::user(
                            structured_output::reprompt_text_answer(output_tool_name),
                        )]);
                        return self.reprompt_for_output();
                    }

                    Ok(self.finish(items, 0))
                }
            }
            RunState::ExecutingTools(batch, policy) => {
                let calls = batch.pending(self.current_turn);
                self.state = RunState::ExecutingTools(batch, policy);
                Ok(AgentRunStep::CallTools { calls })
            }
            RunState::Done(response) => {
                let step = AgentRunStep::Done(response.clone());
                self.state = RunState::Done(response);
                Ok(step)
            }
            state @ RunState::ResolvingToolCalls(_) => {
                self.state = state;
                Err(self.protocol_violation(
                    "next_step called while an invalid tool-call resolution is pending; answer it via resolve_invalid_tool_call first",
                ))
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
        if self.pending_attempt("model_response")? {
            return Err(self.protocol_violation(
                "model_response called after record_streamed_completion_call for the same turn; feed streamed turns via streamed_turn",
            ));
        }

        self.record_completion_call(
            turn.usage,
            ResponseIdentity {
                response_id: turn.response_id,
                provider_request_id: turn.provider_request_id,
            },
            turn.finish_reason,
            turn.raw,
        );
        self.pin_output_tool(&turn.policy);

        let items: Vec<AssistantContent> = turn.choice.clone();
        let has_tool_calls = has_tool_calls(&items);

        self.state = RunState::ResolvingToolCalls(ResolvingState {
            head: turn.head,
            original_choice: turn.choice,
            items,
            next_index: 0,
            policy: turn.policy,
            skipped: BTreeMap::new(),
            recovered: false,
            any_skipped: false,
            has_tool_calls,
        });

        self.advance_resolution()
    }

    /// Latest call's reason when [`FinishReason::truncated_output`] identifies
    /// truncation. Unknown provider reasons do not imply truncation.
    fn record_completion_call(
        &mut self,
        usage: Usage,
        identity: ResponseIdentity,
        finish_reason: Option<FinishReason>,
        raw: serde_json::Value,
    ) -> CompletionCall {
        let call = CompletionCall::new(self.completion_call_index, usage, raw)
            .with_identity(identity)
            .with_finish_reason(finish_reason);
        self.completion_call_index += 1;
        self.completion_calls.push(call.clone());
        self.usage += usage;
        call
    }

    /// Build the run's final [`PromptResponse`], park it in
    /// [`RunState::Done`], and return the `Done` step. Shared by the
    /// output-tool and plain-text finalization paths in `next_step`.
    fn finish(&mut self, content: Vec<AssistantContent>, output_tool_calls: usize) -> AgentRunStep {
        let response = PromptResponse::from_content(content, self.usage)
            .with_messages(self.history.to_vec())
            .with_completion_calls(self.completion_calls.clone())
            .with_output_tool_calls(output_tool_calls);
        self.state = RunState::Done(response.clone());
        AgentRunStep::Done(response)
    }

    /// Park an accepted model turn in [`RunState::AwaitingAdvance`]. Both the
    /// non-streamed (`advance_resolution`) and streamed (`streamed_turn`)
    /// ingestion paths converge here, differing only in the `skipped` map.
    fn finalize_turn(
        &mut self,
        head: AssistantMessage,
        items: Vec<AssistantContent>,
        has_tool_calls: bool,
        skipped: BTreeMap<usize, UserContent>,
        policy: TurnPolicy,
    ) {
        self.state = RunState::AwaitingAdvance(TurnState {
            head,
            items,
            has_tool_calls,
            skipped,
            policy,
        });
    }

    /// Count a tool step that answers a call whose arguments are not a JSON
    /// object, or reset the count when every executed call parsed. Past
    /// [`RunSpec::max_consecutive_malformed_tool_calls`] consecutive steps, when
    /// set, fails the run naming the tool and the parse error.
    fn count_malformed_tool_calls(&mut self, slots: &[ToolSlot]) -> Result<(), PromptError> {
        let malformed = slots.iter().find_map(|slot| match slot {
            ToolSlot::Malformed(tool_call) => tool_call
                .function
                .invalid_arguments
                .as_deref()
                .map(|raw| (tool_call, raw)),
            _ => None,
        });
        let Some((tool_call, raw)) = malformed else {
            self.malformed_tool_call_retries = 0;
            return Ok(());
        };
        self.malformed_tool_call_retries = self.malformed_tool_call_retries.saturating_add(1);
        let Some(limit) = self
            .max_consecutive_malformed_tool_calls
            .filter(|limit| self.malformed_tool_call_retries > *limit)
        else {
            return Ok(());
        };
        let error = policy::arguments_parse_error(raw);
        Err(ProviderError::Response(format!(
            "tool `{}` was called with arguments that are not a JSON object on {} consecutive \
             turns, more than the {} retries allowed: {error}",
            tool_call.function.name, self.malformed_tool_call_retries, limit,
        ))
        .into())
    }

    /// The invalid-call context for a malformed pending call, for the driver
    /// to offer its invalid-call hook before answering the call. Outside a
    /// [`AgentRunStep::CallTools`] step the context lists no tools.
    pub fn malformed_context(
        &self,
        call: &MalformedCall,
        is_streaming: bool,
    ) -> InvalidToolCallContext {
        let policy = match &self.state {
            RunState::ExecutingTools(_, policy) => std::borrow::Cow::Borrowed(policy),
            _ => std::borrow::Cow::Owned(TurnPolicy::default()),
        };
        let raw = call.raw_arguments();
        policy.invalid_call_context(
            call.tool_call(),
            Some(raw.to_owned()),
            // Ends at the turn carrying the call: the hook decides how it is
            // answered, so the closure of pending calls is not part of it.
            build_full_history(self.chat_history.as_deref(), self.history.to_vec()),
            is_streaming,
            InvalidToolCallReason::malformed_arguments(raw),
        )
    }

    /// Validate the recovery policy shared by buffered and streamed turns.
    /// Medium-specific rollback, repair, and skip effects remain at the call
    /// sites; rejection, retry budgeting, and tool-choice checks live here so
    /// the two surfaces cannot drift.
    fn validate_invalid_tool_call_action(
        &mut self,
        action: InvalidToolCallAction,
        tool_call: &ToolCall,
        policy: &TurnPolicy,
        history: &[Message],
    ) -> Result<ValidatedInvalidToolCallAction, PromptError> {
        let unknown = |name: String| policy.unknown_call(name, history.to_vec());
        let rejected = || unknown(tool_call.function.name.to_string());
        let result = match action {
            InvalidToolCallAction::Fail => Err(rejected()),
            InvalidToolCallAction::Retry { feedback } => {
                if self.invalid_tool_call_retries >= self.max_invalid_tool_call_retries {
                    Err(rejected())
                } else {
                    self.invalid_tool_call_retries += 1;
                    Ok(ValidatedInvalidToolCallAction::Retry { feedback })
                }
            }
            InvalidToolCallAction::Repair { tool_name } => {
                if policy.allows(&tool_name) {
                    Ok(ValidatedInvalidToolCallAction::Repair { tool_name })
                } else {
                    Err(unknown(tool_name))
                }
            }
            InvalidToolCallAction::Stop { reason } => Err(self.cancel_error(reason)),
            InvalidToolCallAction::Skip { reason } => {
                if policy.forbids_calls() {
                    Err(rejected())
                } else {
                    Ok(ValidatedInvalidToolCallAction::Skip { reason })
                }
            }
        };

        if result.is_err() {
            self.state = RunState::Failed;
        }
        result
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
    ///   `PromptError::cancelled` and the supplied reason.
    /// - [`InvalidToolCallAction::Skip`] records a synthetic tool result
    ///   and suppresses execution of every tool call in the turn. Rejected
    ///   under a turn whose policy [`forbids_calls`](TurnPolicy::forbids_calls).
    pub fn resolve_invalid_tool_call(
        &mut self,
        action: InvalidToolCallAction,
    ) -> Result<ModelTurnOutcome, PromptError> {
        let mut resolving = self.take_resolving(
            "resolve_invalid_tool_call called without a pending invalid tool call",
        )?;
        let Some(tool_call) = pending_invalid_call(&resolving).cloned() else {
            self.state = RunState::ResolvingToolCalls(resolving);
            return Err(self.protocol_violation(
                "resolve_invalid_tool_call called without a pending invalid tool call",
            ));
        };

        let diagnostic_history = self.diagnostic_history(&resolving);
        let action = self.validate_invalid_tool_call_action(
            action,
            &tool_call,
            &resolving.policy,
            &diagnostic_history,
        )?;

        match action {
            ValidatedInvalidToolCallAction::Retry { feedback } => {
                self.history.commit(assistant_message(
                    resolving.head.clone(),
                    resolving.original_choice.clone(),
                ));
                let Some(user_message) = invalid_tool_retry_user_message(
                    &resolving.original_choice,
                    &tool_call.id,
                    &feedback,
                ) else {
                    return Err(self.fail("invalid tool call retry produced no retry messages"));
                };
                self.history.commit([user_message]);
                self.state = RunState::READY;
                Ok(ModelTurnOutcome::TurnRetried)
            }
            ValidatedInvalidToolCallAction::Repair { tool_name } => {
                if let Some(AssistantContent::ToolCall(tool_call)) =
                    resolving.items.get_mut(resolving.next_index)
                    && let Ok(tool_name) = ToolName::new(tool_name)
                {
                    tool_call.function.name = tool_name;
                }
                resolving.recovered = true;
                self.state = RunState::ResolvingToolCalls(resolving);
                self.advance_resolution()
            }
            ValidatedInvalidToolCallAction::Skip { reason } => {
                let user_content = tool_result_message(
                    tool_call.id.clone(),
                    tool_call.function.name.clone(),
                    reason,
                );
                // Keyed by the call's position: `next_index` is exactly the
                // invalid call's slot in `items`, and later mutations only
                // touch indices at or after it, so earlier keys stay stable.
                resolving.skipped.insert(resolving.next_index, user_content);
                resolving.recovered = true;
                resolving.any_skipped = true;
                resolving.next_index += 1;
                self.state = RunState::ResolvingToolCalls(resolving);
                self.advance_resolution()
            }
        }
    }

    /// Apply the configured unhandled-call policy after hooks decline resolution.
    /// `Fail` reports the invalid call; `Ignore` removes it without suppressing
    /// response hooks or sibling calls. Errors without a pending invalid call.
    pub fn resolve_unhandled_invalid_tool_call(&mut self) -> Result<ModelTurnOutcome, PromptError> {
        match self.unhandled_invalid_tool_call {
            UnhandledInvalidToolCall::Fail => {
                self.resolve_invalid_tool_call(InvalidToolCallAction::fail())
            }
            UnhandledInvalidToolCall::Ignore => self.ignore_invalid_tool_call(),
        }
    }

    /// Resolve the pending invalid tool call by ignoring it: the turn
    /// proceeds as if the model had not made the call.
    pub fn ignore_invalid_tool_call(&mut self) -> Result<ModelTurnOutcome, PromptError> {
        let mut resolving = self.take_resolving(
            "ignore_invalid_tool_call called without a pending invalid tool call",
        )?;

        if pending_invalid_call(&resolving).is_none() {
            self.state = RunState::ResolvingToolCalls(resolving);
            return Err(self.protocol_violation(
                "ignore_invalid_tool_call called without a pending invalid tool call",
            ));
        }

        resolving.items.remove(resolving.next_index);
        resolving.has_tool_calls = has_tool_calls(&resolving.items);
        self.state = RunState::ResolvingToolCalls(resolving);
        self.advance_resolution()
    }

    /// Apply one [`ToolAnswer`] for the pending [`AgentRunStep::CallTools`].
    /// See [`answer_all`](Self::answer_all).
    pub fn answer(&mut self, answer: ToolAnswer) -> Result<(), PromptError> {
        self.answer_all([answer])
    }

    /// Apply answers for the pending [`AgentRunStep::CallTools`], all or none.
    /// The run builds each result; once every call is answered they are
    /// committed as one user message in call order.
    ///
    /// # Errors
    /// A protocol violation, storing nothing, for an answer whose call is not
    /// open. A malformed call's `Stop`, `Fail` or `Repair` ends the run; it
    /// is applied after the other answers, which the run keeps, and the
    /// first such answer in call order wins.
    pub fn answer_all(
        &mut self,
        answers: impl IntoIterator<Item = ToolAnswer>,
    ) -> Result<(), PromptError> {
        let RunState::ExecutingTools(batch, policy) = &self.state else {
            return Err(self.protocol_violation("answer called without a pending CallTools step"));
        };
        // The turn stays put until the batch is answered, so it names the batch.
        let (mut batch, policy, turn) = (batch.clone(), policy.clone(), self.current_turn);
        let mut answers: Vec<ToolAnswer> = answers.into_iter().collect();
        answers.sort_by_key(ToolAnswer::ends_run);
        for ToolAnswer {
            turn: issued,
            index,
            call,
            kind,
        } in answers
        {
            let content = match (batch.slots.get(index), kind) {
                _ if issued != turn => {
                    return Err(self.protocol_violation(&format!(
                        "answer for tool call id `{call}` of model turn {issued}, but the pending calls are from turn {turn}"
                    )));
                }
                (Some(ToolSlot::Execute(tool_call)), AnswerKind::Executed(result))
                    if tool_call.id == call =>
                {
                    tool_result_output(call, tool_call.function.name.clone(), &result)
                }
                (Some(ToolSlot::Malformed(tool_call)), AnswerKind::Malformed(action))
                    if tool_call.id == call =>
                {
                    match malformed_answer(tool_call, action) {
                        Ok(content) => content,
                        Err(end) => {
                            // The run ends with the answers so far kept and the
                            // rest of its batch closed, as the error reports it.
                            self.state = RunState::ExecutingTools(batch, policy);
                            return Err(match end {
                                Ok(reason) => self.fail(reason),
                                Err(error) => {
                                    let closure = self.closure();
                                    self.history.commit(closure);
                                    self.state = RunState::Failed;
                                    error
                                }
                            });
                        }
                    }
                }
                (Some(ToolSlot::Answered(_)), _) => {
                    return Err(self.protocol_violation(&format!(
                        "answer for tool call id `{call}` that is already answered"
                    )));
                }
                _ => {
                    return Err(self.protocol_violation(&format!(
                        "answer for tool call id `{call}` does not match the pending call at position {index}"
                    )));
                }
            };
            if let Some(slot) = batch.slots.get_mut(index) {
                *slot = ToolSlot::Answered(content);
            }
        }
        self.state = match batch.complete() {
            Ok(results) => {
                self.history.commit([results]);
                RunState::READY
            }
            Err(batch) => RunState::ExecutingTools(batch, policy),
        };
        Ok(())
    }

    /// Take the resolving state out of `self.state`, leaving `Failed` behind;
    /// callers restore it on their rejection paths so an out-of-protocol call
    /// does not corrupt a drivable run.
    fn take_resolving(&mut self, violation: &str) -> Result<ResolvingState, PromptError> {
        match std::mem::replace(&mut self.state, RunState::Failed) {
            RunState::ResolvingToolCalls(resolving) => Ok(resolving),
            other => {
                self.state = other;
                Err(self.protocol_violation(violation))
            }
        }
    }

    /// Scan forward for the next invalid tool call; finish the turn when the
    /// scan completes.
    fn advance_resolution(&mut self) -> Result<ModelTurnOutcome, PromptError> {
        let mut resolving =
            self.take_resolving("internal: advance_resolution outside of tool-call resolution")?;
        while let Some(item) = resolving.items.get(resolving.next_index) {
            match item {
                AssistantContent::ToolCall(tool_call)
                    if !resolving.policy.allows(tool_call.function.name.as_str()) =>
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
            head,
            items,
            mut skipped,
            recovered,
            any_skipped,
            has_tool_calls,
            policy,
            ..
        } = resolving;

        // When any tool call was skipped, none of the turn's tool calls
        // execute: peers get a synthetic "not executed" result.
        if any_skipped {
            for (index, item) in items.iter().enumerate() {
                if let AssistantContent::ToolCall(tool_call) = item {
                    skipped
                        .entry(index)
                        .or_insert_with(|| rig_core::transcript::not_executed(tool_call));
                }
            }
        }

        self.finalize_turn(head, items, has_tool_calls, skipped, policy);
        Ok(ModelTurnOutcome::Continue {
            response_hook_suppressed: recovered,
        })
    }

    /// Record a streamed attempt's terminal metadata and aggregate its usage.
    /// All arguments must come from that attempt's final event; do not record a
    /// stream that ended without one. All-`None` counters mean unreported usage.
    ///
    /// Allowed once per model attempt: while awaiting its response, or after
    /// its streamed rollback before the next model step. Other states and
    /// duplicate records return a cancellation error. Abandoned streams must
    /// still be drained for usage.
    pub fn record_streamed_completion_call(
        &mut self,
        usage: Usage,
        identity: ResponseIdentity,
        finish_reason: Option<FinishReason>,
        raw: serde_json::Value,
    ) -> Result<CompletionCall, PromptError> {
        let recordable = match &mut self.state {
            RunState::AwaitingModel { recorded } => !std::mem::replace(recorded, true),
            RunState::PreparingRequest { rollback_owed } => std::mem::take(rollback_owed),
            _ => false,
        };
        if !recordable {
            return Err(self.protocol_violation(
                "record_streamed_completion_call called without a pending or rolled-back model attempt, or twice for one",
            ));
        }

        Ok(self.record_completion_call(usage, identity, finish_reason, raw))
    }

    /// The recovery-hook context for an invalid tool call surfaced
    /// mid-stream by a [`streamed::StreamedTurnAssembler`].
    pub fn streamed_invalid_tool_call_context(
        &self,
        partial: &PartialStreamedTurn,
        invalid: &StreamedInvalidToolCall,
    ) -> InvalidToolCallContext {
        invalid.policy.invalid_call_context(
            &invalid.tool_call,
            invalid.args.clone(),
            self.streamed_diagnostic_history(partial, Some(invalid.tool_call.clone())),
            true,
            invalid.policy.name_reason(&invalid.tool_call),
        )
    }

    /// Resolve an invalid tool call surfaced mid-stream.
    ///
    /// Uses buffered-turn recovery policy with rollback messages from the partial
    /// turn. Retry and skip abandon the stream; repair changes only the name and
    /// rejects malformed arguments. Errors without a pending model step.
    pub fn resolve_streamed_invalid_tool_call(
        &mut self,
        partial: &PartialStreamedTurn,
        invalid: &StreamedInvalidToolCall,
        action: InvalidToolCallAction,
    ) -> Result<StreamedResolution, PromptError> {
        let recorded = self.pending_attempt("resolve_streamed_invalid_tool_call")?;

        // A streamed turn abandoned here never reaches `streamed_turn`, so
        // it pins like a buffered turn that is later retried.
        self.pin_output_tool(&invalid.policy);
        let diagnostic_history =
            self.streamed_diagnostic_history(partial, Some(invalid.tool_call.clone()));
        let action = self.validate_invalid_tool_call_action(
            action,
            &invalid.tool_call,
            &invalid.policy,
            &diagnostic_history,
        )?;

        match action {
            ValidatedInvalidToolCallAction::Retry { feedback } => self.abandon_streamed_turn(
                partial,
                invalid,
                feedback,
                !recorded,
                "invalid tool call retry produced no retry messages",
            ),
            ValidatedInvalidToolCallAction::Repair { tool_name } => {
                Ok(StreamedResolution::Repaired { tool_name })
            }
            ValidatedInvalidToolCallAction::Skip { reason } => self.abandon_streamed_turn(
                partial,
                invalid,
                reason,
                !recorded,
                "invalid tool call skip produced no recovery messages",
            ),
        }
    }

    /// Resolve a streamed call as ignored. The driver must apply the returned
    /// resolution to the assembler so the call does not enter the turn.
    /// Errors without a pending model step.
    pub fn ignore_streamed_invalid_tool_call(&mut self) -> Result<StreamedResolution, PromptError> {
        self.pending_attempt("ignore_streamed_invalid_tool_call")?;
        Ok(StreamedResolution::Ignored)
    }

    /// Shared rollback for the streamed Retry and Skip resolutions: push the
    /// partial turn's rollback messages and abandon the turn, or fail the run
    /// when the partial turn yields no rollback messages. `rollback_owed`
    /// says the abandoned attempt's completion call is still to be recorded.
    fn abandon_streamed_turn(
        &mut self,
        partial: &PartialStreamedTurn,
        invalid: &StreamedInvalidToolCall,
        feedback: String,
        rollback_owed: bool,
        no_messages_reason: &str,
    ) -> Result<StreamedResolution, PromptError> {
        let Some((assistant_message, user_message)) =
            partial.rollback_messages(invalid.tool_call.clone(), feedback)
        else {
            return Err(self.fail(no_messages_reason));
        };
        self.history.commit([assistant_message, user_message]);
        self.state = RunState::PreparingRequest { rollback_owed };
        Ok(StreamedResolution::TurnAbandoned)
    }

    /// Feed the assembled streamed turn for the pending
    /// [`AgentRunStep::CallModel`].
    ///
    /// Rejects remaining disallowed tool names without further recovery. Requires
    /// a pending model step and exactly one prior call to
    /// [`AgentRun::record_streamed_completion_call`] for this attempt; otherwise
    /// returns a protocol error. Accepted turns await [`AgentRun::next_step`].
    pub fn streamed_turn(&mut self, turn: StreamedTurn) -> Result<(), PromptError> {
        if !self.pending_attempt("streamed_turn")? {
            return Err(self.protocol_violation(
                "streamed_turn called before record_streamed_completion_call recorded the turn's completion call",
            ));
        }

        let has_tool_calls = has_tool_calls(&turn.choice);

        for item in &turn.choice {
            let AssistantContent::ToolCall(tool_call) = item else {
                continue;
            };
            if !turn.policy.allows(tool_call.function.name.as_str()) {
                let mut diagnostic_messages = self.history.to_vec();
                diagnostic_messages.extend(assistant_turn(turn.head.clone(), turn.choice.clone()));
                let diagnostic_history =
                    build_full_history(self.chat_history.as_deref(), diagnostic_messages);
                self.state = RunState::Failed;
                return Err(turn
                    .policy
                    .unknown_call(tool_call.function.name.to_string(), diagnostic_history));
            }
        }

        self.pin_output_tool(&turn.policy);
        self.finalize_turn(
            turn.head,
            turn.choice,
            has_tool_calls,
            BTreeMap::new(),
            turn.policy,
        );
        Ok(())
    }

    /// Diagnostic history for a streamed turn: the run's messages plus the
    /// partial assistant turn under inspection.
    fn streamed_diagnostic_history(
        &self,
        partial: &PartialStreamedTurn,
        current_tool_call: Option<ToolCall>,
    ) -> Vec<Message> {
        let mut messages = self.history.to_vec();
        if let Some(assistant) = partial.assistant_message(current_tool_call) {
            messages.push(assistant);
        }
        build_full_history(self.chat_history.as_deref(), messages)
    }

    /// History used for invalid tool-call diagnostics: the run's messages plus
    /// the unmodified assistant turn under inspection.
    fn diagnostic_history(&self, resolving: &ResolvingState) -> Vec<Message> {
        let mut diagnostic_messages = self.history.to_vec();
        diagnostic_messages.extend(assistant_message(
            resolving.head.clone(),
            resolving.original_choice.clone(),
        ));
        build_full_history(self.chat_history.as_deref(), diagnostic_messages)
    }

    /// Whether the pending model attempt's streamed completion call is
    /// recorded; a protocol violation naming `op` when no attempt is pending.
    fn pending_attempt(&self, op: &str) -> Result<bool, PromptError> {
        match self.state {
            RunState::AwaitingModel { recorded } => Ok(recorded),
            _ => {
                Err(self
                    .protocol_violation(&format!("{op} called without a pending CallModel step")))
            }
        }
    }

    fn protocol_violation(&self, reason: &str) -> PromptError {
        self.cancel_error(format!("agent run driver protocol violation: {reason}"))
    }
}

impl AgentRun {
    /// Build a run from a [`RunSpec`], a prompt and an optional prior history.
    ///
    /// Applies the spec's budget, invalid-call retries and output validation;
    /// everything else in the spec is request-shaping the driver reads when it
    /// prepares each model call. `spec.tool_choice` reaches invalid-call
    /// validation only through the turn's [`TurnPolicy`]: pass
    /// `prepared.policy` from [`prepare_request`] to
    /// [`ModelTurn::from_policy`] or [`ModelTurn::new`].
    pub fn from_spec(
        spec: &RunSpec,
        prompt: impl Into<Message>,
        history: Option<Vec<Message>>,
    ) -> Self {
        let mut run = AgentRun::new(prompt)
            .max_turns(spec.effective_max_turns())
            .max_invalid_tool_call_retries(spec.max_invalid_tool_call_retries)
            .with_unhandled_invalid_tool_call(spec.unhandled_invalid_tool_call)
            .with_output_validation(spec.output_schema.clone(), RunSpec::DEFAULT_OUTPUT_RETRIES);
        run.max_consecutive_malformed_tool_calls = spec.max_consecutive_malformed_tool_calls;
        if let Some(history) = history {
            run = run.with_history(history);
        }
        if let Some(name) = spec.output_tool_name.clone() {
            run = run.with_output_tool_name(name);
        }
        run
    }
}

impl AgentRun {
    /// [`with_history`](AgentRun::with_history), rejecting a history that is
    /// not a canonical transcript. Use this when the history comes from
    /// outside the protocol (a memory backend, a resumed run from another
    /// process); `with_history` stays unchecked for callers that built the
    /// history themselves.
    pub fn with_validated_history(self, history: Vec<Message>) -> Result<Self, TranscriptError> {
        validate_canonical(&history)?;
        Ok(self.with_history(history))
    }
}

// Compile-time contract: run state is plain, owned data a host can keep in
// shared state (worker pools, ECS components) on native targets.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<AgentRun>();
    assert_send_sync_static::<AgentRunStep>();
    assert_send_sync_static::<PendingToolCall>();
    assert_send_sync_static::<ModelTurn>();
    assert_send_sync_static::<StreamedTurn>();
    assert_send_sync_static::<StreamedTurnAssembler>();
    assert_send_sync_static::<PromptResponse>();
    assert_send_sync_static::<CompletionCall>();
    assert_send_sync_static::<PromptError>();
    assert_send_sync_static::<TurnTools>();
    assert_send_sync_static::<RunEntry>();
};

/// The result content for a malformed call answered with `action`, or how
/// the answer ends the run: `Ok` with a stop reason, or `Err` with the failure.
fn malformed_answer(
    tool_call: &ToolCall,
    action: Option<InvalidToolCallAction>,
) -> Result<UserContent, Result<String, PromptError>> {
    let (id, name) = (tool_call.id.clone(), tool_call.function.name.clone());
    let raw = tool_call
        .function
        .invalid_arguments
        .as_deref()
        .unwrap_or_default();
    let feedback = match action {
        None => rig_core::transcript::invalid_arguments_feedback(name.as_str(), raw),
        Some(InvalidToolCallAction::Retry { feedback }) => feedback,
        Some(InvalidToolCallAction::Skip { reason }) => {
            let skipped = rig_core::tool::ToolResult::skipped(reason);
            return Ok(tool_result_output(id, name, &skipped));
        }
        Some(InvalidToolCallAction::Stop { reason }) => return Err(Ok(reason)),
        Some(InvalidToolCallAction::Fail | InvalidToolCallAction::Repair { .. }) => {
            return Err(Err(ProviderError::Response(format!(
                "tool `{name}` was called with arguments that are not a JSON object: {}",
                policy::arguments_parse_error(raw)
            ))
            .into()));
        }
    };
    Ok(tool_result_message(id, name, feedback))
}

#[cfg(test)]
mod tests;
