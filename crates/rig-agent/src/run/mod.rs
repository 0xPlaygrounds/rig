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

pub mod output;
pub mod patch;
pub mod prepare;
pub mod spec;
pub use spec::UnhandledInvalidToolCall;
pub mod transcript;

pub use output::OutputMode;
pub use patch::RequestPatch;
pub use prepare::{PrepareError, PreparedRequest, prepare_request};
pub use spec::RunSpec;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use rig_core::completion::{CompletionResponse, FinishReason, ToolDefinition};
use rig_core::error::ProviderError;
use rig_core::streaming::BlockId;

use rig_core::message::{
    AssistantContent, ToolCall, ToolChoice, ToolResult, ToolResultContent, UserContent,
};

use rig_core::completion::{Message, ResponseIdentity, Usage};
pub mod policy;
pub mod response;
pub mod streamed;

pub use policy::{
    InvalidToolCallAction, InvalidToolCallContext, InvalidToolCallReason, RetryRequest,
};
pub use response::{CompletionCall, MemoryAppend, PromptError, PromptResponse};
use rig_core::completion::message::turn_delivered_no_answer;
use rig_core::json_utils;
use transcript::{
    TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER, TranscriptError, assistant_text_from_choice,
    build_full_history, build_history_for_request, invalid_tool_retry_user_message,
    is_empty_assistant_turn, tool_result_message, validate_canonical,
};

pub use streamed::{
    PartialStreamedTurn, StreamedInvalidToolCall, StreamedResolution, StreamedTurn,
    StreamedTurnAssembler, StreamedTurnEvent,
};

/// Build an unknown-tool error with advertised names and diagnostic history.
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
        chat_history,
    }
}

#[derive(Clone, Copy)]
struct InvalidToolCallDiagnostic<'a> {
    tool_call: &'a ToolCall,
    executable_tool_names: &'a BTreeSet<String>,
    allowed_tool_names: &'a BTreeSet<String>,
    history: &'a [Message],
    reason: &'a InvalidToolCallReason,
}

impl InvalidToolCallDiagnostic<'_> {
    fn unknown(&self, tool_name: String) -> PromptError {
        unknown_tool_call_error(
            tool_name,
            self.executable_tool_names.iter().cloned().collect(),
            self.allowed_tool_names.iter().cloned().collect(),
            self.history.to_vec(),
        )
    }

    /// Report the rejected call as an unknown tool or malformed-input response error.
    fn unknown_current(&self) -> PromptError {
        match self.reason {
            InvalidToolCallReason::UnknownTool => {
                self.unknown(self.tool_call.function.name.clone())
            }
            InvalidToolCallReason::MalformedArguments { error } => {
                PromptError::Report(malformed_tool_input_report(self.tool_call, error))
            }
        }
    }

    fn cancelled(&self, reason: String) -> PromptError {
        PromptError::prompt_cancelled(self.history.to_vec(), reason)
    }
}

/// Reconstruct a malformed-input response report from the diagnostic call.
fn malformed_tool_input_report(tool_call: &ToolCall, error: &str) -> rig_core::error::ErrorReport {
    rig_core::error::ErrorReport::new(
        rig_core::error::ErrorKind::Response,
        format!(
            "tool call `{}` arrived with malformed JSON input: {error}",
            tool_call.function.name
        ),
    )
}

enum ValidatedInvalidToolCallAction {
    Retry { feedback: String },
    Repair { tool_name: String },
    Skip { reason: String },
}

/// Required driver action to advance an [`AgentRun`].
#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub struct PendingToolCall {
    /// The tool call emitted by the model (with any repaired tool name applied).
    pub tool_call: ToolCall,
    /// Pre-resolved result for tool calls suppressed by invalid tool-call
    /// recovery. When set, the driver must return this content as the tool
    /// result without executing the tool or invoking tool hooks.
    pub preresolved_result: Option<UserContent>,
    /// Block ID shared by deltas, execution commit, and result. Buffered turns
    /// mint completion-local keys independently of durable `tool_call.id`.
    /// Persist this ID unchanged across resume.
    pub block_id: BlockId,
}

/// A completed model turn fed back to [`AgentRun::model_response`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTurn {
    /// Provider-assigned assistant message ID, when available.
    pub message_id: Option<String>,
    /// Provider-assigned response-scoped ID, when available.
    pub response_id: Option<String>,
    /// The provider's transport request id for this attempt, when reported.
    pub provider_request_id: Option<String>,
    /// The assistant content returned by the model.
    pub choice: Vec<AssistantContent>,
    /// Token usage reported by the provider for this completion request.
    pub usage: Usage,
    /// Executable Rig tools advertised to the provider for this turn.
    pub executable_tool_names: BTreeSet<String>,
    /// Tools allowed by the active [`ToolChoice`] for this turn.
    pub allowed_tool_names: BTreeSet<String>,
    /// Provider-reported terminal reason for this attempt, when available.
    pub finish_reason: Option<FinishReason>,
    /// This attempt's decoded provider response, recorded on its [`CompletionCall`].
    pub raw: serde_json::Value,
}

impl ModelTurn {
    /// Convert a response using the same attempt's prepared tool sets and the
    /// response's normalized finish reason. `prepared` must describe this call.
    pub fn from_response(resp: &CompletionResponse, prepared: &prepare::PreparedRequest) -> Self {
        Self::from_response_parts(
            resp,
            prepared.executable_tool_names.clone(),
            prepared.allowed_tool_names.clone(),
        )
    }

    /// [`from_response`](Self::from_response) for a driver that carries the
    /// per-turn tool-name sets by value instead of the whole prepared
    /// request. The sets must originate from the prepared request of the same
    /// attempt.
    pub fn from_response_parts(
        resp: &CompletionResponse,
        executable_tool_names: BTreeSet<String>,
        allowed_tool_names: BTreeSet<String>,
    ) -> Self {
        Self::new(
            resp.message_id.clone(),
            resp.choice.clone(),
            resp.usage,
            executable_tool_names,
            allowed_tool_names,
            resp.raw.clone(),
        )
        .with_identity(resp.response_id.clone(), resp.provider_request_id.clone())
        .with_finish_reason(resp.finish_reason())
    }

    /// Create a model turn from response parts, the tool names advertised
    /// for the turn, and the provider's own response `raw` (see
    /// [`Self::raw`]).
    pub fn new(
        message_id: Option<String>,
        choice: Vec<AssistantContent>,
        usage: Usage,
        executable_tool_names: BTreeSet<String>,
        allowed_tool_names: BTreeSet<String>,
        raw: serde_json::Value,
    ) -> Self {
        Self {
            message_id,
            response_id: None,
            provider_request_id: None,
            choice,
            usage,
            executable_tool_names,
            allowed_tool_names,
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
    message_id: Option<String>,
    /// The unmodified model output, used for diagnostic histories and retry
    /// messages (repairs are never reflected in those).
    original_choice: Vec<AssistantContent>,
    /// Working copy of the assistant content; repairs rename tool calls here.
    items: Vec<AssistantContent>,
    /// Index of the next item to validate.
    next_index: usize,
    executable_tool_names: BTreeSet<String>,
    allowed_tool_names: BTreeSet<String>,
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
            if !resolving
                .allowed_tool_names
                .contains(&tool_call.function.name) =>
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
    message_id: Option<String>,
    items: Vec<AssistantContent>,
    has_tool_calls: bool,
    /// Keyed by position in `items` (see `ResolvingState::skipped`).
    skipped: BTreeMap<usize, UserContent>,
    /// `(tool_call_id, block_id)` pairs for streamed turns, in
    /// emission order; empty for non-streamed turns.
    block_ids: Vec<(rig_core::message::ToolCallId, BlockId)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RunState {
    /// Ready to emit [`AgentRunStep::CallModel`].
    PreparingRequest,
    /// Waiting for [`AgentRun::model_response`].
    AwaitingModel,
    /// Scanning the model turn's tool calls for validity; may be waiting for
    /// [`AgentRun::resolve_invalid_tool_call`].
    ResolvingToolCalls(ResolvingState),
    /// The turn was accepted; ready to emit [`AgentRunStep::CallTools`] or
    /// [`AgentRunStep::Done`].
    AwaitingAdvance(TurnState),
    /// Waiting for [`AgentRun::tool_results`] for these pending tool calls.
    /// Carrying the calls in the state keeps a serialized run self-contained:
    /// a resumed process re-obtains them from [`AgentRun::next_step`].
    ExecutingTools(Vec<PendingToolCall>),
    /// Terminal: the run completed successfully.
    Done(PromptResponse),
    /// Terminal: the run returned an error.
    Failed,
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
    /// See [`RunSpec::unhandled_invalid_tool_call`].
    unhandled_invalid_tool_call: UnhandledInvalidToolCall,
    tool_choice: Option<ToolChoice>,
    /// Synthetic output-tool name. Its first call finalizes with arguments as
    /// output instead of executing tools.
    output_tool_name: Option<String>,
    /// Schema whose top-level required fields are checked before finalization.
    output_schema: Option<serde_json::Value>,
    /// Budget for re-prompting the model in Tool output mode when it finalizes
    /// without calling the output tool, or calls it with arguments missing
    /// required fields. Exhausting it finalizes best-effort.
    max_output_retries: usize,
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
    rollback_pending: bool,
    /// Set once the current streamed model turn's completion call has been
    /// recorded, rejecting duplicate records; reset when the next
    /// [`AgentRunStep::CallModel`] is emitted.
    streamed_completion_call_recorded: bool,
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
pub const RUN_FORMAT: u32 = 1;

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
            unhandled_invalid_tool_call: UnhandledInvalidToolCall::Fail,
            tool_choice: None,
            output_tool_name: None,
            output_schema: None,
            max_output_retries: 0,
            output_retries: 0,
            chat_history: None,
            new_messages: vec![prompt.into()],
            current_turn: 0,
            usage: Usage::default(),
            completion_calls: Vec::new(),
            completion_call_index: 0,
            invalid_tool_call_retries: 0,
            rollback_pending: false,
            streamed_completion_call_recorded: false,
            previous_model: None,
            turn_tools: None,
            entries: Vec::new(),
            state: RunState::PreparingRequest,
        }
    }

    /// The prompt this run will send on its first model call, while the run
    /// has not yet started. `None` once the first
    /// [`AgentRunStep::CallModel`] has been emitted (the prompt is then run
    /// history, no longer pending input).
    pub fn initial_prompt(&self) -> Option<&Message> {
        (self.current_turn == 0 && matches!(self.state, RunState::PreparingRequest))
            .then(|| self.new_messages.last())
            .flatten()
    }

    /// Replace the pending prompt before the run starts.
    ///
    /// This is the run-start steering point: a driver's pre-run hook may
    /// rewrite the user prompt here, before any model call. Valid only while
    /// [`initial_prompt`](Self::initial_prompt) is `Some`; once the first
    /// [`AgentRunStep::CallModel`] has been emitted the prompt is committed
    /// and rewriting returns [`PromptError::PromptCancelled`].
    pub(crate) fn rewrite_initial_prompt(
        &mut self,
        prompt: impl Into<Message>,
    ) -> Result<(), PromptError> {
        let started = self.current_turn != 0 || !matches!(self.state, RunState::PreparingRequest);
        match self.new_messages.last_mut() {
            Some(slot) if !started => {
                *slot = prompt.into();
                Ok(())
            }
            _ => Err(PromptError::prompt_cancelled(
                self.full_history(),
                "the initial prompt can only be rewritten before the run starts",
            )),
        }
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
    /// [`PromptError::MaxTurnsError`].
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Set the output schema and retry budget for missing output-tool calls or
    /// required fields. Validation checks only top-level required field presence;
    /// exhausting either the output or model-call budget finalizes best-effort.
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

    /// Whether text parses as JSON with no missing top-level required fields.
    /// With no required fields, non-object JSON also passes.
    fn text_satisfies_output_schema(&self, text: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(text.trim())
            .ok()
            .is_some_and(|value| self.missing_required_output_fields(&value).is_empty())
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

    /// Set what the run does with an invalid tool call no hook resolves. See
    /// [`UnhandledInvalidToolCall`].
    pub fn with_unhandled_invalid_tool_call(mut self, policy: UnhandledInvalidToolCall) -> Self {
        self.unhandled_invalid_tool_call = policy;
        self
    }

    /// Set the tool choice active for this run. Used to reject
    /// [`InvalidToolCallAction::Skip`] resolutions under
    /// [`ToolChoice::None`] and reported in invalid tool-call contexts.
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Set the synthetic output-tool name for Tool output mode.
    /// When a model turn calls this tool, the run finalizes with the call's
    /// arguments (serialized JSON) as the response.
    pub fn with_output_tool_name(mut self, name: impl Into<String>) -> Self {
        self.output_tool_name = Some(name.into());
        self
    }

    /// Commit the output-tool name once the driver has resolved it from the
    /// prepared request inside the run loop, where the agent's tool set (and
    /// thus the resolved output mode) is known. Returns whether this call
    /// committed it: the name is pinned for the whole run, so the request
    /// the driver builds each turn stays consistent with the intercept (and
    /// a tool set that shifts mid-run cannot flip the mode); a later call
    /// with a different name is refused, not applied.
    #[must_use = "a refused commit means the run already pinned a name"]
    pub fn commit_output_tool_name(&mut self, name: impl Into<String>) -> bool {
        if self.output_tool_name.is_some() {
            return false;
        }
        self.output_tool_name = Some(name.into());
        true
    }

    /// The synthetic output-tool name committed for this run, if any. The driver
    /// passes this back when preparing later turns so Tool output mode stays
    /// pinned even if the per-turn tool set changes.
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
    /// and tool results), excluding the input history.
    pub fn messages(&self) -> &[Message] {
        &self.new_messages
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
            return Err(PromptError::prompt_cancelled(
                self.full_history(),
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
            return Err(PromptError::prompt_cancelled(
                self.full_history(),
                "model-turn retry does not support tool-bearing model turns; use tool-call hooks instead",
            ));
        }

        match request {
            RetryRequest::Repeat => {}
            RetryRequest::Feedback(feedback) => {
                // Feedback may retry an empty answer, but empty assistant messages
                // must not enter provider history.
                let content = turn.items;
                if !is_empty_assistant_turn(&content) {
                    self.new_messages.push(Message::Assistant {
                        id: turn.message_id,
                        content,
                    });
                }
                self.new_messages.push(Message::user(feedback));
            }
        }

        self.state = RunState::PreparingRequest;
        Ok(())
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
        let tool_call = pending_invalid_call(resolving)?;

        Some(InvalidToolCallContext {
            tool_name: tool_call.function.name.clone(),
            tool_call_id: Some(tool_call.id.clone()),
            // A buffered/unary diagnostic has no live stream block.
            // Correlation uses the typed call ID, including after resume.
            block_id: None,
            args: Some(json_utils::serialize_json_value(
                &tool_call.function.arguments,
            )),
            available_tools: resolving.executable_tool_names.iter().cloned().collect(),
            allowed_tools: resolving.allowed_tool_names.iter().cloned().collect(),
            tool_choice: self.tool_choice.clone(),
            chat_history: self.diagnostic_history(resolving),
            is_streaming: false,
            reason: InvalidToolCallReason::UnknownTool,
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
                        chat_history: self.full_history(),
                        prompt,
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
                    mut block_ids,
                } = turn_state;
                // The first output-tool call is the answer, not executable work;
                // sibling calls must not run after finalization.
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

                    let missing = self.missing_required_output_fields(&args);
                    if !missing.is_empty() && self.can_reprompt_for_output() {
                        self.new_messages.push(Message::Assistant {
                            id: message_id,
                            content: items.clone(),
                        });
                        let feedback = format!(
                            "The `{output_tool_name}` arguments were missing required field(s): \
                             {}. Call `{output_tool_name}` again with every required field.",
                            missing.join(", ")
                        );
                        if let Some(user_message) =
                            invalid_tool_retry_user_message(&items, &tool_call_id, &feedback)
                        {
                            self.new_messages.push(user_message);
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
                    self.new_messages.push(Message::Assistant {
                        id: message_id,
                        content: final_items.clone(),
                    });

                    return Ok(self.finish(output, final_items, output_tool_calls));
                }

                // Reasoning alone is not an answer. Reject answerless truncated turns
                // before committing history, but retain valid empty non-truncated turns.
                if turn_delivered_no_answer(&items)
                    && let Some(reason) = self.truncating_finish_reason()
                {
                    return Err(ProviderError::Response(reason.no_answer_message()).into());
                }

                // Empty turns may succeed but cannot form provider history entries.
                if !is_empty_assistant_turn(&items) {
                    self.new_messages.push(Message::Assistant {
                        id: message_id,
                        content: items.clone(),
                    });
                }

                if has_tool_calls {
                    // Output retries are budgeted per finalization attempt, not per run.
                    self.output_retries = 0;
                    // Allocate assembly keys independently of durable tool identities.
                    // Advance for every content position, matching buffered re-emission.
                    let mut synthetic_blocks = rig_core::streaming::SyntheticIds::tool();
                    let calls: Vec<PendingToolCall> = items
                        .iter()
                        .enumerate()
                        .filter_map(|(index, item)| {
                            let synthetic_block = synthetic_blocks.mint();
                            match item {
                                AssistantContent::ToolCall(tool_call) => {
                                    // Consume pairs positionally so duplicate
                                    // provider IDs within one turn stay
                                    // distinguishable.
                                    let block_id = block_ids
                                        .iter()
                                        .position(|(id, _)| tool_call.id == *id)
                                        .map_or_else(
                                            || synthetic_block,
                                            |pair| block_ids.remove(pair).1,
                                        );
                                    Some(PendingToolCall {
                                        tool_call: tool_call.clone(),
                                        preresolved_result: skipped.get(&index).cloned(),
                                        block_id,
                                    })
                                }
                                _ => None,
                            }
                        })
                        .collect();
                    self.state = RunState::ExecutingTools(calls.clone());
                    Ok(AgentRunStep::CallTools { calls })
                } else {
                    // Accept schema-compatible JSON text without requiring a tool call;
                    // other nonempty answers may consume an output retry.
                    if let Some(output_tool_name) = self.output_tool_name.clone()
                        && !is_empty_assistant_turn(&items)
                        && self.can_reprompt_for_output()
                        && !self.text_satisfies_output_schema(&assistant_text_from_choice(&items))
                    {
                        let feedback = format!(
                            "Provide your final answer by calling the `{output_tool_name}` tool \
                             with the structured result as its arguments, not as plain text."
                        );
                        self.new_messages.push(Message::user(feedback));
                        return self.reprompt_for_output();
                    }

                    Ok(self.finish(assistant_text_from_choice(&items), items, 0))
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
                let step = AgentRunStep::Done(response.clone());
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

        self.record_completion_call(
            turn.usage,
            ResponseIdentity {
                // The message id is also written into run history below.
                message_id: turn.message_id.clone(),
                response_id: turn.response_id,
                provider_request_id: turn.provider_request_id,
            },
            turn.finish_reason,
            turn.raw,
        );

        let items: Vec<AssistantContent> = turn.choice.clone();
        let has_tool_calls = has_tool_calls(&items);

        self.state = RunState::ResolvingToolCalls(ResolvingState {
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
        });

        self.advance_resolution()
    }

    /// Latest call's reason when [`FinishReason::truncated_output`] identifies
    /// truncation. Unknown provider reasons do not imply truncation.
    fn truncating_finish_reason(&self) -> Option<&FinishReason> {
        self.completion_calls
            .last()?
            .finish_reason
            .as_ref()
            .filter(|reason| reason.truncated_output())
    }

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
    fn finish(
        &mut self,
        output: String,
        content: Vec<AssistantContent>,
        output_tool_calls: usize,
    ) -> AgentRunStep {
        let response = PromptResponse::new(output, self.usage)
            .with_messages(self.new_messages.clone())
            .with_completion_calls(self.completion_calls.clone())
            .with_output_tool_calls(output_tool_calls)
            .with_content(content);
        self.state = RunState::Done(response.clone());
        AgentRunStep::Done(response)
    }

    /// Park an accepted model turn in [`RunState::AwaitingAdvance`]. Both the
    /// non-streamed (`advance_resolution`) and streamed (`streamed_turn`)
    /// ingestion paths converge here, differing only in the `skipped` map and
    /// the streamed `block_ids`.
    fn finalize_turn(
        &mut self,
        message_id: Option<String>,
        items: Vec<AssistantContent>,
        has_tool_calls: bool,
        skipped: BTreeMap<usize, UserContent>,
        block_ids: Vec<(rig_core::message::ToolCallId, BlockId)>,
    ) {
        self.state = RunState::AwaitingAdvance(TurnState {
            message_id,
            items,
            has_tool_calls,
            skipped,
            block_ids,
        });
    }

    /// Validate the recovery policy shared by buffered and streamed turns.
    /// Medium-specific rollback, repair, and skip effects remain at the call
    /// sites; rejection, retry budgeting, and tool-choice checks live here so
    /// the two surfaces cannot drift.
    fn validate_invalid_tool_call_action(
        &mut self,
        action: InvalidToolCallAction,
        diagnostic: InvalidToolCallDiagnostic<'_>,
    ) -> Result<ValidatedInvalidToolCallAction, PromptError> {
        let result = match action {
            InvalidToolCallAction::Fail => Err(diagnostic.unknown_current()),
            InvalidToolCallAction::Retry { feedback } => {
                if self.invalid_tool_call_retries >= self.max_invalid_tool_call_retries {
                    Err(diagnostic.unknown_current())
                } else {
                    self.invalid_tool_call_retries += 1;
                    Ok(ValidatedInvalidToolCallAction::Retry { feedback })
                }
            }
            InvalidToolCallAction::Repair { tool_name } => match diagnostic.reason {
                // Repair replaces a *name*; it cannot rewrite argument
                // bytes, so a repair of malformed input would dispatch a
                // tool with arguments the model never produced. Fail closed
                // with the same report `Fail` gives.
                InvalidToolCallReason::MalformedArguments { .. } => {
                    Err(diagnostic.unknown_current())
                }
                InvalidToolCallReason::UnknownTool => {
                    if diagnostic.allowed_tool_names.contains(&tool_name) {
                        Ok(ValidatedInvalidToolCallAction::Repair { tool_name })
                    } else {
                        Err(diagnostic.unknown(tool_name))
                    }
                }
            },
            InvalidToolCallAction::Stop { reason } => Err(diagnostic.cancelled(reason)),
            InvalidToolCallAction::Skip { reason } => {
                if matches!(self.tool_choice, Some(ToolChoice::None)) {
                    Err(diagnostic.unknown_current())
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
    ///   `PromptError::prompt_cancelled` and the supplied reason.
    /// - [`InvalidToolCallAction::Skip`] records a synthetic tool result
    ///   and suppresses execution of every tool call in the turn. Rejected
    ///   under [`ToolChoice::None`].
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
            InvalidToolCallDiagnostic {
                tool_call: &tool_call,
                executable_tool_names: &resolving.executable_tool_names,
                allowed_tool_names: &resolving.allowed_tool_names,
                reason: &InvalidToolCallReason::UnknownTool,
                history: &diagnostic_history,
            },
        )?;

        match action {
            ValidatedInvalidToolCallAction::Retry { feedback } => {
                self.new_messages.push(Message::Assistant {
                    id: resolving.message_id.clone(),
                    content: resolving.original_choice.clone(),
                });
                let Some(user_message) = invalid_tool_retry_user_message(
                    &resolving.original_choice,
                    &tool_call.id,
                    &feedback,
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
            ValidatedInvalidToolCallAction::Repair { tool_name } => {
                if let Some(AssistantContent::ToolCall(tool_call)) =
                    resolving.items.get_mut(resolving.next_index)
                {
                    tool_call.function.name = tool_name;
                }
                resolving.recovered = true;
                self.state = RunState::ResolvingToolCalls(resolving);
                self.advance_resolution()
            }
            ValidatedInvalidToolCallAction::Skip { reason } => {
                let user_content = UserContent::tool_result_for(
                    tool_call.id.clone(),
                    tool_call.provider.clone(),
                    tool_call.function.name.clone(),
                    vec![reason.into()],
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

    /// Feed the tool results for the pending [`AgentRunStep::CallTools`].
    ///
    /// Results may arrive in any order and are appended as one user message.
    /// Each must answer a pending call, with exactly one result per occurrence
    /// of its ID. Invalid or incomplete batches return a cancellation error.
    pub fn tool_results(&mut self, results: Vec<UserContent>) -> Result<(), PromptError> {
        let RunState::ExecutingTools(pending) = &self.state else {
            return Err(
                self.protocol_violation("tool_results called without a pending CallTools step")
            );
        };
        // Match results against pending calls by tool call ID as a multiset,
        // so duplicate provider IDs within one turn stay answerable.
        let mut unanswered: Vec<rig_core::message::ToolCallId> = pending
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
            let Some(index) = unanswered.iter().position(|id| tool_result.call == *id) else {
                return Err(self.protocol_violation(&format!(
                    "tool_results received a result for unknown or already-answered tool call id `{}`",
                    tool_result.call
                )));
            };
            unanswered.swap_remove(index);
        }
        if !unanswered.is_empty() {
            return Err(self.protocol_violation(&format!(
                "tool_results left pending tool call id(s) unanswered: {unanswered:?}"
            )));
        }

        self.new_messages.push(Message::User { content: results });
        self.state = RunState::PreparingRequest;
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
        } = resolving;

        // When any tool call was skipped, none of the turn's tool calls
        // execute: peers get a synthetic "not executed" result.
        if any_skipped {
            for (index, item) in items.iter().enumerate() {
                if let AssistantContent::ToolCall(tool_call) = item {
                    skipped.entry(index).or_insert_with(|| {
                        tool_result_message(
                            tool_call.id.clone(),
                            tool_call.provider.clone(),
                            tool_call.function.name.clone(),
                            TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER.to_string(),
                        )
                    });
                }
            }
        }

        self.finalize_turn(message_id, items, has_tool_calls, skipped, Vec::new());
        Ok(ModelTurnOutcome::Continue {
            response_hook_suppressed: recovered,
        })
    }

    /// Record a streamed attempt's terminal metadata and aggregate its usage.
    /// All arguments must come from that attempt's final event; do not record a
    /// stream that ended without one. All-`None` counters mean unreported usage.
    ///
    /// Allowed once while awaiting a model response, or after a streamed rollback
    /// before the next model step. Other states and duplicate records return a
    /// cancellation error. Abandoned streams must still be drained for usage.
    pub fn record_streamed_completion_call(
        &mut self,
        usage: Usage,
        identity: ResponseIdentity,
        finish_reason: Option<FinishReason>,
        raw: serde_json::Value,
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

        Ok(self.record_completion_call(usage, identity, finish_reason, raw))
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
            block_id: Some(invalid.block_id.clone()),
            args: invalid.args.clone(),
            available_tools: invalid.executable_tool_names.iter().cloned().collect(),
            allowed_tools: invalid.allowed_tool_names.iter().cloned().collect(),
            tool_choice: self.tool_choice.clone(),
            chat_history: self
                .streamed_diagnostic_history(partial, Some(invalid.tool_call.clone())),
            is_streaming: true,
            reason: invalid.reason.clone(),
        }
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
        if !matches!(self.state, RunState::AwaitingModel) {
            return Err(self.protocol_violation(
                "resolve_streamed_invalid_tool_call called without a pending CallModel step",
            ));
        }

        let diagnostic_history =
            self.streamed_diagnostic_history(partial, Some(invalid.tool_call.clone()));
        let action = self.validate_invalid_tool_call_action(
            action,
            InvalidToolCallDiagnostic {
                tool_call: &invalid.tool_call,
                executable_tool_names: &invalid.executable_tool_names,
                allowed_tool_names: &invalid.allowed_tool_names,
                reason: &invalid.reason,
                history: &diagnostic_history,
            },
        )?;

        match action {
            ValidatedInvalidToolCallAction::Retry { feedback } => self.abandon_streamed_turn(
                partial,
                invalid,
                feedback,
                diagnostic_history,
                "invalid tool call retry produced no retry messages",
                None,
            ),
            ValidatedInvalidToolCallAction::Repair { tool_name } => {
                Ok(StreamedResolution::Repaired { tool_name })
            }
            ValidatedInvalidToolCallAction::Skip { reason } => {
                // Synthetic skip reason: emit verbatim text, matching the
                // non-streamed `resolve_invalid_tool_call` skip path (parity) and
                // avoiding re-parsing a rejection message as structured output.
                let skipped_tool_result = ToolResult {
                    call: invalid.tool_call.id.clone(),
                    provider: invalid.tool_call.provider.clone(),
                    name: invalid.tool_call.function.name.clone(),
                    content: vec![ToolResultContent::text(reason.as_str())],
                };
                self.abandon_streamed_turn(
                    partial,
                    invalid,
                    reason,
                    diagnostic_history,
                    "invalid tool call skip produced no recovery messages",
                    Some(skipped_tool_result),
                )
            }
        }
    }

    /// Resolve a streamed call as ignored. The driver must apply the returned
    /// resolution to the assembler so the call does not enter the turn.
    /// Errors without a pending model step.
    pub fn ignore_streamed_invalid_tool_call(&mut self) -> Result<StreamedResolution, PromptError> {
        if !matches!(self.state, RunState::AwaitingModel) {
            return Err(self.protocol_violation(
                "ignore_streamed_invalid_tool_call called without a pending CallModel step",
            ));
        }
        Ok(StreamedResolution::Ignored)
    }

    /// Shared rollback for the streamed Retry and Skip resolutions: push the
    /// partial turn's rollback messages and abandon the turn, or fail the run
    /// when the partial turn yields no rollback messages.
    fn abandon_streamed_turn(
        &mut self,
        partial: &PartialStreamedTurn,
        invalid: &StreamedInvalidToolCall,
        feedback: String,
        diagnostic_history: Vec<Message>,
        no_messages_reason: &str,
        skipped_tool_result: Option<ToolResult>,
    ) -> Result<StreamedResolution, PromptError> {
        let Some((assistant_message, user_message)) =
            partial.rollback_messages(invalid.tool_call.clone(), feedback)
        else {
            self.state = RunState::Failed;
            return Err(PromptError::prompt_cancelled(
                diagnostic_history,
                no_messages_reason,
            ));
        };
        self.new_messages.push(assistant_message);
        self.new_messages.push(user_message);
        self.rollback_pending = true;
        self.state = RunState::PreparingRequest;
        Ok(StreamedResolution::TurnAbandoned {
            skipped_tool_result,
        })
    }

    /// Feed the assembled streamed turn for the pending
    /// [`AgentRunStep::CallModel`].
    ///
    /// Rejects remaining disallowed tool names without further recovery. Requires
    /// a pending model step and exactly one prior call to
    /// [`AgentRun::record_streamed_completion_call`] for this attempt; otherwise
    /// returns a protocol error. Accepted turns await [`AgentRun::next_step`].
    pub fn streamed_turn(&mut self, turn: StreamedTurn) -> Result<(), PromptError> {
        if !matches!(self.state, RunState::AwaitingModel) {
            return Err(
                self.protocol_violation("streamed_turn called without a pending CallModel step")
            );
        }
        if !self.streamed_completion_call_recorded {
            return Err(self.protocol_violation(
                "streamed_turn called before record_streamed_completion_call recorded the turn's completion call",
            ));
        }

        let has_tool_calls = has_tool_calls(&turn.choice);

        for item in &turn.choice {
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
                return Err(unknown_tool_call_error(
                    tool_call.function.name.clone(),
                    turn.executable_tool_names.iter().cloned().collect(),
                    turn.allowed_tool_names.iter().cloned().collect(),
                    diagnostic_history,
                ));
            }
        }

        self.finalize_turn(
            turn.message_id,
            turn.choice,
            has_tool_calls,
            BTreeMap::new(),
            turn.block_ids,
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

impl AgentRun {
    /// Build a run from a [`RunSpec`], a prompt and an optional prior history.
    ///
    /// Applies the spec's budget, invalid-call retries, output validation and
    /// tool choice; everything else in the spec is request-shaping the driver
    /// reads when it prepares each model call.
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
        if let Some(history) = history {
            run = run.with_history(history);
        }
        if let Some(tool_choice) = spec.tool_choice.clone() {
            run = run.with_tool_choice(tool_choice);
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

#[cfg(test)]
mod tests;
