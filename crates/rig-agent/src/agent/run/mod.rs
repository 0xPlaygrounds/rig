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
//! between *any* two steps — while tool calls are pending, or while a model
//! call is in flight — persist it, and resume it later in another process.
//! Note that serialized run state embeds the full conversation accumulated so
//! far, so persisting it inherits whatever sensitivity the conversation
//! content has.
//!
//! # Serialization format
//!
//! Every payload carries a `$schemaVersion` tag ([`RUN_SCHEMA_VERSION`]), and
//! a build reads only the version it writes. Forward *and* backward
//! compatibility are deliberately fail-fast: a mismatched or absent tag is a
//! deserialization error, never a silent reinterpretation of fields whose
//! meaning moved. Version your agent definitions alongside suspended runs.
//!
//! | version | change |
//! | --- | --- |
//! | `1.0` | Initial versioned format. Records the turn's advertised tool names so a run suspended mid-model-call can be resumed. |
//!
//! `AgentRun` deliberately contains no model, tool registry, memory backend, or
//! hook stack. Hand-driving it is a low-level provider integration: the caller
//! owns all IO and any lifecycle policy. To drive a *configured*
//! [`Agent`](crate::agent::Agent) by hand, use
//! [`Agent::drive`](crate::agent::Agent::drive) — the returned
//! [`AgentDriver`](crate::agent::AgentDriver) seeds this machine from the
//! agent's configuration and pairs each turn's completion request with its
//! tool dispatch snapshot, while every side effect (the provider call, tool
//! execution) stays with the caller. To execute an `Agent` with its hooks,
//! memory, retrieval policy, and telemetry, use
//! [`Agent::runner`](crate::agent::Agent::runner); constructing an `AgentRun`
//! directly is not an alternate way to execute an `Agent`, and the driver is a
//! configuration and pairing layer, not a second execution path.
//!
//! [`crate::completion::Prompt::prompt`] and
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
//!
//! # When building the request can fail
//!
//! [`AgentRun::next_step`] consumes a turn the moment it hands out a
//! [`AgentRunStep::CallModel`], which is right for a driver that has nothing
//! fallible to do with it. A driver that must *build* something first — read a
//! tool registry, resolve a model, validate a tool choice — uses the two halves
//! instead, so a build failure costs no turn and leaves the run intact:
//!
//! ```rust,no_run
//! use rig_agent::agent::run::{Advance, AgentRun, ModelCallInputs};
//!
//! # async fn example(run: &mut AgentRun) -> Result<(), Box<dyn std::error::Error>> {
//! match run.advance()? {
//!     Advance::NeedsModelCall => {
//!         let ModelCallInputs { prompt, history } = run.peek_model_call()?;
//!         // Build the request. Failing here changes nothing about `run`:
//!         // no turn spent, still `is_preparing_request()`, retry at will.
//!         # let _ = (prompt, history);
//!         // Only once the request exists does the run advance.
//!         let _turn = run.commit_model_call(None, None)?;
//!         // Send it. If the send fails or its reply is lost, hand the turn
//!         // back with `run.rollback_model_call()?` and prepare again.
//!     }
//!     Advance::CallTools(calls) => {
//!         # let _ = calls;
//!         // Execute, then: run.tool_results(results)?;
//!     }
//!     Advance::Done(response) => println!("{}", response.output),
//! }
//! # Ok(())
//! # }
//! ```

pub mod output_mode;
pub mod streamed;

pub use output_mode::OutputMode;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use rig_core::message::{
    AssistantContent, ToolCall, ToolChoice, ToolResult, ToolResultContent, UserContent,
};

use crate::{
    agent::hook::{InvalidToolCallAction, InvalidToolCallContext, RetryRequest},
    agent::prompt_request::{
        CompletionCall, PromptResponse, TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER,
        assistant_text_from_choice, build_full_history, build_history_for_request,
        invalid_tool_retry_user_message, is_empty_assistant_turn, tool_result_message,
    },
    agent::turn_tools::TurnToolNames,
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

/// The serialization format [`AgentRun`] writes, and the only one it reads.
///
/// Every bump must add a line to the table in the [module docs](self). Forward
/// compatibility is deliberately fail-fast: a build refuses to deserialize a
/// run written by any other version rather than silently reinterpreting fields
/// whose meaning moved.
pub const RUN_SCHEMA_VERSION: &str = "1.0";

/// Fail-fast schema tag on the serialized [`AgentRun`].
///
/// Serializes as [`RUN_SCHEMA_VERSION`] and refuses to deserialize anything
/// else. The field carries no `serde(default)`, so a payload written before
/// versioning existed fails with a missing-field error instead of being
/// silently accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SchemaVersion;

impl Serialize for SchemaVersion {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(RUN_SCHEMA_VERSION)
    }
}

impl<'de> Deserialize<'de> for SchemaVersion {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let found = String::deserialize(deserializer)?;
        if found == RUN_SCHEMA_VERSION {
            return Ok(Self);
        }
        Err(serde::de::Error::custom(format!(
            "serialized agent run has schema version `{found}`, but this build reads and writes \
             `{RUN_SCHEMA_VERSION}`; resume the run with the rig version that suspended it"
        )))
    }
}

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
    pub choice: Vec<AssistantContent>,
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
        choice: Vec<AssistantContent>,
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
#[must_use = "a model turn can need resolution before the run may advance; \
              answer `NeedsResolution` via `resolve_invalid_tool_call` — dropping \
              the outcome surfaces the same problem two steps later as an \
              unrelated protocol violation"]
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
    original_choice: Vec<AssistantContent>,
    /// Working copy of the assistant content; repairs rename tool calls here.
    items: Vec<AssistantContent>,
    /// Index of the next item to validate.
    next_index: usize,
    executable_tool_names: BTreeSet<String>,
    allowed_tool_names: BTreeSet<String>,
    /// Synthetic tool results for skipped tool calls, keyed by the call's
    /// position in `items` — never by the tool-call id, which is empty for
    /// every call on id-less wires (older ollama daemons) and would collide
    /// two skipped calls (and hand a non-skipped id-less call a preresolved
    /// result it never earned).
    skipped: BTreeMap<usize, UserContent>,
    recovered: bool,
    any_skipped: bool,
    has_tool_calls: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TurnState {
    message_id: Option<String>,
    items: Vec<AssistantContent>,
    has_tool_calls: bool,
    /// Keyed by position in `items` (see `ResolvingState::skipped`).
    skipped: BTreeMap<usize, UserContent>,
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
    /// Fail-fast format tag; see [`RUN_SCHEMA_VERSION`].
    #[serde(rename = "$schemaVersion")]
    schema_version: SchemaVersion,
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
    /// Tool names advertised on the most recently committed model call.
    ///
    /// Recorded by [`AgentRun::commit_model_call`] and kept for the whole turn
    /// — through validation, resolution and tool execution — so a driver can
    /// rebuild the turn's [`TurnTools`](crate::agent::TurnTools) in *any*
    /// state, including in a process that only just deserialized the run.
    /// `None` for runs hand-driven through [`AgentRun::next_step`], where the
    /// advertised names arrive with the [`ModelTurn`] instead.
    #[serde(default)]
    advertised_tools: Option<TurnToolNames>,
    /// How many committed model calls were handed back via
    /// [`AgentRun::rollback_model_call`] — sends that failed, or whose reply
    /// was lost. Observability only: the run imposes no bound on it, because
    /// the caller that owns the transport is the only party that can decide
    /// how many attempts are reasonable.
    #[serde(default)]
    model_call_rollbacks: usize,
    state: RunState,
}

/// The inputs for a model call the run is ready to make, read without
/// advancing anything. See [`AgentRun::peek_model_call`].
///
/// Deliberately exhaustive, matching [`AgentRunStep::CallModel`]'s fields:
/// destructuring it is the point, and a caller that must build a request from
/// these should be told by the compiler when there is more to build it from.
#[derive(Debug, Clone)]
pub struct ModelCallInputs {
    /// The prompt message for this turn (the latest message in the run).
    pub prompt: Message,
    /// The chat history preceding `prompt`: the caller-provided input history
    /// followed by messages accumulated by earlier turns.
    pub history: Vec<Message>,
}

/// What [`AgentRun::advance`] resolved to.
///
/// Splits "the run wants another model call" from every step that needs no
/// model call, so a caller whose request preparation can fail does that work
/// *before* anything advances (see [`AgentRun::commit_model_call`]).
///
/// Deliberately exhaustive, like [`AgentRunStep`]: a driver must handle every
/// variant, so adding one is a breaking change by design.
#[derive(Debug)]
pub enum Advance {
    /// The run is parked in `PreparingRequest`: read the inputs with
    /// [`AgentRun::peek_model_call`], build the request, and commit it with
    /// [`AgentRun::commit_model_call`]. Nothing has advanced yet.
    NeedsModelCall,
    /// Execute these tool calls, then feed [`AgentRun::tool_results`].
    CallTools(Vec<PendingToolCall>),
    /// The run is complete.
    Done(PromptResponse),
}

impl AgentRun {
    /// Create a run for one prompt with no input history, a one-model-call
    /// budget, and no invalid tool-call retries.
    pub fn new(prompt: impl Into<Message>) -> Self {
        Self {
            schema_version: SchemaVersion,
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
            advertised_tools: None,
            model_call_rollbacks: 0,
            state: RunState::PreparingRequest,
        }
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
    /// message to the history. Consumes one output-retry and parks the run in
    /// `PreparingRequest`; the caller loops back through [`Self::advance`],
    /// which reports the pending model call without committing it.
    fn reprompt_for_output(&mut self) {
        self.output_retries += 1;
        self.park_for_new_request();
    }

    /// Park the run for a fresh model call, ending the current turn.
    ///
    /// The turn's advertised names end with it. They describe the call the
    /// model was shown, and [`Self::advertised_tools`] is documented as the
    /// record of *this* turn — so a run parked here, in memory or serialized,
    /// must not still report the names of a turn that finished or was
    /// abandoned. Every route back to `PreparingRequest` goes through here so
    /// the invariant cannot be half-applied.
    fn park_for_new_request(&mut self) {
        self.advertised_tools = None;
        self.state = RunState::PreparingRequest;
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

    /// The tool choice active for this run, if one was set.
    ///
    /// A driver preparing this run's requests must honor it: it is the run's
    /// own policy, not the agent's baseline, and it governs both the request
    /// sent to the provider and the run's internal decisions (see
    /// [`Self::with_tool_choice`]).
    pub fn tool_choice(&self) -> Option<&ToolChoice> {
        self.tool_choice.as_ref()
    }

    /// The tool names advertised on the current turn's model call, when the
    /// call was committed with them (see [`Self::commit_model_call`]).
    ///
    /// This is the durable half of the turn's
    /// [`TurnTools`](crate::agent::TurnTools): pairing it with a registry
    /// snapshot reconstitutes the whole thing, which is what lets a driver
    /// resume a run suspended at *any* step rather than only while tool calls
    /// are pending. It is also the record of what the model was actually shown
    /// this turn, which is worth persisting alongside the run for audit.
    ///
    /// `None` for a run hand-driven through [`Self::next_step`], which commits
    /// no names because the driver supplies them with the [`ModelTurn`]
    /// instead, and `None` before the run's first model call.
    pub fn advertised_tools(&self) -> Option<&TurnToolNames> {
        self.advertised_tools.as_ref()
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

    /// Canonical content for the accepted model turn awaiting advancement.
    pub(crate) fn accepted_turn_choice(&self) -> Option<Vec<AssistantContent>> {
        let RunState::AwaitingAdvance(turn) = &self.state else {
            return None;
        };

        // Deliberately not `non_empty(turn.items.clone())`: the helper takes
        // the list by value, so it would pay the clone even when the turn is
        // empty and the copy is discarded. Check first, clone only on the
        // path that keeps it.
        if turn.items.is_empty() {
            return None;
        }
        Some(turn.items.clone())
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
                // The rejected turn may legitimately have carried nothing — that
                // is often *why* a hook rejected it. Cancelling here was
                // unreachable while the streaming accumulator padded empty turns
                // with a fabricated empty-text part; without that padding it
                // would fail exactly the runs a feedback retry exists to rescue.
                // The `is_empty_assistant_turn` check immediately below is what
                // keeps a content-less turn out of history.
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

        self.park_for_new_request();
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
            tool_call_id: Some(tool_call.id.as_str().to_owned()),
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

    /// Whether the run is waiting for its next model call to be prepared —
    /// that is, whether [`Self::peek_model_call`] and
    /// [`Self::commit_model_call`] are valid right now.
    pub fn is_preparing_request(&self) -> bool {
        matches!(self.state, RunState::PreparingRequest)
    }

    /// Read the inputs for the pending model call **without advancing**.
    ///
    /// Pure over committed state: it consumes no turn, moves no state, and
    /// poisons nothing on failure. A caller whose request preparation can fail
    /// — an agent driver reading a tool registry, say — peeks first, builds the
    /// request, and only then calls [`Self::commit_model_call`]. A preparation
    /// failure therefore leaves the run byte-identical to before the attempt,
    /// so it can be retried in this process or serialized and retried in
    /// another.
    ///
    /// # Errors
    /// - [`PromptError::MaxTurnsError`] when the total model-call budget is
    ///   exhausted. The run stays intact, so raising the budget and peeking
    ///   again resumes it.
    /// - [`PromptError::PromptCancelled`] when there is no pending prompt, or
    ///   when the run is not preparing a request (check with
    ///   [`Self::is_preparing_request`], or reach this through
    ///   [`Self::advance`], which reports it).
    pub fn peek_model_call(&self) -> Result<ModelCallInputs, PromptError> {
        if !self.is_preparing_request() {
            return Err(
                self.protocol_violation("peek_model_call called while the run wants no model call")
            );
        }
        let Some((prompt, history_for_turn)) = self.new_messages.split_last() else {
            return Err(PromptError::prompt_cancelled(
                self.full_history(),
                "prompt loop lost its pending prompt",
            ));
        };

        self.check_turn_budget()?;

        Ok(ModelCallInputs {
            prompt: prompt.clone(),
            history: build_history_for_request(self.chat_history.as_deref(), history_for_turn),
        })
    }

    /// Whether another model call fits in the run's budget.
    ///
    /// Checked in **two** places on purpose. [`Self::peek_model_call`] runs it
    /// so a caller learns the budget is spent before building a request it
    /// cannot send; [`Self::commit_model_call`] runs it because that is where
    /// the turn is actually consumed, and the two halves are public — nothing
    /// obliges a caller to peek before every commit, and a caller that peeks
    /// once and re-commits (after a rollback, say) would otherwise drive model
    /// calls forever against a budget never consulted.
    fn check_turn_budget(&self) -> Result<(), PromptError> {
        if self.current_turn < self.max_turns {
            return Ok(());
        }
        let prompt = self
            .new_messages
            .last()
            .cloned()
            .unwrap_or_else(|| Message::user(""));
        Err(PromptError::MaxTurnsError {
            max_turns: self.max_turns,
            chat_history: self.full_history().into(),
            prompt: prompt.into(),
        })
    }

    /// Commit the model call previewed by [`Self::peek_model_call`], returning
    /// its one-based turn index.
    ///
    /// This is the only place a turn is consumed. It is deliberately
    /// infallible and deliberately last: everything that can fail — reading the
    /// tool registry, resolving the output mode, validating the tool choice —
    /// has already happened by the time it runs, so the run never advances into
    /// a call that was never made.
    ///
    /// `advertised` records the turn's tool names on the run (see
    /// [`Self::advertised_tools`]); pass `None` when the caller supplies them
    /// with the [`ModelTurn`] instead. `output_tool_name` fills the run's
    /// committed Tool-mode name once (#1928), pinning the mode for the rest of
    /// the run.
    ///
    /// If the committed call then fails to reach the provider, or its reply is
    /// lost, [`Self::rollback_model_call`] undoes exactly this.
    ///
    /// # Errors
    /// - [`PromptError::PromptCancelled`] when the run is not preparing a
    ///   request — committing a call the run never asked for would consume a
    ///   turn against nothing.
    /// - [`PromptError::MaxTurnsError`] when the model-call budget is spent.
    ///   Checked here as well as in [`Self::peek_model_call`], because this is
    ///   where the turn is actually consumed and a caller is not obliged to
    ///   peek before every commit.
    pub fn commit_model_call(
        &mut self,
        advertised: Option<TurnToolNames>,
        output_tool_name: Option<String>,
    ) -> Result<usize, PromptError> {
        if !self.is_preparing_request() {
            return Err(
                self.protocol_violation("commit_model_call called without a peeked model call")
            );
        }
        self.check_turn_budget()?;
        self.set_output_tool_name(output_tool_name);
        self.advertised_tools = advertised;
        self.current_turn += 1;
        self.rollback_pending = false;
        self.streamed_completion_call_recorded = false;
        self.state = RunState::AwaitingModel;
        Ok(self.current_turn)
    }

    /// Hand back a committed model call that never produced a response.
    ///
    /// For a request that could not be sent, or whose reply is known to be
    /// lost: a transport error, a timeout, a cancelled or vanished provider
    /// job. The turn is refunded — it produced nothing, so it costs nothing —
    /// and the run returns to `PreparingRequest`, where the next
    /// [`Self::peek_model_call`] / [`Self::commit_model_call`] pair (or
    /// [`Self::next_step`]) prepares the request **again, from current
    /// configuration**.
    ///
    /// Re-deriving rather than re-sending is deliberate. A request built for
    /// the failed attempt carries that attempt's tool snapshot, and replaying
    /// it would advertise one set of implementations while a later turn
    /// dispatches another — the exact skew
    /// [`TurnTools`](crate::agent::TurnTools) exists to prevent. The retry
    /// therefore takes a fresh snapshot, a fresh patch, and the same history.
    ///
    /// # Retryable is not replay-safe
    ///
    /// Two independent questions decide whether to roll back, and only the
    /// first has a library answer:
    ///
    /// 1. **Could a retry succeed?** A property of the failure.
    ///    [`CompletionError::is_retryable`](crate::completion::CompletionError::is_retryable)
    ///    answers it.
    /// 2. **Is a retry safe?** Did the request already take effect? Nothing in
    ///    the run can tell "never arrived" from "arrived, reply lost", and
    ///    neither can `is_retryable` — a stream that died *after* the request
    ///    was written is highly retryable and not replay-safe at all.
    ///
    /// Rolling back on question 1 alone is at-least-once: a request that
    /// reached the provider and lost only its reply is billed twice, and any
    /// side effect the model already caused happens again. Only the caller can
    /// settle question 2 — through provider-side idempotency, its own record
    /// of what was transmitted, or a transport that fails before the write.
    /// Roll back when you know the call produced nothing; when you do not
    /// know, prefer failing the run.
    ///
    /// Usage accounting is preserved either way. Tokens the provider already
    /// billed stay in [`Self::usage`] and [`Self::completion_calls`], and a
    /// streamed turn may still record its usage after the rollback (the same
    /// window invalid tool-call recovery uses, see
    /// [`Self::record_streamed_completion_call`]).
    ///
    /// Bounding attempts is the caller's job: the machine performs no IO and
    /// owns no clock, so it cannot know whether a retry is reasonable.
    /// [`Self::model_call_rollbacks`] counts them for a caller that wants a
    /// budget of its own, and
    /// [`CompletionError::is_retryable`](crate::completion::CompletionError::is_retryable)
    /// classifies whether the failure is worth another attempt at all.
    ///
    /// # Errors
    /// [`PromptError::PromptCancelled`] when no model call is in flight.
    pub fn rollback_model_call(&mut self) -> Result<(), PromptError> {
        if !matches!(self.state, RunState::AwaitingModel) {
            return Err(self
                .protocol_violation("rollback_model_call called without a model call in flight"));
        }

        // Refund the turn. `saturating_sub` is belt-and-braces: reaching
        // `AwaitingModel` always went through `commit_model_call`, which
        // incremented it.
        self.current_turn = self.current_turn.saturating_sub(1);
        self.model_call_rollbacks += 1;
        // The next preparation records its own; leaving stale names here would
        // let a later `model_response` validate against a turn that never ran.
        // Keep the streamed-usage window open when the record has not landed
        // yet: a caller that drains a broken stream for usage records it after
        // this point, exactly as invalid tool-call recovery does.
        self.rollback_pending = !self.streamed_completion_call_recorded;
        self.park_for_new_request();
        Ok(())
    }

    /// How many committed model calls were handed back via
    /// [`Self::rollback_model_call`].
    pub fn model_call_rollbacks(&self) -> usize {
        self.model_call_rollbacks
    }

    /// Advance the machine as far as it can go without a model call.
    ///
    /// Returns [`Advance::NeedsModelCall`] the moment the run wants one —
    /// including when an output-mode re-prompt rolls the run back mid-advance
    /// (#1928) — leaving the run in `PreparingRequest` for the caller to
    /// prepare and commit. [`Self::next_step`] is this plus an immediate
    /// commit, for callers with nothing that can fail in between.
    pub fn advance(&mut self) -> Result<Advance, PromptError> {
        loop {
            if self.is_preparing_request() {
                return Ok(Advance::NeedsModelCall);
            }
            match std::mem::replace(&mut self.state, RunState::Failed) {
                // Handled above; re-entering here would advance a run the
                // caller has not prepared a request for.
                RunState::PreparingRequest => {
                    self.state = RunState::PreparingRequest;
                    return Ok(Advance::NeedsModelCall);
                }
                RunState::AwaitingAdvance(turn_state) => {
                    let TurnState {
                        message_id,
                        items,
                        has_tool_calls,
                        skipped,
                        mut internal_call_ids,
                    } = *turn_state;
                    // Tool output mode (#1928): a call to the synthetic output tool
                    // finalizes the run with the call's arguments as the response,
                    // instead of executing it as a tool. First match wins; any
                    // sibling tool calls in the same turn are dropped.
                    if has_tool_calls
                        && let Some(output_tool_name) = self.output_tool_name.clone()
                        && let Some(tool_call) = items.iter().find_map(|item| match item {
                            AssistantContent::ToolCall(tc)
                                if tc.function.name == output_tool_name =>
                            {
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
                                content: items.clone(),
                            });
                            let feedback = format!(
                                "The `{output_tool_name}` arguments were missing required field(s): \
                             {}. Call `{output_tool_name}` again with every required field.",
                                missing.join(", ")
                            );
                            if let Some(user_message) =
                                invalid_tool_retry_user_message(&items, &tool_call_id, feedback)
                            {
                                self.new_messages.push(user_message);
                            }
                            self.reprompt_for_output();
                            continue;
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
                        self.new_messages.push(Message::Assistant {
                            id: message_id,
                            content: final_items.clone(),
                        });

                        let response = PromptResponse::new(output, self.usage)
                            .with_messages(self.new_messages.clone())
                            .with_completion_calls(self.completion_calls.clone())
                            .with_output_tool_calls(output_tool_calls)
                            .with_content(final_items);
                        self.state = RunState::Done(Box::new(response.clone()));
                        return Ok(Advance::Done(response));
                    }

                    // An empty turn is not a lost turn. Cancelling here would fail
                    // runs that previously succeeded: with the fabricated empty-text
                    // padding gone, a textless turn — a tool-call-only turn whose
                    // calls were all dropped, a content-filtered turn, a truncated
                    // stream — arrives honestly empty. `is_empty_assistant_turn`
                    // does the right thing: keep the turn out of history and carry on.
                    if !is_empty_assistant_turn(&items) {
                        self.new_messages.push(Message::Assistant {
                            id: message_id,
                            content: items.clone(),
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
                            .enumerate()
                            .filter_map(|(index, item)| match item {
                                AssistantContent::ToolCall(tool_call) => {
                                    // Consume pairs positionally so duplicate
                                    // provider IDs within one turn stay
                                    // distinguishable.
                                    let internal_call_id = internal_call_ids
                                        .iter()
                                        .position(|(id, _)| tool_call.id == id.as_str())
                                        .map(|pair| internal_call_ids.remove(pair).1);
                                    Some(PendingToolCall {
                                        tool_call: tool_call.clone(),
                                        preresolved_result: skipped.get(&index).cloned(),
                                        internal_call_id,
                                    })
                                }
                                _ => None,
                            })
                            .collect();
                        self.state = RunState::ExecutingTools(calls.clone());
                        return Ok(Advance::CallTools(calls));
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
                            && !is_empty_assistant_turn(&items)
                            && self.can_reprompt_for_output()
                            && !self
                                .text_satisfies_output_schema(&assistant_text_from_choice(&items))
                        {
                            let feedback = format!(
                                "Provide your final answer by calling the `{output_tool_name}` tool \
                             with the structured result as its arguments, not as plain text."
                            );
                            self.new_messages.push(Message::user(feedback));
                            self.reprompt_for_output();
                            continue;
                        }

                        let response =
                            PromptResponse::new(assistant_text_from_choice(&items), self.usage)
                                .with_messages(self.new_messages.clone())
                                .with_completion_calls(self.completion_calls.clone())
                                .with_content(items);
                        self.state = RunState::Done(Box::new(response.clone()));
                        return Ok(Advance::Done(response));
                    }
                }
                RunState::ExecutingTools(calls) => {
                    // Idempotent, like Done: a process resuming a serialized run
                    // re-obtains the pending tool calls from the state itself.
                    self.state = RunState::ExecutingTools(calls.clone());
                    return Ok(Advance::CallTools(calls));
                }
                RunState::Done(response) => {
                    let step = Advance::Done((*response).clone());
                    self.state = RunState::Done(response);
                    return Ok(step);
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
                    return Err(self.protocol_violation(reason));
                }
                RunState::Failed => {
                    return Err(self.protocol_violation(
                        "next_step called after the run already failed or was misdriven",
                    ));
                }
            }
        }
    }

    /// Advance the machine and return the next action for the driver.
    ///
    /// [`Self::advance`] plus an immediate commit of any model call it asks
    /// for — the right entry point for a driver that builds its requests from
    /// the returned inputs with nothing fallible in between. A driver whose
    /// request preparation *can* fail (see [`Self::peek_model_call`]) should
    /// use the two halves instead, so a failure does not consume a turn.
    ///
    /// # Errors
    /// - [`PromptError::MaxTurnsError`] when the total model-call budget is exhausted.
    /// - [`PromptError::PromptCancelled`] when the machine is driven out of
    ///   protocol (for example, calling this while a model response is
    ///   pending).
    pub fn next_step(&mut self) -> Result<AgentRunStep, PromptError> {
        match self.advance()? {
            Advance::NeedsModelCall => {
                let ModelCallInputs { prompt, history } = self.peek_model_call()?;
                let turn = self.commit_model_call(None, None)?;
                Ok(AgentRunStep::CallModel {
                    prompt,
                    history,
                    turn,
                })
            }
            Advance::CallTools(calls) => Ok(AgentRunStep::CallTools { calls }),
            Advance::Done(response) => Ok(AgentRunStep::Done(response)),
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

        let items: Vec<AssistantContent> = turn.choice.clone();
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
    /// ingestion paths converge here, differing only in the `skipped` map and
    /// the streamed `internal_call_ids`.
    fn finalize_turn(
        &mut self,
        message_id: Option<String>,
        items: Vec<AssistantContent>,
        has_tool_calls: bool,
        skipped: BTreeMap<usize, UserContent>,
        internal_call_ids: Vec<(String, String)>,
    ) {
        self.state = RunState::AwaitingAdvance(Box::new(TurnState {
            message_id,
            items,
            has_tool_calls,
            skipped,
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
    pub fn resolve_invalid_tool_call(
        &mut self,
        action: InvalidToolCallAction,
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
                self.park_for_new_request();
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

    /// Discard the pending invalid tool call without marking the turn as
    /// recovered.
    ///
    /// This is the extractor driver's fallback after every invalid-tool hook
    /// has declined to act. Keeping it distinct from [`InvalidToolCallAction::Skip`]
    /// preserves the extractor's legacy response semantics: unrelated calls
    /// disappear, a sibling output call can still finalize the turn, and
    /// response observers still receive the canonical response fields.
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
        resolving.has_tool_calls = resolving
            .items
            .iter()
            .any(|item| matches!(item, AssistantContent::ToolCall(_)));
        // Dropping the last item leaves the turn empty, which is now
        // representable. This used to push a fabricated empty-text part so the
        // content type stayed satisfied; `is_empty_assistant_turn` keeps such a
        // turn out of history further along.
        self.state = RunState::ResolvingToolCalls(resolving);
        self.advance_resolution()
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
            .map(|call| call.tool_call.id.as_str().to_owned())
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
            let Some(index) = unanswered
                .iter()
                .position(|id| tool_result.call == id.as_str())
            else {
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
        self.park_for_new_request();
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
            tool_call_id: Some(invalid.tool_call.id.as_str().to_owned()),
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
                self.park_for_new_request();
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
                    call: invalid.tool_call.id.clone(),
                    provider: invalid.tool_call.provider.clone(),
                    name: invalid.tool_call.function.name.clone(),
                    content: vec![ToolResultContent::text(reason.clone())],
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
                self.park_for_new_request();
                Ok(StreamedResolution::TurnAbandoned {
                    skipped_tool_result: Some(Box::new(skipped_tool_result)),
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
            // `Usage::new()` is the additive identity for `Usage`'s `AddAssign`,
            // so routing the no-usage fallback through `record_completion_call`
            // leaves the run total unchanged while unifying the accounting.
            self.record_completion_call(Usage::new());
            self.streamed_completion_call_recorded = true;
        }

        let has_tool_calls = turn
            .choice
            .iter()
            .any(|item| matches!(item, AssistantContent::ToolCall(_)));

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
            turn.internal_call_ids,
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
            vec![AssistantContent::text(text)],
            Usage::new(),
            tool_names(&["add"]),
            tool_names(&["add"]),
        )
    }

    fn tool_call(id: &str, name: &str) -> AssistantContent {
        // The provider-boundary shape: a non-empty wire id becomes both the
        // durable id and the provider correlator; an empty wire id mints a
        // fresh unique handle (`provider` records the absence).
        AssistantContent::ToolCall(ToolCall::from_wire(
            id,
            ToolFunction::new(name.to_string(), json!({"x": 1})),
        ))
    }

    fn tool_call_turn(id: &str, name: &str) -> ModelTurn {
        ModelTurn::new(
            None,
            vec![tool_call(id, name)],
            Usage::new(),
            tool_names(&["add"]),
            tool_names(&["add"]),
        )
    }

    fn tool_result(id: &str, output: &str) -> UserContent {
        // Every result in these tests answers a call to the `add` tool; the
        // executed tool's name is required data on a result.
        UserContent::tool_result(id, "add", vec![ToolResultContent::text(output)])
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

    /// A model call that was committed and then failed to reach the provider:
    /// the turn is refunded, the run is drivable again, and the retry takes
    /// the turn the failure did not.
    #[test]
    fn rollback_model_call_refunds_the_turn_and_re_prepares() {
        let mut run = AgentRun::new("add things").max_turns(1);
        expect_call_model(&mut run);
        assert_eq!(run.turn(), 1);

        run.rollback_model_call().expect("rollback should succeed");
        assert_eq!(run.turn(), 0, "a call that produced nothing costs nothing");
        assert_eq!(run.model_call_rollbacks(), 1);
        assert!(run.is_preparing_request());

        // The budget of one is intact, so the retry is possible at all.
        let (_, _, turn) = expect_call_model(&mut run);
        assert_eq!(turn, 1);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add"))
                .expect("the retried turn is accepted"),
        );
    }

    /// The advertised names belong to the call that was handed back, so they
    /// must not survive it: a later `model_response` validating against a turn
    /// that never ran would be validating against nothing.
    #[test]
    fn rollback_model_call_drops_the_advertised_tools() {
        let mut run = AgentRun::new("add things").max_turns(2);
        run.advance().expect("advance should succeed");
        run.peek_model_call().expect("peek should succeed");
        run.commit_model_call(
            Some(TurnToolNames {
                executable: tool_names(&["add"]),
                allowed: tool_names(&["add"]),
            }),
            None,
        )
        .expect("commit should succeed");
        assert!(run.advertised_tools().is_some());

        run.rollback_model_call().expect("rollback should succeed");
        assert_eq!(run.advertised_tools(), None);
    }

    /// Tokens the provider already billed stay billed. A streamed turn that
    /// died mid-stream may still learn its usage afterwards, so the recording
    /// window stays open across the rollback — the same window invalid
    /// tool-call recovery uses.
    #[test]
    fn rollback_model_call_preserves_streamed_usage_accounting() {
        let mut run = AgentRun::new("add things").max_turns(2);
        expect_call_model(&mut run);
        run.record_streamed_completion_call(Usage {
            total_tokens: 11,
            ..Usage::new()
        })
        .expect("usage recorded before the failure");

        run.rollback_model_call().expect("rollback should succeed");
        assert_eq!(
            run.usage().total_tokens,
            11,
            "billed tokens survive the rollback"
        );
        assert_eq!(run.completion_calls().len(), 1);

        // A second record for the same dead turn is still rejected...
        run.record_streamed_completion_call(Usage::new())
            .expect_err("the turn already recorded its usage");

        // ...and the retried turn records its own.
        expect_call_model(&mut run);
        run.record_streamed_completion_call(Usage {
            total_tokens: 7,
            ..Usage::new()
        })
        .expect("the retried turn records its own usage");
        assert_eq!(run.usage().total_tokens, 18);
        assert_eq!(run.completion_calls().len(), 2);
    }

    /// A stream that broke before reporting usage can still report it after
    /// the rollback, which is why the window is left open when nothing was
    /// recorded.
    #[test]
    fn rollback_model_call_leaves_the_streamed_usage_window_open() {
        let mut run = AgentRun::new("add things").max_turns(2);
        expect_call_model(&mut run);
        run.rollback_model_call().expect("rollback should succeed");
        run.record_streamed_completion_call(Usage {
            total_tokens: 3,
            ..Usage::new()
        })
        .expect("a drained stream may report usage after the rollback");
        assert_eq!(run.usage().total_tokens, 3);
    }

    #[test]
    fn rollback_model_call_requires_a_call_in_flight() {
        let mut run = AgentRun::new("add things").max_turns(2);
        run.rollback_model_call()
            .expect_err("nothing is in flight before the first call");

        expect_call_model(&mut run);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add"))
                .expect("model_response should succeed"),
        );
        expect_call_tools(&mut run);
        run.rollback_model_call()
            .expect_err("tool calls are pending, not a model call");
    }

    /// The transition is state, not driver memory, so it survives suspension.
    #[test]
    fn rollback_model_call_survives_a_serde_round_trip() {
        let mut run = AgentRun::new("add things").max_turns(1);
        expect_call_model(&mut run);
        run.rollback_model_call().expect("rollback should succeed");

        let serialized = serde_json::to_string(&run).expect("run should serialize");
        let mut restored: AgentRun =
            serde_json::from_str(&serialized).expect("run should deserialize");
        assert_eq!(restored.turn(), 0);
        assert_eq!(restored.model_call_rollbacks(), 1);
        let (_, _, turn) = expect_call_model(&mut restored);
        assert_eq!(turn, 1);
    }

    /// `advertised_tools` is the record of what the model was shown on the
    /// *current* turn, so every route back to `PreparingRequest` must end the
    /// turn's names with it — not just the rollback path. A run parked for a
    /// fresh call and then serialized would otherwise report a turn that is
    /// over.
    #[test]
    fn every_route_back_to_preparing_ends_the_turns_advertised_names() {
        let advertised = TurnToolNames::new(["add"], ["add"]);

        // Normal progression: tool results end the turn.
        let mut run = AgentRun::new("add things").max_turns(3);
        run.advance().expect("advance");
        run.peek_model_call().expect("peek");
        run.commit_model_call(Some(advertised.clone()), None)
            .expect("commit");
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add"))
                .expect("model_response"),
        );
        expect_call_tools(&mut run);
        assert!(run.advertised_tools().is_some(), "the turn is still live");
        run.tool_results(vec![tool_result("call_1", "2")])
            .expect("tool_results");
        assert_eq!(
            run.advertised_tools(),
            None,
            "tool results end the turn that advertised them"
        );

        // Hook-driven turn retry.
        let mut run = AgentRun::new("add things").max_turns(3);
        run.advance().expect("advance");
        run.peek_model_call().expect("peek");
        run.commit_model_call(Some(advertised.clone()), None)
            .expect("commit");
        expect_continue(
            run.model_response(text_turn("nope"))
                .expect("model_response"),
        );
        run.retry_model_turn(RetryRequest::Repeat)
            .expect("retry should succeed");
        assert_eq!(
            run.advertised_tools(),
            None,
            "a retried turn's names do not outlive it"
        );
    }

    /// The budget is checked where it is *spent*, not only where it is
    /// previewed. Both halves are public, and nothing obliges a caller to peek
    /// before every commit — a caller that peeks once and re-commits (after a
    /// rollback, say) would otherwise drive model calls forever.
    #[test]
    fn commit_model_call_enforces_the_turn_budget_without_a_peek() {
        let mut run = AgentRun::new("add things").max_turns(1);

        // Spend the budget through the normal path.
        expect_call_model(&mut run);
        run.rollback_model_call().expect("rollback should succeed");
        expect_call_model(&mut run);
        assert_eq!(run.turn(), 1);
        expect_continue(
            run.model_response(tool_call_turn("call_1", "add"))
                .expect("model_response should succeed"),
        );
        expect_call_tools(&mut run);
        run.tool_results(vec![tool_result("call_1", "2")])
            .expect("tool_results should succeed");

        // The run wants another call and the budget is spent. Committing
        // without peeking must not get one.
        assert!(run.is_preparing_request());
        let err = run
            .commit_model_call(None, None)
            .expect_err("the budget is spent");
        assert!(
            matches!(err, PromptError::MaxTurnsError { .. }),
            "expected MaxTurnsError, got {err:?}"
        );
        assert_eq!(run.turn(), 1, "a refused commit consumes nothing");
    }

    /// The two halves are a public protocol, so driving them out of order is
    /// an error a caller can handle — not a `debug_assert` that vanishes in
    /// release builds.
    #[test]
    fn peek_and_commit_reject_being_driven_out_of_protocol() {
        let mut run = AgentRun::new("add things").max_turns(2);
        expect_call_model(&mut run);

        run.peek_model_call()
            .expect_err("a model call is already in flight");
        run.commit_model_call(None, None)
            .expect_err("there is no peeked call to commit");
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
            vec![tool_call("call_1", "add"), tool_call("call_2", "add")],
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
                if matches!(content.first(), Some(UserContent::ToolResult(_)))
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

    #[test]
    fn invalid_tool_call_repair_renames_and_suppresses_response_hook() {
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
            vec![tool_call("call_1", "unknown"), tool_call("call_2", "add")],
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

    /// Two ID-LESS calls (older ollama daemons issue no tool-call id) must
    /// not collide in the skipped map: each mints its own unique correlation
    /// handle at the provider boundary, so skipping the invalid one leaves
    /// the valid peer with its own "not executed" result, and each call reads
    /// back its OWN preresolved result — position, not id, is the key.
    #[test]
    fn id_less_calls_keep_distinct_skip_results() {
        let mut run = AgentRun::new("call things").max_turns(2);

        expect_call_model(&mut run);
        let turn = ModelTurn::new(
            None,
            vec![tool_call("", "unknown"), tool_call("", "add")],
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
        assert_ne!(
            calls[0].tool_call.id, calls[1].tool_call.id,
            "id-less calls mint distinct correlation handles, never a shared sentinel"
        );
        assert!(
            calls
                .iter()
                .all(|call| !call.tool_call.id.is_empty() && call.tool_call.provider.is_none()),
            "minted handles are non-empty and record the provider's absence"
        );
        let results: Vec<String> = calls
            .iter()
            .map(|call| match call.preresolved_result.as_ref() {
                Some(rig_core::message::UserContent::ToolResult(result)) => result
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        rig_core::message::ToolResultContent::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect(),
                _ => panic!("both calls carry preresolved results"),
            })
            .collect();
        assert_eq!(
            results[0], "not available",
            "the skipped call reads its own feedback"
        );
        assert_eq!(
            results[1], TOOL_NOT_EXECUTED_DUE_TO_INVALID_PEER,
            "the peer reads its own synthetic result, not the skipped one\'s"
        );
    }

    #[test]
    fn skip_under_tool_choice_none_fails() {
        let mut run = AgentRun::new("call something").with_tool_choice(ToolChoice::None);

        expect_call_model(&mut run);
        expect_needs_resolution(
            run.model_response(ModelTurn::new(
                None,
                vec![tool_call("call_1", "add")],
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

    /// A suspended run in the current serialization: tagged assistant content,
    /// and `"usage": null` on the completion call pinning `CompletionCall`'s
    /// null tolerance (the pre-monoid `Option` encoding must still map to
    /// zero-valued usage). Fields added since are defaulted in; only the
    /// version tag gates the load.
    const SUSPENDED_RUN_FIXTURE: &str = r#"{"$schemaVersion":"1.0","max_turns":2,"max_invalid_tool_call_retries":0,"tool_choice":null,"chat_history":null,"new_messages":[{"role":"user","content":[{"type":"text","text":"add things"}]},{"role":"assistant","id":null,"content":[{"type":"toolcall","id":"call_1","function":{"name":"add","arguments":{"x":1}},"signature":null,"additional_params":null}]}],"current_turn":1,"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"cached_input_tokens":0,"cache_creation_input_tokens":0,"tool_use_prompt_tokens":0,"reasoning_tokens":0},"completion_calls":[{"call_index":0,"usage":null}],"completion_call_index":1,"invalid_tool_call_retries":0,"rollback_pending":false,"streamed_completion_call_recorded":false,"state":{"ExecutingTools":[{"tool_call":{"id":"call_1","function":{"name":"add","arguments":{"x":1}},"signature":null,"additional_params":null},"preresolved_result":null,"internal_call_id":null}]}}"#;

    #[test]
    fn agent_run_deserializes_suspended_state_with_defaulted_fields() {
        let mut restored: AgentRun = serde_json::from_str(SUSPENDED_RUN_FIXTURE)
            .expect("in-version suspended run should deserialize");
        assert_eq!(restored.completion_calls()[0].usage, Usage::new());
        assert_eq!(
            restored.advertised_tools(),
            None,
            "a run suspended by a raw hand-driver records no advertised tools"
        );

        let calls = expect_call_tools(&mut restored);
        assert_eq!(calls.len(), 1);
        restored
            .tool_results(vec![tool_result("call_1", "2")])
            .expect("tool_results should succeed");
        expect_call_model(&mut restored);
    }

    /// A prose warning cannot fail a load; the version tag can. A payload
    /// written by another format version is rejected outright rather than
    /// silently reinterpreted — including one written before versioning
    /// existed, which has no tag at all.
    #[test]
    fn agent_run_rejects_foreign_schema_versions() {
        let untagged = SUSPENDED_RUN_FIXTURE.replace(r#""$schemaVersion":"1.0","#, "");
        let err = serde_json::from_str::<AgentRun>(&untagged)
            .expect_err("an untagged run must not deserialize");
        assert!(
            err.to_string().contains("$schemaVersion"),
            "error should name the missing tag, got: {err}"
        );

        let future =
            SUSPENDED_RUN_FIXTURE.replace(r#""$schemaVersion":"1.0""#, r#""$schemaVersion":"2.0""#);
        let err = serde_json::from_str::<AgentRun>(&future)
            .expect_err("a run from another version must not deserialize");
        let message = err.to_string();
        assert!(
            message.contains("2.0") && message.contains(RUN_SCHEMA_VERSION),
            "error should name both versions, got: {message}"
        );
    }

    #[test]
    fn serialized_run_carries_the_schema_version() {
        let run = AgentRun::new("add things");
        let json: serde_json::Value = serde_json::to_value(&run).expect("run should serialize");
        assert_eq!(
            json.get("$schemaVersion").and_then(|v| v.as_str()),
            Some(RUN_SCHEMA_VERSION)
        );
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
        // Direct value comparison: with `additional_params` a named field,
        // a restored message is identical to the live one — no serialized-form
        // detour that would hide a round-trip divergence.
        assert_eq!(resumed.messages, uninterrupted.messages);
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
            vec![tool_call(id, name)],
            Usage::new(),
            tool_names(&["add"]),
            tool_names(&["add", name]),
        )
    }

    fn output_tool_turn_with_args(id: &str, name: &str, arguments: serde_json::Value) -> ModelTurn {
        ModelTurn::new(
            None,
            vec![AssistantContent::ToolCall(ToolCall::from_wire(
                id,
                ToolFunction::new(name.to_string(), arguments),
            ))],
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
                        answered.insert(result.call.to_string());
                    }
                }
            }
        }
        for message in messages {
            if let Message::Assistant { content, .. } = message {
                for item in content.iter() {
                    if let AssistantContent::ToolCall(call) = item {
                        assert!(
                            answered.contains(call.id.as_str()),
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
            vec![
                tool_call("call_1", "add"),
                tool_call("call_2", "final_result"),
            ],
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
        let two_calls = vec![tool_call("c1", "add"), tool_call("c2", "add")];
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
}
