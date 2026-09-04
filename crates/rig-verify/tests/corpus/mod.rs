//! The effect corpus's two interpreters and the program table they share.
//!
//! Every golden effect log under `fixtures/*.effects.json` is one row of
//! one matrix: a [`Program`] the producing root test built verbatim,
//! replayed here by the bus-driven engine and by a hand driver of
//! `AgentRun` with **no provider, no tool, no memory and no index behind
//! any key**. The oracle is the record as data — kind, outcome and, for a
//! stream recorded with its events, the event sequence — position by
//! position. See `golden_replay.rs` for the original corpus and the
//! `corpus_*.rs` modules for the matrices.
//!
//! # The dimensions of an effect trace
//!
//! What an agent program can ask the bus, and how it is served. Every
//! matrix module prunes this table and says why.
//!
//! | axis | values |
//! |---|---|
//! | completion transport | unary · streamed, events dropped · streamed, events kept · streamed on a delta wire (tool names and arguments as deltas) |
//! | tool shape | none · one call then answer · two calls in one turn · two turns · zero-arg tool · a tool that errors |
//! | tool id wire | provider id (anthropic) · id-less, minted `tool-<n>` (gemini) · dual `call_id`/`item_id` (openai) |
//! | serving | `serial_per_handler` false · true; `tool_concurrency` 1 · 2; capacities default · 1 |
//! | memory | none · `Load` + `Append` · `Load` of an empty conversation · `Clear` from a hook (after the load, or after the append) · two runs in one log · explicit history (bypassed) · a `Load` or `Append` that fails |
//! | retrieval | none · `dynamic_context(n, index)` (`TopN`) · `retrieved_tools(n, index, toolset)` (`TopNIds`) · both |
//! | embedding, rerank, custom | never dispatched by the agent itself: an index embeds its query inside the handler (`RetrieveAdapter`), and nothing in `rig-agent` reranks; a hook dispatches `Embed`, `Rerank` (a mock reranker: no keyed cassette suite) and a host's `Custom<E>` over the host's bus (Matrices I, N, O) |
//! | hooks | none · observe-only · `on_dispatch` → `Patch` · `Deny` · `on_outcome` → `Replace` · `on_invalid_tool_call` → `Retry` · `Repair` · `Skip` · `on_completion_call` → request patch · a hook that dispatches through `HookContext` · a stop at every point (`on_run_start`, `on_model_select`, `on_completion_call`, `on_dispatch`, `on_outcome`, `on_model_turn`, a delta) |
//! | model routing | one model · `model_route` with `on_model_select` choosing the other |
//! | output | text · `output_schema` under `Native` · `Tool` (the output tool's call, reprompted when missing or incomplete) · `Prompted` |
//! | bus ownership | own bus (`bus` in the header) · a host's bus via `over_bus` (`bus: None`) |
//! | run continuation | one run · serialize after the first tool turn, resume on the same replay bus (every kind of program: hooks, memory, a committed output tool, a route, an ignored call, a streamed head) |
//! | per-turn shaping | none · a request patch on one turn (`tool_choice`, `extra_context`, `preamble`, `max_tokens`, `additional_params`, `active_tools`, `history`) · patches merged from several hooks · a route on the first turn · a route registered after build |
//! | hook identity | the type name · a name the hook gives itself (`AgentHook::name`) |
//! | causality | a consumer's own dispatch · nested from a tool's `Serve` through its sink's dispatcher (depth 1 · 2) · from a detached sink's resolver · from a spawned thread; the target another key · the same key (refused under serial serving, served under concurrent); the parent answered · cancelled with the child in flight · cancelled with the child queued (Matrix Q) |
//! | interpreters | the bus engine · the hand driver · the resumed engine · the Bevy host replaying the log as a script |
//! | outcome kind | success · `Cancelled` · handler error (`ErrorReport`) · a divergence (refused) |
//! | invalid call | none · unary, resolved by a hook · streamed, resolved mid-stream · unresolved under `Fail` · under `Ignore` |
//!
//! # What the original ten goldens cover
//!
//! Unary completion; memory load and append; a streamed turn with events
//! and one tool; two tools in one turn under serial serving; two tool-call
//! turns on id-less and on dual-id wires; an invalid call retried by a
//! hook; a consumer cancel. They do not cover retrieval, a hook that
//! patches, denies or replaces, model routing, structured output, a host's
//! bus, a resumed run, a handler error, `tool_choice`, `max_tokens`,
//! `additional_params`, static context, an appended or absent preamble, or
//! a prior history. Those are the matrices.

#![allow(dead_code)] // every test target uses a different subset

use std::time::Duration;

use futures::StreamExt;
use rig_agent::agent::{
    ModelSelection, ModelSelectionAction, ObservationAction, ReasoningDelta, TextDelta,
    ToolCallDelta,
};
use rig_agent::bus::{Bus, Dispatcher, MemoryHandle, ModelHandle, ToolHandle};
use rig_agent::run::{OutputMode, UnhandledInvalidToolCall};
use rig_agent::{
    AgentBuilder, AgentHook, HookContext,
    agent::{
        CompletionCallAction, CompletionCallEvent, DispatchAction, DispatchEvent, ModelTurnAction,
        ModelTurnFinished, MultiTurnStreamItem, OutcomeAction, OutcomeEvent, RequestPatch,
        RetryRequest, RunSettled, RunStart, RunStartAction, StepEventKind, StreamingError,
    },
    completion::PromptError,
    run::{
        AgentRun, AgentRunStep, InvalidToolCallAction, InvalidToolCallContext, ModelTurn,
        ModelTurnOutcome, PendingToolCall, PromptResponse, RunSpec, StreamedResolution,
        StreamedTurnAssembler, StreamedTurnEvent, prepare_request,
    },
    tool::{RegisteredTool, server::ToolServer},
};
use rig_core::{
    completion::{CompletionRequestBuilder, Document},
    effect::{EffectFamily, EffectRecord, HandlerKey, MemoryOutcome},
    error::ErrorKind,
    id::ConversationId,
    message::ToolChoice,
    message::{AssistantContent, Message, UserContent},
    tool::{ToolContext, ToolOutput},
    transcript::tool_result_output,
};
use rig_effect_log::{Checkpoint, EffectLog, EffectLogRecorder, EffectLogReplayer, RequestCheck};

/// A hook the producer added, by type: the header names hooks by their
/// type's last path segment, so the replay's hook is a type of the same
/// name (defined here) making the same decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hook {
    /// `on_invalid_tool_call` → retry with feedback naming `add`.
    RetryUnknownTool,
    /// Observes every family; decides nothing.
    ObserveEverything,
    /// `on_dispatch` → `Patch`: `add` runs with `{"x":40,"y":2}`.
    PatchAddArgs,
    /// `on_dispatch` → `Deny` (skip) for `add`.
    DenyAdd,
    /// `on_outcome` → `Replace`: the model sees `99` for `add`.
    ReplaceAddResult,
    /// `on_outcome` → `Replace`: a text answer becomes `REPLACED`.
    ReplaceAnswer,
    /// `on_completion_call` → a request patch overriding the preamble.
    PreambleOverride,
    /// `on_model_turn_finished` → retry with feedback until `DONE`.
    DemandDone,
    /// `on_run_start` dispatches `add(1, 2)` through the run's bus.
    LookupBeforeRun,
    /// `on_model_select` → `Select(fast)` on every turn after the first.
    RouteAfterFirstTurn,
    /// `on_run_start` → `Stop`.
    StopAtStart,
    /// `on_model_select` → `Stop`.
    StopAtModelSelect,
    /// `on_completion_call` → `Stop`.
    StopAtCompletionCall,
    /// `on_dispatch` → `Deny(Cancelled)` for `add`.
    CancelAddDispatch,
    /// `on_outcome` → `Replace(Err(Cancelled))` for `add`'s result.
    CancelAddOutcome,
    /// `on_outcome` → `Replace(Err(Cancelled))` on a text answer.
    CancelAnswer,
    /// `on_model_turn_finished` → `Stop` on every turn.
    StopAfterTurn,
    /// `on_model_turn_finished` → `Stop` at the turn with no tool call.
    StopAtAnswer,
    /// `on_text_delta` → `Stop`.
    StopOnTextDelta,
    /// `on_tool_call_delta` → `Stop`.
    StopOnToolCallDelta,
    /// `on_reasoning_delta` → `Stop`.
    StopOnReasoningDelta,
    /// Observes `on_run_settled`; decides nothing.
    RecordSettled,
    /// `on_invalid_tool_call` → `Repair { tool_name: "add" }`.
    RepairToAdd,
    /// `on_invalid_tool_call` → `Skip { reason }`.
    SkipUnknown,
    /// `on_run_start` → a host note (Matrix I).
    NoteAtStart,
    /// `on_completion_call` → a host note before every completion.
    NoteAtCompletionCall,
    /// `on_outcome` → a host note after every tool answer.
    NoteAtOutcome,
    /// `on_run_settled` → a host note after the answer.
    NoteAtSettled,
    /// `on_run_start` → two host notes dispatched together.
    NoteTwice,
    /// `on_run_start` → a bind the host refuses; the run goes on.
    NoteUnserved,
    /// `on_run_start` → the prompt embedded through the host's model.
    EmbedPrompt,
    /// `on_run_start` → `Clear`, which lands after the run's `Load` (Matrix J).
    ClearAtStart,
    /// `on_run_settled` → `Clear` after the run's `Append`.
    ClearAtSettled,
    /// `on_tool_call_delta` → `Stop` on the delta naming the tool (Matrix K).
    StopOnToolNameDelta,
    /// `on_tool_call_delta` → `Stop` on the first arguments delta.
    StopOnToolArgumentsDelta,
    /// `on_completion_call` → `tool_choice: Required` on turn 1 (Matrix M).
    PatchToolChoiceRequiredFirst,
    /// `on_completion_call` → `tool_choice: None` on turn 2.
    PatchToolChoiceNoneSecond,
    /// `on_completion_call` → a context document on every turn.
    PatchExtraContext,
    /// `on_completion_call` → `max_tokens: 5` on turn 2.
    PatchMaxTokensSecond,
    /// `on_completion_call` → thinking (and temperature 1.0) on turn 2.
    PatchThinkingSecond,
    /// `on_completion_call` → the pirate preamble on turn 2.
    PatchPreambleSecond,
    /// `on_completion_call` → no tools advertised on turn 2.
    PatchActiveToolsNoneSecond,
    /// `on_completion_call` → a prior exchange as turn 1's history.
    PatchHistoryFirst,
    /// `on_model_select` → `Select("fast")` on the first turn only.
    RouteOnFirstTurn,
    /// `on_model_select` → `Select("late")` on every turn: a route
    /// registered after build.
    SelectLate,
    /// `on_model_turn_finished` → `Stop` after turn `n`, named by `n`
    /// (Matrix O: a stateful hook).
    StopAfterTurnN(usize),
    /// `on_run_start` → two documents reranked through the host's
    /// reranker.
    RerankDocs,
    /// `on_run_start` → a host note the host's layer denies; the hook sees
    /// `Denied` and the run goes on (Matrix T).
    NoteDeniedAtStart,
    /// `on_run_start` observes the history the run starts with and asserts
    /// it is the replacement a memory layer made (Matrix P).
    HistoryIsReplaced,
    /// `on_run_start` dispatches a custom effect that does not serialize:
    /// the hook sees `Request` with the serde message, nothing reaches the
    /// bus, the run goes on (Matrix T, L3).
    NoteUnserializableAtStart,
    /// `on_run_start` → `n` host notes, one after another, named by `n`
    /// (Matrix T, L2: two hundred records beside the streamed one).
    NotesAtStart(usize),
}

pub const SKIP_REASON: &str = "no such tool; skipped";

/// The builder's `output_mode`, as data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    Native,
    Tool,
    Prompted,
}

impl Output {
    pub fn mode(self) -> OutputMode {
        match self {
            Self::Native => OutputMode::Native,
            Self::Tool => OutputMode::Tool,
            Self::Prompted => OutputMode::Prompted,
        }
    }
}

/// The runner's policy for an invalid call no hook resolves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unhandled {
    /// `UnhandledInvalidToolCall::Fail`: the run fails at the record.
    Fail,
    /// `UnhandledInvalidToolCall::Ignore`: the call is dropped, the run
    /// goes on.
    Ignore,
}

pub const STOP_AT_START: &str = "stopped at run start";
pub const STOP_AT_MODEL_SELECT: &str = "stopped at model selection";
pub const STOP_AT_COMPLETION_CALL: &str = "stopped before the completion call";
pub const CANCEL_ADD_DISPATCH: &str = "add is cancelled before the bus";
pub const CANCEL_ADD_OUTCOME: &str = "add is cancelled after the bus";
pub const CANCEL_ANSWER: &str = "the answer is cancelled";
pub const STOP_AFTER_TURN: &str = "stopped after the model turn";
pub const STOP_AT_ANSWER: &str = "stopped at the answer turn";
pub const STOP_ON_TEXT_DELTA: &str = "stopped on the first text delta";
pub const STOP_ON_TOOL_CALL_DELTA: &str = "stopped on the first tool-call delta";
pub const STOP_ON_REASONING_DELTA: &str = "stopped on the first reasoning delta";

pub const PIRATE_PREAMBLE: &str = "You are a pirate. Answer in one short sentence.";
pub const DENY_REASON: &str = "add is disabled for this run";
pub const REPLACED_RESULT: &str = "99";
pub const REPLACED_ANSWER: &str = "REPLACED";
pub const DONE_FEEDBACK: &str = "End your answer with the word DONE.";
pub const LOOKUP_ARGS: &str = r#"{"x":1,"y":2}"#;
pub const LOOKUP_KEY: &str = "golden/tool:add#0";
pub const PATCHED_ARGS: &str = r#"{"x":40,"y":2}"#;
pub const ROUTE: &str = "fast";

/// How the producer's run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ending {
    /// A final answer, the last completion's text.
    Answer,
    /// `PromptError::MaxTurnsError`: the model-call budget ran out with the
    /// model still calling tools (a per-run `tool_choice` that forces a
    /// call does this). Every record is a success; the run is not.
    MaxTurns,
    /// `PromptError::Report` of kind `ProviderResponse`: the completion
    /// record's outcome is the provider's error and the run fails at it.
    ProviderError,
    /// `PromptError::UnknownToolCall`: the model called a tool the program
    /// does not advertise and no hook resolved it; the run fails at the
    /// completion record.
    UnknownToolCall,
    /// `PromptError::MemoryError`: the conversation's `Load` failed; the
    /// run fails at the memory record before any completion.
    MemoryError,
    /// `PromptError::PromptCancelled` with this reason: a hook stopped the
    /// run. The records are those the engine made before the stop.
    Cancelled(&'static str),
    /// `PromptError::Report` (or a stream's `Report` item) of this kind: a
    /// layer denied or replaced what the run needed (Matrices P and T).
    Failed(ErrorKind),
}

/// The producer's tool choice, as data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    Auto,
    None,
    Required,
    Specific(&'static str),
}

impl Choice {
    pub fn tool_choice(self) -> ToolChoice {
        match self {
            Self::Auto => ToolChoice::Auto,
            Self::None => ToolChoice::None,
            Self::Required => ToolChoice::Required,
            Self::Specific(name) => ToolChoice::Specific {
                function_names: vec![name.to_owned()],
            },
        }
    }
}

/// One golden's program: what the producing root test built, verbatim.
pub struct Program {
    pub fixture: &'static str,
    pub owner: &'static str,
    /// `None` is `without_preamble()`.
    pub preamble: Option<&'static str>,
    /// `append_preamble(doc)` after `preamble`.
    pub append_preamble: Option<&'static str>,
    /// `context(doc)` static documents, in order.
    pub context: &'static [&'static str],
    pub prompt: &'static str,
    /// The history the runner was given (`history(..)`), if any.
    pub history: Option<fn() -> Vec<Message>>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub additional_params: Option<fn() -> serde_json::Value>,
    pub tool_choice: Option<Choice>,
    /// `output_schema_raw(schema)`.
    pub output_schema: Option<fn() -> serde_json::Value>,
    /// The builder's `default_max_turns`, part of the run spec the header
    /// hashes; a runner-level `max_turns` is not.
    pub default_max_turns: Option<usize>,
    pub max_turns: Option<usize>,
    pub tool_concurrency: Option<usize>,
    /// The producer ran `stream_prompt`: the model is asked for a stream.
    pub streamed: bool,
    /// The producer attached conversation memory under this id.
    pub conversation: Option<&'static str>,
    /// The producer's hooks, in registration order.
    pub hooks: &'static [Hook],
    pub invalid_retries: usize,
    /// The producer dropped the stream mid-way: the one record is a
    /// `Cancelled` completion and the run never finishes. On replay the
    /// replayer answers that record as the cancel it was — after the
    /// events it kept, if any, else as the consumer's first item.
    pub cancel_after_first_delta: bool,
    pub ending: Ending,
    /// The run's output when a hook replaced it; else the golden's last
    /// completion text.
    pub expected_output: Option<&'static str>,
    /// A second model registered as a route under this label.
    pub route: Option<&'static str>,
    /// `dynamic_context(samples, index)`: a `TopN` retrieval before every
    /// model call, its documents in the request.
    pub dynamic_context: Option<usize>,
    /// `retrieved_tools(sample, index, toolset)`: a `TopNIds` retrieval
    /// before every model call, the named tools advertised.
    pub retrieved_tools: Option<usize>,
    /// The names of the retrievable tools (advertised only when retrieved;
    /// every other tool in the required row is always advertised).
    pub retrievable: &'static [&'static str],
    /// The runner's `unhandled_invalid_tool_call` policy.
    pub unhandled: Unhandled,
    /// `output_mode(mode)`; `None` is the builder's `Auto`.
    pub output_mode: Option<Output>,
    /// A second `prompt` on the same agent after the first run settles:
    /// two runs, one log (Matrix J).
    pub second_prompt: Option<&'static str>,
    /// A route the producer registered after build (`register_model`):
    /// served under `<owner>/model:<label>` and in the handler table, but
    /// not in the required row, so the replay registers its replayer on
    /// the bus before the agent is built rather than through the builder
    /// (Matrix M).
    pub late_route: Option<&'static str>,
    /// The `lookup` tool nests a dispatch through its sink's dispatcher
    /// (Matrix Q): what it dispatches, and from where.
    pub nesting: Option<Nesting>,
    /// The host's bus serves serially (a host-bus golden does not name
    /// its policy; the replay's host runs the producer's).
    pub host_serial: bool,
    /// The producer dropped the run once a host handler signalled it was
    /// reached — the never-answering handler (Matrix Q: the tool call and
    /// its child are cancelled together) or the suspending layer's world
    /// (Matrix P: the tool call is cancelled mid-suspend); the run never
    /// finishes.
    pub cancel_when_reached: bool,
    /// The layers the producer registered around the program's handlers,
    /// as the header names them (Matrix P): per key, outermost first.
    pub layers: &'static [LayerSpec],
}

/// A layer the producer wrapped around one of the program's handlers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerSpec {
    pub at: LayerAt,
    pub layer: LayerKind,
}

/// The key a layer wraps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerAt {
    /// The agent's `add` tool (`<owner>/tool:add#0`).
    Tool,
    /// The agent's model key.
    Model,
    /// The agent's memory key.
    Memory,
    /// The host's note handler (`host/note`).
    Note,
}

/// The layers of the corpus, by name: each is a hand-written `Intercept`
/// in this module (and, verbatim, in the producers' `goldens.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// `before` → `deny(DENY_REASON)` for `add`: the hook `DenyAdd` as a layer.
    DenyAdd,
    /// `before` → `Patch` of `add`'s arguments to `PATCHED_ARGS`: `PatchAddArgs` as a layer.
    PatchAddArgs,
    /// `after` → `Replace` of `add`'s output with `REPLACED_RESULT`: `ReplaceAddResult` as a layer.
    ReplaceAddResult,
    /// `before` → `Patch` of `add`'s arguments to `PATCHED_AGAIN_ARGS` (the host's own policy).
    PatchAgain,
    /// `before` suspends until the world answers as `Answer` says.
    Approval(Answer),
    /// `before` → `Patch` of another family: `Internal`, no dispatch.
    WrongFamily,
    /// `after` → `Replace(Err(Cancelled))` on a completion.
    CancelStream,
    /// `after` → `Replace` of a memory `Load`'s answer with the bypass history.
    ReplaceLoad,
    /// `before` → `deny(HOST_DENY_REASON)` for everything.
    DenyAll,
}

/// What the world answers a suspended dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Answer {
    Approve,
    Deny,
    /// Never: the world signals it was asked and holds the channel; the
    /// producer drops the run (`cancel_when_reached`).
    Never,
}

pub const PATCHED_AGAIN_ARGS: &str = r#"{"x":30,"y":12}"#;
pub const HOST_DENY_REASON: &str = "denied by the host";
pub const WORLD_DENY_REASON: &str = "blocked by the world";
pub const CANCEL_STREAM_REASON: &str = "the answer is cancelled by a layer";

/// What the `lookup` tool dispatches from inside its own service, through
/// the dispatcher its sink carries (`rig_agent::bus::SinkDispatch`), so the child
/// record names the tool's record as its parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nesting {
    pub child: NestedChild,
    /// The nested dispatch is made from a spawned OS thread, blocking on
    /// its first poll — the case a thread-keyed re-entrancy check could
    /// not see. Serial programs only: the refusal is decided at the send.
    pub from_thread: bool,
    /// The tool detaches its sink and a spawned task answers, dispatching
    /// the child through the detached sink's dispatcher.
    pub detached: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NestedChild {
    /// A completion on the agent's model key: the question in the args.
    Completion,
    /// A host note (`host/note`).
    Note,
    /// The tool's own key, with `leaf: true` so the child does not nest
    /// again: refused under serial serving, served under concurrent.
    Same,
    /// The host's relay (`host/relay`), which itself dispatches a note:
    /// a chain of three.
    Relay,
    /// The host's handler that never answers (`host/never`): the child is
    /// in flight when the run is dropped.
    Never,
    /// Two dispatches to `host/never` at once: under a serial host the
    /// second is queued behind the first when the run is dropped.
    NeverTwice,
}

pub const NESTING: Nesting = Nesting {
    child: NestedChild::Completion,
    from_thread: false,
    detached: false,
};

impl Program {
    pub const DEFAULT: Program = Program {
        fixture: "",
        owner: "golden",
        preamble: Some(""),
        append_preamble: None,
        context: &[],
        prompt: "",
        history: None,
        temperature: None,
        max_tokens: None,
        additional_params: None,
        tool_choice: None,
        output_schema: None,
        default_max_turns: None,
        max_turns: None,
        tool_concurrency: None,
        streamed: false,
        conversation: None,
        hooks: &[],
        invalid_retries: 0,
        cancel_after_first_delta: false,
        ending: Ending::Answer,
        expected_output: None,
        route: None,
        dynamic_context: None,
        retrieved_tools: None,
        retrievable: &[],
        unhandled: Unhandled::Fail,
        output_mode: None,
        second_prompt: None,
        late_route: None,
        nesting: None,
        host_serial: false,
        cancel_when_reached: false,
        layers: &[],
    };

    /// The preamble the run spec holds: the builder's, with any appended
    /// document.
    fn spec_preamble(&self) -> Option<String> {
        let base = self.preamble.map(str::to_owned);
        match self.append_preamble {
            Some(doc) => Some(format!("{}\n{doc}", base.unwrap_or_default())),
            None => base,
        }
    }

    fn static_context(&self) -> Vec<Document> {
        self.context
            .iter()
            .enumerate()
            .map(|(n, text)| Document {
                id: format!("static_doc_{n}"),
                text: (*text).to_owned(),
                additional_props: Default::default(),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The hooks, by name.

/// The producer's hook, verbatim (`tests/common/goldens.rs`): the header
/// names it by type name, so the replay's hook is the same type.
struct RetryUnknownTool;

impl AgentHook for RetryUnknownTool {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(retry_feedback(&context.tool_name))
    }
}

pub fn retry_feedback(tool_name: &str) -> InvalidToolCallAction {
    InvalidToolCallAction::Retry {
        feedback: format!("there is no tool named {tool_name}; use add"),
    }
}

struct ObserveEverything;

impl AgentHook for ObserveEverything {
    fn observes(&self, _kind: StepEventKind) -> bool {
        true
    }
}

pub struct PatchAddArgs;

impl AgentHook for PatchAddArgs {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name() == Some("add") {
            DispatchAction::rewrite_tool_args(event.kind, serde_json::json!({"x": 40, "y": 2}))
        } else {
            DispatchAction::proceed()
        }
    }
}

struct DenyAdd;

impl AgentHook for DenyAdd {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name() == Some("add") {
            DispatchAction::skip(DENY_REASON)
        } else {
            DispatchAction::proceed()
        }
    }
}

struct ReplaceAddResult;

impl AgentHook for ReplaceAddResult {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if event.tool_name() == Some("add") && event.tool_result().is_some() {
            OutcomeAction::rewrite_tool_result(&event, REPLACED_RESULT)
        } else {
            OutcomeAction::proceed()
        }
    }
}

struct ReplaceAnswer;

impl AgentHook for ReplaceAnswer {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let Some(response) = event.completion() else {
            return OutcomeAction::proceed();
        };
        if response
            .choice
            .iter()
            .any(|content| matches!(content, AssistantContent::ToolCall(_)))
        {
            return OutcomeAction::proceed();
        }
        let mut replacement = response.clone();
        replacement.choice = vec![AssistantContent::text(REPLACED_ANSWER)];
        OutcomeAction::replace(Ok(rig_core::effect::Outcome::Completion(replacement)))
    }
}

struct PreambleOverride;

impl AgentHook for PreambleOverride {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        CompletionCallAction::patch(RequestPatch::new().preamble(PIRATE_PREAMBLE))
    }
}

struct DemandDone;

impl AgentHook for DemandDone {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if answer_text(event.content).contains("DONE") {
            ModelTurnAction::continue_run()
        } else {
            ModelTurnAction::retry_with_feedback(DONE_FEEDBACK)
        }
    }
}

struct LookupBeforeRun;

impl AgentHook for LookupBeforeRun {
    async fn on_run_start(&self, ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        let tool = ctx
            .tool(&HandlerKey::from(LOOKUP_KEY))
            .expect("the run's bus serves add");
        let answer = tool
            .dispatch(rig_core::effect::ToolCallRequest {
                name: "add".to_owned(),
                args: LOOKUP_ARGS.to_owned(),
            })
            .await
            .expect("add answers");
        assert_eq!(answer.output().render(), "3");
        RunStartAction::continue_run()
    }
}

struct RouteAfterFirstTurn;

impl AgentHook for RouteAfterFirstTurn {
    fn on_model_select(
        &self,
        _ctx: &HookContext,
        event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        if event.previous_model.is_some() {
            ModelSelectionAction::select(ROUTE)
        } else {
            ModelSelectionAction::continue_run()
        }
    }
}

struct StopAtStart;
impl AgentHook for StopAtStart {
    async fn on_run_start(&self, _ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        RunStartAction::stop(STOP_AT_START)
    }
}

struct StopAtModelSelect;
impl AgentHook for StopAtModelSelect {
    fn on_model_select(
        &self,
        _ctx: &HookContext,
        _event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        ModelSelectionAction::stop(STOP_AT_MODEL_SELECT)
    }
}

struct StopAtCompletionCall;
impl AgentHook for StopAtCompletionCall {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        CompletionCallAction::stop(STOP_AT_COMPLETION_CALL)
    }
}

struct CancelAddDispatch;
impl AgentHook for CancelAddDispatch {
    async fn on_dispatch(&self, _ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        if event.tool_name() == Some("add") {
            DispatchAction::stop(CANCEL_ADD_DISPATCH)
        } else {
            DispatchAction::proceed()
        }
    }
}

struct CancelAddOutcome;
impl AgentHook for CancelAddOutcome {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if event.tool_name() == Some("add") && event.tool_result().is_some() {
            OutcomeAction::stop(CANCEL_ADD_OUTCOME)
        } else {
            OutcomeAction::proceed()
        }
    }
}

struct CancelAnswer;
impl AgentHook for CancelAnswer {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        match event.completion() {
            Some(response)
                if !response
                    .choice
                    .iter()
                    .any(|c| matches!(c, AssistantContent::ToolCall(_))) =>
            {
                OutcomeAction::stop(CANCEL_ANSWER)
            }
            _ => OutcomeAction::proceed(),
        }
    }
}

struct StopAfterTurn;
impl AgentHook for StopAfterTurn {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        ModelTurnAction::stop(STOP_AFTER_TURN)
    }
}

struct StopAtAnswer;
impl AgentHook for StopAtAnswer {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if event
            .content
            .iter()
            .any(|c| matches!(c, AssistantContent::ToolCall(_)))
        {
            ModelTurnAction::continue_run()
        } else {
            ModelTurnAction::stop(STOP_AT_ANSWER)
        }
    }
}

struct StopOnTextDelta;
impl AgentHook for StopOnTextDelta {
    async fn on_text_delta(&self, _ctx: &HookContext, _event: TextDelta<'_>) -> ObservationAction {
        ObservationAction::stop(STOP_ON_TEXT_DELTA)
    }
}

struct StopOnToolCallDelta;
impl AgentHook for StopOnToolCallDelta {
    async fn on_tool_call_delta(
        &self,
        _ctx: &HookContext,
        _event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        ObservationAction::stop(STOP_ON_TOOL_CALL_DELTA)
    }
}

struct StopOnReasoningDelta;
impl AgentHook for StopOnReasoningDelta {
    async fn on_reasoning_delta(
        &self,
        _ctx: &HookContext,
        _event: ReasoningDelta<'_>,
    ) -> ObservationAction {
        ObservationAction::stop(STOP_ON_REASONING_DELTA)
    }
}

/// The producer's settled observer, by name; observes nothing here.
struct RecordSettled;
impl AgentHook for RecordSettled {}

struct RepairToAdd;
impl AgentHook for RepairToAdd {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        _context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(InvalidToolCallAction::repair("add"))
    }
}

struct SkipUnknown;
impl AgentHook for SkipUnknown {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        _context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(InvalidToolCallAction::skip(SKIP_REASON))
    }
}

fn answer_text(content: &[AssistantContent]) -> String {
    content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// The builder with `hooks` added in order, by name.
pub fn with_hooks<S>(builder: AgentBuilder<S>, hooks: &[Hook]) -> AgentBuilder<S> {
    add_hooks(builder, hooks)
}

// ---------------------------------------------------------------------------
// Matrix I: a host's own effect, the same type the producer dispatched.

/// The host's key for its custom handler.
pub const NOTE_KEY: &str = "host/note";
/// The host's key for its embedding model.
pub const EMBED_KEY: &str = "host/embed";

/// A host-defined effect: a note of where in the run it was taken. The
/// producer's type of the same kind label; the payload is data.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Note {
    pub at: String,
}

/// The host's answer to a [`Note`].
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct NoteAck {
    pub accepted: bool,
    pub at: String,
}

impl rig_core::effect::CustomEffect for Note {
    const KIND: &'static str = "corpus:note";
    type Answer = NoteAck;
}

pub fn note_key() -> rig_core::effect::Key<rig_core::effect::family::Custom<Note>> {
    rig_core::effect::Key::new_unchecked(HandlerKey::from(NOTE_KEY))
}

pub fn embed_key() -> rig_core::effect::Key<rig_core::effect::family::Embed> {
    rig_core::effect::Key::new_unchecked(HandlerKey::from(EMBED_KEY))
}

async fn take_note(ctx: &HookContext, at: &str) {
    let host = ctx.bind(&note_key()).expect("the host serves notes");
    let ack = host
        .dispatch(Note { at: at.to_owned() })
        .await
        .expect("the host acknowledges");
    assert!(ack.accepted && ack.at == at, "{ack:?}");
}

struct NoteAtStart;

impl AgentHook for NoteAtStart {
    async fn on_run_start(&self, ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        take_note(ctx, "start").await;
        RunStartAction::continue_run()
    }
}

struct NoteAtCompletionCall;

impl AgentHook for NoteAtCompletionCall {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        take_note(ctx, "completion_call").await;
        CompletionCallAction::Continue
    }
}

struct NoteAtOutcome;

impl AgentHook for NoteAtOutcome {
    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        if event.kind.family() == EffectFamily::Tool {
            take_note(ctx, "outcome").await;
        }
        OutcomeAction::proceed()
    }
}

struct NoteAtSettled;

impl AgentHook for NoteAtSettled {
    async fn on_run_settled(&self, ctx: &HookContext, _event: RunSettled<'_>) {
        take_note(ctx, "settled").await;
    }
}

struct NoteTwice;

impl AgentHook for NoteTwice {
    async fn on_run_start(&self, ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        let host = ctx.bind(&note_key()).expect("the host serves notes");
        let first = host.dispatch(Note {
            at: "first".to_owned(),
        });
        let second = host.dispatch(Note {
            at: "second".to_owned(),
        });
        let (first, second) = futures::join!(first, second);
        assert_eq!(first.expect("acknowledged").at, "first");
        assert_eq!(second.expect("acknowledged").at, "second");
        RunStartAction::continue_run()
    }
}

struct NoteUnserved;

impl AgentHook for NoteUnserved {
    async fn on_run_start(&self, ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        let refused = ctx.bind(&note_key()).expect_err("no host serves notes");
        assert_eq!(
            refused.kind,
            rig_core::error::ErrorKind::HandlerUnavailable,
            "{refused:?}"
        );
        RunStartAction::continue_run()
    }
}

struct EmbedPrompt;

impl AgentHook for EmbedPrompt {
    async fn on_run_start(&self, ctx: &HookContext, event: RunStart<'_>) -> RunStartAction {
        let host = ctx.bind(&embed_key()).expect("the host serves embeddings");
        let text = event.prompt.rag_text().expect("a text prompt");
        let outputs = host
            .dispatch(rig_core::effect::EmbedInputs::Texts(vec![text]))
            .await
            .expect("the host embeds");
        match outputs {
            rig_core::effect::EmbedOutputs::Texts(response) => {
                assert_eq!(response.embeddings.len(), 1, "{response:?}")
            }
            rig_core::effect::EmbedOutputs::Images(_) => panic!("a text embedding"),
        }
        RunStartAction::continue_run()
    }
}

// ---------------------------------------------------------------------------
// Matrix J: memory cleared from a hook.

struct NoteDeniedAtStart;

impl AgentHook for NoteDeniedAtStart {
    async fn on_run_start(&self, ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        let host = ctx.bind(&note_key()).expect("the host serves notes");
        let report = host
            .dispatch(Note {
                at: "start".to_owned(),
            })
            .await
            .expect_err("the host's layer denies the note");
        assert_eq!(report.kind, ErrorKind::Denied, "{report:?}");
        assert_eq!(report.message, HOST_DENY_REASON);
        RunStartAction::continue_run()
    }
}

/// A host effect whose `Serialize` fails: it never has a wire form, so it
/// never reaches the bus (L3).
#[derive(Debug)]
pub struct Unserializable;

impl serde::Serialize for Unserializable {
    fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom(
            "this effect refuses to serialize",
        ))
    }
}

impl<'de> serde::Deserialize<'de> for Unserializable {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        <()>::deserialize(deserializer).map(|()| Self)
    }
}

impl rig_core::effect::CustomEffect for Unserializable {
    const KIND: &'static str = "corpus:unserializable";
    type Answer = NoteAck;
}

pub const UNSERIALIZABLE_KEY: &str = "host/unserializable";

pub fn unserializable_key()
-> rig_core::effect::Key<rig_core::effect::family::Custom<Unserializable>> {
    rig_core::effect::Key::new_unchecked(HandlerKey::from(UNSERIALIZABLE_KEY))
}

/// What the hook sees when its effect has no wire form.
pub fn assert_unserializable(report: &rig_core::error::ErrorReport) {
    assert_eq!(report.kind, ErrorKind::Request, "{report:?}");
    assert!(
        report.message.contains("did not serialize")
            && report.message.contains("refuses to serialize"),
        "{}",
        report.message
    );
}

struct NoteUnserializableAtStart;

impl AgentHook for NoteUnserializableAtStart {
    async fn on_run_start(&self, ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        let host = ctx
            .bind(&unserializable_key())
            .expect("the host serves the kind");
        let report = host
            .dispatch(Unserializable)
            .await
            .expect_err("no wire form, no dispatch");
        assert_unserializable(&report);
        RunStartAction::continue_run()
    }
}

struct NotesAtStart(usize);

impl AgentHook for NotesAtStart {
    fn name(&self) -> Option<String> {
        Some(format!("NotesAtStart({})", self.0))
    }

    async fn on_run_start(&self, ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        for n in 0..self.0 {
            take_note(ctx, &format!("start-{n}")).await;
        }
        RunStartAction::continue_run()
    }
}

struct HistoryIsReplaced;

impl AgentHook for HistoryIsReplaced {
    async fn on_run_start(&self, _ctx: &HookContext, event: RunStart<'_>) -> RunStartAction {
        assert_eq!(
            event.history,
            bypass_history(),
            "the run starts with the history the memory layer put in the Load's place"
        );
        RunStartAction::continue_run()
    }
}

/// The history the memory layer answers a `Load` with, and the one the
/// bypass program hands the runner (Matrix J): two turns naming Ada.
pub fn bypass_history() -> Vec<Message> {
    vec![
        Message::user("My name is Ada."),
        Message::assistant("Hello, Ada."),
    ]
}

// ---------------------------------------------------------------------------
// Matrix P: the layers, hand-written `Intercept`s. Program, not record:
// the producers' `goldens.rs` holds the same types verbatim.

fn is_add(kind: &rig_core::effect::EffectKind) -> bool {
    matches!(kind, rig_core::effect::EffectKind::ToolCall { name, .. } if name == "add")
}

fn patch_add(kind: &rig_core::effect::EffectKind, args: &str) -> rig_core::serve::Decision {
    match kind {
        rig_core::effect::EffectKind::ToolCall { name, .. } if name == "add" => {
            rig_core::serve::Decision::Patch(rig_core::effect::EffectKind::ToolCall {
                name: name.clone(),
                args: args.to_owned(),
            })
        }
        _ => rig_core::serve::Decision::Proceed,
    }
}

/// The hook `DenyAdd`, as a layer.
pub struct DenyAddLayer;

impl rig_core::serve::Intercept for DenyAddLayer {
    fn name(&self) -> String {
        "DenyAddLayer".to_owned()
    }
    async fn before(
        &self,
        _id: rig_core::effect::EffectId,
        kind: &rig_core::effect::EffectKind,
    ) -> rig_core::serve::Decision {
        if is_add(kind) {
            rig_core::serve::Decision::deny(DENY_REASON)
        } else {
            rig_core::serve::Decision::Proceed
        }
    }
    async fn after(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
        _outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
    ) -> rig_core::serve::Verdict {
        rig_core::serve::Verdict::Keep
    }
}

/// The hook `PatchAddArgs`, as a layer.
pub struct PatchAddArgsLayer;

impl rig_core::serve::Intercept for PatchAddArgsLayer {
    fn name(&self) -> String {
        "PatchAddArgsLayer".to_owned()
    }
    async fn before(
        &self,
        _id: rig_core::effect::EffectId,
        kind: &rig_core::effect::EffectKind,
    ) -> rig_core::serve::Decision {
        patch_add(kind, PATCHED_ARGS)
    }
    async fn after(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
        _outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
    ) -> rig_core::serve::Verdict {
        rig_core::serve::Verdict::Keep
    }
}

/// The host's own patch of `add`'s arguments, beneath the agent's.
pub struct PatchAgainLayer;

impl rig_core::serve::Intercept for PatchAgainLayer {
    fn name(&self) -> String {
        "PatchAgainLayer".to_owned()
    }
    async fn before(
        &self,
        _id: rig_core::effect::EffectId,
        kind: &rig_core::effect::EffectKind,
    ) -> rig_core::serve::Decision {
        patch_add(kind, PATCHED_AGAIN_ARGS)
    }
    async fn after(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
        _outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
    ) -> rig_core::serve::Verdict {
        rig_core::serve::Verdict::Keep
    }
}

/// The hook `ReplaceAddResult`, as a layer.
pub struct ReplaceAddResultLayer;

impl rig_core::serve::Intercept for ReplaceAddResultLayer {
    fn name(&self) -> String {
        "ReplaceAddResultLayer".to_owned()
    }
    async fn before(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
    ) -> rig_core::serve::Decision {
        rig_core::serve::Decision::Proceed
    }
    async fn after(
        &self,
        _id: rig_core::effect::EffectId,
        kind: &rig_core::effect::EffectKind,
        outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
    ) -> rig_core::serve::Verdict {
        match outcome {
            Ok(rig_core::effect::Outcome::ToolResult { result }) if is_add(kind) => {
                rig_core::serve::Verdict::Replace(Ok(rig_core::effect::Outcome::ToolResult {
                    result: result
                        .clone()
                        .with_output(ToolOutput::text(REPLACED_RESULT)),
                }))
            }
            _ => rig_core::serve::Verdict::Keep,
        }
    }
}

/// An approval gate: `before` sends the dispatch to the world and waits.
pub struct ApprovalLayer {
    pub asks: tokio::sync::mpsc::UnboundedSender<(
        rig_core::effect::EffectId,
        futures::channel::oneshot::Sender<rig_core::serve::Decision>,
    )>,
}

impl rig_core::serve::Intercept for ApprovalLayer {
    fn name(&self) -> String {
        "ApprovalLayer".to_owned()
    }
    async fn before(
        &self,
        id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
    ) -> rig_core::serve::Decision {
        let (decide, decided) = futures::channel::oneshot::channel();
        self.asks.send((id, decide)).expect("the world listens");
        match decided.await {
            Ok(decision) => decision,
            Err(futures::channel::oneshot::Canceled) => {
                rig_core::serve::Decision::Deny(rig_core::error::ErrorReport::new(
                    ErrorKind::Internal,
                    "layer `ApprovalLayer`: the world closed the answer channel without deciding",
                ))
            }
        }
    }
    async fn after(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
        _outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
    ) -> rig_core::serve::Verdict {
        rig_core::serve::Verdict::Keep
    }
}

/// A patch of another family: never a dispatch.
pub struct WrongFamilyLayer;

impl rig_core::serve::Intercept for WrongFamilyLayer {
    fn name(&self) -> String {
        "WrongFamilyLayer".to_owned()
    }
    async fn before(
        &self,
        _id: rig_core::effect::EffectId,
        kind: &rig_core::effect::EffectKind,
    ) -> rig_core::serve::Decision {
        if is_add(kind) {
            rig_core::serve::Decision::Patch(rig_core::effect::EffectKind::Custom {
                kind: std::sync::Arc::from("corpus:wrong"),
                payload: serde_json::Value::Null,
            })
        } else {
            rig_core::serve::Decision::Proceed
        }
    }
    async fn after(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
        _outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
    ) -> rig_core::serve::Verdict {
        rig_core::serve::Verdict::Keep
    }
}

/// `after` on a completion → the answer is cancelled.
pub struct CancelStreamLayer;

impl rig_core::serve::Intercept for CancelStreamLayer {
    fn name(&self) -> String {
        "CancelStreamLayer".to_owned()
    }
    async fn before(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
    ) -> rig_core::serve::Decision {
        rig_core::serve::Decision::Proceed
    }
    async fn after(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
        _outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
    ) -> rig_core::serve::Verdict {
        rig_core::serve::Verdict::Replace(Err(rig_core::error::ErrorReport::new(
            ErrorKind::Cancelled,
            CANCEL_STREAM_REASON,
        )))
    }
}

/// `after` on a memory `Load` → the bypass history in the store's place.
pub struct ReplaceLoadLayer;

impl rig_core::serve::Intercept for ReplaceLoadLayer {
    fn name(&self) -> String {
        "ReplaceLoadLayer".to_owned()
    }
    async fn before(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
    ) -> rig_core::serve::Decision {
        rig_core::serve::Decision::Proceed
    }
    async fn after(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
        outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
    ) -> rig_core::serve::Verdict {
        match outcome {
            Ok(rig_core::effect::Outcome::Memory(MemoryOutcome::Loaded { .. })) => {
                rig_core::serve::Verdict::Replace(Ok(rig_core::effect::Outcome::Memory(
                    MemoryOutcome::Loaded {
                        messages: bypass_history(),
                    },
                )))
            }
            _ => rig_core::serve::Verdict::Keep,
        }
    }
}

/// The host denies everything on the key.
pub struct DenyAllLayer;

impl rig_core::serve::Intercept for DenyAllLayer {
    fn name(&self) -> String {
        "DenyAllLayer".to_owned()
    }
    async fn before(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
    ) -> rig_core::serve::Decision {
        rig_core::serve::Decision::deny(HOST_DENY_REASON)
    }
    async fn after(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
        _outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
    ) -> rig_core::serve::Verdict {
        rig_core::serve::Verdict::Keep
    }
}

/// The world the suspending layers ask: answers as the program says, and
/// signals `reached` when it holds an answer forever.
pub type Asks = tokio::sync::mpsc::UnboundedSender<(
    rig_core::effect::EffectId,
    futures::channel::oneshot::Sender<rig_core::serve::Decision>,
)>;

pub fn spawn_world(answer: Answer, reached: std::sync::Arc<tokio::sync::Notify>) -> Asks {
    let (asks, mut asked): (Asks, _) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Some((_, decide)) = asked.recv().await {
            match answer {
                Answer::Approve => {
                    let _ = decide.send(rig_core::serve::Decision::Proceed);
                }
                Answer::Deny => {
                    let _ = decide.send(rig_core::serve::Decision::deny(WORLD_DENY_REASON));
                }
                Answer::Never => {
                    reached.notify_one();
                    held.push(decide);
                }
            }
        }
    });
    asks
}

/// `handler` under the program's layers at `at`, outermost first as the
/// header names them (so wrapped innermost first).
pub fn layered(
    handler: rig_core::serve::ErasedHandler,
    program: &Program,
    at: LayerAt,
    world: &Option<Asks>,
) -> rig_core::serve::ErasedHandler {
    let mut handler = handler;
    for spec in program.layers.iter().rev().filter(|spec| spec.at == at) {
        handler = match spec.layer {
            LayerKind::DenyAdd => handler.layered(DenyAddLayer),
            LayerKind::PatchAddArgs => handler.layered(PatchAddArgsLayer),
            LayerKind::ReplaceAddResult => handler.layered(ReplaceAddResultLayer),
            LayerKind::PatchAgain => handler.layered(PatchAgainLayer),
            LayerKind::Approval(_) => handler.layered(ApprovalLayer {
                asks: world.clone().expect("a world for the suspending layer"),
            }),
            LayerKind::WrongFamily => handler.layered(WrongFamilyLayer),
            LayerKind::CancelStream => handler.layered(CancelStreamLayer),
            LayerKind::ReplaceLoad => handler.layered(ReplaceLoadLayer),
            LayerKind::DenyAll => handler.layered(DenyAllLayer),
        };
    }
    handler
}

/// The names the header lists for the program's layers: the handler
/// table's order (by key), outermost first within a key.
pub fn layer_names(program: &Program, owner: &str) -> Vec<String> {
    let key_of = |at: LayerAt| match at {
        LayerAt::Tool => format!("{owner}/tool:add#0"),
        LayerAt::Model => format!("{owner}/model:default"),
        LayerAt::Memory => format!("{owner}/memory"),
        LayerAt::Note => NOTE_KEY.to_owned(),
    };
    let mut keyed: Vec<(String, &LayerSpec)> = program
        .layers
        .iter()
        .map(|spec| (key_of(spec.at), spec))
        .collect();
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    keyed
        .into_iter()
        .map(|(_, spec)| match spec.layer {
            LayerKind::DenyAdd => "DenyAddLayer",
            LayerKind::PatchAddArgs => "PatchAddArgsLayer",
            LayerKind::ReplaceAddResult => "ReplaceAddResultLayer",
            LayerKind::PatchAgain => "PatchAgainLayer",
            LayerKind::Approval(_) => "ApprovalLayer",
            LayerKind::WrongFamily => "WrongFamilyLayer",
            LayerKind::CancelStream => "CancelStreamLayer",
            LayerKind::ReplaceLoad => "ReplaceLoadLayer",
            LayerKind::DenyAll => "DenyAllLayer",
        })
        .map(str::to_owned)
        .collect()
}

// ---------------------------------------------------------------------------
// Matrix Q: the nesting tool and the host's relay and never-answering
// handlers. Program, not record: the producer's copies live in
// `tests/common/goldens.rs`; both interpreters register these.

pub const NESTING_TOOL_KEY: &str = "golden/tool:lookup#0";
pub const RELAY_KEY: &str = "host/relay";
pub const NEVER_KEY: &str = "host/never";
pub const NESTED_PREAMBLE: &str = "Answer in one word.";

/// The relay's effect: it takes a note on the host's behalf.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct RelayNote {
    pub at: String,
}

impl rig_core::effect::CustomEffect for RelayNote {
    const KIND: &'static str = "corpus:relay";
    type Answer = NoteAck;
}

/// The never-answering effect.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Hold;

impl rig_core::effect::CustomEffect for Hold {
    const KIND: &'static str = "corpus:hold";
    type Answer = NoteAck;
}

pub fn relay_key() -> rig_core::effect::Key<rig_core::effect::family::Custom<RelayNote>> {
    rig_core::effect::Key::new_unchecked(HandlerKey::from(RELAY_KEY))
}

pub fn never_key() -> rig_core::effect::Key<rig_core::effect::family::Custom<Hold>> {
    rig_core::effect::Key::new_unchecked(HandlerKey::from(NEVER_KEY))
}

/// The `lookup` tool's arguments: a question, or a leaf marker.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct LookupArgs {
    #[serde(default)]
    pub q: String,
    #[serde(default)]
    pub leaf: bool,
}

pub fn lookup_parameters() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": "The question to look up"},
            "leaf": {"type": "boolean", "description": "Answer directly, without looking further"}
        },
        "required": ["q"]
    })
}

/// A tool that dispatches from inside its own service, through the
/// dispatcher its sink carries: every child names this call as its
/// parent. Which child, and from where, is the program's `Nesting`.
pub struct Lookup {
    pub nesting: Nesting,
    pub model_key: HandlerKey,
}

fn tool_text(text: String) -> rig_core::effect::Outcome {
    rig_core::effect::Outcome::ToolResult {
        result: rig_core::tool::ToolResult::success(ToolOutput::text(text)),
    }
}

impl Lookup {
    /// The child's answer as the tool's text, dispatched through
    /// `dispatcher` (the sink's, parented by this call).
    async fn nest(&self, dispatcher: Dispatcher, args: LookupArgs) -> String {
        match self.nesting.child {
            NestedChild::Completion => {
                let model: ModelHandle = dispatcher.handle(&self.model_key).expect("the model");
                let request = CompletionRequestBuilder::unbound(args.q.as_str())
                    .preamble(NESTED_PREAMBLE.to_owned())
                    .temperature(0.0)
                    .build();
                let response = model
                    .complete(request)
                    .await
                    .expect("the nested completion");
                response
                    .choice
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::Text(text) => Some(text.text.trim().to_owned()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            NestedChild::Note => {
                let ack = dispatcher
                    .bind(&note_key())
                    .expect("the host serves notes")
                    .dispatch(Note {
                        at: "lookup".to_owned(),
                    })
                    .await
                    .expect("acknowledged");
                format!("noted:{}", ack.at)
            }
            NestedChild::Same => {
                let handle: ToolHandle = dispatcher
                    .handle(&HandlerKey::from(NESTING_TOOL_KEY))
                    .expect("the tool's own key");
                let call = handle.call("lookup", r#"{"q":"","leaf":true}"#, ToolContext::new());
                let answer = if self.nesting.from_thread {
                    // From another thread: a thread-keyed check would not
                    // see this as re-entrant; the chain does. The refusal is
                    // decided at the send, so the block returns at once.
                    std::thread::spawn(move || futures::executor::block_on(call))
                        .join()
                        .expect("the nested thread")
                } else {
                    call.await
                };
                match answer {
                    Ok(answer) => format!("served:{}", answer.result.output().render()),
                    Err(report) => format!("refused:{:?}", report.kind),
                }
            }
            NestedChild::Relay => {
                let ack = dispatcher
                    .bind(&relay_key())
                    .expect("the host serves the relay")
                    .dispatch(RelayNote {
                        at: "lookup".to_owned(),
                    })
                    .await
                    .expect("relayed");
                format!("relayed:{}", ack.at)
            }
            NestedChild::Never => {
                let held = dispatcher
                    .bind(&never_key())
                    .expect("the host holds")
                    .dispatch(Hold);
                match held.await {
                    Ok(ack) => format!("answered:{}", ack.at),
                    Err(report) => format!("failed:{:?}", report.kind),
                }
            }
            NestedChild::NeverTwice => {
                let host = dispatcher.bind(&never_key()).expect("the host holds");
                let first = host.dispatch(Hold);
                let second = host.dispatch(Hold);
                match futures::join!(first, second) {
                    (Ok(first), Ok(second)) => format!("answered:{}:{}", first.at, second.at),
                    (Err(report), _) | (_, Err(report)) => format!("failed:{:?}", report.kind),
                }
            }
        }
    }
}

impl rig_core::serve::Serve for Lookup {
    type Family = rig_core::effect::family::Tool;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: HandlerKey::from(NESTING_TOOL_KEY),
            family: rig_core::effect::FamilyDescriptor::Tool {
                name: "lookup".to_owned(),
                description: "Look a question up".to_owned(),
                parameters: lookup_parameters(),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: rig_core::effect::EffectKind, sink: rig_core::serve::OutcomeSink) {
        let rig_core::effect::EffectKind::ToolCall { args, .. } = kind else {
            sink.resolve(Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::Request,
                "a tool call",
            )))
            .await;
            return;
        };
        let args: LookupArgs = serde_json::from_str(&args).unwrap_or_default();
        if args.leaf {
            sink.resolve(Ok(tool_text("leaf".to_owned()))).await;
            return;
        }
        if self.nesting.detached {
            // Answered later, by a task holding the detached sink: the
            // child is dispatched through the detached sink's dispatcher.
            let sink = sink.detach();
            let dispatcher =
                rig_agent::bus::SinkDispatch::dispatcher(&sink).expect("a scoped sink");
            assert_eq!(dispatcher.parent(), Some(sink.id()));
            let lookup = Lookup {
                nesting: Nesting {
                    detached: false,
                    ..self.nesting
                },
                model_key: self.model_key.clone(),
            };
            tokio::spawn(async move {
                let text = lookup.nest(dispatcher, args).await;
                sink.resolve(Ok(tool_text(text))).await;
            });
            return;
        }
        let dispatcher = rig_agent::bus::SinkDispatch::dispatcher(&sink).expect("a scoped sink");
        assert_eq!(dispatcher.parent(), Some(sink.id()));
        let text = self.nest(dispatcher, args).await;
        sink.resolve(Ok(tool_text(text))).await;
    }
}

/// The host's relay: takes a note through its own sink's dispatcher and
/// answers with the acknowledgement — the middle of a chain of three.
pub struct Relay;

impl rig_core::serve::Serve for Relay {
    type Family = rig_core::effect::family::Custom<RelayNote>;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: HandlerKey::from(RELAY_KEY),
            family: rig_core::effect::FamilyDescriptor::Custom {
                kind: <RelayNote as rig_core::effect::CustomEffect>::KIND.to_owned(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: rig_core::effect::EffectKind, sink: rig_core::serve::OutcomeSink) {
        let rig_core::effect::EffectKind::Custom { payload, .. } = kind else {
            sink.resolve(Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::Request,
                "a relay note",
            )))
            .await;
            return;
        };
        let note: RelayNote = serde_json::from_value(payload).expect("a relay note");
        let dispatcher = rig_agent::bus::SinkDispatch::dispatcher(&sink).expect("a scoped sink");
        let ack = dispatcher
            .bind(&note_key())
            .expect("the host serves notes")
            .dispatch(Note {
                at: format!("relay<{}", note.at),
            })
            .await
            .expect("acknowledged");
        sink.resolve(Ok(rig_core::effect::Outcome::Custom(
            serde_json::to_value(NoteAck {
                accepted: ack.accepted,
                at: ack.at,
            })
            .expect("an ack serializes"),
        )))
        .await;
    }
}

/// The host's handler that never answers: it signals that it was reached
/// and holds the dispatch until the consumer goes.
pub struct Never {
    pub reached: std::sync::Arc<tokio::sync::Notify>,
}

impl rig_core::serve::Serve for Never {
    type Family = rig_core::effect::family::Custom<Hold>;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: HandlerKey::from(NEVER_KEY),
            family: rig_core::effect::FamilyDescriptor::Custom {
                kind: <Hold as rig_core::effect::CustomEffect>::KIND.to_owned(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _kind: rig_core::effect::EffectKind, sink: rig_core::serve::OutcomeSink) {
        self.reached.notify_one();
        futures::future::pending::<()>().await;
        drop(sink);
    }
}

/// The conversation every memory program loads and saves under.
pub const CONVERSATION: &str = "golden-conversation";

async fn clear_conversation(ctx: &HookContext) {
    let memory = ctx
        .memory(&HandlerKey::from("golden/memory"))
        .expect("the run's bus serves memory");
    let outcome = memory
        .dispatch(rig_core::effect::MemoryOp::Clear {
            conversation: ConversationId::from(CONVERSATION),
        })
        .await
        .expect("the memory clears");
    assert!(
        matches!(outcome, rig_core::effect::MemoryOutcome::Cleared),
        "{outcome:?}"
    );
}

struct ClearAtStart;

impl AgentHook for ClearAtStart {
    async fn on_run_start(&self, ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
        clear_conversation(ctx).await;
        RunStartAction::continue_run()
    }
}

struct ClearAtSettled;

impl AgentHook for ClearAtSettled {
    async fn on_run_settled(&self, ctx: &HookContext, _event: RunSettled<'_>) {
        clear_conversation(ctx).await;
    }
}

// ---------------------------------------------------------------------------
// Matrix K: stops on the delta wire.

pub const STOP_ON_TOOL_NAME_DELTA: &str = "stop on the tool's name delta";
pub const STOP_ON_TOOL_ARGUMENTS_DELTA: &str = "stop on the tool's arguments delta";

struct StopOnToolNameDelta;

impl AgentHook for StopOnToolNameDelta {
    async fn on_tool_call_delta(
        &self,
        _ctx: &HookContext,
        event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        if event.tool_name.is_some() {
            ObservationAction::stop(STOP_ON_TOOL_NAME_DELTA)
        } else {
            ObservationAction::continue_run()
        }
    }
}

struct StopOnToolArgumentsDelta;

impl AgentHook for StopOnToolArgumentsDelta {
    async fn on_tool_call_delta(
        &self,
        _ctx: &HookContext,
        event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        if event.tool_name.is_none() && !event.delta.is_empty() {
            ObservationAction::stop(STOP_ON_TOOL_ARGUMENTS_DELTA)
        } else {
            ObservationAction::continue_run()
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix M: per-turn shaping.

pub const SHAPING_CONTEXT_ID: &str = "shaping-context";
pub const SHAPING_CONTEXT: &str =
    "Definition of a glarb-glarb: an ancient farming tool from planet Jiro.";
pub const LATE_ROUTE: &str = "late";

fn shaping_document() -> Document {
    Document {
        id: SHAPING_CONTEXT_ID.to_owned(),
        text: SHAPING_CONTEXT.to_owned(),
        additional_props: Default::default(),
    }
}

/// The patch one of the corpus's hooks makes on `turn`, as the hook makes it.
pub fn hook_patch(hook: Hook, turn: usize) -> Option<RequestPatch> {
    match hook {
        Hook::PatchToolChoiceRequiredFirst => {
            (turn == 1).then(|| RequestPatch::new().tool_choice(ToolChoice::Required))
        }
        Hook::PatchToolChoiceNoneSecond => {
            (turn == 2).then(|| RequestPatch::new().tool_choice(ToolChoice::None))
        }
        Hook::PatchExtraContext => Some(RequestPatch::new().context(shaping_document())),
        Hook::PatchMaxTokensSecond => (turn == 2).then(|| RequestPatch::new().max_tokens(5)),
        Hook::PatchThinkingSecond => (turn == 2).then(|| {
            RequestPatch::new().temperature(1.0).additional_params(
                serde_json::json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } }),
            )
        }),
        Hook::PatchPreambleSecond => {
            (turn == 2).then(|| RequestPatch::new().preamble(PIRATE_PREAMBLE))
        }
        Hook::PatchActiveToolsNoneSecond => {
            (turn == 2).then(|| RequestPatch::new().active_tools(Vec::<String>::new()))
        }
        Hook::PatchHistoryFirst => (turn == 1).then(|| {
            RequestPatch::new().history(vec![
                Message::user("My name is Ada."),
                Message::assistant("Hello, Ada."),
            ])
        }),
        Hook::PreambleOverride => Some(RequestPatch::new().preamble(PIRATE_PREAMBLE)),
        Hook::RetryUnknownTool
        | Hook::ObserveEverything
        | Hook::PatchAddArgs
        | Hook::DenyAdd
        | Hook::ReplaceAddResult
        | Hook::ReplaceAnswer
        | Hook::DemandDone
        | Hook::LookupBeforeRun
        | Hook::RouteAfterFirstTurn
        | Hook::StopAtStart
        | Hook::StopAtModelSelect
        | Hook::StopAtCompletionCall
        | Hook::CancelAddDispatch
        | Hook::CancelAddOutcome
        | Hook::CancelAnswer
        | Hook::StopAfterTurn
        | Hook::StopAtAnswer
        | Hook::StopOnTextDelta
        | Hook::StopOnToolCallDelta
        | Hook::StopOnReasoningDelta
        | Hook::RecordSettled
        | Hook::RepairToAdd
        | Hook::SkipUnknown
        | Hook::NoteAtStart
        | Hook::NoteAtCompletionCall
        | Hook::NoteAtOutcome
        | Hook::NoteAtSettled
        | Hook::NoteTwice
        | Hook::NoteUnserved
        | Hook::EmbedPrompt
        | Hook::ClearAtStart
        | Hook::ClearAtSettled
        | Hook::StopOnToolNameDelta
        | Hook::StopOnToolArgumentsDelta
        | Hook::RouteOnFirstTurn
        | Hook::SelectLate
        | Hook::StopAfterTurnN(_)
        | Hook::RerankDocs
        | Hook::NoteDeniedAtStart
        | Hook::HistoryIsReplaced
        | Hook::NoteUnserializableAtStart
        | Hook::NotesAtStart(_) => None,
    }
}

/// The program's request patch for `turn`: every patching hook's, merged in
/// registration order as the hook stack merges them.
pub fn patch_for_turn(program: &Program, turn: usize) -> Option<RequestPatch> {
    program
        .hooks
        .iter()
        .filter_map(|hook| hook_patch(*hook, turn))
        .reduce(RequestPatch::merge)
}

macro_rules! patch_hook {
    ($name:ident, $hook:expr) => {
        struct $name;

        impl AgentHook for $name {
            async fn on_completion_call(
                &self,
                _ctx: &HookContext,
                event: CompletionCallEvent<'_>,
            ) -> CompletionCallAction {
                match hook_patch($hook, event.turn) {
                    Some(patch) => CompletionCallAction::patch(patch),
                    None => CompletionCallAction::Continue,
                }
            }
        }
    };
}

patch_hook!(
    PatchToolChoiceRequiredFirst,
    Hook::PatchToolChoiceRequiredFirst
);
patch_hook!(PatchToolChoiceNoneSecond, Hook::PatchToolChoiceNoneSecond);
patch_hook!(PatchExtraContext, Hook::PatchExtraContext);
patch_hook!(PatchMaxTokensSecond, Hook::PatchMaxTokensSecond);
patch_hook!(PatchThinkingSecond, Hook::PatchThinkingSecond);
patch_hook!(PatchPreambleSecond, Hook::PatchPreambleSecond);
patch_hook!(PatchActiveToolsNoneSecond, Hook::PatchActiveToolsNoneSecond);
patch_hook!(PatchHistoryFirst, Hook::PatchHistoryFirst);

struct RouteOnFirstTurn;

impl AgentHook for RouteOnFirstTurn {
    fn on_model_select(
        &self,
        _ctx: &HookContext,
        event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        if event.previous_model.is_none() {
            ModelSelectionAction::select(ROUTE)
        } else {
            ModelSelectionAction::continue_run()
        }
    }
}

struct SelectLate;

impl AgentHook for SelectLate {
    fn on_model_select(
        &self,
        _ctx: &HookContext,
        _event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        ModelSelectionAction::select(LATE_ROUTE)
    }
}

// ---------------------------------------------------------------------------
// Matrix O: a stateful hook, a host's reranker.

pub fn stop_after_turn_reason(n: usize) -> String {
    format!("stopped after turn {n}")
}

struct StopAfterTurnN(usize);

impl AgentHook for StopAfterTurnN {
    fn name(&self) -> Option<String> {
        Some(format!("StopAfterTurn({})", self.0))
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        if event.turn == self.0 {
            ModelTurnAction::stop(stop_after_turn_reason(self.0))
        } else {
            ModelTurnAction::continue_run()
        }
    }
}

pub const RERANK_KEY: &str = "host/rerank";
pub const RERANK_DOCUMENTS: [&str; 2] = ["the harbor label", "the orchard label"];

pub fn rerank_key() -> rig_core::effect::Key<rig_core::effect::family::Rerank> {
    rig_core::effect::Key::new_unchecked(HandlerKey::from(RERANK_KEY))
}

pub fn rerank_request(query: &str) -> rig_core::effect::RerankRequest {
    rig_core::effect::RerankRequest {
        query: query.to_owned(),
        documents: RERANK_DOCUMENTS
            .iter()
            .map(|doc| (*doc).to_owned())
            .collect(),
    }
}

struct RerankDocs;

impl AgentHook for RerankDocs {
    async fn on_run_start(&self, ctx: &HookContext, event: RunStart<'_>) -> RunStartAction {
        let host = ctx.bind(&rerank_key()).expect("the host serves reranking");
        let query = event.prompt.rag_text().expect("a text prompt");
        let ranked = host
            .dispatch(rerank_request(&query))
            .await
            .expect("the host reranks");
        assert_eq!(ranked.results.len(), 2, "{ranked:?}");
        RunStartAction::continue_run()
    }
}

fn add_hooks<S>(mut builder: AgentBuilder<S>, hooks: &[Hook]) -> AgentBuilder<S> {
    for hook in hooks {
        builder = match hook {
            Hook::RetryUnknownTool => builder.add_hook(RetryUnknownTool),
            Hook::ObserveEverything => builder.add_hook(ObserveEverything),
            Hook::PatchAddArgs => builder.add_hook(PatchAddArgs),
            Hook::DenyAdd => builder.add_hook(DenyAdd),
            Hook::ReplaceAddResult => builder.add_hook(ReplaceAddResult),
            Hook::ReplaceAnswer => builder.add_hook(ReplaceAnswer),
            Hook::PreambleOverride => builder.add_hook(PreambleOverride),
            Hook::DemandDone => builder.add_hook(DemandDone),
            Hook::LookupBeforeRun => builder.add_hook(LookupBeforeRun),
            Hook::RouteAfterFirstTurn => builder.add_hook(RouteAfterFirstTurn),
            Hook::StopAtStart => builder.add_hook(StopAtStart),
            Hook::StopAtModelSelect => builder.add_hook(StopAtModelSelect),
            Hook::StopAtCompletionCall => builder.add_hook(StopAtCompletionCall),
            Hook::CancelAddDispatch => builder.add_hook(CancelAddDispatch),
            Hook::CancelAddOutcome => builder.add_hook(CancelAddOutcome),
            Hook::CancelAnswer => builder.add_hook(CancelAnswer),
            Hook::StopAfterTurn => builder.add_hook(StopAfterTurn),
            Hook::StopAtAnswer => builder.add_hook(StopAtAnswer),
            Hook::StopOnTextDelta => builder.add_hook(StopOnTextDelta),
            Hook::StopOnToolCallDelta => builder.add_hook(StopOnToolCallDelta),
            Hook::StopOnReasoningDelta => builder.add_hook(StopOnReasoningDelta),
            Hook::RecordSettled => builder.add_hook(RecordSettled),
            Hook::RepairToAdd => builder.add_hook(RepairToAdd),
            Hook::SkipUnknown => builder.add_hook(SkipUnknown),
            Hook::NoteAtStart => builder.add_hook(NoteAtStart),
            Hook::NoteAtCompletionCall => builder.add_hook(NoteAtCompletionCall),
            Hook::NoteAtOutcome => builder.add_hook(NoteAtOutcome),
            Hook::NoteAtSettled => builder.add_hook(NoteAtSettled),
            Hook::NoteTwice => builder.add_hook(NoteTwice),
            Hook::NoteUnserved => builder.add_hook(NoteUnserved),
            Hook::EmbedPrompt => builder.add_hook(EmbedPrompt),
            Hook::ClearAtStart => builder.add_hook(ClearAtStart),
            Hook::ClearAtSettled => builder.add_hook(ClearAtSettled),
            Hook::StopOnToolNameDelta => builder.add_hook(StopOnToolNameDelta),
            Hook::StopOnToolArgumentsDelta => builder.add_hook(StopOnToolArgumentsDelta),
            Hook::PatchToolChoiceRequiredFirst => builder.add_hook(PatchToolChoiceRequiredFirst),
            Hook::PatchToolChoiceNoneSecond => builder.add_hook(PatchToolChoiceNoneSecond),
            Hook::PatchExtraContext => builder.add_hook(PatchExtraContext),
            Hook::PatchMaxTokensSecond => builder.add_hook(PatchMaxTokensSecond),
            Hook::PatchThinkingSecond => builder.add_hook(PatchThinkingSecond),
            Hook::PatchPreambleSecond => builder.add_hook(PatchPreambleSecond),
            Hook::PatchActiveToolsNoneSecond => builder.add_hook(PatchActiveToolsNoneSecond),
            Hook::PatchHistoryFirst => builder.add_hook(PatchHistoryFirst),
            Hook::RouteOnFirstTurn => builder.add_hook(RouteOnFirstTurn),
            Hook::SelectLate => builder.add_hook(SelectLate),
            Hook::StopAfterTurnN(n) => builder.add_hook(StopAfterTurnN(*n)),
            Hook::RerankDocs => builder.add_hook(RerankDocs),
            Hook::NoteDeniedAtStart => builder.add_hook(NoteDeniedAtStart),
            Hook::HistoryIsReplaced => builder.add_hook(HistoryIsReplaced),
            Hook::NoteUnserializableAtStart => builder.add_hook(NoteUnserializableAtStart),
            Hook::NotesAtStart(n) => builder.add_hook(NotesAtStart(*n)),
        };
    }
    builder
}

// ---------------------------------------------------------------------------
// Goldens and the oracle.

pub async fn within<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("a replay never hangs")
}

/// Where the goldens live.
pub fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

pub fn golden(fixture: &str) -> EffectLog {
    let path = format!(
        "{}/fixtures/{fixture}.effects.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("the golden fixture is committed");
    serde_json::from_str(&text).expect("the golden fixture loads")
}

/// A record as data: its kind, outcome, published tool output and events.
pub fn as_data(record: &EffectRecord) -> serde_json::Value {
    serde_json::json!({
        "key": record.key,
        "kind": record.kind,
        "outcome": record.outcome,
        "tool_output": record.tool_output,
        "events": record.events,
    })
}

/// The oracle: the replayed log is the golden, record by record, in the
/// log's order — which is dispatch order across every key (ids are minted
/// at dispatch and the recorder keeps them in that order), so a dispatch
/// at the wrong point between two keys is a divergence, not a per-key
/// match. Both logs' ids must be strictly increasing for that to hold.
pub fn assert_same_records(replayed: &EffectLog, log: &EffectLog, interpreter: &str) {
    for (name, which) in [("the golden", log), ("the replay", replayed)] {
        let ids: Vec<u64> = which.iter().map(|record| record.id.as_u64()).collect();
        assert!(
            ids.windows(2).all(|pair| pair[0] < pair[1]),
            "{interpreter}: {name}'s records are in dispatch order: {ids:?}"
        );
    }
    // Causality as data: a record's parent, if any, is an earlier record
    // of the same log, and the replay's chain is the golden's, by position.
    let parent_positions = |which: &EffectLog, name: &str| -> Vec<Option<usize>> {
        which
            .iter()
            .enumerate()
            .map(|(position, record)| {
                record.parent.map(|parent| {
                    let at = which
                        .iter()
                        .position(|candidate| candidate.id == parent)
                        .unwrap_or_else(|| {
                            panic!(
                                "{interpreter}: {name}'s record {position} names parent {parent:?}, which is not in the log"
                            )
                        });
                    assert!(
                        at < position,
                        "{interpreter}: {name}'s record {position} names a later record as its parent"
                    );
                    at
                })
            })
            .collect()
    };
    let golden_parents = parent_positions(log, "the golden");
    let replayed_parents = parent_positions(replayed, "the replay");
    let replayed: Vec<_> = replayed.iter().map(as_data).collect();
    let recorded: Vec<_> = log.iter().map(as_data).collect();
    for (position, (got, want)) in replayed_parents.iter().zip(&golden_parents).enumerate() {
        assert_eq!(
            got, want,
            "{interpreter}: record {position}'s parent differs from the golden's"
        );
    }
    for (position, (got, want)) in replayed.iter().zip(&recorded).enumerate() {
        assert_eq!(
            got, want,
            "{interpreter}: record {position} differs from the golden"
        );
    }
    assert_eq!(
        replayed.len(),
        recorded.len(),
        "{interpreter}: the golden has {} records, the replay {}",
        recorded.len(),
        replayed.len()
    );
}

pub fn keeps_events(log: &EffectLog) -> bool {
    log.iter().any(|record| record.events.is_some())
}

/// The run's output as the golden's last completion gives it: its text,
/// or — in Tool output mode — the output tool's arguments serialized as
/// the run serializes them (`final_result`, or the collision-safe name
/// the run picked).
pub fn golden_answer(log: &EffectLog) -> String {
    log.iter()
        .rev()
        .find_map(|record| match &record.outcome {
            Ok(rig_core::effect::Outcome::Completion(response)) => {
                let output_call = response.choice.iter().find_map(|content| match content {
                    AssistantContent::ToolCall(call)
                        if call.function.name.starts_with("final_result") =>
                    {
                        Some(rig_core::json_utils::serialize_json_value(
                            &call.function.arguments,
                        ))
                    }
                    _ => None,
                });
                Some(output_call.unwrap_or_else(|| {
                    response
                        .choice
                        .iter()
                        .filter_map(|content| match content {
                            AssistantContent::Text(text) => Some(text.text.clone()),
                            _ => None,
                        })
                        .collect::<String>()
                }))
            }
            _ => None,
        })
        .expect("the golden ends in a completion")
}

/// The bus, with the golden's policy, the model replayer registered and a
/// recorder attached; the tool replayers in a server the agent advertises
/// from (the required row's tools, dispatched or not).
pub struct Replay {
    pub log: EffectLog,
    pub dispatcher: Dispatcher,
    pub registrar: rig_agent::bus::Registrar,
    pub recorder: EffectLogRecorder,
    pub driver: tokio::task::JoinHandle<()>,
    pub model_key: HandlerKey,
    pub memory_key: HandlerKey,
    /// Signalled when the host's never-answering handler is reached
    /// (Matrix Q's cancelled cells) or the world holds a suspended layer's
    /// answer forever (Matrix P's).
    pub reached: Option<std::sync::Arc<tokio::sync::Notify>>,
    /// The world a suspending layer asks (Matrix P).
    pub world: Option<Asks>,
    /// How every replayer of this replay compares requests (Matrix R).
    pub check: RequestCheck,
}

impl Replay {
    pub fn open(program: &Program) -> Self {
        Self::open_checking(program, RequestCheck::Payload)
    }

    /// A replayer for `key` over `source`, comparing requests as this
    /// replay does.
    pub fn replayer(
        &self,
        source: &EffectLog,
        key: &HandlerKey,
    ) -> Result<EffectLogReplayer, rig_core::error::ErrorReport> {
        EffectLogReplayer::for_key(source, key).map(|replayer| replayer.checking(self.check))
    }

    pub fn open_checking(program: &Program, check: RequestCheck) -> Self {
        let log = golden(program.fixture);
        EffectLogReplayer::check_header(&log).expect("a current format");
        // A golden recorded over a host's bus names no policy: the host
        // sized its bus. The replay's host uses the default, or the
        // producer's where the program names it.
        let mut bus = log.header.bus.unwrap_or_default();
        if program.host_serial {
            assert!(log.header.bus.is_none(), "a host-bus program");
            bus.serial_per_handler = true;
        }
        let (dispatcher, registrar, mut driver) = Bus::channel_with(bus);
        let model_key = HandlerKey::from(format!("{}/model:default", program.owner));
        let memory_key = HandlerKey::from(format!("{}/memory", program.owner));
        // The world of a suspending layer, and what it signals.
        let mut reached = None;
        let world = program
            .layers
            .iter()
            .find_map(|spec| match spec.layer {
                LayerKind::Approval(answer) => Some(answer),
                _ => None,
            })
            .map(|answer| {
                let notify = std::sync::Arc::new(tokio::sync::Notify::new());
                reached = Some(std::sync::Arc::clone(&notify));
                spawn_world(answer, notify)
            });
        let model = EffectLogReplayer::for_key(&log, &model_key)
            .expect("the model's records")
            .checking(check);
        driver
            .register_erased(
                model_key.clone(),
                layered(
                    rig_core::serve::ErasedHandler::new(model),
                    program,
                    LayerAt::Model,
                    &world,
                ),
            )
            .expect("a fresh key");
        let recorder = if keeps_events(&log) {
            EffectLogRecorder::keeping_stream_events()
        } else {
            EffectLogRecorder::new()
        };
        // The host's own handlers — a custom effect, an embedding model —
        // are in the signature (the trace's row), never in the required
        // row (the agent's); the host registers them as it did when it
        // recorded, from the log, before the agent is built. The `host/`
        // prefix decides *which* signature keys are the host's to register
        // (the builder registers the agent's own); describing them is the
        // replayer's, from the handler table, and a key it cannot describe
        // is refused by name rather than skipped.
        // The host's nesting handlers are program: registered as the
        // producer registered them, never replayed (Matrix Q).
        if program.nesting.is_some() {
            driver
                .register_erased(
                    HandlerKey::from(RELAY_KEY),
                    rig_core::serve::ErasedHandler::new(Relay),
                )
                .expect("a fresh key");
            let notify = std::sync::Arc::new(tokio::sync::Notify::new());
            driver
                .register_erased(
                    HandlerKey::from(NEVER_KEY),
                    rig_core::serve::ErasedHandler::new(Never {
                        reached: std::sync::Arc::clone(&notify),
                    }),
                )
                .expect("a fresh key");
            reached = Some(notify);
        }
        // The host's keys come from the handler table (the header's first
        // source): a key the host served but a layer denied every dispatch
        // to has no record and is not in the signature, yet must be served.
        let host_keys: Vec<HandlerKey> = log
            .header
            .handlers
            .iter()
            .map(|handler| &handler.key)
            .filter(|key| key.parts().owner.as_deref() == Some("host"))
            .filter(|key| {
                program.nesting.is_none()
                    || (key.as_str() != RELAY_KEY && key.as_str() != NEVER_KEY)
            })
            .cloned()
            .collect();
        for key in host_keys {
            let replayer = EffectLogReplayer::for_key(&log, &key)
                .expect("the host handler's records")
                .checking(check);
            let handler = if key.as_str() == NOTE_KEY {
                layered(
                    rig_core::serve::ErasedHandler::new(replayer),
                    program,
                    LayerAt::Note,
                    &world,
                )
            } else {
                rig_core::serve::ErasedHandler::new(replayer)
            };
            driver.register_erased(key, handler).expect("a fresh key");
        }
        // A route registered after build is served the way the producer
        // served it: on the bus, outside the builder and the row.
        if let Some(label) = program.late_route {
            let key = HandlerKey::from(format!("{}/model:{label}", program.owner));
            let replayer = EffectLogReplayer::for_key(&log, &key)
                .expect("the late route's records")
                .checking(check);
            driver
                .register_erased(key, rig_core::serve::ErasedHandler::new(replayer))
                .expect("a fresh key");
        }
        driver.record_to(recorder.clone());
        let driver = tokio::spawn(driver);
        Self {
            log,
            dispatcher,
            registrar,
            recorder,
            driver,
            model_key,
            memory_key,
            reached,
            world,
            check,
        }
    }

    pub fn route_key(&self, label: &str) -> HandlerKey {
        HandlerKey::from(format!("{}/model:{label}", self.log_owner()))
    }

    pub fn log_owner(&self) -> String {
        self.model_key
            .parts()
            .owner
            .map(|owner| owner.to_string())
            .expect("the model key names its owner")
    }

    pub fn tool_keys(&self) -> Vec<HandlerKey> {
        self.log
            .header
            .required
            .iter()
            .filter(|(_, family)| **family == EffectFamily::Tool)
            .map(|(key, _)| key.clone())
            .collect()
    }

    pub fn tool_server(&self) -> rig_agent::tool::server::ToolServerHandle {
        self.tool_server_for(&Program::DEFAULT)
    }

    /// The program's tool registry over the log: every required tool from
    /// its replayer, the retrievable ones behind the recorded index's
    /// replayer under the program's owner.
    pub fn tool_server_for(&self, program: &Program) -> rig_agent::tool::server::ToolServerHandle {
        let mut server = ToolServer::new().owner(self.log_owner());
        let mut retrievable = rig_agent::tool::ToolSet::default();
        for key in self.tool_keys() {
            let parts = key.parts();
            assert_eq!(
                parts.kind.as_deref(),
                Some("tool"),
                "a tool key names its tool"
            );
            let name = parts.label.as_ref();
            let tool = match program.nesting {
                Some(nesting) if key.as_str() == NESTING_TOOL_KEY => {
                    RegisteredTool::from_handler(Lookup {
                        nesting,
                        model_key: self.model_key.clone(),
                    })
                    .expect("a tool-family handler")
                }
                _ if key.as_str().ends_with("/tool:add#0") && !program.layers.is_empty() => {
                    let replayer = self
                        .replayer(&self.log, &key)
                        .expect("a required tool is described");
                    let handler = layered(
                        rig_core::serve::ErasedHandler::new(replayer),
                        program,
                        LayerAt::Tool,
                        &self.world,
                    );
                    RegisteredTool::from_handler(handler).expect("a tool-family handler")
                }
                _ => {
                    let replayer = self
                        .replayer(&self.log, &key)
                        .expect("a required tool is described");
                    RegisteredTool::from_handler(replayer).expect("a tool-family replayer")
                }
            };
            if program.retrievable.contains(&name) {
                retrievable.add_registered(tool);
            } else {
                server = server.registered_tool(tool);
            }
        }
        if let Some(sample) = program.retrieved_tools {
            let key = HandlerKey::from(format!("{}/retrieve:tools#0", self.log_owner()));
            let replayer = self
                .replayer(&self.log, &key)
                .expect("the tool index's records");
            server = server.retrieved_tools_handler(sample, replayer, retrievable);
        }
        server.run()
    }

    pub fn context_key(&self) -> HandlerKey {
        HandlerKey::from(format!("{}/retrieve:context#0", self.log_owner()))
    }

    pub async fn close(self) -> EffectLog {
        drop((self.dispatcher, self.registrar));
        within(self.driver).await.expect("driver task");
        self.recorder.take()
    }
}

// ---------------------------------------------------------------------------
// The bus engine.

/// The program's agent over the replay bus, its memory, route and
/// context handlers replaying from `source` (the whole golden for a
/// fresh run; the golden's tail for a resumed one, whose head the hand
/// driver's handlers answered).
pub fn build_agent(
    replay: &Replay,
    program: &Program,
    server: rig_agent::tool::server::ToolServerHandle,
    source: &EffectLog,
) -> rig_agent::Agent {
    let agent = build_agent_unchecked(replay, program, server, source);
    agent
        .check_replayable(&replay.log)
        .expect("the same program as the one recorded");
    agent
}

/// [`build_agent`] without the replay check, for a row that pins the
/// refusal.
pub fn build_agent_unchecked(
    replay: &Replay,
    program: &Program,
    server: rig_agent::tool::server::ToolServerHandle,
    source: &EffectLog,
) -> rig_agent::Agent {
    let mut builder = AgentBuilder::over_bus(
        replay.dispatcher.clone(),
        replay.registrar.clone(),
        program.owner,
        replay.model_key.clone(),
    )
    .name(program.owner)
    .tool_server_handle(server);
    builder = match program.preamble {
        Some(preamble) => builder.preamble(preamble),
        None => builder.without_preamble(),
    };
    if let Some(doc) = program.append_preamble {
        builder = builder.append_preamble(doc);
    }
    for doc in program.context {
        builder = builder.context(*doc);
    }
    if let Some(temperature) = program.temperature {
        builder = builder.temperature(temperature);
    }
    if let Some(max_tokens) = program.max_tokens {
        builder = builder.max_tokens(max_tokens);
    }
    if let Some(params) = program.additional_params {
        builder = builder.additional_params(params());
    }
    if let Some(choice) = program.tool_choice {
        builder = builder.tool_choice(choice.tool_choice());
    }
    if let Some(schema) = program.output_schema {
        builder = builder.output_schema_raw(
            serde_json::from_value(schema()).expect("the producer's schema is a schema"),
        );
    }
    if let Some(mode) = program.output_mode {
        builder = builder.output_mode(mode.mode());
    }
    if let Some(default_max_turns) = program.default_max_turns {
        builder = builder.default_max_turns(default_max_turns);
    }
    builder = add_hooks(builder, program.hooks);
    if let Some(conversation) = program.conversation {
        let memory = replay
            .replayer(source, &replay.memory_key)
            .expect("the conversation's records");
        let memory = layered(
            rig_core::serve::ErasedHandler::new(memory),
            program,
            LayerAt::Memory,
            &replay.world,
        );
        builder = builder.memory_handler(memory).conversation(conversation);
    }
    // A route is the agent's to register (`model_route_handler`), as the
    // producer's `model_route` was; the host bus serves only the default
    // model, as the producer's client did.
    if let Some(label) = program.route {
        let route = replay
            .replayer(source, &replay.route_key(label))
            .expect("the route is in the required row");
        builder = builder.model_route_handler(label, route);
    }
    if let Some(samples) = program.dynamic_context {
        let index = replay
            .replayer(source, &replay.context_key())
            .expect("the context index's records");
        builder = builder.dynamic_context_handler(samples, index);
    }
    builder.build()
}

pub async fn bus_engine_reproduces(program: &Program) {
    let replay = Replay::open(program);
    let server = replay.tool_server_for(program);
    let agent = build_agent(&replay, program, server, &replay.log);

    let prompts: Vec<&str> = std::iter::once(program.prompt)
        .chain(program.second_prompt)
        .collect();
    let mut output = None;
    for prompt in prompts {
        output = if program.streamed {
            let mut runner = agent.stream_prompt(prompt);
            if let Some(history) = program.history {
                runner = runner.history(history());
            }
            if let Some(max_turns) = program.max_turns {
                runner = runner.max_turns(max_turns);
            }
            if let Some(concurrency) = program.tool_concurrency {
                runner = runner.tool_concurrency(concurrency);
            }
            runner = runner
                .max_invalid_tool_call_retries(program.invalid_retries)
                .unhandled_invalid_tool_call(unhandled_policy(program));
            let mut stream = runner.stream().await;
            let mut output = None;
            let mut failed_as_expected = false;
            while let Some(item) = within(stream.next()).await {
                match item {
                    Ok(MultiTurnStreamItem::FinalResponse(response)) => {
                        output = Some(response.output);
                    }
                    Err(StreamingError::Report(report))
                        if program.cancel_after_first_delta
                            && report.kind == rig_core::error::ErrorKind::Cancelled =>
                    {
                        break;
                    }
                    Err(StreamingError::Prompt(error))
                        if program.ending == Ending::MaxTurns
                            && matches!(*error, PromptError::MaxTurnsError { .. }) =>
                    {
                        failed_as_expected = true;
                    }
                    Err(StreamingError::Prompt(error))
                        if program.ending == Ending::UnknownToolCall
                            && matches!(*error, PromptError::UnknownToolCall { .. }) =>
                    {
                        failed_as_expected = true;
                    }
                    Err(StreamingError::Prompt(error))
                        if program.ending == Ending::MemoryError
                            && matches!(*error, PromptError::MemoryError(_)) =>
                    {
                        failed_as_expected = true;
                    }
                    Err(StreamingError::Prompt(error))
                        if matches!(
                            (&*error, program.ending),
                            (PromptError::PromptCancelled { reason, .. }, Ending::Cancelled(expected))
                                if reason == expected
                        ) =>
                    {
                        failed_as_expected = true;
                    }
                    Err(StreamingError::Report(report))
                        if program.ending == Ending::ProviderError
                            && report.kind == rig_core::error::ErrorKind::ProviderResponse =>
                    {
                        failed_as_expected = true;
                    }
                    Err(StreamingError::Report(report))
                        if program.ending == Ending::Failed(report.kind) =>
                    {
                        failed_as_expected = true;
                    }
                    Err(StreamingError::Completion(_))
                        if program.ending == Ending::ProviderError =>
                    {
                        failed_as_expected = true;
                    }
                    Err(error) => {
                        panic!("the replayer answered every request it recognised: {error:?}")
                    }
                    Ok(_) => {}
                }
            }
            drop(stream);
            if program.cancel_after_first_delta {
                // The driver resolves the cancelled dispatch on its own task.
                for _ in 0..64 {
                    tokio::task::yield_now().await;
                }
                None
            } else if program.ending != Ending::Answer {
                assert!(failed_as_expected, "the run ends in {:?}", program.ending);
                None
            } else {
                Some(output.expect("the stream yields a final response"))
            }
        } else {
            let mut runner = agent.prompt(prompt);
            if let Some(history) = program.history {
                runner = runner.history(history());
            }
            if let Some(max_turns) = program.max_turns {
                runner = runner.max_turns(max_turns);
            }
            if let Some(concurrency) = program.tool_concurrency {
                runner = runner.tool_concurrency(concurrency);
            }
            runner = runner
                .max_invalid_tool_call_retries(program.invalid_retries)
                .unhandled_invalid_tool_call(unhandled_policy(program));
            if program.cancel_when_reached {
                // The run is dropped once the nested child is reached: the
                // tool call and its chain are cancelled together.
                let reached = replay.reached.clone().expect("a nesting program");
                tokio::select! {
                    finished = within(runner.run()) => {
                        panic!("the run finished before the child was reached: {finished:?}")
                    }
                    () = within(reached.notified()) => {}
                }
                for _ in 0..64 {
                    tokio::task::yield_now().await;
                }
                continue;
            }
            match (within(runner.run()).await, program.ending) {
                (Ok(response), Ending::Answer) => Some(response.output),
                (Err(PromptError::MaxTurnsError { .. }), Ending::MaxTurns)
                | (Err(PromptError::UnknownToolCall { .. }), Ending::UnknownToolCall)
                | (Err(PromptError::MemoryError(_)), Ending::MemoryError) => None,
                (Err(PromptError::Report(report)), Ending::ProviderError)
                    if report.kind == rig_core::error::ErrorKind::ProviderResponse =>
                {
                    None
                }
                (Err(PromptError::Report(report)), Ending::Failed(kind)) if report.kind == kind => {
                    None
                }
                (Err(PromptError::PromptCancelled { reason, .. }), Ending::Cancelled(expected))
                    if reason == expected =>
                {
                    None
                }
                (Ok(response), ending) => {
                    panic!("the run ends in {ending:?}, not an answer: {response:?}")
                }
                (Err(error), _) => {
                    panic!("the replayer answered every request it recognised: {error:?}")
                }
            }
        };
    }
    if let Some(output) = output {
        assert_eq!(
            output,
            program
                .expected_output
                .map_or_else(|| golden_answer(&replay.log), str::to_owned)
        );
    }
    drop(agent);
    let log = replay.log.clone();
    let replayed = replay.close().await;
    assert_same_records(&replayed, &log, "bus engine");
}

// ---------------------------------------------------------------------------
// The hand driver: `AgentRun` stepped by this test, every step dispatched
// over the bus handles the engine would use.

/// The tool handles the program advertises, by tool name.
pub fn tool_handles(replay: &Replay) -> Vec<(String, ToolHandle)> {
    replay
        .tool_keys()
        .into_iter()
        .map(|key| {
            let handle: ToolHandle = replay.dispatcher.handle(&key).expect("a tool handle");
            (handle.name(), handle)
        })
        .collect()
}

/// The hand driver's dispatch of a turn's tool calls, making each hook's
/// decision itself: a patched call runs with the patched arguments, a
/// denied call never reaches the bus and the model sees the reason as its
/// result (`ToolResult::skipped`, as the engine shapes it), a replaced
/// result reaches the model as the replacement while the record holds the
/// tool's answer.
pub async fn call_tools(
    calls: Vec<PendingToolCall>,
    tools: &[(String, ToolHandle)],
    concurrency: usize,
    hooks: &[Hook],
    notes: Option<&rig_agent::bus::Handle<rig_core::effect::family::Custom<Note>>>,
) -> Result<Vec<UserContent>, &'static str> {
    let dispatch = |call: PendingToolCall| async move {
        if let Some(preresolved) = call.preresolved_result {
            return Ok(preresolved);
        }
        let name = call.tool_call.function.name.clone();
        let is_add = name == "add";
        if is_add && hooks.contains(&Hook::CancelAddDispatch) {
            // `Deny(Cancelled)`: the call never reaches the bus; the run stops.
            return Err(CANCEL_ADD_DISPATCH);
        }
        if is_add && hooks.contains(&Hook::DenyAdd) {
            return Ok(tool_result_output(
                call.tool_call.id.clone(),
                call.tool_call.provider.clone(),
                name,
                ToolOutput::text(DENY_REASON),
            ));
        }
        let (_, handle) = tools
            .iter()
            .find(|(tool, _)| *tool == name)
            .unwrap_or_else(|| panic!("the program advertises `{name}`"));
        let args = if is_add && hooks.contains(&Hook::PatchAddArgs) {
            PATCHED_ARGS.to_owned()
        } else {
            call.tool_call.function.arguments.to_string()
        };
        // The bus's answer, mapped as the engine maps it: a layer's denial
        // is the skipped result the model sees, a cancel stops the run, any
        // other failure is a failed result the model sees.
        let answer = match within(handle.call(name.clone(), args, ToolContext::new())).await {
            Ok(answer) => answer,
            Err(report) if report.kind == ErrorKind::Cancelled => {
                return Err(CANCEL_ADD_DISPATCH);
            }
            Err(report) if report.kind == ErrorKind::Denied => {
                return Ok(tool_result_output(
                    call.tool_call.id.clone(),
                    call.tool_call.provider.clone(),
                    name,
                    rig_core::tool::ToolResult::skipped(report.message)
                        .output()
                        .clone(),
                ));
            }
            Err(report) => {
                return Ok(tool_result_output(
                    call.tool_call.id.clone(),
                    call.tool_call.provider.clone(),
                    name,
                    rig_core::tool::ToolResult::failed(
                        rig_core::tool::ToolExecutionError::other(report.message.clone())
                            .with_model_feedback(report.message),
                    )
                    .output()
                    .clone(),
                ));
            }
        };
        // The outcome hook's dispatch, inside the tool's dispatch as the
        // engine fires it (Matrix I).
        if hooks.contains(&Hook::NoteAtOutcome) {
            let ack = within(notes.expect("a note hook").dispatch(Note {
                at: "outcome".to_owned(),
            }))
            .await
            .expect("the replayer acknowledged");
            assert!(ack.accepted && ack.at == "outcome", "{ack:?}");
        }
        // The model-visible output of the result, failed or not: what the
        // engine shapes into the transcript.
        if is_add && hooks.contains(&Hook::CancelAddOutcome) {
            // `Replace(Err(Cancelled))`: the tool ran and is recorded; the
            // run stops.
            return Err(CANCEL_ADD_OUTCOME);
        }
        let mut output = answer.result.output().clone();
        if is_add && hooks.contains(&Hook::ReplaceAddResult) {
            output = ToolOutput::text(REPLACED_RESULT);
        }
        // The engine's own shaping of a result (`rig_core::transcript`).
        Ok(tool_result_output(
            call.tool_call.id.clone(),
            call.tool_call.provider.clone(),
            name,
            output,
        ))
    };
    futures::stream::iter(calls)
        .map(dispatch)
        .buffered(concurrency.max(1))
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect()
}

/// The header's hook list for `program`: `DynamicContext` first when the
/// builder registered one, then the hooks by name, then the layers.
pub fn program_hooks(program: &Program, owner: &str) -> Vec<String> {
    program
        .dynamic_context
        .map(|_| "DynamicContext".to_owned())
        .into_iter()
        .chain(program.hooks.iter().map(|hook| hook_name(*hook)))
        .chain(layer_names(program, owner))
        .collect()
}

/// The name the header records for a hook: its type's last path segment.
pub fn hook_name(hook: Hook) -> String {
    let name = match hook {
        Hook::RetryUnknownTool => "RetryUnknownTool",
        Hook::ObserveEverything => "ObserveEverything",
        Hook::PatchAddArgs => "PatchAddArgs",
        Hook::DenyAdd => "DenyAdd",
        Hook::ReplaceAddResult => "ReplaceAddResult",
        Hook::ReplaceAnswer => "ReplaceAnswer",
        Hook::PreambleOverride => "PreambleOverride",
        Hook::DemandDone => "DemandDone",
        Hook::LookupBeforeRun => "LookupBeforeRun",
        Hook::RouteAfterFirstTurn => "RouteAfterFirstTurn",
        Hook::StopAtStart => "StopAtStart",
        Hook::StopAtModelSelect => "StopAtModelSelect",
        Hook::StopAtCompletionCall => "StopAtCompletionCall",
        Hook::CancelAddDispatch => "CancelAddDispatch",
        Hook::CancelAddOutcome => "CancelAddOutcome",
        Hook::CancelAnswer => "CancelAnswer",
        Hook::StopAfterTurn => "StopAfterTurn",
        Hook::StopAtAnswer => "StopAtAnswer",
        Hook::StopOnTextDelta => "StopOnTextDelta",
        Hook::StopOnToolCallDelta => "StopOnToolCallDelta",
        Hook::StopOnReasoningDelta => "StopOnReasoningDelta",
        Hook::RecordSettled => "RecordSettled",
        Hook::RepairToAdd => "RepairToAdd",
        Hook::SkipUnknown => "SkipUnknown",
        Hook::NoteAtStart => "NoteAtStart",
        Hook::NoteAtCompletionCall => "NoteAtCompletionCall",
        Hook::NoteAtOutcome => "NoteAtOutcome",
        Hook::NoteAtSettled => "NoteAtSettled",
        Hook::NoteTwice => "NoteTwice",
        Hook::NoteUnserved => "NoteUnserved",
        Hook::EmbedPrompt => "EmbedPrompt",
        Hook::ClearAtStart => "ClearAtStart",
        Hook::ClearAtSettled => "ClearAtSettled",
        Hook::StopOnToolNameDelta => "StopOnToolNameDelta",
        Hook::StopOnToolArgumentsDelta => "StopOnToolArgumentsDelta",
        Hook::PatchToolChoiceRequiredFirst => "PatchToolChoiceRequiredFirst",
        Hook::PatchToolChoiceNoneSecond => "PatchToolChoiceNoneSecond",
        Hook::PatchExtraContext => "PatchExtraContext",
        Hook::PatchMaxTokensSecond => "PatchMaxTokensSecond",
        Hook::PatchThinkingSecond => "PatchThinkingSecond",
        Hook::PatchPreambleSecond => "PatchPreambleSecond",
        Hook::PatchActiveToolsNoneSecond => "PatchActiveToolsNoneSecond",
        Hook::PatchHistoryFirst => "PatchHistoryFirst",
        Hook::RouteOnFirstTurn => "RouteOnFirstTurn",
        Hook::SelectLate => "SelectLate",
        Hook::StopAfterTurnN(n) => return format!("StopAfterTurn({n})"),
        Hook::RerankDocs => "RerankDocs",
        Hook::NoteDeniedAtStart => "NoteDeniedAtStart",
        Hook::HistoryIsReplaced => "HistoryIsReplaced",
        Hook::NoteUnserializableAtStart => "NoteUnserializableAtStart",
        Hook::NotesAtStart(n) => return format!("NotesAtStart({n})"),
    };
    name.to_owned()
}

/// What the bus engine's `check_replayable` checks, for the hand driver:
/// the header names this program — its spec hash (the builder's spec:
/// the default budget, no runner retries), its hook stack, and every key
/// it will dispatch to in the required row.
pub fn assert_header_names_the_program(replay: &Replay, program: &Program) {
    let header = &replay.log.header;
    let builder_spec = RunSpec {
        max_turns: Some(program.default_max_turns.unwrap_or(1)),
        max_invalid_tool_call_retries: 0,
        unhandled_invalid_tool_call: UnhandledInvalidToolCall::Fail,
        ..run_spec(program)
    };
    assert_eq!(
        header.run_spec,
        rig_effect_log::stable_hash(&builder_spec).ok(),
        "the header's spec hash is this program's"
    );
    // `dynamic_context` is a hook of the builder's own (`DynamicContext`),
    // named in the header like any other; the producers register it
    // before their own hooks.
    let hooks: Vec<String> = program
        .dynamic_context
        .map(|_| "DynamicContext".to_owned())
        .into_iter()
        .chain(program.hooks.iter().map(|hook| hook_name(*hook)))
        .chain(layer_names(program, &replay.log_owner()))
        .collect();
    assert_eq!(
        header.hooks, hooks,
        "the header's hook stack is this program's"
    );
    let mut needed = vec![(replay.model_key.clone(), EffectFamily::Completion)];
    if program.conversation.is_some() {
        needed.push((replay.memory_key.clone(), EffectFamily::Memory));
    }
    if let Some(label) = program.route {
        needed.push((replay.route_key(label), EffectFamily::Completion));
    }
    if program.dynamic_context.is_some() {
        needed.push((replay.context_key(), EffectFamily::Retrieve));
    }
    if program.retrieved_tools.is_some() {
        needed.push((
            HandlerKey::from(format!("{}/retrieve:tools#0", replay.log_owner())),
            EffectFamily::Retrieve,
        ));
    }
    for (key, family) in needed {
        assert_eq!(
            header.required.get(&key),
            Some(&family),
            "the required row names `{key}`"
        );
    }
}

fn unhandled_policy(program: &Program) -> UnhandledInvalidToolCall {
    match program.unhandled {
        Unhandled::Fail => UnhandledInvalidToolCall::Fail,
        Unhandled::Ignore => UnhandledInvalidToolCall::Ignore,
    }
}

pub fn run_spec(program: &Program) -> RunSpec {
    RunSpec {
        preamble: program.spec_preamble(),
        static_context: program.static_context(),
        additional_params: program.additional_params.map(|params| params()),
        max_tokens: program.max_tokens,
        temperature: program.temperature,
        tool_choice: program.tool_choice.map(Choice::tool_choice),
        // The runner's budget, else the builder's default, as the engine
        // resolves it.
        max_turns: program.max_turns.or(program.default_max_turns),
        max_invalid_tool_call_retries: program.invalid_retries,
        output_schema: program.output_schema.map(|schema| schema()),
        output_mode: program.output_mode.map_or(OutputMode::Auto, Output::mode),
        unhandled_invalid_tool_call: unhandled_policy(program),
        ..RunSpec::new()
    }
}

/// The id of the last record the hand-driven head consumes: the end of
/// the first run of tool records (and the custom notes a hook dispatches
/// inside them) after a completion. Everything after it is the tail the
/// resumed engine replays.
fn head_end_id(log: &EffectLog, tool_turns: usize) -> u64 {
    let records = &log.records;
    let mut cursor = 0;
    let mut end = 0;
    for _ in 0..tool_turns {
        let first_completion = cursor
            + records[cursor..]
                .iter()
                .position(|record| record.kind.family() == EffectFamily::Completion)
                .expect("a resumed program has a completion");
        let first_tool = first_completion
            + records[first_completion..]
                .iter()
                .position(|record| record.kind.family() == EffectFamily::Tool)
                .expect("a resumed program has a tool turn");
        end = first_tool;
        while end + 1 < records.len()
            && matches!(
                records[end + 1].kind.family(),
                EffectFamily::Tool | EffectFamily::Custom
            )
        {
            end += 1;
        }
        cursor = end + 1;
    }
    records[end].id.as_u64()
}

/// The golden's records after `tool_turns` tool turns, under its header.
fn tail_after(log: &EffectLog, tool_turns: usize) -> EffectLog {
    let head_end = head_end_id(log, tool_turns);
    EffectLog {
        header: log.header.clone(),
        records: log
            .records
            .iter()
            .filter(|record| record.id.as_u64() > head_end)
            .cloned()
            .collect(),
    }
}

/// `Resume::Never` never suspends, so no tail is ever asked for it.
fn unreachable_resume() -> EffectLog {
    panic!("a run that never suspends has no tail")
}

/// The resumed engine's tail: the persisted run resumed on the replay bus
/// by the program's agent, whose memory, route and context handlers are
/// re-registered from the golden's tail (the head's handlers answered the
/// rest). `Ok(Some)` is the answer, `Ok(None)` a non-answer ending the
/// program expects, `Err` the hook's cancel reason.
async fn resumed_tail(
    replay: &Replay,
    program: &Program,
    server: rig_agent::tool::server::ToolServerHandle,
    state: String,
    tail_log: EffectLog,
) -> Result<Option<PromptResponse>, &'static str> {
    if program.conversation.is_some() {
        replay.registrar.deregister(&replay.memory_key);
    }
    if let Some(label) = program.route {
        replay.registrar.deregister(&replay.route_key(label));
    }
    if program.dynamic_context.is_some() {
        replay.registrar.deregister(&replay.context_key());
    }
    let restored: AgentRun = serde_json::from_str(&state).expect("the run state restores");
    let agent = build_agent(replay, program, server, &tail_log);
    let mut runner = agent.runner("ignored").resume(restored);
    if let Some(max_turns) = program.max_turns {
        runner = runner.max_turns(max_turns);
    }
    if let Some(concurrency) = program.tool_concurrency {
        runner = runner.tool_concurrency(concurrency);
    }
    runner = runner
        .max_invalid_tool_call_retries(program.invalid_retries)
        .unhandled_invalid_tool_call(unhandled_policy(program));
    let outcome = if program.streamed {
        let mut stream = runner.stream().await;
        let mut output = None;
        let mut failure = None;
        while let Some(item) = within(stream.next()).await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(response)) => output = Some(response),
                Ok(_) => {}
                Err(StreamingError::Prompt(error)) => failure = Some(*error),
                Err(error) => panic!("the resumed stream: {error:?}"),
            }
        }
        drop(stream);
        match failure {
            Some(error) => Err(error),
            None => Ok(output.expect("the resumed stream yields a final response")),
        }
    } else {
        within(runner.run()).await
    };
    drop(agent);
    match (outcome, program.ending) {
        (Ok(response), Ending::Answer) => Ok(Some(response)),
        (Err(PromptError::MaxTurnsError { .. }), Ending::MaxTurns)
        | (Err(PromptError::UnknownToolCall { .. }), Ending::UnknownToolCall) => Ok(None),
        (Err(PromptError::PromptCancelled { reason, .. }), Ending::Cancelled(expected))
            if reason == expected =>
        {
            Err(expected)
        }
        (Ok(response), ending) => {
            panic!("the resumed run ends in {ending:?}, not an answer: {response:?}")
        }
        (Err(error), _) => panic!("the resumed run: {error:?}"),
    }
}

pub async fn hand_driver_reproduces(program: &Program) {
    hand_drive(program, Resume::Never).await;
}

/// How the hand driver hands the run on (Matrices L and R).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resume {
    /// Never: the hand driver takes the program to its end.
    Never,
    /// After the first tool turn's results: the serialized run resumed on
    /// the same replay bus, whose replayers hold the tail (Matrix L).
    AfterFirstToolTurn,
    /// After `tool_turns` tool turns' results, through a `Checkpoint`: the
    /// head's serialized run and position become the checkpoint (JSON
    /// round trip), the continuation is `EffectLog::from_checkpoint` over
    /// `against`, and the resumed engine replays it under `check`
    /// (Matrix R).
    Checkpoint {
        tool_turns: usize,
        check: RequestCheck,
        against: Against,
    },
}

/// What a checkpoint's continuation is replayed against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Against {
    /// The log's tail from the checkpoint: the continuation.
    Tail,
    /// The full log in the tail's place: refused by its first id.
    FullLog,
}

impl Resume {
    fn tool_turns(self) -> Option<usize> {
        match self {
            Self::Never => None,
            Self::AfterFirstToolTurn => Some(1),
            Self::Checkpoint { tool_turns, .. } => Some(tool_turns),
        }
    }
}

/// Matrix R: the run continued from a checkpoint.
pub async fn checkpoint_reproduces(
    program: &Program,
    tool_turns: usize,
    check: RequestCheck,
    against: Against,
) {
    hand_drive(
        program,
        Resume::Checkpoint {
            tool_turns,
            check,
            against,
        },
    )
    .await;
}

/// The run continued: the hand driver takes the program up to and
/// including its first tool turn's results, serializes the `AgentRun`, and
/// the bus engine resumes it on the same replay bus — whose replayers have
/// answered the head and hold the tail — to the golden's ending. The
/// recorded log is the whole golden: the head by hand, the tail by the
/// engine, one record sequence. A resumed run loads no memory and saves
/// none (the runner skips both on `resume`), so the head loads and the
/// driver appends; the resumed engine fires the settled hooks itself.
pub async fn resume_reproduces(program: &Program) {
    hand_drive(program, Resume::AfterFirstToolTurn).await;
}

async fn hand_drive(program: &Program, resume: Resume) {
    let check = match resume {
        Resume::Checkpoint { check, .. } => check,
        Resume::Never | Resume::AfterFirstToolTurn => RequestCheck::Payload,
    };
    let replay = Replay::open_checking(program, check);
    let resume_after = resume.tool_turns();
    let resumes = resume_after.is_some();
    let server = replay.tool_server_for(program);
    server.attach(&replay.registrar);
    // The context index, registered by the driver as the builder would.
    let context: Option<rig_agent::bus::Handle<rig_core::effect::family::Retrieve>> =
        program.dynamic_context.map(|_| {
            let key = replay.context_key();
            let replayer = replay
                .replayer(&replay.log, &key)
                .expect("the context index's records");
            replay
                .registrar
                .register_erased(key.clone(), rig_core::serve::ErasedHandler::new(replayer))
                .expect("a fresh key");
            replay.dispatcher.handle(&key).expect("the context index")
        });
    let tools = tool_handles(&replay);
    let model: ModelHandle = replay
        .dispatcher
        .handle(&replay.model_key)
        .expect("the model");
    // The route, registered by the driver as the agent would register it,
    // selected on every turn after the first when the program's hook does.
    let route: Option<ModelHandle> = program.route.map(|label| {
        let key = replay.route_key(label);
        let replayer = replay
            .replayer(&replay.log, &key)
            .expect("the route is in the required row");
        replay
            .registrar
            .register_erased(key.clone(), rig_core::serve::ErasedHandler::new(replayer))
            .expect("a fresh key");
        replay.dispatcher.handle(&key).expect("the route")
    });
    let late_route: Option<ModelHandle> = program.late_route.map(|label| {
        replay
            .dispatcher
            .handle(&replay.route_key(label))
            .expect("the late route")
    });
    let memory: Option<(MemoryHandle, ConversationId)> = program.conversation.map(|id| {
        let replayer = replay
            .replayer(&replay.log, &replay.memory_key)
            .expect("the conversation's records");
        replay
            .registrar
            .register_erased(
                replay.memory_key.clone(),
                layered(
                    rig_core::serve::ErasedHandler::new(replayer),
                    program,
                    LayerAt::Memory,
                    &replay.world,
                ),
            )
            .expect("a fresh key");
        let handle: MemoryHandle = replay
            .dispatcher
            .handle(&replay.memory_key)
            .expect("the memory");
        (handle, ConversationId::from(id))
    });
    assert_header_names_the_program(&replay, program);
    let spec = run_spec(program);
    // Explicit history bypasses memory for the run, as the runner does.
    // The host's handler the note hooks dispatch to, bound as the hooks
    // bind it (a refused bind is the unserved program's point).
    let notes = program
        .hooks
        .iter()
        .any(|hook| {
            matches!(
                hook,
                Hook::NoteAtStart
                    | Hook::NoteAtCompletionCall
                    | Hook::NoteAtOutcome
                    | Hook::NoteAtSettled
                    | Hook::NoteTwice
                    | Hook::NotesAtStart(_)
            )
        })
        .then(|| {
            replay
                .dispatcher
                .bind(&note_key())
                .expect("the host serves notes")
        });
    let note = |at: &'static str| {
        let notes = notes.as_ref();
        async move {
            let ack = within(
                notes
                    .expect("a note hook")
                    .dispatch(Note { at: at.to_owned() }),
            )
            .await
            .expect("the replayer acknowledged");
            assert!(ack.accepted && ack.at == at, "{ack:?}");
        }
    };
    if program.hooks.contains(&Hook::NoteUnserved) {
        let refused = replay
            .dispatcher
            .bind(&note_key())
            .expect_err("no host serves notes");
        assert_eq!(refused.kind, rig_core::error::ErrorKind::HandlerUnavailable);
    }
    assert!(
        !(resumes && program.second_prompt.is_some()),
        "a resumed program is one run"
    );
    let mut resumed: Option<String> = None;
    // The default model's label, as the engine names it to the selection
    // hook: the key's label.
    let default_label = {
        let parts = replay.model_key.parts();
        assert_eq!(
            parts.kind.as_deref(),
            Some("model"),
            "the model key names its label"
        );
        parts.label.to_string()
    };
    let prompts: Vec<&str> = std::iter::once(program.prompt)
        .chain(program.second_prompt)
        .collect();
    let mut last_response = None;
    for prompt in prompts {
        let mut load_failed = false;
        let history = match (program.history, &memory) {
            (Some(history), _) => Some(history()),
            (None, Some((handle, id))) => match within(handle.load(id.clone())).await {
                Ok(history) => Some(history),
                Err(report) if program.ending == Ending::MemoryError => {
                    assert!(
                        matches!(
                            report.kind,
                            rig_core::error::ErrorKind::MemoryBackend | ErrorKind::Denied
                        ),
                        "{report:?}"
                    );
                    load_failed = true;
                    None
                }
                Err(report) => panic!("the replayer answered the load: {report:?}"),
            },
            (None, None) => None,
        };
        if load_failed {
            // As the engine: the run fails at the load, before any completion.
            drop((model, route, late_route, tools, memory, context));
            let log = replay.log.clone();
            let replayed = replay.close().await;
            assert_same_records(&replayed, &log, "hand driver");
            return;
        }
        // The run-start hook's clear: the hook fires after the load, so the
        // clear lands between the load and the first completion.
        if program.hooks.contains(&Hook::ClearAtStart) {
            let (handle, id) = memory.as_ref().expect("a memory program");
            within(handle.clear(id.clone()))
                .await
                .expect("the replayer answered the clear");
        }
        if program.hooks.contains(&Hook::LookupBeforeRun) {
            // The hook's own dispatch, before the first completion.
            let (_, add) = tools
                .iter()
                .find(|(tool, _)| tool == "add")
                .expect("the program advertises add");
            let answer = within(add.call("add", LOOKUP_ARGS, ToolContext::new()))
                .await
                .expect("the replayer answered the hook's call");
            assert_eq!(answer.result.output().render(), "3");
        }
        // The run-start hooks' dispatches, before the first completion.
        if program.hooks.contains(&Hook::NoteAtStart) {
            note("start").await;
        }
        if let Some(Hook::NotesAtStart(n)) = program
            .hooks
            .iter()
            .find(|hook| matches!(hook, Hook::NotesAtStart(_)))
        {
            for n in 0..*n {
                note(Box::leak(format!("start-{n}").into_boxed_str())).await;
            }
        }
        if program.hooks.contains(&Hook::NoteUnserializableAtStart) {
            // The hook's effect has no wire form: refused at the send,
            // nothing reaches the bus.
            let host = replay
                .dispatcher
                .bind(&unserializable_key())
                .expect("the host serves the kind");
            let report = within(host.dispatch(Unserializable))
                .await
                .expect_err("no wire form, no dispatch");
            assert_unserializable(&report);
        }
        if program.hooks.contains(&Hook::NoteDeniedAtStart) {
            // The host's layer denies the note: the hook sees `Denied`.
            let host = replay
                .dispatcher
                .bind(&note_key())
                .expect("the host serves notes");
            let report = within(host.dispatch(Note {
                at: "start".to_owned(),
            }))
            .await
            .expect_err("denied by the host's layer");
            assert_eq!(report.kind, ErrorKind::Denied, "{report:?}");
        }
        if program.hooks.contains(&Hook::NoteTwice) {
            let host = notes.as_ref().expect("a note hook");
            let first = within(host.dispatch(Note {
                at: "first".to_owned(),
            }));
            let second = within(host.dispatch(Note {
                at: "second".to_owned(),
            }));
            let (first, second) = futures::join!(first, second);
            assert_eq!(first.expect("acknowledged").at, "first");
            assert_eq!(second.expect("acknowledged").at, "second");
        }
        if program.hooks.contains(&Hook::RerankDocs) {
            let host = replay
                .dispatcher
                .bind(&rerank_key())
                .expect("the host serves reranking");
            let ranked = within(host.dispatch(rerank_request(prompt)))
                .await
                .expect("the replayer reranked");
            assert_eq!(ranked.results.len(), 2, "{ranked:?}");
        }
        if program.hooks.contains(&Hook::EmbedPrompt) {
            let host = replay
                .dispatcher
                .bind(&embed_key())
                .expect("the host serves embeddings");
            let outputs = within(host.dispatch(rig_core::effect::EmbedInputs::Texts(vec![
                program.prompt.to_owned(),
            ])))
            .await
            .expect("the replayer embedded");
            assert!(matches!(outputs, rig_core::effect::EmbedOutputs::Texts(_)));
        }
        let mut run = AgentRun::from_spec(&spec, prompt, history);
        let mut tool_turns_done = 0usize;
        // The routing hook selects the route once a model has been asked
        // (`previous_model` is set): the first call goes to the default.
        let mut asked_before = false;
        // The hook that stopped the run, with its reason: the same decision at
        // the same point the engine makes it.
        let mut cancelled: Option<&'static str> = None;
        // A stop before any dispatch: at run start, at model selection, or
        // before the completion call. The hand driver has no seam for those
        // hooks (nothing was dispatched to drive); the row asserts what the
        // engine's empty log asserts, the header over no records.
        let stop_before_any = [
            (Hook::StopAtStart, STOP_AT_START),
            (Hook::StopAtModelSelect, STOP_AT_MODEL_SELECT),
            (Hook::StopAtCompletionCall, STOP_AT_COMPLETION_CALL),
        ]
        .into_iter()
        .find(|(hook, _)| program.hooks.contains(hook))
        .map(|(_, reason)| reason);
        let response = loop {
            if let Some(reason) = stop_before_any {
                cancelled = Some(reason);
                break None;
            }
            let step = match (run.next_step(), program.ending) {
                (Ok(step), _) => step,
                (Err(PromptError::MaxTurnsError { .. }), Ending::MaxTurns) => break None,
                (Err(error), _) => panic!("a step: {error:?}"),
            };
            match step {
                AgentRunStep::CallModel {
                    prompt,
                    history,
                    turn,
                } => {
                    // The step's turn number, before `turn` is rebound to
                    // the model's turn below.
                    let turn_number = turn;
                    // The selection hook's choice: a route after the first
                    // turn, on the first turn only, or on every turn.
                    let (model, label) = match &route {
                        Some(route)
                            if program.hooks.contains(&Hook::RouteAfterFirstTurn)
                                && asked_before =>
                        {
                            (route, program.route.expect("a route"))
                        }
                        Some(route)
                            if program.hooks.contains(&Hook::RouteOnFirstTurn) && !asked_before =>
                        {
                            (route, program.route.expect("a route"))
                        }
                        _ if program.hooks.contains(&Hook::SelectLate) => (
                            late_route.as_ref().expect("a late route"),
                            program.late_route.expect("a late route"),
                        ),
                        _ => (&model, default_label.as_str()),
                    };
                    asked_before = true;
                    // The retrievals the engine performs at the boundary, in its
                    // order: the context (a completion-call hook), then the tool
                    // index; the query is the current prompt's text, else the
                    // last text in history, as the engine derives it.
                    let query = prompt
                        .rag_text()
                        .or_else(|| history.iter().rev().find_map(Message::rag_text))
                        .unwrap_or_default();
                    // The completion-call hooks' patches for this turn, merged
                    // as the stack merges them.
                    let mut turn_patch = patch_for_turn(program, turn);
                    if let (Some(context), Some(samples)) = (&context, program.dynamic_context) {
                        let req = rig_core::vector_store::request::VectorSearchRequest::builder()
                            .query(query.clone())
                            .samples(samples as u64)
                            .build();
                        let results = within(context.top_n::<serde_json::Value>(req))
                            .await
                            .expect("the replayer answered the context query");
                        let docs = results.into_iter().map(|(_, id, value)| Document {
                            id,
                            text: serde_json::to_string_pretty(&value)
                                .unwrap_or_else(|_| value.to_string()),
                            additional_props: Default::default(),
                        });
                        turn_patch = Some(turn_patch.unwrap_or_default().extra_context(docs));
                    }
                    let mut dynamic_tool_ids = Vec::new();
                    for (key, kind) in server.retrieval_effects(Some(query.clone())) {
                        match within(replay.dispatcher.dispatch(&key, kind))
                            .await
                            .expect("the replayer answered the tool query")
                        {
                            rig_core::effect::Outcome::Documents(
                                rig_core::effect::RetrievedDocuments::Ids(ids),
                            ) => dynamic_tool_ids.extend(ids.into_iter().map(|(_, id)| id)),
                            other => panic!("retrieved ids, not {other:?}"),
                        }
                    }
                    let definitions = server
                        .snapshot_with_dynamic(&dynamic_tool_ids)
                        .take_definitions();
                    let prepared = prepare_request(
                        &spec,
                        &model.capabilities(),
                        &history,
                        definitions,
                        run.output_tool_name(),
                        turn_patch.as_ref(),
                    )
                    .expect("prepared");
                    // The driver's routing state, persisted on the run as the
                    // engine persists it — after the request prepared, so a
                    // preparation failure leaves it unchanged — so a resumed
                    // engine's selection hook sees it.
                    run.set_previous_model(rig_core::completion::ModelRef::new(label));
                    run.set_output_tool_name(prepared.output_tool_name.clone());
                    run.advertise_tools(turn, prepared.tools.clone());
                    let executable = prepared.executable_tool_names.clone();
                    let allowed = prepared.allowed_tool_names.clone();
                    let request = prepared
                        .apply(CompletionRequestBuilder::unbound(prompt))
                        .build();
                    // The completion-call hook's dispatch, before the completion.
                    if program.hooks.contains(&Hook::NoteAtCompletionCall) {
                        note("completion_call").await;
                    }
                    let turn = if program.streamed {
                        let mut stream = model.stream(request);
                        let mut assembler = StreamedTurnAssembler::new(executable, allowed);
                        let mut provider_failed = false;
                        let mut delta_stop: Option<&'static str> = None;
                        let mut turn_abandoned = false;
                        let mut unknown_tool_call = false;
                        while let Some(event) = within(stream.next()).await {
                            let event = match event {
                                Ok(event) => event,
                                Err(report)
                                    if program.cancel_after_first_delta
                                        && report.kind == rig_core::error::ErrorKind::Cancelled =>
                                {
                                    break;
                                }
                                Err(report)
                                    if program.ending == Ending::ProviderError
                                        && report.kind
                                            == rig_core::error::ErrorKind::ProviderResponse =>
                                {
                                    provider_failed = true;
                                    break;
                                }
                                Err(report) if program.ending == Ending::Failed(report.kind) => {
                                    provider_failed = true;
                                    break;
                                }
                                Err(report) => {
                                    panic!(
                                        "the replayer re-emitted the recorded stream: {report:?}"
                                    )
                                }
                            };
                            // The observe-only hooks' stops, at the delta they
                            // fire on: the engine leaves the stream there.
                            if let rig_core::streaming::StreamEvent::BlockDelta { delta, .. } =
                                &event
                            {
                                let stop = match delta {
                                    rig_core::streaming::Delta::Text { .. }
                                        if program.hooks.contains(&Hook::StopOnTextDelta) =>
                                    {
                                        Some(STOP_ON_TEXT_DELTA)
                                    }
                                    rig_core::streaming::Delta::ToolName { .. }
                                    | rig_core::streaming::Delta::ToolArguments { .. }
                                        if program.hooks.contains(&Hook::StopOnToolCallDelta) =>
                                    {
                                        Some(STOP_ON_TOOL_CALL_DELTA)
                                    }
                                    rig_core::streaming::Delta::ToolName { .. }
                                        if program.hooks.contains(&Hook::StopOnToolNameDelta) =>
                                    {
                                        Some(STOP_ON_TOOL_NAME_DELTA)
                                    }
                                    rig_core::streaming::Delta::ToolArguments { arguments }
                                        if program
                                            .hooks
                                            .contains(&Hook::StopOnToolArgumentsDelta)
                                            && !arguments.is_empty() =>
                                    {
                                        Some(STOP_ON_TOOL_ARGUMENTS_DELTA)
                                    }
                                    rig_core::streaming::Delta::Reasoning { .. }
                                        if program.hooks.contains(&Hook::StopOnReasoningDelta) =>
                                    {
                                        Some(STOP_ON_REASONING_DELTA)
                                    }
                                    _ => None,
                                };
                                if stop.is_some() {
                                    delta_stop = stop;
                                    break;
                                }
                            }
                            let events = assembler.ingest(&event).expect("a well-formed stream");
                            // An invalid call surfaced mid-stream: resolved as the
                            // engine resolves it — the hook's decision, else the
                            // runner's policy — through the run's streamed seam.
                            for streamed in events {
                                let StreamedTurnEvent::InvalidToolCall(invalid) = streamed else {
                                    continue;
                                };
                                let partial = assembler.partial_turn(stream.message_id.clone());
                                let action = if program.hooks.contains(&Hook::RetryUnknownTool) {
                                    Some(retry_feedback(&invalid.tool_call.function.name))
                                } else if program.hooks.contains(&Hook::RepairToAdd) {
                                    Some(InvalidToolCallAction::repair("add"))
                                } else if program.hooks.contains(&Hook::SkipUnknown) {
                                    Some(InvalidToolCallAction::skip(SKIP_REASON))
                                } else {
                                    None
                                };
                                let resolved = match (action, program.unhandled) {
                                    (Some(action), _) => run.resolve_streamed_invalid_tool_call(
                                        &partial, &invalid, action,
                                    ),
                                    (None, Unhandled::Fail) => run
                                        .resolve_streamed_invalid_tool_call(
                                            &partial,
                                            &invalid,
                                            InvalidToolCallAction::fail(),
                                        ),
                                    (None, Unhandled::Ignore) => {
                                        run.ignore_streamed_invalid_tool_call()
                                    }
                                };
                                match resolved {
                                    Ok(resolution @ StreamedResolution::Repaired { .. })
                                    | Ok(resolution @ StreamedResolution::Ignored) => {
                                        let replayed =
                                            assembler.resolve_pending_invalid(&resolution);
                                        assert!(
                                            replayed.iter().all(|e| !matches!(
                                                e,
                                                StreamedTurnEvent::InvalidToolCall(_)
                                            )),
                                            "a repair names an allowed tool"
                                        );
                                    }
                                    Ok(resolution @ StreamedResolution::TurnAbandoned { .. }) => {
                                        assembler.resolve_pending_invalid(&resolution);
                                        turn_abandoned = true;
                                    }
                                    Err(error) => {
                                        assert!(
                                            program.ending == Ending::UnknownToolCall
                                                && matches!(
                                                    error,
                                                    PromptError::UnknownToolCall { .. }
                                                ),
                                            "{error:?}"
                                        );
                                        unknown_tool_call = true;
                                    }
                                }
                            }
                            if turn_abandoned || unknown_tool_call {
                                break;
                            }
                        }
                        if unknown_tool_call {
                            drop(stream);
                            break None;
                        }
                        if turn_abandoned {
                            // As the engine: the abandoned turn's stream is drained
                            // for its usage, then the next model call is asked for.
                            while within(stream.next()).await.is_some() {}
                            drop(stream);
                            continue;
                        }
                        if let Some(reason) = delta_stop {
                            // As the engine: the model's stream is dropped at
                            // the delta, so the dispatch is recorded as a cancel.
                            drop(stream);
                            for _ in 0..64 {
                                tokio::task::yield_now().await;
                            }
                            cancelled = Some(reason);
                            break None;
                        }
                        if program.cancel_after_first_delta {
                            drop(stream);
                            for _ in 0..64 {
                                tokio::task::yield_now().await;
                            }
                            break None;
                        }
                        if program.ending == Ending::ProviderError {
                            assert!(
                                provider_failed,
                                "the stream fails with the provider's error"
                            );
                            drop(stream);
                            break None;
                        }
                        let usage = stream.usage();
                        let snapshot = stream.snapshot();
                        let streamed = assembler.finish(stream.message_id.clone(), &snapshot);
                        ModelTurn::new(
                            streamed.message_id,
                            streamed.choice,
                            usage,
                            streamed.executable_tool_names,
                            streamed.allowed_tool_names,
                        )
                    } else {
                        let response = match (within(model.complete(request)).await, program.ending)
                        {
                            (Ok(response), _) => response,
                            (Err(report), Ending::ProviderError)
                                if report.kind == rig_core::error::ErrorKind::ProviderResponse =>
                            {
                                break None;
                            }
                            (Err(report), Ending::Failed(kind)) if report.kind == kind => {
                                break None;
                            }
                            (Err(report), _) => {
                                panic!("the replayer recognised the request: {report:?}")
                            }
                        };
                        ModelTurn::from_response_parts(&response, executable, allowed)
                    };
                    let choice = turn.choice.clone();
                    let mut outcome = run.model_response(turn).expect("a model turn");
                    let mut unknown_tool_call = false;
                    while let ModelTurnOutcome::NeedsResolution(invalid) = outcome {
                        // The hook's decision, else the runner's policy — as
                        // the engine resolves an invalid call.
                        let action = if program.hooks.contains(&Hook::RetryUnknownTool) {
                            Some(retry_feedback(&invalid.tool_name))
                        } else if program.hooks.contains(&Hook::RepairToAdd) {
                            Some(InvalidToolCallAction::repair("add"))
                        } else if program.hooks.contains(&Hook::SkipUnknown) {
                            Some(InvalidToolCallAction::skip(SKIP_REASON))
                        } else {
                            None
                        };
                        let resolved = match action {
                            Some(action) => run.resolve_invalid_tool_call(action),
                            None => run.resolve_unhandled_invalid_tool_call(),
                        };
                        match resolved {
                            Ok(next) => outcome = next,
                            Err(error) => {
                                assert!(
                                    program.ending == Ending::UnknownToolCall
                                        && matches!(error, PromptError::UnknownToolCall { .. }),
                                    "{error:?}"
                                );
                                unknown_tool_call = true;
                                break;
                            }
                        }
                    }
                    if unknown_tool_call {
                        break None;
                    }
                    // The outcome and model-turn hooks, as the engine settles a
                    // turn: a replacement of the accepted choice, then a retry.
                    let has_tool_calls = choice
                        .iter()
                        .any(|content| matches!(content, AssistantContent::ToolCall(_)));
                    // The outcome hook's cancel on a text answer, then the
                    // model-turn hooks' stops, as the engine settles a turn.
                    if program.hooks.contains(&Hook::CancelAnswer) && !has_tool_calls {
                        cancelled = Some(CANCEL_ANSWER);
                        break None;
                    }
                    if program.hooks.contains(&Hook::StopAfterTurn) {
                        cancelled = Some(STOP_AFTER_TURN);
                        break None;
                    }
                    // The stateful stop, at its turn: the reason the hook formats is
                    // the program's ending, a literal.
                    if let Some(n) = program.hooks.iter().find_map(|hook| match hook {
                        Hook::StopAfterTurnN(n) if *n == turn_number => Some(*n),
                        _ => None,
                    }) {
                        let Ending::Cancelled(expected) = program.ending else {
                            panic!("a stateful stop ends the run");
                        };
                        assert_eq!(stop_after_turn_reason(n), expected);
                        cancelled = Some(expected);
                        break None;
                    }
                    if program.hooks.contains(&Hook::StopAtAnswer) && !has_tool_calls {
                        cancelled = Some(STOP_AT_ANSWER);
                        break None;
                    }
                    if program.hooks.contains(&Hook::ReplaceAnswer) && !has_tool_calls {
                        run.replace_accepted_turn_choice(vec![AssistantContent::text(
                            REPLACED_ANSWER,
                        )])
                        .expect("a text turn is replaceable");
                    }
                    if program.hooks.contains(&Hook::DemandDone)
                        && !has_tool_calls
                        && !answer_text(&choice).contains("DONE")
                    {
                        run.retry_model_turn(RetryRequest::Feedback(DONE_FEEDBACK.to_owned()))
                            .expect("a text turn is retryable");
                    }
                }
                AgentRunStep::CallTools { calls } => {
                    if program.cancel_when_reached {
                        // As the engine: the calls are dropped once the nested
                        // child is reached, and the chain is cancelled.
                        let reached = replay.reached.clone().expect("a nesting program");
                        let calling = call_tools(
                            calls,
                            &tools,
                            program.tool_concurrency.unwrap_or(1),
                            program.hooks,
                            notes.as_ref(),
                        );
                        tokio::select! {
                            results = within(calling) => {
                                panic!("the tool answered before its child was reached: {results:?}")
                            }
                            () = within(reached.notified()) => {}
                        }
                        for _ in 0..64 {
                            tokio::task::yield_now().await;
                        }
                        break None;
                    }
                    let results = match call_tools(
                        calls,
                        &tools,
                        program.tool_concurrency.unwrap_or(1),
                        program.hooks,
                        notes.as_ref(),
                    )
                    .await
                    {
                        Ok(results) => results,
                        Err(reason) => {
                            cancelled = Some(reason);
                            break None;
                        }
                    };
                    run.tool_results(results).expect("results for every call");
                    tool_turns_done += 1;
                    if resume_after == Some(tool_turns_done) && resumed.is_none() {
                        // The suspension: the state a driver persists between
                        // steps, with the next model call pending.
                        resumed =
                            Some(serde_json::to_string(&run).expect("the run state serializes"));
                        break None;
                    }
                }
                AgentRunStep::Done(response) => break Some(response),
            }
        };
        let response = match resumed.take() {
            None => response,
            Some(state) => {
                let tail = match resume {
                    Resume::Checkpoint {
                        tool_turns,
                        against,
                        ..
                    } => {
                        // The cut: the head's position and serialized run as
                        // a checkpoint, round-tripped as JSON, then the
                        // continuation it names.
                        let at = replay
                            .log
                            .iter()
                            .position(|record| {
                                record.id.as_u64() > head_end_id(&replay.log, tool_turns)
                            })
                            .unwrap_or(replay.log.len());
                        let (checkpoint, tail) = replay.log.checkpoint(
                            at,
                            serde_json::from_str::<serde_json::Value>(&state)
                                .expect("the run state is JSON"),
                        );
                        let checkpoint: Checkpoint<serde_json::Value> = serde_json::from_str(
                            &serde_json::to_string(&checkpoint).expect("a checkpoint serializes"),
                        )
                        .expect("a checkpoint restores");
                        assert_eq!(checkpoint.at, at);
                        match against {
                            Against::Tail => EffectLog::from_checkpoint(&checkpoint, tail)
                                .expect("the tail follows its checkpoint"),
                            Against::FullLog => {
                                // The full log in the tail's place is refused
                                // by its first id, before any dispatch.
                                let refused =
                                    EffectLog::from_checkpoint(&checkpoint, replay.log.clone())
                                        .expect_err("a full log is not the tail");
                                assert!(
                                    refused.message.starts_with(&format!(
                                        "resume refused: the checkpoint at {at} expects record"
                                    )) && refused.message.ends_with(&format!(
                                        "the tail begins at {}",
                                        replay.log[0].id
                                    )),
                                    "{}",
                                    refused.message
                                );
                                drop((model, route, late_route, tools, memory, context, notes));
                                replay.close().await;
                                return;
                            }
                        }
                    }
                    Resume::AfterFirstToolTurn => tail_after(&replay.log, 1),
                    Resume::Never => unreachable_resume(),
                };
                match resumed_tail(&replay, program, server.clone(), state, tail).await {
                    Ok(response) => response,
                    Err(reason) => {
                        cancelled = Some(reason);
                        None
                    }
                }
            }
        };
        if let Ending::Cancelled(expected) = program.ending {
            assert_eq!(
                cancelled,
                Some(expected),
                "the driver stopped where the hook does"
            );
        } else {
            assert_eq!(cancelled, None, "no hook stops this program");
        }
        let Some(response) = response else {
            drop((model, route, late_route, tools, memory, context));
            let log = replay.log.clone();
            let replayed = replay.close().await;
            assert_same_records(&replayed, &log, "hand driver");
            return;
        };
        if let (Some((handle, id)), None) = (&memory, program.history) {
            // As the engine: a failed append is logged and the answer stands.
            if let Err(report) =
                within(handle.append(id.clone(), response.messages.clone().unwrap_or_default()))
                    .await
            {
                assert_eq!(report.kind, rig_core::error::ErrorKind::MemoryBackend);
            }
        }
        // The settled hooks, once per run, after the append: the clear,
        // the host note.
        if program.hooks.contains(&Hook::ClearAtSettled) && !resumes {
            let (handle, id) = memory.as_ref().expect("a memory program");
            within(handle.clear(id.clone()))
                .await
                .expect("the replayer answered the clear");
        }
        if program.hooks.contains(&Hook::NoteAtSettled) && !resumes {
            note("settled").await;
        }
        last_response = Some(response);
    }
    let response = last_response.expect("a run");
    assert_eq!(
        response.output,
        program
            .expected_output
            .map_or_else(|| golden_answer(&replay.log), str::to_owned)
    );
    drop((model, route, late_route, tools, memory, context, notes));
    let log = replay.log.clone();
    let replayed = replay.close().await;
    assert_same_records(&replayed, &log, "hand driver");
}

/// Both interpreters, as two tests each, for the rows named: `test: PROGRAM`.
#[macro_export]
macro_rules! both_interpreters {
    ($($test:ident: $program:ident),* $(,)?) => {
        mod bus_engine {
            $(
                #[tokio::test]
                async fn $test() {
                    $crate::corpus::bus_engine_reproduces(&super::$program).await;
                }
            )*
        }
        mod hand_driver {
            $(
                #[tokio::test]
                async fn $test() {
                    $crate::corpus::hand_driver_reproduces(&super::$program).await;
                }
            )*
        }
        /// The third interpreter: the program as an agent graph in a Bevy
        /// world.
        mod world_agent {
            $(
                #[test]
                fn $test() {
                    $crate::corpus::world::world_agent_reproduces(&super::$program);
                }
            )*
        }
    };
}

pub mod world;
pub mod world_hooks;
pub mod world_nesting;
pub mod world_resume;
