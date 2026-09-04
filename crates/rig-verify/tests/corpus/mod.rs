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
//! | completion transport | unary · streamed, events dropped · streamed, events kept |
//! | tool shape | none · one call then answer · two calls in one turn · two turns · zero-arg tool · a tool that errors |
//! | tool id wire | provider id (anthropic) · id-less, minted `tool-<n>` (gemini) · dual `call_id`/`item_id` (openai) |
//! | serving | `serial_per_handler` false · true; `tool_concurrency` 1 · 2; capacities default · 1 |
//! | memory | none · `Load` + `Append` · `Load` of an empty conversation |
//! | retrieval | none · `dynamic_context(n, index)` (`TopN`) · `retrieved_tools(n, index, toolset)` (`TopNIds`) · both |
//! | embedding, rerank | never dispatched by the agent: an index embeds its query inside the handler (`RetrieveAdapter`), and nothing in `rig-agent` reranks; a host dispatches those families over its own bus |
//! | hooks | none · observe-only · `on_dispatch` → `Patch` · `Deny` · `on_outcome` → `Replace` · `on_invalid_tool_call` → `Retry` · `on_completion_call` → request patch · a hook that dispatches through `HookContext` |
//! | model routing | one model · `model_route` with `on_model_select` choosing the other |
//! | output | text · `output_schema` |
//! | bus ownership | own bus (`bus` in the header) · a host's bus via `over_bus` (`bus: None`) |
//! | run continuation | one run · serialize mid-run, resume on a fresh bus |
//! | outcome kind | success · `Cancelled` · handler error (`ErrorReport`) · a divergence (refused) |
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
use rig_agent::run::{OutputMode, UnhandledInvalidToolCall};
use rig_agent::{
    AgentBuilder, AgentHook, HookContext,
    agent::{
        CompletionCallAction, CompletionCallEvent, DispatchAction, DispatchEvent, ModelTurnAction,
        ModelTurnFinished, MultiTurnStreamItem, OutcomeAction, OutcomeEvent, RequestPatch,
        RetryRequest, RunStart, RunStartAction, StepEventKind, StreamingError,
    },
    completion::PromptError,
    run::{
        AgentRun, AgentRunStep, InvalidToolCallAction, InvalidToolCallContext, ModelTurn,
        ModelTurnOutcome, PendingToolCall, RunSpec, StreamedResolution, StreamedTurnAssembler,
        StreamedTurnEvent, prepare_request,
    },
    tool::{RegisteredTool, server::ToolServer},
};
use rig_bus::{Bus, Dispatcher, MemoryHandle, ModelHandle, ToolHandle};
use rig_core::{
    completion::{CompletionRequestBuilder, Document},
    effect::{EffectFamily, EffectRecord, HandlerKey},
    id::ConversationId,
    message::ToolChoice,
    message::{AssistantContent, Message, UserContent},
    tool::{ToolContext, ToolOutput},
    transcript::tool_result_output,
};
use rig_effect_log::{EffectLog, EffectLogRecorder, EffectLogReplayer};

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
    /// `PromptError::PromptCancelled` with this reason: a hook stopped the
    /// run. The records are those the engine made before the stop.
    Cancelled(&'static str),
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
}

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

fn retry_feedback(tool_name: &str) -> InvalidToolCallAction {
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

struct PatchAddArgs;

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
                context: ToolContext::new(),
            })
            .await
            .expect("add answers");
        assert_eq!(answer.result.output().render(), "3");
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

pub fn golden(fixture: &str) -> EffectLog {
    let path = format!(
        "{}/fixtures/{fixture}.effects.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("the golden fixture is committed");
    serde_json::from_str(&text).expect("the golden fixture loads")
}

/// A record as data: its kind, its outcome and its events, if any.
pub fn as_data(record: &EffectRecord) -> serde_json::Value {
    serde_json::json!({
        "key": record.key,
        "kind": record.kind,
        "outcome": record.outcome,
        "events": record.events,
    })
}

pub fn assert_same_records(replayed: &EffectLog, log: &EffectLog, interpreter: &str) {
    let replayed: Vec<_> = replayed.iter().map(as_data).collect();
    let recorded: Vec<_> = log.iter().map(as_data).collect();
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
    pub registrar: rig_bus::Registrar,
    pub recorder: EffectLogRecorder,
    pub driver: tokio::task::JoinHandle<()>,
    pub model_key: HandlerKey,
    pub memory_key: HandlerKey,
}

impl Replay {
    pub fn open(program: &Program) -> Self {
        let log = golden(program.fixture);
        EffectLogReplayer::check_header(&log).expect("a current format");
        // A golden recorded over a host's bus names no policy: the host
        // sized its bus. The replay's host uses the default.
        let bus = log.header.bus.unwrap_or_default();
        let (dispatcher, registrar, mut driver) = Bus::channel_with(bus);
        let model_key = HandlerKey::from(format!("{}/model:default", program.owner));
        let memory_key = HandlerKey::from(format!("{}/memory", program.owner));
        let model = EffectLogReplayer::for_key(&log, &model_key).expect("the model's records");
        driver
            .register_erased(
                model_key.clone(),
                rig_core::serve::ErasedHandler::new(model),
            )
            .expect("a fresh key");
        let recorder = if keeps_events(&log) {
            EffectLogRecorder::keeping_stream_events()
        } else {
            EffectLogRecorder::new()
        };
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
        }
    }

    pub fn route_key(&self, label: &str) -> HandlerKey {
        HandlerKey::from(format!("{}/model:{label}", self.log_owner()))
    }

    pub fn log_owner(&self) -> String {
        self.model_key
            .as_str()
            .rsplit_once("/model:")
            .map(|(owner, _)| owner.to_owned())
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
            let name = key
                .as_str()
                .rsplit_once("tool:")
                .map(|(_, rest)| rest.split_once('#').map_or(rest, |(name, _)| name))
                .expect("a tool key names its tool");
            let replayer =
                EffectLogReplayer::for_key(&self.log, &key).expect("a required tool is described");
            let tool = RegisteredTool::from_handler(replayer).expect("a tool-family replayer");
            if program.retrievable.contains(&name) {
                retrievable.add_registered(tool);
            } else {
                server = server.registered_tool(tool);
            }
        }
        if let Some(sample) = program.retrieved_tools {
            let key = HandlerKey::from(format!("{}/retrieve:tools#0", self.log_owner()));
            let replayer =
                EffectLogReplayer::for_key(&self.log, &key).expect("the tool index's records");
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

pub async fn bus_engine_reproduces(program: &Program) {
    let replay = Replay::open(program);
    let server = replay.tool_server_for(program);
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
        let memory = EffectLogReplayer::for_key(&replay.log, &replay.memory_key)
            .expect("the conversation's records");
        builder = builder.memory_handler(memory).conversation(conversation);
    }
    // A route is the agent's to register (`model_route_handler`), as the
    // producer's `model_route` was; the host bus serves only the default
    // model, as the producer's client did.
    if let Some(label) = program.route {
        let route = EffectLogReplayer::for_key(&replay.log, &replay.route_key(label))
            .expect("the route is in the required row");
        builder = builder.model_route_handler(label, route);
    }
    if let Some(samples) = program.dynamic_context {
        let index = EffectLogReplayer::for_key(&replay.log, &replay.context_key())
            .expect("the context index's records");
        builder = builder.dynamic_context_handler(samples, index);
    }
    let agent = builder.build();
    agent
        .check_replayable(&replay.log)
        .expect("the same program as the one recorded");

    let output = if program.streamed {
        let mut runner = agent.stream_prompt(program.prompt);
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
                Err(StreamingError::Completion(error))
                    if program.ending == Ending::ProviderError =>
                {
                    let _ = error;
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
        let mut runner = agent.prompt(program.prompt);
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
        match (within(runner.run()).await, program.ending) {
            (Ok(response), Ending::Answer) => Some(response.output),
            (Err(PromptError::MaxTurnsError { .. }), Ending::MaxTurns)
            | (Err(PromptError::UnknownToolCall { .. }), Ending::UnknownToolCall) => None,
            (Err(PromptError::Report(report)), Ending::ProviderError)
                if report.kind == rig_core::error::ErrorKind::ProviderResponse =>
            {
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
        let answer = within(handle.call(name.clone(), args, ToolContext::new()))
            .await
            .expect("the replayer answered the recorded call");
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

/// The run spec the producer's builder and runner amount to.
/// The name the header records for a hook: its type's last path segment.
fn hook_name(hook: Hook) -> &'static str {
    match hook {
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
    }
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
    let hooks: Vec<&str> = program
        .dynamic_context
        .map(|_| "DynamicContext")
        .into_iter()
        .chain(program.hooks.iter().map(|hook| hook_name(*hook)))
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

pub async fn hand_driver_reproduces(program: &Program) {
    let replay = Replay::open(program);
    let server = replay.tool_server_for(program);
    server.attach(&replay.registrar);
    // The context index, registered by the driver as the builder would.
    let context: Option<rig_bus::Handle<rig_core::effect::family::Retrieve>> =
        program.dynamic_context.map(|_| {
            let key = replay.context_key();
            let replayer =
                EffectLogReplayer::for_key(&replay.log, &key).expect("the context index's records");
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
        let replayer = EffectLogReplayer::for_key(&replay.log, &key)
            .expect("the route is in the required row");
        replay
            .registrar
            .register_erased(key.clone(), rig_core::serve::ErasedHandler::new(replayer))
            .expect("a fresh key");
        replay.dispatcher.handle(&key).expect("the route")
    });
    let memory: Option<(MemoryHandle, ConversationId)> = program.conversation.map(|id| {
        let replayer = EffectLogReplayer::for_key(&replay.log, &replay.memory_key)
            .expect("the conversation's records");
        replay
            .registrar
            .register_erased(
                replay.memory_key.clone(),
                rig_core::serve::ErasedHandler::new(replayer),
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
    let history = match (program.history, &memory) {
        (Some(history), _) => Some(history()),
        (None, Some((handle, id))) => Some(
            within(handle.load(id.clone()))
                .await
                .expect("the replayer answered the load"),
        ),
        (None, None) => None,
    };
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
    let patch = program
        .hooks
        .contains(&Hook::PreambleOverride)
        .then(|| RequestPatch::new().preamble(PIRATE_PREAMBLE));
    let mut run = AgentRun::from_spec(&spec, program.prompt, history);
    // The routing hook selects the route once a model has been asked
    // (`previous_model` is set): the first call goes to the default.
    let mut asked_before = false;
    // The hook that stopped the run, with its reason: the same decision at
    // the same point the engine makes it.
    let mut cancelled: Option<&'static str> = None;
    // A stop before any dispatch: at run start, at model selection, or
    // before the completion call.
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
                let model = match (&route, program.hooks.contains(&Hook::RouteAfterFirstTurn)) {
                    (Some(route), true) if asked_before => route,
                    _ => &model,
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
                let mut turn_patch = patch.clone();
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
                run.set_output_tool_name(prepared.output_tool_name.clone());
                run.advertise_tools(turn, prepared.tools.clone());
                let executable = prepared.executable_tool_names.clone();
                let allowed = prepared.allowed_tool_names.clone();
                let request = prepared
                    .apply(CompletionRequestBuilder::unbound(prompt))
                    .build();
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
                            Err(report) => {
                                panic!("the replayer re-emitted the recorded stream: {report:?}")
                            }
                        };
                        // The observe-only hooks' stops, at the delta they
                        // fire on: the engine leaves the stream there.
                        if let rig_core::streaming::StreamEvent::BlockDelta { delta, .. } = &event {
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
                                (Some(action), _) => run
                                    .resolve_streamed_invalid_tool_call(&partial, &invalid, action),
                                (None, Unhandled::Fail) => run.resolve_streamed_invalid_tool_call(
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
                                    let replayed = assembler.resolve_pending_invalid(&resolution);
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
                                            && matches!(error, PromptError::UnknownToolCall { .. }),
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
                    let response = match (within(model.complete(request)).await, program.ending) {
                        (Ok(response), _) => response,
                        (Err(report), Ending::ProviderError)
                            if report.kind == rig_core::error::ErrorKind::ProviderResponse =>
                        {
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
                if program.hooks.contains(&Hook::StopAtAnswer) && !has_tool_calls {
                    cancelled = Some(STOP_AT_ANSWER);
                    break None;
                }
                if program.hooks.contains(&Hook::ReplaceAnswer) && !has_tool_calls {
                    run.replace_accepted_turn_choice(vec![AssistantContent::text(REPLACED_ANSWER)])
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
                let results = match call_tools(
                    calls,
                    &tools,
                    program.tool_concurrency.unwrap_or(1),
                    program.hooks,
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
            }
            AgentRunStep::Done(response) => break Some(response),
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
        drop((model, route, tools, memory, context));
        let log = replay.log.clone();
        let replayed = replay.close().await;
        assert_same_records(&replayed, &log, "hand driver");
        return;
    };
    if let (Some((handle, id)), None) = (&memory, program.history) {
        within(handle.append(id.clone(), response.messages.clone().unwrap_or_default()))
            .await
            .expect("the replayer answered the append");
    }
    assert_eq!(
        response.output,
        program
            .expected_output
            .map_or_else(|| golden_answer(&replay.log), str::to_owned)
    );
    drop((model, route, tools, memory, context));
    let log = replay.log.clone();
    let replayed = replay.close().await;
    assert_same_records(&replayed, &log, "hand driver");
}

/// The run continued: the hand driver takes the program up to and
/// including its first tool call's result, serializes the `AgentRun`, and
/// the bus engine resumes it on the same replay bus — whose replayers have
/// answered the head and hold the tail — to the golden's answer. The
/// recorded log is the whole golden: the head by hand, the tail by the
/// engine, one record sequence.
pub async fn resume_reproduces(program: &Program) {
    assert!(
        program.hooks.is_empty() && program.conversation.is_none() && program.route.is_none(),
        "resume rows are plain tool programs"
    );
    let replay = Replay::open(program);
    let server = replay.tool_server();
    server.attach(&replay.registrar);
    let tools = tool_handles(&replay);
    let model: ModelHandle = replay
        .dispatcher
        .handle(&replay.model_key)
        .expect("the model");
    assert_header_names_the_program(&replay, program);
    let spec = run_spec(program);
    let definitions = server.static_tool_defs();
    let mut run = AgentRun::from_spec(&spec, program.prompt, None);
    // Up to and including the first tool turn's results; then suspended
    // with the next model call pending, the state a driver persists
    // between steps.
    loop {
        match run.next_step().expect("a step") {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                let prepared = prepare_request(
                    &spec,
                    &model.capabilities(),
                    &history,
                    definitions.clone(),
                    run.output_tool_name(),
                    None,
                )
                .expect("prepared");
                run.set_output_tool_name(prepared.output_tool_name.clone());
                run.advertise_tools(turn, prepared.tools.clone());
                let executable = prepared.executable_tool_names.clone();
                let allowed = prepared.allowed_tool_names.clone();
                let request = prepared
                    .apply(CompletionRequestBuilder::unbound(prompt))
                    .build();
                let turn = if program.streamed {
                    let mut stream = model.stream(request);
                    let mut assembler = StreamedTurnAssembler::new(executable, allowed);
                    while let Some(event) = within(stream.next()).await {
                        let event = event.expect("the replayer re-emitted the recorded stream");
                        assembler.ingest(&event).expect("a well-formed stream");
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
                    let response = within(model.complete(request))
                        .await
                        .expect("the replayer recognised the request");
                    ModelTurn::from_response_parts(&response, executable, allowed)
                };
                run.model_response(turn).expect("a model turn");
            }
            AgentRunStep::CallTools { calls } => {
                let results = call_tools(calls, &tools, program.tool_concurrency.unwrap_or(1), &[])
                    .await
                    .expect("no hook stops a resume row");
                run.tool_results(results).expect("results for every call");
                break;
            }
            AgentRunStep::Done(_) => panic!("the program has a tool turn"),
        }
    }
    let state = serde_json::to_string(&run).expect("the run state serializes");
    drop((model, tools));
    let restored: AgentRun = serde_json::from_str(&state).expect("the run state restores");

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
    if let Some(temperature) = program.temperature {
        builder = builder.temperature(temperature);
    }
    if let Some(default_max_turns) = program.default_max_turns {
        builder = builder.default_max_turns(default_max_turns);
    }
    let agent = builder.build();
    agent
        .check_replayable(&replay.log)
        .expect("the same program as the one recorded");
    let mut runner = agent.runner("ignored").resume(restored);
    if let Some(max_turns) = program.max_turns {
        runner = runner.max_turns(max_turns);
    }
    if let Some(concurrency) = program.tool_concurrency {
        runner = runner.tool_concurrency(concurrency);
    }
    // The medium the producer ran on: a streamed program's tail asks the
    // model for a stream, as its record says.
    let output = if program.streamed {
        let mut stream = runner.stream().await;
        let mut output = None;
        while let Some(item) = within(stream.next()).await {
            match item {
                Ok(MultiTurnStreamItem::FinalResponse(response)) => output = Some(response.output),
                Ok(_) => {}
                Err(error) => panic!("the resumed stream: {error:?}"),
            }
        }
        drop(stream);
        output.expect("the resumed stream yields a final response")
    } else {
        within(runner.run()).await.expect("the resumed run").output
    };
    assert_eq!(output, golden_answer(&replay.log));
    drop(agent);
    let log = replay.log.clone();
    let replayed = replay.close().await;
    assert_same_records(&replayed, &log, "hand driver head, bus engine tail");
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
    };
}
