//! Golden effect logs: the effect-bus cassette corpus.
//!
//! A producing test runs an agent program against the cassette transport
//! with `record_effects()` and either writes the log to
//! `crates/rig-verify/fixtures/<name>.effects.json` (under
//! `RIG_REGENERATE_GOLDEN=1`) or asserts the run's log equals the committed
//! one as data — so the root suite itself detects drift between a cassette
//! and its golden. rig-verify replays every golden with no provider at
//! all. Goldens are re-recorded by their producer, never edited by hand.

use rig::effect::EffectFamily;
use rig::message::Message;
use rig_effect_log::EffectLog;

/// The families of a log's records, in order: the shape a producer asserts.
#[allow(dead_code)] // not every target records
pub(crate) fn families(log: &EffectLog) -> Vec<EffectFamily> {
    log.records
        .iter()
        .map(|record| record.kind.family())
        .collect()
}

/// The output schema the request-shape matrix constrains an answer to,
/// as one literal both the producer and the rig-verify replay build the
/// program from (`crates/rig-verify/tests/corpus_request_shape.rs`).
#[allow(dead_code)]
pub(crate) const EVENT_SCHEMA: &str = r#"{"type":"object","properties":{"title":{"type":"string"},"category":{"type":"string"},"summary":{"type":"string"}},"required":["title","category","summary"]}"#;

#[allow(dead_code)]
pub(crate) fn event_schema() -> schemars::Schema {
    serde_json::from_str(EVENT_SCHEMA).expect("the schema literal is a schema")
}

/// The prior history the request-shape matrix's history cell runs with;
/// the replay builds the same two turns.
#[allow(dead_code)]
pub(crate) fn prior_history() -> Vec<Message> {
    vec![
        Message::user("My name is Ada."),
        Message::assistant("Nice to meet you, Ada."),
    ]
}

/// The committed golden's path.
pub(crate) fn golden_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates/rig-verify/fixtures")
        .join(format!("{name}.effects.json"))
}

/// Write `log` as the golden `name` under `RIG_REGENERATE_GOLDEN=1`, else
/// assert it equals the committed golden byte for byte (the header is part
/// of the oracle: a program that changed refuses before it diverges).
///
/// In record mode this is a no-op: a golden is generated from the
/// *replayed* cassette, never from a live recording, because the cassette
/// is written with placeholders for provider ids (`msg_REDACTED_1`,
/// `toolu_REDACTED_1`, …) and the golden must hold the same, or the first
/// replay diverges on an id the record never held. A panic here would
/// also discard the cassette the run just recorded (the wrapper writes a
/// cassette only when the test body returns), so the loop is: record on
/// the producer's filter, then regenerate the golden in replay mode.
pub(crate) fn golden_effects(name: &str, log: &EffectLog) {
    if std::env::var("RIG_PROVIDER_TEST_MODE").is_ok_and(|mode| mode.eq_ignore_ascii_case("record"))
    {
        assert!(
            std::env::var_os("RIG_REGENERATE_GOLDEN").is_none(),
            "golden `{name}`: record the cassette first, then regenerate the golden in replay mode"
        );
        return;
    }
    let rendered = serde_json::to_string_pretty(log).expect("the log serializes");
    let path = golden_path(name);
    if std::env::var_os("RIG_REGENERATE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().expect("a parent")).expect("fixtures dir");
        std::fs::write(&path, format!("{rendered}\n")).expect("the golden file writes");
        return;
    }
    let committed = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "no golden fixture at {}; run with RIG_REGENERATE_GOLDEN=1",
            path.display()
        )
    });
    assert_eq!(
        committed.trim_end(),
        rendered,
        "the agent's effects diverged from golden `{name}`; if the change is deliberate, regenerate it"
    );
}

/// The corpus's recovery hook: an unknown tool is retried once with
/// feedback. A hook is program, not record — the effect-log header names
/// it by type, so every producer that records a recovery and the
/// rig-verify replay use this one type.
#[allow(dead_code)] // used by the recovery producer, not every target
pub(crate) struct RetryUnknownTool;

impl rig::agent::AgentHook for RetryUnknownTool {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &rig::agent::HookContext,
        context: &rig::agent::InvalidToolCallContext,
    ) -> Option<rig::agent::InvalidToolCallAction> {
        Some(rig::agent::InvalidToolCallAction::Retry {
            feedback: format!("there is no tool named {}; use add", context.tool_name),
        })
    }
}

// ---------------------------------------------------------------------------
// The hook matrix's hooks (Matrix B, `tests/providers/anthropic/cassette/
// corpus_hooks.rs`, `crates/rig-verify/tests/corpus_hooks.rs`). Hooks are
// program: the header names each by type, and the rig-verify replay
// defines a type of the same name making the same decision. Every hook is
// stateless, so its decision is a function of the event alone (the
// header cannot tell two hooks of one type with different state apart).

#[allow(dead_code)]
pub(crate) const PIRATE_PREAMBLE: &str = "You are a pirate. Answer in one short sentence.";
#[allow(dead_code)]
pub(crate) const DENY_REASON: &str = "add is disabled for this run";
#[allow(dead_code)]
pub(crate) const REPLACED_RESULT: &str = "99";
#[allow(dead_code)]
pub(crate) const REPLACED_ANSWER: &str = "REPLACED";
#[allow(dead_code)]
pub(crate) const DONE_FEEDBACK: &str = "End your answer with the word DONE.";
#[allow(dead_code)]
pub(crate) const LOOKUP_ARGS: &str = r#"{"x":1,"y":2}"#;
#[allow(dead_code)]
pub(crate) const LOOKUP_KEY: &str = "golden/tool:add#0";

/// Opts into observing every dispatch family; decides nothing.
#[allow(dead_code)]
pub(crate) struct ObserveEverything;

impl rig::agent::AgentHook for ObserveEverything {
    fn observes(&self, _kind: rig::agent::StepEventKind) -> bool {
        true
    }
}

/// `on_dispatch` → `Patch`: `add` runs with `{"x":40,"y":2}` whatever the
/// model asked (the record holds the patched call; history keeps the
/// model's).
#[allow(dead_code)]
pub(crate) struct PatchAddArgs;

impl rig::agent::AgentHook for PatchAddArgs {
    async fn on_dispatch(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::DispatchEvent<'_>,
    ) -> rig::agent::DispatchAction {
        if event.tool_name() == Some("add") {
            rig::agent::DispatchAction::rewrite_tool_args(
                event.kind,
                serde_json::json!({"x": 40, "y": 2}),
            )
        } else {
            rig::agent::DispatchAction::proceed()
        }
    }
}

/// `on_dispatch` → `Deny` (a skip): `add` never reaches the bus; the model
/// sees the reason as the tool's result.
#[allow(dead_code)]
pub(crate) struct DenyAdd;

impl rig::agent::AgentHook for DenyAdd {
    async fn on_dispatch(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::DispatchEvent<'_>,
    ) -> rig::agent::DispatchAction {
        if event.tool_name() == Some("add") {
            rig::agent::DispatchAction::skip(DENY_REASON)
        } else {
            rig::agent::DispatchAction::proceed()
        }
    }
}

/// `on_outcome` → `Replace`: the model sees `99` for `add`, the record
/// holds what the tool answered.
#[allow(dead_code)]
pub(crate) struct ReplaceAddResult;

impl rig::agent::AgentHook for ReplaceAddResult {
    async fn on_outcome(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::OutcomeEvent<'_>,
    ) -> rig::agent::OutcomeAction {
        if event.tool_name() == Some("add") && event.tool_result().is_some() {
            rig::agent::OutcomeAction::rewrite_tool_result(&event, REPLACED_RESULT)
        } else {
            rig::agent::OutcomeAction::proceed()
        }
    }
}

/// `on_outcome` → `Replace` on a completion: a text answer is replaced by
/// `REPLACED`; the record holds the model's.
#[allow(dead_code)]
pub(crate) struct ReplaceAnswer;

impl rig::agent::AgentHook for ReplaceAnswer {
    async fn on_outcome(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::OutcomeEvent<'_>,
    ) -> rig::agent::OutcomeAction {
        let Some(response) = event.completion() else {
            return rig::agent::OutcomeAction::proceed();
        };
        if response
            .choice
            .iter()
            .any(|content| matches!(content, rig::message::AssistantContent::ToolCall(_)))
        {
            return rig::agent::OutcomeAction::proceed();
        }
        let mut replacement = response.clone();
        replacement.choice = vec![rig::message::AssistantContent::text(REPLACED_ANSWER)];
        rig::agent::OutcomeAction::replace(Ok(rig::effect::Outcome::Completion(replacement)))
    }
}

/// `on_completion_call` → a request patch overriding the preamble: the
/// request holds the pirate preamble, the spec holds the base.
#[allow(dead_code)]
pub(crate) struct PreambleOverride;

impl rig::agent::AgentHook for PreambleOverride {
    async fn on_completion_call(
        &self,
        _ctx: &rig::agent::HookContext,
        _event: rig::agent::CompletionCallEvent<'_>,
    ) -> rig::agent::CompletionCallAction {
        rig::agent::CompletionCallAction::patch(
            rig::agent::RequestPatch::new().preamble(PIRATE_PREAMBLE),
        )
    }
}

/// `on_model_turn_finished` → `Retry` with feedback until the answer holds
/// `DONE`: a second completion is a record, the decision is program.
#[allow(dead_code)]
pub(crate) struct DemandDone;

impl rig::agent::AgentHook for DemandDone {
    async fn on_model_turn_finished(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::ModelTurnFinished<'_>,
    ) -> rig::agent::ModelTurnAction {
        let text: String = event
            .content
            .iter()
            .filter_map(|content| match content {
                rig::message::AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        if text.contains("DONE") {
            rig::agent::ModelTurnAction::continue_run()
        } else {
            rig::agent::ModelTurnAction::retry_with_feedback(DONE_FEEDBACK)
        }
    }
}

/// `on_run_start` dispatches `add(1, 2)` through the run's bus: a hook's
/// own effect is a record under the tool's key, before the first
/// completion.
#[allow(dead_code)]
pub(crate) struct LookupBeforeRun;

impl rig::agent::AgentHook for LookupBeforeRun {
    async fn on_run_start(
        &self,
        ctx: &rig::agent::HookContext,
        _event: rig::agent::RunStart<'_>,
    ) -> rig::agent::RunStartAction {
        let tool = ctx
            .tool(&rig::effect::HandlerKey::from(LOOKUP_KEY))
            .expect("the run's bus serves add");
        let answer = tool
            .dispatch(rig::effect::ToolCallRequest {
                name: "add".to_owned(),
                args: LOOKUP_ARGS.to_owned(),
                context: rig::tool::ToolContext::new(),
            })
            .await
            .expect("add answers");
        assert_eq!(answer.result.output().render(), "3");
        rig::agent::RunStartAction::continue_run()
    }
}

/// `on_model_select` → `Select("fast")` on every turn after the first:
/// the route answers once the default model has been asked once.
#[allow(dead_code)]
pub(crate) struct RouteAfterFirstTurn;

impl rig::agent::AgentHook for RouteAfterFirstTurn {
    fn on_model_select(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::ModelSelection<'_>,
    ) -> rig::agent::ModelSelectionAction {
        if event.previous_model.is_some() {
            rig::agent::ModelSelectionAction::select("fast")
        } else {
            rig::agent::ModelSelectionAction::continue_run()
        }
    }
}

// ---------------------------------------------------------------------------
// The outcome matrix's tool (Matrix D).

#[allow(dead_code)]
pub(crate) const BROKEN_ADD: &str = "the adder is broken";

#[derive(serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct FailingAddArgs {
    pub(crate) x: i64,
    pub(crate) y: i64,
}

/// An `add` that fails every call: the tool record's outcome is a failed
/// result, which the model sees and answers around.
#[allow(dead_code)]
pub(crate) struct FailingAdd;

impl rig::tool::Tool for FailingAdd {
    const NAME: &'static str = "add";
    type Args = FailingAddArgs;
    type Output = i64;
    type Error = rig::tool::ToolExecutionError;

    fn description(&self) -> String {
        "adds two integers".into()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]})
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: FailingAddArgs,
    ) -> Result<i64, Self::Error> {
        Err(rig::tool::ToolExecutionError::other(BROKEN_ADD))
    }
}

// ---------------------------------------------------------------------------
// The outcome matrix's long-argument tool: a call whose arguments stream
// for long enough that a consumer's drop lands mid-call.

#[derive(serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct NoteArgs {
    pub(crate) title: String,
    pub(crate) body: String,
}

/// Writes a note; its `body` is what the model streams at length.
#[allow(dead_code)]
pub(crate) struct WriteNote;

impl rig::tool::Tool for WriteNote {
    const NAME: &'static str = "write_note";
    type Args = NoteArgs;
    type Output = String;
    type Error = rig::tool::ToolExecutionError;

    fn description(&self) -> String {
        "writes a note with a title and a body".into()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}}, "required": ["title", "body"]})
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: NoteArgs,
    ) -> Result<String, Self::Error> {
        Ok(format!("saved {} ({} chars)", args.title, args.body.len()))
    }
}

// ---------------------------------------------------------------------------
// The retrieval matrix (Matrix A): an index of facts for `dynamic_context`
// and a toolset of retrievable tools for `retrieved_tools`, each embedded
// by the provider under test.

#[allow(dead_code)]
pub(crate) const FACTS: [&str; 3] = [
    "A flurbo is a green alien that lives on cold planets.",
    "A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
    "A linglingdong is a term used by inhabitants of the far side of the moon to describe humans.",
];
#[allow(dead_code)]
pub(crate) const FACT_PROMPT: &str = "What is a glarb-glarb? Answer in one sentence.";
#[allow(dead_code)]
pub(crate) const RETRIEVED_TOOLS_PREAMBLE: &str =
    "You are a calculator. You must use the provided tools for every arithmetic operation.";
#[allow(dead_code)]
pub(crate) const SUBTRACT_PROMPT: &str =
    "Subtract 8 from 50 with the subtract tool, then reply with just the number.";
#[allow(dead_code)]
pub(crate) const ADD_THEN_SUBTRACT_PROMPT: &str = "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the subtract tool. Report the final number.";

/// The facts, embedded by `model`, as an in-memory index (ids `doc0`..).
#[allow(dead_code)]
pub(crate) async fn facts_index<M: rig::embeddings::EmbeddingModel + Clone>(
    model: M,
    facts: &[&str],
) -> rig::vector_store::in_memory_store::InMemoryVectorIndex<String, M> {
    let store = if facts.is_empty() {
        rig::vector_store::in_memory_store::InMemoryVectorStore::<String>::default()
    } else {
        let embeddings = rig::embeddings::EmbeddingsBuilder::new(model.clone())
            .documents(facts.iter().map(|fact| (*fact).to_owned()))
            .expect("documents should be added")
            .build()
            .await
            .expect("fact embeddings should succeed");
        rig::vector_store::in_memory_store::InMemoryVectorStore::from_documents(embeddings)
    };
    store.index(model)
}

/// The toolset's embeddable schemas, embedded by `model`, as an index keyed
/// by tool name.
#[allow(dead_code)]
pub(crate) async fn tool_index<M: rig::embeddings::EmbeddingModel + Clone>(
    model: M,
    toolset: &rig::tool::ToolSet,
) -> rig::vector_store::in_memory_store::InMemoryVectorIndex<rig::embeddings::ToolSchema, M> {
    let embeddings = rig::embeddings::EmbeddingsBuilder::new(model.clone())
        .documents(toolset.schemas().expect("tool schemas should build"))
        .expect("documents should be added")
        .build()
        .await
        .expect("tool schema embeddings should succeed");
    rig::vector_store::in_memory_store::InMemoryVectorStore::from_documents_with_id_f(
        embeddings,
        |tool| tool.name.clone(),
    )
    .index(model)
}

#[derive(Debug, thiserror::Error)]
#[error("init error")]
#[allow(dead_code)]
pub(crate) struct NoInit;

macro_rules! retrievable_operation {
    ($name:ident, $tool_name:literal, $description:literal, $embedding_doc:literal, $op:expr) => {
        #[derive(Clone, Default, serde::Deserialize, serde::Serialize)]
        #[allow(dead_code)]
        pub(crate) struct $name;

        impl rig::tool::Tool for $name {
            const NAME: &'static str = $tool_name;
            type Error = rig::tool::ToolExecutionError;
            type Args = crate::goldens::FailingAddArgs;
            type Output = i64;

            fn description(&self) -> String {
                $description.to_string()
            }

            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]})
            }

            async fn call(
                &self,
                _context: &mut rig::tool::ToolContext,
                args: Self::Args,
            ) -> Result<Self::Output, Self::Error> {
                let op: fn(i64, i64) -> i64 = $op;
                Ok(op(args.x, args.y))
            }
        }

        impl rig::tool::ToolEmbedding for $name {
            type InitError = NoInit;
            type Context = ();
            type State = ();

            fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
                Ok(Self)
            }

            fn embedding_docs(&self) -> Vec<String> {
                vec![$embedding_doc.into()]
            }

            fn context(&self) -> Self::Context {}
        }
    };
}

retrievable_operation!(
    EmbedAdd,
    "add",
    "Add x and y together",
    "Add two numbers together to get their sum",
    |x, y| x + y
);
retrievable_operation!(
    EmbedSubtract,
    "subtract",
    "Subtract y from x",
    "Subtract one number from another to get their difference",
    |x, y| x - y
);

/// The retrievable toolset: `add` and `subtract`, in that order.
#[allow(dead_code)]
pub(crate) fn retrievable_toolset() -> rig::tool::ToolSet {
    let mut toolset = rig::tool::ToolSet::default();
    toolset
        .add_retrieved_tool(EmbedAdd)
        .expect("the tool context serializes");
    toolset
        .add_retrieved_tool(EmbedSubtract)
        .expect("the tool context serializes");
    toolset
}

// ---------------------------------------------------------------------------
// The endings matrix's hooks (Matrix F): every `Stop` in the hook surface,
// each a stateless type deciding from its event alone, plus an
// observe-only hook that records what `on_run_settled` saw (producer-side
// assertion; the header names it like any other hook).

#[allow(dead_code)]
pub(crate) const STOP_AT_START: &str = "stopped at run start";
#[allow(dead_code)]
pub(crate) const STOP_AT_MODEL_SELECT: &str = "stopped at model selection";
#[allow(dead_code)]
pub(crate) const STOP_AT_COMPLETION_CALL: &str = "stopped before the completion call";
#[allow(dead_code)]
pub(crate) const CANCEL_ADD_DISPATCH: &str = "add is cancelled before the bus";
#[allow(dead_code)]
pub(crate) const CANCEL_ADD_OUTCOME: &str = "add is cancelled after the bus";
#[allow(dead_code)]
pub(crate) const CANCEL_ANSWER: &str = "the answer is cancelled";
#[allow(dead_code)]
pub(crate) const STOP_AFTER_TURN: &str = "stopped after the model turn";
#[allow(dead_code)]
pub(crate) const STOP_AT_ANSWER: &str = "stopped at the answer turn";
#[allow(dead_code)]
pub(crate) const STOP_ON_TEXT_DELTA: &str = "stopped on the first text delta";
#[allow(dead_code)]
pub(crate) const STOP_ON_TOOL_CALL_DELTA: &str = "stopped on the first tool-call delta";
#[allow(dead_code)]
pub(crate) const STOP_ON_REASONING_DELTA: &str = "stopped on the first reasoning delta";

#[allow(dead_code)]
pub(crate) struct StopAtStart;
impl rig::agent::AgentHook for StopAtStart {
    async fn on_run_start(
        &self,
        _ctx: &rig::agent::HookContext,
        _event: rig::agent::RunStart<'_>,
    ) -> rig::agent::RunStartAction {
        rig::agent::RunStartAction::stop(STOP_AT_START)
    }
}

#[allow(dead_code)]
pub(crate) struct StopAtModelSelect;
impl rig::agent::AgentHook for StopAtModelSelect {
    fn on_model_select(
        &self,
        _ctx: &rig::agent::HookContext,
        _event: rig::agent::ModelSelection<'_>,
    ) -> rig::agent::ModelSelectionAction {
        rig::agent::ModelSelectionAction::stop(STOP_AT_MODEL_SELECT)
    }
}

#[allow(dead_code)]
pub(crate) struct StopAtCompletionCall;
impl rig::agent::AgentHook for StopAtCompletionCall {
    async fn on_completion_call(
        &self,
        _ctx: &rig::agent::HookContext,
        _event: rig::agent::CompletionCallEvent<'_>,
    ) -> rig::agent::CompletionCallAction {
        rig::agent::CompletionCallAction::stop(STOP_AT_COMPLETION_CALL)
    }
}

/// `on_dispatch` → `Deny(Cancelled)` for `add`: the run stops before the
/// tool reaches the bus.
#[allow(dead_code)]
pub(crate) struct CancelAddDispatch;
impl rig::agent::AgentHook for CancelAddDispatch {
    async fn on_dispatch(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::DispatchEvent<'_>,
    ) -> rig::agent::DispatchAction {
        if event.tool_name() == Some("add") {
            rig::agent::DispatchAction::stop(CANCEL_ADD_DISPATCH)
        } else {
            rig::agent::DispatchAction::proceed()
        }
    }
}

/// `on_outcome` → `Replace(Err(Cancelled))` for `add`'s result: the tool
/// ran and is recorded; the run stops after.
#[allow(dead_code)]
pub(crate) struct CancelAddOutcome;
impl rig::agent::AgentHook for CancelAddOutcome {
    async fn on_outcome(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::OutcomeEvent<'_>,
    ) -> rig::agent::OutcomeAction {
        if event.tool_name() == Some("add") && event.tool_result().is_some() {
            rig::agent::OutcomeAction::stop(CANCEL_ADD_OUTCOME)
        } else {
            rig::agent::OutcomeAction::proceed()
        }
    }
}

/// `on_outcome` → `Replace(Err(Cancelled))` on a text answer.
#[allow(dead_code)]
pub(crate) struct CancelAnswer;
impl rig::agent::AgentHook for CancelAnswer {
    async fn on_outcome(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::OutcomeEvent<'_>,
    ) -> rig::agent::OutcomeAction {
        match event.completion() {
            Some(response)
                if !response
                    .choice
                    .iter()
                    .any(|c| matches!(c, rig::message::AssistantContent::ToolCall(_))) =>
            {
                rig::agent::OutcomeAction::stop(CANCEL_ANSWER)
            }
            _ => rig::agent::OutcomeAction::proceed(),
        }
    }
}

#[allow(dead_code)]
pub(crate) struct StopAfterTurn;
impl rig::agent::AgentHook for StopAfterTurn {
    async fn on_model_turn_finished(
        &self,
        _ctx: &rig::agent::HookContext,
        _event: rig::agent::ModelTurnFinished<'_>,
    ) -> rig::agent::ModelTurnAction {
        rig::agent::ModelTurnAction::stop(STOP_AFTER_TURN)
    }
}

/// Stops at the turn that carries no tool call — the answer turn of a
/// tool program, so the tool turn's records precede the stop.
#[allow(dead_code)]
pub(crate) struct StopAtAnswer;
impl rig::agent::AgentHook for StopAtAnswer {
    async fn on_model_turn_finished(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::ModelTurnFinished<'_>,
    ) -> rig::agent::ModelTurnAction {
        if event
            .content
            .iter()
            .any(|c| matches!(c, rig::message::AssistantContent::ToolCall(_)))
        {
            rig::agent::ModelTurnAction::continue_run()
        } else {
            rig::agent::ModelTurnAction::stop(STOP_AT_ANSWER)
        }
    }
}

#[allow(dead_code)]
pub(crate) struct StopOnTextDelta;
impl rig::agent::AgentHook for StopOnTextDelta {
    async fn on_text_delta(
        &self,
        _ctx: &rig::agent::HookContext,
        _event: rig::agent::TextDelta<'_>,
    ) -> rig::agent::ObservationAction {
        rig::agent::ObservationAction::stop(STOP_ON_TEXT_DELTA)
    }
}

#[allow(dead_code)]
pub(crate) struct StopOnToolCallDelta;
impl rig::agent::AgentHook for StopOnToolCallDelta {
    async fn on_tool_call_delta(
        &self,
        _ctx: &rig::agent::HookContext,
        _event: rig::agent::ToolCallDelta<'_>,
    ) -> rig::agent::ObservationAction {
        rig::agent::ObservationAction::stop(STOP_ON_TOOL_CALL_DELTA)
    }
}

#[allow(dead_code)]
pub(crate) struct StopOnReasoningDelta;
impl rig::agent::AgentHook for StopOnReasoningDelta {
    async fn on_reasoning_delta(
        &self,
        _ctx: &rig::agent::HookContext,
        _event: rig::agent::ReasoningDelta<'_>,
    ) -> rig::agent::ObservationAction {
        rig::agent::ObservationAction::stop(STOP_ON_REASONING_DELTA)
    }
}

/// Observe-only: what `on_run_settled` saw, for the producer to assert.
/// Not a record, and its state is not identity (the header names the
/// type); the replay's hook of the same name observes nothing.
#[derive(Clone, Default)]
#[allow(dead_code)]
pub(crate) struct RecordSettled(pub(crate) std::sync::Arc<std::sync::Mutex<Option<String>>>);
impl rig::agent::AgentHook for RecordSettled {
    async fn on_run_settled(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::RunSettled<'_>,
    ) {
        let seen = match event.outcome {
            rig::agent::SettledOutcome::Response(response) => {
                format!("response:{}", response.output)
            }
            rig::agent::SettledOutcome::Error(reason) => format!("error:{reason}"),
        };
        *self.0.lock().expect("settled") = Some(seen);
    }
}

// ---------------------------------------------------------------------------
// The invalid-call matrix's hooks (Matrix G).

#[allow(dead_code)]
pub(crate) const SKIP_REASON: &str = "no such tool; skipped";

/// `on_invalid_tool_call` → `Repair { tool_name: "add" }`: the unknown
/// call is re-targeted to `add` with its arguments.
#[allow(dead_code)]
pub(crate) struct RepairToAdd;
impl rig::agent::AgentHook for RepairToAdd {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &rig::agent::HookContext,
        _context: &rig::agent::InvalidToolCallContext,
    ) -> Option<rig::agent::InvalidToolCallAction> {
        Some(rig::agent::InvalidToolCallAction::Repair {
            tool_name: "add".to_owned(),
        })
    }
}

/// `on_invalid_tool_call` → `Skip { reason }`: the model sees the reason
/// as the call's result and goes on.
#[allow(dead_code)]
pub(crate) struct SkipUnknown;
impl rig::agent::AgentHook for SkipUnknown {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &rig::agent::HookContext,
        _context: &rig::agent::InvalidToolCallContext,
    ) -> Option<rig::agent::InvalidToolCallAction> {
        Some(rig::agent::InvalidToolCallAction::Skip {
            reason: SKIP_REASON.to_owned(),
        })
    }
}

// ---------------------------------------------------------------------------
// Matrix I: a host's own effect, dispatched by hooks over the host's bus.

/// The host's key for its custom handler.
#[allow(dead_code)]
pub(crate) const NOTE_KEY: &str = "host/note";
/// The host's key for its embedding model.
#[allow(dead_code)]
pub(crate) const EMBED_KEY: &str = "host/embed";

/// A host-defined effect: a note of where in the run it was taken.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct Note {
    pub at: String,
}

/// The host's answer to a [`Note`].
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct NoteAck {
    pub accepted: bool,
    pub at: String,
}

impl rig::effect::CustomEffect for Note {
    const KIND: &'static str = "corpus:note";
    type Answer = NoteAck;
}

/// The host's handler for [`Note`]: acknowledges every note with where
/// it was taken.
#[allow(dead_code)]
pub(crate) struct NoteTaker;

impl rig::serve::Serve for NoteTaker {
    type Family = rig::effect::family::Custom<Note>;

    fn descriptor(&self) -> rig::effect::HandlerDescriptor {
        rig::effect::HandlerDescriptor {
            key: rig::effect::HandlerKey::from(NOTE_KEY),
            family: rig::effect::FamilyDescriptor::Custom {
                kind: <Note as rig::effect::CustomEffect>::KIND.to_owned(),
            },
        }
    }

    async fn serve(&self, kind: rig::effect::EffectKind, sink: rig::serve::OutcomeSink) {
        let outcome = match kind {
            rig::effect::EffectKind::Custom { payload, .. } => {
                match serde_json::from_value::<Note>(payload) {
                    Ok(note) => Ok(rig::effect::Outcome::Custom(
                        serde_json::to_value(NoteAck {
                            accepted: true,
                            at: note.at,
                        })
                        .expect("an ack serializes"),
                    )),
                    Err(error) => Err(rig::error::ErrorReport::new(
                        rig::error::ErrorKind::Request,
                        format!("not a note: {error}"),
                    )),
                }
            }
            other => Err(rig::error::ErrorReport::new(
                rig::error::ErrorKind::Request,
                format!("a note, not {other:?}"),
            )),
        };
        sink.resolve(outcome).await;
    }
}

#[allow(dead_code)]
fn note_key() -> rig::effect::Key<rig::effect::family::Custom<Note>> {
    rig::effect::Key::new_unchecked(rig::effect::HandlerKey::from(NOTE_KEY))
}

/// Dispatch a note from a hook, asserting the host acknowledged it.
#[allow(dead_code)]
async fn take_note(ctx: &rig::agent::HookContext, at: &str) {
    let host = ctx.bind(&note_key()).expect("the host serves notes");
    let ack = host
        .dispatch(Note { at: at.to_owned() })
        .await
        .expect("the host acknowledges");
    assert!(ack.accepted && ack.at == at, "{ack:?}");
}

/// `on_run_start` → a note, before the first completion.
#[allow(dead_code)]
pub(crate) struct NoteAtStart;

impl rig::agent::AgentHook for NoteAtStart {
    async fn on_run_start(
        &self,
        ctx: &rig::agent::HookContext,
        _event: rig::agent::RunStart<'_>,
    ) -> rig::agent::RunStartAction {
        take_note(ctx, "start").await;
        rig::agent::RunStartAction::continue_run()
    }
}

/// `on_completion_call` → a note before every completion.
#[allow(dead_code)]
pub(crate) struct NoteAtCompletionCall;

impl rig::agent::AgentHook for NoteAtCompletionCall {
    async fn on_completion_call(
        &self,
        ctx: &rig::agent::HookContext,
        _event: rig::agent::CompletionCallEvent<'_>,
    ) -> rig::agent::CompletionCallAction {
        take_note(ctx, "completion_call").await;
        rig::agent::CompletionCallAction::Continue
    }
}

/// `on_outcome` → a note after every tool answer (a completion's answer
/// is left alone).
#[allow(dead_code)]
pub(crate) struct NoteAtOutcome;

impl rig::agent::AgentHook for NoteAtOutcome {
    async fn on_outcome(
        &self,
        ctx: &rig::agent::HookContext,
        event: rig::agent::OutcomeEvent<'_>,
    ) -> rig::agent::OutcomeAction {
        if event.kind.family() == rig::effect::EffectFamily::Tool {
            take_note(ctx, "outcome").await;
        }
        rig::agent::OutcomeAction::Proceed
    }
}

/// `on_run_settled` → a note after the run's answer: the last dispatch
/// the run makes, after the record that answered it.
#[allow(dead_code)]
pub(crate) struct NoteAtSettled;

impl rig::agent::AgentHook for NoteAtSettled {
    async fn on_run_settled(
        &self,
        ctx: &rig::agent::HookContext,
        _event: rig::agent::RunSettled<'_>,
    ) {
        take_note(ctx, "settled").await;
    }
}

/// `on_run_start` → two notes dispatched together (their order is the
/// bus's).
#[allow(dead_code)]
pub(crate) struct NoteTwice;

impl rig::agent::AgentHook for NoteTwice {
    async fn on_run_start(
        &self,
        ctx: &rig::agent::HookContext,
        _event: rig::agent::RunStart<'_>,
    ) -> rig::agent::RunStartAction {
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
        rig::agent::RunStartAction::continue_run()
    }
}

/// `on_run_start` → a bind to a key the host never registered: the hook
/// sees the refusal and lets the run go on; nothing is dispatched.
#[allow(dead_code)]
pub(crate) struct NoteUnserved;

impl rig::agent::AgentHook for NoteUnserved {
    async fn on_run_start(
        &self,
        ctx: &rig::agent::HookContext,
        _event: rig::agent::RunStart<'_>,
    ) -> rig::agent::RunStartAction {
        let refused = ctx.bind(&note_key()).expect_err("no host serves notes");
        assert_eq!(
            refused.kind,
            rig::error::ErrorKind::HandlerUnavailable,
            "{refused:?}"
        );
        rig::agent::RunStartAction::continue_run()
    }
}

/// `on_run_start` → the prompt's text embedded through the host's
/// embedding model.
#[allow(dead_code)]
pub(crate) struct EmbedPrompt;

impl rig::agent::AgentHook for EmbedPrompt {
    async fn on_run_start(
        &self,
        ctx: &rig::agent::HookContext,
        event: rig::agent::RunStart<'_>,
    ) -> rig::agent::RunStartAction {
        let key: rig::effect::Key<rig::effect::family::Embed> =
            rig::effect::Key::new_unchecked(rig::effect::HandlerKey::from(EMBED_KEY));
        let host = ctx.bind(&key).expect("the host serves embeddings");
        let text = event.prompt.rag_text().expect("a text prompt");
        let outputs = host
            .dispatch(rig::effect::EmbedInputs::Texts(vec![text]))
            .await
            .expect("the host embeds");
        match outputs {
            rig::effect::EmbedOutputs::Texts(response) => {
                assert_eq!(response.embeddings.len(), 1, "{response:?}")
            }
            rig::effect::EmbedOutputs::Images(_) => panic!("a text embedding"),
        }
        rig::agent::RunStartAction::continue_run()
    }
}

// ---------------------------------------------------------------------------
// Matrix J: memory operations.

/// The agent's memory key (`<owner>/memory`, the owner `golden`).
#[allow(dead_code)]
pub(crate) const MEMORY_KEY: &str = "golden/memory";
/// The conversation every memory cell loads and saves under.
#[allow(dead_code)]
pub(crate) const CONVERSATION: &str = "golden-conversation";

/// Clear the conversation from a hook, through the run's memory handle.
#[allow(dead_code)]
async fn clear_conversation(ctx: &rig::agent::HookContext) {
    let memory = ctx
        .memory(&rig::effect::HandlerKey::from(MEMORY_KEY))
        .expect("the run's bus serves memory");
    let outcome = memory
        .dispatch(rig::effect::MemoryOp::Clear {
            conversation: rig::id::ConversationId::from(CONVERSATION),
        })
        .await
        .expect("the memory clears");
    assert!(
        matches!(outcome, rig::effect::MemoryOutcome::Cleared),
        "{outcome:?}"
    );
}

/// `on_run_start` → `Clear`; the hook fires after the run's `Load`, so
/// the clear lands between the load and the append.
#[allow(dead_code)]
pub(crate) struct ClearAtStart;

impl rig::agent::AgentHook for ClearAtStart {
    async fn on_run_start(
        &self,
        ctx: &rig::agent::HookContext,
        _event: rig::agent::RunStart<'_>,
    ) -> rig::agent::RunStartAction {
        clear_conversation(ctx).await;
        rig::agent::RunStartAction::continue_run()
    }
}

/// `on_run_settled` → `Clear` after the run's `Append`.
#[allow(dead_code)]
pub(crate) struct ClearAtSettled;

impl rig::agent::AgentHook for ClearAtSettled {
    async fn on_run_settled(
        &self,
        ctx: &rig::agent::HookContext,
        _event: rig::agent::RunSettled<'_>,
    ) {
        clear_conversation(ctx).await;
    }
}

/// An in-memory conversation store whose `load` or `append` refuses.
#[allow(dead_code)]
pub(crate) struct FailingMemory {
    inner: rig::memory::InMemoryConversationMemory,
    fail_load: bool,
    fail_append: bool,
}

#[allow(dead_code)]
impl FailingMemory {
    pub(crate) fn load_fails() -> Self {
        Self {
            inner: rig::memory::InMemoryConversationMemory::new(),
            fail_load: true,
            fail_append: false,
        }
    }

    pub(crate) fn append_fails() -> Self {
        Self {
            inner: rig::memory::InMemoryConversationMemory::new(),
            fail_load: false,
            fail_append: true,
        }
    }
}

fn refused(op: &str) -> rig::memory::MemoryError {
    rig::memory::MemoryError::Backend(format!("the store refused the {op}").into())
}

impl rig::memory::ConversationMemory for FailingMemory {
    fn load<'a>(
        &'a self,
        conversation_id: &'a rig::id::ConversationId,
    ) -> rig::wasm_compat::WasmBoxedFuture<
        'a,
        Result<Vec<rig::message::Message>, rig::memory::MemoryError>,
    > {
        if self.fail_load {
            Box::pin(async { Err(refused("load")) })
        } else {
            self.inner.load(conversation_id)
        }
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a rig::id::ConversationId,
        messages: Vec<rig::message::Message>,
    ) -> rig::wasm_compat::WasmBoxedFuture<'a, Result<(), rig::memory::MemoryError>> {
        if self.fail_append {
            Box::pin(async { Err(refused("append")) })
        } else {
            self.inner.append(conversation_id, messages)
        }
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a rig::id::ConversationId,
    ) -> rig::wasm_compat::WasmBoxedFuture<'a, Result<(), rig::memory::MemoryError>> {
        self.inner.clear(conversation_id)
    }
}

// ---------------------------------------------------------------------------
// Matrix K: the delta wire.

#[allow(dead_code)]
pub(crate) const STOP_ON_TOOL_NAME_DELTA: &str = "stop on the tool's name delta";
#[allow(dead_code)]
pub(crate) const STOP_ON_TOOL_ARGUMENTS_DELTA: &str = "stop on the tool's arguments delta";

/// `on_tool_call_delta` → `Stop` on the delta that names the tool.
#[allow(dead_code)]
pub(crate) struct StopOnToolNameDelta;
impl rig::agent::AgentHook for StopOnToolNameDelta {
    async fn on_tool_call_delta(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::ToolCallDelta<'_>,
    ) -> rig::agent::ObservationAction {
        if event.tool_name.is_some() {
            rig::agent::ObservationAction::stop(STOP_ON_TOOL_NAME_DELTA)
        } else {
            rig::agent::ObservationAction::continue_run()
        }
    }
}

/// `on_tool_call_delta` → `Stop` on the first arguments delta.
#[allow(dead_code)]
pub(crate) struct StopOnToolArgumentsDelta;
impl rig::agent::AgentHook for StopOnToolArgumentsDelta {
    async fn on_tool_call_delta(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::ToolCallDelta<'_>,
    ) -> rig::agent::ObservationAction {
        if event.tool_name.is_none() && !event.delta.is_empty() {
            rig::agent::ObservationAction::stop(STOP_ON_TOOL_ARGUMENTS_DELTA)
        } else {
            rig::agent::ObservationAction::continue_run()
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix M: per-turn shaping through `on_completion_call` and `on_model_select`.

#[allow(dead_code)]
pub(crate) const SHAPING_CONTEXT_ID: &str = "shaping-context";
#[allow(dead_code)]
pub(crate) const SHAPING_CONTEXT: &str =
    "Definition of a glarb-glarb: an ancient farming tool from planet Jiro.";
#[allow(dead_code)]
pub(crate) const LATE_ROUTE: &str = "late";

#[allow(dead_code)]
fn shaping_document() -> rig::completion::Document {
    rig::completion::Document {
        id: SHAPING_CONTEXT_ID.to_owned(),
        text: SHAPING_CONTEXT.to_owned(),
        additional_props: Default::default(),
    }
}

/// A patch applied on one turn only, or on every turn.
macro_rules! patch_hook {
    ($(#[$doc:meta])* $name:ident, |$turn:ident| $patch:expr) => {
        $(#[$doc])*
        #[allow(dead_code)]
        pub(crate) struct $name;

        impl rig::agent::AgentHook for $name {
            async fn on_completion_call(
                &self,
                _ctx: &rig::agent::HookContext,
                event: rig::agent::CompletionCallEvent<'_>,
            ) -> rig::agent::CompletionCallAction {
                let $turn = event.turn;
                match $patch {
                    Some(patch) => rig::agent::CompletionCallAction::patch(patch),
                    None => rig::agent::CompletionCallAction::Continue,
                }
            }
        }
    };
}

patch_hook!(
    /// `tool_choice: Required` on the first turn only.
    PatchToolChoiceRequiredFirst,
    |turn| (turn == 1).then(|| rig::agent::RequestPatch::new().tool_choice(rig::message::ToolChoice::Required))
);
patch_hook!(
    /// `tool_choice: None` on the second turn only.
    PatchToolChoiceNoneSecond,
    |turn| (turn == 2).then(|| rig::agent::RequestPatch::new().tool_choice(rig::message::ToolChoice::None))
);
patch_hook!(
    /// A context document on every turn.
    PatchExtraContext,
    |_turn| Some(rig::agent::RequestPatch::new().context(shaping_document()))
);
patch_hook!(
    /// `max_tokens: 5` on the second turn only.
    PatchMaxTokensSecond,
    |turn| (turn == 2).then(|| rig::agent::RequestPatch::new().max_tokens(5))
);
patch_hook!(
    /// Extended thinking (and the temperature it needs) on the second turn only.
    PatchThinkingSecond,
    |turn| (turn == 2).then(|| {
        rig::agent::RequestPatch::new()
            .temperature(1.0)
            .additional_params(serde_json::json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } }))
    })
);
patch_hook!(
    /// The pirate preamble on the second turn only.
    PatchPreambleSecond,
    |turn| (turn == 2).then(|| rig::agent::RequestPatch::new().preamble(PIRATE_PREAMBLE))
);
patch_hook!(
    /// No tools advertised on the second turn.
    PatchActiveToolsNoneSecond,
    |turn| (turn == 2).then(|| rig::agent::RequestPatch::new().active_tools(Vec::<String>::new()))
);
patch_hook!(
    /// A prior exchange as the first turn's history.
    PatchHistoryFirst,
    |turn| (turn == 1).then(|| {
        rig::agent::RequestPatch::new().history(vec![
            rig::message::Message::user("My name is Ada."),
            rig::message::Message::assistant("Hello, Ada."),
        ])
    })
);

/// `on_model_select` → `Select("fast")` on the first turn (no model asked
/// yet), `Continue` after: the reverse of `RouteAfterFirstTurn`.
#[allow(dead_code)]
pub(crate) struct RouteOnFirstTurn;

impl rig::agent::AgentHook for RouteOnFirstTurn {
    fn on_model_select(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::ModelSelection<'_>,
    ) -> rig::agent::ModelSelectionAction {
        if event.previous_model.is_none() {
            rig::agent::ModelSelectionAction::select("fast")
        } else {
            rig::agent::ModelSelectionAction::continue_run()
        }
    }
}

/// `on_model_select` → `Select("late")` on every turn: a route the agent
/// registered after build (`register_model`), which the required row does
/// not name.
#[allow(dead_code)]
pub(crate) struct SelectLate;

impl rig::agent::AgentHook for SelectLate {
    fn on_model_select(
        &self,
        _ctx: &rig::agent::HookContext,
        _event: rig::agent::ModelSelection<'_>,
    ) -> rig::agent::ModelSelectionAction {
        rig::agent::ModelSelectionAction::select(LATE_ROUTE)
    }
}

// ---------------------------------------------------------------------------
// Matrix O: the oracle and the header.

/// `on_model_turn_finished` → `Stop` after turn `n`, named by `n`: a
/// stateful hook whose header name carries its state.
#[allow(dead_code)]
pub(crate) struct StopAfterTurnN(pub usize);

#[allow(dead_code)]
pub(crate) fn stop_after_turn_reason(n: usize) -> String {
    format!("stopped after turn {n}")
}

impl rig::agent::AgentHook for StopAfterTurnN {
    fn name(&self) -> Option<String> {
        Some(format!("StopAfterTurn({})", self.0))
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::ModelTurnFinished<'_>,
    ) -> rig::agent::ModelTurnAction {
        if event.turn == self.0 {
            rig::agent::ModelTurnAction::stop(stop_after_turn_reason(self.0))
        } else {
            rig::agent::ModelTurnAction::continue_run()
        }
    }
}

/// The host's key for its reranker.
#[allow(dead_code)]
pub(crate) const RERANK_KEY: &str = "host/rerank";
#[allow(dead_code)]
pub(crate) const RERANK_DOCUMENTS: [&str; 2] = ["the harbor label", "the orchard label"];

/// A reranker that ranks by document length, longest first: a mock behind
/// a `RerankAdapter`, since no keyed provider in the tree has a rerank
/// cassette suite.
#[allow(dead_code)]
pub(crate) struct MockRerank;

impl rig::rerank::RerankModel for MockRerank {
    fn max_documents(&self) -> usize {
        16
    }

    async fn rerank(
        &self,
        _query: &str,
        documents: Vec<String>,
    ) -> Result<rig::rerank::RerankResponse, rig::rerank::RerankError> {
        let mut results: Vec<rig::rerank::RerankResult> = documents
            .iter()
            .enumerate()
            .map(|(index, document)| rig::rerank::RerankResult {
                index,
                document: Some(document.clone()),
                relevance_score: document.len() as f64 / 100.0,
            })
            .collect();
        results.sort_by(|left, right| right.relevance_score.total_cmp(&left.relevance_score));
        let mut response = rig::rerank::RerankResponse::new(results, "mock");
        response.model = Some("mock-rerank".to_owned());
        Ok(response)
    }
}

/// `on_run_start` → the prompt reranks two documents through the host's
/// reranker.
#[allow(dead_code)]
pub(crate) struct RerankDocs;

impl rig::agent::AgentHook for RerankDocs {
    async fn on_run_start(
        &self,
        ctx: &rig::agent::HookContext,
        event: rig::agent::RunStart<'_>,
    ) -> rig::agent::RunStartAction {
        let key: rig::effect::Key<rig::effect::family::Rerank> =
            rig::effect::Key::new_unchecked(rig::effect::HandlerKey::from(RERANK_KEY));
        let host = ctx.bind(&key).expect("the host serves reranking");
        let query = event.prompt.rag_text().expect("a text prompt");
        let ranked = host
            .dispatch(rig::effect::RerankRequest {
                query,
                documents: RERANK_DOCUMENTS
                    .iter()
                    .map(|doc| (*doc).to_owned())
                    .collect(),
            })
            .await
            .expect("the host reranks");
        assert_eq!(ranked.results.len(), 2, "{ranked:?}");
        rig::agent::RunStartAction::continue_run()
    }
}

// ---------------------------------------------------------------------------
// Matrix Q: causal dispatch. The `lookup` tool dispatches from inside its
// own service through the dispatcher its sink carries, so the child record
// names the tool's record as its parent; the host's relay nests once more;
// the host's `never` handler holds a dispatch until its consumer goes. The
// rig-verify replay registers the same handlers (`corpus/mod.rs`): program,
// not record.

#[allow(dead_code)]
pub(crate) const NESTING_TOOL_KEY: &str = "golden/tool:lookup#0";
#[allow(dead_code)]
pub(crate) const RELAY_KEY: &str = "host/relay";
#[allow(dead_code)]
pub(crate) const NEVER_KEY: &str = "host/never";
#[allow(dead_code)]
pub(crate) const NESTED_PREAMBLE: &str = "Answer in one word.";

/// What the `lookup` tool dispatches, and from where.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct Nesting {
    pub child: NestedChild,
    /// The nested dispatch is made from a spawned OS thread, blocking on
    /// its first poll (serial programs only: the refusal is decided at the
    /// send).
    pub from_thread: bool,
    /// The tool detaches its sink; a spawned task answers, dispatching the
    /// child through the detached sink's dispatcher.
    pub detached: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum NestedChild {
    /// A completion on the agent's model key: the question in the args.
    Completion,
    /// A host note.
    Note,
    /// The tool's own key with `leaf: true`: refused under serial serving,
    /// served under concurrent.
    Same,
    /// The host's relay, which itself dispatches a note: a chain of three.
    Relay,
    /// The host's never-answering handler: the child is in flight when the
    /// run is dropped.
    Never,
    /// Two dispatches to the never-answering handler at once: under a
    /// serial host the second is queued when the run is dropped.
    NeverTwice,
}

/// The relay's effect.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct RelayNote {
    pub at: String,
}

impl rig::effect::CustomEffect for RelayNote {
    const KIND: &'static str = "corpus:relay";
    type Answer = NoteAck;
}

/// The never-answering effect.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct Hold;

impl rig::effect::CustomEffect for Hold {
    const KIND: &'static str = "corpus:hold";
    type Answer = NoteAck;
}

#[allow(dead_code)]
fn relay_key() -> rig::effect::Key<rig::effect::family::Custom<RelayNote>> {
    rig::effect::Key::new_unchecked(rig::effect::HandlerKey::from(RELAY_KEY))
}

#[allow(dead_code)]
fn never_key() -> rig::effect::Key<rig::effect::family::Custom<Hold>> {
    rig::effect::Key::new_unchecked(rig::effect::HandlerKey::from(NEVER_KEY))
}

/// The `lookup` tool's arguments: a question, or a leaf marker.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct LookupArgs {
    #[serde(default)]
    pub q: String,
    #[serde(default)]
    pub leaf: bool,
}

#[allow(dead_code)]
pub(crate) fn lookup_parameters() -> serde_json::Value {
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
/// dispatcher its sink carries.
#[allow(dead_code)]
pub(crate) struct Lookup {
    pub nesting: Nesting,
    pub model_key: rig::effect::HandlerKey,
}

#[allow(dead_code)]
fn tool_text(context: rig::tool::ToolContext, text: String) -> rig::effect::Outcome {
    rig::effect::Outcome::ToolResult {
        result: rig::core::tool::ToolResult::success(rig::core::tool::ToolOutput::text(text)),
        context,
    }
}

impl Lookup {
    #[allow(dead_code)]
    async fn nest(&self, dispatcher: rig::bus::Dispatcher, args: LookupArgs) -> String {
        match self.nesting.child {
            NestedChild::Completion => {
                let model: rig::bus::ModelHandle =
                    dispatcher.handle(&self.model_key).expect("the model");
                let request =
                    rig::core::completion::CompletionRequestBuilder::unbound(args.q.as_str())
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
                        rig::core::message::AssistantContent::Text(text) => {
                            Some(text.text.trim().to_owned())
                        }
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
                let handle: rig::bus::ToolHandle = dispatcher
                    .handle(&rig::effect::HandlerKey::from(NESTING_TOOL_KEY))
                    .expect("the tool's own key");
                let call = handle.call(
                    "lookup",
                    r#"{"q":"","leaf":true}"#,
                    rig::tool::ToolContext::new(),
                );
                let answer = if self.nesting.from_thread {
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

impl rig::serve::Serve for Lookup {
    type Family = rig::effect::family::Tool;

    fn descriptor(&self) -> rig::effect::HandlerDescriptor {
        rig::effect::HandlerDescriptor {
            key: rig::effect::HandlerKey::from(NESTING_TOOL_KEY),
            family: rig::effect::FamilyDescriptor::Tool {
                name: "lookup".to_owned(),
                description: "Look a question up".to_owned(),
                parameters: lookup_parameters(),
                embedding: None,
            },
        }
    }

    async fn serve(&self, kind: rig::effect::EffectKind, sink: rig::serve::OutcomeSink) {
        let rig::effect::EffectKind::ToolCall { args, context, .. } = kind else {
            sink.resolve(Err(rig::error::ErrorReport::new(
                rig::error::ErrorKind::Request,
                "a tool call",
            )))
            .await;
            return;
        };
        let args: LookupArgs = serde_json::from_str(&args).unwrap_or_default();
        if args.leaf {
            sink.resolve(Ok(tool_text(context, "leaf".to_owned())))
                .await;
            return;
        }
        if self.nesting.detached {
            let sink = sink.detach();
            let dispatcher = rig::bus::SinkDispatch::dispatcher(&sink).expect("a scoped sink");
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
                sink.resolve(Ok(tool_text(context, text))).await;
            });
            return;
        }
        let dispatcher = rig::bus::SinkDispatch::dispatcher(&sink).expect("a scoped sink");
        assert_eq!(dispatcher.parent(), Some(sink.id()));
        let text = self.nest(dispatcher, args).await;
        sink.resolve(Ok(tool_text(context, text))).await;
    }
}

/// The host's relay: takes a note through its own sink's dispatcher.
#[allow(dead_code)]
pub(crate) struct Relay;

impl rig::serve::Serve for Relay {
    type Family = rig::effect::family::Custom<RelayNote>;

    fn descriptor(&self) -> rig::effect::HandlerDescriptor {
        rig::effect::HandlerDescriptor {
            key: rig::effect::HandlerKey::from(RELAY_KEY),
            family: rig::effect::FamilyDescriptor::Custom {
                kind: <RelayNote as rig::effect::CustomEffect>::KIND.to_owned(),
            },
        }
    }

    async fn serve(&self, kind: rig::effect::EffectKind, sink: rig::serve::OutcomeSink) {
        let rig::effect::EffectKind::Custom { payload, .. } = kind else {
            sink.resolve(Err(rig::error::ErrorReport::new(
                rig::error::ErrorKind::Request,
                "a relay note",
            )))
            .await;
            return;
        };
        let note: RelayNote = serde_json::from_value(payload).expect("a relay note");
        let dispatcher = rig::bus::SinkDispatch::dispatcher(&sink).expect("a scoped sink");
        let ack = dispatcher
            .bind(&note_key())
            .expect("the host serves notes")
            .dispatch(Note {
                at: format!("relay<{}", note.at),
            })
            .await
            .expect("acknowledged");
        sink.resolve(Ok(rig::effect::Outcome::Custom(
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
#[allow(dead_code)]
pub(crate) struct Never {
    pub reached: std::sync::Arc<tokio::sync::Notify>,
}

impl rig::serve::Serve for Never {
    type Family = rig::effect::family::Custom<Hold>;

    fn descriptor(&self) -> rig::effect::HandlerDescriptor {
        rig::effect::HandlerDescriptor {
            key: rig::effect::HandlerKey::from(NEVER_KEY),
            family: rig::effect::FamilyDescriptor::Custom {
                kind: <Hold as rig::effect::CustomEffect>::KIND.to_owned(),
            },
        }
    }

    async fn serve(&self, _kind: rig::effect::EffectKind, sink: rig::serve::OutcomeSink) {
        self.reached.notify_one();
        futures::future::pending::<()>().await;
        drop(sink);
    }
}

/// The parent of every record, by position in the log.
#[allow(dead_code)]
pub(crate) fn parent_positions(log: &EffectLog) -> Vec<Option<usize>> {
    log.records
        .iter()
        .map(|record| {
            record.parent.map(|parent| {
                log.records
                    .iter()
                    .position(|r| r.id == parent)
                    .expect("a parent in the log")
            })
        })
        .collect()
}
