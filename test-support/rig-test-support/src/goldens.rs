//! Golden effect logs: the effect-bus cassette corpus.
//!
//! A producing test runs an agent program against the cassette transport
//! with an explicit `EffectLogRecorder` and either writes the log to
//! `crates/rig-cassette/fixtures/effects/<name>.effects.json` (under
//! `RIG_REGENERATE_GOLDEN=1`) or asserts the run's log equals the committed
//! one as data — so the root suite itself detects drift between a cassette
//! and its golden. rig-cassette replays every golden with no provider at
//! all. Goldens are re-recorded by their producer, never edited by hand.

use rig_core::effect::EffectFamily;

use rig_core::message::Message;

use rig_cassette::effect_log::EffectLog;

#[path = "goldens/world.rs"]
mod world;
pub(crate) use world::attach_world_recorder;
pub use world::{capture_world_program, capture_world_programs, world_golden_test};

/// The families of a log's records, in order: the shape a producer asserts.
#[allow(dead_code)] // not every target records
pub fn families(log: &EffectLog) -> Vec<EffectFamily> {
    log.records
        .iter()
        .map(|record| record.kind.family())
        .collect()
}

/// The output schema the request-shape matrix constrains an answer to,
/// as one literal both the producer and the rig-cassette replay build the
/// program from (`crates/rig-cassette/tests/corpus_request_shape.rs`).
#[allow(dead_code)]
pub const EVENT_SCHEMA: &str = r#"{"type":"object","properties":{"title":{"type":"string"},"category":{"type":"string"},"summary":{"type":"string"}},"required":["title","category","summary"]}"#;

#[allow(dead_code)]
/// Parse the fixed event schema used by structured-output corpus cells.
pub fn event_schema() -> schemars::Schema {
    serde_json::from_str(EVENT_SCHEMA).expect("the schema literal is a schema")
}

/// The prior history the request-shape matrix's history cell runs with;
/// the replay builds the same two turns.
#[allow(dead_code)]
pub fn prior_history() -> Vec<Message> {
    vec![
        Message::user("My name is Ada."),
        Message::assistant("Nice to meet you, Ada."),
    ]
}

/// The committed golden's path.
pub fn golden_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/rig-cassette/fixtures/effects")
        .join(format!("{name}.effects.json"))
}

/// Write `log` as the golden `name` under `RIG_REGENERATE_GOLDEN=1`, else
/// assert it equals the committed golden byte for byte (the header is part
/// of the oracle: a program that changed refuses before it diverges).
///
/// In record mode this is a no-op: a golden is generated from the
/// *replayed* cassette, never from a live recording, because the golden
/// must hold exactly the bytes replay serves (the finalized cassette after
/// scrubbing and volatile-field normalization), or the first replay diverges. A panic here would
/// also discard the cassette the run just recorded (the wrapper writes a
/// cassette only when the test body returns), so the loop is: record on
/// the producer's filter, then regenerate the golden in replay mode.
pub fn golden_effects(name: &str, log: &EffectLog) {
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

/// Compare a native log with its world fixture as JSON data.
/// Observed delivery boundaries are excluded because HTTP scheduling varies.
/// Requires [`capture_world_programs`] and pre-dispatch program captures.
/// In replay mode, `RIG_REGENERATE_GOLDEN` writes the fixture instead.
/// Record mode does nothing and rejects simultaneous regeneration.
/// Names must be a single file stem, not a path.
pub fn world_golden_effects(name: &str, log: &EffectLog) {
    assert!(
        !name.is_empty()
            && name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_'),
        "world golden names must contain only letters, digits, and underscores"
    );
    if std::env::var("RIG_PROVIDER_TEST_MODE").is_ok_and(|mode| mode.eq_ignore_ascii_case("record"))
    {
        assert!(
            std::env::var_os("RIG_REGENERATE_GOLDEN").is_none(),
            "world golden `{name}`: regenerate only in replay mode"
        );
        return;
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/rig-cassette/fixtures/effects/world")
        .join(format!("{name}.effects.json"));
    if std::env::var_os("RIG_REGENERATE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().expect("world corpus directory"))
            .expect("create world corpus directory");
        let rendered = serde_json::to_string_pretty(log).expect("the world log serializes");
        std::fs::write(&path, format!("{rendered}\n")).expect("write world golden");
        world::programs(
            &path.with_file_name(format!("{name}.programs.json")),
            log,
            true,
        );
        return;
    }
    let committed = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "world golden {}: {error}; regenerate in replay mode",
            path.display()
        )
    });
    world::programs(
        &path.with_file_name(format!("{name}.programs.json")),
        log,
        false,
    );
    let expected: serde_json::Value = serde_json::from_str(&committed).expect("world golden JSON");
    assert_eq!(
        world::without_delivery_boundaries(expected),
        world::without_delivery_boundaries(
            serde_json::to_value(log).expect("the world log serializes")
        ),
        "the world's effects diverged from world golden `{name}`"
    );
}

/// The corpus's recovery hook: an unknown tool is retried once with
/// feedback. A hook is program, not record — the effect-log header names
/// it by type, so every producer that records a recovery and the
/// rig-cassette replay use this one type.
#[allow(dead_code)] // used by the recovery producer, not every target
pub struct RetryUnknownTool;

impl rig_agent::agent::AgentHook for RetryUnknownTool {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        context: &rig_agent::agent::InvalidToolCallContext,
    ) -> Option<rig_agent::agent::InvalidToolCallAction> {
        Some(rig_agent::agent::InvalidToolCallAction::Retry {
            feedback: format!("there is no tool named {}; use add", context.tool_name),
        })
    }
}

// ---------------------------------------------------------------------------
// The hook matrix's hooks (Matrix B, `crates/rig-cassette/tests/providers/anthropic/cassette/
// corpus_hooks.rs`, `crates/rig-cassette/tests/corpus_hooks.rs`). Hooks are
// program: the header names each by type, and the rig-cassette replay
// defines a type of the same name making the same decision. Every hook is
// stateless, so its decision is a function of the event alone (the
// header cannot tell two hooks of one type with different state apart).

#[allow(dead_code)]
/// Fixed pirate instruction used by preamble-patching cells.
pub const PIRATE_PREAMBLE: &str = "You are a pirate. Answer in one short sentence.";
#[allow(dead_code)]
/// Expected denial text when the run disables the add tool.
pub const DENY_REASON: &str = "add is disabled for this run";
#[allow(dead_code)]
/// Fixed tool-result replacement used by outcome hooks.
pub const REPLACED_RESULT: &str = "99";
#[allow(dead_code)]
/// Fixed final-answer replacement used by steering hooks.
pub const REPLACED_ANSWER: &str = "REPLACED";
#[allow(dead_code)]
/// Feedback requesting the final DONE marker.
pub const DONE_FEEDBACK: &str = "End your answer with the word DONE.";
#[allow(dead_code)]
/// Fixed JSON arguments for the add-tool lookup dispatch.
pub const LOOKUP_ARGS: &str = r#"{"x":1,"y":2}"#;
#[allow(dead_code)]
/// Handler key of the corpus's registered add tool.
pub const LOOKUP_KEY: &str = "golden/tool:add#0";

/// Opts into observing every dispatch family; decides nothing.
#[allow(dead_code)]
pub struct ObserveEverything;

impl rig_agent::agent::AgentHook for ObserveEverything {
    fn observes(&self, _kind: rig_agent::agent::StepEventKind) -> bool {
        true
    }
}

/// `on_dispatch` → `Patch`: `add` runs with `{"x":40,"y":2}` whatever the
/// model asked (the record holds the patched call; history keeps the
/// model's).
#[allow(dead_code)]
pub struct PatchAddArgs;

impl rig_agent::agent::AgentHook for PatchAddArgs {
    async fn on_dispatch(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::DispatchEvent<'_>,
    ) -> rig_agent::agent::DispatchAction {
        if event.tool_name() == Some("add") {
            rig_agent::agent::DispatchAction::rewrite_tool_args(
                event.kind,
                serde_json::json!({"x": 40, "y": 2}),
            )
        } else {
            rig_agent::agent::DispatchAction::proceed()
        }
    }
}

/// `on_dispatch` → `Deny` (a skip): `add` never reaches the bus; the model
/// sees the reason as the tool's result.
#[allow(dead_code)]
pub struct DenyAdd;

impl rig_agent::agent::AgentHook for DenyAdd {
    async fn on_dispatch(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::DispatchEvent<'_>,
    ) -> rig_agent::agent::DispatchAction {
        if event.tool_name() == Some("add") {
            rig_agent::agent::DispatchAction::skip(DENY_REASON)
        } else {
            rig_agent::agent::DispatchAction::proceed()
        }
    }
}

/// `on_outcome` → `Replace`: the model sees `99` for `add`, the record
/// holds what the tool answered.
#[allow(dead_code)]
pub struct ReplaceAddResult;

impl rig_agent::agent::AgentHook for ReplaceAddResult {
    async fn on_outcome(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::OutcomeEvent<'_>,
    ) -> rig_agent::agent::OutcomeAction {
        if event.tool_name() == Some("add") && event.tool_result().is_some() {
            rig_agent::agent::OutcomeAction::rewrite_tool_result(&event, REPLACED_RESULT)
        } else {
            rig_agent::agent::OutcomeAction::proceed()
        }
    }
}

/// `on_outcome` → `Replace` on a completion: a text answer is replaced by
/// `REPLACED`; the record holds the model's.
#[allow(dead_code)]
pub struct ReplaceAnswer;

impl rig_agent::agent::AgentHook for ReplaceAnswer {
    async fn on_outcome(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::OutcomeEvent<'_>,
    ) -> rig_agent::agent::OutcomeAction {
        let Some(response) = event.completion() else {
            return rig_agent::agent::OutcomeAction::proceed();
        };
        if response
            .choice
            .iter()
            .any(|content| matches!(content, rig_core::message::AssistantContent::ToolCall(_)))
        {
            return rig_agent::agent::OutcomeAction::proceed();
        }
        let mut replacement = response.clone();
        replacement.choice = vec![rig_core::message::AssistantContent::text(REPLACED_ANSWER)];
        rig_agent::agent::OutcomeAction::replace(Ok(rig_core::effect::Outcome::Completion(
            replacement,
        )))
    }
}

/// `on_completion_call` → a request patch overriding the preamble: the
/// request holds the pirate preamble, the spec holds the base.
#[allow(dead_code)]
pub struct PreambleOverride;

impl rig_agent::agent::AgentHook for PreambleOverride {
    async fn on_completion_call(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::CompletionCallEvent<'_>,
    ) -> rig_agent::agent::CompletionCallAction {
        rig_agent::agent::CompletionCallAction::patch(
            rig_agent::agent::RequestPatch::new().preamble(PIRATE_PREAMBLE),
        )
    }
}

/// `on_model_turn_finished` → `Retry` with feedback until the answer holds
/// `DONE`: a second completion is a record, the decision is program.
#[allow(dead_code)]
pub struct DemandDone;

impl rig_agent::agent::AgentHook for DemandDone {
    async fn on_model_turn_finished(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::ModelTurnFinished<'_>,
    ) -> rig_agent::agent::ModelTurnAction {
        let text: String = event
            .content
            .iter()
            .filter_map(|content| match content {
                rig_core::message::AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect();
        if text.contains("DONE") {
            rig_agent::agent::ModelTurnAction::continue_run()
        } else {
            rig_agent::agent::ModelTurnAction::retry_with_feedback(DONE_FEEDBACK)
        }
    }
}

/// `on_run_start` dispatches `add(1, 2)` through the run's bus: a hook's
/// own effect is a record under the tool's key, before the first
/// completion.
#[allow(dead_code)]
pub struct LookupBeforeRun;

impl rig_agent::agent::AgentHook for LookupBeforeRun {
    async fn on_run_start(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        let tool = ctx
            .tool(&rig_core::effect::HandlerKey::from(LOOKUP_KEY))
            .expect("the run's bus serves add");
        let answer = tool
            .dispatch(rig_core::effect::ToolCallRequest {
                name: "add".to_owned(),
                args: LOOKUP_ARGS.to_owned(),
            })
            .await
            .expect("add answers");
        assert_eq!(answer.output().render(), "3");
        rig_agent::agent::RunStartAction::continue_run()
    }
}

/// `on_model_select` → `Select("fast")` on every turn after the first:
/// the route answers once the default model has been asked once.
#[allow(dead_code)]
pub struct RouteAfterFirstTurn;

impl rig_agent::agent::AgentHook for RouteAfterFirstTurn {
    fn on_model_select(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::ModelSelection<'_>,
    ) -> rig_agent::agent::ModelSelectionAction {
        if event.previous_model.is_some() {
            rig_agent::agent::ModelSelectionAction::select("fast")
        } else {
            rig_agent::agent::ModelSelectionAction::continue_run()
        }
    }
}

// ---------------------------------------------------------------------------
// The outcome matrix's tool (Matrix D).

#[allow(dead_code)]
/// Expected failure message from the deliberately broken adder.
pub const BROKEN_ADD: &str = "the adder is broken";

#[derive(serde::Deserialize)]
#[allow(dead_code)]
/// Integer operands accepted by the corpus arithmetic tools.
pub struct FailingAddArgs {
    /// First arithmetic operand.
    pub x: i64,
    /// Second arithmetic operand.
    pub y: i64,
}

/// An `add` that fails every call: the tool record's outcome is a failed
/// result, which the model sees and answers around.
#[allow(dead_code)]
pub struct FailingAdd;

impl rig_core::tool::Tool for FailingAdd {
    const NAME: &'static str = "add";
    type Args = FailingAddArgs;
    type Output = i64;
    type Error = rig_core::tool::ToolExecutionError;

    fn description(&self) -> String {
        "adds two integers".into()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]})
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        _args: FailingAddArgs,
    ) -> Result<i64, Self::Error> {
        Err(rig_core::tool::ToolExecutionError::other(BROKEN_ADD))
    }
}

// ---------------------------------------------------------------------------
// The outcome matrix's long-argument tool: a call whose arguments stream
// for long enough that a consumer's drop lands mid-call.

#[derive(serde::Deserialize)]
#[allow(dead_code)]
/// Arguments accepted by the note-writing tool.
pub struct NoteArgs {
    /// Title of the note to write.
    pub title: String,
    /// Body of the note to write.
    pub body: String,
}

/// Writes a note; its `body` is what the model streams at length.
#[allow(dead_code)]
pub struct WriteNote;

impl rig_core::tool::Tool for WriteNote {
    const NAME: &'static str = "write_note";
    type Args = NoteArgs;
    type Output = String;
    type Error = rig_core::tool::ToolExecutionError;

    fn description(&self) -> String {
        "writes a note with a title and a body".into()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}}, "required": ["title", "body"]})
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
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
/// Fixed documents used by retrieval corpus cells.
pub const FACTS: [&str; 3] = [
    "A flurbo is a green alien that lives on cold planets.",
    "A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
    "A linglingdong is a term used by inhabitants of the far side of the moon to describe humans.",
];
#[allow(dead_code)]
/// Question answered from the fixed retrieval documents.
pub const FACT_PROMPT: &str = "What is a glarb-glarb? Answer in one sentence.";
#[allow(dead_code)]
/// System instruction requiring use of retrieved arithmetic tools.
pub const RETRIEVED_TOOLS_PREAMBLE: &str =
    "You are a calculator. You must use the provided tools for every arithmetic operation.";
#[allow(dead_code)]
/// Prompt exercising the retrieved subtraction tool.
pub const SUBTRACT_PROMPT: &str =
    "Subtract 8 from 50 with the subtract tool, then reply with just the number.";
#[allow(dead_code)]
/// Prompt requiring an add call followed by a dependent subtraction call.
pub const ADD_THEN_SUBTRACT_PROMPT: &str = "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the subtract tool. Report the final number.";

/// The facts, embedded by `model`, as an in-memory index (ids `doc0`..).
#[allow(dead_code)]
pub async fn facts_index<M: rig_core::embeddings::EmbeddingModel + Clone>(
    model: M,
    facts: &[&str],
) -> rig_core::vector_store::in_memory_store::InMemoryVectorIndex<String, M> {
    let store = if facts.is_empty() {
        rig_core::vector_store::in_memory_store::InMemoryVectorStore::<String>::default()
    } else {
        let embeddings = rig_core::embeddings::EmbeddingsBuilder::new(model.clone())
            .documents(facts.iter().map(|fact| (*fact).to_owned()))
            .expect("documents should be added")
            .build()
            .await
            .expect("fact embeddings should succeed");
        rig_core::vector_store::in_memory_store::InMemoryVectorStore::from_documents(embeddings)
    };
    store.index(model)
}

/// The toolset's embeddable schemas, embedded by `model`, as an index keyed
/// by tool name.
#[allow(dead_code)]
pub async fn tool_index<M: rig_core::embeddings::EmbeddingModel + Clone>(
    model: M,
    toolset: &rig_agent::tool::ToolSet,
) -> rig_core::vector_store::in_memory_store::InMemoryVectorIndex<rig_core::embeddings::ToolSchema, M>
{
    let embeddings = rig_core::embeddings::EmbeddingsBuilder::new(model.clone())
        .documents(toolset.schemas().expect("tool schemas should build"))
        .expect("documents should be added")
        .build()
        .await
        .expect("tool schema embeddings should succeed");
    rig_core::vector_store::in_memory_store::InMemoryVectorStore::from_documents_with_id_f(
        embeddings,
        |tool| tool.name.clone(),
    )
    .index(model)
}

#[derive(Debug, thiserror::Error)]
#[error("init error")]
#[allow(dead_code)]
/// Initialization error type for the retrievable arithmetic tools.
pub struct NoInit;

macro_rules! retrievable_operation {
    ($name:ident, $tool_name:literal, $description:literal, $embedding_doc:literal, $op:expr) => {
        #[derive(Clone, Default, serde::Deserialize, serde::Serialize)]
        #[allow(dead_code)]
        /// Retrievable arithmetic tool with a fixed name, schema, and embedding document.
        pub struct $name;

        impl rig_core::tool::Tool for $name {
            const NAME: &'static str = $tool_name;
            type Error = rig_core::tool::ToolExecutionError;
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
                _context: &mut rig_core::tool::ToolContext,
                args: Self::Args,
            ) -> Result<Self::Output, Self::Error> {
                let op: fn(i64, i64) -> i64 = $op;
                Ok(op(args.x, args.y))
            }
        }

        impl rig_core::tool::ToolEmbedding for $name {
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
pub fn retrievable_toolset() -> rig_agent::tool::ToolSet {
    let mut toolset = rig_agent::tool::ToolSet::default();
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
/// Expected stop reason at run start.
pub const STOP_AT_START: &str = "stopped at run start";
#[allow(dead_code)]
/// Expected stop reason at model selection.
pub const STOP_AT_MODEL_SELECT: &str = "stopped at model selection";
#[allow(dead_code)]
/// Expected stop reason before a completion call.
pub const STOP_AT_COMPLETION_CALL: &str = "stopped before the completion call";
#[allow(dead_code)]
/// Expected cancellation reason before add dispatch reaches the bus.
pub const CANCEL_ADD_DISPATCH: &str = "add is cancelled before the bus";
#[allow(dead_code)]
/// Expected cancellation reason after add dispatch returns from the bus.
pub const CANCEL_ADD_OUTCOME: &str = "add is cancelled after the bus";
#[allow(dead_code)]
/// Expected cancellation reason for the final answer.
pub const CANCEL_ANSWER: &str = "the answer is cancelled";
#[allow(dead_code)]
/// Expected stop reason after a model turn.
pub const STOP_AFTER_TURN: &str = "stopped after the model turn";
#[allow(dead_code)]
/// Expected stop reason at the answer turn.
pub const STOP_AT_ANSWER: &str = "stopped at the answer turn";
#[allow(dead_code)]
/// Expected stop reason on the first text delta.
pub const STOP_ON_TEXT_DELTA: &str = "stopped on the first text delta";
#[allow(dead_code)]
/// Expected stop reason on the first tool-call delta.
pub const STOP_ON_TOOL_CALL_DELTA: &str = "stopped on the first tool-call delta";
#[allow(dead_code)]
/// Expected stop reason on the first reasoning delta.
pub const STOP_ON_REASONING_DELTA: &str = "stopped on the first reasoning delta";

#[allow(dead_code)]
/// Hook that stops the run at its start event.
pub struct StopAtStart;
impl rig_agent::agent::AgentHook for StopAtStart {
    async fn on_run_start(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        rig_agent::agent::RunStartAction::stop(STOP_AT_START)
    }
}

#[allow(dead_code)]
/// Hook that stops the run during model selection.
pub struct StopAtModelSelect;
impl rig_agent::agent::AgentHook for StopAtModelSelect {
    fn on_model_select(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::ModelSelection<'_>,
    ) -> rig_agent::agent::ModelSelectionAction {
        rig_agent::agent::ModelSelectionAction::stop(STOP_AT_MODEL_SELECT)
    }
}

#[allow(dead_code)]
/// Hook that stops before the completion call.
pub struct StopAtCompletionCall;
impl rig_agent::agent::AgentHook for StopAtCompletionCall {
    async fn on_completion_call(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::CompletionCallEvent<'_>,
    ) -> rig_agent::agent::CompletionCallAction {
        rig_agent::agent::CompletionCallAction::stop(STOP_AT_COMPLETION_CALL)
    }
}

/// `on_dispatch` → `Deny(Cancelled)` for `add`: the run stops before the
/// tool reaches the bus.
#[allow(dead_code)]
pub struct CancelAddDispatch;
impl rig_agent::agent::AgentHook for CancelAddDispatch {
    async fn on_dispatch(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::DispatchEvent<'_>,
    ) -> rig_agent::agent::DispatchAction {
        if event.tool_name() == Some("add") {
            rig_agent::agent::DispatchAction::stop(CANCEL_ADD_DISPATCH)
        } else {
            rig_agent::agent::DispatchAction::proceed()
        }
    }
}

/// `on_outcome` → `Replace(Err(Cancelled))` for `add`'s result: the tool
/// ran and is recorded; the run stops after.
#[allow(dead_code)]
pub struct CancelAddOutcome;
impl rig_agent::agent::AgentHook for CancelAddOutcome {
    async fn on_outcome(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::OutcomeEvent<'_>,
    ) -> rig_agent::agent::OutcomeAction {
        if event.tool_name() == Some("add") && event.tool_result().is_some() {
            rig_agent::agent::OutcomeAction::stop(CANCEL_ADD_OUTCOME)
        } else {
            rig_agent::agent::OutcomeAction::proceed()
        }
    }
}

/// `on_outcome` → `Replace(Err(Cancelled))` on a text answer.
#[allow(dead_code)]
pub struct CancelAnswer;
impl rig_agent::agent::AgentHook for CancelAnswer {
    async fn on_outcome(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::OutcomeEvent<'_>,
    ) -> rig_agent::agent::OutcomeAction {
        match event.completion() {
            Some(response)
                if !response
                    .choice
                    .iter()
                    .any(|c| matches!(c, rig_core::message::AssistantContent::ToolCall(_))) =>
            {
                rig_agent::agent::OutcomeAction::stop(CANCEL_ANSWER)
            }
            _ => rig_agent::agent::OutcomeAction::proceed(),
        }
    }
}

#[allow(dead_code)]
/// Hook that stops after the model turn.
pub struct StopAfterTurn;
impl rig_agent::agent::AgentHook for StopAfterTurn {
    async fn on_model_turn_finished(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::ModelTurnFinished<'_>,
    ) -> rig_agent::agent::ModelTurnAction {
        rig_agent::agent::ModelTurnAction::stop(STOP_AFTER_TURN)
    }
}

/// Stops at the turn that carries no tool call — the answer turn of a
/// tool program, so the tool turn's records precede the stop.
#[allow(dead_code)]
pub struct StopAtAnswer;
impl rig_agent::agent::AgentHook for StopAtAnswer {
    async fn on_model_turn_finished(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::ModelTurnFinished<'_>,
    ) -> rig_agent::agent::ModelTurnAction {
        if event
            .content
            .iter()
            .any(|c| matches!(c, rig_core::message::AssistantContent::ToolCall(_)))
        {
            rig_agent::agent::ModelTurnAction::continue_run()
        } else {
            rig_agent::agent::ModelTurnAction::stop(STOP_AT_ANSWER)
        }
    }
}

#[allow(dead_code)]
/// Hook that stops on the first text delta.
pub struct StopOnTextDelta;
impl rig_agent::agent::AgentHook for StopOnTextDelta {
    async fn on_text_delta(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::TextDelta<'_>,
    ) -> rig_agent::agent::ObservationAction {
        rig_agent::agent::ObservationAction::stop(STOP_ON_TEXT_DELTA)
    }
}

#[allow(dead_code)]
/// Hook that stops on the first tool-call delta.
pub struct StopOnToolCallDelta;
impl rig_agent::agent::AgentHook for StopOnToolCallDelta {
    async fn on_tool_call_delta(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::ToolCallDelta<'_>,
    ) -> rig_agent::agent::ObservationAction {
        rig_agent::agent::ObservationAction::stop(STOP_ON_TOOL_CALL_DELTA)
    }
}

#[allow(dead_code)]
/// Hook that stops on the first reasoning delta.
pub struct StopOnReasoningDelta;
impl rig_agent::agent::AgentHook for StopOnReasoningDelta {
    async fn on_reasoning_delta(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::ReasoningDelta<'_>,
    ) -> rig_agent::agent::ObservationAction {
        rig_agent::agent::ObservationAction::stop(STOP_ON_REASONING_DELTA)
    }
}

/// Observe-only: what `on_run_settled` saw, for the producer to assert.
/// Not a record, and its state is not identity (the header names the
/// type); the replay's hook of the same name observes nothing.
#[derive(Clone, Default)]
#[allow(dead_code)]
pub struct RecordSettled(pub std::sync::Arc<std::sync::Mutex<Option<String>>>);
impl rig_agent::agent::AgentHook for RecordSettled {
    async fn on_run_settled(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::RunSettled<'_>,
    ) {
        let seen = match event.outcome {
            rig_agent::agent::SettledOutcome::Response(response) => {
                format!("response:{}", response.output)
            }
            rig_agent::agent::SettledOutcome::Error(reason) => format!("error:{reason}"),
        };
        *self.0.lock().expect("settled") = Some(seen);
    }
}

// ---------------------------------------------------------------------------
// The invalid-call matrix's hooks (Matrix G).

#[allow(dead_code)]
/// Expected result when an unknown tool call is skipped.
pub const SKIP_REASON: &str = "no such tool; skipped";

/// `on_invalid_tool_call` → `Repair { tool_name: "add" }`: the unknown
/// call is re-targeted to `add` with its arguments.
#[allow(dead_code)]
pub struct RepairToAdd;
impl rig_agent::agent::AgentHook for RepairToAdd {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _context: &rig_agent::agent::InvalidToolCallContext,
    ) -> Option<rig_agent::agent::InvalidToolCallAction> {
        Some(rig_agent::agent::InvalidToolCallAction::Repair {
            tool_name: "add".to_owned(),
        })
    }
}

/// `on_invalid_tool_call` → `Skip { reason }`: the model sees the reason
/// as the call's result and goes on.
#[allow(dead_code)]
pub struct SkipUnknown;
impl rig_agent::agent::AgentHook for SkipUnknown {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _context: &rig_agent::agent::InvalidToolCallContext,
    ) -> Option<rig_agent::agent::InvalidToolCallAction> {
        Some(rig_agent::agent::InvalidToolCallAction::Skip {
            reason: SKIP_REASON.to_owned(),
        })
    }
}

// ---------------------------------------------------------------------------
// Matrix I: a host's own effect, dispatched by hooks over the host's bus.

/// The host's key for its custom handler.
#[allow(dead_code)]
pub const NOTE_KEY: &str = "host/note";
/// The host's key for its embedding model.
#[allow(dead_code)]
pub const EMBED_KEY: &str = "host/embed";

/// A host-defined effect: a note of where in the run it was taken.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct Note {
    /// Lifecycle location recorded by the note dispatch.
    pub at: String,
}

/// The host's answer to a [`Note`].
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct NoteAck {
    /// Whether the handler accepted the note.
    pub accepted: bool,
    /// Lifecycle location carried through the acknowledgement.
    pub at: String,
}

impl rig_core::effect::CustomEffect for Note {
    const KIND: &'static str = "corpus:note";
    type Answer = NoteAck;
}

/// The host's handler for [`Note`]: acknowledges every note with where
/// it was taken.
#[allow(dead_code)]
pub struct NoteTaker;

impl rig_core::serve::Serve for NoteTaker {
    type Family = rig_core::effect::family::Custom<Note>;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: rig_core::effect::HandlerKey::from(NOTE_KEY),
            family: rig_core::effect::FamilyDescriptor::Custom {
                kind: <Note as rig_core::effect::CustomEffect>::KIND.to_owned(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(
        &self,
        kind: rig_core::effect::EffectKind,
        _dispatch: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        let outcome = match kind {
            rig_core::effect::EffectKind::Custom { payload, .. } => {
                match serde_json::from_value::<Note>(payload) {
                    Ok(note) => Ok(rig_core::effect::Outcome::Custom {
                        payload: serde_json::to_value(NoteAck {
                            accepted: true,
                            at: note.at,
                        })
                        .expect("an ack serializes"),
                    }),
                    Err(error) => Err(rig_core::error::ErrorReport::new(
                        rig_core::error::ErrorKind::Request,
                        format!("not a note: {error}"),
                    )),
                }
            }
            other => Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::Request,
                format!("a note, not {other:?}"),
            )),
        };
        rig_core::serve::Reply::Outcome(outcome)
    }
}

#[allow(dead_code)]
fn note_key() -> rig_core::effect::Key<rig_core::effect::family::Custom<Note>> {
    rig_core::effect::Key::new_unchecked(rig_core::effect::HandlerKey::from(NOTE_KEY))
}

/// Dispatch a note from a hook, asserting the host acknowledged it.
#[allow(dead_code)]
async fn take_note(ctx: &rig_agent::agent::HookContext, at: &str) {
    let host = ctx.bind(&note_key()).expect("the host serves notes");
    let ack = host
        .dispatch(Note { at: at.to_owned() })
        .await
        .expect("the host acknowledges");
    assert!(ack.accepted && ack.at == at, "{ack:?}");
}

/// `on_run_start` → a note, before the first completion.
#[allow(dead_code)]
pub struct NoteAtStart;

impl rig_agent::agent::AgentHook for NoteAtStart {
    async fn on_run_start(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        take_note(ctx, "start").await;
        rig_agent::agent::RunStartAction::continue_run()
    }
}

/// `on_completion_call` → a note before every completion.
#[allow(dead_code)]
pub struct NoteAtCompletionCall;

impl rig_agent::agent::AgentHook for NoteAtCompletionCall {
    async fn on_completion_call(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::CompletionCallEvent<'_>,
    ) -> rig_agent::agent::CompletionCallAction {
        take_note(ctx, "completion_call").await;
        rig_agent::agent::CompletionCallAction::Continue
    }
}

/// `on_outcome` → a note after every tool answer (a completion's answer
/// is left alone).
#[allow(dead_code)]
pub struct NoteAtOutcome;

impl rig_agent::agent::AgentHook for NoteAtOutcome {
    async fn on_outcome(
        &self,
        ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::OutcomeEvent<'_>,
    ) -> rig_agent::agent::OutcomeAction {
        if event.kind.family() == rig_core::effect::EffectFamily::Tool {
            take_note(ctx, "outcome").await;
        }
        rig_agent::agent::OutcomeAction::Proceed
    }
}

/// `on_run_settled` → a note after the run's answer: the last dispatch
/// the run makes, after the record that answered it.
#[allow(dead_code)]
pub struct NoteAtSettled;

impl rig_agent::agent::AgentHook for NoteAtSettled {
    async fn on_run_settled(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunSettled<'_>,
    ) {
        take_note(ctx, "settled").await;
    }
}

/// `on_run_start` → two notes dispatched together (their order is the
/// bus's).
#[allow(dead_code)]
pub struct NoteTwice;

impl rig_agent::agent::AgentHook for NoteTwice {
    async fn on_run_start(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
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
        rig_agent::agent::RunStartAction::continue_run()
    }
}

/// `on_run_start` → a bind to a key the host never registered: the hook
/// sees the refusal and lets the run go on; nothing is dispatched.
#[allow(dead_code)]
pub struct NoteUnserved;

impl rig_agent::agent::AgentHook for NoteUnserved {
    async fn on_run_start(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        let refused = ctx.bind(&note_key()).expect_err("no host serves notes");
        assert_eq!(
            refused.kind,
            rig_core::error::ErrorKind::HandlerUnavailable,
            "{refused:?}"
        );
        rig_agent::agent::RunStartAction::continue_run()
    }
}

/// `on_run_start` → the prompt's text embedded through the host's
/// embedding model.
#[allow(dead_code)]
pub struct EmbedPrompt;

impl rig_agent::agent::AgentHook for EmbedPrompt {
    async fn on_run_start(
        &self,
        ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        let key: rig_core::effect::Key<rig_core::effect::family::Embed> =
            rig_core::effect::Key::new_unchecked(rig_core::effect::HandlerKey::from(EMBED_KEY));
        let host = ctx.bind(&key).expect("the host serves embeddings");
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
        rig_agent::agent::RunStartAction::continue_run()
    }
}

// ---------------------------------------------------------------------------
// Matrix J: memory operations.

/// The agent's memory key (`<owner>/memory`, the owner `golden`).
#[allow(dead_code)]
pub const MEMORY_KEY: &str = "golden/memory";
/// The conversation every memory cell loads and saves under.
#[allow(dead_code)]
pub const CONVERSATION: &str = "golden-conversation";

/// Clear the conversation from a hook, through the run's memory handle.
#[allow(dead_code)]
async fn clear_conversation(ctx: &rig_agent::agent::HookContext) {
    let memory = ctx
        .memory(&rig_core::effect::HandlerKey::from(MEMORY_KEY))
        .expect("the run's bus serves memory");
    let outcome = memory
        .dispatch(rig_core::effect::MemoryOp::Clear {
            conversation: rig_core::id::ConversationId::from(CONVERSATION),
        })
        .await
        .expect("the memory clears");
    assert!(
        matches!(outcome, rig_core::effect::MemoryOutcome::Cleared),
        "{outcome:?}"
    );
}

/// `on_run_start` → `Clear`; the hook fires after the run's `Load`, so
/// the clear lands between the load and the append.
#[allow(dead_code)]
pub struct ClearAtStart;

impl rig_agent::agent::AgentHook for ClearAtStart {
    async fn on_run_start(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        clear_conversation(ctx).await;
        rig_agent::agent::RunStartAction::continue_run()
    }
}

/// `on_run_settled` → `Clear` after the run's `Append`.
#[allow(dead_code)]
pub struct ClearAtSettled;

impl rig_agent::agent::AgentHook for ClearAtSettled {
    async fn on_run_settled(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunSettled<'_>,
    ) {
        clear_conversation(ctx).await;
    }
}

/// An in-memory conversation store whose `load` or `append` refuses.
#[allow(dead_code)]
pub struct FailingMemory {
    inner: rig_core::memory::InMemoryConversationMemory,
    fail_load: bool,
    fail_append: bool,
}

#[allow(dead_code)]
impl FailingMemory {
    /// Construct memory that fails loads while allowing appends.
    pub fn load_fails() -> Self {
        Self {
            inner: rig_core::memory::InMemoryConversationMemory::new(),
            fail_load: true,
            fail_append: false,
        }
    }

    /// Construct memory that loads successfully but fails appends.
    pub fn append_fails() -> Self {
        Self {
            inner: rig_core::memory::InMemoryConversationMemory::new(),
            fail_load: false,
            fail_append: true,
        }
    }
}

fn refused(op: &str) -> rig_core::memory::MemoryError {
    rig_core::memory::MemoryError::Backend(format!("the store refused the {op}").into())
}

impl rig_core::memory::ConversationMemory for FailingMemory {
    fn load<'a>(
        &'a self,
        conversation_id: &'a rig_core::id::ConversationId,
    ) -> rig_core::wasm_compat::WasmBoxedFuture<
        'a,
        Result<Vec<rig_core::message::Message>, rig_core::memory::MemoryError>,
    > {
        if self.fail_load {
            Box::pin(async { Err(refused("load")) })
        } else {
            self.inner.load(conversation_id)
        }
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a rig_core::id::ConversationId,
        messages: Vec<rig_core::message::Message>,
    ) -> rig_core::wasm_compat::WasmBoxedFuture<'a, Result<(), rig_core::memory::MemoryError>> {
        if self.fail_append {
            Box::pin(async { Err(refused("append")) })
        } else {
            self.inner.append(conversation_id, messages)
        }
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a rig_core::id::ConversationId,
    ) -> rig_core::wasm_compat::WasmBoxedFuture<'a, Result<(), rig_core::memory::MemoryError>> {
        self.inner.clear(conversation_id)
    }
}

// ---------------------------------------------------------------------------
// Matrix K: the delta wire.

#[allow(dead_code)]
/// Expected stop reason for a tool-name delta.
pub const STOP_ON_TOOL_NAME_DELTA: &str = "stop on the tool's name delta";
#[allow(dead_code)]
/// Expected stop reason for a tool-arguments delta.
pub const STOP_ON_TOOL_ARGUMENTS_DELTA: &str = "stop on the tool's arguments delta";

/// `on_tool_call_delta` → `Stop` on the delta that names the tool.
#[allow(dead_code)]
pub struct StopOnToolNameDelta;
impl rig_agent::agent::AgentHook for StopOnToolNameDelta {
    async fn on_tool_call_delta(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::ToolCallDelta<'_>,
    ) -> rig_agent::agent::ObservationAction {
        if event.tool_name.is_some() {
            rig_agent::agent::ObservationAction::stop(STOP_ON_TOOL_NAME_DELTA)
        } else {
            rig_agent::agent::ObservationAction::continue_run()
        }
    }
}

/// `on_tool_call_delta` → `Stop` on the first arguments delta.
#[allow(dead_code)]
pub struct StopOnToolArgumentsDelta;
impl rig_agent::agent::AgentHook for StopOnToolArgumentsDelta {
    async fn on_tool_call_delta(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::ToolCallDelta<'_>,
    ) -> rig_agent::agent::ObservationAction {
        if event.tool_name.is_none() && !event.delta.is_empty() {
            rig_agent::agent::ObservationAction::stop(STOP_ON_TOOL_ARGUMENTS_DELTA)
        } else {
            rig_agent::agent::ObservationAction::continue_run()
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix M: per-turn shaping through `on_completion_call` and `on_model_select`.

#[allow(dead_code)]
/// Document identifier appended by the request-shaping hook.
pub const SHAPING_CONTEXT_ID: &str = "shaping-context";
#[allow(dead_code)]
/// Document text appended by the request-shaping hook.
pub const SHAPING_CONTEXT: &str =
    "Definition of a glarb-glarb: an ancient farming tool from planet Jiro.";
#[allow(dead_code)]
/// Model route selected by the late-routing hook.
pub const LATE_ROUTE: &str = "late";

#[allow(dead_code)]
fn shaping_document() -> rig_agent::completion::Document {
    rig_agent::completion::Document {
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
        pub struct $name;

        impl rig_agent::agent::AgentHook for $name {
            async fn on_completion_call(
                &self,
                _ctx: &rig_agent::agent::HookContext,
                event: rig_agent::agent::CompletionCallEvent<'_>,
            ) -> rig_agent::agent::CompletionCallAction {
                let $turn = event.turn;
                match $patch {
                    Some(patch) => rig_agent::agent::CompletionCallAction::patch(patch),
                    None => rig_agent::agent::CompletionCallAction::Continue,
                }
            }
        }
    };
}

patch_hook!(
    /// `tool_choice: Required` on the first turn only.
    PatchToolChoiceRequiredFirst,
    |turn| (turn == 1).then(|| rig_agent::agent::RequestPatch::new().tool_choice(rig_core::message::ToolChoice::Required))
);
patch_hook!(
    /// `tool_choice: None` on the second turn only.
    PatchToolChoiceNoneSecond,
    |turn| (turn == 2).then(|| rig_agent::agent::RequestPatch::new().tool_choice(rig_core::message::ToolChoice::None))
);
patch_hook!(
    /// A context document on every turn.
    PatchExtraContext,
    |_turn| Some(rig_agent::agent::RequestPatch::new().context(shaping_document()))
);
patch_hook!(
    /// `max_tokens: 5` on the second turn only.
    PatchMaxTokensSecond,
    |turn| (turn == 2).then(|| rig_agent::agent::RequestPatch::new().max_tokens(5))
);
patch_hook!(
    /// Extended thinking (and the temperature it needs) on the second turn only.
    PatchThinkingSecond,
    |turn| (turn == 2).then(|| {
        rig_agent::agent::RequestPatch::new()
            .temperature(1.0)
            .additional_params(serde_json::json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } }))
    })
);
patch_hook!(
    /// The pirate preamble on the second turn only.
    PatchPreambleSecond,
    |turn| (turn == 2).then(|| rig_agent::agent::RequestPatch::new().preamble(PIRATE_PREAMBLE))
);
patch_hook!(
    /// No tools advertised on the second turn.
    PatchActiveToolsNoneSecond,
    |turn| (turn == 2).then(|| rig_agent::agent::RequestPatch::new().active_tools(Vec::<String>::new()))
);
patch_hook!(
    /// A prior exchange as the first turn's history.
    PatchHistoryFirst,
    |turn| (turn == 1).then(|| {
        rig_agent::agent::RequestPatch::new().history(vec![
            rig_core::message::Message::user("My name is Ada."),
            rig_core::message::Message::assistant("Hello, Ada."),
        ])
    })
);

/// `on_model_select` → `Select("fast")` on the first turn (no model asked
/// yet), `Continue` after: the reverse of `RouteAfterFirstTurn`.
#[allow(dead_code)]
pub struct RouteOnFirstTurn;

impl rig_agent::agent::AgentHook for RouteOnFirstTurn {
    fn on_model_select(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::ModelSelection<'_>,
    ) -> rig_agent::agent::ModelSelectionAction {
        if event.previous_model.is_none() {
            rig_agent::agent::ModelSelectionAction::select("fast")
        } else {
            rig_agent::agent::ModelSelectionAction::continue_run()
        }
    }
}

/// `on_model_select` → `Select("late")` on every turn: a route the agent
/// registered after build (`register_model`), which the required row does
/// not name.
#[allow(dead_code)]
pub struct SelectLate;

impl rig_agent::agent::AgentHook for SelectLate {
    fn on_model_select(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::ModelSelection<'_>,
    ) -> rig_agent::agent::ModelSelectionAction {
        rig_agent::agent::ModelSelectionAction::select(LATE_ROUTE)
    }
}

// ---------------------------------------------------------------------------
// Matrix O: the oracle and the header.

/// `on_model_turn_finished` → `Stop` after turn `n`, named by `n`: a
/// stateful hook whose header name carries its state.
#[allow(dead_code)]
pub struct StopAfterTurnN(pub usize);

#[allow(dead_code)]
/// Build the fixed stop reason for the specified model turn.
pub fn stop_after_turn_reason(n: usize) -> String {
    format!("stopped after turn {n}")
}

impl rig_agent::agent::AgentHook for StopAfterTurnN {
    fn name(&self) -> Option<String> {
        Some(format!("StopAfterTurn({})", self.0))
    }

    async fn on_model_turn_finished(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::ModelTurnFinished<'_>,
    ) -> rig_agent::agent::ModelTurnAction {
        if event.turn == self.0 {
            rig_agent::agent::ModelTurnAction::stop(stop_after_turn_reason(self.0))
        } else {
            rig_agent::agent::ModelTurnAction::continue_run()
        }
    }
}

/// The host's key for its reranker.
#[allow(dead_code)]
pub const RERANK_KEY: &str = "host/rerank";
#[allow(dead_code)]
/// Fixed documents used by reranking corpus cells.
pub const RERANK_DOCUMENTS: [&str; 2] = ["the harbor label", "the orchard label"];

/// A reranker that ranks by document length, longest first: a mock behind
/// a `RerankAdapter`, since no keyed provider in the tree has a rerank
/// cassette suite.
#[allow(dead_code)]
pub struct MockRerank;

impl rig_core::rerank::RerankModel for MockRerank {
    fn max_documents(&self) -> usize {
        16
    }

    async fn rerank(
        &self,
        _query: &str,
        documents: Vec<String>,
    ) -> Result<rig_core::rerank::RerankResponse, rig_core::error::ProviderError> {
        let mut results: Vec<rig_core::rerank::RerankResult> = documents
            .iter()
            .enumerate()
            .map(|(index, document)| rig_core::rerank::RerankResult {
                index,
                document: Some(document.clone()),
                relevance_score: document.len() as f64 / 100.0,
            })
            .collect();
        results.sort_by(|left, right| right.relevance_score.total_cmp(&left.relevance_score));
        let mut response = rig_core::rerank::RerankResponse::new(results, "mock");
        response.model = Some("mock-rerank".to_owned());
        Ok(response)
    }
}

/// `on_run_start` → the prompt reranks two documents through the host's
/// reranker.
#[allow(dead_code)]
pub struct RerankDocs;

impl rig_agent::agent::AgentHook for RerankDocs {
    async fn on_run_start(
        &self,
        ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        let key: rig_core::effect::Key<rig_core::effect::family::Rerank> =
            rig_core::effect::Key::new_unchecked(rig_core::effect::HandlerKey::from(RERANK_KEY));
        let host = ctx.bind(&key).expect("the host serves reranking");
        let query = event.prompt.rag_text().expect("a text prompt");
        let ranked = host
            .dispatch(rig_core::effect::RerankRequest {
                query,
                documents: RERANK_DOCUMENTS
                    .iter()
                    .map(|doc| (*doc).to_owned())
                    .collect(),
            })
            .await
            .expect("the host reranks");
        assert_eq!(ranked.results.len(), 2, "{ranked:?}");
        rig_agent::agent::RunStartAction::continue_run()
    }
}

// ---------------------------------------------------------------------------
// Matrix Q: causal dispatch. The `lookup` tool dispatches from inside its
// own service through the dispatcher its sink carries, so the child record
// names the tool's record as its parent; the host's relay nests once more;
// the host's `never` handler holds a dispatch until its consumer goes. The
// rig-cassette replay registers the same handlers (`corpus/mod.rs`): program,
// not record.

#[allow(dead_code)]
/// Handler key of the tool that performs nested dispatches.
pub const NESTING_TOOL_KEY: &str = "golden/tool:lookup#0";
#[allow(dead_code)]
/// Handler key of the host's nested-note relay.
pub const RELAY_KEY: &str = "host/relay";
#[allow(dead_code)]
/// Handler key of the host service that never answers.
pub const NEVER_KEY: &str = "host/never";
#[allow(dead_code)]
/// System instruction used for nested completion requests.
pub const NESTED_PREAMBLE: &str = "Answer in one word.";

/// What the `lookup` tool dispatches, and from where.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub struct Nesting {
    /// Kind of child dispatch performed by the lookup tool.
    pub child: NestedChild,
    /// The nested dispatch is made from a spawned OS thread, blocking on
    /// its first poll (serial programs only: the refusal is decided at the
    /// send).
    pub from_thread: bool,
    /// The tool detaches its sink; a spawned task answers, dispatching the
    /// child through the detached sink's dispatcher.
    pub detached: bool,
    /// The nested completion carries no temperature (the gpt-5 family
    /// takes only its default).
    pub no_temperature: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
/// Nested-dispatch behavior exercised by a corpus cell.
pub enum NestedChild {
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
pub struct RelayNote {
    /// Location marker propagated through the relay and its child note.
    pub at: String,
}

impl rig_core::effect::CustomEffect for RelayNote {
    const KIND: &'static str = "corpus:relay";
    type Answer = NoteAck;
}

/// The never-answering effect.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct Hold;

impl rig_core::effect::CustomEffect for Hold {
    const KIND: &'static str = "corpus:hold";
    type Answer = NoteAck;
}

#[allow(dead_code)]
fn relay_key() -> rig_core::effect::Key<rig_core::effect::family::Custom<RelayNote>> {
    rig_core::effect::Key::new_unchecked(rig_core::effect::HandlerKey::from(RELAY_KEY))
}

#[allow(dead_code)]
fn never_key() -> rig_core::effect::Key<rig_core::effect::family::Custom<Hold>> {
    rig_core::effect::Key::new_unchecked(rig_core::effect::HandlerKey::from(NEVER_KEY))
}

/// The `lookup` tool's arguments: a question, or a leaf marker.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct LookupArgs {
    #[serde(default)]
    /// Question passed to the nested lookup or completion.
    pub q: String,
    #[serde(default)]
    /// Whether to answer directly without another nested dispatch.
    pub leaf: bool,
}

#[allow(dead_code)]
/// Return the fixed JSON parameter schema of the lookup tool.
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
/// dispatcher its sink carries.
#[allow(dead_code)]
pub struct Lookup {
    /// Child-dispatch and execution-context settings for the lookup.
    pub nesting: Nesting,
    /// Registered model handler used for nested completions.
    pub model_key: rig_core::effect::HandlerKey,
}

#[allow(dead_code)]
fn tool_text(text: String) -> rig_core::effect::Outcome {
    rig_core::effect::Outcome::ToolResult {
        result: rig_core::tool::ToolResult::success(rig_core::tool::ToolOutput::text(text)),
    }
}

impl Lookup {
    #[allow(dead_code)]
    async fn nest(&self, dispatcher: rig_agent::bus::Dispatcher, args: LookupArgs) -> String {
        match self.nesting.child {
            NestedChild::Completion => {
                let model: rig_agent::bus::ModelHandle =
                    dispatcher.handle(&self.model_key).expect("the model");
                let mut request =
                    rig_core::completion::CompletionRequestBuilder::unbound(args.q.as_str())
                        .preamble(NESTED_PREAMBLE.to_owned());
                if !self.nesting.no_temperature {
                    request = request.temperature(0.0);
                }
                let request = request.build();
                let response = model
                    .complete(request)
                    .await
                    .expect("the nested completion");
                response
                    .choice
                    .iter()
                    .filter_map(|content| match content {
                        rig_core::message::AssistantContent::Text(text) => {
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
                let handle: rig_agent::bus::ToolHandle = dispatcher
                    .handle(&rig_core::effect::HandlerKey::from(NESTING_TOOL_KEY))
                    .expect("the tool's own key");
                let call = handle.call(
                    "lookup",
                    r#"{"q":"","leaf":true}"#,
                    rig_core::tool::ToolContext::new(),
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

impl rig_core::serve::Serve for Lookup {
    type Family = rig_core::effect::family::Tool;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: rig_core::effect::HandlerKey::from(NESTING_TOOL_KEY),
            family: rig_core::effect::FamilyDescriptor::Tool {
                name: "lookup".to_owned(),
                description: "Look a question up".to_owned(),
                parameters: lookup_parameters(),
                embedding: None,
            },
            layers: Vec::new(),
        }
    }

    async fn serve(
        &self,
        kind: rig_core::effect::EffectKind,
        dispatch: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        let rig_core::effect::EffectKind::ToolCall { args, .. } = kind else {
            return rig_core::serve::Reply::Outcome(Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::Request,
                "a tool call",
            )));
        };
        let args: LookupArgs = serde_json::from_str(&args).unwrap_or_default();
        if args.leaf {
            return rig_core::serve::Reply::Outcome(Ok(tool_text("leaf".to_owned())));
        }
        if self.nesting.detached {
            let (resolver, answer) = rig_core::serve::deferred();
            let dispatcher =
                rig_agent::bus::DispatchScope::dispatcher(&dispatch).expect("a scoped sink");
            assert_eq!(dispatcher.parent(), Some(dispatch.id()));
            let lookup = Lookup {
                nesting: Nesting {
                    detached: false,
                    ..self.nesting
                },
                model_key: self.model_key.clone(),
            };
            tokio::spawn(async move {
                let text = lookup.nest(dispatcher, args).await;
                let _ = resolver.resolve(Ok(tool_text(text)));
            });
            return rig_core::serve::Reply::Outcome(answer.await);
        }
        let dispatcher =
            rig_agent::bus::DispatchScope::dispatcher(&dispatch).expect("a scoped sink");
        assert_eq!(dispatcher.parent(), Some(dispatch.id()));
        let text = self.nest(dispatcher, args).await;
        rig_core::serve::Reply::Outcome(Ok(tool_text(text)))
    }
}

/// The host's relay: takes a note through its own sink's dispatcher.
#[allow(dead_code)]
pub struct Relay;

impl rig_core::serve::Serve for Relay {
    type Family = rig_core::effect::family::Custom<RelayNote>;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: rig_core::effect::HandlerKey::from(RELAY_KEY),
            family: rig_core::effect::FamilyDescriptor::Custom {
                kind: <RelayNote as rig_core::effect::CustomEffect>::KIND.to_owned(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(
        &self,
        kind: rig_core::effect::EffectKind,
        dispatch: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        let rig_core::effect::EffectKind::Custom { payload, .. } = kind else {
            return rig_core::serve::Reply::Outcome(Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::Request,
                "a relay note",
            )));
        };
        let note: RelayNote = serde_json::from_value(payload).expect("a relay note");
        let dispatcher =
            rig_agent::bus::DispatchScope::dispatcher(&dispatch).expect("a scoped dispatch");
        let ack = dispatcher
            .bind(&note_key())
            .expect("the host serves notes")
            .dispatch(Note {
                at: format!("relay<{}", note.at),
            })
            .await
            .expect("acknowledged");
        rig_core::serve::Reply::Outcome(Ok(rig_core::effect::Outcome::Custom {
            payload: serde_json::to_value(NoteAck {
                accepted: ack.accepted,
                at: ack.at,
            })
            .expect("an ack serializes"),
        }))
    }
}

/// The host's handler that never answers: it signals that it was reached
/// and holds the dispatch until the consumer goes.
#[allow(dead_code)]
pub struct Never {
    /// Notification fired when the never-answering service is entered.
    pub reached: std::sync::Arc<tokio::sync::Notify>,
}

impl rig_core::serve::Serve for Never {
    type Family = rig_core::effect::family::Custom<Hold>;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: rig_core::effect::HandlerKey::from(NEVER_KEY),
            family: rig_core::effect::FamilyDescriptor::Custom {
                kind: <Hold as rig_core::effect::CustomEffect>::KIND.to_owned(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(
        &self,
        _kind: rig_core::effect::EffectKind,
        _dispatch: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        self.reached.notify_one();
        std::future::pending().await
    }
}

/// The parent of every record, by position in the log.
#[allow(dead_code)]
pub fn parent_positions(log: &EffectLog) -> Vec<Option<usize>> {
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

// ---------------------------------------------------------------------------
// Matrix P: the layers, hand-written `Intercept`s; Matrix T's `Denied`
// cells. Program, not record: `crates/rig-cassette/tests/corpus/mod.rs`
// holds the same types verbatim.

#[allow(dead_code)]
/// First fixed argument replacement used by dispatch layers.
pub const PATCHED_ARGS: &str = r#"{"x":40,"y":2}"#;
#[allow(dead_code)]
/// Second fixed argument replacement used to prove layer chaining.
pub const PATCHED_AGAIN_ARGS: &str = r#"{"x":30,"y":12}"#;
#[allow(dead_code)]
/// Expected denial message from a host layer.
pub const HOST_DENY_REASON: &str = "denied by the host";
#[allow(dead_code)]
/// Expected denial message from the external approval world.
pub const WORLD_DENY_REASON: &str = "blocked by the world";
#[allow(dead_code)]
/// Expected cancellation message from a stream-intercepting layer.
pub const CANCEL_STREAM_REASON: &str = "the answer is cancelled by a layer";

/// The history the memory layer answers a `Load` with: two turns naming Ada.
#[allow(dead_code)]
pub fn replaced_history() -> Vec<Message> {
    vec![
        Message::user("My name is Ada."),
        Message::assistant("Hello, Ada."),
    ]
}

#[allow(dead_code)]
fn is_add(kind: &rig_core::effect::EffectKind) -> bool {
    matches!(kind, rig_core::effect::EffectKind::ToolCall { name, .. } if name == "add")
}

#[allow(dead_code)]
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

macro_rules! keep_after {
    () => {
        async fn after(
            &self,
            _id: rig_core::effect::EffectId,
            _kind: &rig_core::effect::EffectKind,
            _outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
        ) -> rig_core::serve::Verdict {
            rig_core::serve::Verdict::Keep
        }
    };
}

macro_rules! proceed_before {
    () => {
        async fn before(
            &self,
            _id: rig_core::effect::EffectId,
            _kind: &rig_core::effect::EffectKind,
        ) -> rig_core::serve::Decision {
            rig_core::serve::Decision::Proceed
        }
    };
}

/// The hook `DenyAdd`, as a layer.
#[allow(dead_code)]
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
    keep_after!();
}

/// The hook `PatchAddArgs`, as a layer.
#[allow(dead_code)]
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
    keep_after!();
}

/// The host's own patch of `add`'s arguments, beneath the agent's.
#[allow(dead_code)]
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
    keep_after!();
}

/// The hook `ReplaceAddResult`, as a layer.
#[allow(dead_code)]
pub struct ReplaceAddResultLayer;

impl rig_core::serve::Intercept for ReplaceAddResultLayer {
    fn name(&self) -> String {
        "ReplaceAddResultLayer".to_owned()
    }
    proceed_before!();
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
                        .with_output(rig_core::tool::ToolOutput::text(REPLACED_RESULT)),
                }))
            }
            _ => rig_core::serve::Verdict::Keep,
        }
    }
}

/// The world a suspending layer asks.
#[allow(dead_code)]
pub type Asks = tokio::sync::mpsc::UnboundedSender<(
    rig_core::effect::EffectId,
    futures::channel::oneshot::Sender<rig_core::serve::Decision>,
)>;

/// An approval gate: `before` sends the dispatch to the world and waits.
#[allow(dead_code)]
pub struct ApprovalLayer {
    /// Channel carrying approval requests and their decision senders.
    pub asks: Asks,
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
                    rig_core::error::ErrorKind::Internal,
                    "layer `ApprovalLayer`: the world closed the answer channel without deciding",
                ))
            }
        }
    }
    keep_after!();
}

/// What the world answers a suspended dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Answer {
    /// Allow the suspended dispatch to proceed.
    Approve,
    /// Reject the suspended dispatch with the fixed world denial.
    Deny,
    /// Never: the world signals it was asked and holds the channel.
    Never,
}

/// Spawn the world: answers as `answer` says; signals `reached` when it
/// holds an answer forever.
#[allow(dead_code)]
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

/// A patch of another family: never a dispatch.
#[allow(dead_code)]
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
    keep_after!();
}

/// `after` on a completion → the answer is cancelled.
#[allow(dead_code)]
pub struct CancelStreamLayer;

impl rig_core::serve::Intercept for CancelStreamLayer {
    fn name(&self) -> String {
        "CancelStreamLayer".to_owned()
    }
    proceed_before!();
    async fn after(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
        _outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
    ) -> rig_core::serve::Verdict {
        rig_core::serve::Verdict::Replace(Err(rig_core::error::ErrorReport::new(
            rig_core::error::ErrorKind::Cancelled,
            CANCEL_STREAM_REASON,
        )))
    }
}

/// `after` on a memory `Load` → the replacement history in the store's place.
#[allow(dead_code)]
pub struct ReplaceLoadLayer;

impl rig_core::serve::Intercept for ReplaceLoadLayer {
    fn name(&self) -> String {
        "ReplaceLoadLayer".to_owned()
    }
    proceed_before!();
    async fn after(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &rig_core::effect::EffectKind,
        outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>,
    ) -> rig_core::serve::Verdict {
        match outcome {
            Ok(rig_core::effect::Outcome::Memory(rig_core::effect::MemoryOutcome::Loaded {
                ..
            })) => rig_core::serve::Verdict::Replace(Ok(rig_core::effect::Outcome::Memory(
                rig_core::effect::MemoryOutcome::Loaded {
                    messages: replaced_history(),
                },
            ))),
            _ => rig_core::serve::Verdict::Keep,
        }
    }
}

/// The host denies everything on the key.
#[allow(dead_code)]
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
    keep_after!();
}

/// `on_run_start` → a host note the host's layer denies; the hook sees
/// `Denied` and the run goes on.
#[allow(dead_code)]
pub struct NoteDeniedAtStart;

impl rig_agent::agent::AgentHook for NoteDeniedAtStart {
    async fn on_run_start(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        let host = ctx.bind(&note_key()).expect("the host serves notes");
        let report = host
            .dispatch(Note {
                at: "start".to_owned(),
            })
            .await
            .expect_err("the host's layer denies the note");
        assert_eq!(
            report.kind,
            rig_core::error::ErrorKind::Denied,
            "{report:?}"
        );
        assert_eq!(report.message, HOST_DENY_REASON);
        rig_agent::agent::RunStartAction::continue_run()
    }
}

/// `on_run_start` asserts the run starts with the history the memory
/// layer put in the `Load`'s place.
#[allow(dead_code)]
pub struct HistoryIsReplaced;

impl rig_agent::agent::AgentHook for HistoryIsReplaced {
    async fn on_run_start(
        &self,
        _ctx: &rig_agent::agent::HookContext,
        event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        assert_eq!(event.history, replaced_history());
        rig_agent::agent::RunStartAction::continue_run()
    }
}

/// The `add` tool of the anthropic cells (`test-support/rig-test-support/src/support.rs`'s
/// `Adder`, verbatim in name, description and schema, so a layered cell's
/// handler table and requests are the hook cells' bytes), available to
/// every target.
#[derive(serde::Deserialize, serde::Serialize)]
#[allow(dead_code)]
pub struct AddArgs {
    /// First number to add.
    pub x: i32,
    /// Second number to add.
    pub y: i32,
}

#[allow(dead_code)]
/// Adder matching the provider corpus tool's name, description, and schema.
pub struct Adder;

impl rig_core::tool::Tool for Adder {
    const NAME: &'static str = "add";
    type Error = rig_core::tool::ToolExecutionError;
    type Args = AddArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::from_str(
            r#"{"type":"object","properties":{"x":{"type":"number","description":"The first number to add"},"y":{"type":"number","description":"The second number to add"}},"required":["x","y"]}"#,
        )
        .expect("adder schema should deserialize")
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

/// The agent's `add` tool under `layers` (outermost first), registered
/// through a tool server named `golden` so the key is `golden/tool:add#0`.
#[allow(dead_code)]
pub fn add_tool_under(
    layers: impl FnOnce(rig_core::serve::ErasedHandler) -> rig_core::serve::ErasedHandler,
) -> rig_agent::tool::server::ToolServerHandle {
    let adder =
        rig_core::serve::ErasedHandler::new(rig_core::serve::adapters::ToolAdapter::new(Adder));
    rig_agent::tool::server::ToolServer::new()
        .owner("golden")
        .registered_tool(
            rig_agent::tool::RegisteredTool::from_handler(layers(adder))
                .expect("a tool-family handler"),
        )
        .run()
}

// ---------------------------------------------------------------------------
// Matrix T's L3 and L4 cells: a host effect that does not serialize; host
// handlers a program registers and never dispatches to.

/// A host effect whose `Serialize` fails: it never has a wire form.
#[derive(Debug)]
#[allow(dead_code)]
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

#[allow(dead_code)]
/// Handler key used by the deliberately unserializable custom effect.
pub const UNSERIALIZABLE_KEY: &str = "host/unserializable";

#[allow(dead_code)]
fn unserializable_key() -> rig_core::effect::Key<rig_core::effect::family::Custom<Unserializable>> {
    rig_core::effect::Key::new_unchecked(rig_core::effect::HandlerKey::from(UNSERIALIZABLE_KEY))
}

/// The host's handler for the kind: counts what reaches it (nothing
/// should) and would acknowledge.
#[allow(dead_code)]
pub struct NeverAsked {
    /// Number of dispatches that reached this handler; expected to remain zero.
    pub reached: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl rig_core::serve::Serve for NeverAsked {
    type Family = rig_core::effect::family::Custom<Unserializable>;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: rig_core::effect::HandlerKey::from(UNSERIALIZABLE_KEY),
            family: rig_core::effect::FamilyDescriptor::Custom {
                kind: <Unserializable as rig_core::effect::CustomEffect>::KIND.to_owned(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(
        &self,
        _kind: rig_core::effect::EffectKind,
        _dispatch: rig_core::serve::Dispatch,
    ) -> rig_core::serve::Reply {
        self.reached
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        rig_core::serve::Reply::Outcome(Ok(rig_core::effect::Outcome::Custom {
            payload: serde_json::to_value(NoteAck {
                accepted: true,
                at: "never".to_owned(),
            })
            .expect("an ack serializes"),
        }))
    }
}

/// `on_run_start` dispatches the unserializable effect: the hook sees
/// `Request` with the serde message and the run goes on.
#[allow(dead_code)]
pub struct NoteUnserializableAtStart;

impl rig_agent::agent::AgentHook for NoteUnserializableAtStart {
    async fn on_run_start(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        let host = ctx
            .bind(&unserializable_key())
            .expect("the host serves the kind");
        let report = host
            .dispatch(Unserializable)
            .await
            .expect_err("no wire form, no dispatch");
        assert_eq!(
            report.kind,
            rig_core::error::ErrorKind::Request,
            "{report:?}"
        );
        assert!(
            report.message.contains("did not serialize")
                && report.message.contains("refuses to serialize"),
            "{}",
            report.message
        );
        rig_agent::agent::RunStartAction::continue_run()
    }
}

/// `on_run_start` → `n` host notes, one after another; named by `n`.
#[allow(dead_code)]
pub struct NotesAtStart(pub usize);

impl rig_agent::agent::AgentHook for NotesAtStart {
    fn name(&self) -> Option<String> {
        Some(format!("NotesAtStart({})", self.0))
    }

    async fn on_run_start(
        &self,
        ctx: &rig_agent::agent::HookContext,
        _event: rig_agent::agent::RunStart<'_>,
    ) -> rig_agent::agent::RunStartAction {
        for n in 0..self.0 {
            take_note(ctx, &format!("start-{n}")).await;
        }
        rig_agent::agent::RunStartAction::continue_run()
    }
}
