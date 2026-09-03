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
