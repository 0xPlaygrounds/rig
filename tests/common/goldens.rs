//! Golden effect logs: the effect-bus cassette corpus.
//!
//! A producing test runs an agent program against the cassette transport
//! with `record_effects()` and either writes the log to
//! `crates/rig-verify/fixtures/<name>.effects.json` (under
//! `RIG_REGENERATE_GOLDEN=1`) or asserts the run's log equals the committed
//! one as data — so the root suite itself detects drift between a cassette
//! and its golden. rig-verify replays every golden with no provider at
//! all. Goldens are re-recorded by their producer, never edited by hand.

use rig_effect_log::EffectLog;

/// The committed golden's path.
pub(crate) fn golden_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates/rig-verify/fixtures")
        .join(format!("{name}.effects.json"))
}

/// Write `log` as the golden `name` under `RIG_REGENERATE_GOLDEN=1`, else
/// assert it equals the committed golden byte for byte (the header is part
/// of the oracle: a program that changed refuses before it diverges).
pub(crate) fn golden_effects(name: &str, log: &EffectLog) {
    // A golden is generated from the *replayed* cassette, never from a live
    // recording: the cassette is written with placeholders for provider ids
    // (`msg_REDACTED_1`, `toolu_REDACTED_1`, …) and the golden must hold the
    // same, or the first replay diverges on an id the record never held.
    assert!(
        !std::env::var("RIG_PROVIDER_TEST_MODE")
            .is_ok_and(|mode| mode.eq_ignore_ascii_case("record")),
        "golden `{name}`: record the cassette first, then regenerate the golden in replay mode"
    );
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
