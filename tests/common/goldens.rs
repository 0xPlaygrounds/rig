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
