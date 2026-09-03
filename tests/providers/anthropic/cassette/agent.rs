//! Anthropic agent completion smoke test.

use rig::prelude::*;
use rig::providers::anthropic;

use super::super::support::with_anthropic_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    with_anthropic_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .preamble(BASIC_PREAMBLE)
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed")
            .output;

        assert_nonempty_response(&response);
    })
    .await;
}

/// The golden effect log: `completion_smoke` recorded as effects. The
/// committed file (`crates/rig-verify/fixtures/anthropic_completion_smoke.effects.json`)
/// is what `rig-verify`'s two interpreters must both reproduce, kind for
/// kind, from a replayer. Regenerate with `RIG_REGENERATE_GOLDEN=1` after a
/// deliberate change to the request the agent builds (never by hand); the
/// cassette itself is untouched either way.
#[tokio::test]
async fn completion_smoke_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .record_effects()
            .build();
        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed")
            .output;
        assert_nonempty_response(&response);
        let log = agent.take_effect_log().expect("recording");
        let rendered = serde_json::to_string_pretty(&log).expect("the log serializes");
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("crates/rig-verify/fixtures/anthropic_completion_smoke.effects.json");
        if std::env::var_os("RIG_REGENERATE_GOLDEN").is_some() {
            std::fs::create_dir_all(path.parent().expect("a parent")).expect("fixtures dir");
            std::fs::write(&path, format!("{rendered}\n")).expect("the golden file writes");
            return;
        }
        let committed = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("no golden fixture at {}; run with RIG_REGENERATE_GOLDEN=1", path.display()));
        assert_eq!(
            committed.trim_end(),
            rendered,
            "the agent's effects diverged from the golden log; if the change is deliberate, regenerate it"
        );
    })
    .await;
}
