//! Anthropic agent completion smoke test.

use rig::providers::anthropic;
use rig_cassette::agent::AgentReplayExt;

use super::super::support::with_anthropic_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

/// The golden effect log: `completion_smoke` recorded as effects. The
/// committed file (`crates/rig-cassette/fixtures/effects/anthropic_completion_smoke.effects.json`)
/// is what `rig-cassette`'s two interpreters must both reproduce, kind for
/// kind, from a replayer. Regenerate with `RIG_REGENERATE_GOLDEN=1` after a
/// deliberate change to the request the agent builds (never by hand); the
/// cassette itself is untouched either way.
#[tokio::test]
async fn completion_smoke_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("agent/completion_smoke", |client| async move {
        let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
        let agent =
            rig::AgentBuilder::new(client.completion(anthropic::completion::CLAUDE_SONNET_4_6))
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .record_to(recorder.clone())
                .build();
        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed")
            .output();
        assert_nonempty_response(&response);
        let log = agent.stamp(recorder.take());
        crate::goldens::golden_effects("anthropic_completion_smoke", &log);
    })
    .await;
}
