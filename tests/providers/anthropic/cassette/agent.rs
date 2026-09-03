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
        crate::goldens::golden_effects("anthropic_completion_smoke", &log);
    })
    .await;
}

/// Golden `anthropic_memory_conversation`: the same completion over a
/// conversation memory. The load (empty), the completion and the append
/// are all records; the request bytes are the smoke cassette's, so no new
/// recording is needed.
#[tokio::test]
async fn memory_conversation_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(anthropic::completion::CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .memory(rig::memory::InMemoryConversationMemory::new())
            .conversation("golden-conversation")
            .record_effects()
            .build();
        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed")
            .output;
        assert_nonempty_response(&response);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            log.records
                .iter()
                .map(|record| record.kind.family())
                .collect::<Vec<_>>(),
            [
                rig::effect::EffectFamily::Memory,
                rig::effect::EffectFamily::Completion,
                rig::effect::EffectFamily::Memory,
            ],
            "load, completion, append"
        );
        crate::goldens::golden_effects("anthropic_memory_conversation", &log);
    })
    .await;
}
