//! The effect corpus's Anthropic recordings: producers of the golden effect
//! logs rig-verify replays (`crates/rig-verify/tests/golden_replay.rs`).
//! Each records once against the cassette transport and writes its golden
//! under `RIG_REGENERATE_GOLDEN=1`, else asserts equality with it.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::effect::EffectFamily;
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig::streaming::{Delta, StreamEvent};

use super::super::support::with_anthropic_cassette;
use crate::support::{Adder, TOOLS_PREAMBLE};

fn families(log: &rig::effect_log::EffectLog) -> Vec<EffectFamily> {
    log.records
        .iter()
        .map(|record| record.kind.family())
        .collect()
}

/// One tool call, then the final answer: `[Completion, Tool, Completion]`.
#[tokio::test]
async fn tool_call_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("effect_corpus/tool_call_turn", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .record_effects()
            .build();
        let response = agent
            .prompt("Use the add tool to add 17 and 25, then reply with just the number.")
            .max_turns(3)
            .await
            .expect("the agent answers");
        assert!(response.output.contains("42"), "{}", response.output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion
            ]
        );
        crate::goldens::golden_effects("anthropic_tool_call_turn", &log);
    })
    .await;
}

/// A streamed turn whose consumer drops the stream after the first text
/// delta: the completion record's outcome is `Cancelled`, not a handler or
/// provider failure (the corpus prompt's risk 6, now with evidence).
#[tokio::test]
async fn cancelled_stream_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("effect_corpus/cancelled_stream", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble("You are a concise assistant. Answer directly.")
            .temperature(0.0)
            .record_effects()
            .build();
        {
            let mut stream = agent
                .stream_prompt(
                    "Write a 600-word essay on the history of the Rust programming language.",
                )
                .stream()
                .await;
            while let Some(item) = stream.next().await {
                if let Ok(MultiTurnStreamItem::StreamAssistantItem(StreamEvent::BlockDelta {
                    delta: Delta::Text { .. },
                    ..
                })) = item
                {
                    break;
                }
            }
            // Dropped here, mid-stream.
        }
        // The driver resolves the cancelled dispatch on its own task.
        for _ in 0..64 {
            tokio::task::yield_now().await;
        }
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        let report = log.records[0]
            .outcome
            .as_ref()
            .expect_err("a dropped stream is recorded as a cancel");
        assert_eq!(report.kind, rig::error::ErrorKind::Cancelled, "{report:?}");
        crate::goldens::golden_effects("anthropic_cancelled_stream", &log);
    })
    .await;
}
