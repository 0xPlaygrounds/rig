//! Matrix O of the effect corpus, the live cells (`CLAUDE_SONNET_4_6`,
//! temperature 0): two tool calls served concurrently with a host note
//! inside each dispatch (the cross-key order the golden pins), and a
//! stateful stop whose header name carries its state. Producers of the
//! goldens `crates/rig-verify/tests/corpus_oracle.rs` replays by both
//! interpreters. Every cell is a new recording under
//! `tests/cassettes/anthropic/corpus_oracle/`.

use rig::agent::AgentBuilder;
use rig::bus::Bus;
use rig::completion::PromptError;
use rig::effect::{EffectFamily, HandlerKey};
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::with_anthropic_corpus_oracle_cassette;
use crate::goldens::{
    NOTE_KEY, NoteAtOutcome, NoteTaker, StopAfterTurnN, families, stop_after_turn_reason,
};
use crate::support::{
    Adder, AlphaSignal, BetaSignal, TOOLS_PREAMBLE, TWO_TOOL_STREAM_PREAMBLE,
    TWO_TOOL_STREAM_PROMPT,
};

const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";

/// Two tool calls in one turn under `tool_concurrency: 2`, a host note
/// inside each dispatch: the recorder orders the six records as they were
/// dispatched, and the replay must agree.
#[tokio::test]
async fn concurrent_notes_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_oracle_cassette("corpus_oracle/concurrent_notes", |client| async move {
        let (dispatcher, registrar, mut driver) = Bus::channel();
        let model_key = HandlerKey::from("golden/model:default");
        driver
            .register_erased(
                model_key.clone(),
                rig::serve::ErasedHandler::new(rig::serve::adapters::CompletionAdapter::new(
                    "default",
                    client.completion_model(CLAUDE_SONNET_4_6),
                )),
            )
            .expect("a fresh key");
        driver
            .register_erased(
                HandlerKey::from(NOTE_KEY),
                rig::serve::ErasedHandler::new(NoteTaker),
            )
            .expect("a fresh key");
        let recorder = rig::effect_log::EffectLogRecorder::new();
        driver.record_to(recorder.clone());
        let driver = tokio::spawn(driver);
        let agent =
            AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
                .name("golden")
                .preamble(TWO_TOOL_STREAM_PREAMBLE)
                .temperature(0.0)
                .tool(AlphaSignal)
                .tool(BetaSignal)
                .add_hook(NoteAtOutcome)
                .build();
        let output = agent
            .prompt(TWO_TOOL_STREAM_PROMPT)
            .max_turns(8)
            .tool_concurrency(2)
            .await
            .expect("the agent answers")
            .output;
        assert!(!output.is_empty());
        let log = agent.stamp(recorder.take());
        drop((agent, dispatcher, registrar));
        driver.await.expect("the host's driver");
        let fams = families(&log);
        assert_eq!(fams.len(), 6, "{fams:?}");
        assert_eq!(fams[0], EffectFamily::Completion);
        assert_eq!(fams[5], EffectFamily::Completion);
        assert_eq!(
            fams[1..5]
                .iter()
                .filter(|family| **family == EffectFamily::Tool)
                .count(),
            2
        );
        eprintln!("ORDER {fams:?}");
        crate::goldens::golden_effects("anthropic_oracle_concurrent_notes", &log);
    })
    .await;
}

/// A stateful stop: `StopAfterTurn(2)` ends the run after the answer
/// turn, and the header names the hook with its state.
#[tokio::test]
async fn stop_after_turn_two_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_oracle_cassette(
        "corpus_oracle/stop_after_turn_two",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .add_hook(StopAfterTurnN(2))
                .record_effects()
                .build();
            let error = agent
                .prompt(ADD_PROMPT)
                .max_turns(3)
                .await
                .expect_err("the hook stops the run");
            assert!(
                matches!(&error, PromptError::PromptCancelled { reason, .. } if *reason == stop_after_turn_reason(2)),
                "{error:?}"
            );
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion
                ]
            );
            assert_eq!(log.header.hooks, ["StopAfterTurn(2)"]);
            crate::goldens::golden_effects("anthropic_oracle_stop_after_turn_two", &log);
        },
    )
    .await;
}
