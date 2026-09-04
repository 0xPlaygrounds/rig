//! Matrix I of the effect corpus: a host's own effect, dispatched by the
//! agent's hooks over the host's bus (`CLAUDE_SONNET_4_6`, temperature
//! 0). The host registers the model under the agent's key and its own
//! `NoteTaker` under `host/note`, drives the bus and records; the agent
//! stamps the log. Producers of the goldens
//! `crates/rig-verify/tests/corpus_host.rs` replays by both interpreters;
//! the enumeration lives there. Every cell is a new recording under
//! `tests/cassettes/anthropic/corpus_host/`.

use futures::StreamExt;
use rig::agent::{AgentBuilder, MultiTurnStreamItem};
use rig::bus::{Bus, BusConfig};
use rig::effect::{EffectFamily, EffectKind, HandlerKey};
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::with_anthropic_corpus_host_cassette;
use crate::goldens::{
    NOTE_KEY, NoteAtCompletionCall, NoteAtOutcome, NoteAtSettled, NoteAtStart, NoteTaker,
    NoteTwice, NoteUnserved, families,
};
use crate::support::{Adder, BASIC_PREAMBLE, TOOLS_PREAMBLE};

const PROMPT: &str = "Reply with the single word: ready.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";

/// The hooks a cell registers, in order.
#[derive(Clone, Copy)]
enum Hooks {
    AtStart,
    AtCompletionCall,
    AtOutcome,
    AtSettled,
    StartAndSettled,
    Twice,
    Unserved,
}

/// What a cell asks of the host.
struct Host {
    /// Register the note taker.
    notes: bool,
    /// The host's serving policy.
    serial: bool,
    /// Keep stream events.
    streamed: bool,
    /// Advertise `add` and ask for a sum.
    with_tool: bool,
}

const PLAIN: Host = Host {
    notes: true,
    serial: false,
    streamed: false,
    with_tool: false,
};

async fn final_output(stream: &mut rig::agent::StreamingResult) -> String {
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    output.expect("a final response")
}

fn with_hooks<S>(builder: AgentBuilder<S>, hooks: Hooks) -> AgentBuilder<S> {
    match hooks {
        Hooks::AtStart => builder.add_hook(NoteAtStart),
        Hooks::AtCompletionCall => builder.add_hook(NoteAtCompletionCall),
        Hooks::AtOutcome => builder.add_hook(NoteAtOutcome),
        Hooks::AtSettled => builder.add_hook(NoteAtSettled),
        Hooks::StartAndSettled => builder.add_hook(NoteAtStart).add_hook(NoteAtSettled),
        Hooks::Twice => builder.add_hook(NoteTwice),
        Hooks::Unserved => builder.add_hook(NoteUnserved),
    }
}

/// The program over the host's bus, with `hooks` registered on the agent.
async fn over_host(
    client: rig::providers::anthropic::Client,
    host: Host,
    hooks: Hooks,
) -> rig::effect_log::EffectLog {
    let config = BusConfig {
        serial_per_handler: host.serial,
        ..BusConfig::default()
    };
    let (dispatcher, registrar, mut driver) = Bus::channel_with(config);
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
    if host.notes {
        driver
            .register_erased(
                HandlerKey::from(NOTE_KEY),
                rig::serve::ErasedHandler::new(NoteTaker),
            )
            .expect("a fresh key");
    }
    let recorder = if host.streamed {
        rig::effect_log::EffectLogRecorder::keeping_stream_events()
    } else {
        rig::effect_log::EffectLogRecorder::new()
    };
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let builder =
        AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
            .name("golden")
            .temperature(0.0);
    let agent = if host.with_tool {
        with_hooks(builder.preamble(TOOLS_PREAMBLE).tool(Adder), hooks).build()
    } else {
        with_hooks(builder.preamble(BASIC_PREAMBLE), hooks).build()
    };
    let prompt = if host.with_tool { ADD_PROMPT } else { PROMPT };
    let output = if host.streamed {
        let mut stream = agent.stream_prompt(prompt).max_turns(3).stream().await;
        let output = final_output(&mut stream).await;
        drop(stream);
        output
    } else {
        agent
            .prompt(prompt)
            .max_turns(3)
            .await
            .expect("the agent answers")
            .output
    };
    if host.with_tool {
        assert!(output.contains("42"), "{output}");
    }
    let log = agent.stamp(recorder.take());
    drop((agent, dispatcher, registrar));
    driver.await.expect("the host's driver");
    assert_eq!(log.header.bus, None, "the policy is the host's");
    log
}

fn note_ats(log: &rig::effect_log::EffectLog) -> Vec<String> {
    log.iter()
        .filter(|record| record.key.as_str() == NOTE_KEY)
        .map(|record| match &record.kind {
            EffectKind::Custom { kind, payload } => {
                assert_eq!(&**kind, "corpus:note");
                payload["at"]
                    .as_str()
                    .expect("a note names its point")
                    .to_owned()
            }
            other => panic!("a note, not {other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn custom_at_start_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_host_cassette("corpus_host/custom_at_start", |client| async move {
        let log = over_host(client, PLAIN, Hooks::AtStart).await;
        assert_eq!(
            families(&log),
            [EffectFamily::Custom, EffectFamily::Completion]
        );
        assert_eq!(note_ats(&log), ["start"]);
        crate::goldens::golden_effects("anthropic_host_custom_at_start", &log);
    })
    .await;
}

#[tokio::test]
async fn custom_at_completion_call_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_host_cassette(
        "corpus_host/custom_at_completion_call",
        |client| async move {
            let log = over_host(client, PLAIN, Hooks::AtCompletionCall).await;
            assert_eq!(
                families(&log),
                [EffectFamily::Custom, EffectFamily::Completion]
            );
            assert_eq!(note_ats(&log), ["completion_call"]);
            crate::goldens::golden_effects("anthropic_host_custom_at_completion_call", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn custom_at_outcome_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_host_cassette("corpus_host/custom_at_outcome", |client| async move {
        let host = Host {
            with_tool: true,
            ..PLAIN
        };
        let log = over_host(client, host, Hooks::AtOutcome).await;
        assert_eq!(
            families(&log),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Custom,
                EffectFamily::Completion
            ]
        );
        assert_eq!(note_ats(&log), ["outcome"]);
        crate::goldens::golden_effects("anthropic_host_custom_at_outcome", &log);
    })
    .await;
}

/// A dispatch from `on_run_settled`, after the answer: the recorder is
/// still tapping the host's bus, so the record follows the completion
/// that answered the run.
#[tokio::test]
async fn custom_at_settled_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_host_cassette("corpus_host/custom_at_settled", |client| async move {
        let log = over_host(client, PLAIN, Hooks::AtSettled).await;
        assert_eq!(
            families(&log),
            [EffectFamily::Completion, EffectFamily::Custom]
        );
        assert_eq!(note_ats(&log), ["settled"]);
        crate::goldens::golden_effects("anthropic_host_custom_at_settled", &log);
    })
    .await;
}

#[tokio::test]
async fn custom_start_and_settled_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_host_cassette(
        "corpus_host/custom_start_and_settled",
        |client| async move {
            let log = over_host(client, PLAIN, Hooks::StartAndSettled).await;
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Custom,
                    EffectFamily::Completion,
                    EffectFamily::Custom
                ]
            );
            assert_eq!(note_ats(&log), ["start", "settled"]);
            crate::goldens::golden_effects("anthropic_host_custom_start_and_settled", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn custom_twice_serial_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_host_cassette("corpus_host/custom_twice_serial", |client| async move {
        let host = Host {
            serial: true,
            ..PLAIN
        };
        let log = over_host(client, host, Hooks::Twice).await;
        assert_eq!(
            families(&log),
            [
                EffectFamily::Custom,
                EffectFamily::Custom,
                EffectFamily::Completion
            ]
        );
        assert_eq!(note_ats(&log), ["first", "second"]);
        crate::goldens::golden_effects("anthropic_host_custom_twice_serial", &log);
    })
    .await;
}

#[tokio::test]
async fn custom_twice_concurrent_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_host_cassette(
        "corpus_host/custom_twice_concurrent",
        |client| async move {
            let log = over_host(client, PLAIN, Hooks::Twice).await;
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Custom,
                    EffectFamily::Custom,
                    EffectFamily::Completion
                ]
            );
            assert_eq!(note_ats(&log), ["first", "second"]);
            crate::goldens::golden_effects("anthropic_host_custom_twice_concurrent", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn custom_at_start_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_host_cassette(
        "corpus_host/custom_at_start_streamed",
        |client| async move {
            let host = Host {
                streamed: true,
                ..PLAIN
            };
            let log = over_host(client, host, Hooks::AtStart).await;
            assert_eq!(
                families(&log),
                [EffectFamily::Custom, EffectFamily::Completion]
            );
            assert!(log.records[1].events.is_some(), "events are kept");
            crate::goldens::golden_effects("anthropic_host_custom_at_start_streamed", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn custom_at_outcome_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_host_cassette(
        "corpus_host/custom_at_outcome_streamed",
        |client| async move {
            let host = Host {
                streamed: true,
                with_tool: true,
                ..PLAIN
            };
            let log = over_host(client, host, Hooks::AtOutcome).await;
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Custom,
                    EffectFamily::Completion
                ]
            );
            assert_eq!(note_ats(&log), ["outcome"]);
            crate::goldens::golden_effects("anthropic_host_custom_at_outcome_streamed", &log);
        },
    )
    .await;
}

/// The host registered no note taker: the hook's bind is refused, the
/// run goes on, and nothing of the hook reaches the log but its name.
#[tokio::test]
async fn custom_unserved_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_host_cassette("corpus_host/custom_unserved", |client| async move {
        let host = Host {
            notes: false,
            ..PLAIN
        };
        let log = over_host(client, host, Hooks::Unserved).await;
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert_eq!(log.header.hooks, ["NoteUnserved"]);
        crate::goldens::golden_effects("anthropic_host_custom_unserved", &log);
    })
    .await;
}
