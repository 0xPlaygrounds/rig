//! Matrix N of the effect corpus, the openai rows: the pass-2 shapes on
//! this wire where the wire changes the record. Producers of the goldens
//! `crates/rig-verify/tests/corpus_breadth.rs` replays by both
//! interpreters; the enumeration lives there. Every cell is a new
//! recording under `tests/cassettes/openai/corpus_breadth/`.

use futures::StreamExt;
use rig::agent::{AgentBuilder, MultiTurnStreamItem, StreamingError};
use rig::bus::Bus;
use rig::completion::PromptError;
use rig::effect::{EffectFamily, HandlerKey};
use rig::prelude::*;
use rig::providers::openai;
use rig::run::OutputMode;

use super::super::support::with_openai_corpus_breadth_cassette;
use crate::goldens::{
    CANCEL_ADD_DISPATCH, CONVERSATION, CancelAddDispatch, EMBED_KEY, NOTE_KEY, NoteAtOutcome,
    NoteTaker, STOP_ON_TEXT_DELTA, StopOnTextDelta, event_schema, families,
};
use crate::support::{Adder, BASIC_PREAMBLE, TOOLS_PREAMBLE};

const MODEL: &str = openai::GPT_4O;
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const STRUCTURED_OUTPUT_PROMPT: &str =
    "Return a concise event object for a local Rust meetup in Seattle.";
const ESSAY_PROMPT: &str =
    "Write four paragraphs about the history of the Rust programming language.";
const PROMPT: &str = "Reply with the single word: ready.";
const SECOND_PROMPT: &str = "Now reply with the single word: again.";

fn assert_event(output: &str) {
    let object: serde_json::Value =
        serde_json::from_str(output).unwrap_or_else(|_| panic!("the schema's object: {output}"));
    assert!(
        object["title"].is_string() && object["summary"].is_string(),
        "{object}"
    );
}

async fn final_output(stream: &mut rig::agent::StreamingResult) -> Result<String, String> {
    let mut output = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(response)) => output = Some(response.output),
            Ok(_) => {}
            Err(StreamingError::Prompt(error)) => match *error {
                PromptError::PromptCancelled { reason, .. } => return Err(reason),
                other => panic!("the stream yields: {other:?}"),
            },
            Err(other) => panic!("the stream yields: {other:?}"),
        }
    }
    Ok(output.expect("a final response"))
}

/// A host's bus with the model, and the host's note taker or embedding model.
fn host_bus(
    client: &openai::Client,
    notes: bool,
    embeds: bool,
) -> (
    rig::bus::Dispatcher,
    rig::bus::Registrar,
    rig::bus::BusDriver,
    HandlerKey,
) {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let model_key = HandlerKey::from("golden/model:default");
    driver
        .register_erased(
            model_key.clone(),
            rig::serve::ErasedHandler::new(rig::serve::adapters::CompletionAdapter::new(
                "default",
                client.completion_model(MODEL),
            )),
        )
        .expect("a fresh key");
    if notes {
        driver
            .register_erased(
                HandlerKey::from(NOTE_KEY),
                rig::serve::ErasedHandler::new(NoteTaker),
            )
            .expect("a fresh key");
    }
    if embeds {
        driver
            .register_erased(
                HandlerKey::from(EMBED_KEY),
                rig::serve::ErasedHandler::new(rig::serve::adapters::EmbedAdapter::new(
                    "host",
                    client.embedding_model(openai::TEXT_EMBEDDING_3_SMALL),
                )),
            )
            .expect("a fresh key");
    }
    (dispatcher, registrar, driver, model_key)
}

/// `Tool` output mode, streamed with events: the output tool's call as
/// this wire ids it.
#[tokio::test]
async fn output_tool_streamed_effect_log_is_the_golden_fixture() {
    with_openai_corpus_breadth_cassette(
        "corpus_breadth/output_tool_streamed",
        |client| async move {
            let agent = client
                .agent(MODEL)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .output_schema_raw(event_schema())
                .output_mode(OutputMode::Tool)
                .record_effects_with_events()
                .build();
            let mut stream = agent.stream_prompt(STRUCTURED_OUTPUT_PROMPT).stream().await;
            let output = final_output(&mut stream).await.expect("the run answers");
            drop(stream);
            assert_event(&output);
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(log.records[0].events.is_some(), "events are kept");
            crate::goldens::golden_effects("openai_breadth_output_tool_streamed", &log);
        },
    )
    .await;
}

/// A stop on the first text delta of a long streamed answer: the
/// completion is cancelled at the delta.
#[tokio::test]
async fn text_delta_stop_effect_log_is_the_golden_fixture() {
    with_openai_corpus_breadth_cassette("corpus_breadth/text_delta_stop", |client| async move {
        let agent = client
            .agent(MODEL)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .add_hook(StopOnTextDelta)
            .record_effects_with_events()
            .build();
        let mut stream = agent.stream_prompt(ESSAY_PROMPT).stream().await;
        let reason = final_output(&mut stream)
            .await
            .expect_err("the hook stops the run");
        drop(stream);
        for _ in 0..64 {
            tokio::task::yield_now().await;
        }
        assert_eq!(reason, STOP_ON_TEXT_DELTA);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert_eq!(
            log.records[0]
                .outcome
                .as_ref()
                .err()
                .map(|report| report.kind),
            Some(rig::error::ErrorKind::Cancelled),
            "{:?}",
            log.records[0].outcome
        );
        crate::goldens::golden_effects("openai_breadth_text_delta_stop", &log);
    })
    .await;
}

/// `on_dispatch` → `Deny(Cancelled)` on the tool: the completion is
/// recorded, the tool never reaches the bus.
#[tokio::test]
async fn tool_dispatch_cancelled_effect_log_is_the_golden_fixture() {
    with_openai_corpus_breadth_cassette(
        "corpus_breadth/tool_dispatch_cancelled",
        |client| async move {
            let agent = client
                .agent(MODEL)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .add_hook(CancelAddDispatch)
                .record_effects()
                .build();
            let error = agent
                .prompt(ADD_PROMPT)
                .max_turns(3)
                .await
                .expect_err("the hook stops the run");
            assert!(
                matches!(&error, PromptError::PromptCancelled { reason, .. } if reason == CANCEL_ADD_DISPATCH),
                "{error:?}"
            );
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            crate::goldens::golden_effects("openai_breadth_tool_dispatch_cancelled", &log);
        },
    )
    .await;
}

/// A host's custom note inside the tool's dispatch, beside this wire's
/// tool-call ids.
#[tokio::test]
async fn custom_at_outcome_effect_log_is_the_golden_fixture() {
    with_openai_corpus_breadth_cassette("corpus_breadth/custom_at_outcome", |client| async move {
        let (dispatcher, registrar, mut driver, model_key) = host_bus(&client, true, false);
        let recorder = rig::effect_log::EffectLogRecorder::new();
        driver.record_to(recorder.clone());
        let driver = tokio::spawn(driver);
        let agent =
            AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .add_hook(NoteAtOutcome)
                .build();
        let output = agent
            .prompt(ADD_PROMPT)
            .max_turns(3)
            .await
            .expect("the agent answers")
            .output;
        assert!(output.contains("42"), "{output}");
        let log = agent.stamp(recorder.take());
        drop((agent, dispatcher, registrar));
        driver.await.expect("the host's driver");
        assert_eq!(
            families(&log),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Custom,
                EffectFamily::Completion
            ]
        );
        crate::goldens::golden_effects("openai_breadth_custom_at_outcome", &log);
    })
    .await;
}

/// `Prompted` output mode, streamed with events, on the Responses wire.
#[tokio::test]
async fn prompted_streamed_effect_log_is_the_golden_fixture() {
    with_openai_corpus_breadth_cassette("corpus_breadth/prompted_streamed", |client| async move {
        let agent = client
            .agent(MODEL)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .output_schema_raw(event_schema())
            .output_mode(OutputMode::Prompted)
            .record_effects_with_events()
            .build();
        let mut stream = agent.stream_prompt(STRUCTURED_OUTPUT_PROMPT).stream().await;
        let output = final_output(&mut stream).await.expect("the run answers");
        drop(stream);
        assert_event(&output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        crate::goldens::golden_effects("openai_breadth_prompted_streamed", &log);
    })
    .await;
}

/// Two runs over one conversation on the Responses wire: the second
/// load holds the first append.
#[tokio::test]
async fn memory_two_runs_effect_log_is_the_golden_fixture() {
    with_openai_corpus_breadth_cassette("corpus_breadth/memory_two_runs", |client| async move {
        let agent = client
            .agent(MODEL)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .memory(rig::memory::InMemoryConversationMemory::new())
            .conversation(CONVERSATION)
            .record_effects()
            .build();
        for prompt in [PROMPT, SECOND_PROMPT] {
            let output = agent
                .prompt(prompt)
                .await
                .expect("the agent answers")
                .output;
            assert!(!output.is_empty());
        }
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [
                EffectFamily::Memory,
                EffectFamily::Completion,
                EffectFamily::Memory,
                EffectFamily::Memory,
                EffectFamily::Completion,
                EffectFamily::Memory,
            ]
        );
        crate::goldens::golden_effects("openai_breadth_memory_two_runs", &log);
    })
    .await;
}
