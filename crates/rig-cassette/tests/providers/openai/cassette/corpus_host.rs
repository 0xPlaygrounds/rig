//! Matrix I of the effect corpus, the embedding cells: a hook embeds the
//! prompt through the host's `ModelAdapter` (`text-embedding-3-small`)
//! before the completion (`gpt-4o`, temperature 0). Both are on the wire;
//! each cell is a new recording under `crates/rig-cassette/fixtures/cassettes/openai/corpus_host/`.

use futures::StreamExt;
use rig::DynModel;
use rig::agent::{AgentBuilder, MultiTurnStreamItem};
use rig::bus::Bus;
use rig::effect::{EffectFamily, HandlerKey};
use rig::operation::Completion;
use rig::providers::openai;
use rig_cassette::agent::AgentReplayExt;

use super::super::support::{OpenAiCassette, with_openai_corpus_host_cassette};
use crate::goldens::{EMBED_KEY, EmbedPrompt, families};
use crate::support::BASIC_PREAMBLE;

const PROMPT: &str = "Reply with the single word: ready.";

async fn final_output(stream: &mut rig::agent::StreamingResult) -> String {
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output());
        }
    }
    output.expect("a final response")
}

async fn embeds_over_host(
    client: OpenAiCassette,
    streamed: bool,
) -> rig::cassette::effect_log::EffectLog {
    let completion = client.openai.completion(openai::GPT_4O);
    embeds_over_host_with(client, completion, streamed).await
}

/// [`embeds_over_host`], with the agent's completion model given.
async fn embeds_over_host_with(
    client: OpenAiCassette,
    completion: impl Into<DynModel<Completion>>,
    streamed: bool,
) -> rig::cassette::effect_log::EffectLog {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let model_key = HandlerKey::from("golden/model:default");
    driver
        .register_erased(
            model_key.clone(),
            rig::serve::ErasedHandler::new(rig::serve::adapters::ModelAdapter::new(
                "default", completion,
            )),
        )
        .expect("a fresh key");
    driver
        .register_erased(
            HandlerKey::from(EMBED_KEY),
            rig::serve::ErasedHandler::new(rig::serve::adapters::ModelAdapter::new(
                "host",
                client
                    .openai
                    .embedding(openai::TEXT_EMBEDDING_3_SMALL, None),
            )),
        )
        .expect("a fresh key");
    let recorder = if streamed {
        rig::cassette::effect_log::EffectLogRecorder::keeping_stream_events()
    } else {
        rig::cassette::effect_log::EffectLogRecorder::new()
    };
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let agent = AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
        .name("golden")
        .preamble(BASIC_PREAMBLE)
        .temperature(0.0)
        .add_hook(EmbedPrompt)
        .build();
    let output = if streamed {
        let mut stream = agent.prompt(PROMPT).stream();
        let output = final_output(&mut stream).await;
        drop(stream);
        output
    } else {
        agent
            .prompt(PROMPT)
            .await
            .expect("the agent answers")
            .output()
    };
    assert!(output.to_lowercase().contains("ready"), "{output}");
    let log = agent.stamp(recorder.take());
    drop((agent, dispatcher, registrar));
    driver.await.expect("the host's driver");
    assert_eq!(
        families(&log),
        [EffectFamily::Embed, EffectFamily::Completion]
    );
    assert!(
        !log.header
            .required
            .contains_key(&HandlerKey::from(EMBED_KEY)),
        "the host's embedding is not in the agent's row"
    );
    assert!(
        log.header
            .signature
            .contains_key(&HandlerKey::from(EMBED_KEY)),
        "but it is in the signature"
    );
    log
}

#[tokio::test]
async fn embed_prompt_effect_log_is_the_golden_fixture() {
    with_openai_corpus_host_cassette("corpus_host/embed_prompt", |client| async move {
        let log = embeds_over_host(client, false).await;
        crate::goldens::golden_effects("openai_host_embed_prompt", &log);
    })
    .await;
}

#[tokio::test]
async fn embed_prompt_streamed_effect_log_is_the_golden_fixture() {
    with_openai_corpus_host_cassette("corpus_host/embed_prompt_streamed", |client| async move {
        let log = embeds_over_host(client, true).await;
        assert!(log.records[1].events.is_some(), "events are kept");
        crate::goldens::golden_effects("openai_host_embed_prompt_streamed", &log);
    })
    .await;
}

/// An agent whose completion model is chosen at runtime from configuration
/// and erased to its operation records the same effect log, observed over
/// the bus, and replays the same recording.
#[tokio::test]
async fn a_model_chosen_from_config_yields_the_golden_effect_log() {
    with_openai_corpus_host_cassette("corpus_host/embed_prompt", |client| async move {
        let reference = rig::providers::registry::ProviderRef::parse("openai/gpt-4o")
            .expect("a registered reference");
        let config = client.openai.config.clone();
        let http = rig_test_support::rebased::Rebased::new(
            "https://api.openai.com/v1",
            config.base_url.clone(),
            client.openai.http.clone(),
        );
        let completion: DynModel<Completion> =
            reference.completion_model_with(config.api_key, http);
        let log = embeds_over_host_with(client, completion, false).await;
        // The golden's one producer is the test above; this one reads it as
        // data. A live recording's bytes differ, so it is replay-only.
        if std::env::var("RIG_PROVIDER_TEST_MODE").is_ok_and(|mode| mode == "record") {
            return;
        }
        let golden = crate::goldens::golden_path("openai_host_embed_prompt");
        let committed = std::fs::read_to_string(golden).expect("the golden is committed");
        let rendered = serde_json::to_string_pretty(&log).expect("the log serializes");
        assert_eq!(committed.trim_end(), rendered);
    })
    .await;
}
