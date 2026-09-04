//! Matrix I of the effect corpus, the embedding cells: a hook embeds the
//! prompt through the host's `EmbedAdapter` (`text-embedding-3-small`)
//! before the completion (`gpt-4o`, temperature 0). Both are on the wire;
//! each cell is a new recording under `tests/cassettes/openai/corpus_host/`.

use futures::StreamExt;
use rig::agent::{AgentBuilder, MultiTurnStreamItem};
use rig::bus::Bus;
use rig::effect::{EffectFamily, HandlerKey};
use rig::prelude::*;
use rig::providers::openai;

use super::super::support::with_openai_corpus_host_cassette;
use crate::goldens::{EMBED_KEY, EmbedPrompt, families};
use crate::support::BASIC_PREAMBLE;

const PROMPT: &str = "Reply with the single word: ready.";

async fn final_output(stream: &mut rig::agent::StreamingResult) -> String {
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    output.expect("a final response")
}

async fn embeds_over_host(client: openai::Client, streamed: bool) -> rig::effect_log::EffectLog {
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let model_key = HandlerKey::from("golden/model:default");
    driver
        .register_erased(
            model_key.clone(),
            rig::serve::ErasedHandler::new(rig::serve::adapters::CompletionAdapter::new(
                "default",
                client.completion_model(openai::GPT_4O),
            )),
        )
        .expect("a fresh key");
    driver
        .register_erased(
            HandlerKey::from(EMBED_KEY),
            rig::serve::ErasedHandler::new(rig::serve::adapters::EmbedAdapter::new(
                "host",
                client.embedding_model(openai::TEXT_EMBEDDING_3_SMALL),
            )),
        )
        .expect("a fresh key");
    let recorder = if streamed {
        rig::effect_log::EffectLogRecorder::keeping_stream_events()
    } else {
        rig::effect_log::EffectLogRecorder::new()
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
        let mut stream = agent.stream_prompt(PROMPT).stream().await;
        let output = final_output(&mut stream).await;
        drop(stream);
        output
    } else {
        agent
            .prompt(PROMPT)
            .await
            .expect("the agent answers")
            .output
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
