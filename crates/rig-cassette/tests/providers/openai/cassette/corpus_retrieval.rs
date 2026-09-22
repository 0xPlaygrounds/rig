//! Matrix A of the effect corpus, the OpenAI cells: retrieval effects over
//! an index embedded by `text-embedding-3-small` and a `gpt-4o` agent on
//! the Responses wire (dual-id tool calls). Producers of the goldens
//! `crates/rig-cassette/tests/corpus_retrieval.rs` replays; the enumeration
//! lives there. Every cell is a new recording under
//! `crates/rig-cassette/fixtures/cassettes/openai/corpus_retrieval/`.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::effect::{EffectFamily, Outcome, RetrievedDocuments};
use rig::prelude::*;
use rig::providers::openai;
use rig_cassette::agent::AgentReplayExt;

use super::super::support::with_openai_corpus_retrieval_cassette;
use crate::goldens::{
    FACT_PROMPT, FACTS, RETRIEVED_TOOLS_PREAMBLE, SUBTRACT_PROMPT, facts_index, families,
    retrievable_toolset, tool_index,
};
use crate::support::BASIC_PREAMBLE;

const EMBEDDING: &str = openai::TEXT_EMBEDDING_3_SMALL;
const MODEL: &str = openai::GPT_4O;

fn first_retrieved_ids(log: &rig::cassette::effect_log::EffectLog) -> Vec<String> {
    log.records
        .iter()
        .find_map(|record| match &record.outcome {
            Ok(Outcome::Documents(RetrievedDocuments::Scored(docs))) => {
                Some(docs.iter().map(|(_, id, _)| id.clone()).collect())
            }
            Ok(Outcome::Documents(RetrievedDocuments::Ids(ids))) => {
                Some(ids.iter().map(|(_, id)| id.clone()).collect())
            }
            _ => None,
        })
        .expect("a retrieval")
}

async fn final_output(stream: &mut rig::agent::StreamingResult) -> String {
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    output.expect("a final response")
}

#[tokio::test]
async fn dynamic_context_one_effect_log_is_the_golden_fixture() {
    with_openai_corpus_retrieval_cassette(
        "corpus_retrieval/dynamic_context_one",
        |client| async move {
            let index = facts_index(client.openai.embedding(EMBEDDING, None), &FACTS).await;
            let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
            let agent = client
                .openai
                .agent(MODEL)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .dynamic_context(1, index)
                .record_to(recorder.clone())
                .build();
            let response = agent.prompt(FACT_PROMPT).await.expect("the agent answers");
            assert!(!response.output.is_empty());
            let log = agent.stamp(recorder.take());
            assert_eq!(
                families(&log),
                [EffectFamily::Retrieve, EffectFamily::Completion]
            );
            assert_eq!(first_retrieved_ids(&log).len(), 1);
            crate::goldens::golden_effects("openai_retrieval_dynamic_context_one", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn dynamic_context_one_streamed_effect_log_is_the_golden_fixture() {
    with_openai_corpus_retrieval_cassette(
        "corpus_retrieval/dynamic_context_one_streamed",
        |client| async move {
            let index = facts_index(client.openai.embedding(EMBEDDING, None), &FACTS).await;
            let recorder = rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events();
            let agent = client
                .openai
                .agent(MODEL)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .dynamic_context(1, index)
                .record_to(recorder.clone())
                .build();
            let mut stream = agent.prompt(FACT_PROMPT).stream();
            let output = final_output(&mut stream).await;
            drop(stream);
            assert!(!output.is_empty());
            let log = agent.stamp(recorder.take());
            assert_eq!(
                families(&log),
                [EffectFamily::Retrieve, EffectFamily::Completion]
            );
            assert!(log.records[1].events.is_some(), "events are kept");
            crate::goldens::golden_effects("openai_retrieval_dynamic_context_one_streamed", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn retrieved_tools_one_effect_log_is_the_golden_fixture() {
    with_openai_corpus_retrieval_cassette(
        "corpus_retrieval/retrieved_tools_one",
        |client| async move {
            let toolset = retrievable_toolset();
            let index = tool_index(client.openai.embedding(EMBEDDING, None), &toolset).await;
            let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
            let agent = client
                .openai
                .agent(MODEL)
                .name("golden")
                .preamble(RETRIEVED_TOOLS_PREAMBLE)
                .temperature(0.0)
                .retrieved_tools(1, index, toolset)
                .record_to(recorder.clone())
                .build();
            let response = agent
                .prompt(SUBTRACT_PROMPT)
                .max_turns(3)
                .await
                .expect("the agent answers");
            assert!(response.output.contains("42"), "{}", response.output);
            let log = agent.stamp(recorder.take());
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Retrieve,
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Retrieve,
                    EffectFamily::Completion
                ]
            );
            assert_eq!(first_retrieved_ids(&log), ["subtract"]);
            crate::goldens::golden_effects("openai_retrieval_retrieved_tools_one", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn retrieved_tools_one_streamed_effect_log_is_the_golden_fixture() {
    with_openai_corpus_retrieval_cassette(
        "corpus_retrieval/retrieved_tools_one_streamed",
        |client| async move {
            let toolset = retrievable_toolset();
            let index = tool_index(client.openai.embedding(EMBEDDING, None), &toolset).await;
            let recorder = rig_cassette::effect_log::EffectLogRecorder::keeping_stream_events();
            let agent = client
                .openai
                .agent(MODEL)
                .name("golden")
                .preamble(RETRIEVED_TOOLS_PREAMBLE)
                .temperature(0.0)
                .retrieved_tools(1, index, toolset)
                .record_to(recorder.clone())
                .build();
            let mut stream = agent.prompt(SUBTRACT_PROMPT).max_turns(3).stream();
            let output = final_output(&mut stream).await;
            drop(stream);
            assert!(output.contains("42"), "{output}");
            let log = agent.stamp(recorder.take());
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Retrieve,
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Retrieve,
                    EffectFamily::Completion
                ]
            );
            assert!(log.records[1].events.is_some(), "events are kept");
            crate::goldens::golden_effects("openai_retrieval_retrieved_tools_one_streamed", &log);
        },
    )
    .await;
}

#[tokio::test]
async fn context_and_tools_effect_log_is_the_golden_fixture() {
    with_openai_corpus_retrieval_cassette(
        "corpus_retrieval/context_and_tools",
        |client| async move {
            let facts = facts_index(client.openai.embedding(EMBEDDING, None), &FACTS).await;
            let toolset = retrievable_toolset();
            let tools = tool_index(client.openai.embedding(EMBEDDING, None), &toolset).await;
            let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
            let agent = client
                .openai
                .agent(MODEL)
                .name("golden")
                .preamble(RETRIEVED_TOOLS_PREAMBLE)
                .temperature(0.0)
                .dynamic_context(1, facts)
                .retrieved_tools(1, tools, toolset)
                .record_to(recorder.clone())
                .build();
            let response = agent
                .prompt(SUBTRACT_PROMPT)
                .max_turns(3)
                .await
                .expect("the agent answers");
            assert!(response.output.contains("42"), "{}", response.output);
            let log = agent.stamp(recorder.take());
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Retrieve,
                    EffectFamily::Retrieve,
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Retrieve,
                    EffectFamily::Retrieve,
                    EffectFamily::Completion
                ]
            );
            crate::goldens::golden_effects("openai_retrieval_context_and_tools", &log);
        },
    )
    .await;
}
