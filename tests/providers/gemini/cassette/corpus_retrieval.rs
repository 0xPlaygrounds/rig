//! Matrix A of the effect corpus, the Gemini cells: retrieval effects. A
//! `dynamic_context` index and a `retrieved_tools` index, both embedded by
//! Gemini (`gemini-embedding-001`) and queried by the agent over the bus,
//! so every `Retrieve` dispatch is a record. Producers of the goldens
//! `crates/rig-verify/tests/corpus_retrieval.rs` replays by both
//! interpreters; the enumeration lives there.
//!
//! Every cell is a new recording under
//! `tests/cassettes/gemini/corpus_retrieval/`: the document and tool-schema
//! embeddings at build time, the query embedding at prompt time, and the
//! completion turns.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::effect::{EffectFamily, EffectKind, Outcome, RetrieveQuery, RetrievedDocuments};
use rig::prelude::*;
use rig::providers::gemini;

use super::super::support::with_gemini_corpus_retrieval_cassette;
use crate::goldens::{
    ADD_THEN_SUBTRACT_PROMPT, EmbedSubtract, FACT_PROMPT, FACTS, RETRIEVED_TOOLS_PREAMBLE,
    SUBTRACT_PROMPT, facts_index, families, retrievable_toolset, tool_index,
};
use crate::support::{Adder, BASIC_PREAMBLE};

const EMBEDDING: &str = gemini::embedding::EMBEDDING_001;
const MODEL: &str = gemini::completion::GEMINI_2_5_FLASH;

/// The ids each `Retrieve` record answered with, per record.
fn retrieved_ids(log: &rig::effect_log::EffectLog) -> Vec<Vec<String>> {
    log.records
        .iter()
        .filter_map(|record| match &record.outcome {
            Ok(Outcome::Documents(RetrievedDocuments::Scored(docs))) => {
                Some(docs.iter().map(|(_, id, _)| id.clone()).collect())
            }
            Ok(Outcome::Documents(RetrievedDocuments::Ids(ids))) => {
                Some(ids.iter().map(|(_, id)| id.clone()).collect())
            }
            _ => None,
        })
        .collect()
}

/// The documents each `TopN` record answered with, rendered.
fn retrieved_texts(log: &rig::effect_log::EffectLog) -> Vec<String> {
    log.records
        .iter()
        .filter_map(|record| match &record.outcome {
            Ok(Outcome::Documents(RetrievedDocuments::Scored(docs))) => Some(
                docs.iter()
                    .map(|(_, _, value)| value.to_string())
                    .collect::<Vec<_>>()
                    .join(" | "),
            ),
            _ => None,
        })
        .collect()
}

fn retrieve_kinds(log: &rig::effect_log::EffectLog) -> Vec<&'static str> {
    log.records
        .iter()
        .filter_map(|record| match &record.kind {
            EffectKind::Retrieve {
                query: RetrieveQuery::TopN { .. },
            } => Some("top_n"),
            EffectKind::Retrieve {
                query: RetrieveQuery::TopNIds { .. },
            } => Some("top_n_ids"),
            _ => None,
        })
        .collect()
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

/// `dynamic_context(1, facts)`: one `TopN` retrieval, then the completion
/// whose request holds the retrieved fact as a document.
#[tokio::test]
async fn dynamic_context_one_effect_log_is_the_golden_fixture() {
    with_gemini_corpus_retrieval_cassette(
        "corpus_retrieval/dynamic_context_one",
        |client| async move {
            let index = facts_index(client.embedding_model(EMBEDDING), &FACTS).await;
            let agent = client
                .agent(MODEL)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .dynamic_context(1, index)
                .record_effects()
                .build();
            let response = agent.prompt(FACT_PROMPT).await.expect("the agent answers");
            assert!(!response.output.is_empty());
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [EffectFamily::Retrieve, EffectFamily::Completion]
            );
            assert_eq!(retrieve_kinds(&log), ["top_n"]);
            assert_eq!(retrieved_ids(&log)[0].len(), 1);
            assert!(
                retrieved_texts(&log)[0].contains("glarb-glarb"),
                "the fact about the prompt's word: {:?}",
                retrieved_texts(&log)
            );
            assert_eq!(
                log.header
                    .required
                    .get(&rig::effect::HandlerKey::from("golden/retrieve:context#0")),
                Some(&EffectFamily::Retrieve)
            );
            crate::goldens::golden_effects("gemini_retrieval_dynamic_context_one", &log);
        },
    )
    .await;
}

/// `dynamic_context(2, facts)`, streamed with events.
#[tokio::test]
async fn dynamic_context_two_streamed_effect_log_is_the_golden_fixture() {
    with_gemini_corpus_retrieval_cassette(
        "corpus_retrieval/dynamic_context_two_streamed",
        |client| async move {
            let index = facts_index(client.embedding_model(EMBEDDING), &FACTS).await;
            let agent = client
                .agent(MODEL)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .dynamic_context(2, index)
                .record_effects_with_events()
                .build();
            let mut stream = agent.stream_prompt(FACT_PROMPT).stream().await;
            let output = final_output(&mut stream).await;
            drop(stream);
            assert!(!output.is_empty());
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [EffectFamily::Retrieve, EffectFamily::Completion]
            );
            assert_eq!(retrieved_ids(&log)[0].len(), 2);
            assert!(log.records[1].events.is_some(), "events are kept");
            crate::goldens::golden_effects("gemini_retrieval_dynamic_context_two_streamed", &log);
        },
    )
    .await;
}

/// `dynamic_context(5, facts)` over three facts: the index answers with
/// all three.
#[tokio::test]
async fn dynamic_context_over_sampled_effect_log_is_the_golden_fixture() {
    with_gemini_corpus_retrieval_cassette(
        "corpus_retrieval/dynamic_context_over_sampled",
        |client| async move {
            let index = facts_index(client.embedding_model(EMBEDDING), &FACTS).await;
            let agent = client
                .agent(MODEL)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .dynamic_context(5, index)
                .record_effects()
                .build();
            let response = agent.prompt(FACT_PROMPT).await.expect("the agent answers");
            assert!(!response.output.is_empty());
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [EffectFamily::Retrieve, EffectFamily::Completion]
            );
            assert_eq!(retrieved_ids(&log)[0].len(), 3, "every fact, no more");
            crate::goldens::golden_effects("gemini_retrieval_dynamic_context_over_sampled", &log);
        },
    )
    .await;
}

/// `dynamic_context(1, empty)`: the query is embedded, the index answers
/// with no documents, the request carries none.
#[tokio::test]
async fn dynamic_context_empty_index_effect_log_is_the_golden_fixture() {
    with_gemini_corpus_retrieval_cassette(
        "corpus_retrieval/dynamic_context_empty_index",
        |client| async move {
            let index = facts_index(client.embedding_model(EMBEDDING), &[]).await;
            let agent = client
                .agent(MODEL)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .dynamic_context(1, index)
                .record_effects()
                .build();
            let response = agent.prompt(FACT_PROMPT).await.expect("the agent answers");
            assert!(!response.output.is_empty());
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [EffectFamily::Retrieve, EffectFamily::Completion]
            );
            assert_eq!(retrieved_ids(&log), [Vec::<String>::new()]);
            let request = match &log.records[1].kind {
                EffectKind::Completion { request, .. } => request,
                other => panic!("a completion, not {other:?}"),
            };
            assert!(request.documents.is_empty());
            crate::goldens::golden_effects("gemini_retrieval_dynamic_context_empty_index", &log);
        },
    )
    .await;
}

/// `retrieved_tools(1, tools)`: a `TopNIds` retrieval before every model
/// call, the retrieved tool advertised and called.
#[tokio::test]
async fn retrieved_tools_one_effect_log_is_the_golden_fixture() {
    with_gemini_corpus_retrieval_cassette(
        "corpus_retrieval/retrieved_tools_one",
        |client| async move {
            let toolset = retrievable_toolset();
            let index = tool_index(client.embedding_model(EMBEDDING), &toolset).await;
            let agent = client
                .agent(MODEL)
                .name("golden")
                .preamble(RETRIEVED_TOOLS_PREAMBLE)
                .temperature(0.0)
                .retrieved_tools(1, index, toolset)
                .record_effects()
                .build();
            let response = agent
                .prompt(SUBTRACT_PROMPT)
                .max_turns(3)
                .await
                .expect("the agent answers");
            assert!(response.output.contains("42"), "{}", response.output);
            let log = agent.take_effect_log().expect("recording");
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
            assert_eq!(retrieve_kinds(&log), ["top_n_ids", "top_n_ids"]);
            assert_eq!(retrieved_ids(&log)[0], ["subtract"]);
            assert_eq!(
                log.header
                    .required
                    .get(&rig::effect::HandlerKey::from("golden/retrieve:tools#0")),
                Some(&EffectFamily::Retrieve)
            );
            crate::goldens::golden_effects("gemini_retrieval_retrieved_tools_one", &log);
        },
    )
    .await;
}

/// A static `add` and a retrievable `subtract`: the static tool is always
/// advertised, the retrieved one when its index says so.
#[tokio::test]
async fn retrieved_tools_with_static_effect_log_is_the_golden_fixture() {
    with_gemini_corpus_retrieval_cassette(
        "corpus_retrieval/retrieved_tools_with_static",
        |client| async move {
            let mut toolset = rig::tool::ToolSet::default();
            toolset
                .add_retrieved_tool(EmbedSubtract)
                .expect("the tool context serializes");
            let index = tool_index(client.embedding_model(EMBEDDING), &toolset).await;
            let agent = client
                .agent(MODEL)
                .name("golden")
                .preamble(RETRIEVED_TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .retrieved_tools(1, index, toolset)
                .record_effects()
                .build();
            let response = agent
                .prompt(ADD_THEN_SUBTRACT_PROMPT)
                .max_turns(6)
                .await
                .expect("the agent answers");
            assert!(response.output.contains("21"), "{}", response.output);
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Retrieve,
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Retrieve,
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Retrieve,
                    EffectFamily::Completion
                ]
            );
            crate::goldens::golden_effects("gemini_retrieval_retrieved_tools_with_static", &log);
        },
    )
    .await;
}

/// Both: the context retrieval, then the tool retrieval, before every
/// model call, in that order.
#[tokio::test]
async fn context_and_tools_effect_log_is_the_golden_fixture() {
    with_gemini_corpus_retrieval_cassette(
        "corpus_retrieval/context_and_tools",
        |client| async move {
            let facts = facts_index(client.embedding_model(EMBEDDING), &FACTS).await;
            let toolset = retrievable_toolset();
            let tools = tool_index(client.embedding_model(EMBEDDING), &toolset).await;
            let agent = client
                .agent(MODEL)
                .name("golden")
                .preamble(RETRIEVED_TOOLS_PREAMBLE)
                .temperature(0.0)
                .dynamic_context(1, facts)
                .retrieved_tools(1, tools, toolset)
                .record_effects()
                .build();
            let response = agent
                .prompt(SUBTRACT_PROMPT)
                .max_turns(3)
                .await
                .expect("the agent answers");
            assert!(response.output.contains("42"), "{}", response.output);
            let log = agent.take_effect_log().expect("recording");
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
            assert_eq!(
                retrieve_kinds(&log),
                ["top_n", "top_n_ids", "top_n", "top_n_ids"]
            );
            crate::goldens::golden_effects("gemini_retrieval_context_and_tools", &log);
        },
    )
    .await;
}
