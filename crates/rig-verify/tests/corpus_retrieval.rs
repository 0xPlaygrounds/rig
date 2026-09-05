//! Matrix A: retrieval effects.
//!
//! The family the original corpus never recorded. `dynamic_context`
//! dispatches a `TopN` retrieval from a completion-call hook before every
//! model call and patches the retrieved documents into the request;
//! `retrieved_tools` dispatches a `TopNIds` retrieval at the request
//! boundary and advertises the named tools. Both go over the bus under the
//! agent's keys (`<owner>/retrieve:context#0`, `<owner>/retrieve:tools#0`),
//! both are in the required row, and on replay a replayer answers them
//! from the record: no index, no embedding model, no provider. The
//! embeddings themselves are not effects — an index embeds its query
//! inside the retrieval handler — so the `Embed` and `Rerank` families
//! stay unrecorded by an agent program; a host that dispatches them over
//! its own bus is outside this corpus.
//!
//! # Dimensions
//!
//! | axis | values recorded |
//! |---|---|
//! | retrieval source | `dynamic_context` · `retrieved_tools` · both |
//! | `samples` | 1 · 2 · 5 over three documents |
//! | index | three facts · empty · two tool schemas · one tool schema beside a static tool |
//! | embedding provider | gemini (`gemini-embedding-001`) · openai (`text-embedding-3-small`) |
//! | completion wire | gemini (id-less calls) · openai (dual-id calls) |
//! | transport | unary · streamed with events |
//!
//! Full cross-product: 3 × 3 × 4 × 2 × 2 × 2 = 288. Recorded: the 12 cells
//! below. Pruned: an embedding provider crossed with another provider's
//! completion (a cassette records one provider's wire); `samples` beyond
//! 1 for tool retrieval (a two-tool set retrieved whole is the static case
//! with an extra record); a cohere or other embedding wire (the retrieval
//! record holds the index's answer, not the embedding, so a third wire
//! adds a cassette and no new record shape); an empty tool index (a
//! program with no tool to advertise is the completion smoke).
//!
//! # Cells
//!
//! | golden | producer | shape |
//! |---|---|---|
//! | `gemini_retrieval_dynamic_context_one` | gemini `corpus_retrieval.rs` `dynamic_context_one_…` | `[Retrieve(TopN), Completion]`, `doc1` in the request |
//! | `gemini_retrieval_dynamic_context_two_streamed` | `dynamic_context_two_streamed_…` | the same with two documents, events kept |
//! | `gemini_retrieval_dynamic_context_over_sampled` | `dynamic_context_over_sampled_…` | `samples: 5`, three documents answered |
//! | `gemini_retrieval_dynamic_context_empty_index` | `dynamic_context_empty_index_…` | no documents answered, none in the request |
//! | `gemini_retrieval_retrieved_tools_one` | `retrieved_tools_one_…` | `[Retrieve(TopNIds), Completion, Tool, Retrieve, Completion]`, `subtract` retrieved and called |
//! | `gemini_retrieval_retrieved_tools_with_static` | `retrieved_tools_with_static_…` | a static `add` and a retrieved `subtract`, two tool turns |
//! | `gemini_retrieval_context_and_tools` | `context_and_tools_…` | `[Retrieve(TopN), Retrieve(TopNIds), Completion, Tool, Retrieve, Retrieve, Completion]` |
//! | `openai_retrieval_dynamic_context_one` | openai `corpus_retrieval.rs` `dynamic_context_one_…` | `[Retrieve, Completion]` |
//! | `openai_retrieval_dynamic_context_one_streamed` | `dynamic_context_one_streamed_…` | the same, events kept |
//! | `openai_retrieval_retrieved_tools_one` | `retrieved_tools_one_…` | `[Retrieve, Completion, Tool, Retrieve, Completion]`, dual-id call |
//! | `openai_retrieval_retrieved_tools_one_streamed` | `retrieved_tools_one_streamed_…` | the same, events kept |
//! | `openai_retrieval_context_and_tools` | `context_and_tools_…` | both retrievals before every model call |
//!
//! Every cell is a new recording: the index's embeddings at build time,
//! the query embedding at prompt time, the completion turns.
//!
//! # What the matrix found
//!
//! - The replay had no way to register a retrieval index from a log: the
//!   builder's `dynamic_context` and the tool server's `retrieved_tools`
//!   take a `VectorStoreIndex`, and a replayer is a handler.
//!   `dynamic_context_handler` and `retrieved_tools_handler` register any
//!   retrieval-family handler under the same keys, as `memory_handler` does
//!   for memory.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

mod corpus;

use corpus::Program;

const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const FACT_PROMPT: &str = "What is a glarb-glarb? Answer in one sentence.";
const RETRIEVED_TOOLS_PREAMBLE: &str =
    "You are a calculator. You must use the provided tools for every arithmetic operation.";
const SUBTRACT_PROMPT: &str =
    "Subtract 8 from 50 with the subtract tool, then reply with just the number.";
const ADD_THEN_SUBTRACT_PROMPT: &str = "First add 20 and 5 with the add tool. Then subtract 4 from that sum with the subtract tool. Report the final number.";

const CONTEXT: Program = Program {
    preamble: Some(BASIC_PREAMBLE),
    prompt: FACT_PROMPT,
    temperature: Some(0.0),
    ..Program::DEFAULT
};
const TOOLS: Program = Program {
    preamble: Some(RETRIEVED_TOOLS_PREAMBLE),
    prompt: SUBTRACT_PROMPT,
    temperature: Some(0.0),
    max_turns: Some(3),
    retrieved_tools: Some(1),
    retrievable: &["add", "subtract"],
    ..Program::DEFAULT
};

const GEMINI_DYNAMIC_CONTEXT_ONE: Program = Program {
    fixture: "gemini_retrieval_dynamic_context_one",
    dynamic_context: Some(1),
    ..CONTEXT
};
const GEMINI_DYNAMIC_CONTEXT_TWO_STREAMED: Program = Program {
    fixture: "gemini_retrieval_dynamic_context_two_streamed",
    dynamic_context: Some(2),
    streamed: true,
    ..CONTEXT
};
const GEMINI_DYNAMIC_CONTEXT_OVER_SAMPLED: Program = Program {
    fixture: "gemini_retrieval_dynamic_context_over_sampled",
    dynamic_context: Some(5),
    ..CONTEXT
};
const GEMINI_DYNAMIC_CONTEXT_EMPTY_INDEX: Program = Program {
    fixture: "gemini_retrieval_dynamic_context_empty_index",
    dynamic_context: Some(1),
    ..CONTEXT
};
const GEMINI_RETRIEVED_TOOLS_ONE: Program = Program {
    fixture: "gemini_retrieval_retrieved_tools_one",
    ..TOOLS
};
const GEMINI_RETRIEVED_TOOLS_WITH_STATIC: Program = Program {
    fixture: "gemini_retrieval_retrieved_tools_with_static",
    prompt: ADD_THEN_SUBTRACT_PROMPT,
    max_turns: Some(6),
    retrievable: &["subtract"],
    ..TOOLS
};
const GEMINI_CONTEXT_AND_TOOLS: Program = Program {
    fixture: "gemini_retrieval_context_and_tools",
    dynamic_context: Some(1),
    ..TOOLS
};
const OPENAI_DYNAMIC_CONTEXT_ONE: Program = Program {
    fixture: "openai_retrieval_dynamic_context_one",
    dynamic_context: Some(1),
    ..CONTEXT
};
const OPENAI_DYNAMIC_CONTEXT_ONE_STREAMED: Program = Program {
    fixture: "openai_retrieval_dynamic_context_one_streamed",
    dynamic_context: Some(1),
    streamed: true,
    ..CONTEXT
};
const OPENAI_RETRIEVED_TOOLS_ONE: Program = Program {
    fixture: "openai_retrieval_retrieved_tools_one",
    ..TOOLS
};
const OPENAI_RETRIEVED_TOOLS_ONE_STREAMED: Program = Program {
    fixture: "openai_retrieval_retrieved_tools_one_streamed",
    streamed: true,
    ..TOOLS
};
const OPENAI_CONTEXT_AND_TOOLS: Program = Program {
    fixture: "openai_retrieval_context_and_tools",
    dynamic_context: Some(1),
    ..TOOLS
};

both_interpreters! {
    gemini_dynamic_context_one: GEMINI_DYNAMIC_CONTEXT_ONE,
    gemini_dynamic_context_two_streamed: GEMINI_DYNAMIC_CONTEXT_TWO_STREAMED,
    gemini_dynamic_context_over_sampled: GEMINI_DYNAMIC_CONTEXT_OVER_SAMPLED,
    gemini_dynamic_context_empty_index: GEMINI_DYNAMIC_CONTEXT_EMPTY_INDEX,
    gemini_retrieved_tools_one: GEMINI_RETRIEVED_TOOLS_ONE,
    gemini_retrieved_tools_with_static: GEMINI_RETRIEVED_TOOLS_WITH_STATIC,
    gemini_context_and_tools: GEMINI_CONTEXT_AND_TOOLS,
    openai_dynamic_context_one: OPENAI_DYNAMIC_CONTEXT_ONE,
    openai_dynamic_context_one_streamed: OPENAI_DYNAMIC_CONTEXT_ONE_STREAMED,
    openai_retrieved_tools_one: OPENAI_RETRIEVED_TOOLS_ONE,
    openai_retrieved_tools_one_streamed: OPENAI_RETRIEVED_TOOLS_ONE_STREAMED,
    openai_context_and_tools: OPENAI_CONTEXT_AND_TOOLS,
}

/// The required row names every retrieval index, and the retrieved
/// definition in each request equals the one the record holds: the
/// replayer serves the row's tools from the recorded requests.
#[test]
fn every_retrieval_index_is_in_the_required_row() {
    for (program, keys) in [
        (
            &GEMINI_DYNAMIC_CONTEXT_ONE,
            vec!["golden/retrieve:context#0"],
        ),
        (&GEMINI_RETRIEVED_TOOLS_ONE, vec!["golden/retrieve:tools#0"]),
        (
            &GEMINI_CONTEXT_AND_TOOLS,
            vec!["golden/retrieve:context#0", "golden/retrieve:tools#0"],
        ),
        (
            &OPENAI_CONTEXT_AND_TOOLS,
            vec!["golden/retrieve:context#0", "golden/retrieve:tools#0"],
        ),
    ] {
        let log = corpus::golden(program.fixture);
        for key in keys {
            assert_eq!(
                log.header
                    .required
                    .get(&rig_core::effect::HandlerKey::from(key)),
                Some(&rig_core::effect::EffectFamily::Retrieve),
                "{}: {key}",
                program.fixture
            );
        }
    }
}
