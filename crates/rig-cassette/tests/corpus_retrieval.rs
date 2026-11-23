//! Matrix A: retrieval effects.
//!
//! `dynamic_context` dispatches a `TopN` retrieval from a completion-call
//! hook before every model call and patches the retrieved documents into the
//! request. `retrieved_tools` dispatches a `TopNIds` retrieval at the request
//! boundary and advertises the named tools. Both go over the bus under the
//! agent's keys (`<owner>/retrieve:context#0`, `<owner>/retrieve:tools#0`),
//! both are in the required row, and on replay a replayer answers them from
//! the record through `dynamic_context_handler` and `retrieved_tools_handler`:
//! no index, no embedding model, no provider.
//!
//! The embeddings themselves are not effects, since an index embeds its
//! query inside the retrieval handler. So an agent program records no `Embed`
//! or `Rerank` effect, and a third embedding wire would add a cassette but no
//! new record shape.

use crate::corpus;

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
