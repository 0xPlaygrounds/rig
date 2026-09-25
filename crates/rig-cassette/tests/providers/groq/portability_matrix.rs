//! Groq continues histories other wires produced: Anthropic-signed and
//! Gemini-signed reasoning beside a tool exchange.

use super::support::with_groq_cassette_result;
use crate::history_survival::portability::{Cell, Source};
use rig::providers::openai::OpenAI;
use rig::wire::Wire as _;

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: OpenAI, cell: Cell) -> rig::Model<rig::providers::openai::wire::OpenAiWire> {
    client.completion(cell.model).on(rig::transport())
}

const fn cell(source: Source) -> Cell {
    Cell {
        provider: "groq",
        model: "openai/gpt-oss-20b",
        params,
        max_tokens: 1024,
        source,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_groq_cassette_result, family: portability_case;
    #[tokio::test]
    from_anthropic: ("portability_matrix/from_anthropic", configured_result, cell(Source::Anthropic));
    #[tokio::test]
    from_gemini: ("portability_matrix/from_gemini", configured_result, cell(Source::Gemini));
}
