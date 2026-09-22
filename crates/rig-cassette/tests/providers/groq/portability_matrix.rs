//! Groq continues histories other wires produced: Anthropic-signed and
//! Gemini-signed reasoning beside a tool exchange.

use rig::completion::CompletionModel;

use super::support::{BoundGroq, with_groq_cassette_result};
use crate::history_survival::portability::{Cell, Source};

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: BoundGroq, cell: Cell) -> impl CompletionModel + 'static {
    client.completion(cell.model)
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
