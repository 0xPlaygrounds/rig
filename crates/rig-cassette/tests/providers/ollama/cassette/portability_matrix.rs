//! Ollama continues histories other wires produced: Anthropic-signed,
//! OpenAI-encrypted and Gemini-signed reasoning beside a tool exchange,
//! against a local daemon.

use rig::completion::CompletionModel;

use super::super::support::{BoundOllama, with_ollama_cassette};
use crate::history_survival::portability::{Cell, Source};

fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({ "think": false }))
}

fn model(client: BoundOllama, cell: Cell) -> impl CompletionModel + 'static {
    client.completion(cell.model)
}

const fn cell(source: Source) -> Cell {
    Cell {
        provider: "ollama",
        model: "qwen3:4b",
        params,
        max_tokens: 1024,
        source,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_ollama_cassette, family: portability_case;
    #[tokio::test]
    from_anthropic: ("portability_matrix/from_anthropic", configured, cell(Source::Anthropic));
    #[tokio::test]
    from_openai_responses: ("portability_matrix/from_openai_responses", configured, cell(Source::OpenAiResponses));
    #[tokio::test]
    from_gemini: ("portability_matrix/from_gemini", configured, cell(Source::Gemini));
}
