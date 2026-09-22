//! Chat Completions continues histories other wires produced: Anthropic-
//! signed, OpenAI-encrypted and Gemini-signed reasoning beside a tool
//! exchange.

use rig::completion::CompletionModel;

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::history_survival::portability::{Cell, Source};

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: OpenAiCassette, cell: Cell) -> impl CompletionModel + 'static {
    client.openai.chat(cell.model)
}

const fn cell(source: Source) -> Cell {
    Cell {
        provider: "openai",
        model: "gpt-4.1-mini",
        params,
        max_tokens: 512,
        source,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_openai_cassette, family: portability_case;
    #[tokio::test]
    chat_from_anthropic: ("portability_matrix/chat_from_anthropic", configured, cell(Source::Anthropic));
    #[tokio::test]
    chat_from_openai_responses: ("portability_matrix/chat_from_openai_responses", configured, cell(Source::OpenAiResponses));
    #[tokio::test]
    chat_from_gemini: ("portability_matrix/chat_from_gemini", configured, cell(Source::Gemini));
}
