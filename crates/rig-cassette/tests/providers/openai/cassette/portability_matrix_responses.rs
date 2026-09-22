//! Responses continues histories other wires produced: Anthropic-signed,
//! Gemini-signed and DeepSeek plain reasoning beside a tool exchange.

use rig::completion::CompletionModel;

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::history_survival::portability::{Cell, Source};

fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({ "reasoning": { "effort": "low" } }))
}

fn model(client: OpenAiCassette, cell: Cell) -> impl CompletionModel + 'static {
    client.openai.responses(cell.model)
}

const fn cell(source: Source) -> Cell {
    Cell {
        provider: "openai",
        model: "gpt-5-mini",
        params,
        max_tokens: 2048,
        source,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_openai_cassette, family: portability_case;
    #[tokio::test]
    responses_from_anthropic: ("portability_matrix/responses_from_anthropic", configured, cell(Source::Anthropic));
    #[tokio::test]
    responses_from_gemini: ("portability_matrix/responses_from_gemini", configured, cell(Source::Gemini));
    #[tokio::test]
    responses_from_deepseek: ("portability_matrix/responses_from_deepseek", configured, cell(Source::DeepSeek));
}
