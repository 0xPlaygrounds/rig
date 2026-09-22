//! Gemini continues histories other wires produced: Anthropic-signed,
//! OpenAI-encrypted and DeepSeek plain reasoning beside a tool exchange.

use rig::completion::CompletionModel;

use super::super::support::{BoundGemini, with_gemini_cassette};
use crate::history_survival::portability::{Cell, Source};

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: BoundGemini, cell: Cell) -> impl CompletionModel + 'static {
    client.completion(cell.model)
}

const fn cell(source: Source) -> Cell {
    Cell {
        provider: "gemini",
        model: "gemini-2.5-flash",
        params,
        max_tokens: 2048,
        source,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_gemini_cassette, family: portability_case;
    #[tokio::test]
    from_anthropic: ("portability_matrix/from_anthropic", configured, cell(Source::Anthropic));
    #[tokio::test]
    from_openai_responses: ("portability_matrix/from_openai_responses", configured, cell(Source::OpenAiResponses));
    #[tokio::test]
    from_deepseek: ("portability_matrix/from_deepseek", configured, cell(Source::DeepSeek));
}
