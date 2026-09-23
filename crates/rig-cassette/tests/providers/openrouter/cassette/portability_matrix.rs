//! OpenRouter, routing to an Anthropic model, continues histories other
//! wires produced: Anthropic-signed, OpenAI-encrypted and Gemini-signed
//! reasoning beside a tool exchange.
//!
//! Rig replays reasoning only to its issuer. Claude through OpenRouter
//! shares the Anthropic issuer, so the Anthropic row replays its signed
//! thinking; the other rows carry reasoning another issuer produced, which
//! Rig omits (Anthropic rejects OpenAI ciphertext replayed as
//! `redacted_thinking`), so those targets continue from the tool exchange
//! and text alone.

use rig::completion::CompletionModel;

use super::super::support::{BoundOpenRouter, with_openrouter_cassette};
use crate::history_survival::portability::{Cell, Source};

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: BoundOpenRouter, cell: Cell) -> impl CompletionModel + 'static {
    client.completion(cell.model)
}

const fn cell(source: Source) -> Cell {
    Cell {
        provider: "openrouter",
        model: "anthropic/claude-haiku-4.5",
        params,
        max_tokens: 512,
        source,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_openrouter_cassette, family: portability_case;
    #[tokio::test]
    from_anthropic: ("portability_matrix/from_anthropic", configured, cell(Source::Anthropic));
    #[tokio::test]
    from_openai_responses: ("portability_matrix/from_openai_responses", configured, cell(Source::OpenAiResponses));
    #[tokio::test]
    from_gemini: ("portability_matrix/from_gemini", configured, cell(Source::Gemini));
}
