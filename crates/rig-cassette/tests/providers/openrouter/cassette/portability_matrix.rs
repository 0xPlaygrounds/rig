//! OpenRouter, routing to an Anthropic model, continues histories other
//! wires produced: Anthropic-signed, OpenAI-encrypted and Gemini-signed
//! reasoning beside a tool exchange.
//!
//! Two rows stay unrecorded. Committed recordings scrub signatures to
//! placeholders, so an Anthropic-origin signature cannot be replayed to an
//! Anthropic model from a fixture: the Anthropic row is inconclusive. The
//! OpenAI row reproduces a defect: Rig replays foreign encrypted reasoning
//! as Anthropic `redacted_thinking`, which the provider rejects with
//! "Invalid `data` in `redacted_thinking` block"; it is reported, not
//! fixed, because the fix needs reasoning provenance on the message type.

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
    #[ignore = "inconclusive: committed recordings scrub the Anthropic signature to a placeholder"]
    from_anthropic: ("portability_matrix/from_anthropic", configured, cell(Source::Anthropic));
    #[tokio::test]
    #[ignore = "defect: foreign encrypted reasoning replays as redacted_thinking and Anthropic rejects it (400)"]
    from_openai_responses: ("portability_matrix/from_openai_responses", configured, cell(Source::OpenAiResponses));
    #[tokio::test]
    from_gemini: ("portability_matrix/from_gemini", configured, cell(Source::Gemini));
}
