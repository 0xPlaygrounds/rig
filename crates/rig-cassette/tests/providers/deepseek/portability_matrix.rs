//! DeepSeek continues histories other wires produced: Anthropic-signed,
//! OpenAI-encrypted and Gemini-signed reasoning beside a tool exchange.

use rig::providers::deepseek;
use rig_test_support::cassette_models::OpenAiModels;

use super::support::with_deepseek_cassette;
use crate::history_survival::portability::{Cell, Source};

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: OpenAiModels, cell: Cell) -> rig::Model<rig::providers::openai::wire::OpenAiWire> {
    client.completion(cell.model)
}

const fn cell(source: Source) -> Cell {
    Cell {
        provider: "deepseek",
        model: deepseek::DEEPSEEK_V4_FLASH,
        params,
        max_tokens: 512,
        source,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_deepseek_cassette, family: portability_case;
    #[tokio::test]
    from_openai_responses: ("portability_matrix/from_openai_responses", configured, cell(Source::OpenAiResponses));
}
