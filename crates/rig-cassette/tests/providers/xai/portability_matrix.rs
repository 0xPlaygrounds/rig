//! xAI continues histories other wires produced on the Responses dialect:
//! Anthropic-signed and Gemini-signed reasoning beside a tool exchange.

use rig::providers::xai;
use rig_test_support::endpoint::Endpoint;

use super::support::with_xai_cassette;
use crate::history_survival::portability::{Cell, Source};

fn params() -> Option<serde_json::Value> {
    None
}

fn model(
    client: Endpoint<rig::providers::openai::OpenAI>,
    cell: Cell,
) -> rig::Model<rig::providers::openai::wire::OpenAiWire, rig::http_client::BoxedHttpClient> {
    client.completion(cell.model)
}

const fn cell(source: Source) -> Cell {
    Cell {
        provider: "xai",
        model: xai::GROK_3_MINI,
        params,
        max_tokens: 2048,
        source,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_xai_cassette, family: portability_case;
    #[tokio::test]
    from_anthropic: ("portability_matrix/from_anthropic", configured, cell(Source::Anthropic));
    #[tokio::test]
    from_gemini: ("portability_matrix/from_gemini", configured, cell(Source::Gemini));
}
