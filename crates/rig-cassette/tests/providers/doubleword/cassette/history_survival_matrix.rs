//! Doubleword history survival: tool-call ids across three prompts.

use super::super::support::with_doubleword_cassette;
use crate::history_survival::driver::{Cell, Expect, Transport};
use rig::providers::openai::OpenAI;

fn params() -> Option<serde_json::Value> {
    None
}

fn model(
    client: OpenAI,
    cell: Cell,
) -> rig::Model<rig::providers::openai::wire::OpenAiWire, rig::http_client::BoxedHttpClient> {
    rig::model(client.completion(cell.model))
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "doubleword",
        model: rig::providers::doubleword::QWEN3_5_397B_A17B,
        params,
        max_tokens: 2048,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_doubleword_cassette, family: history_survival_case;
    #[tokio::test]
    unary: ("history_survival_matrix/unary", configured, cell(Transport::Unary, Expect::TOOLS_ONLY));
    #[tokio::test]
    streaming: ("history_survival_matrix/streaming", configured, cell(Transport::Streaming, Expect::TOOLS_ONLY));
}
