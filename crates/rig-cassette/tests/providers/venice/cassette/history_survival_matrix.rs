//! Venice history survival: tool-call ids across three prompts.

use super::super::support::with_venice_cassette;
use crate::history_survival::driver::{Cell, Expect, Transport};
use rig::providers::openai::OpenAI;

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: OpenAI, cell: Cell) -> rig::Model<rig::providers::openai::wire::OpenAiWire> {
    rig::model(client.completion(cell.model))
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "venice",
        model: rig::providers::venice::MISTRAL_SMALL_3_2_24B,
        params,
        max_tokens: 2048,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_venice_cassette, family: history_survival_case;
    #[tokio::test]
    unary: ("history_survival_matrix/unary", configured, cell(Transport::Unary, Expect::TOOLS_ONLY));
    #[tokio::test]
    streaming: ("history_survival_matrix/streaming", configured, cell(Transport::Streaming, Expect::TOOLS_ONLY));
}
