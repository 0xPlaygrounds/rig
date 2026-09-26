//! Cohere v2 history survival: tool-call ids across three prompts.

use super::super::support::with_cohere_cassette;
use crate::history_survival::driver::{Cell, Expect, Transport};
use rig::providers::cohere::wire::Cohere;

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: Cohere, cell: Cell) -> rig::Model<rig::providers::cohere::Chat> {
    rig::model(client.completion(cell.model))
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "cohere",
        model: rig::providers::cohere::COMMAND_A_03_2025,
        params,
        max_tokens: 2048,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_cohere_cassette, family: history_survival_case;
    #[tokio::test]
    unary: ("history_survival_matrix/unary", configured, cell(Transport::Unary, Expect::TOOLS_ONLY));
    #[tokio::test]
    streaming: ("history_survival_matrix/streaming", configured, cell(Transport::Streaming, Expect::TOOLS_ONLY));
}
