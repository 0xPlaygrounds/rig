//! Cohere v2 history survival: tool-call ids across three prompts.

use rig::completion::CompletionModel;

use super::super::support::{BoundCohere, with_cohere_cassette};
use crate::history_survival::driver::{Cell, Expect, Transport};

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: BoundCohere, cell: Cell) -> impl CompletionModel + 'static {
    client.completion(cell.model)
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
