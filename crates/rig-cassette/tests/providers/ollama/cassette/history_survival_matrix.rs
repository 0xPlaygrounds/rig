//! Ollama history survival: `thinking`, daemon-issued tool-call ids and
//! name-correlated results across three prompts against a local daemon.

use super::super::support::with_ollama_cassette;
use crate::history_survival::driver::{Cell, Expect, Transport};
use rig_test_support::cassette_models::OllamaModels;

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: OllamaModels, cell: Cell) -> rig::Model<rig::providers::ollama::Chat> {
    client.completion(cell.model)
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "ollama",
        model: "qwen3:4b",
        params,
        max_tokens: 8192,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_ollama_cassette, family: history_survival_case;
    #[tokio::test]
    unary: ("history_survival_matrix/unary", configured, cell(Transport::Unary, Expect::REASONING));
    #[tokio::test]
    streaming: ("history_survival_matrix/streaming", configured, cell(Transport::Streaming, Expect::REASONING));
}
