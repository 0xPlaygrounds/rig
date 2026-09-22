//! Ollama history survival: `thinking`, daemon-issued tool-call ids and
//! name-correlated results across three prompts against a local daemon.

use rig::completion::CompletionModel;

use super::super::support::{BoundOllama, with_ollama_cassette};
use crate::history_survival::driver::{Cell, Expect, Transport};

fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({ "think": true }))
}

fn model(client: BoundOllama, cell: Cell) -> impl CompletionModel + 'static {
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
