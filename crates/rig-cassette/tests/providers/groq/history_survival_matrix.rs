//! Groq history survival: tool-call ids across three prompts on an
//! OpenAI-compatible wire, with parsed reasoning delivered but not replayed.
//!
//! Both cells are the live regression for the `reasoning_content` replay
//! that Groq rejected with HTTP 400 before the fix. After it, the parsed
//! reasoning reaches the normalized history and is dropped from the next
//! request; whether Groq would accept it back under `reasoning` is
//! unverified.

use rig::completion::CompletionModel;

use super::support::{BoundGroq, with_groq_cassette_result};
use crate::history_survival::driver::{Cell, Expect, Transport};

fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({ "reasoning_format": "parsed" }))
}

fn model(client: BoundGroq, cell: Cell) -> impl CompletionModel + 'static {
    client.completion(cell.model)
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "groq",
        model: "openai/gpt-oss-20b",
        params,
        max_tokens: 4096,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_groq_cassette_result, family: history_survival_case;
    #[tokio::test]
    unary: ("history_survival_matrix/unary", configured_result, cell(Transport::Unary, Expect::REASONING));
    #[tokio::test]
    streaming: ("history_survival_matrix/streaming", configured_result, cell(Transport::Streaming, Expect::REASONING));
}
