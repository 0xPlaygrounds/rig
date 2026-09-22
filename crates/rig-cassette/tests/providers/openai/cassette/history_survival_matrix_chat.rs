//! Chat Completions history survival: tool-call ids across three prompts.
//!
//! Chat Completions replays no reasoning, so the opaque fields under test
//! are the provider-issued tool-call ids of parallel and retried calls.

use rig::completion::CompletionModel;

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::history_survival::driver::{Cell, Expect, Transport};

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: OpenAiCassette, cell: Cell) -> impl CompletionModel + 'static {
    client.openai.chat(cell.model)
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "openai",
        model: "gpt-4.1-mini",
        params,
        max_tokens: 2048,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_openai_cassette, family: history_survival_case;
    #[tokio::test]
    chat_unary: ("history_survival_matrix/chat_unary", configured, cell(Transport::Unary, Expect::TOOLS_ONLY));
    #[tokio::test]
    chat_streaming: ("history_survival_matrix/chat_streaming", configured, cell(Transport::Streaming, Expect::TOOLS_ONLY));
}
