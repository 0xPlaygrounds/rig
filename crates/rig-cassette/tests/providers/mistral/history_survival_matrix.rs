//! Mistral history survival: nine-character tool-call ids across three
//! prompts.
//!
//! Unrecorded: every attempt answered HTTP 429 "Rate limit exceeded" before
//! the first model call, and the key reports
//! `x-ratelimit-limit-req-minute: 0`, so the account has no request
//! allowance at all rather than a transient limit.

use super::support::with_mistral_cassette_result;
use crate::history_survival::driver::{Cell, Expect, Transport};
use rig::providers::openai::OpenAI;
use rig::wire::Wire as _;

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: OpenAI, cell: Cell) -> rig::Model<rig::providers::openai::wire::OpenAiWire> {
    client.completion(cell.model).on(rig::transport())
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "mistral",
        model: super::DEFAULT_MODEL,
        params,
        max_tokens: 2048,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_mistral_cassette_result, family: history_survival_case;
    #[tokio::test]
    #[ignore = "unrecorded: the key's request allowance is zero (x-ratelimit-limit-req-minute: 0); every attempt answered HTTP 429"]
    unary: ("history_survival_matrix/unary", configured_result, cell(Transport::Unary, Expect::TOOLS_ONLY));
    #[tokio::test]
    #[ignore = "unrecorded: the key's request allowance is zero (x-ratelimit-limit-req-minute: 0); every attempt answered HTTP 429"]
    streaming: ("history_survival_matrix/streaming", configured_result, cell(Transport::Streaming, Expect::TOOLS_ONLY));
}
