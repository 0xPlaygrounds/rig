//! OpenRouter history survival: Anthropic-signed reasoning delivered through
//! `reasoning_details`, and tool-call ids, across three prompts.

use super::super::support::with_openrouter_cassette;
use crate::history_survival::driver::{Cell, Expect, Transport};
use rig::providers::openai::OpenAI;

fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({
        "reasoning": { "max_tokens": 1024 },
        "include_reasoning": true
    }))
}

fn model(
    client: OpenAI,
    cell: Cell,
) -> rig::Model<rig::providers::openai::wire::OpenAiWire, rig::http_client::BoxedHttpClient> {
    rig::model(client.completion(cell.model))
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "openrouter",
        model: "anthropic/claude-haiku-4.5",
        params,
        max_tokens: 4096,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_openrouter_cassette, family: history_survival_case;
    #[tokio::test]
    unary: ("history_survival_matrix/unary", configured, cell(Transport::Unary, Expect::SIGNED));
    #[tokio::test]
    streaming: ("history_survival_matrix/streaming", configured, cell(Transport::Streaming, Expect::SIGNED));
}
