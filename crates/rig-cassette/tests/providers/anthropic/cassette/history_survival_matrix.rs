//! Anthropic history survival: thinking signatures and `tool_use` ids across
//! three prompts.
//!
//! Unrecorded: the workspace API limit was exhausted at recording time
//! (HTTP 400 "You have reached your specified workspace API usage limits",
//! access restored 2026-10-01). The corpus check still covers the delivered
//! signatures in the existing Anthropic recordings.

use rig::completion::CompletionModel;
use rig::providers::anthropic::completion::CLAUDE_HAIKU_4_5;

use super::super::support::with_anthropic_cassette;
use crate::history_survival::driver::{Cell, Expect, Transport};

fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } }))
}

fn model(
    client: rig::driver::Bound<rig::providers::anthropic::wire::Anthropic>,
    cell: Cell,
) -> impl CompletionModel + 'static {
    client.completion(cell.model)
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "anthropic",
        model: CLAUDE_HAIKU_4_5,
        params,
        max_tokens: 4096,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_anthropic_cassette, family: history_survival_case;
    #[tokio::test]
    #[ignore = "unrecorded: the Anthropic workspace API limit is exhausted until 2026-10-01"]
    unary: ("history_survival_matrix/unary", configured, cell(Transport::Unary, Expect::SIGNED));
    #[tokio::test]
    #[ignore = "unrecorded: the Anthropic workspace API limit is exhausted until 2026-10-01"]
    streaming: ("history_survival_matrix/streaming", configured, cell(Transport::Streaming, Expect::SIGNED));
}
