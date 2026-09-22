//! DeepSeek history survival: unsigned reasoning and tool-call ids across
//! three prompts. `reasoning_content` is never replayed, so the wire-level
//! rule tracks the call ids.

use rig::completion::CompletionModel;
use rig::providers::deepseek;

use super::support::{BoundDeepSeek, with_deepseek_cassette};
use crate::history_survival::driver::{Cell, Expect, Transport};

fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({ "thinking": { "type": "enabled" } }))
}

fn model(client: BoundDeepSeek, cell: Cell) -> impl CompletionModel + 'static {
    client.completion(cell.model)
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "deepseek",
        model: deepseek::DEEPSEEK_V4_FLASH,
        params,
        max_tokens: 4096,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_deepseek_cassette, family: history_survival_case;
    #[tokio::test]
    unary: ("history_survival_matrix/unary", configured, cell(Transport::Unary, Expect::REASONING));
    #[tokio::test]
    streaming: ("history_survival_matrix/streaming", configured, cell(Transport::Streaming, Expect::REASONING));
}
