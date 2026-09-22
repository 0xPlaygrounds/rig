//! Responses history survival: encrypted reasoning, reasoning item ids and
//! function-call ids across three prompts, plus an image tool result.

use rig::completion::CompletionModel;

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::history_survival::driver::{Cell, Expect, Transport};

fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({ "reasoning": { "effort": "low" } }))
}

fn model(client: OpenAiCassette, cell: Cell) -> impl CompletionModel + 'static {
    client.openai.responses(cell.model)
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "openai",
        model: "gpt-5-mini",
        params,
        max_tokens: 4096,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_openai_cassette, family: history_survival_case;
    #[tokio::test]
    responses_unary: ("history_survival_matrix/responses_unary", configured, cell(Transport::Unary, Expect::ENCRYPTED));
    #[tokio::test]
    responses_streaming: ("history_survival_matrix/responses_streaming", configured, cell(Transport::Streaming, Expect::ENCRYPTED));
    #[tokio::test]
    responses_unary_image_tool_result: ("history_survival_matrix/responses_unary_image_tool_result", configured, cell(Transport::Unary, Expect::ENCRYPTED.with_image()));
}
