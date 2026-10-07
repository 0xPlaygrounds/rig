//! Responses history survival: encrypted reasoning, reasoning item ids and
//! function-call ids across three prompts, plus an image tool result.

use rig::completion::Effort;

use super::super::support::{OpenAiCassette, effort, stateless, with_openai_cassette};
use crate::history_survival::Options;
use crate::history_survival::driver::{Cell, Expect, Transport};

fn params() -> Option<serde_json::Value> {
    None
}

fn options() -> Options {
    Options::new(effort(Effort::Low), stateless())
}

fn model(
    client: OpenAiCassette,
    cell: Cell,
) -> rig::Model<
    rig::providers::openai::responses_api::wire::Responses,
    rig::http_client::DynHttpClient,
> {
    client.openai.responses(cell.model)
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "openai",
        model: "gpt-5-mini",
        params,
        options,
        max_tokens: 4096,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_openai_cassette, family: history_survival_case;
    #[tokio::test]
    responses_unary_image_tool_result: ("history_survival_matrix/responses_unary_image_tool_result", configured, cell(Transport::Unary, Expect::ENCRYPTED.with_image()));
}
