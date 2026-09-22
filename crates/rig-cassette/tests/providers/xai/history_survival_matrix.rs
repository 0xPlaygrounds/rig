//! xAI history survival: id-only reasoning items and function-call ids on
//! the Responses dialect across three prompts. The reasoning item carries
//! no content, so the normalized history holds no reasoning block; the
//! wire-level rule still requires its id back.

use rig::completion::CompletionModel;
use rig::providers::xai;

use super::support::with_xai_cassette;
use crate::history_survival::driver::{Cell, Expect, Transport};

fn params() -> Option<serde_json::Value> {
    None
}

fn model(
    client: rig::driver::Bound<rig::providers::openai::OpenAI>,
    cell: Cell,
) -> impl CompletionModel + 'static {
    client.completion(cell.model)
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "xai",
        model: xai::GROK_3_MINI,
        params,
        max_tokens: 2048,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_xai_cassette, family: history_survival_case;
    #[tokio::test]
    unary: ("history_survival_matrix/unary", configured, cell(Transport::Unary, Expect::TOOLS_ONLY));
    #[tokio::test]
    streaming: ("history_survival_matrix/streaming", configured, cell(Transport::Streaming, Expect::TOOLS_ONLY));
}
