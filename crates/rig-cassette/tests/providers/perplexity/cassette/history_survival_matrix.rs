//! Perplexity history survival: not executable.
//!
//! The Sonar dialect declares `supports_tools: false`
//! (`rig-core/src/providers/openai/wire/dialects.rs`), and the survival
//! task is tool-driven, so the cells are registered as unsupported rather
//! than recorded.

use rig::completion::CompletionModel;
use rig::providers::perplexity;

use super::super::support::{BoundPerplexity, with_perplexity_cassette};
use crate::history_survival::driver::{Cell, Expect, Transport};

fn params() -> Option<serde_json::Value> {
    None
}

fn model(client: BoundPerplexity, cell: Cell) -> impl CompletionModel + 'static {
    client.completion(cell.model)
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "perplexity",
        model: perplexity::SONAR,
        params,
        max_tokens: 2048,
        transport,
        expect,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_perplexity_cassette, family: history_survival_case;
    #[tokio::test]
    #[ignore = "unsupported: the Perplexity dialect declares supports_tools: false"]
    unary: ("history_survival_matrix/unary", configured, cell(Transport::Unary, Expect::TOOLS_ONLY));
    #[tokio::test]
    #[ignore = "unsupported: the Perplexity dialect declares supports_tools: false"]
    streaming: ("history_survival_matrix/streaming", configured, cell(Transport::Streaming, Expect::TOOLS_ONLY));
}
