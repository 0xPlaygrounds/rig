//! Gemini history survival: thought signatures and function-call ids across
//! three prompts, plus an image tool result.

use rig::completion::CompletionModel;

use super::super::support::{BoundGemini, with_gemini_cassette};
use crate::history_survival::driver::{Cell, Expect, Transport};

fn params() -> Option<serde_json::Value> {
    Some(serde_json::json!({
        "generationConfig": {
            "thinkingConfig": { "thinkingBudget": 1024, "includeThoughts": true }
        }
    }))
}

fn model(client: BoundGemini, cell: Cell) -> impl CompletionModel + 'static {
    client.completion(cell.model)
}

const fn cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        provider: "gemini",
        model: "gemini-2.5-flash",
        params,
        max_tokens: 4096,
        transport,
        expect,
    }
}

/// `gemini-2.5-flash` answers an image inside a function response with
/// "Multimodal function responses are not supported for this model"; the
/// image cell uses the model the multimodal tool-result regression records.
const fn image_cell(transport: Transport, expect: Expect) -> Cell {
    Cell {
        model: rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW,
        ..cell(transport, expect)
    }
}

crate::matrix::case_matrix! {
    wrapper: with_gemini_cassette, family: history_survival_case;
    #[tokio::test]
    unary: ("history_survival_matrix/unary", configured, cell(Transport::Unary, Expect::SIGNED));
    #[tokio::test]
    streaming: ("history_survival_matrix/streaming", configured, cell(Transport::Streaming, Expect::SIGNED));
    #[tokio::test]
    unary_image_tool_result: ("history_survival_matrix/unary_image_tool_result", configured, image_cell(Transport::Unary, Expect::SIGNATURES.with_image()));
}
