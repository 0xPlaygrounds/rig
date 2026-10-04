//! Copilot's Responses route, which its Codex models take: system messages
//! stay in `input`, and every request carries the editor envelope.

use rig_core::providers::copilot::wire::CopilotWire;
use rig_core::providers::copilot::{self, Copilot};

use super::openai_responses::ResponsesHistory;

fn wire(model: &str) -> CopilotWire {
    Copilot::new("tid=test;exp=0").completion(model).wire
}

rig_history_conformance::history_conformance_suite! {
    wire: "copilot",
    fixture: ResponsesHistory {
        wire,
        model: copilot::GPT_5_3_CODEX,
        other_model: "gpt-5.2-codex",
        text_only_model: Some("gpt-5.3-codex-spark"),
    },
}
