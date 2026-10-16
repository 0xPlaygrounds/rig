//! Copilot's Responses route, which its Codex models take: system messages
//! stay in `input`, and every request carries the editor envelope.

use rig_core::providers::copilot;
use rig_core::providers::openai::wire::{OpenAIConfig, OpenAiWire};

use super::openai_responses::ResponsesHistory;

fn wire(model: &str) -> OpenAiWire {
    OpenAiWire::new(
        OpenAIConfig::with_key(&copilot::DIALECT, "tid=test;exp=0"),
        model,
    )
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
