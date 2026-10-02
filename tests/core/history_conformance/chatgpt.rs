//! The ChatGPT (Codex) dialect of the Responses wire: every system message
//! goes to `instructions`, and the gateway always answers with a stream.

use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;

use super::openai_responses::ResponsesHistory;

fn wire(model: &str) -> Responses {
    Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, "test-token"),
        model,
    )
}

rig_core::history_conformance_suite! {
    wire: "chatgpt",
    fixture: ResponsesHistory {
        wire,
        model: chatgpt::GPT_5_4,
        other_model: chatgpt::GPT_5_3_CODEX,
        text_only_model: Some(chatgpt::GPT_5_3_CODEX_SPARK),
    },
}
