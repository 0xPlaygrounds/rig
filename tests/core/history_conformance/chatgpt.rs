//! The ChatGPT (Codex) dialect of the Responses wire: every system message
//! goes to `instructions`, and the gateway always answers with a stream.

use rig_core::providers::chatgpt;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;

use super::openai_responses::ResponsesHistory;

// These are the models the fixtures were recorded with.
const GPT_5_4: &str = "gpt-5.4";
const GPT_5_3_CODEX: &str = "gpt-5.3-codex";
const GPT_5_3_CODEX_SPARK: &str = "gpt-5.3-codex-spark";

fn wire(model: &str) -> Responses {
    Responses::new(
        OpenAIConfig::with_key(&chatgpt::DIALECT, "test-token"),
        model,
    )
}

rig_history_conformance::history_conformance_suite! {
    wire: "chatgpt",
    fixture: ResponsesHistory {
        wire,
        model: GPT_5_4,
        other_model: GPT_5_3_CODEX,
        text_only_model: Some(GPT_5_3_CODEX_SPARK),
    },
}
