//! Migrated from `examples/gemini_streaming.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig, ThinkingConfig, ThinkingLevel,
};
use rig::streaming::StreamingPrompt;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn thinking_config_streaming_prompt() {
    let generation_config = GenerationConfig {
        thinking_config: Some(ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(ThinkingLevel::Medium),
            include_thoughts: Some(true),
        }),
        ..Default::default()
    };
    let params = AdditionalParameters::default().with_config(generation_config);
    let agent = gemini::Client::from_env()
        .agent(gemini::completion::GEMINI_3_FLASH_PREVIEW)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .additional_params(serde_json::to_value(params).expect("params should serialize"))
        .build();

    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
