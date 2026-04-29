//! Gemini streaming coverage, including the migrated example path.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::gemini;
use rig_core::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig, ThinkingConfig, ThinkingLevel,
};
use rig_core::streaming::StreamingPrompt;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn streaming_smoke() {
    let thinking_config = GenerationConfig {
        thinking_config: Some(ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(ThinkingLevel::Medium),
            include_thoughts: Some(true),
        }),
        ..Default::default()
    };
    let additional_params = AdditionalParameters::default().with_config(thinking_config);

    let client = gemini::Client::from_env().expect("client should build");
    let agent = client
        .agent(gemini::completion::GEMINI_3_FLASH_PREVIEW)
        .preamble(STREAMING_PREAMBLE)
        .additional_params(
            serde_json::to_value(additional_params)
                .expect("Gemini thinking config should serialize"),
        )
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn example_streaming_prompt() {
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
        .expect("client should build")
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
