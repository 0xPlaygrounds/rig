//! Gemini streaming tools smoke test covering the additional-params request path.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};
use rig::streaming::StreamingPrompt;

use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn streaming_tools_smoke() {
    let additional_params =
        AdditionalParameters::default().with_config(GenerationConfig::default());

    let client = gemini::Client::from_env();
    let agent = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .additional_params(
            serde_json::to_value(additional_params)
                .expect("Gemini additional params should serialize"),
        )
        .build();

    let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
