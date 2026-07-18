//! Gemini structured output smoke test.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig_agent::test_utils::decode_structured_output;

use super::super::support::with_gemini_cassette;
use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
};

#[tokio::test]
async fn structured_output_smoke() {
    with_gemini_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let agent = client
                .agent("gemini-3-flash-preview")
                .output_schema::<SmokeStructuredOutput>()
                .build();

            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");
            let structured: SmokeStructuredOutput =
                decode_structured_output("gemini_structured_output_smoke", &response)
                    .expect("structured output should deserialize");

            assert_smoke_structured_output(&structured);
        },
    )
    .await;
}

#[cfg(feature = "bevy")]
#[tokio::test]
async fn bevy_native_structured_output_uses_provider_schema() {
    use rig::bevy::{AgentSpec, BevyRuntime, policy::OutputMode};
    use rig::completion::AssistantContent;
    use rig::providers::gemini;

    with_gemini_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let runtime = BevyRuntime::default();
            let agent = runtime.spawn_agent(
                AgentSpec::new(client.completion_model(gemini::completion::GEMINI_3_FLASH_PREVIEW))
                    .output_schema::<SmokeStructuredOutput>()
                    .output_mode(OutputMode::Native),
            );

            let outcome = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("Bevy structured output should succeed");
            let text = outcome
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            let structured: SmokeStructuredOutput =
                serde_json::from_str(&text).expect("Gemini structured output should deserialize");
            assert_smoke_structured_output(&structured);
        },
    )
    .await;
}
