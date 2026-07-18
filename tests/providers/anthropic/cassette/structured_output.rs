//! Anthropic structured output smoke test.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig_agent::test_utils::decode_structured_output;

use super::super::support::with_anthropic_cassette;
use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_smoke_structured_output,
};

#[tokio::test]
async fn structured_output_smoke() {
    with_anthropic_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .output_schema::<SmokeStructuredOutput>()
                .build();

            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");
            let structured: SmokeStructuredOutput =
                decode_structured_output("anthropic_structured_output_smoke", &response)
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

    with_anthropic_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let runtime = BevyRuntime::default();
            let agent = runtime.spawn_agent(
                AgentSpec::new(client.completion_model(CLAUDE_SONNET_4_6))
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
            let structured: SmokeStructuredOutput = serde_json::from_str(&text)
                .expect("Anthropic structured output should deserialize");
            assert_smoke_structured_output(&structured);
        },
    )
    .await;
}

#[cfg(feature = "bevy")]
#[tokio::test]
async fn bevy_prompted_structured_output_uses_schema_instruction() {
    use rig::bevy::{AgentSpec, BevyRuntime, policy::OutputMode};
    use rig::completion::AssistantContent;

    with_anthropic_cassette(
        "structured_output/bevy_prompted_structured_output",
        |client| async move {
            let outcome = BevyRuntime::default()
                .spawn_agent(
                    AgentSpec::new(client.completion_model(CLAUDE_SONNET_4_6))
                        .output_schema::<SmokeStructuredOutput>()
                        .output_mode(OutputMode::Prompted),
                )
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("Bevy prompted structured output should succeed");
            let text = outcome
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<String>();
            let structured: SmokeStructuredOutput =
                serde_json::from_str(&text).expect("Anthropic prompted output should deserialize");
            assert_smoke_structured_output(&structured);
        },
    )
    .await;
}
