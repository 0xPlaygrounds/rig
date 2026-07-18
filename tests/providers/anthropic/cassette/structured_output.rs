//! Anthropic structured output smoke test.

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::prelude::AgentClientExt;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig_agent::test_utils::decode_structured_output;
use rig_bevy::{LocalRuntime, TenantId};

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

#[tokio::test]
async fn bevy_local_prompted_structured_output() {
    with_anthropic_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let request = model
                .completion_request(STRUCTURED_OUTPUT_PROMPT)
                .output_schema(schemars::schema_for!(SmokeStructuredOutput))
                .build();
            let mut runtime = LocalRuntime::new(model, TenantId::new());
            let result = runtime
                .run_structured::<SmokeStructuredOutput>(
                    request,
                    rig_bevy::OutputMode::Prompted,
                    false,
                    false,
                )
                .await
                .expect("Bevy structured run");
            assert_smoke_structured_output(&result.output);
        },
    )
    .await;
}
