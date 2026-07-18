//! Gemini structured output smoke test.

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::prelude::AgentClientExt;
use rig::providers::gemini;
use rig_agent::test_utils::decode_structured_output;
use rig_bevy::{LocalRuntime, TenantId};

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

#[tokio::test]
async fn bevy_local_native_structured_output() {
    with_gemini_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let model = client.completion_model("gemini-3-flash-preview");
            let request = model
                .completion_request(STRUCTURED_OUTPUT_PROMPT)
                .output_schema(schemars::schema_for!(SmokeStructuredOutput))
                .build();
            let mut runtime = LocalRuntime::new(model, TenantId::new());
            let result = runtime
                .run_structured::<SmokeStructuredOutput>(
                    request,
                    rig_bevy::OutputMode::Native,
                    true,
                    false,
                )
                .await
                .expect("Bevy structured run");
            assert_smoke_structured_output(&result.output);
            let _: gemini::completion::gemini_api_types::GenerateContentResponse =
                result.raw_response;
        },
    )
    .await;
}
