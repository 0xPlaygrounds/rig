//! Native tool stress: real provider, real tools, ordered application policies.
use super::super::{support::with_gemini_cassette, tools_support::CodewordLookup};
use super::ecs_stress_tools_runtime as runtime;
use rig::providers::gemini;
#[tokio::test]
async fn tool_error_guidance_drives_model_retry_blocking() {
    rig_test_support::goldens::world_golden_test(
        async {
            let lookup = CodewordLookup::default();
            let lookup_calls = lookup.counter.clone();
            with_gemini_cassette(
        "hook_stress_tools/tool_error_guidance_drives_model_retry_blocking",
        |client| async move {
            let mut ecs = runtime::agent(
                client.completion(gemini::completion::GEMINI_2_5_FLASH),
                "You look up team codewords with the lookup_codeword tool. If the tool returns \
                     an error with guidance, follow that guidance and try again, then report the \
                     codeword you obtain.",
                "stress-agent",
                0.0,
            );
            ecs.tool(lookup);
            let response =
                runtime::prompt(&mut ecs, "Look up the codeword for the red team.", 5).await;
            assert!(
                lookup_calls.count() >= 2,
                "the model should retry the lookup after the error guidance, saw {} call(s)",
                lookup_calls.count()
            );
            assert!(
                response.to_ascii_lowercase().contains("azure-falcon"),
                "the model should report the recovered codeword: {response:?}"
            );
        },
    )
    .await;
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "gemini_stress_tools_tool_error_guidance_drives_model_retry_blocking",
                log,
            )
        },
    )
    .await
}
