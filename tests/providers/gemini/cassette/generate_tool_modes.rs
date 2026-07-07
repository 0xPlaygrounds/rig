//! Gemini `toolConfig.functionCallingConfig` mode regression tests.
//!
//! Existing `tool_choice` cassettes cover `Specific` (ANY + allowed names) and
//! `None`; this module locks the remaining `Required` mapping: bare
//! `{"mode": "ANY"}` with no allowedFunctionNames must force a function call.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::client::CompletionClient;
use rig::completion::CompletionModel;
use rig::message::{AssistantContent, ToolChoice};
use rig::providers::gemini;
use rig::tool::Tool;

use super::super::support::with_gemini_cassette;
use crate::support::{Adder, TOOLS_PREAMBLE};

#[tokio::test]
async fn required_maps_to_any_and_forces_function_call() {
    with_gemini_cassette(
        "generate_tool_modes/required_maps_to_any_and_forces_function_call",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request("Please greet me.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .temperature(0.0)
                .tool(Adder.definition().await)
                .tool_choice(ToolChoice::Required)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("required tool choice completion should succeed");

            let names: Vec<_> = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call.function.name.clone()),
                    _ => None,
                })
                .collect();
            assert!(
                !names.is_empty(),
                "toolConfig mode ANY must force a function call even for a chat prompt, \
                 got {:?}",
                response.choice
            );
            assert!(
                names.iter().all(|name| name == Adder::NAME),
                "only the provided tool can be called, saw {names:?}"
            );
        },
    )
    .await;
}
