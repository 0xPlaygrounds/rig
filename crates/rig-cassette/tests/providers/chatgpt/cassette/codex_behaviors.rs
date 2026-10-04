//! ChatGPT/Codex Responses backend behavior regression tests.
//!
//! Locks down strict tool schemas, ChatGPT-specific request shaping, SSE
//! reconstruction, and system/default instruction behavior.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::message::AssistantContent;
use rig::providers::chatgpt;
use rig::tool::Tool;
use rig_test_support::cassette_models::MapWire;

use super::super::support::with_chatgpt_cassette;
use crate::support::{Adder, TOOLS_PREAMBLE};
use rig::completion::CompletionRequest;

#[tokio::test]
async fn strict_tools_opt_in_roundtrip() {
    with_chatgpt_cassette(
        "codex_behaviors/strict_tools_opt_in_roundtrip",
        |client| async move {
            // The recorded request body locks the strict-tools contract:
            // `strict: true` plus the sanitized schema (additionalProperties
            // false, all properties required) must be accepted by the backend.
            let model = client
                .completion(chatgpt::GPT_5_4)
                .map_wire(|wire| wire.with_strict_tools());
            let request = CompletionRequest::new("Use the add tool to add 7 and 5.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&Adder));

            let response = model
                .call(request)
                .await
                .expect("strict-tools completion should succeed");

            let tool_call = response
                .choice
                .iter()
                .find_map(|content| match content {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
                    _ => None,
                })
                .expect("strict tool call should be produced");
            assert_eq!(tool_call.function.name, Adder::NAME);
            assert_eq!(
                tool_call
                    .function
                    .arguments
                    .get("x")
                    .and_then(serde_json::Value::as_f64),
                Some(7.0),
                "strict-mode arguments should carry both required fields: {:?}",
                tool_call.function.arguments
            );
            assert_eq!(
                tool_call
                    .function
                    .arguments
                    .get("y")
                    .and_then(serde_json::Value::as_f64),
                Some(5.0),
                "strict-mode arguments should carry both required fields: {:?}",
                tool_call.function.arguments
            );
        },
    )
    .await;
}
