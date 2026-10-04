//! Anthropic Messages API `tool_choice` regression tests.
//!
//! Locks down every `tool_choice` shape Rig sends to the Messages API:
//! `required` (mapped to Anthropic's `{"type": "any"}`), `none`, and a
//! specific named tool (`{"type": "tool", "name": ...}`).
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::completion::FinishReason;
use rig::message::{AssistantContent, ToolChoice};
use rig::providers::anthropic;
use rig::tool::Tool;

use super::super::support::with_anthropic_cassette;
use crate::support::{Adder, TOOLS_PREAMBLE};
use rig::completion::CompletionRequest;

fn tool_call_names(choice: &[AssistantContent]) -> Vec<String> {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.function.name.clone()),
            _ => None,
        })
        .map(String::from)
        .collect()
}

#[tokio::test]
async fn required_maps_to_any_and_forces_tool_use() {
    with_anthropic_cassette(
        "messages_tool_choice/required_maps_to_any_and_forces_tool_use",
        |client| async move {
            let model = client.completion(anthropic::completion::CLAUDE_SONNET_4_6);
            let request = CompletionRequest::new("Please greet me.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .max_tokens(1024)
                .tool(rig::tool::tool_definition(&Adder))
                .tool_choice(ToolChoice::Required);

            let response = model
                .call(request)
                .await
                .expect("required tool choice completion should succeed");

            let names = tool_call_names(&response.choice);
            assert!(
                !names.is_empty(),
                "tool_choice=required (Anthropic `any`) must force a tool_use even for a \
                 chat prompt, got {:?}",
                response.choice
            );
            assert!(
                names.iter().all(|name| name == Adder::NAME),
                "only the provided tool can be called, saw {names:?}"
            );
            assert_eq!(
                response.finish_reason(),
                Some(FinishReason::ToolCalls),
                "a forced tool_use turn should preserve the tool_use stop reason"
            );
        },
    )
    .await;
}
