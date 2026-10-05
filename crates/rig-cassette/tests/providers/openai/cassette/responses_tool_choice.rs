//! OpenAI Responses API `tool_choice` regression tests.
//!
//! Locks down every `tool_choice` shape the Responses API accepts from Rig:
//! `required` (forced tool use), `none` (suppressed tool use), a single
//! specific function (`{"type": "function", "name": ...}`), and multiple
//! specific functions (`{"type": "allowed_tools", ...}`).
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::message::{AssistantContent, ToolChoice};
use rig::providers::openai;
use rig::tool::Tool;

use super::super::support::with_openai_cassette;
use crate::support::{Adder, AlphaSignal, Subtract, TOOLS_PREAMBLE};
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
async fn specific_multiple_functions_use_allowed_tools() {
    with_openai_cassette(
        "responses_tool_choice/specific_multiple_functions_use_allowed_tools",
        |client| async move {
            let model = client.openai.completion(openai::GPT_4O);
            let request = CompletionRequest::new("What is 2 plus 3? Use exactly one tool.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&Adder))
                .tool(rig::tool::tool_definition(&Subtract))
                .tool(rig::tool::tool_definition(&AlphaSignal))
                .tool_choice(ToolChoice::Specific {
                    function_names: vec![
                        rig_core::tool::tool_name::<Adder>(),
                        rig_core::tool::tool_name::<Subtract>(),
                    ],
                });

            let response = model
                .call(request)
                .await
                .expect("allowed-tools tool choice completion should succeed");

            let names = tool_call_names(&response.choice);
            assert!(
                !names.is_empty(),
                "allowed_tools in required mode must force a tool call, got {:?}",
                response.choice
            );
            assert!(
                names
                    .iter()
                    .all(|name| name == Adder::NAME || name == Subtract::NAME),
                "only allowed tools may be called, saw {names:?}"
            );
            assert!(
                names.iter().any(|name| name == Adder::NAME),
                "an addition prompt restricted to add/subtract should call add, saw {names:?}"
            );
        },
    )
    .await;
}
