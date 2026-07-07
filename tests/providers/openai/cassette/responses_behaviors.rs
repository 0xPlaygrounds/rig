//! OpenAI Responses API behavior regression tests.
//!
//! Locks down strict-tool opt-in, incomplete-response surfacing, and
//! system-instruction placement as input items.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::client::CompletionClient;
use rig::completion::{Chat, CompletionModel, Message};
use rig::message::AssistantContent;
use rig::providers::openai;
use rig::providers::openai::responses_api::ResponseStatus;
use rig::tool::Tool;

use super::super::support::with_openai_cassette;
use crate::support::{Adder, TOOLS_PREAMBLE};

#[tokio::test]
async fn strict_tools_opt_in_roundtrip() {
    with_openai_cassette(
        "responses_behaviors/strict_tools_opt_in_roundtrip",
        |client| async move {
            // The recorded request body locks the strict-tools contract:
            // `strict: true` plus the sanitized schema (additionalProperties
            // false, all properties required) must be accepted by the API.
            let model = client.completion_model(openai::GPT_4O).with_strict_tools();
            let request = model
                .completion_request("Use the add tool to add 7 and 5.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .tool(Adder.definition().await)
                .build();

            let response = model
                .completion(request)
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
                    .and_then(|value| value.as_f64()),
                Some(7.0),
                "strict-mode arguments should carry both required fields: {:?}",
                tool_call.function.arguments
            );
            assert_eq!(
                tool_call
                    .function
                    .arguments
                    .get("y")
                    .and_then(|value| value.as_f64()),
                Some(5.0),
                "strict-mode arguments should carry both required fields: {:?}",
                tool_call.function.arguments
            );
        },
    )
    .await;
}

#[tokio::test]
async fn incomplete_response_surfaces_partial_output() {
    with_openai_cassette(
        "responses_behaviors/incomplete_response_surfaces_partial_output",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let request = model
                .completion_request(
                    "Write a story of at least 150 words about a lighthouse keeper.",
                )
                .preamble("You are a storyteller.".to_string())
                .max_tokens(16)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("an incomplete response should still convert, not error");

            assert_eq!(
                response.raw_response.status,
                ResponseStatus::Incomplete,
                "hitting max_output_tokens should mark the response incomplete"
            );
            let reason = response
                .raw_response
                .incomplete_details
                .as_ref()
                .map(|details| details.reason.as_str());
            assert_eq!(
                reason,
                Some("max_output_tokens"),
                "incomplete_details should carry the truncation reason"
            );
            let text: String = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                !text.trim().is_empty(),
                "partial output text should still be surfaced"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn system_messages_as_input_items_mid_conversation() {
    with_openai_cassette(
        "responses_behaviors/system_messages_as_input_items_mid_conversation",
        |client| async move {
            // The recorded request body locks the placement contract: with
            // `with_system_instructions_as_messages`, the preamble and the
            // mid-conversation system message are sent as `system` input
            // items instead of the top-level `instructions` field.
            let agent = client
                .with_system_instructions_as_messages()
                .agent(openai::GPT_4O)
                .preamble("You are a concise assistant.")
                .build();
            let mut history = vec![
                Message::user("Hello!"),
                Message::assistant("Hi! How can I help you today?"),
                Message::system(
                    "The user's codename is FALCON-9. Always refer to the user by codename.",
                ),
            ];

            let result = agent
                .chat("What is my codename?", &mut history)
                .await
                .expect("chat with a mid-conversation system message should succeed");

            assert!(
                result.contains("FALCON-9"),
                "the mid-conversation system message must reach the model, got {result:?}"
            );
        },
    )
    .await;
}
