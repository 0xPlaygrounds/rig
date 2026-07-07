//! ChatGPT/Codex Responses backend behavior regression tests.
//!
//! Locks down strict tool schemas, ChatGPT-specific request shaping, SSE
//! reconstruction, and system/default instruction behavior.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::client::CompletionClient;
use rig::completion::{Chat, CompletionModel, Message};
use rig::message::AssistantContent;
use rig::providers::chatgpt;
use rig::tool::Tool;

use super::super::support::{with_chatgpt_cassette, with_chatgpt_cassette_default_instructions};
use crate::support::{Adder, TOOLS_PREAMBLE, assistant_text_response};

#[tokio::test]
async fn strict_tools_opt_in_roundtrip() {
    with_chatgpt_cassette(
        "codex_behaviors/strict_tools_opt_in_roundtrip",
        |client| async move {
            // The recorded request body locks the strict-tools contract:
            // `strict: true` plus the sanitized schema (additionalProperties
            // false, all properties required) must be accepted by the backend.
            let model = client
                .completion_model(chatgpt::GPT_5_4)
                .with_strict_tools();
            let request = model
                .completion_request("Use the add tool to add 7 and 5.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .tool(rig_core::tool::tool_definition(&Adder))
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
async fn store_false_and_prompt_cache_fields_roundtrip() {
    with_chatgpt_cassette(
        "codex_behaviors/store_false_and_prompt_cache_fields_roundtrip",
        |client| async move {
            let model = client.completion_model(chatgpt::GPT_5_4);
            let response = model
                .completion(
                    model
                        .completion_request("Reply with exactly this marker: CODEX-STORE-FALSE")
                        .preamble("Return only the requested marker.".to_string())
                        .build(),
                )
                .await
                .expect("basic ChatGPT/Codex completion should succeed");

            let text = assistant_text_response(&response.choice).expect("response text");
            assert!(
                text.contains("CODEX-STORE-FALSE"),
                "final answer should preserve the requested marker, got {text:?}"
            );
            assert_eq!(
                response.raw_response.additional_parameters.store,
                Some(false),
                "ChatGPT provider must force store=false"
            );
            assert!(
                response
                    .raw_response
                    .additional_parameters
                    .prompt_cache_key
                    .as_deref()
                    .is_some_and(|value| !value.is_empty()),
                "ChatGPT backend should return a prompt cache key that cassettes scrub"
            );
            assert!(response.usage.input_tokens > 0);
            assert!(response.usage.output_tokens > 0);
        },
    )
    .await;
}

#[tokio::test]
async fn explicit_preamble_and_mid_conversation_system_messages_are_instructions() {
    with_chatgpt_cassette(
        "codex_behaviors/explicit_preamble_and_mid_conversation_system_messages_are_instructions",
        |client| async move {
            // ChatGPT rejects `system` items in `input`; the recorded request
            // body locks that the provider lifts both the preamble and later
            // system messages into the top-level `instructions` field.
            let agent = client
                .agent(chatgpt::GPT_5_4)
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

#[tokio::test]
async fn default_instructions_merge_with_explicit_preamble() {
    with_chatgpt_cassette_default_instructions(
        "codex_behaviors/default_instructions_merge_with_explicit_preamble",
        "Default instruction marker: always include DEFAULT-CODEX-MARKER when asked for the default marker.",
        |client| async move {
            let agent = client
                .agent(chatgpt::GPT_5_4)
                .preamble("Explicit instruction marker: also include EXPLICIT-CODEX-MARKER.")
                .build();
            let mut history = Vec::<Message>::new();

            let result = agent
                .chat(
                    "List the default marker and the explicit marker, and nothing else.",
                    &mut history,
                )
                .await
                .expect("default and explicit instructions should both reach the backend");

            assert!(
                result.contains("DEFAULT-CODEX-MARKER")
                    && result.contains("EXPLICIT-CODEX-MARKER"),
                "merged instructions should influence the answer, got {result:?}"
            );
        },
    )
    .await;
}
