//! ChatGPT/Codex Responses backend behavior regression tests.
//!
//! Locks down strict tool schemas, ChatGPT-specific request shaping, SSE
//! reconstruction, and system/default instruction behavior.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::completion::{CompletionModel, Message};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::chatgpt;
use rig::providers::openai::responses_api;
use rig::tool::Tool;
use serde::Deserialize;

use super::super::support::{with_chatgpt_cassette, with_chatgpt_cassette_default_instructions};
use crate::cassettes::recorded_interaction_bodies;
use crate::support::{Adder, TOOLS_PREAMBLE};

const CHATGPT_PROVIDER: &str = "chatgpt";

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
            let request = model
                .completion_request("Use the add tool to add 7 and 5.")
                .preamble(TOOLS_PREAMBLE.to_string())
                .tool(rig::tool::tool_definition(&Adder))
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

#[tokio::test]
async fn store_false_and_prompt_cache_fields_roundtrip() {
    let scenario = "codex_behaviors/store_false_and_prompt_cache_fields_roundtrip";
    with_chatgpt_cassette(
        "codex_behaviors/store_false_and_prompt_cache_fields_roundtrip",
        |client| async move {
            let model = client.completion(chatgpt::GPT_5_4);
            // `store` and `prompt_cache_key` are Responses-API wire fields
            // with no normalized home, so they are read off the backend's own
            // response type — deserialized from the one reply's `raw`, which
            // for a dialect that answers even a unary request with an event
            // stream is the envelope the decoder reassembled from the terminal
            // `response.completed` frame. That frame carries no output items,
            // and the cassette records a single interaction, so the
            // assistant-text check that used to ride along here lives only in
            // `codex_sessions`. The marker prompt is kept verbatim so the
            // request still matches the recorded cassette.
            let response = model
                .completion(
                    model
                        .completion_request("Reply with exactly this marker: CODEX-STORE-FALSE")
                        .preamble("Return only the requested marker.".to_string())
                        .build(),
                )
                .await
                .expect("basic ChatGPT/Codex completion should succeed");
            let raw = responses_api::CompletionResponse::deserialize(&response.raw)
                .expect("`raw` is the serialized responses_api::CompletionResponse");

            assert_eq!(
                raw.additional_parameters.store,
                Some(false),
                "ChatGPT provider must force store=false"
            );
            assert!(
                raw.additional_parameters
                    .prompt_cache_key
                    .as_deref()
                    .is_some_and(|value| !value.is_empty()),
                "ChatGPT backend should return a prompt cache key that cassettes scrub"
            );
            let usage = raw
                .usage
                .as_ref()
                .expect("ChatGPT/Codex completion should report usage");
            assert!(usage.input_tokens > 0);
            assert!(usage.output_tokens > 0);
        },
    )
    .await;

    // `store` is rig's own request field, so the recorded request bytes are
    // where that contract actually lives; the read off the reply above only
    // proves the backend echoed it back.
    let interactions = recorded_interaction_bodies(CHATGPT_PROVIDER, scenario);
    assert_eq!(
        interactions.len(),
        1,
        "{scenario}: the scenario must record exactly one interaction"
    );
    let request: serde_json::Value =
        serde_json::from_str(&interactions[0].0).expect("the recorded request body should be JSON");
    assert_eq!(
        request.get("store"),
        Some(&serde_json::Value::Bool(false)),
        "the request ChatGPT receives must carry store=false"
    );
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
                result.output.contains("FALCON-9"),
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
                result.output.contains("DEFAULT-CODEX-MARKER")
                    && result.output.contains("EXPLICIT-CODEX-MARKER"),
                "merged instructions should influence the answer, got {result:?}"
            );
        },
    )
    .await;
}
