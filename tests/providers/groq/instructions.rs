//! Live instruction-routing checks for Groq.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, Prompt};
use rig::message::Message;
use rig::providers::groq;
use rig::streaming::StreamingPrompt;

use crate::support::{
    INSTRUCTIONS_CONFLICT_PROMPT, INSTRUCTIONS_EXPECTED_MARKER, INSTRUCTIONS_PREAMBLE,
    assert_contains_any_case_insensitive, collect_stream_final_response,
};

use super::STREAMING_REASONING_MODEL;

const EXPLICIT_SYSTEM_MARKER: &str = "maple-harbor-6217";
const COMBINED_SYSTEM_MARKER: &str = "spruce-copper-8841";
const STREAMING_SYSTEM_MARKER: &str = "willow-stream-5039";
const STREAMING_SYSTEM_PREAMBLE: &str =
    "Reply with exactly the text `willow-stream-5039` and no other text.";

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn preamble_is_honored() {
    let response = groq::Client::from_env()
        .expect("client should build")
        .agent(STREAMING_REASONING_MODEL)
        .preamble(INSTRUCTIONS_PREAMBLE)
        .build()
        .prompt(INSTRUCTIONS_CONFLICT_PROMPT)
        .await
        .expect("Groq completion should succeed");

    assert_contains_any_case_insensitive(&response, &[INSTRUCTIONS_EXPECTED_MARKER]);
}

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn explicit_chat_history_system_message_is_honored() {
    let model = groq::Client::from_env()
        .expect("client should build")
        .completion_model(STREAMING_REASONING_MODEL);

    let response = model
        .completion_request("Reply with the exact text required by the system message.")
        .message(Message::system(format!(
            "Reply with exactly the text `{EXPLICIT_SYSTEM_MARKER}` and no other text."
        )))
        .send()
        .await
        .expect("Groq completion should succeed");

    assert_contains_any_case_insensitive(
        &response
            .choice
            .iter()
            .filter_map(|content| match content {
                rig::message::AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<String>(),
        &[EXPLICIT_SYSTEM_MARKER],
    );
}

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn preamble_and_explicit_system_message_are_both_honored() {
    let model = groq::Client::from_env()
        .expect("client should build")
        .completion_model(STREAMING_REASONING_MODEL);

    let response = model
        .completion_request("Reply with the exact marker required by the system messages.")
        .preamble(format!(
            "This preamble must be delivered as a string system message. The final answer marker is `{COMBINED_SYSTEM_MARKER}`."
        ))
        .message(Message::system(format!(
            "Reply with exactly `{COMBINED_SYSTEM_MARKER}` and no other text."
        )))
        .send()
        .await
        .expect("Groq completion should succeed");

    let text = response
        .choice
        .iter()
        .filter_map(|content| match content {
            rig::message::AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_contains_any_case_insensitive(&text, &[COMBINED_SYSTEM_MARKER]);
}

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn streaming_preamble_system_message_is_honored() {
    let agent = groq::Client::from_env()
        .expect("client should build")
        .agent(STREAMING_REASONING_MODEL)
        .preamble(STREAMING_SYSTEM_PREAMBLE)
        .build();

    let mut stream = agent
        .stream_prompt("Reply with the exact text required by the instruction-routing test.")
        .await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("Groq streaming completion should succeed");

    assert_contains_any_case_insensitive(&response, &[STREAMING_SYSTEM_MARKER]);
}
