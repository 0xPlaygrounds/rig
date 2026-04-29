//! ChatGPT completion normalization smoke tests.

use futures::StreamExt;
use rig_core::client::CompletionClient;
use rig_core::completion::CompletionModel;
use rig_core::message::AssistantContent;
use rig_core::message::Message;
use rig_core::streaming::{StreamedAssistantContent, StreamingPrompt};

use crate::chatgpt::{LIVE_MODEL, live_builder, live_client};
use crate::support::{
    assert_contains_any_case_insensitive, assert_nonempty_response, collect_stream_final_response,
};

fn aggregated_text(choice: &rig_core::OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn default_instructions_fill_required_instructions() {
    let client = live_builder()
        .default_instructions("Always answer with the single word cedar.")
        .build()
        .expect("ChatGPT client should build");

    let agent = client.agent(LIVE_MODEL).build();
    let mut stream = agent
        .stream_prompt("Reply with the exact word from the instructions.")
        .await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("default-instructions streaming completion should succeed");

    assert_contains_any_case_insensitive(&response, &["cedar"]);
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn system_messages_are_lifted_into_instructions() {
    let model = live_client().completion_model(LIVE_MODEL);

    let request = model
        .completion_request("Reply with the exact word from the system message.")
        .message(Message::system("Always answer with the single word maple."))
        .build();
    let mut stream = model
        .stream(request)
        .await
        .expect("system-message stream should succeed");

    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let StreamedAssistantContent::Text(delta) =
            item.expect("system-message stream item should succeed")
        {
            text.push_str(&delta.text);
        }
    }
    if text.trim().is_empty() {
        text = aggregated_text(&stream.choice);
    }
    assert_nonempty_response(&text);
    assert_contains_any_case_insensitive(&text, &["maple"]);
}
