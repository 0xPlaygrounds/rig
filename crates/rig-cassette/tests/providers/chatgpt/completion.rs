//! ChatGPT completion normalization smoke tests.

use rig::streaming::Item;
use futures::StreamExt;
use rig::message::AssistantContent;
use rig::message::Message;
use rig::streaming::StreamEvent;

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::support::{
    assert_contains_any_case_insensitive, assert_nonempty_response, collect_stream_final_response,
};
use rig::completion::CompletionRequest;

fn aggregated_text(choice: &[AssistantContent]) -> String {
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
    // The instructions merged ahead of every preamble live on the provider
    // config, so they are set by mapping the bound config, not by rebuilding
    // the transport.
    let client = live_client()
        .await
        .map_config(|config| config.with_instructions("Always answer with the single word cedar."));

    let agent = rig::AgentBuilder::new(client.completion(LIVE_MODEL)).build();
    let mut stream = agent
        .prompt("Reply with the exact word from the instructions.")
        .stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("default-instructions streaming completion should succeed");

    assert_contains_any_case_insensitive(&response, &["cedar"]);
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn system_messages_are_lifted_into_instructions() {
    let model = live_client().await.completion(LIVE_MODEL);

    let request = CompletionRequest::new("Reply with the exact word from the system message.")
        .message(Message::system("Always answer with the single word maple."));
    let mut stream = model
        .stream(request)
        .expect("system-message stream should succeed");

    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let Item::Event(StreamEvent::Text { text: delta, .. }) = item.expect("system-message stream item should succeed")
        {
            text.push_str(&delta);
        }
    }
    if text.trim().is_empty() {
        text = aggregated_text(&stream.partial().choice);
    }
    assert_nonempty_response(&text);
    assert_contains_any_case_insensitive(&text, &["maple"]);
}
