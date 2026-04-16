//! ChatGPT completion normalization smoke tests.

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::message::Message;
use rig::providers::chatgpt;

use crate::chatgpt::{live_builder, live_client, response_text};
use crate::support::{assert_contains_any_case_insensitive, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn default_instructions_fill_required_instructions() {
    let client = live_builder()
        .default_instructions("Always answer with the single word cedar.")
        .build()
        .expect("ChatGPT client should build");

    let response = client
        .agent(chatgpt::GPT_5_3_CHAT_LATEST)
        .build()
        .prompt("Reply with the exact word from the instructions.")
        .await
        .expect("default-instructions completion should succeed");

    assert_contains_any_case_insensitive(&response, &["cedar"]);
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn system_messages_are_lifted_into_instructions() {
    let model = live_client().completion_model(chatgpt::GPT_5_3_CHAT_LATEST);

    let response = model
        .completion(
            model
                .completion_request("Reply with the exact word from the system message.")
                .message(Message::system("Always answer with the single word maple."))
                .build(),
        )
        .await
        .expect("system-message completion should succeed");

    let text = response_text(&response.choice);
    assert_nonempty_response(&text);
    assert_contains_any_case_insensitive(&text, &["maple"]);
}
