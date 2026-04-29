//! Moonshot reasoning-history roundtrip smoke test.

use rig_core::OneOrMany;
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::CompletionModel;
use rig_core::message::{AssistantContent, Message, Reasoning};
use rig_core::providers::moonshot;

use crate::support::{assert_contains_any_case_insensitive, assert_nonempty_response};

fn response_text(choice: &rig_core::OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn assistant_reasoning_content_roundtrips_in_history() {
    let model = moonshot::Client::from_env()
        .expect("moonshot client should build")
        .completion_model(moonshot::KIMI_K2_5);
    let assistant = Message::Assistant {
        id: None,
        content: OneOrMany::many(vec![
            AssistantContent::Reasoning(Reasoning::new("Remember the chosen color.")),
            AssistantContent::text("Understood. I will remember teal."),
        ])
        .expect("assistant content"),
    };

    let response = model
        .completion(
            model
                .completion_request("What color was I asked to remember? Reply with one word.")
                .message(Message::user("Remember the secret color is teal."))
                .message(assistant)
                .build(),
        )
        .await
        .expect("reasoning-history completion should succeed");

    let text = response_text(&response.choice);
    assert_nonempty_response(&text);
    assert_contains_any_case_insensitive(&text, &["teal"]);
}
