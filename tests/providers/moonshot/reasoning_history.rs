//! Moonshot reasoning-history roundtrip smoke test.
use rig::message::{AssistantContent, Message, Reasoning};
use rig::providers::moonshot;

use crate::support::{assert_contains_any_case_insensitive, assert_nonempty_response};
use rig::completion::CompletionRequest;

fn response_text(choice: &[AssistantContent]) -> String {
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
    let model = moonshot::from_env()
        .expect("MOONSHOT_API_KEY should be set")
        .completion(moonshot::KIMI_K3);
    let assistant = Message::Assistant {
        id: None,
        content: vec![
            AssistantContent::Reasoning(Reasoning::new("Remember the chosen color.")),
            AssistantContent::text("Understood. I will remember teal."),
        ],
    };

    let response = model
        .call(
            CompletionRequest::new("What color was I asked to remember? Reply with one word.")
                .message(Message::user("Remember the secret color is teal."))
                .message(assistant),
        )
        .await
        .expect("reasoning-history completion should succeed");

    let text = response_text(&response.choice);
    assert_nonempty_response(&text);
    assert_contains_any_case_insensitive(&text, &["teal"]);
}
