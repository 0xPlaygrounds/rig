//! Moonshot reasoning-history roundtrip smoke test.
use rig::message::{AssistantContent, Message, Reasoning};
use rig::providers::moonshot;
use rig::providers::openai::wire::{self as openai_wire, OpenAI};
use rig::wire::Wire as _;

use crate::support::{assert_contains_any_case_insensitive, assert_nonempty_response};
use rig::completion::CompletionRequestBuilder;

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
    let model = OpenAI::from_env_with(&openai_wire::MOONSHOT)
        .expect("MOONSHOT_API_KEY should be set")
        .completion(moonshot::KIMI_K3)
        .on(rig::transport());
    let assistant = Message::Assistant {
        id: None,
        content: vec![
            AssistantContent::Reasoning(Reasoning::new("Remember the chosen color.")),
            AssistantContent::text("Understood. I will remember teal."),
        ],
    };

    let response = model
        .call(
            CompletionRequestBuilder::new(
                "What color was I asked to remember? Reply with one word.",
            )
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
