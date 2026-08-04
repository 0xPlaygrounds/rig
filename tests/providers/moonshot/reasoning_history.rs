//! Moonshot reasoning-history roundtrip smoke test.

use rig::OneOrMany;
use rig::completion::CompletionRequest;
use rig::http_runtime::HttpRuntime;
use rig::message::{AssistantContent, Message, Reasoning};
use rig::providers::moonshot;

use crate::support::{assert_contains_any_case_insensitive, assert_nonempty_response};

fn response_text(choice: &rig::OneOrMany<AssistantContent>) -> String {
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
    let cfg = moonshot::functions::Config::from_env(moonshot::KIMI_K2_5)
        .expect("moonshot config should build");
    let rt = HttpRuntime::new();
    let assistant = Message::Assistant {
        id: None,
        content: OneOrMany::many(vec![
            AssistantContent::Reasoning(Reasoning::new("Remember the chosen color.")),
            AssistantContent::text("Understood. I will remember teal."),
        ])
        .expect("assistant content"),
    };

    let response = moonshot::functions::complete(
        &cfg,
        &rt,
        CompletionRequest::with_history(
            vec![
                Message::user("Remember the secret color is teal."),
                assistant,
            ],
            "What color was I asked to remember? Reply with one word.",
        ),
    )
    .await
    .expect("reasoning-history completion should succeed");

    let text = response_text(&response.choice);
    assert_nonempty_response(&text);
    assert_contains_any_case_insensitive(&text, &["teal"]);
}
