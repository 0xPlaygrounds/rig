//! Perplexity multi-turn chat cassette coverage.

use rig::client::AgentClientExt;
use rig::completion::{Chat, Message};
use rig::providers::perplexity;

use crate::support::assert_contains_any_case_insensitive;

use super::super::support::with_perplexity_cassette;

#[tokio::test]
async fn chat_history_smoke() {
    with_perplexity_cassette("chat/chat_history_smoke", |client| async move {
        let agent = client
            .agent(perplexity::SONAR)
            .preamble("You are a memory test assistant. Keep answers short.")
            .max_tokens(48)
            .additional_params(serde_json::json!({"search_context_size": "low"}))
            .build();
        let mut history = Vec::<Message>::new();

        let first = agent
            .chat(
                "Remember this code word for the next turn: amber-rig.",
                &mut history,
            )
            .await
            .expect("first chat turn should succeed");
        assert_contains_any_case_insensitive(&first, &["amber", "remember"]);

        let second = agent
            .chat("What code word did I ask you to remember?", &mut history)
            .await
            .expect("second chat turn should succeed");
        assert_contains_any_case_insensitive(&second, &["amber-rig", "amber"]);
    })
    .await;
}
