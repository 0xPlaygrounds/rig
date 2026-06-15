//! Cassette coverage for mistral.rs through Rig's OpenAI Responses API client.

use rig::client::CompletionClient;
use rig::completion::{Chat, Prompt};

use crate::support::{assert_contains_all_case_insensitive, assert_nonempty_response};

use super::super::support::{SYSTEM_PROMPT, model_name, with_mistralrs_cassette};

#[tokio::test]
async fn responses_api_no_think_returns_text() {
    with_mistralrs_cassette(
        "responses_api/responses_api_no_think_returns_text",
        |client| async move {
            let agent = client
                .agent(model_name())
                .preamble(SYSTEM_PROMPT)
                .max_tokens(128)
                .build();

            let response = agent
                .prompt("/no_think Explain token usage reporting in one sentence.")
                .await
                .expect("Responses API /no_think prompt should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn responses_api_reasoning_plus_answer_completes() {
    with_mistralrs_cassette(
        "responses_api/responses_api_reasoning_plus_answer_completes",
        |client| async move {
            let agent = client
                .agent(model_name())
                .preamble(SYSTEM_PROMPT)
                .max_tokens(512)
                .build();

            let response = agent
                .prompt(
                    "Think briefly, then answer in one sentence why local OpenAI-compatible servers should report token usage.",
                )
                .await
                .expect("Responses API reasoning plus answer prompt should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn responses_api_multi_turn_replays_history() {
    with_mistralrs_cassette(
        "responses_api/responses_api_multi_turn_replays_history",
        |client| async move {
            let agent = client
                .agent(model_name())
                .preamble(SYSTEM_PROMPT)
                .max_tokens(256)
                .build();
            let mut history = Vec::new();

            let _first = agent
                .chat(
                    "Think briefly, then answer in one sentence why usage accounting matters.",
                    &mut history,
                )
                .await
                .expect("first multi-turn Responses API chat should succeed");
            let first_history_len = history.len();
            let second = agent
                .chat("/no_think Reply with exactly: OK", &mut history)
                .await
                .expect("second multi-turn Responses API chat should succeed");

            assert!(
                first_history_len > 0 && history.len() > first_history_len,
                "multi-turn history should be updated; first_len={first_history_len}, final_len={}",
                history.len()
            );
            assert_contains_all_case_insensitive(&second, &["OK"]);
        },
    )
    .await;
}
