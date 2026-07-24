//! Cassette coverage for mistral.rs through Rig's OpenAI Responses API client.

use rig::completion::{Chat, CompletionModel, Prompt};
use rig::message::AssistantContent;
use rig::prelude::*;

use crate::support::{assert_contains_all_case_insensitive, assert_nonempty_response};

use super::super::support::{SYSTEM_PROMPT, model_name, with_mistralrs_cassette};

#[tokio::test]
async fn responses_api_no_think_returns_text() {
    with_mistralrs_cassette(
        "responses_api/responses_api_no_think_returns_text",
        |client| async move {
            let agent = client
                .with_system_instructions_as_messages()
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
            let model = client
                .with_system_instructions_as_messages()
                .completion_model(model_name());
            let request = model
                .completion_request(
                    "Think briefly, then answer in one sentence why local OpenAI-compatible servers should report token usage.",
                )
                .preamble(SYSTEM_PROMPT.to_owned())
                .max_tokens(512)
                .build();
            let response = model
                .completion(request)
                .await
                .expect("Responses API reasoning plus answer prompt should succeed");
            let text = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<String>();

            assert_nonempty_response(&text);
            assert!(
                response
                    .raw_response
                    .provider_reasoning
                    .as_deref()
                    .is_some_and(|reasoning| !reasoning.trim().is_empty()),
                "string-shaped provider reasoning should remain available"
            );
            assert_eq!(response.raw_response.reasoning_metadata, None);
            assert_eq!(response.raw_response.reasoning_context, None);
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
                .with_system_instructions_as_messages()
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
