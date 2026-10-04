//! Cassette-backed Cohere non-streaming completion coverage, on the Chat
//! Completions wire of Cohere's OpenAI Compatibility API.

use rig::completion::{AssistantContent, FinishReason, Message};

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::support::{
    BASIC_PREAMBLE, BASIC_PROMPT, assert_contains_any_case_insensitive, assert_nonempty_response,
};
use rig::completion::CompletionRequest;

#[tokio::test]
async fn completion_smoke() {
    with_cohere_cassette("agent/completion_smoke", |client| async move {
        let agent = rig::AgentBuilder::new(client.completion(CASSETTE_MODEL))
            .preamble(BASIC_PREAMBLE)
            .temperature(0.2)
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");

        assert_nonempty_response(&response.output());
    })
    .await;
}

#[tokio::test]
async fn usage_is_reported_from_token_counts() {
    with_cohere_cassette(
        "agent/usage_is_reported_from_token_counts",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let request = CompletionRequest::new(BASIC_PROMPT).preamble(BASIC_PREAMBLE.to_string());

            // The normalized usage and the reply's own `usage` come out of the
            // cassette's one recorded interaction.
            let response = model
                .call(request)
                .await
                .expect("completion should succeed");
            let count = |pointer: &str| {
                response
                    .raw
                    .pointer(pointer)
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or_else(|| panic!("Cohere should report `{pointer}`"))
            };
            let input = count("/usage/prompt_tokens");
            let output = count("/usage/completion_tokens");

            assert_eq!(response.usage.input_tokens, Some(input));
            assert_eq!(response.usage.output_tokens, Some(output));
            assert_eq!(
                response.usage.total_tokens,
                Some(count("/usage/total_tokens"))
            );
            assert_eq!(
                response.usage.cached_input_tokens,
                Some(count("/usage/prompt_tokens_details/cached_tokens"))
            );
            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        },
    )
    .await;
}

#[tokio::test]
async fn max_tokens_sets_max_tokens_finish_reason() {
    with_cohere_cassette(
        "agent/max_tokens_sets_max_tokens_finish_reason",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let request =
                CompletionRequest::new("Write a detailed fifty-word description of the ocean.")
                    .max_tokens(4);

            let response = model
                .call(request)
                .await
                .expect("capped completion should succeed");
            assert_eq!(response.raw["choices"][0]["finish_reason"], "length");
            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
        },
    )
    .await;
}

#[tokio::test]
async fn multiturn_history_is_accepted() {
    with_cohere_cassette("agent/multiturn_history_is_accepted", |client| async move {
        let model = client.completion(CASSETTE_MODEL);
        let request = CompletionRequest::new("What code word did I ask you to remember?")
            .message(Message::user(
                "Remember the code word cobalt-orchid for my next question.",
            ))
            .message(Message::assistant(
                "Understood. I will remember the code word cobalt-orchid.",
            ))
            .max_tokens(32);

        let response = model
            .call(request)
            .await
            .expect("multi-turn history should be accepted");
        let text = response
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<String>();

        assert_contains_any_case_insensitive(&text, &["cobalt-orchid", "cobalt orchid"]);
    })
    .await;
}

#[tokio::test]
async fn stop_sequences_are_forwarded() {
    with_cohere_cassette("agent/stop_sequences_are_forwarded", |client| async move {
        let model = client.completion(CASSETTE_MODEL);
        let request = CompletionRequest::new("Output exactly this sequence: alpha<END>omega")
            .temperature(0.0)
            .max_tokens(32)
            .additional_params(serde_json::json!({
                "seed": 7,
                "stop": ["<END>"]
            }));

        let response = model
            .call(request)
            .await
            .expect("stop sequence request should succeed");
        let text = response
            .choice
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<String>();

        assert!(
            !text.contains("omega"),
            "the stop sequence ends the text: {text}"
        );
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    })
    .await;
}

#[tokio::test]
async fn sampling_parameters_are_forwarded() {
    with_cohere_cassette(
        "agent/sampling_parameters_are_forwarded",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let request = CompletionRequest::new("Reply with one short sentence about rain.")
                .temperature(0.2)
                .max_tokens(24)
                .additional_params(serde_json::json!({
                    "seed": 11,
                    "top_p": 0.8,
                    // Cohere takes one penalty at a time.
                    "frequency_penalty": 0.1
                }));

            let response = model
                .call(request)
                .await
                .expect("documented sampling parameters should be accepted");
            let text = response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<String>();

            assert_nonempty_response(&text);
        },
    )
    .await;
}
