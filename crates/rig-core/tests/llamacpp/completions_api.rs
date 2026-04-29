//! Migrated from `examples/openai_agent_completions_api.rs` against a local llama.cpp server.

use rig_core::client::CompletionClient;
use rig_core::completion::CompletionModel;
use rig_core::completion::Prompt;
use rig_core::telemetry::ProviderResponseExt;

use crate::support::{
    RAW_TEXT_RESPONSE_PREAMBLE, RAW_TEXT_RESPONSE_PROMPT, assert_contains_all_case_insensitive,
    assert_nonempty_response, assistant_text_response,
};

use super::support;

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn completions_api_agent_prompt() {
    let agent = support::client()
        .completion_model(support::model_name())
        .completions_api()
        .into_agent_builder()
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent
        .prompt("Hello world!")
        .await
        .expect("completions api prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn completions_api_raw_response_text_matches_normalized_choice_text() {
    let client = support::completions_client();
    let response = client
        .completion_model(support::model_name())
        .completion_request(RAW_TEXT_RESPONSE_PROMPT)
        .preamble(RAW_TEXT_RESPONSE_PREAMBLE.to_string())
        .send()
        .await
        .expect("raw completions api request should succeed");

    let normalized_text = assistant_text_response(&response.choice)
        .expect("normalized completions api response should contain assistant text");
    let raw_text = response
        .raw_response
        .get_text_response()
        .expect("raw completions api response should contain assistant text");

    assert_nonempty_response(&normalized_text);
    assert_nonempty_response(&raw_text);
    assert_contains_all_case_insensitive(&raw_text, &["cedar", "maple"]);
    assert_eq!(raw_text.trim(), normalized_text.trim());
}
