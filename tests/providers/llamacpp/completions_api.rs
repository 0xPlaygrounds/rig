//! Migrated from `examples/openai_agent_completions_api.rs` against a local llama.cpp server.

use rig::completion::CompletionModel;
use rig::completion::NormalizeCompletionResponse;
use rig::completion::Prompt;
use rig::prelude::*;
use rig::telemetry::ProviderResponseExt;

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
    let model = client.completion_model(support::model_name());
    let request = model
        .completion_request(RAW_TEXT_RESPONSE_PROMPT)
        .preamble(RAW_TEXT_RESPONSE_PREAMBLE.to_string())
        .build();
    // One request, two views: `raw_completion` returns llama.cpp's own wire
    // response and the provider-local conversion produces exactly what
    // `CompletionModel::completion` would have returned for it.
    let raw = model
        .raw_completion(request)
        .await
        .expect("raw completions api request should succeed");
    let raw_text = raw
        .get_text_response()
        .expect("raw completions api response should contain assistant text");
    let response: rig::completion::CompletionResponse = raw
        .normalize("openai")
        .expect("raw completions api response should normalize");

    let normalized_text = assistant_text_response(&response.choice)
        .expect("normalized completions api response should contain assistant text");

    assert_nonempty_response(&normalized_text);
    assert_nonempty_response(&raw_text);
    assert_contains_all_case_insensitive(&raw_text, &["cedar", "maple"]);
    assert_eq!(raw_text.trim(), normalized_text.trim());
}
