//! Migrated from `examples/openai_agent_completions_api.rs` against a local llama.cpp server.

use rig::completion::Prompt;
use rig::completion::{CompletionModel, CompletionRequest};
use rig::prelude::*;

use crate::support::{
    RAW_TEXT_RESPONSE_PREAMBLE, RAW_TEXT_RESPONSE_PROMPT, assert_contains_all_case_insensitive,
    assert_nonempty_response, assistant_text_response,
};

use super::support;

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn completions_api_agent_prompt() {
    let agent = support::completions_client()
        .agent(support::model_name())
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
    // The normalized `CompletionResponse` no longer exposes the raw provider
    // payload, and this test runs live-only (no cassette to re-read the wire
    // body from), so the previous raw-vs-normalized comparison now asserts the
    // exact expected content directly on the normalized text.
    let client = support::completions_client();
    let model = client.completion_model(support::model_name());
    let response = model
        .completion(CompletionRequest::with_history(
            Some(RAW_TEXT_RESPONSE_PREAMBLE),
            Vec::new(),
            RAW_TEXT_RESPONSE_PROMPT,
        ))
        .await
        .expect("raw completions api request should succeed");

    let normalized_text = assistant_text_response(&response.choice)
        .expect("normalized completions api response should contain assistant text");

    assert_nonempty_response(&normalized_text);
    assert_contains_all_case_insensitive(&normalized_text, &["cedar", "maple"]);
}
