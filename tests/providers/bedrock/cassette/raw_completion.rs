//! AWS Bedrock raw completion cassette coverage ported from OpenAI completions tests.

use rig::bedrock;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::telemetry::ProviderResponseExt;

use super::super::support::with_bedrock_cassette;
use crate::support::{
    RAW_TEXT_RESPONSE_PREAMBLE, RAW_TEXT_RESPONSE_PROMPT, assert_contains_all_case_insensitive,
    assert_nonempty_response, assistant_text_response,
};

#[tokio::test]
async fn raw_response_text_matches_normalized_choice_text() {
    with_bedrock_cassette(
        "raw_completion/raw_response_text_matches_normalized_choice_text",
        |client| async move {
            let model = client.completion_model(bedrock::completion::AMAZON_NOVA_LITE);
            let request = model
                .completion_request(RAW_TEXT_RESPONSE_PROMPT)
                .preamble(RAW_TEXT_RESPONSE_PREAMBLE.to_string())
                .temperature(0.0)
                .build();

            // `completion` is `raw_completion` followed by exactly this
            // conversion, so raw-vs-normalized parity is still checked against
            // one recorded interaction.
            let raw = model
                .raw_completion(request)
                .await
                .expect("raw Bedrock request should succeed");
            let raw_text = raw
                .get_text_response()
                .expect("raw Bedrock response should contain assistant text");
            let response: rig::completion::CompletionResponse = raw
                .try_into()
                .expect("raw Bedrock response should normalize");
            let normalized_text = assistant_text_response(&response.choice)
                .expect("normalized Bedrock response should contain assistant text");

            assert_nonempty_response(&normalized_text);
            assert_nonempty_response(&raw_text);
            assert_contains_all_case_insensitive(&raw_text, &["cedar", "maple"]);
            assert_eq!(raw_text.trim(), normalized_text.trim());
        },
    )
    .await;
}
