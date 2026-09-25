//! AWS Bedrock raw completion cassette coverage ported from OpenAI completions tests.

use rig::bedrock;
use rig::bedrock::types::converse_output::{ContentBlock, InternalConverseOutput};
use serde::Deserialize;

use super::super::support::with_bedrock_cassette;
use crate::support::{
    RAW_TEXT_RESPONSE_PREAMBLE, RAW_TEXT_RESPONSE_PROMPT, assert_contains_all_case_insensitive,
    assert_nonempty_response, assistant_text_response,
};
use rig::completion::CompletionRequestBuilder;

#[tokio::test]
async fn raw_response_text_matches_normalized_choice_text() {
    with_bedrock_cassette(
        "raw_completion/raw_response_text_matches_normalized_choice_text",
        |client| async move {
            let model = client.completion(bedrock::completion::AMAZON_NOVA_LITE);
            let request = CompletionRequestBuilder::new(RAW_TEXT_RESPONSE_PROMPT)
                .preamble(RAW_TEXT_RESPONSE_PREAMBLE.to_string())
                .temperature(0.0)
                .build();

            // `raw` is the unary Converse frame the response was normalized
            // from, so raw-vs-normalized parity is checked against one
            // recorded interaction.
            let response = model
                .call(request, None)
                .await
                .expect("Bedrock request should succeed");
            let raw = InternalConverseOutput::deserialize(&response.raw)
                .expect("raw should deserialize into the Converse frame");
            let raw_text = raw
                .output
                .as_ref()
                .and_then(|output| output.as_message().ok())
                .map(|message| {
                    message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text(text) => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .filter(|text| !text.is_empty())
                .expect("raw Bedrock response should contain assistant text");
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
