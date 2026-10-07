//! Cassette-backed Cohere non-streaming completion coverage, on the Chat
//! Completions wire of Cohere's OpenAI Compatibility API.

use rig::completion::{AssistantContent, FinishReason};

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig::completion::{CompletionRequest, GenerationOptions, ProviderOptions};
use rig::providers::cohere::extension::{CohereExt, CohereOptions};

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
async fn stop_sequences_are_forwarded() {
    with_cohere_cassette("agent/stop_sequences_are_forwarded", |client| async move {
        let model = client.completion(CASSETTE_MODEL);
        let request = CompletionRequest::new("Output exactly this sequence: alpha<END>omega")
            .temperature(0.0)
            .max_tokens(32)
            .options(GenerationOptions::default().seed(7).stop(["<END>"]));

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
                .options(GenerationOptions::default().seed(11).top_p(0.8))
                .provider_options(
                    ProviderOptions::new()
                        // Cohere takes one penalty at a time.
                        .with::<CohereExt>(&CohereOptions::default().frequency_penalty(0.1))
                        .expect("Cohere options serialize"),
                );

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
