//! Gemini streaming coverage, including the migrated example path.

use futures::StreamExt;
use rig::completion::FinishReason;
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig, ThinkingConfig, ThinkingLevel,
};
use rig::streaming::Item;
use rig::streaming::StreamEvent;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
    collect_stream_final_response_and_provider_final,
};
use rig::completion::CompletionRequest;

#[tokio::test]
async fn streaming_smoke() {
    let thinking_config = GenerationConfig {
        thinking_config: Some(ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(ThinkingLevel::Medium),
            include_thoughts: Some(true),
        }),
        ..Default::default()
    };
    let additional_params = AdditionalParameters::default().with_config(thinking_config);

    super::super::support::with_gemini_cassette("streaming/streaming_smoke", |client| async move {
        let agent =
            rig::AgentBuilder::new(client.completion(gemini::completion::GEMINI_3_FLASH_PREVIEW))
                .preamble(STREAMING_PREAMBLE)
                .additional_params(
                    serde_json::to_value(additional_params)
                        .expect("Gemini thinking config should serialize"),
                )
                .build();

        let mut stream = agent.prompt(STREAMING_PROMPT).stream();
        let (response, provider_final) =
            collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
        assert!(provider_final.usage.total_tokens.is_some_and(|n| n > 0));
    })
    .await;
}

#[tokio::test]
async fn example_streaming_prompt() {
    let generation_config = GenerationConfig {
        thinking_config: Some(ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(ThinkingLevel::Medium),
            include_thoughts: Some(true),
        }),
        ..Default::default()
    };
    let params = AdditionalParameters::default().with_config(generation_config);
    super::super::support::with_gemini_cassette(
        "streaming/example_streaming_prompt",
        |client| async move {
            let agent = rig::AgentBuilder::new(
                client.completion(gemini::completion::GEMINI_3_FLASH_PREVIEW),
            )
            .preamble("Be precise and concise.")
            .temperature(0.5)
            .additional_params(serde_json::to_value(params).expect("params should serialize"))
            .build();

            let mut stream = agent
                .prompt("When and where and what type is the next solar eclipse?")
                .stream();
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming prompt should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn final_metadata_exposes_finish_reason_and_model_version() {
    super::super::support::with_gemini_cassette(
        "streaming/final_metadata_exposes_finish_reason_and_model_version",
        |client| async move {
            let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
            let request =
                CompletionRequest::new("Reply with exactly: final metadata ok").temperature(0.0);
            let mut stream = model.stream(request).expect("stream should start");

            let mut text = String::new();
            while let Some(chunk) = stream.next().await {
                if let Item::Event(StreamEvent::Text { text: delta, .. }) =
                    chunk.expect("stream chunk should succeed")
                {
                    text.push_str(&delta);
                }
            }
            let final_response = stream
                .finish()
                .await
                .expect("stream should yield final metadata");

            assert_nonempty_response(&text);
            assert!(
                matches!(final_response.finish_reason(), Some(FinishReason::Stop)),
                "expected STOP finish reason, got {:?}",
                final_response.finish_reason()
            );
            assert_eq!(
                final_response.model.as_deref(),
                Some(gemini::completion::GEMINI_2_5_FLASH),
                "expected resolved Gemini model version to be surfaced"
            );
            assert!(
                final_response.usage.is_reported(),
                "expected final response to expose token usage"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn final_metadata_handles_terminal_finish_reason_chunk() {
    super::super::support::with_gemini_cassette(
        "streaming/final_metadata_handles_terminal_finish_reason_chunk",
        |client| async move {
            let model = client.completion(gemini::completion::GEMINI_2_5_FLASH);
            let request =
                CompletionRequest::new("Reply with exactly: contentless final metadata ok")
                    .temperature(0.0);
            let mut stream = model.stream(request).expect("stream should start");

            let mut text = String::new();
            while let Some(chunk) = stream.next().await {
                if let Item::Event(StreamEvent::Text { text: delta, .. }) =
                    chunk.expect("stream chunk should succeed")
                {
                    text.push_str(&delta);
                }
            }
            let final_response = stream
                .finish()
                .await
                .expect("stream should yield final metadata");

            assert_eq!(text.trim(), "contentless final metadata ok");
            assert!(
                matches!(final_response.finish_reason(), Some(FinishReason::Stop)),
                "expected STOP finish reason from contentless terminal chunk, got {:?}",
                final_response.finish_reason()
            );
            assert_eq!(
                final_response.model.as_deref(),
                Some(gemini::completion::GEMINI_2_5_FLASH),
                "expected modelVersion from terminal chunks to be retained"
            );
            let usage = final_response.usage;
            assert!(
                usage.input_tokens.is_some_and(|n| n > 0),
                "expected positive input token usage, got {usage:?}"
            );
            assert!(
                usage.output_tokens.is_some_and(|n| n > 0),
                "expected positive output token usage, got {usage:?}"
            );
            assert!(
                usage.total_tokens.unwrap_or(0)
                    >= usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0),
                "expected total token usage to include input and output tokens, got {usage:?}"
            );
        },
    )
    .await;
}
