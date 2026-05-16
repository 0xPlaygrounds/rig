//! Gemini streaming coverage, including the migrated example path.

use futures::StreamExt;
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, GetTokenUsage};
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, FinishReason, GenerationConfig, ThinkingConfig, ThinkingLevel,
};
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

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
        let agent = client
            .agent(gemini::completion::GEMINI_3_FLASH_PREVIEW)
            .preamble(STREAMING_PREAMBLE)
            .additional_params(
                serde_json::to_value(additional_params)
                    .expect("Gemini thinking config should serialize"),
            )
            .build();

        let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
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
            let agent = client
                .agent(gemini::completion::GEMINI_3_FLASH_PREVIEW)
                .preamble("Be precise and concise.")
                .temperature(0.5)
                .additional_params(serde_json::to_value(params).expect("params should serialize"))
                .build();

            let mut stream = agent
                .stream_prompt("When and where and what type is the next solar eclipse?")
                .await;
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
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request("Reply with exactly: final metadata ok")
                .temperature(0.0)
                .build();
            let mut stream = model.stream(request).await.expect("stream should start");

            let mut text = String::new();
            let mut final_response = None;
            let mut final_response_count = 0;
            while let Some(chunk) = stream.next().await {
                match chunk.expect("stream chunk should succeed") {
                    StreamedAssistantContent::Text(delta) => text.push_str(&delta.text),
                    StreamedAssistantContent::Final(response) => {
                        final_response_count += 1;
                        final_response = Some(response);
                    }
                    _ => {}
                }
            }

            assert_nonempty_response(&text);
            assert_eq!(
                final_response_count, 1,
                "stream should yield exactly one final response"
            );
            let final_response = final_response.expect("stream should yield final metadata");
            assert!(
                matches!(final_response.finish_reason, Some(FinishReason::Stop)),
                "expected STOP finish reason, got {:?}",
                final_response.finish_reason
            );
            assert_eq!(
                final_response.model_version.as_deref(),
                Some(gemini::completion::GEMINI_2_5_FLASH),
                "expected resolved Gemini model version to be surfaced"
            );
            assert!(
                final_response.token_usage().is_some(),
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
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let request = model
                .completion_request("Reply with exactly: contentless final metadata ok")
                .temperature(0.0)
                .build();
            let mut stream = model.stream(request).await.expect("stream should start");

            let mut text = String::new();
            let mut final_response = None;
            let mut final_response_count = 0;
            while let Some(chunk) = stream.next().await {
                match chunk.expect("stream chunk should succeed") {
                    StreamedAssistantContent::Text(delta) => text.push_str(&delta.text),
                    StreamedAssistantContent::Final(response) => {
                        final_response_count += 1;
                        final_response = Some(response);
                    }
                    _ => {}
                }
            }

            assert_eq!(text.trim(), "contentless final metadata ok");
            assert_eq!(
                final_response_count, 1,
                "terminal finish chunk should yield exactly one final response"
            );
            let final_response = final_response.expect("stream should yield final metadata");
            assert!(
                matches!(final_response.finish_reason, Some(FinishReason::Stop)),
                "expected STOP finish reason from contentless terminal chunk, got {:?}",
                final_response.finish_reason
            );
            assert_eq!(
                final_response.model_version.as_deref(),
                Some(gemini::completion::GEMINI_2_5_FLASH),
                "expected modelVersion from terminal chunks to be retained"
            );
            let usage = final_response
                .token_usage()
                .expect("expected final response to expose token usage");
            assert!(
                usage.input_tokens > 0,
                "expected positive input token usage, got {usage:?}"
            );
            assert!(
                usage.output_tokens > 0,
                "expected positive output token usage, got {usage:?}"
            );
            assert!(
                usage.total_tokens >= usage.input_tokens + usage.output_tokens,
                "expected total token usage to include input and output tokens, got {usage:?}"
            );
        },
    )
    .await;
}
