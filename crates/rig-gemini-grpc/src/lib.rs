//! Google Gemini gRPC API client and Rig integration
//!
//! This module provides gRPC-based access to the Gemini API, offering better
//! performance and type safety compared to the REST API.
//!
//! # Example
//! ```no_run
//! use rig_core::client::CompletionClient;
//! use rig_gemini_grpc::{Client, completion::GEMINI_2_0_FLASH};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let client = Client::new("YOUR_API_KEY").await?;
//!
//! let completion_model = client.completion_model(GEMINI_2_0_FLASH);
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;
pub mod embedding;
pub mod streaming;

pub use client::Client;

// Include the generated proto code
mod proto {
    #![allow(clippy::all)]
    #![allow(warnings)]
    tonic::include_proto!("google.ai.generativelanguage.v1beta");
}

// Re-export commonly used proto types
pub use proto::{
    Content, EmbedContentRequest, EmbedContentResponse, GenerateContentRequest,
    GenerateContentResponse, Part, generative_service_client::GenerativeServiceClient,
};

// Implement GetCompletionMetadata for proto::GenerateContentResponse to support streaming
impl rig_core::completion::GetCompletionMetadata for proto::GenerateContentResponse {
    fn token_usage(&self) -> rig_core::completion::Usage {
        self.usage_metadata
            .as_ref()
            .map(|u| rig_core::completion::Usage {
                input_tokens: u.prompt_token_count as u64,
                output_tokens: u.candidates_token_count as u64,
                total_tokens: u.total_token_count as u64,
                cached_input_tokens: u.cached_content_token_count as u64,
                cache_creation_input_tokens: 0,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            })
            .unwrap_or_default()
    }

    fn terminal_metadata(&self) -> Option<rig_core::completion::CompletionTerminalMetadata> {
        let reason = self.candidates.first()?.finish_reason;
        gemini_terminal_metadata(reason)
    }
}

fn gemini_terminal_metadata(
    reason: i32,
) -> Option<rig_core::completion::CompletionTerminalMetadata> {
    use rig_core::completion::{CompletionFinishReason, CompletionTerminalMetadata};

    if reason == proto::candidate::FinishReason::Unspecified as i32 {
        return None;
    }
    let Ok(reason) = proto::candidate::FinishReason::try_from(reason) else {
        return Some(
            CompletionTerminalMetadata::new(CompletionFinishReason::Unknown)
                .with_raw_reason(reason.to_string()),
        );
    };
    let canonical = match reason {
        proto::candidate::FinishReason::Stop => CompletionFinishReason::Stop,
        proto::candidate::FinishReason::MaxTokens => CompletionFinishReason::Length,
        proto::candidate::FinishReason::Safety
        | proto::candidate::FinishReason::Recitation
        | proto::candidate::FinishReason::Language
        | proto::candidate::FinishReason::Blocklist
        | proto::candidate::FinishReason::ProhibitedContent
        | proto::candidate::FinishReason::Spii
        | proto::candidate::FinishReason::ImageSafety
        | proto::candidate::FinishReason::ImageProhibitedContent
        | proto::candidate::FinishReason::ImageRecitation => CompletionFinishReason::ContentFilter,
        _ => CompletionFinishReason::Unknown,
    };
    Some(CompletionTerminalMetadata::new(canonical).with_raw_reason(reason.as_str_name()))
}

fn gemini_protocol_finish_reason_error(
    response: &proto::GenerateContentResponse,
) -> Option<rig_core::completion::CompletionError> {
    use proto::candidate::FinishReason;

    let candidate = response.candidates.first()?;
    let reason = FinishReason::try_from(candidate.finish_reason).ok()?;
    if !matches!(
        reason,
        FinishReason::MalformedFunctionCall
            | FinishReason::UnexpectedToolCall
            | FinishReason::TooManyToolCalls
    ) {
        return None;
    }

    Some(match serde_json::to_string(response) {
        Ok(body) => rig_core::completion::CompletionError::from_provider_body(body),
        Err(error) => rig_core::completion::CompletionError::ProviderError(format!(
            "Gemini gRPC stopped with {}: {}; failed to preserve response: {error}",
            reason.as_str_name(),
            candidate.finish_message.as_deref().unwrap_or("no details")
        )),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::completion::CompletionFinishReason;

    #[test]
    fn grpc_finish_reason_normalization_preserves_proto_name() {
        for (reason, expected) in [
            (
                proto::candidate::FinishReason::Stop,
                CompletionFinishReason::Stop,
            ),
            (
                proto::candidate::FinishReason::MaxTokens,
                CompletionFinishReason::Length,
            ),
            (
                proto::candidate::FinishReason::Safety,
                CompletionFinishReason::ContentFilter,
            ),
            (
                proto::candidate::FinishReason::Other,
                CompletionFinishReason::Unknown,
            ),
        ] {
            let metadata = gemini_terminal_metadata(reason as i32)
                .expect("terminal proto reason should produce metadata");
            assert_eq!(metadata.reason(), expected);
            assert_eq!(metadata.raw_reason(), Some(reason.as_str_name()));
        }
        assert_eq!(
            gemini_terminal_metadata(proto::candidate::FinishReason::Unspecified as i32),
            None
        );
        let unknown = gemini_terminal_metadata(999)
            .expect("an unknown nonzero proto value should produce metadata");
        assert_eq!(unknown.reason(), CompletionFinishReason::Unknown);
        assert_eq!(unknown.raw_reason(), Some("999"));
    }

    #[test]
    fn grpc_protocol_finish_reasons_remain_provider_errors() {
        for reason in [
            proto::candidate::FinishReason::MalformedFunctionCall,
            proto::candidate::FinishReason::UnexpectedToolCall,
            proto::candidate::FinishReason::TooManyToolCalls,
        ] {
            let response = proto::GenerateContentResponse {
                candidates: vec![proto::Candidate {
                    finish_reason: reason as i32,
                    finish_message: Some("invalid tool protocol".to_string()),
                    ..Default::default()
                }],
                ..Default::default()
            };
            let error = gemini_protocol_finish_reason_error(&response)
                .expect("protocol finish reason should remain an error");
            let body = error
                .provider_response_json()
                .expect("provider response should be JSON")
                .expect("provider response should be preserved");
            assert_eq!(body["candidates"][0]["finish_reason"], reason as i32);
            assert_eq!(
                body["candidates"][0]["finish_message"],
                "invalid tool protocol"
            );
        }

        let response = proto::GenerateContentResponse {
            candidates: vec![proto::Candidate {
                finish_reason: proto::candidate::FinishReason::Stop as i32,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(gemini_protocol_finish_reason_error(&response).is_none());
    }
}
