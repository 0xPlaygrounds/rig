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
        self.candidates
            .first()
            .and_then(|candidate| terminal_metadata_from_finish_reason(candidate.finish_reason))
    }
}

pub(crate) fn terminal_metadata_from_finish_reason(
    finish_reason: i32,
) -> Option<rig_core::completion::CompletionTerminalMetadata> {
    use rig_core::completion::{CompletionFinishReason, CompletionTerminalMetadata};

    if finish_reason == proto::candidate::FinishReason::Unspecified as i32 {
        return None;
    }
    let Ok(provider_reason) = proto::candidate::FinishReason::try_from(finish_reason) else {
        return Some(
            CompletionTerminalMetadata::new(CompletionFinishReason::Unknown)
                .with_raw_reason(finish_reason.to_string()),
        );
    };
    let reason = match provider_reason {
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
    Some(CompletionTerminalMetadata::new(reason).with_raw_reason(provider_reason.as_str_name()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::completion::CompletionFinishReason;

    #[test]
    fn normalizes_grpc_finish_reasons_and_default_absence() {
        let length =
            terminal_metadata_from_finish_reason(proto::candidate::FinishReason::MaxTokens as i32)
                .expect("length metadata");
        assert_eq!(length.reason(), CompletionFinishReason::Length);
        assert_eq!(length.raw_reason(), Some("MAX_TOKENS"));
        assert_eq!(
            terminal_metadata_from_finish_reason(
                proto::candidate::FinishReason::Unspecified as i32
            ),
            None
        );
    }
}
