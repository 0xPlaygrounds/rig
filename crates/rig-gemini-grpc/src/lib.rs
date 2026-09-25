//! Gemini completion and embedding wires over the gRPC API.
//!
//! ```no_run
//! use rig_core::wire::Wire as _;
//! use rig_gemini_grpc::{GeminiGrpc, completion::{GEMINI_2_0_FLASH, GenerateContent}};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let transport = GeminiGrpc::new("YOUR_API_KEY").await?;
//! let model = GenerateContent::new(GEMINI_2_0_FLASH).on(transport);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;
pub mod embedding;
pub mod streaming;

pub use client::GeminiGrpc;

/// Generated Gemini protobuf messages and service client.
///
/// ```
/// use rig_gemini_grpc::proto::GenerateContentResponse;
///
/// let response = GenerateContentResponse::default();
/// assert!(response.candidates.is_empty());
/// ```
pub mod proto {
    #![allow(clippy::all)]
    #![allow(warnings)]
    tonic::include_proto!("google.ai.generativelanguage.v1beta");
}

pub use proto::{
    Content, EmbedContentRequest, EmbedContentResponse, GenerateContentRequest,
    GenerateContentResponse, Part, generative_service_client::GenerativeServiceClient,
};

impl From<&proto::GenerateContentResponse> for rig_core::completion::Usage {
    fn from(response: &proto::GenerateContentResponse) -> Self {
        response
            .usage_metadata
            .as_ref()
            .map(|u| rig_core::completion::Usage {
                input_tokens: Some(u.prompt_token_count as u64),
                output_tokens: Some(u.candidates_token_count as u64),
                total_tokens: Some(u.total_token_count as u64),
                cached_input_tokens: Some(u.cached_content_token_count as u64),
                cache_creation_input_tokens: None,
                tool_use_prompt_tokens: None,
                reasoning_tokens: None,
            })
            .unwrap_or_default()
    }
}
