//! Gemini completion and embedding models using the gRPC API.
//!
//! ```no_run
//! use rig_core::driver::CompletionProvider;
//! use rig_gemini_grpc::{Client, completion::GEMINI_2_0_FLASH};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let client = Client::new("YOUR_API_KEY").await?;
//!
//! let completion_model = client.completion(GEMINI_2_0_FLASH);
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;
pub mod embedding;
pub mod streaming;

pub use client::Client;

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
