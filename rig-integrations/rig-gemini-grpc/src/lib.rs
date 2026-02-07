//! Google Gemini gRPC API client and Rig integration
//!
//! This module provides gRPC-based access to the Gemini API, offering better
//! performance and type safety compared to the REST API.
//!
//! # Example
//! ```no_run
//! use rig_gemini_grpc::{Client, completion::GEMINI_2_0_FLASH};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
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

// Implement GetTokenUsage for proto::GenerateContentResponse to support streaming
impl rig::completion::GetTokenUsage for proto::GenerateContentResponse {
    fn token_usage(&self) -> Option<rig::completion::Usage> {
        self.usage_metadata
            .as_ref()
            .map(|u| rig::completion::Usage {
                input_tokens: u.prompt_token_count as u64,
                output_tokens: u.candidates_token_count as u64,
                total_tokens: u.total_token_count as u64,
                cached_input_tokens: u.cached_content_token_count as u64,
            })
    }
}
