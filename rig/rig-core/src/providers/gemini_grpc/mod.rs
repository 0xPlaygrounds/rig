//! Google Gemini gRPC API client and Rig integration
//!
//! This module provides gRPC-based access to the Gemini API, offering better
//! performance and type safety compared to the REST API.
//!
//! # Example
//! ```no_run
//! use rig::providers::gemini_grpc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = gemini_grpc::Client::new("YOUR_API_KEY").await?;
//!
//! let completion_model = client.completion_model(gemini_grpc::completion::GEMINI_2_0_FLASH);
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
    tonic::include_proto!("gemini");
}

// Re-export commonly used proto types
pub use proto::{
    Content, GenerateContentRequest, GenerateContentResponse, Part,
    generative_service_client::GenerativeServiceClient,
};

// Implement GetTokenUsage for proto::GenerateContentResponse to support streaming
impl crate::completion::GetTokenUsage for proto::GenerateContentResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        self.usage_metadata
            .as_ref()
            .map(|u| crate::completion::Usage {
                input_tokens: u.prompt_token_count as u64,
                output_tokens: u.candidates_token_count.unwrap_or(0) as u64,
                total_tokens: u.total_token_count as u64,
            })
    }
}
