//! Gemini completion and embedding wires over the gRPC API.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_gemini_grpc::{GeminiGrpc, completion::{GEMINI_3_8_FLASH, GenerateContent}};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let transport = GeminiGrpc::new("YOUR_API_KEY").await?;
//! let model = transport.completion(GEMINI_3_8_FLASH);
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
        completion::map_usage(response.usage_metadata.as_ref())
    }
}
