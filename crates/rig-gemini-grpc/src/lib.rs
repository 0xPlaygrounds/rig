//! Gemini completion and embedding wires over the gRPC API.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_gemini_grpc::{GeminiGrpc, completion::{GEMINI_2_0_FLASH, GenerateContent}};
//!
//! # async fn example() -> Result<(), rig_gemini_grpc::GeminiGrpcError> {
//! let transport = GeminiGrpc::new("YOUR_API_KEY").await?;
//! let model = transport.completion(GEMINI_2_0_FLASH);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;
pub mod embedding;
pub mod rest;
pub mod streaming;

pub use client::{GeminiGrpc, GeminiGrpcError};

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

/// The usage the REST wire reads from this reply's JSON.
impl From<&proto::GenerateContentResponse> for rig_core::completion::Usage {
    fn from(response: &proto::GenerateContentResponse) -> Self {
        rest::to_rest(response)
            .ok()
            .and_then(|json| json.get("usageMetadata").map(usage_of))
            .unwrap_or_default()
    }
}

use rig_core::providers::gemini::completion::gemini_api_types::usage_of;
