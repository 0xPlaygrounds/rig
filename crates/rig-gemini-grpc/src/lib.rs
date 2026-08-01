//! Google Gemini gRPC API integration for Rig.
//!
//! gRPC-based access to the Gemini API, offering better performance and type
//! safety than the REST API.
//!
//! # Entry point: [`functions`]
//!
//! The crate's face is data-oriented: a serde
//! [`functions::Config`] / [`functions::EmbeddingConfig`] describing *how* to
//! reach Gemini, plus free functions
//! [`functions::complete`] / [`functions::open_stream`] /
//! [`functions::embed`] / [`functions::embed_batches`].
//!
//! Because gRPC is a non-HTTP transport, a connected tonic channel cannot be
//! plain data — [`Client`] is that live handle, built from a config by
//! [`functions::client_from_config`] (or from the environment by
//! [`Client::from_env`]) and passed to each free function.
//!
//! ```no_run
//! use rig_gemini_grpc::{completion::GEMINI_2_0_FLASH, functions};
//! use rig_core::completion::CompletionRequest;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let cfg = functions::Config::new(GEMINI_2_0_FLASH);
//! let client = functions::client_from_config(&cfg).await?;
//!
//! let response = functions::complete(
//!     &client,
//!     &cfg.model,
//!     CompletionRequest::from_prompt("Hello!"),
//! )
//! .await?;
//! # let _ = response;
//! # Ok(())
//! # }
//! ```
//!
//! With the root `rig` crate's `gemini-grpc` feature and `rig::prelude::*`,
//! this same live client supports `client.agent(model)` and the bound direct
//! completion facade. Both paths seed the runtime with this connected channel.
//!
//! [`completion`], [`embedding`] and [`streaming`] hold the wire conversions
//! and model-identifier constants the free functions are built from.

pub mod client;
pub mod completion;
pub mod embedding;
pub mod functions;
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

// Normalize the proto usage metadata into rig's usage type for both the unary
// and streaming completion paths.
impl proto::GenerateContentResponse {
    /// Token usage reported by the API, zero-valued when missing.
    pub fn token_usage(&self) -> rig_core::completion::Usage {
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
}
