//! The xAI completion wire over xAI's gRPC API.
//!
//! Requests are built by rig-core's xAI Chat Completions encoder and sent as
//! the `GetCompletionsRequest` they describe; replies are read back as Chat
//! Completions chunks by the same decoder, so history, tools, reasoning and
//! typed options behave as on the REST wire.
//!
//! ```no_run
//! use rig_xai_grpc::XaiGrpc;
//!
//! # async fn example() -> Result<(), rig_xai_grpc::XaiGrpcError> {
//! let transport = XaiGrpc::new("YOUR_API_KEY").await?;
//! let model = transport.completion("grok-4.20-0309-non-reasoning");
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

mod auth;
mod chat;
pub mod client;
pub mod completion;
pub mod json;
pub mod streaming;

pub use client::{XaiGrpc, XaiGrpcError};

/// Messages and the `Chat` service client generated from xAI's protos.
///
/// ```
/// use rig_xai_grpc::proto::GetChatCompletionResponse;
///
/// let response = GetChatCompletionResponse::default();
/// assert!(response.outputs.is_empty());
/// ```
pub mod proto {
    #![allow(clippy::all)]
    #![allow(warnings)]
    tonic::include_proto!("xai_api");
}

pub use proto::{
    GetChatCompletionChunk, GetChatCompletionResponse, GetCompletionsRequest,
    chat_client::ChatClient,
};
