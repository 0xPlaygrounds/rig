#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! Vertex AI model completions through Rig's completion traits.
//! Configure Application Default Credentials or supply credentials or a prediction
//! service through [`ClientBuilder`]. ADC construction requires a Tokio runtime
//! that remains alive and driven for the client's lifetime.
//!
//! ```no_run
//! use rig_vertexai::Client;
//!
//! # async fn example() -> Result<(), rig_vertexai::client::VertexAiClientError> {
//! let client = Client::from_env()?;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;
pub(crate) mod types;

pub use client::{Client, ClientBuilder};
pub use types::completion_response::VERTEX_TEXT_EXTRAS_KEY;
