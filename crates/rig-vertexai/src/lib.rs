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
//! The Vertex AI completion wire and the [`VertexAi`] transport it is sent
//! through. Configure Application Default Credentials or supply credentials
//! or a prediction service through [`VertexAiBuilder`]. ADC construction
//! requires a Tokio runtime that remains alive and driven for the
//! transport's lifetime.
//!
//! ```no_run
//! use rig_core::wire::Wire as _;
//! use rig_vertexai::{VertexAi, completion::{GEMINI_2_5_FLASH, GenerateContent}};
//!
//! # async fn example() -> Result<(), rig_vertexai::client::VertexAiClientError> {
//! let model = GenerateContent::new(GEMINI_2_5_FLASH).on(VertexAi::from_env()?);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;
pub(crate) mod types;

pub use client::{VertexAi, VertexAiBuilder};
pub use types::completion_response::VERTEX_TEXT_EXTRAS_KEY;
