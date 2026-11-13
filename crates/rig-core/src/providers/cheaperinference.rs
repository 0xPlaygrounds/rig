//! [Cheaper Inference](https://cheaperinference.com/docs)'s OpenAI-compatible API.
//!
//! [`from_env`] reads `CHEAPER_INFERENCE_API_KEY` and the optional
//! `CHEAPER_INFERENCE_BASE_URL`. [`new`] takes the key explicitly. Select a
//! model identifier from the provider's [catalog](https://cheaperinference.com/models).
//!
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::providers::cheaperinference;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let reply = cheaperinference::from_env()?
//!     .chat("deepseek-v4-flash")
//!     .call(CompletionRequest::new("Summarize retrieval-augmented generation."))
//!     .await?;
//! # let _ = reply;
//! # Ok(())
//! # }
//! ```

/// The provider key used by the dialect, registry and reply origin.
pub const PROVIDER_NAME: &str = "cheaperinference";

crate::client::macros::openai_vendor!(
    crate::providers::openai::wire::CHEAPERINFERENCE,
    "Cheaper Inference"
);

#[cfg(test)]
mod tests;
