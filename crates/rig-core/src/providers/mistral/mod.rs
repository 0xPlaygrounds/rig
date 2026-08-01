//! Mistral API integration.
//!
//! # Example
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::http_runtime::HttpRuntime;
//! use rig_core::providers::mistral;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = mistral::functions::Config::from_env(mistral::MISTRAL_LARGE)?;
//! let rt = HttpRuntime::new();
//!
//! let request = CompletionRequest::from_prompt("Who are you?");
//! let response = mistral::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

pub mod completion;
pub mod embedding;
pub mod functions;
pub mod model_listing;
pub mod transcription;

crate::providers::client::define_http_client! {
    config = functions::Config,
    default_base_url = functions::DEFAULT_BASE_URL,
    api_key_required = true,
}
crate::providers::client::impl_http_embedding_config_factory!(Client, functions::EmbeddingConfig);

impl Client {
    /// Materialize transcription configuration sharing this connection.
    pub fn transcription_config(&self, model: impl Into<String>) -> functions::Config {
        self.config(model)
    }
}

pub use completion::*;
pub use embedding::*;
pub use transcription::*;
