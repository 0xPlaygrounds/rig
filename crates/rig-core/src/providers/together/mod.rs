//! Together AI API integration
//!
//! # Example
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::http_runtime::HttpRuntime;
//! use rig_core::providers::together;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = together::functions::Config::from_env(together::LLAMA_3_70B_INSTRUCT_TURBO)?;
//! let rt = HttpRuntime::new();
//!
//! let request = CompletionRequest::from_prompt("Who are you?");
//! let response = together::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

pub mod completion;
pub mod embedding;
pub mod functions;

crate::providers::client::define_http_client! {
    config = functions::Config,
    default_base_url = functions::DEFAULT_BASE_URL,
    api_key_required = true,
}
crate::providers::client::impl_http_embedding_config_factory!(Client, functions::EmbeddingConfig);

pub use completion::*;
pub use embedding::*;
