//! Doubleword inference API integration.
//!
//! [Doubleword](https://docs.doubleword.ai) is an OpenAI-compatible inference
//! provider. This integration covers the **realtime** tier: synchronous chat
//! completions and streaming, plus embeddings on the same endpoint.
//! Doubleword's cheaper **async** and **batch** tiers run through the
//! OpenAI-compatible Batch API (`/v1/batches`); Rig support for them is not
//! yet included.
//!
//! Set `DOUBLEWORD_API_KEY` (and optionally `DOUBLEWORD_BASE_URL`) to use
//! [`functions::Config::from_env`].
//!
//! # Example
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::http_runtime::HttpRuntime;
//! use rig_core::providers::doubleword;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = doubleword::functions::Config::from_env(doubleword::QWEN3_5_9B)?;
//! let rt = HttpRuntime::new();
//!
//! let request = CompletionRequest::from_prompt("What is Rig?");
//! let response = doubleword::functions::complete(&cfg, &rt, request).await?;
//! # let _ = response;
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
