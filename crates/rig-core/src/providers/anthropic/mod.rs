//! Anthropic API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::providers::anthropic;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = anthropic::functions::Config::from_env(anthropic::completion::CLAUDE_SONNET_4_6)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//!
//! let request = rig_core::completion::CompletionRequest::from_prompt("Hello world!");
//! let response = anthropic::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

pub mod completion;
pub mod functions;
pub mod model_listing;
pub mod streaming;

crate::providers::client::define_http_client! {
    config = functions::Config,
    default_base_url = functions::DEFAULT_BASE_URL,
    api_key_required = true,
}

impl ClientBuilder {
    /// Add an Anthropic beta feature header to every request made by the client.
    ///
    /// This is primarily useful for APIs, such as Files, whose beta selection
    /// belongs to the connection rather than a completion-model config.
    pub fn anthropic_beta(self, beta: impl Into<String>) -> Self {
        self.extra_header("anthropic-beta", beta)
    }
}
