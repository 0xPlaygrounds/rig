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
