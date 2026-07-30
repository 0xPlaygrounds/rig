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

pub use completion::*;
pub use embedding::*;
