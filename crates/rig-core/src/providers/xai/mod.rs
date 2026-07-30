//! xAI API integration
//!
//! # Example
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::http_runtime::HttpRuntime;
//! use rig_core::providers::xai;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = xai::functions::Config::from_env(xai::GROK_3)?;
//! let rt = HttpRuntime::new();
//!
//! let request = CompletionRequest::from_prompt("Who are you?");
//! let response = xai::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

mod api;
#[cfg(feature = "audio")]
pub mod audio_generation;
pub mod completion;
pub mod functions;
#[cfg(feature = "image")]
pub mod image_generation;
mod streaming;

#[cfg(feature = "audio")]
pub use audio_generation::TTS_1;
pub use completion::{
    CompletionResponse, GROK_2_1212, GROK_2_IMAGE_1212, GROK_2_VISION_1212, GROK_3, GROK_3_FAST,
    GROK_3_MINI, GROK_3_MINI_FAST, GROK_4,
};
#[cfg(feature = "image")]
pub use image_generation::{GROK_IMAGINE_IMAGE, GROK_IMAGINE_IMAGE_PRO};
