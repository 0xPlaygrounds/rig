//! xAI API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::xai;
//!
//! let client = xai::Client::new("YOUR_API_KEY");
//!
//! let grok = client.completion_model(xai::GROK_3);
//! ```

mod api;
pub mod client;
pub mod completion;
mod streaming;

pub use client::Client;
pub use completion::{
    CompletionModel, CompletionResponse, GROK_2_1212, GROK_2_IMAGE_1212, GROK_2_VISION_1212,
    GROK_3, GROK_3_FAST, GROK_3_MINI, GROK_3_MINI_FAST, GROK_4,
};
