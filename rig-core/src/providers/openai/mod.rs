//! OpenAI API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::openai;
//!
//! let client = openai::Client::new("YOUR_API_KEY");
//!
//! let gpt4o = client.completion_model(openai::GPT_4O);
//! ```
pub mod client;
pub mod completion;
pub mod embedding;
pub mod image_generation;
pub mod streaming;
pub mod transcription;

pub use client::*;
pub use completion::*;
pub use embedding::*;
pub use image_generation::*;
pub use streaming::*;
pub use transcription::*;
