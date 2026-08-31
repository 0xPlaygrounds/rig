//! Create a new completion model with the given name
//!
//! # Example
//! ```ignore
//! use rig_core::{
//!     client::CompletionClient,
//!     providers::huggingface::{client, completion},
//! };
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize the Huggingface client
//! let client = client::Client::new("your-huggingface-api-key")?;
//!
//! let completion_model = client.completion_model(completion::GEMMA_2);
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
pub mod transcription;

// Hoisted to the provider root, where the deleted `rig-reqwest` alias tree
// put them and where callers name them.
pub use client::{Client, ClientBuilder, SubProvider};
pub use completion::CompletionModel;
#[cfg(feature = "image")]
pub use image_generation::ImageGenerationModel;
#[cfg(feature = "image")]
pub use image_generation::image_generation_models::*;
pub use transcription::TranscriptionModel;
