//! Create a new completion model with the given name
//!
//! # Example
//! ```
//! use rig::providers::huggingface::{client::self, completion::self}
//!
//! // Initialize the Huggingface client
//! let client = client::Client::new("your-huggingface-api-key");
//!
//! let completion_model = client.completion_model(completion::GEMMA_2);
//! ```

pub mod client;
pub mod completion;

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
pub mod streaming;
pub mod transcription;

pub use client::{Client, ClientBuilder, SubProvider};
#[cfg(feature = "image")]
pub use image_generation::image_generation_models::*;
