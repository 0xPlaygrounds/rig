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
pub mod image_generation;
pub mod streaming;
pub mod transcription;

pub use client::{Client, ClientBuilder, SubProvider};
pub use completion::{
    GEMMA_2, META_LLAMA_3_1, PHI_4, QWEN_QVQ_PREVIEW, QWEN2_5, QWEN2_5_CODER, QWEN2_VL,
    SMALLTHINKER_PREVIEW,
};

#[cfg(feature = "image")]
pub use image_generation::{FLUX_1, KOLORS, STABLE_DIFFUSION_3};
pub use transcription::{WHISPER_LARGE_V3, WHISPER_LARGE_V3_TURBO, WHISPER_SMALL};
