//! OpenRouter Inference API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::openrouter};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = openrouter::Client::new("YOUR_API_KEY")?;
//!
//! let sonar = client.completion_model(openrouter::PERPLEXITY_SONAR_PRO);
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;
pub mod client;
pub mod completion;
pub mod embedding;
pub mod model_listing;
pub mod streaming;
pub mod transcription;

#[cfg(feature = "audio")]
pub use audio_generation::{AudioGenerationModel, GPT_4O_MINI_TTS, KOKORO_82M, VOXTRAL_MINI_TTS};
pub use client::*;
pub use completion::*;
pub use embedding::*;
pub use model_listing::OpenRouterModelLister;
pub use transcription::{
    CHIRP_3, GPT_4O_MINI_TRANSCRIBE, GPT_4O_TRANSCRIBE, TranscriptionModel, TranscriptionResponse,
    WHISPER_1, WHISPER_LARGE_V3, WHISPER_LARGE_V3_TURBO,
};
