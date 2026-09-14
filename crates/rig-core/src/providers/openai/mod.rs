//! OpenAI API clients and models. Enable the `openai` Cargo feature.

pub mod client;
pub mod model_listing;
pub use super::openai_compatible::*;
pub use client::*;
pub use model_listing::*;
pub mod transcription;
pub use transcription::*;
#[cfg(feature = "image")]
pub mod image_generation;
#[cfg(feature = "image")]
pub use image_generation::*;
#[cfg(feature = "audio")]
pub mod audio_generation;
#[cfg(feature = "audio")]
pub use audio_generation::{
    AudioGenerationModel, CompletionsAudioGenerationModel, TTS_1, TTS_1_HD,
};
