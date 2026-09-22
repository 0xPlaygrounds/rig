//! Speech model identifiers for OpenAI's `/audio/speech` endpoint.
//!
//! ```
//! use rig_core::providers::openai::{OpenAI, audio_generation::TTS_1};
//! let wire = OpenAI::new("key").speech(TTS_1);
//! ```

pub const TTS_1: &str = "tts-1";
pub const TTS_1_HD: &str = "tts-1-hd";
