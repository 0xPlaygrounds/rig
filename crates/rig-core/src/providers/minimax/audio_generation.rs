//! MiniMax's text-to-speech models and reply.
//!
//! The request's voice is the `voice_id` of a system, cloned or designed
//! voice. Other `voice_setting` fields (`vol`, `pitch`, `emotion`) and
//! top-level fields such as `audio_setting` or `language_boost` go in
//! `additional_params`; a `voice_setting` there adds to the derived one.
//! Rig asks for hex audio, which is the only form it decodes.
//!
//! ```no_run
//! use rig_core::audio_generation::AudioGenerationRequestBuilder;
//! use rig_core::providers::minimax;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = minimax::from_env()?.audio_generation(minimax::SPEECH_2_8_HD);
//! let request = AudioGenerationRequestBuilder::new("Hello", "English_Graceful_Lady")
//!     .additional_params(serde_json::json!({ "audio_setting": { "format": "wav" } }))
//!     .build();
//! let audio = model.call(request).await?.audio;
//! # let _ = audio;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

use crate::audio_generation::AudioGenerationResponse;
use crate::error::ProviderError;

/// `speech-2.8-hd`
pub const SPEECH_2_8_HD: &str = "speech-2.8-hd";
/// `speech-2.8-turbo`
pub const SPEECH_2_8_TURBO: &str = "speech-2.8-turbo";
/// `speech-2.6-hd`
pub const SPEECH_2_6_HD: &str = "speech-2.6-hd";
/// `speech-2.6-turbo`
pub const SPEECH_2_6_TURBO: &str = "speech-2.6-turbo";
/// `speech-02-hd`
pub const SPEECH_02_HD: &str = "speech-02-hd";
/// `speech-02-turbo`
pub const SPEECH_02_TURBO: &str = "speech-02-turbo";

/// MiniMax's non-streaming `POST /v1/t2a_v2` reply.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SpeechReply {
    /// The audio. MiniMax may send `null` here.
    #[serde(default)]
    pub data: Option<SpeechData>,
    /// Audio length, size, format and billed characters, verbatim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_info: Option<serde_json::Value>,
    /// MiniMax's trace id.
    #[serde(default)]
    pub trace_id: Option<String>,
}

/// The audio of a [`SpeechReply`].
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SpeechData {
    /// The audio, hex-encoded, in the requested `audio_setting.format`.
    #[serde(default)]
    pub audio: String,
}

impl SpeechReply {
    /// The decoded audio, with the trace id as the response id. A reply
    /// without hex audio, such as one asked for `output_format: url`, is an
    /// error.
    pub fn into_response(self) -> Result<AudioGenerationResponse, ProviderError> {
        let encoded = self.data.map(|data| data.audio).unwrap_or_default();
        let Some(audio) = decode_hex(&encoded).filter(|audio| !audio.is_empty()) else {
            return Err(ProviderError::Response(
                "MiniMax returned no hex audio; rig decodes only `output_format: hex`".to_owned(),
            ));
        };
        Ok(AudioGenerationResponse {
            response_id: self.trace_id.filter(|id| !id.is_empty()),
            ..AudioGenerationResponse::new(audio)
        })
    }
}

/// The bytes `hex` spells, or `None` when it is not an even run of hex digits.
fn decode_hex(hex: &str) -> Option<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    let digit = |byte: &u8| char::from(*byte).to_digit(16);
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| match pair {
            [high, low] => u8::try_from(digit(high)? * 16 + digit(low)?).ok(),
            _ => None,
        })
        .collect()
}
