//! xAI's model identifiers and its dialect.
//!
//! xAI speaks the OpenAI wires, so it has no client and no completion model
//! of its own: [`DIALECT`] carries the base URL, the `XAI_API_KEY` variable,
//! the `x-request-id` header and the quirks below — its completion is the
//! Responses endpoint at `/v1/responses`, and its image and speech endpoints
//! are OpenAI-shaped under `/v1`.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::openai::OpenAI;
//! use rig_core::providers::xai;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // The wire; `.bind(transport)` joins it to a socket.
//! let grok = OpenAI::from_env_with(&xai::DIALECT)?.completion(xai::GROK_3);
//! # let _ = grok;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "audio")]
pub mod audio_generation;
#[cfg(feature = "image")]
pub mod image_generation;

#[cfg(feature = "audio")]
pub use audio_generation::TTS_1;
#[cfg(feature = "image")]
pub use image_generation::{GROK_IMAGINE_IMAGE, GROK_IMAGINE_IMAGE_PRO};

/// xAI completion models.
pub const GROK_2_1212: &str = "grok-2-1212";
pub const GROK_2_VISION_1212: &str = "grok-2-vision-1212";
pub const GROK_3: &str = "grok-3";
pub const GROK_3_FAST: &str = "grok-3-fast";
pub const GROK_3_MINI: &str = "grok-3-mini";
pub const GROK_3_MINI_FAST: &str = "grok-3-mini-fast";
pub const GROK_2_IMAGE_1212: &str = "grok-2-image-1212";
pub const GROK_4: &str = "grok-4-0709";

use crate::providers::openai::responses_api::SystemInstructionsPlacement;
use crate::providers::openai::wire::{
    Dialect, ImageBody, Quirks, ResponsesContract, ResponsesQuirks, Route, SpeechBody,
};

/// xAI, as an OpenAI dialect.
///
/// Every field is what this gateway does differently. Its endpoints live
/// under `/v1` on a bare host; its text-to-speech endpoint is `/v1/tts` and
/// takes a body of its own, as does its image endpoint. Its Responses
/// endpoint rejects top-level `instructions` so system messages stay in
/// `input`, answers a 200 with its error envelope, publishes a finished
/// function call at its `output_item.done` rather than at the terminal, and
/// its native structured output does not compose with tool calls.
pub const DIALECT: Dialect = Dialect {
    request_id_header: Some("x-request-id"),
    quirks: Quirks {
        completion_route: Route::Responses,
        completion_path: "/v1/chat/completions",
        embeddings_path: "/v1/embeddings",
        models_path: "/v1/models",
        verify_path: "/v1/api-key",
        transcription_path: "/v1/audio/transcriptions",
        image_generation_path: "/v1/images/generations",
        audio_generation_path: "/v1/tts",
        image_body: ImageBody::Xai,
        speech_body: SpeechBody::Xai,
        responses: ResponsesQuirks {
            path: "/v1/responses",
            system_instructions: SystemInstructionsPlacement::InputSystemMessages,
            contract: ResponsesContract::Xai,
            ..ResponsesQuirks::openai()
        },
        ..Quirks::openai()
    },
    ..Dialect::gateway("xai", "https://api.x.ai", "XAI_API_KEY")
};
