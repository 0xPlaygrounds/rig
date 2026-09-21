//! xAI's model identifiers and its dialect.
//!
//! [`DIALECT`] configures the Responses endpoint and reads `XAI_API_KEY`.
//!
//! ```no_run
//! use rig_core::providers::openai::OpenAI;
//! use rig_core::providers::xai;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
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

/// Identifier for the Grok 2 December 2024 model.
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

/// xAI endpoint and encoding configuration. Responses system messages remain
/// in `input`; error envelopes may arrive with HTTP 200. Completed function calls
/// arrive at `output_item.done`, and structured output cannot combine with tools.
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
