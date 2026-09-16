//! xAI API client and Rig integration
//!
//! # Example
//! ```ignore
//! use rig_core::{client::CompletionClient, providers::xai};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = xai::Client::new("YOUR_API_KEY")?;
//!
//! let grok = client.completion_model(xai::GROK_3);
//! # Ok(())
//! # }
//! ```

pub(crate) mod api;
#[cfg(feature = "audio")]
pub mod audio_generation;
pub mod client;
pub mod completion;
#[cfg(feature = "image")]
pub mod image_generation;

#[cfg(feature = "audio")]
pub use audio_generation::{AudioGenerationModel, TTS_1};
pub use client::Client;
pub use completion::{CompletionModel, CompletionResponse};
#[cfg(feature = "image")]
pub use image_generation::{GROK_IMAGINE_IMAGE, GROK_IMAGINE_IMAGE_PRO, ImageGenerationModel};

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
use crate::providers::openai::responses_api::wire::{Dialect, Quirks, RequestShape};

/// xAI, as a Responses dialect.
///
/// Every field is what this gateway does differently: its endpoint lives
/// under `/v1`, it takes its own input shape (see [`api`]), it rejects
/// top-level `instructions` so system messages stay in `input`, it answers
/// a 200 with its error envelope, it publishes a finished function call at
/// its `output_item.done` rather than at the terminal, and its native
/// structured output does not compose with tool calls.
pub const DIALECT: Dialect = Dialect {
    name: "xai",
    base_url: "https://api.x.ai",
    api_key_env: "XAI_API_KEY",
    base_url_env: None,
    request_id_header: Some("x-request-id"),
    quirks: Quirks {
        path: "/v1/responses",
        system_instructions: SystemInstructionsPlacement::InputSystemMessages,
        request: RequestShape::Xai,
        base_url_env_alias: None,
        account_id_env: None,
        default_instructions: None,
        instructions_env: None,
        identity: None,
        always_streams: false,
        relaxed_content_type: false,
        codex_parameter_subset: false,
        error_envelope_in_success: true,
        repair_envelope_less_frames: false,
        native_output_with_tools: false,
    },
};
