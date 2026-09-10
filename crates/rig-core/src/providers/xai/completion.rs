//! xAI completion support through its OpenAI-compatible Responses API.

pub use crate::providers::openai::responses_api::CompletionResponse;

use super::client::XAi;

/// xAI completion model, driven by the shared Responses implementation.
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
    crate::providers::openai::responses_api::GenericResponsesCompletionModel<XAi, H>;

/// xAI completion models.
pub const GROK_2_1212: &str = "grok-2-1212";
pub const GROK_2_VISION_1212: &str = "grok-2-vision-1212";
pub const GROK_3: &str = "grok-3";
pub const GROK_3_FAST: &str = "grok-3-fast";
pub const GROK_3_MINI: &str = "grok-3-mini";
pub const GROK_3_MINI_FAST: &str = "grok-3-mini-fast";
pub const GROK_2_IMAGE_1212: &str = "grok-2-image-1212";
pub const GROK_4: &str = "grok-4-0709";

#[cfg(test)]
mod tests;
