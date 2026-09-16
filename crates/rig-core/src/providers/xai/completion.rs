//! xAI completion support through its OpenAI-compatible Responses API.

pub use crate::providers::openai::responses_api::CompletionResponse;

use super::client::XAi;

/// xAI completion model, driven by the shared Responses implementation.
pub type CompletionModel<H = crate::http_client::BoxedHttpClient> =
    crate::providers::openai::responses_api::GenericResponsesCompletionModel<XAi, H>;

// The model identifiers are provider data and live beside the dialect in
// `super`; they are re-exported here so `xai::completion::GROK_3` keeps
// resolving while the client layer stands.
pub use super::{
    GROK_2_1212, GROK_2_IMAGE_1212, GROK_2_VISION_1212, GROK_3, GROK_3_FAST, GROK_3_MINI,
    GROK_3_MINI_FAST, GROK_4,
};

#[cfg(test)]
mod tests;
