//! Candle's typed request options and reply extras: the sampler settings
//! with no portable [`GenerationOptions`] form, over the model's
//! [`GenerationConfig`](crate::GenerationConfig) defaults, and the local
//! generation record each reply carries.
//!
//! [`GenerationOptions`]: rig_core::completion::GenerationOptions
//!
//! ```
//! use rig_candle::extension::{CandleOptions};
//! use rig_core::completion::CompletionRequest;
//!
//! let options = CandleOptions::default().top_k(40).repeat_penalty(1.2);
//! let request = CompletionRequest::new("hi").provider_option(options);
//! ```

use rig_core::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use rig_core::message::Api;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::FinishReason;

/// The `candle` provider: its key, [`CandleOptions`] and [`CandleExtras`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CandleExt;

impl ProviderExtension for CandleExt {
    const PROVIDER: &'static str = crate::types::PROVIDER_NAME;
    type Options = CandleOptions;
    type Extras = CandleExtras;
}

/// Candle's request options. A field left unset keeps the model's default.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CandleOptions {
    /// The sampler settings every generation reads.
    #[serde(rename = "*")]
    pub shared: CandleShared,
}

impl ExtensionOptions for CandleOptions {
    type Ext = CandleExt;
}

impl CandleOptions {
    /// Sample from the `top_k` most likely tokens. It must be positive and
    /// no larger than the vocabulary.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.shared.top_k = Some(top_k);
        self
    }

    /// Penalize tokens repeated in the recent context; `1.0` disables it.
    pub fn repeat_penalty(mut self, penalty: f32) -> Self {
        self.shared.repeat_penalty = Some(penalty);
        self
    }

    /// How many recent tokens the repeat penalty considers.
    pub fn repeat_last_n(mut self, last_n: usize) -> Self {
        self.shared.repeat_last_n = Some(last_n);
        self
    }
}

/// The sampler settings every generation reads.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CandleShared {
    /// `top_k`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// `repeat_penalty`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    /// `repeat_last_n`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<usize>,
}

/// The local generation record of a reply, the fields of
/// [`CandleCompletionResponse`](crate::CandleCompletionResponse). A field
/// the record does not carry is `None`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct CandleExtras {
    /// The decoded text, without the prompt and the stop token.
    pub text: Option<String>,
    /// The prompt's token count.
    pub prompt_tokens: Option<u64>,
    /// The sampled tokens, an end token included.
    pub generated_tokens: Option<u64>,
    /// The output limit the request and the defaults chose.
    pub requested_max_tokens: Option<u64>,
    /// The output limit left by the model's context.
    pub effective_max_tokens: Option<u64>,
    /// Why generation ended.
    pub finish_reason: Option<FinishReason>,
    /// The prompt's prefill time, in milliseconds.
    pub prefill_duration_ms: Option<u64>,
    /// The time to the first sampled token, in milliseconds.
    pub time_to_first_token_ms: Option<u64>,
    /// The prefill and generation time, in milliseconds.
    pub generation_duration_ms: Option<u64>,
    /// The generated tokens per second.
    pub tokens_per_second: Option<f64>,
}

impl ReplyExtras for CandleExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Self::deserialize(raw)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
