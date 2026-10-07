//! llama.cpp's typed request options and reply extras
//! (<https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>).
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::llamacpp::extension::{LlamaCppExt, LlamaCppOptions};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = LlamaCppOptions::new().top_k(40).min_p(0.05);
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<LlamaCppExt>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// llama.cpp's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LlamaCppExt;

impl ProviderExtension for LlamaCppExt {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = LlamaCppOptions;
    type Extras = LlamaCppExtras;
}

/// llama.cpp's request options.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct LlamaCppOptions {
    /// The fields every route takes.
    #[serde(rename = "*")]
    pub shared: LlamaCppShared,
}

/// The fields `llama-server` takes.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct LlamaCppShared {
    /// Arguments to the model's chat template, such as `enable_thinking`.
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub chat_template_kwargs: Map<String, Value>,
    /// How the reply carries the reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_format: Option<ReasoningFormat>,
    /// How many of the most likely tokens to report per position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_probs: Option<u32>,
    /// The samplers, in the order they run.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub samplers: Vec<String>,
    /// Sample from the `k` most likely tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// The minimum probability of a token, relative to the most likely one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    /// Locally typical sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f64>,
    /// Mirostat sampling: 0 off, 1 Mirostat, 2 Mirostat 2.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<u8>,
    /// Mirostat's target entropy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f64>,
    /// Mirostat's learning rate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_eta: Option<f64>,
    /// The server slot that runs the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_slot: Option<i32>,
    /// Whether every streamed chunk carries timings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings_per_token: Option<bool>,
}

/// How a reply carries the reasoning.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReasoningFormat {
    /// Left in the content.
    None,
    /// In `reasoning_content`.
    Deepseek,
    /// In `reasoning_content`, and in the content's `<think>` tags too.
    DeepseekLegacy,
    /// As the server's template decides.
    Auto,
}

impl LlamaCppOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pass `key`, `value` to the model's chat template.
    pub fn chat_template_kwarg(mut self, key: impl Into<String>, value: Value) -> Self {
        self.shared.chat_template_kwargs.insert(key.into(), value);
        self
    }

    /// Carry the reasoning as `format`.
    pub fn reasoning_format(mut self, format: ReasoningFormat) -> Self {
        self.shared.reasoning_format = Some(format);
        self
    }

    /// Report the `count` most likely tokens per position.
    pub fn n_probs(mut self, count: u32) -> Self {
        self.shared.n_probs = Some(count);
        self
    }

    /// Run `samplers`, in order.
    pub fn samplers(mut self, samplers: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.shared.samplers = samplers.into_iter().map(Into::into).collect();
        self
    }

    /// Sample from the `k` most likely tokens.
    pub fn top_k(mut self, k: i32) -> Self {
        self.shared.top_k = Some(k);
        self
    }

    /// Set the minimum relative token probability.
    pub fn min_p(mut self, p: f64) -> Self {
        self.shared.min_p = Some(p);
        self
    }

    /// Set locally typical sampling.
    pub fn typical_p(mut self, p: f64) -> Self {
        self.shared.typical_p = Some(p);
        self
    }

    /// Set the Mirostat version: 0 off, 1 or 2.
    pub fn mirostat(mut self, version: u8) -> Self {
        self.shared.mirostat = Some(version);
        self
    }

    /// Set Mirostat's target entropy.
    pub fn mirostat_tau(mut self, tau: f64) -> Self {
        self.shared.mirostat_tau = Some(tau);
        self
    }

    /// Set Mirostat's learning rate.
    pub fn mirostat_eta(mut self, eta: f64) -> Self {
        self.shared.mirostat_eta = Some(eta);
        self
    }

    /// Run the request on server slot `slot`.
    pub fn id_slot(mut self, slot: i32) -> Self {
        self.shared.id_slot = Some(slot);
        self
    }

    /// Whether every streamed chunk carries timings.
    pub fn timings_per_token(mut self, per_token: bool) -> Self {
        self.shared.timings_per_token = Some(per_token);
        self
    }
}

impl ExtensionOptions for LlamaCppOptions {}

/// llama.cpp's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LlamaCppExtras {
    /// How long the prompt and the completion took.
    pub timings: Option<Timings>,
}

/// How long a `llama-server` request took.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct Timings {
    /// Prompt tokens reused from the cache.
    #[serde(default)]
    pub cache_n: Option<u64>,
    /// Prompt tokens evaluated.
    #[serde(default)]
    pub prompt_n: Option<u64>,
    /// Milliseconds spent on the prompt.
    #[serde(default)]
    pub prompt_ms: Option<f64>,
    /// Milliseconds per prompt token.
    #[serde(default)]
    pub prompt_per_token_ms: Option<f64>,
    /// Prompt tokens per second.
    #[serde(default)]
    pub prompt_per_second: Option<f64>,
    /// Tokens generated.
    #[serde(default)]
    pub predicted_n: Option<u64>,
    /// Milliseconds spent generating.
    #[serde(default)]
    pub predicted_ms: Option<f64>,
    /// Milliseconds per generated token.
    #[serde(default)]
    pub predicted_per_token_ms: Option<f64>,
    /// Generated tokens per second.
    #[serde(default)]
    pub predicted_per_second: Option<f64>,
}

impl ReplyExtras for LlamaCppExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            timings: reply_field(raw, "/timings")?,
        })
    }
}

#[cfg(test)]
mod tests;
