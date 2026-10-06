//! Ollama's typed request options and reply extras. The shared section
//! goes to both routes, the OpenAI-compatible `/v1` and the native
//! `/api/chat`; the `"ollama.chat"` section, with the model parameters, goes
//! to the native route only and is skipped on `/v1`.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::ollama::extension::{KeepAlive, Ollama, OllamaOptions};
//!
//! let options = OllamaOptions::default()
//!     .keep_alive(KeepAlive::duration("5m"))
//!     .num_ctx(8192);
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<Ollama>(&options)?);
//! # Ok::<(), rig_core::completion::OptionsError>(())
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// The `ollama` provider: its key, [`OllamaOptions`] and [`OllamaExtras`].
#[derive(Debug)]
pub enum Ollama {}

impl ProviderExtension for Ollama {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = OllamaOptions;
    type Extras = OllamaExtras;
}

/// Ollama's request options.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OllamaOptions {
    /// The fields both routes read.
    #[serde(rename = "*")]
    pub shared: OllamaShared,
    /// The fields only `/api/chat` reads.
    #[serde(rename = "ollama.chat")]
    pub chat: OllamaNative,
}

impl ExtensionOptions for OllamaOptions {}

impl OllamaOptions {
    /// How long the daemon keeps the model loaded after the request
    /// (`keep_alive`).
    pub fn keep_alive(mut self, keep_alive: KeepAlive) -> Self {
        self.shared.keep_alive = Some(keep_alive);
        self
    }

    /// The context window, in tokens (`options.num_ctx`).
    pub fn num_ctx(mut self, num_ctx: u32) -> Self {
        self.chat.options.num_ctx = Some(num_ctx);
        self
    }

    /// The prompt tokens kept when the context is shifted
    /// (`options.num_keep`).
    pub fn num_keep(mut self, num_keep: u32) -> Self {
        self.chat.options.num_keep = Some(num_keep);
        self
    }

    /// Sample from the `top_k` most likely tokens (`options.top_k`).
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.chat.options.top_k = Some(top_k);
        self
    }

    /// Drop tokens below `min_p` times the likeliest token's probability
    /// (`options.min_p`).
    pub fn min_p(mut self, min_p: f64) -> Self {
        self.chat.options.min_p = Some(min_p);
        self
    }

    /// Penalize repeated tokens (`options.repeat_penalty`).
    pub fn repeat_penalty(mut self, penalty: f64) -> Self {
        self.chat.options.repeat_penalty = Some(penalty);
        self
    }

    /// How far back the repeat penalty looks, in tokens; `0` disables it
    /// and `-1` is the context window (`options.repeat_last_n`).
    pub fn repeat_last_n(mut self, last_n: i32) -> Self {
        self.chat.options.repeat_last_n = Some(last_n);
        self
    }

    /// The model layers offloaded to the GPU (`options.num_gpu`).
    pub fn num_gpu(mut self, num_gpu: i32) -> Self {
        self.chat.options.num_gpu = Some(num_gpu);
        self
    }

    /// The CPU threads generation uses (`options.num_thread`).
    pub fn num_thread(mut self, num_thread: u32) -> Self {
        self.chat.options.num_thread = Some(num_thread);
        self
    }

    /// Return each generated token's log probability (`logprobs`), read
    /// back as [`OllamaExtras::logprobs`].
    pub fn logprobs(mut self, logprobs: bool) -> Self {
        self.chat.logprobs = Some(logprobs);
        self
    }

    /// The most likely alternatives returned beside each token
    /// (`top_logprobs`).
    pub fn top_logprobs(mut self, top_logprobs: u32) -> Self {
        self.chat.top_logprobs = Some(top_logprobs);
        self
    }
}

/// The fields both Ollama routes read.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OllamaShared {
    /// `keep_alive`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<KeepAlive>,
}

/// How long the daemon keeps a model loaded after a request.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum KeepAlive {
    /// A duration string such as `"5m"`; a negative one keeps the model
    /// loaded.
    Duration(String),
    /// A number of seconds; `0` unloads the model at once and a negative
    /// number keeps it loaded.
    Seconds(i64),
}

impl KeepAlive {
    /// A duration string such as `"5m"` or `"1h"`.
    pub fn duration(duration: impl Into<String>) -> Self {
        Self::Duration(duration.into())
    }

    /// A number of seconds.
    pub fn seconds(seconds: i64) -> Self {
        Self::Seconds(seconds)
    }
}

/// The fields only `/api/chat` reads.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OllamaNative {
    /// `options`: model parameters, beside the ones the request and its
    /// generation options set.
    #[serde(skip_serializing_if = "ModelOptions::is_empty")]
    pub options: ModelOptions,
    /// `logprobs`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// `top_logprobs`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
}

/// The model parameters `/api/chat` reads in `options`. `temperature`,
/// `num_predict`, `top_p`, `seed` and `stop` are not here: the request and
/// its generation options set them.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ModelOptions {
    /// `num_ctx`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    /// `num_keep`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_keep: Option<u32>,
    /// `top_k`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// `min_p`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    /// `repeat_penalty`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    /// `repeat_last_n`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    /// `num_gpu`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<i32>,
    /// `num_thread`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<u32>,
}

impl ModelOptions {
    /// Whether no parameter is set.
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

/// The fields of an `/api/chat` reply Rig does not normalize, read from
/// the whole reply or a stream's final record. A `/v1` reply carries only
/// [`Self::model`] and leaves the rest `None`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct OllamaExtras {
    /// `model`, on both routes.
    pub model: Option<String>,
    /// `created_at`.
    pub created_at: Option<String>,
    /// `done_reason` as the daemon spells it (`stop`, `length`, ...).
    pub done_reason: Option<String>,
    /// `total_duration`, in nanoseconds.
    pub total_duration: Option<u64>,
    /// `load_duration`, in nanoseconds.
    pub load_duration: Option<u64>,
    /// `prompt_eval_duration`, in nanoseconds.
    pub prompt_eval_duration: Option<u64>,
    /// `eval_duration`, in nanoseconds.
    pub eval_duration: Option<u64>,
    /// `prompt_eval_count`.
    pub prompt_eval_count: Option<u64>,
    /// `prompt_eval_cached_count`: the prompt tokens read from the cache.
    pub prompt_eval_cached_count: Option<u64>,
    /// `eval_count`.
    pub eval_count: Option<u64>,
    /// `logprobs`, when [`OllamaOptions::logprobs`] asked for them.
    pub logprobs: Option<Vec<Logprob>>,
}

/// One generated token's log probability.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct Logprob {
    /// `token`.
    pub token: Option<String>,
    /// `logprob`.
    pub logprob: Option<f64>,
    /// `bytes`, the token's UTF-8 bytes.
    pub bytes: Option<Vec<u8>>,
    /// `top_logprobs`, the likeliest alternatives, when
    /// [`OllamaOptions::top_logprobs`] asked for them.
    pub top_logprobs: Option<Vec<Logprob>>,
}

impl ReplyExtras for OllamaExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Self::deserialize(raw)
    }
}

#[cfg(test)]
mod tests;
