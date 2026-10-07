//! DeepSeek's typed reply extras. DeepSeek has no typed request option:
//! thinking and its effort are generation options.
//!
//! ```
//! use rig_core::completion::CompletionResponse;
//! use rig_core::providers::deepseek::extension::DeepSeekExt;
//!
//! fn cache_hits(reply: &CompletionResponse) -> Option<u64> {
//!     reply.extras::<DeepSeekExt>()?.ok()?.prompt_cache_hit_tokens
//! }
//! # let _ = cache_hits;
//! ```

use serde::Serialize;
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// DeepSeek's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeepSeekExt;

impl ProviderExtension for DeepSeekExt {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = DeepSeekOptions;
    type Extras = DeepSeekExtras;
}

/// DeepSeek's request options: none beyond the generation options.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct DeepSeekOptions {}

impl DeepSeekOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ExtensionOptions for DeepSeekOptions {}

/// DeepSeek's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeepSeekExtras {
    /// Prompt tokens read from the context cache.
    pub prompt_cache_hit_tokens: Option<u64>,
    /// Prompt tokens not in the context cache.
    pub prompt_cache_miss_tokens: Option<u64>,
    /// Completion tokens spent reasoning.
    pub reasoning_tokens: Option<u64>,
    /// The backend configuration's fingerprint.
    pub system_fingerprint: Option<String>,
}

impl ReplyExtras for DeepSeekExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            prompt_cache_hit_tokens: reply_field(raw, "/usage/prompt_cache_hit_tokens")?,
            prompt_cache_miss_tokens: reply_field(raw, "/usage/prompt_cache_miss_tokens")?,
            reasoning_tokens: reply_field(
                raw,
                "/usage/completion_tokens_details/reasoning_tokens",
            )?,
            system_fingerprint: reply_field(raw, "/system_fingerprint")?,
        })
    }
}

#[cfg(test)]
mod tests;
