//! DeepSeek model identifiers and typed response fields, including cache usage
//! and reasoning content. [`from_env`] and [`new`] build a client on the [`DEEPSEEK`](crate::providers::openai::wire::DEEPSEEK) dialect.
//!
//! ```no_run
//! use rig_core::providers::deepseek;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let deepseek_chat = deepseek::from_env()?.chat(deepseek::DEEPSEEK_V4_FLASH);
//! # let _ = deepseek_chat;
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

use crate::providers::openai;

/// `deepseek-flash` completion model. DeepSeek points this unversioned name at
/// its latest Flash model, which is DeepSeek-V4.1-Flash as of September 2026.
pub const DEEPSEEK_FLASH: &str = "deepseek-flash";
pub const DEEPSEEK_V4_FLASH: &str = "deepseek-v4-flash";
pub const DEEPSEEK_V4_PRO: &str = "deepseek-v4-pro";

/// DeepSeek's reply, read back from
/// [`crate::completion::CompletionResponse::raw`]. Reasoning text is the
/// assistant message's `reasoning` field.
pub type CompletionResponse = openai::completion::ChatCompletionResponse<Usage>;

/// DeepSeek's accounting: the OpenAI-compatible counters plus its prompt-cache split.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Usage {
    /// The OpenAI-compatible counters.
    #[serde(flatten)]
    pub openai: openai::Usage,
    /// Prompt tokens served from DeepSeek's context cache.
    #[serde(default)]
    pub prompt_cache_hit_tokens: u32,
    /// Prompt tokens that missed the context cache.
    #[serde(default)]
    pub prompt_cache_miss_tokens: u32,
}

crate::client::macros::openai_vendor!(crate::providers::openai::wire::DEEPSEEK, "DeepSeek");
