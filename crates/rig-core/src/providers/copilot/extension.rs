//! GitHub Copilot's typed reply extras, on Chat Completions and Responses
//! alike. Copilot has no typed request option; its intent is a header,
//! set with
//! [`CopilotWire::with_intent`](crate::providers::copilot::wire::CopilotWire::with_intent).
//!
//! ```
//! use rig_core::completion::CompletionResponse;
//! use rig_core::providers::copilot::extension::Copilot;
//!
//! fn billed(reply: &CompletionResponse) -> Option<u64> {
//!     reply.extras::<Copilot>()?.ok()?.copilot_usage?.total_nano_aiu
//! }
//! # let _ = billed;
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// Copilot's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Copilot;

impl ProviderExtension for Copilot {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = CopilotOptions;
    type Extras = CopilotExtras;
}

/// Copilot's request options: none.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CopilotOptions {}

impl CopilotOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ExtensionOptions for CopilotOptions {}

/// Copilot's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CopilotExtras {
    /// What the request was billed. Both routes.
    pub copilot_usage: Option<CopilotUsage>,
    /// The content filter's verdicts on the prompt. Chat only.
    pub prompt_filter_results: Option<Vec<Value>>,
}

/// What a Copilot request was billed.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct CopilotUsage {
    /// The tokens billed, by kind.
    #[serde(default)]
    pub token_details: Option<Vec<TokenDetail>>,
    /// The total, in billionths of an AI unit.
    #[serde(default)]
    pub total_nano_aiu: Option<u64>,
}

/// The tokens of one kind a request was billed.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct TokenDetail {
    /// The kind, such as `input`, `output` or `cache_read`.
    #[serde(default)]
    pub token_type: Option<String>,
    /// How many tokens.
    #[serde(default)]
    pub token_count: Option<u64>,
    /// The tokens one price applies to.
    #[serde(default)]
    pub batch_size: Option<u64>,
    /// The price per batch, in billionths of an AI unit.
    #[serde(default)]
    pub cost_per_batch: Option<u64>,
}

impl ReplyExtras for CopilotExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            copilot_usage: reply_field(raw, "/copilot_usage")?,
            prompt_filter_results: reply_field(raw, "/prompt_filter_results")?,
        })
    }
}

#[cfg(test)]
mod tests;
