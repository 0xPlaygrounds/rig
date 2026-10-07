//! xAI's typed request options and reply extras, on Chat Completions and
//! Responses alike (<https://docs.x.ai/developers/cost-tracking>).
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::xai::extension::{XaiExt, XaiOptions};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = XaiOptions::new().prompt_cache_key("conversation-42");
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<XaiExt>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use serde::Serialize;
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// xAI's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct XaiExt;

impl ProviderExtension for XaiExt {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = XaiOptions;
    type Extras = XaiExtras;
}

/// xAI's request options.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct XaiOptions {
    /// The fields both routes take.
    #[serde(rename = "*")]
    pub shared: XaiShared,
}

/// The fields xAI takes on both routes.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct XaiShared {
    /// Routes requests that share a prefix to the same cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
}

impl XaiOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Route the prompt cache by `key`.
    pub fn prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.shared.prompt_cache_key = Some(key.into());
        self
    }
}

impl ExtensionOptions for XaiOptions {}

/// xAI's reply fields, from `usage` on both routes. Each is `None` when
/// the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct XaiExtras {
    /// The request's cost in ticks of 10^-10 USD.
    pub cost_in_usd_ticks: Option<u64>,
    /// Search sources the request read.
    pub num_sources_used: Option<u64>,
    /// Server-side tool calls the request made.
    pub num_server_side_tools_used: Option<u64>,
    /// Server-side tool calls per tool, such as `web_search_calls`.
    pub server_side_tool_usage_details: Option<Value>,
}

impl ReplyExtras for XaiExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            cost_in_usd_ticks: reply_field(raw, "/usage/cost_in_usd_ticks")?,
            num_sources_used: reply_field(raw, "/usage/num_sources_used")?,
            num_server_side_tools_used: reply_field(raw, "/usage/num_server_side_tools_used")?,
            server_side_tool_usage_details: reply_field(
                raw,
                "/usage/server_side_tool_usage_details",
            )?,
        })
    }
}

#[cfg(test)]
mod tests;
