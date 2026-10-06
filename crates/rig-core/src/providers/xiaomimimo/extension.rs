//! Xiaomi MiMo's typed reply extras on its OpenAI-format and Messages APIs.
//! MiMo has no typed request option: its web search is a server tool,
//! declared in `additional_params.tools`.
//!
//! ```
//! use rig_core::completion::CompletionResponse;
//! use rig_core::providers::xiaomimimo::extension::XiaomiMimo;
//!
//! fn annotations(reply: &CompletionResponse) -> usize {
//!     reply
//!         .extras::<XiaomiMimo>()
//!         .and_then(Result::ok)
//!         .and_then(|extras| extras.annotations)
//!         .map_or(0, |annotations| annotations.len())
//! }
//! # let _ = annotations;
//! ```

use serde::Serialize;
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;
use crate::providers::anthropic::extension::MessagesStop;
use crate::providers::anthropic::wire::MESSAGES_API;

/// Xiaomi MiMo's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct XiaomiMimo;

impl ProviderExtension for XiaomiMimo {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = XiaomiMimoOptions;
    type Extras = XiaomiMimoExtras;
}

/// Xiaomi MiMo's request options: none.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct XiaomiMimoOptions {}

impl XiaomiMimoOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ExtensionOptions for XiaomiMimoOptions {}

/// Xiaomi MiMo's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct XiaomiMimoExtras {
    /// The message's annotations, such as web-search citations.
    /// OpenAI-format API only.
    pub annotations: Option<Vec<Value>>,
    /// `stop_reason`, verbatim. Messages route.
    pub stop_reason: Option<String>,
    /// `stop_sequence`: the stop sequence the turn ended on. Messages route.
    pub stop_sequence: Option<String>,
}

impl ReplyExtras for XiaomiMimoExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        if api.as_str() == MESSAGES_API {
            let MessagesStop {
                stop_reason,
                stop_sequence,
            } = MessagesStop::read("Xiaomi MiMo", api, raw)?;
            return Ok(Self {
                stop_reason,
                stop_sequence,
                ..Self::default()
            });
        }
        if api.as_str() != "openai.chat" {
            return Ok(Self::default());
        }
        Ok(Self {
            annotations: reply_field(raw, "/choices/0/message/annotations")?,
            ..Self::default()
        })
    }
}

#[cfg(test)]
mod tests;
