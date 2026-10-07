//! Cohere's typed request options and reply extras. The shared section
//! goes to both routes, the Compatibility API and the native chat API; the
//! `"cohere.chat"` section goes to the native API only and is skipped on a
//! request the Compatibility API takes.
//!
//! ```
//! use rig_core::completion::CompletionRequest;
//! use rig_core::providers::cohere::extension::{CitationMode, CohereOptions};
//!
//! let options = CohereOptions::default()
//!     .frequency_penalty(0.2)
//!     .citation_mode(CitationMode::Fast);
//! let request = CompletionRequest::new("hi").provider_option(options);
//! ```

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// The `cohere` provider: its key, [`CohereOptions`] and [`CohereExtras`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CohereExt;

impl ProviderExtension for CohereExt {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = CohereOptions;
    type Extras = CohereExtras;
}

/// Cohere's request options.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CohereOptions {
    /// The fields both routes read.
    #[serde(rename = "*")]
    pub shared: CohereShared,
    /// The fields only the native chat API reads.
    #[serde(rename = "cohere.chat")]
    pub chat: CohereNative,
}

impl ExtensionOptions for CohereOptions {
    type Ext = CohereExt;
}

impl CohereOptions {
    /// Penalize tokens by how often they appeared (`frequency_penalty`,
    /// 0.0 to 1.0).
    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.shared.frequency_penalty = Some(penalty);
        self
    }

    /// Penalize tokens that appeared at all (`presence_penalty`, 0.0 to
    /// 1.0).
    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.shared.presence_penalty = Some(penalty);
        self
    }

    /// How the native API cites the documents (`citation_options.mode`).
    pub fn citation_mode(mut self, mode: CitationMode) -> Self {
        self.chat.citation_mode = Some(mode);
        self
    }

    /// The safety instruction the native API adds (`safety_mode`).
    pub fn safety_mode(mut self, mode: SafetyMode) -> Self {
        self.chat.safety_mode = Some(mode);
        self
    }

    /// The request's queue priority on the native API (`priority`): lower
    /// is served first, and 0 is the default.
    pub fn priority(mut self, priority: u32) -> Self {
        self.chat.priority = Some(priority);
        self
    }

    /// Sample from the `top_k` most likely tokens on the native API (`k`,
    /// 0 to 500).
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.chat.top_k = Some(top_k);
        self
    }

    /// Return each generated token's log probability on the native API
    /// (`logprobs`), read back as [`CohereExtras::logprobs`].
    pub fn logprobs(mut self, logprobs: bool) -> Self {
        self.chat.logprobs = Some(logprobs);
        self
    }
}

/// The fields both Cohere routes read.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CohereShared {
    /// `frequency_penalty`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// `presence_penalty`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
}

/// The fields only the native chat API reads.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CohereNative {
    /// `citation_options.mode`.
    #[serde(
        rename = "citation_options",
        serialize_with = "citation_options",
        skip_serializing_if = "Option::is_none"
    )]
    pub citation_mode: Option<CitationMode>,
    /// `safety_mode`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_mode: Option<SafetyMode>,
    /// `priority`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<u32>,
    /// `k`.
    #[serde(rename = "k", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// `logprobs`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
}

/// `{"mode": <mode>}`, the `citation_options` object.
fn citation_options<S: Serializer>(
    mode: &Option<CitationMode>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    #[derive(Serialize)]
    struct CitationOptions<'a> {
        mode: &'a Option<CitationMode>,
    }
    CitationOptions { mode }.serialize(serializer)
}

/// How the native API cites the documents a reply uses.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CitationMode {
    /// `ENABLED`.
    Enabled,
    /// `DISABLED`.
    Disabled,
    /// `FAST`.
    Fast,
    /// `ACCURATE`.
    Accurate,
    /// `OFF`.
    Off,
}

/// The safety instruction the native API adds to the prompt.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SafetyMode {
    /// `CONTEXTUAL`, the default.
    Contextual,
    /// `STRICT`.
    Strict,
    /// `OFF`.
    Off,
}

/// The fields of a Cohere reply Rig does not normalize. Every field but
/// [`Self::id`] comes from a native chat API reply; a Compatibility API
/// reply leaves it `None`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CohereExtras {
    /// `id`, on both routes.
    pub id: Option<String>,
    /// `finish_reason` as Cohere spells it (`COMPLETE`, `MAX_TOKENS`, ...).
    pub finish_reason: Option<String>,
    /// `usage.billed_units`.
    pub billed_units: Option<BilledUnits>,
    /// `usage.tokens`.
    pub tokens: Option<Tokens>,
    /// `usage.cached_tokens`: the part of `tokens.input_tokens` read from
    /// the cache, which `billed_units` leaves out.
    pub cached_tokens: Option<f64>,
    /// `message.tool_plan`: the model's plan before its tool calls.
    pub tool_plan: Option<String>,
    /// `logprobs`, when [`CohereOptions::logprobs`] asked for them.
    pub logprobs: Option<Vec<Logprob>>,
}

/// The units Cohere bills a request by.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct BilledUnits {
    /// `input_tokens`.
    pub input_tokens: Option<f64>,
    /// `output_tokens`.
    pub output_tokens: Option<f64>,
    /// `search_units`.
    pub search_units: Option<f64>,
    /// `classifications`.
    pub classifications: Option<f64>,
}

/// The tokens a request read and wrote, system overhead included.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct Tokens {
    /// `input_tokens`.
    pub input_tokens: Option<f64>,
    /// `output_tokens`.
    pub output_tokens: Option<f64>,
}

/// The log probabilities of one stretch of generated text.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct Logprob {
    /// `token_ids`.
    pub token_ids: Vec<u32>,
    /// `text`, the tokens decoded.
    pub text: Option<String>,
    /// `logprobs`, one per token.
    pub logprobs: Vec<f64>,
}

/// The value at `pointer` in `raw`, or `None` when it is absent or `null`.
fn at<T: DeserializeOwned>(raw: &Value, pointer: &str) -> Result<Option<T>, serde_json::Error> {
    match raw.pointer(pointer) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => T::deserialize(value).map(Some),
    }
}

impl ReplyExtras for CohereExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            id: at(raw, "/id")?,
            finish_reason: at(raw, "/finish_reason")?,
            billed_units: at(raw, "/usage/billed_units")?,
            tokens: at(raw, "/usage/tokens")?,
            cached_tokens: at(raw, "/usage/cached_tokens")?,
            tool_plan: at(raw, "/message/tool_plan")?,
            logprobs: at(raw, "/logprobs")?,
        })
    }
}

#[cfg(test)]
mod tests;
