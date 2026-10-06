//! OpenAI's typed request options and reply extras. The Chat Completions
//! section is sent only on that route.
//!
//! ```
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::openai::extension::{ChatOptions, OpenAi, OpenAiOptions};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let options = OpenAiOptions::new().chat(ChatOptions::new().logprobs(true).top_logprobs(3));
//! let request = CompletionRequest::new("hi")
//!     .provider_options(ProviderOptions::new().with::<OpenAi>(&options)?);
//! # let _ = request;
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// OpenAI's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpenAi;

impl ProviderExtension for OpenAi {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = OpenAiOptions;
    type Extras = OpenAiExtras;
}

/// OpenAI's request options, by route.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OpenAiOptions {
    /// Sent on Chat Completions only.
    #[serde(rename = "openai.chat")]
    pub chat: ChatOptions,
}

impl OpenAiOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Send `chat` on Chat Completions.
    pub fn chat(mut self, chat: ChatOptions) -> Self {
        self.chat = chat;
        self
    }
}

impl ExtensionOptions for OpenAiOptions {}

/// The fields only Chat Completions takes
/// (<https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create>).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ChatOptions {
    /// Bias per token id, from -100 to 100.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub logit_bias: BTreeMap<String, i32>,
    /// Predicted output, which speeds up a regeneration that mostly repeats
    /// it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<Prediction>,
    /// Whether the reply carries each output token's log probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// How many of the most likely tokens to report per position, 0 to 20.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    /// Penalizes tokens by how often they already appear, -2.0 to 2.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// Penalizes tokens that already appear, -2.0 to 2.0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// The output modalities, for models that also speak.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub modalities: Vec<Modality>,
    /// The spoken output, when `modalities` holds audio.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioOutput>,
    /// Web search, for the search models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,
}

impl ChatOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bias token id `token` by `bias`.
    pub fn logit_bias(mut self, token: u32, bias: i32) -> Self {
        self.logit_bias.insert(token.to_string(), bias);
        self
    }

    /// Predict the output as `content`.
    pub fn prediction(mut self, content: impl Into<String>) -> Self {
        self.prediction = Some(Prediction::content(content));
        self
    }

    /// Whether the reply carries log probabilities.
    pub fn logprobs(mut self, logprobs: bool) -> Self {
        self.logprobs = Some(logprobs);
        self
    }

    /// Report the `count` most likely tokens per position.
    pub fn top_logprobs(mut self, count: u8) -> Self {
        self.top_logprobs = Some(count);
        self
    }

    /// Set the frequency penalty.
    pub fn frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    /// Set the presence penalty.
    pub fn presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Answer in `modalities`.
    pub fn modalities(mut self, modalities: impl IntoIterator<Item = Modality>) -> Self {
        self.modalities = modalities.into_iter().collect();
        self
    }

    /// Speak with `voice` in `format`.
    pub fn audio(mut self, voice: impl Into<String>, format: AudioFormat) -> Self {
        self.audio = Some(AudioOutput {
            voice: voice.into(),
            format,
        });
        self
    }

    /// Search the web with `options`.
    pub fn web_search_options(mut self, options: WebSearchOptions) -> Self {
        self.web_search_options = Some(options);
        self
    }
}

/// Predicted output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Prediction {
    /// Text the output is expected to mostly repeat.
    Content {
        /// The predicted text.
        content: String,
    },
}

impl Prediction {
    /// A prediction of `content`.
    pub fn content(content: impl Into<String>) -> Self {
        Self::Content {
            content: content.into(),
        }
    }
}

/// An output modality.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Text.
    Text,
    /// Speech.
    Audio,
}

/// The spoken output's voice and encoding.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AudioOutput {
    /// The voice, such as `alloy`.
    pub voice: String,
    /// The encoding.
    pub format: AudioFormat,
}

/// A spoken output's encoding.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    /// WAV.
    Wav,
    /// AAC.
    Aac,
    /// MP3.
    Mp3,
    /// FLAC.
    Flac,
    /// Opus.
    Opus,
    /// Raw 16-bit PCM.
    Pcm16,
}

/// How a search model searches the web.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct WebSearchOptions {
    /// How much search context goes into the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<SearchContextSize>,
    /// Where the user is, to localize results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocation>,
}

impl WebSearchOptions {
    /// The defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Put `size` search context into the prompt.
    pub fn search_context_size(mut self, size: SearchContextSize) -> Self {
        self.search_context_size = Some(size);
        self
    }

    /// Localize results to `location`.
    pub fn user_location(mut self, location: ApproximateLocation) -> Self {
        self.user_location = Some(UserLocation::Approximate {
            approximate: location,
        });
        self
    }
}

/// How much search context goes into the prompt.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchContextSize {
    /// Least context, cheapest.
    Low,
    /// The default.
    Medium,
    /// Most context.
    High,
}

/// Where the user is.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum UserLocation {
    /// An approximate location.
    Approximate {
        /// The location.
        approximate: ApproximateLocation,
    },
}

/// An approximate location; every field is optional.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ApproximateLocation {
    /// The city, such as `San Francisco`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    /// The two-letter ISO country code, such as `US`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// The region, such as `California`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// The IANA time zone, such as `America/Los_Angeles`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

impl ApproximateLocation {
    /// No field set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the city.
    pub fn city(mut self, city: impl Into<String>) -> Self {
        self.city = Some(city.into());
        self
    }

    /// Set the two-letter country code.
    pub fn country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    /// Set the region.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set the IANA time zone.
    pub fn timezone(mut self, timezone: impl Into<String>) -> Self {
        self.timezone = Some(timezone.into());
        self
    }
}

/// OpenAI's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OpenAiExtras {
    /// The service tier that served the request. Both routes.
    pub service_tier: Option<String>,
    /// The backend configuration's fingerprint. Chat only.
    pub system_fingerprint: Option<String>,
    /// Prompt token details. Chat only.
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Completion token details. Chat only.
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// The message's annotations, such as URL citations. Chat only.
    pub annotations: Option<Vec<Value>>,
}

/// Where a Chat reply's prompt tokens went.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct PromptTokensDetails {
    /// Tokens read from the prompt cache.
    #[serde(default)]
    pub cached_tokens: Option<u64>,
    /// Audio input tokens.
    #[serde(default)]
    pub audio_tokens: Option<u64>,
    /// Tokens written to the prompt cache.
    #[serde(default)]
    pub cache_write_tokens: Option<u64>,
}

/// Where a Chat reply's completion tokens went.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct CompletionTokensDetails {
    /// Tokens spent reasoning.
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
    /// Audio output tokens.
    #[serde(default)]
    pub audio_tokens: Option<u64>,
    /// Predicted tokens that appeared in the output.
    #[serde(default)]
    pub accepted_prediction_tokens: Option<u64>,
    /// Predicted tokens that did not, billed as output.
    #[serde(default)]
    pub rejected_prediction_tokens: Option<u64>,
}

impl ReplyExtras for OpenAiExtras {
    fn from_reply(api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        let service_tier = reply_field(raw, "/service_tier")?;
        if api.as_str() != "openai.chat" {
            return Ok(Self {
                service_tier,
                ..Self::default()
            });
        }
        Ok(Self {
            service_tier,
            system_fingerprint: reply_field(raw, "/system_fingerprint")?,
            prompt_tokens_details: reply_field(raw, "/usage/prompt_tokens_details")?,
            completion_tokens_details: reply_field(raw, "/usage/completion_tokens_details")?,
            annotations: reply_field(raw, "/choices/0/message/annotations")?,
        })
    }
}

#[cfg(test)]
mod tests;
