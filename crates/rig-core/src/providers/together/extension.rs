//! Together AI's typed request options and reply extras
//! (<https://docs.together.ai/reference/chat-completions-1>).
//!
//! ```
//! use rig_core::completion::CompletionRequest;
//! use rig_core::providers::together::extension::{TogetherOptions};
//!
//! let options = TogetherOptions::new().top_k(40).repetition_penalty(1.1);
//! let request = CompletionRequest::new("hi").provider_option(options);
//! # let _ = request;
//! ```

use serde::Serialize;
use serde_json::{Map, Value};

use crate::completion::provider_options::reply_field;
use crate::completion::{ExtensionOptions, ProviderExtension, ReplyExtras};
use crate::message::Api;

/// Together AI's extension marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TogetherExt;

impl ProviderExtension for TogetherExt {
    const PROVIDER: &'static str = super::PROVIDER_NAME;
    type Options = TogetherOptions;
    type Extras = TogetherExtras;
}

/// Together AI's request options.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TogetherOptions {
    /// The fields every route takes.
    #[serde(rename = "*")]
    pub shared: TogetherShared,
}

/// The fields Together AI takes.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TogetherShared {
    /// Arguments to the model's chat template, such as `enable_thinking`.
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub chat_template_kwargs: Map<String, Value>,
    /// Sample from the `k` most likely tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// The minimum probability of a token, relative to the most likely one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    /// Penalizes tokens already in the prompt and the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    /// The moderation model that screens the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_model: Option<String>,
}

impl TogetherOptions {
    /// No option set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pass `key`, `value` to the model's chat template.
    pub fn chat_template_kwarg(mut self, key: impl Into<String>, value: Value) -> Self {
        self.shared.chat_template_kwargs.insert(key.into(), value);
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

    /// Set the repetition penalty.
    pub fn repetition_penalty(mut self, penalty: f64) -> Self {
        self.shared.repetition_penalty = Some(penalty);
        self
    }

    /// Screen the request with moderation model `model`.
    pub fn safety_model(mut self, model: impl Into<String>) -> Self {
        self.shared.safety_model = Some(model.into());
        self
    }
}

impl ExtensionOptions for TogetherOptions {
    type Ext = TogetherExt;
}

/// Together AI's reply fields. Each is `None` when the reply lacks it.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TogetherExtras {
    /// Warnings about the request.
    pub warnings: Option<Vec<Value>>,
    /// The first choice's reasoning.
    pub reasoning: Option<String>,
}

impl ReplyExtras for TogetherExtras {
    fn from_reply(_api: &Api, raw: &Value) -> Result<Self, serde_json::Error> {
        Ok(Self {
            warnings: reply_field(raw, "/warnings")?,
            reasoning: reply_field(raw, "/choices/0/message/reasoning")?,
        })
    }
}

#[cfg(test)]
mod tests;
