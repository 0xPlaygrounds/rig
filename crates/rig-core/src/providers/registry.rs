//! One name for a provider.
//!
//! Every gateway rig speaks is a [`Dialect`](openai::wire::Dialect) const —
//! data, not code — and every dialect's name is unique across the whole
//! tree. So a provider is nameable: [`by_name`] resolves one to the
//! configuration that talks to it, [`all`] enumerates every name this build
//! knows, and [`ModelRef`] is the `"provider:model"` pair a host stores when
//! it has nothing to override.
//!
//! ```
//! use rig_core::providers::{ModelRef, ProviderConfig};
//!
//! let reference: ModelRef = "deepseek:deepseek-chat".parse()?;
//! assert_eq!(reference.provider(), "deepseek");
//! assert_eq!(reference.model(), "deepseek-chat");
//! // The short form is the provider's default configuration: no base URL,
//! // no route override, no betas. Anything else is `ProviderConfig`.
//! assert!(matches!(reference.config(), ProviderConfig::OpenAi(_)));
//! assert_eq!(reference.to_string(), "deepseek:deepseek-chat");
//! # Ok::<(), rig_core::providers::UnknownProvider>(())
//! ```
//!
//! What a `ModelRef` does **not** do is validate the model id. rig ships
//! model-name constants, not a catalog: a provider's list changes without a
//! rig release, and refusing a model this build has not heard of would make
//! every new model wait for one. A wrong model id is the provider's 404,
//! which is where that fact lives.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::{anthropic, gemini, openai};
use crate::wire::Secret;

/// Which provider serves a request, as its own configuration.
///
/// The enumeration is over *request shapes* — the formats rig-core has a
/// [`Wire`](crate::wire::Wire) for — not over providers: a gateway is its
/// dialect, which is data, so every OpenAI-shaped and Anthropic-shaped
/// provider in the tree is one of these three variants. Adding a gateway
/// adds a `Dialect` const; only a genuinely new request format adds a
/// variant here.
///
/// Serialized under the `provider` tag, with the credential redacted by
/// [`Secret`] — so a config round-trips through a file or a scene carrying
/// everything but the key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum ProviderConfig {
    /// Every OpenAI-shaped provider, by dialect: OpenAI itself, Azure,
    /// DeepSeek, Groq, Venice, OpenRouter, xAI, ChatGPT, … over Chat
    /// Completions or Responses per the config's `route`.
    OpenAi(openai::wire::OpenAI),
    /// Anthropic's Messages API, by dialect: Anthropic itself and the
    /// `-anthropic` endpoints of zAI, MiniMax, Moonshot and Xiaomi MiMo.
    Anthropic(anthropic::wire::Anthropic),
    /// Gemini's GenerateContent.
    Gemini(gemini::Gemini),
}

impl ProviderConfig {
    /// The provider's descriptor name, as records, telemetry and
    /// [`ModelRef`] name it.
    pub fn provider(&self) -> &str {
        match self {
            Self::OpenAi(config) => config.dialect.name,
            Self::Anthropic(config) => config.dialect.name,
            Self::Gemini(_) => GEMINI,
        }
    }

    /// The base URL every request resolves against.
    pub fn base_url(&self) -> &str {
        match self {
            Self::OpenAi(config) => &config.base_url,
            Self::Anthropic(config) => &config.base_url,
            Self::Gemini(config) => &config.base_url,
        }
    }

    /// The credential this configuration would send.
    pub fn credential(&self) -> &Secret {
        match self {
            Self::OpenAi(config) => &config.api_key,
            Self::Anthropic(config) => &config.api_key,
            Self::Gemini(config) => &config.api_key,
        }
    }

    /// The same configuration carrying `secret`.
    ///
    /// A persisted configuration has no credential — [`Secret`] drops it on
    /// the way out and back in — so whoever loads one puts the resolved
    /// credential in here, and nowhere else.
    pub fn with_credential(mut self, secret: Secret) -> Self {
        match &mut self {
            Self::OpenAi(config) => config.api_key = secret,
            Self::Anthropic(config) => config.api_key = secret,
            Self::Gemini(config) => config.api_key = secret,
        }
        self
    }

    /// The environment variables this configuration reads when it is built
    /// from the environment: the credential's variable first, then the base
    /// URL's where the dialect names one.
    ///
    /// The question a host asks *before* running: "what does this scene
    /// need?" — answerable off the data alone, with no resolver, no
    /// environment access and no request.
    pub fn required_env(&self) -> impl Iterator<Item = &'static str> {
        let (api_key_env, base_url_env) = match self {
            Self::OpenAi(config) => (config.dialect.api_key_env, config.dialect.base_url_env),
            Self::Anthropic(config) => (config.dialect.api_key_env, config.dialect.base_url_env),
            Self::Gemini(_) => (gemini::API_KEY_ENV, None),
        };
        std::iter::once(api_key_env).chain(base_url_env)
    }
}

impl fmt::Display for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.provider())
    }
}

/// Gemini's provider name. It speaks one format at one host, so it has no
/// `Dialect` to carry the name for it.
const GEMINI: &str = "gemini";

/// The provider named `name`, configured the way it is by default and with
/// no credential.
///
/// `None` is "this build has no such provider"; [`all`] is the list that
/// makes that answer actionable.
pub fn by_name(name: &str) -> Option<ProviderConfig> {
    if name == GEMINI {
        return Some(ProviderConfig::Gemini(gemini::Gemini::new(
            Secret::default(),
        )));
    }
    if let Some(dialect) = openai::wire::by_name(name) {
        return Some(ProviderConfig::OpenAi(openai::wire::OpenAI::with_key(
            dialect,
            Secret::default(),
        )));
    }
    anthropic::wire::Dialect::by_name(name).map(|dialect| {
        ProviderConfig::Anthropic(anthropic::wire::Anthropic::with_dialect(
            Secret::default(),
            &dialect,
        ))
    })
}

/// Every provider name this build knows, OpenAI-shaped first, then
/// Anthropic-shaped, then Gemini.
///
/// The order is the declaration order of the dialect tables, which is what
/// makes a diagnostic listing them stable between runs.
pub fn all() -> impl Iterator<Item = &'static str> {
    openai::wire::all()
        .map(|dialect| dialect.name)
        .chain(anthropic::wire::all().map(|dialect| dialect.name))
        .chain(std::iter::once(GEMINI))
}

/// A provider name this build does not know, with the names it does.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnknownProvider {
    /// The string named no provider at all.
    #[error(
        "`{reference}` is not a `provider:model` reference: name the provider before the colon \
         (for example `openai:gpt-5.2`)"
    )]
    Unqualified {
        /// What was parsed.
        reference: String,
    },
    /// The provider is not one this build ships.
    #[error("unknown provider `{provider}`; this build knows {known}")]
    Provider {
        /// The name that resolved to nothing.
        provider: String,
        /// Every name that would have resolved, comma-separated.
        known: String,
    },
    /// The model half was empty.
    #[error("`{reference}` names the provider `{provider}` but no model")]
    NoModel {
        /// What was parsed.
        reference: String,
        /// The provider half, which did resolve.
        provider: String,
    },
}

/// A provider and a model, by name: `"deepseek:deepseek-chat"`.
///
/// The short form of "which model serves this". It resolves to the
/// provider's default configuration, so it says nothing about a base URL, a
/// route, an api-version or a beta — a host that needs one of those stores
/// a [`ProviderConfig`] instead. Serialized as the one string it is, which
/// is the point: a config file or a scene that overrides nothing should not
/// have to spell out a provider object.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ModelRef {
    provider: String,
    model: String,
}

impl ModelRef {
    /// The reference to `model` on `provider`, if this build knows the
    /// provider.
    pub fn new(provider: &str, model: impl Into<String>) -> Result<Self, UnknownProvider> {
        if by_name(provider).is_none() {
            return Err(UnknownProvider::Provider {
                provider: provider.to_owned(),
                known: all().collect::<Vec<_>>().join(", "),
            });
        }
        Ok(Self {
            provider: provider.to_owned(),
            model: model.into(),
        })
    }

    /// The provider's name.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// The model id, verbatim. rig does not check it against a catalog: a
    /// provider adds models without a rig release, so a model this build has
    /// not heard of is the provider's 404, not a parse error.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The provider's default configuration, with no credential.
    ///
    /// Infallible: a `ModelRef` exists only for a provider that resolved.
    pub fn config(&self) -> ProviderConfig {
        by_name(&self.provider).unwrap_or_else(|| {
            unreachable!("a `ModelRef` is only built for a provider this build knows")
        })
    }
}

impl FromStr for ModelRef {
    type Err = UnknownProvider;

    fn from_str(reference: &str) -> Result<Self, Self::Err> {
        let Some((provider, model)) = reference.split_once(':') else {
            return Err(UnknownProvider::Unqualified {
                reference: reference.to_owned(),
            });
        };
        if model.is_empty() {
            return Err(UnknownProvider::NoModel {
                reference: reference.to_owned(),
                provider: provider.to_owned(),
            });
        }
        Self::new(provider, model)
    }
}

impl TryFrom<String> for ModelRef {
    type Error = UnknownProvider;

    fn try_from(reference: String) -> Result<Self, Self::Error> {
        reference.parse()
    }
}

impl From<ModelRef> for String {
    fn from(reference: ModelRef) -> Self {
        reference.to_string()
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.provider, self.model)
    }
}

#[cfg(test)]
mod tests;
