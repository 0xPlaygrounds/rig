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
use crate::driver::Bind;
use crate::http_client::BoxedHttpClient;
use crate::serve::{ErasedHandler, adapters::CompletionAdapter};
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

    /// This configuration's completion handler for `model`, advertised
    /// under `label`, over `transport`.
    ///
    /// The one place a named provider becomes something that answers: a
    /// host that stores providers as data (rig-ecs's bindings, a config
    /// file) builds them here rather than keeping its own table of which
    /// config yields which wire.
    pub fn completion_adapter(
        self,
        model: &str,
        label: &str,
        transport: BoxedHttpClient,
    ) -> ErasedHandler {
        match self {
            Self::OpenAi(config) => ErasedHandler::new(CompletionAdapter::new(
                label,
                config.bind(transport).completion(model),
            )),
            Self::Anthropic(config) => ErasedHandler::new(CompletionAdapter::new(
                label,
                config.bind(transport).completion(model),
            )),
            Self::Gemini(config) => ErasedHandler::new(CompletionAdapter::new(
                label,
                config.bind(transport).completion(model),
            )),
        }
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

/// A provider this build knows, by name.
///
/// The type is the proof: the only way to obtain one is [`by_name`] or
/// [`all`], so *holding* a `ProviderId` means the name resolves, and
/// [`Self::config`] is infallible without an `unwrap`, an `unreachable!` or
/// an `Option` that cannot be `None`. The invariant lives in the type
/// rather than in a comment asking the reader to trust the constructor.
///
/// `Copy`, because it is a `&'static` dialect and a name — equality and
/// hashing are by name, which is what a caller means by "the same
/// provider".
#[derive(Debug, Clone, Copy)]
pub struct ProviderId(Shape);

/// Which format's table the name came out of, and the entry it named. Both
/// halves are `&'static`: a dialect is a const, so an id borrows it rather
/// than copying a configuration out of it.
#[derive(Debug, Clone, Copy)]
enum Shape {
    OpenAi(&'static openai::wire::Dialect),
    Anthropic(&'static anthropic::wire::Dialect),
    Gemini,
}

impl ProviderId {
    /// The name, as the registry spells it.
    pub fn name(&self) -> &'static str {
        match self.0 {
            Shape::OpenAi(dialect) => dialect.name,
            Shape::Anthropic(dialect) => dialect.name,
            Shape::Gemini => GEMINI,
        }
    }

    /// The provider configured the way it is by default, with no
    /// credential.
    ///
    /// Infallible by construction — see the type's documentation.
    pub fn config(&self) -> ProviderConfig {
        match self.0 {
            Shape::OpenAi(dialect) => {
                ProviderConfig::OpenAi(openai::wire::OpenAI::with_key(dialect, Secret::default()))
            }
            Shape::Anthropic(dialect) => ProviderConfig::Anthropic(
                anthropic::wire::Anthropic::with_dialect(Secret::default(), dialect),
            ),
            Shape::Gemini => ProviderConfig::Gemini(gemini::Gemini::new(Secret::default())),
        }
    }

    /// The environment this provider reads when it is built from the
    /// environment.
    pub fn required_env(&self) -> Vec<&'static str> {
        self.config().required_env().collect()
    }
}

impl PartialEq for ProviderId {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for ProviderId {}

impl std::hash::Hash for ProviderId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The provider named `name`.
///
/// `None` is "this build has no such provider"; [`all`] is the list that
/// makes that answer actionable.
pub fn by_name(name: &str) -> Option<ProviderId> {
    if name == GEMINI {
        return Some(ProviderId(Shape::Gemini));
    }
    if let Some(dialect) = openai::wire::by_name(name) {
        return Some(ProviderId(Shape::OpenAi(dialect)));
    }
    anthropic::wire::all()
        .find(|dialect| dialect.name == name)
        .map(|dialect| ProviderId(Shape::Anthropic(dialect)))
}

/// Every provider this build knows, OpenAI-shaped first, then
/// Anthropic-shaped, then Gemini.
///
/// The order is the declaration order of the dialect tables, so a
/// diagnostic listing them is stable between runs.
pub fn all() -> impl Iterator<Item = ProviderId> {
    openai::wire::all()
        .map(|dialect| ProviderId(Shape::OpenAi(dialect)))
        .chain(anthropic::wire::all().map(|dialect| ProviderId(Shape::Anthropic(dialect))))
        .chain(std::iter::once(ProviderId(Shape::Gemini)))
}

/// Every known name, for a refusal that tells the caller what would have
/// worked.
fn known_names() -> String {
    all().map(|id| id.name()).collect::<Vec<_>>().join(", ")
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
    /// The provider, as a name this build is known to have. Not a config
    /// snapshot — equality compares the name the user wrote — and not a
    /// bare `String` either, so reading the configuration back cannot fail.
    provider: ProviderId,
    model: String,
}

impl ModelRef {
    /// The reference to `model` on `provider`, if this build knows the
    /// provider.
    pub fn new(provider: &str, model: impl Into<String>) -> Result<Self, UnknownProvider> {
        let provider = by_name(provider).ok_or_else(|| UnknownProvider::Provider {
            provider: provider.to_owned(),
            known: known_names(),
        })?;
        Ok(Self {
            provider,
            model: model.into(),
        })
    }

    /// The provider.
    pub fn id(&self) -> ProviderId {
        self.provider
    }

    /// The provider's name.
    pub fn provider(&self) -> &'static str {
        self.provider.name()
    }

    /// The model id, verbatim. rig does not check it against a catalog: a
    /// provider adds models without a rig release, so a model this build has
    /// not heard of is the provider's 404, not a parse error.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The provider's default configuration, with no credential.
    pub fn config(&self) -> ProviderConfig {
        self.provider.config()
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
        write!(f, "{}:{}", self.provider.name(), self.model)
    }
}

/// Which provider and model serve something, in either of the two forms a
/// host writes: the name, or the configuration.
///
/// One field, two spellings, and the serialized form is the reason both
/// exist: a host that overrides nothing writes `"deepseek:deepseek-chat"`
/// and a host that needs a base URL, a route, a beta or an api-version
/// writes the configuration out.
///
/// Read by the *shape of the input* rather than by trying variants in turn:
/// a string is a name and a map is a configuration, decided in
/// `deserialize` before either is parsed. `#[serde(untagged)]` would read
/// the same documents, but it reports a wrong field as "data did not match
/// any variant", where this reports the field — which is the whole reason
/// the untyped `extra_params` bag was deleted from the binding in the
/// first place.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ProviderRef {
    /// `"provider:model"` — the provider's default configuration.
    Named(ModelRef),
    /// A configuration written out, with the model beside it.
    Configured {
        /// The provider, configured.
        config: ProviderConfig,
        /// The model id.
        model: String,
    },
}

impl ProviderRef {
    /// The provider's name.
    pub fn provider(&self) -> &str {
        match self {
            Self::Named(reference) => reference.provider(),
            Self::Configured { config, .. } => config.provider(),
        }
    }

    /// The model id.
    pub fn model(&self) -> &str {
        match self {
            Self::Named(reference) => reference.model(),
            Self::Configured { model, .. } => model,
        }
    }

    /// The configuration to talk to it with, credential-less either way.
    pub fn config(&self) -> ProviderConfig {
        match self {
            Self::Named(reference) => reference.config(),
            Self::Configured { config, .. } => config.clone(),
        }
    }

    /// The environment this reference reads when built from the
    /// environment.
    pub fn required_env(&self) -> Vec<&'static str> {
        self.config().required_env().collect()
    }
}

impl<'de> Deserialize<'de> for ProviderRef {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// The long form, named so serde attributes an error to the field
        /// that was wrong rather than to the union.
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Configured {
            config: ProviderConfig,
            model: String,
        }

        struct EitherForm;

        impl<'de> serde::de::Visitor<'de> for EitherForm {
            type Value = ProviderRef;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a `provider:model` string, or a map of `config` and `model`")
            }

            fn visit_str<E: serde::de::Error>(self, reference: &str) -> Result<Self::Value, E> {
                reference.parse().map(ProviderRef::Named).map_err(E::custom)
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                map: M,
            ) -> Result<Self::Value, M::Error> {
                let Configured { config, model } =
                    Deserialize::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
                Ok(ProviderRef::Configured { config, model })
            }
        }

        deserializer.deserialize_any(EitherForm)
    }
}

impl From<ModelRef> for ProviderRef {
    fn from(reference: ModelRef) -> Self {
        Self::Named(reference)
    }
}

impl fmt::Display for ProviderRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.provider(), self.model())
    }
}

#[cfg(test)]
mod tests;
