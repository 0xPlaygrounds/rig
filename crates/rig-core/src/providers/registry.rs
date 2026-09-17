//! One name for a provider.
//!
//! Every gateway rig speaks is a [`Dialect`](openai::wire::Dialect) const —
//! data, not code — and every dialect's name is unique across the whole
//! tree. So a provider is nameable: [`by_name`] resolves one to the
//! configuration that talks to it, [`all`] enumerates every name this build
//! knows, and [`ProviderRef`] is the `"provider:model"` pair a host stores when
//! it has nothing to override.
//!
//! ```
//! use rig_core::providers::{ProviderConfig, ProviderRef};
//!
//! let reference: ProviderRef = "deepseek:deepseek-chat".parse()?;
//! assert_eq!(reference.provider(), "deepseek");
//! assert_eq!(reference.model(), "deepseek-chat");
//! // A vendor with two doors is asked which: `"zai:glm-4.6"` refuses and
//! // lists them, rather than picking a host for you.
//! assert!("zai:glm-4.6".parse::<ProviderRef>().is_err());
//! let messages: ProviderRef = "zai/anthropic:glm-4.6".parse()?;
//! assert_eq!(messages.provider(), "zai");
//! // The short form is the provider's default configuration: no base URL,
//! // no route override, no betas. Anything else is `ProviderConfig`.
//! assert!(matches!(reference.config(), ProviderConfig::OpenAi(_)));
//! assert_eq!(reference.to_string(), "deepseek:deepseek-chat");
//! # Ok::<(), rig_core::providers::UnknownProvider>(())
//! ```
//!
//! What a `ProviderRef` does **not** do is validate the model id. rig ships
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
    /// [`ProviderRef`] name it.
    pub fn provider(&self) -> &str {
        match self {
            Self::OpenAi(config) => config.dialect.name,
            Self::Anthropic(config) => config.dialect.name,
            Self::Gemini(_) => GEMINI,
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
        let (api_key_env, base_url_env) = self.env();
        std::iter::once(api_key_env).chain(base_url_env)
    }

    /// The credential's variable and the base URL's, as the dialect names
    /// them. Both are `&'static`, which is what lets a caller holding a
    /// temporary configuration still hand out an iterator.
    fn env(&self) -> (&'static str, Option<&'static str>) {
        match self {
            Self::OpenAi(config) => (config.dialect.api_key_env, config.dialect.base_url_env),
            Self::Anthropic(config) => (config.dialect.api_key_env, config.dialect.base_url_env),
            Self::Gemini(_) => (gemini::API_KEY_ENV, None),
        }
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

/// Gemini's vendor name. It speaks one format at one host, so it has no
/// `Dialect` to carry the name for it.
const GEMINI: &str = "gemini";

/// Which request format an endpoint speaks.
///
/// Orthogonal to *whose* endpoint it is: a vendor may front one door or
/// several, and z.ai, MiniMax, Moonshot and Xiaomi MiMo each front two —
/// an OpenAI-shaped one and an Anthropic-shaped one, at different hosts,
/// under the same vendor name. Keeping the two facts apart is why nothing
/// in this tree is called `zai-anthropic`: that would name a vendor that
/// does not exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    /// OpenAI's shape — Chat Completions or Responses, per the
    /// configuration's `route`.
    OpenAi,
    /// Anthropic's Messages shape.
    Anthropic,
    /// Gemini's GenerateContent.
    Gemini,
}

impl Format {
    /// How the format is written in a reference's qualifier
    /// (`zai/anthropic`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
        }
    }

    /// The format a qualifier names.
    pub fn by_name(name: &str) -> Option<Self> {
        [Self::OpenAi, Self::Anthropic, Self::Gemini]
            .into_iter()
            .find(|format| format.as_str() == name)
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One provider: a vendor's endpoint in one format.
///
/// The identity is the *pair*. The only way to obtain one is [`resolve`],
/// [`endpoints`] or [`all`], so holding a `ProviderId` means the pair
/// exists, and [`Self::config`] is infallible without an `unwrap`, an
/// `unreachable!` or an `Option` that cannot be `None` — the invariant
/// lives in the type rather than in a comment asking the reader to trust
/// the constructor.
#[derive(Debug, Clone, Copy)]
pub struct ProviderId(Endpoint);

/// The dialect an id names, and with it the format. Both halves are
/// `&'static`: a dialect is a const, so an id borrows it rather than
/// copying a configuration out of it.
#[derive(Debug, Clone, Copy)]
enum Endpoint {
    OpenAi(&'static openai::wire::Dialect),
    Anthropic(&'static anthropic::wire::Dialect),
    Gemini,
}

impl ProviderId {
    /// The vendor's name: `"zai"` for both of z.ai's doors.
    pub fn vendor(&self) -> &'static str {
        match self.0 {
            Endpoint::OpenAi(dialect) => dialect.name,
            Endpoint::Anthropic(dialect) => dialect.name,
            Endpoint::Gemini => GEMINI,
        }
    }

    /// Which format this door speaks.
    pub fn format(&self) -> Format {
        match self.0 {
            Endpoint::OpenAi(_) => Format::OpenAi,
            Endpoint::Anthropic(_) => Format::Anthropic,
            Endpoint::Gemini => Format::Gemini,
        }
    }

    /// How a reference spells this provider: the vendor alone when it has
    /// one door, `vendor/format` when it has several.
    ///
    /// The short spelling is not a convenience — it is the whole name for
    /// every vendor but four, and writing `openai/openai` would be noise.
    pub fn spelling(&self) -> String {
        if endpoints(self.vendor()).count() > 1 {
            format!("{}/{}", self.vendor(), self.format())
        } else {
            self.vendor().to_owned()
        }
    }

    /// The provider configured the way it is by default, with no
    /// credential.
    ///
    /// Infallible by construction — see the type's documentation.
    pub fn config(&self) -> ProviderConfig {
        match self.0 {
            Endpoint::OpenAi(dialect) => {
                ProviderConfig::OpenAi(openai::wire::OpenAI::with_key(dialect, Secret::default()))
            }
            Endpoint::Anthropic(dialect) => ProviderConfig::Anthropic(
                anthropic::wire::Anthropic::with_dialect(Secret::default(), dialect),
            ),
            Endpoint::Gemini => ProviderConfig::Gemini(gemini::Gemini::new(Secret::default())),
        }
    }
}

impl PartialEq for ProviderId {
    fn eq(&self, other: &Self) -> bool {
        self.vendor() == other.vendor() && self.format() == other.format()
    }
}

impl Eq for ProviderId {}

impl std::hash::Hash for ProviderId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.vendor().hash(state);
        self.format().hash(state);
    }
}

/// Every provider this build knows: OpenAI-shaped first, then
/// Anthropic-shaped, then Gemini, in the dialect tables' declaration
/// order, so a diagnostic listing them is stable between runs.
pub fn all() -> impl Iterator<Item = ProviderId> {
    openai::wire::all()
        .map(|dialect| ProviderId(Endpoint::OpenAi(dialect)))
        .chain(anthropic::wire::all().map(|dialect| ProviderId(Endpoint::Anthropic(dialect))))
        .chain(std::iter::once(ProviderId(Endpoint::Gemini)))
}

/// Every vendor name this build knows, each once however many doors it
/// fronts.
pub fn vendors() -> impl Iterator<Item = &'static str> {
    let mut seen: Vec<&'static str> = Vec::new();
    all().filter_map(move |id| {
        let vendor = id.vendor();
        (!seen.contains(&vendor)).then(|| {
            seen.push(vendor);
            vendor
        })
    })
}

/// Every door `vendor` fronts, in format order.
pub fn endpoints(vendor: &str) -> impl Iterator<Item = ProviderId> + use<> {
    let vendor = vendor.to_owned();
    all().filter(move |id| id.vendor() == vendor)
}

/// The provider `spec` names: a vendor (`"deepseek"`) or a vendor and a
/// format (`"zai/anthropic"`).
///
/// A vendor with one door needs no qualifier. A vendor with several and no
/// qualifier is [`UnknownProvider::Ambiguous`] listing its doors — never a
/// silent pick, because the doors are different hosts and a binding that
/// quietly talked to the wrong one is worse than one that refuses.
pub fn resolve(spec: &str) -> Result<ProviderId, UnknownProvider> {
    let (vendor, format) = match spec.split_once('/') {
        Some((vendor, format)) => (vendor, Some(format)),
        None => (spec, None),
    };
    let doors: Vec<ProviderId> = endpoints(vendor).collect();
    if doors.is_empty() {
        return Err(UnknownProvider::Vendor {
            vendor: vendor.to_owned(),
            known: vendors().collect::<Vec<_>>().join(", "),
        });
    }
    match format {
        Some(format) => {
            let format = Format::by_name(format).ok_or_else(|| UnknownProvider::Format {
                vendor: vendor.to_owned(),
                format: format.to_owned(),
                known: spellings(&doors),
            })?;
            doors
                .into_iter()
                .find(|id| id.format() == format)
                .ok_or_else(|| UnknownProvider::Format {
                    vendor: vendor.to_owned(),
                    format: format.as_str().to_owned(),
                    known: spellings(&endpoints(vendor).collect::<Vec<_>>()),
                })
        }
        None => match doors.as_slice() {
            [only] => Ok(*only),
            several => Err(UnknownProvider::Ambiguous {
                vendor: vendor.to_owned(),
                doors: spellings(several),
            }),
        },
    }
}

/// How a set of providers is written, for a refusal that teaches.
fn spellings(ids: &[ProviderId]) -> String {
    ids.iter()
        .map(ProviderId::spelling)
        .collect::<Vec<_>>()
        .join(", ")
}

/// A reference this build cannot resolve, with what it could have.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnknownProvider {
    /// The string named no vendor and no model.
    #[error(
        "`{reference}` is not a `provider:model` reference: name the provider before the colon \
         (for example `openai:gpt-5.2`, or `zai/anthropic:glm-4.6`)"
    )]
    Unqualified {
        /// What was parsed.
        reference: String,
    },
    /// The vendor is not one this build ships.
    #[error("unknown provider `{vendor}`; this build knows {known}")]
    Vendor {
        /// The name that resolved to nothing.
        vendor: String,
        /// Every vendor that would have resolved.
        known: String,
    },
    /// The vendor fronts several endpoints and the reference named none of
    /// them.
    #[error("`{vendor}` fronts more than one endpoint; name which: {doors}")]
    Ambiguous {
        /// The vendor.
        vendor: String,
        /// Its doors, as they are written.
        doors: String,
    },
    /// The vendor does not front an endpoint in that format.
    #[error("`{vendor}` has no `{format}` endpoint; it fronts {known}")]
    Format {
        /// The vendor.
        vendor: String,
        /// The format the reference asked for.
        format: String,
        /// The doors it does front.
        known: String,
    },
}

/// Which provider and model serve something, in either of the two forms a
/// host writes.
///
/// `"deepseek:deepseek-chat"` when nothing is overridden — one line in a
/// config file or a scene — and the configuration written out when a base
/// URL, a route, a beta or an api-version has to be named. One type, not
/// two: a name is a provider plus a model, which is what the long form is
/// as well.
///
/// Written and read by *shape*, symmetrically and by hand: the short form
/// is the string it parsed from, the long form is a map of `config` and
/// `model`, and the reader decides which by what it is given rather than
/// by trying variants. `#[serde(untagged)]` would read the same documents
/// but, once both variants have failed, could only report "data did not
/// match any variant".
///
/// What neither form validates is the model id. rig ships model-name
/// constants, not a catalog: a provider adds models without a rig release,
/// so an id this build has not heard of is the provider's 404, which is
/// where that fact lives.
#[derive(Debug, Clone, PartialEq)]
pub enum ProviderRef {
    /// A provider by name, on its default configuration.
    Named {
        /// The provider.
        provider: ProviderId,
        /// The model id.
        model: String,
    },
    /// A configuration written out, with the model beside it.
    Configured {
        /// The provider, configured.
        config: ProviderConfig,
        /// The model id.
        model: String,
    },
}

impl ProviderRef {
    /// The reference to `model` on the provider `spec` names — a vendor,
    /// or a vendor and a format (`"zai/anthropic"`).
    pub fn named(spec: &str, model: impl Into<String>) -> Result<Self, UnknownProvider> {
        Ok(Self::Named {
            provider: resolve(spec)?,
            model: model.into(),
        })
    }

    /// The vendor's name. Both of a two-door vendor's endpoints report the
    /// same one, which is the point: `zai` is one company.
    pub fn provider(&self) -> &str {
        match self {
            Self::Named { provider, .. } => provider.vendor(),
            Self::Configured { config, .. } => config.provider(),
        }
    }

    /// How this reference is written: the provider's spelling and the
    /// model, which is exactly what it parses from and serializes as.
    pub fn spelling(&self) -> String {
        match self {
            Self::Named { provider, model } => format!("{}:{model}", provider.spelling()),
            Self::Configured { config, model } => format!("{}:{model}", config.provider()),
        }
    }

    /// The model id, verbatim.
    pub fn model(&self) -> &str {
        match self {
            Self::Named { model, .. } | Self::Configured { model, .. } => model,
        }
    }

    /// The configuration to talk to it with, credential-less either way.
    pub fn config(&self) -> ProviderConfig {
        match self {
            Self::Named { provider, .. } => provider.config(),
            Self::Configured { config, .. } => config.clone(),
        }
    }

    /// The environment this reference reads when built from the
    /// environment.
    ///
    /// The variables are `&'static`, so the configuration this reads them
    /// off is a temporary and nothing is allocated.
    pub fn required_env(&self) -> impl Iterator<Item = &'static str> {
        let (api_key_env, base_url_env) = self.config().env();
        std::iter::once(api_key_env).chain(base_url_env)
    }
}

impl fmt::Display for ProviderRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.spelling())
    }
}

impl FromStr for ProviderRef {
    type Err = UnknownProvider;

    fn from_str(reference: &str) -> Result<Self, Self::Err> {
        // Both halves have to be there: a bare model name has no provider
        // to resolve, and a bare provider names no model to send.
        match reference.split_once(':') {
            Some((provider, model)) if !model.is_empty() => Self::named(provider, model),
            _ => Err(UnknownProvider::Unqualified {
                reference: reference.to_owned(),
            }),
        }
    }
}

impl Serialize for ProviderRef {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            // The one string it parsed from, which is also its `Display`.
            Self::Named { .. } => serializer.serialize_str(&self.spelling()),
            Self::Configured { config, model } => {
                use serde::ser::SerializeStruct;
                let mut map = serializer.serialize_struct("ProviderRef", 2)?;
                map.serialize_field("config", config)?;
                map.serialize_field("model", model)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ProviderRef {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// The long form, named so serde attributes an error to the value
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
                reference.parse().map_err(E::custom)
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

#[cfg(test)]
mod tests;
