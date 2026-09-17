//! Naming a provider as data: one vocabulary for "which provider, which
//! protocol, which model".
//!
//! A host that stores a model choice in a config file, a scene or a database
//! row needs two different things, and conflating them is what this module
//! exists to prevent:
//!
//! - a **registered selection** — [`ProviderId`], a validated
//!   `(vendor, format)` pair such as `deepseek/openai` or `zai/anthropic`.
//!   It names a provider this build ships and nothing else: its preset
//!   configuration follows the provider's defaults today.
//! - a **configuration** — [`ProviderConfig`], one of the crate's existing
//!   provider configuration types. It carries explicit choices: the host, the
//!   route, the API version, the beta flags, where system instructions go.
//!
//! [`ProviderRef`] is either of those plus a model identifier, and is the one
//! type a host persists.
//!
//! # Syntax
//!
//! A reference reads as `vendor[/format]:model`. Shorthand is an *input*
//! convenience: a bare vendor resolves when this build registers exactly one
//! protocol family for it, and every write is qualified.
//!
//! ```
//! use rig_core::providers::registry::{Format, ProviderRef};
//!
//! // Shorthand in, qualified out.
//! let reference = ProviderRef::parse("deepseek:deepseek-chat")?;
//! assert_eq!(reference.to_string(), "deepseek/openai:deepseek-chat");
//!
//! // A vendor with two doors must name one.
//! assert!(ProviderRef::parse("zai:glm-4.6").is_err());
//! let zai = ProviderRef::parse("zai/anthropic:glm-4.6")?;
//! assert_eq!(zai.id().format(), Format::Anthropic);
//!
//! // A model identifier may contain `:` and `/`; only the first `:` splits.
//! let local = ProviderRef::parse("llamacpp/openai:qwen3:4b")?;
//! assert_eq!(local.model, "qwen3:4b");
//! # Ok::<(), rig_core::providers::registry::RefError>(())
//! ```
//!
//! # Persistence
//!
//! A registered reference serializes as its canonical string; an explicit
//! configuration serializes as a structured object, losing nothing but the
//! credential (see [`Secret`]). Deserialization dispatches on the shape, so
//! the serialized form requires a self-describing format — JSON in this
//! repository.
//!
//! Canonicalization is intentional: reading `deepseek:deepseek-chat` and
//! saving it writes `deepseek/openai:deepseek-chat`. A round trip preserves
//! meaning, not input bytes. The guarantee a qualified write buys is narrow
//! and worth stating exactly: **registering another protocol family for a
//! vendor cannot make an already-written qualified reference ambiguous.** It
//! says nothing about a removed endpoint, a renamed identifier, a changed
//! preset default or a retired model; a short reference selects a preset, it
//! does not freeze one.
//!
//! `copilot/openai` takes an already-exchanged Copilot session token, not a
//! GitHub OAuth token. Its preset derives the endpoint from that token, and
//! [`ProviderConfig::completion_handler`] uses Copilot's model-dependent
//! routing and editor headers. An explicit configuration preserves its host,
//! route and instruction placement rather than replacing them with the
//! preset. Token exchange remains the host's responsibility.
//!
//! # Registry, not telemetry
//!
//! `vendor/format` is registry syntax. The provider string a record and a
//! span carry is still the dialect's own `name`
//! ([`Wire::name`]) — `deepseek`, `gcp.gemini`,
//! `azure.openai`. The two have different contracts and neither renames the
//! other.

use std::fmt;
use std::hash::{Hash, Hasher};

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::completion::CompletionModel;
use crate::driver::{Bind, Bound};
use crate::http_client::BoxedHttpClient;
use crate::operation::Completion;
use crate::providers::{anthropic, copilot, gemini, openai};
use crate::serve::ErasedHandler;
use crate::serve::adapters::CompletionAdapter;
use crate::wire::{HasCompletion, Secret, Wire};

/// A protocol family: the request grammar a provider speaks, and so which of
/// this crate's configuration types describes it.
///
/// A family is not a route. `openai` covers both Chat Completions and the
/// Responses endpoint, because both are the same configuration
/// ([`openai::wire::OpenAI`]) and the endpoint is that configuration's own
/// `route` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Format {
    /// OpenAI's grammar: Chat Completions and the Responses endpoint.
    OpenAi,
    /// Anthropic's Messages grammar.
    Anthropic,
    /// Gemini's GenerateContent grammar.
    Gemini,
}

impl Format {
    /// Every protocol family, in registration order. The one table: the
    /// serialized spelling, the parser and this list cannot disagree.
    pub const ALL: [Format; 3] = [Format::OpenAi, Format::Anthropic, Format::Gemini];

    /// The family's stable serialized name.
    pub fn as_str(self) -> &'static str {
        match self {
            Format::OpenAi => "openai",
            Format::Anthropic => "anthropic",
            Format::Gemini => "gemini",
        }
    }

    /// The family `name` spells, or `None`.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|format| format.as_str() == name)
    }

    /// Every family's name, for an error message.
    fn names() -> String {
        Self::ALL
            .iter()
            .map(|format| format.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for Format {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Format {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Self::named(&name).ok_or_else(|| {
            de::Error::custom(format!(
                "`{name}` is not a protocol family ({})",
                Format::names()
            ))
        })
    }
}

/// Which registered provider a [`ProviderId`] names.
///
/// The one decomposition of an identity into a vendor and a family: every
/// accessor reads this, and nothing else rediscovers it. Each arm holds the
/// provider's existing dialect definition rather than restating it, so the
/// registry cannot disagree with the tables it is built from.
#[derive(Debug, Clone, Copy)]
enum Registered {
    OpenAi(openai::wire::Dialect),
    Anthropic(anthropic::wire::Dialect),
    Gemini,
}

/// A registered provider selection: a vendor this build ships, paired with
/// one protocol family it speaks.
///
/// Opaque and minted only here, from a provider's own dialect definition, so
/// a pair the crate does not describe — `openai/anthropic`,
/// `anthropic/gemini` — has no representation, and no public unchecked
/// constructor or derived deserializer can make one:
/// [`Deserialize`](ProviderId#impl-Deserialize<'de>-for-ProviderId) goes
/// through [`resolve`](Self::resolve) like any other input.
///
/// "Registered" means *listed in this build's dialect tables*, which is what
/// [`all`](Self::all) yields and [`resolve`](Self::resolve) accepts. A
/// dialect const that exists but is missing from those tables — an
/// in-progress provider — is still describable through
/// [`ProviderConfig::id`], and will not resolve; adding the const to its
/// table is what registers it.
///
/// [`Display`](fmt::Display) writes the canonical `vendor/format`.
#[derive(Debug, Clone, Copy)]
pub struct ProviderId(Registered);

/// Identity is the `(vendor, format)` pair, never the dialect's payload: two
/// ids that name the same selection are the same id.
impl PartialEq for ProviderId {
    fn eq(&self, other: &Self) -> bool {
        self.vendor() == other.vendor() && self.format() == other.format()
    }
}

impl Eq for ProviderId {}

impl Hash for ProviderId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vendor().hash(state);
        self.format().hash(state);
    }
}

impl ProviderId {
    /// Every selection this build registers: one per OpenAI-shaped dialect,
    /// one per Messages-format dialect, and Gemini.
    pub fn all() -> impl Iterator<Item = ProviderId> {
        openai::wire::all()
            .map(|dialect| ProviderId(Registered::OpenAi(*dialect)))
            .chain(
                anthropic::wire::all().map(|dialect| ProviderId(Registered::Anthropic(*dialect))),
            )
            .chain(std::iter::once(ProviderId(Registered::Gemini)))
    }

    /// The selection `vendor` names in `format`, or `None` when this build
    /// registers no such pair.
    pub fn new(vendor: &str, format: Format) -> Option<Self> {
        match format {
            Format::OpenAi => {
                openai::wire::by_name(vendor).map(|dialect| Self(Registered::OpenAi(*dialect)))
            }
            Format::Anthropic => anthropic::wire::Dialect::by_name(vendor)
                .map(|dialect| Self(Registered::Anthropic(dialect))),
            Format::Gemini => (vendor == gemini::PROVIDER_NAME).then_some(Self(Registered::Gemini)),
        }
    }

    /// The vendor, spelled as the provider's own descriptor name.
    pub fn vendor(&self) -> &'static str {
        match &self.0 {
            Registered::OpenAi(dialect) => dialect.name,
            Registered::Anthropic(dialect) => dialect.name,
            Registered::Gemini => gemini::PROVIDER_NAME,
        }
    }

    /// The protocol family.
    pub fn format(&self) -> Format {
        match &self.0 {
            Registered::OpenAi(_) => Format::OpenAi,
            Registered::Anthropic(_) => Format::Anthropic,
            Registered::Gemini => Format::Gemini,
        }
    }

    /// Every selection registered for `vendor`, in registration order.
    pub fn vendor_selections(vendor: &str) -> impl Iterator<Item = ProviderId> + '_ {
        Self::all().filter(move |id| id.vendor() == vendor)
    }

    /// Resolve `vendor` or `vendor/format`.
    ///
    /// A bare vendor is accepted only when this build registers exactly one
    /// family for it; otherwise the error names the qualified alternatives,
    /// each of which this resolver accepts.
    pub fn resolve(selection: &str) -> Result<Self, SelectionError> {
        let malformed = || SelectionError::Malformed {
            selection: selection.to_owned(),
        };
        let (vendor, format) = match selection.split_once('/') {
            Some((_, rest)) if rest.contains('/') => return Err(malformed()),
            Some((vendor, format)) => (vendor, Some(format)),
            None => (selection, None),
        };
        if vendor.is_empty() {
            return Err(malformed());
        }
        let Some(format) = format else {
            let mut registered = Self::vendor_selections(vendor);
            let first = registered.next().ok_or_else(|| SelectionError::Unknown {
                vendor: vendor.to_owned(),
            })?;
            return match registered.next() {
                None => Ok(first),
                Some(_) => Err(SelectionError::Ambiguous {
                    vendor: vendor.to_owned(),
                    alternatives: alternatives(vendor),
                }),
            };
        };
        let Some(family) = Format::named(format) else {
            return Err(SelectionError::UnknownFormat {
                format: format.to_owned(),
            });
        };
        Self::new(vendor, family).ok_or_else(|| {
            let alternatives = alternatives(vendor);
            match alternatives.is_empty() {
                true => SelectionError::Unknown {
                    vendor: vendor.to_owned(),
                },
                false => SelectionError::Unregistered {
                    vendor: vendor.to_owned(),
                    format: family,
                    alternatives,
                },
            }
        })
    }

    /// This selection's preset configuration, credentialed with `api_key`.
    ///
    /// Infallible: every registered selection holds a definition that already
    /// exists, so there is nothing to look up and nothing to fail.
    pub fn config(&self, api_key: impl Into<Secret>) -> ProviderConfig {
        match &self.0 {
            Registered::OpenAi(dialect) if dialect.name == copilot::PROVIDER_NAME => {
                let provider = copilot::wire::Copilot::new(api_key);
                ProviderConfig::OpenAi(
                    openai::wire::OpenAI::with_key(dialect, provider.api_key)
                        .with_base_url(provider.base_url),
                )
            }
            Registered::OpenAi(dialect) => {
                ProviderConfig::OpenAi(openai::wire::OpenAI::with_key(dialect, api_key))
            }
            Registered::Anthropic(dialect) => ProviderConfig::Anthropic(
                anthropic::wire::Anthropic::with_dialect(api_key, dialect),
            ),
            Registered::Gemini => ProviderConfig::Gemini(gemini::Gemini::new(api_key)),
        }
    }

    /// The environment variable the provider documents its credential under —
    /// borrowed, because the dialect tables already hold it.
    pub fn api_key_env(&self) -> &'static str {
        match &self.0 {
            Registered::OpenAi(dialect) => dialect.api_key_env,
            Registered::Anthropic(dialect) => dialect.api_key_env,
            Registered::Gemini => gemini::API_KEY_ENV,
        }
    }

    /// Whether this selection needs a credential at all.
    ///
    /// A local `llama-server` authenticates optionally, so naming its
    /// variable is a hint rather than a requirement.
    pub fn requires_credential(&self) -> bool {
        match &self.0 {
            Registered::OpenAi(dialect) => {
                !matches!(dialect.quirks.auth, openai::wire::Auth::OptionalBearer)
            }
            Registered::Anthropic(_) | Registered::Gemini => true,
        }
    }
}

/// The qualified spellings registered for `vendor`, in registration order.
fn alternatives(vendor: &str) -> Vec<String> {
    ProviderId::vendor_selections(vendor)
        .map(|id| id.to_string())
        .collect()
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.vendor(), self.format())
    }
}

impl Serialize for ProviderId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ProviderId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let selection = String::deserialize(deserializer)?;
        Self::resolve(&selection).map_err(de::Error::custom)
    }
}

/// Why a provider selection did not resolve.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SelectionError {
    /// No registered provider goes by this vendor name.
    #[error("no registered provider is named `{vendor}`")]
    Unknown {
        /// The vendor named.
        vendor: String,
    },
    /// The vendor is registered, but not for this protocol family.
    #[error(
        "`{vendor}` speaks no {format} endpoint in this build (it speaks {})",
        alternatives.join(", ")
    )]
    Unregistered {
        /// The vendor named.
        vendor: String,
        /// The family named.
        format: Format,
        /// The vendor's registered selections, canonically spelled.
        alternatives: Vec<String>,
    },
    /// A bare vendor with more than one registered family.
    #[error(
        "`{vendor}` names more than one registered selection: name one of {}",
        alternatives.join(", ")
    )]
    Ambiguous {
        /// The vendor named.
        vendor: String,
        /// The vendor's registered selections, canonically spelled.
        alternatives: Vec<String>,
    },
    /// The family is not one this crate knows.
    #[error("`{format}` is not a protocol family ({})", Format::names())]
    UnknownFormat {
        /// The family named.
        format: String,
    },
    /// Not `vendor` or `vendor/format` at all.
    #[error("`{selection}` is not a provider selection: expected `vendor` or `vendor/format`")]
    Malformed {
        /// What was given.
        selection: String,
    },
}

/// A provider's configuration, by protocol family: this crate's existing
/// configuration types, not a copy of their fields.
///
/// Serializes externally tagged — `{"openai": {…}}` — so the family is read
/// before the configuration, and a misspelled option is an error naming the
/// field rather than a failure to match any variant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ProviderConfig {
    /// An OpenAI-shaped provider, on either of its two endpoints.
    #[serde(rename = "openai")]
    OpenAi(openai::wire::OpenAI),
    /// A Messages-format provider.
    #[serde(rename = "anthropic")]
    Anthropic(anthropic::wire::Anthropic),
    /// Gemini.
    #[serde(rename = "gemini")]
    Gemini(gemini::Gemini),
}

impl ProviderConfig {
    /// The registered selection this configuration speaks: its dialect's
    /// identity, whatever host or options it carries. The family is
    /// [`ProviderId::format`] on it.
    pub fn id(&self) -> ProviderId {
        ProviderId(match self {
            Self::OpenAi(provider) => Registered::OpenAi(provider.dialect),
            Self::Anthropic(provider) => Registered::Anthropic(provider.dialect),
            Self::Gemini(_) => Registered::Gemini,
        })
    }

    /// Whether the credential is empty — which it always is after a round
    /// trip, since [`Secret`] does not serialize.
    pub fn is_unauthenticated(&self) -> bool {
        match self {
            Self::OpenAi(provider) => provider.api_key.is_empty(),
            Self::Anthropic(provider) => provider.api_key.is_empty(),
            Self::Gemini(provider) => provider.api_key.is_empty(),
        }
    }

    /// The same configuration with `api_key` as its credential: how a host
    /// rehydrates a configuration it loaded from data.
    pub fn with_credential(mut self, api_key: impl Into<Secret>) -> Self {
        let api_key = api_key.into();
        match &mut self {
            Self::OpenAi(provider) => provider.api_key = api_key,
            Self::Anthropic(provider) => provider.api_key = api_key,
            Self::Gemini(provider) => provider.api_key = api_key,
        }
        self
    }

    /// The completion wire for `model`, bound to `http`, erased behind a
    /// [`CompletionAdapter`] labelled `label`.
    ///
    /// A handler built from data uses the provider's own completion wire,
    /// including Copilot's model-dependent routing and request envelope.
    /// Explicit hosts, routes and typed options are preserved.
    pub fn completion_handler(
        &self,
        label: &str,
        model: &str,
        http: BoxedHttpClient,
    ) -> ErasedHandler {
        match self {
            Self::OpenAi(provider) if provider.dialect.name == copilot::PROVIDER_NAME => erase(
                copilot::wire::CopilotWire::from_openai(provider.clone(), model),
                label,
                http,
            ),
            Self::OpenAi(provider) => erase(provider.completion(model), label, http),
            Self::Anthropic(provider) => erase(provider.completion(model), label, http),
            Self::Gemini(provider) => erase(provider.completion(model), label, http),
        }
    }
}

/// Bind the provider's completion wire to `http` and erase it under `label`.
fn erase<W>(wire: W, label: &str, http: BoxedHttpClient) -> ErasedHandler
where
    W: Wire<Op = Completion>,
    Bound<W, BoxedHttpClient>: CompletionModel + 'static,
{
    ErasedHandler::new(CompletionAdapter::new(label, wire.bind(http)))
}

/// Which provider a [`ProviderRef`] names: the registry's preset for a
/// selection, or an explicit configuration.
#[derive(Clone, Debug, PartialEq)]
pub enum Provider {
    /// A registered selection, whose configuration is the registry's preset.
    Registered(ProviderId),
    /// An explicit configuration, carrying its own host and options.
    Configured(ProviderConfig),
}

/// A model identifier paired with the provider that serves it: what a host
/// stores when it stores "which model".
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderRef {
    /// The provider.
    pub provider: Provider,
    /// The provider's own model identifier. Non-empty: the string form
    /// separates the selection from the model at the first `:`, so an empty
    /// model has no spelling [`parse`](Self::parse) would read back.
    pub model: String,
}

impl ProviderRef {
    /// A reference to `model` on a registered selection.
    pub fn registered(id: ProviderId, model: impl Into<String>) -> Self {
        Self {
            provider: Provider::Registered(id),
            model: model.into(),
        }
    }

    /// A reference to `model` on an explicit configuration.
    pub fn configured(config: ProviderConfig, model: impl Into<String>) -> Self {
        Self {
            provider: Provider::Configured(config),
            model: model.into(),
        }
    }

    /// Parse `vendor[/format]:model`.
    ///
    /// The first `:` separates the selection from the model, so a model
    /// identifier may contain `:` and `/` freely.
    pub fn parse(text: &str) -> Result<Self, RefError> {
        let Some((selection, model)) = text.split_once(':') else {
            return Err(RefError::NoModel {
                reference: text.to_owned(),
            });
        };
        if model.is_empty() {
            return Err(RefError::NoModel {
                reference: text.to_owned(),
            });
        }
        Ok(Self::registered(ProviderId::resolve(selection)?, model))
    }

    /// The registered selection this reference names — the preset's own, or
    /// the configuration's dialect. The canonical identity label diagnostics
    /// print.
    pub fn id(&self) -> ProviderId {
        match &self.provider {
            Provider::Registered(id) => *id,
            Provider::Configured(config) => config.id(),
        }
    }

    /// The configuration to build from, credentialed with `api_key`: the
    /// registry's preset, or the explicit configuration rehydrated.
    pub fn config(&self, api_key: impl Into<Secret>) -> ProviderConfig {
        match &self.provider {
            Provider::Registered(id) => id.config(api_key),
            Provider::Configured(config) => config.clone().with_credential(api_key),
        }
    }
}

impl std::str::FromStr for ProviderRef {
    type Err = RefError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        Self::parse(text)
    }
}

/// The canonical identity spelling: `vendor/format:model`, for a registered
/// selection and for a configuration alike.
///
/// This is a *label*, not the serialized form: an explicit configuration's
/// host and options are not in it, and writing those is `Serialize`'s job.
impl fmt::Display for ProviderRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.id(), self.model)
    }
}

/// Why a provider reference did not parse.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RefError {
    /// No model identifier after the selection.
    #[error("`{reference}` names no model: expected `vendor[/format]:model`")]
    NoModel {
        /// What was given.
        reference: String,
    },
    /// The selection did not resolve.
    #[error(transparent)]
    Selection(#[from] SelectionError),
}

/// The field names of the object form, which is also what a wrong shape is
/// reported against.
const REF_FIELDS: &[&str] = &["config", "model"];

/// A registered reference writes its canonical string; a configured one
/// writes `{config, model}`. Nothing else: the string form cannot express a
/// host or an option, so a configuration must never be written as one.
impl Serialize for ProviderRef {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match &self.provider {
            Provider::Registered(_) => serializer.collect_str(self),
            Provider::Configured(config) => {
                let mut object = serializer.serialize_struct("ProviderRef", 2)?;
                object.serialize_field("config", config)?;
                object.serialize_field("model", &self.model)?;
                object.end()
            }
        }
    }
}

/// Symmetric with `Serialize`: a string is a registered reference, a map is a
/// configuration.
///
/// Dispatching on the shape rather than buffering the input keeps the
/// configuration's own field errors intact, and needs a self-describing
/// format — which JSON is.
impl<'de> Deserialize<'de> for ProviderRef {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(RefVisitor)
    }
}

struct RefVisitor;

impl<'de> Visitor<'de> for RefVisitor {
    type Value = ProviderRef;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "a provider reference `vendor[/format]:model`, or an object with `config` and `model`",
        )
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
        ProviderRef::parse(text).map_err(de::Error::custom)
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut config: Option<ProviderConfig> = None;
        let mut model: Option<String> = None;
        while let Some(field) = map.next_key::<String>()? {
            match field.as_str() {
                "config" => {
                    if config.is_some() {
                        return Err(de::Error::duplicate_field("config"));
                    }
                    config = Some(map.next_value()?);
                }
                "model" => {
                    if model.is_some() {
                        return Err(de::Error::duplicate_field("model"));
                    }
                    model = Some(map.next_value()?);
                }
                unknown => return Err(de::Error::unknown_field(unknown, REF_FIELDS)),
            }
        }
        Ok(ProviderRef::configured(
            config.ok_or_else(|| de::Error::missing_field("config"))?,
            model.ok_or_else(|| de::Error::missing_field("model"))?,
        ))
    }
}

#[cfg(test)]
mod tests;
