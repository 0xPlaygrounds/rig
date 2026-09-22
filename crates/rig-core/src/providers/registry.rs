//! Serializable provider selections, explicit configurations, and model references.
//!
//! [`ProviderRef`] pairs a nonempty model ID with a registered preset or an
//! explicit configuration. References discard credentials and require a
//! self-describing serialization format; hosts supply credentials at execution.
//!
//! ```
//! use rig_core::providers::registry::ProviderRef;
//! let reference = ProviderRef::parse("deepseek:deepseek-chat")?;
//! assert_eq!(reference.to_string(), "deepseek/openai:deepseek-chat");
//! # Ok::<(), rig_core::providers::registry::RefError>(())
//! ```

use std::fmt;
use std::hash::{Hash, Hasher};

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::completion::CompletionModel;
use crate::driver::{Bind, Bound};
use crate::http_client::BoxedHttpClient;
use crate::operation::Completion;
use crate::providers::{anthropic, gemini, openai};
use crate::serve::ErasedHandler;
use crate::serve::adapters::CompletionAdapter;
use crate::wire::{HasCompletion, Secret, Wire};

/// Every dialect this build knows, for
/// [`openai::wire::Dialect`]'s [`Deserialize`](serde::Deserialize) lookup.
///
/// Keyed by [`openai::wire::Dialect::name`], so the regional and endpoint variants that
/// share a provider name are not listed: a stored wire keeps its base URL,
/// which is what distinguishes them.
pub(crate) const OPENAI_DIALECTS: &[&openai::wire::Dialect] = &[
    &openai::wire::OPENAI,
    &openai::wire::AZURE,
    &openai::wire::DEEPSEEK,
    &openai::wire::GROQ,
    &openai::wire::HYPERBOLIC,
    &openai::wire::MIRA,
    &openai::wire::PERPLEXITY,
    &openai::wire::TOGETHER,
    &openai::wire::HUGGINGFACE,
    &openai::wire::LLAMACPP,
    &openai::wire::MISTRAL,
    &openai::wire::OPENROUTER,
    &openai::wire::VENICE,
    &openai::wire::DOUBLEWORD,
    &openai::wire::ZAI,
    &openai::wire::MINIMAX,
    &openai::wire::MOONSHOT,
    &openai::wire::XIAOMIMIMO,
    &crate::providers::xai::DIALECT,
    &crate::providers::chatgpt::DIALECT,
    &crate::providers::copilot::wire::DIALECT,
];

/// A provider's request grammar and configuration type.
/// The OpenAI family includes Chat Completions and Responses; select the endpoint
/// through [`openai::wire::OpenAI::with_route`].
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
    /// Every protocol family, in registration order.
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

/// Registered dialect definition and its configuration family.
#[derive(Debug, Clone, Copy)]
enum Registered {
    OpenAi(openai::wire::Dialect),
    Anthropic(anthropic::wire::Dialect),
    Gemini,
}

/// Validated vendor and protocol-family pair from this build's dialect tables.
/// Construction and deserialization reject unregistered pairs. Equality and
/// hashing use the pair, not dialect options; display emits `vendor/format`.
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

    /// Build this selection's preset with `api_key`. Copilot requires an exchanged
    /// session token, not a GitHub OAuth token.
    pub fn config(&self, api_key: impl Into<Secret>) -> ProviderConfig {
        match &self.0 {
            Registered::OpenAi(dialect) => {
                ProviderConfig::OpenAi(openai::wire::OpenAI::with_key(dialect, api_key))
            }
            Registered::Anthropic(dialect) => ProviderConfig::Anthropic(
                anthropic::wire::Anthropic::with_dialect(api_key, dialect),
            ),
            Registered::Gemini => ProviderConfig::Gemini(gemini::Gemini::new(api_key)),
        }
    }

    /// Environment variable named by the registered credential configuration.
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

/// A provider configuration tagged by protocol family, such as `{"openai": {…}}`.
/// Serialization rejects unregistered or modified dialect definitions. Set
/// persistent hosts and typed options on the configuration, not the dialect.
/// Deserialization reports unknown configuration fields by name.
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
    /// The catalog selection with this dialect's name, if registered.
    /// Host and option overrides do not change the selection; its preset
    /// remains the catalog's, never a custom dialect's payload.
    pub fn id(&self) -> Option<ProviderId> {
        ProviderId::new(self.vendor(), self.format())
    }

    /// The configured dialect's name, including unregistered dialects.
    pub fn vendor(&self) -> &'static str {
        match self {
            Self::OpenAi(provider) => provider.dialect.name,
            Self::Anthropic(provider) => provider.dialect.name,
            Self::Gemini(_) => gemini::PROVIDER_NAME,
        }
    }

    /// The configuration family, independently of catalog membership.
    pub fn format(&self) -> Format {
        match self {
            Self::OpenAi(_) => Format::OpenAi,
            Self::Anthropic(_) => Format::Anthropic,
            Self::Gemini(_) => Format::Gemini,
        }
    }

    /// Whether the credential is empty, as it is after serialization round-trip
    /// because [`Secret`] omits its value.
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

/// A nonempty model identifier paired with a credential-free provider selection.
/// Construction validates the identifier; both fields are read-only.
///
/// ```compile_fail
/// use rig_core::providers::registry::ProviderRef;
/// let Ok(mut reference) = ProviderRef::parse("deepseek:deepseek-chat") else { return; };
/// reference.model.clear(); // the validated identifier is private
/// ```
///
/// Provider data is also read-only, so credentials cannot be inserted afterwards:
///
/// ```compile_fail
/// use rig_core::providers::registry::ProviderRef;
/// let Ok(mut reference) = ProviderRef::parse("deepseek:deepseek-chat") else { return; };
/// reference.provider = reference.provider().clone();
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderRef {
    /// The provider.
    provider: Provider,
    /// The provider's own model identifier. Non-empty: the string form
    /// separates the selection from the model at the first `:`, so an empty
    /// model has no spelling [`parse`](Self::parse) would read back.
    model: String,
}

impl ProviderRef {
    /// A reference to a non-empty `model` on a registered selection.
    /// Returns [`RefError::EmptyModel`] for an empty identifier.
    pub fn registered(id: ProviderId, model: impl Into<String>) -> Result<Self, RefError> {
        Self::new(Provider::Registered(id), model.into())
    }

    /// A reference to a non-empty `model` on an explicit configuration.
    /// Removes the credential: references are persistent recipes, and the
    /// host supplies credentials through [`config`](Self::config) at construction time.
    /// The host stays fixed, even if it is a preset's default. For Copilot,
    /// use a registered reference to follow the resolved token's proxy endpoint,
    /// or configure the intended endpoint before creating this reference.
    /// Returns [`RefError::EmptyModel`] for an empty identifier.
    pub fn configured(config: ProviderConfig, model: impl Into<String>) -> Result<Self, RefError> {
        Self::new(
            Provider::Configured(config.with_credential("")),
            model.into(),
        )
    }

    /// The credential-free provider recipe.
    pub fn provider(&self) -> &Provider {
        &self.provider
    }

    fn new(provider: Provider, model: String) -> Result<Self, RefError> {
        if model.is_empty() {
            return Err(RefError::EmptyModel);
        }
        Ok(Self { provider, model })
    }

    /// The non-empty model identifier.
    pub fn model(&self) -> &str {
        &self.model
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
        Self::registered(ProviderId::resolve(selection)?, model)
    }

    /// The catalog selection this reference names, if its dialect is registered.
    pub fn id(&self) -> Option<ProviderId> {
        match &self.provider {
            Provider::Registered(id) => Some(*id),
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

/// Display `vendor/format:model` for registered and configured references.
/// This label omits hosts and options; use serialization to preserve them.
impl fmt::Display for ProviderRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.provider {
            Provider::Registered(id) => write!(f, "{id}:{}", self.model),
            Provider::Configured(config) => {
                write!(f, "{}/{}:{}", config.vendor(), config.format(), self.model)
            }
        }
    }
}

/// Why a provider reference did not parse.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RefError {
    /// A constructor or structured reference supplied an empty model identifier.
    #[error("model identifier must not be empty")]
    EmptyModel,
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

/// Serialize registered references as canonical strings and configured references
/// as `{config, model}` objects to preserve their hosts and options.
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

/// Deserialize strings as registered references and maps as explicit configurations.
/// Requires a self-describing format and preserves configuration field errors.
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
        ProviderRef::configured(
            config.ok_or_else(|| de::Error::missing_field("config"))?,
            model.ok_or_else(|| de::Error::missing_field("model"))?,
        )
        .map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests;
